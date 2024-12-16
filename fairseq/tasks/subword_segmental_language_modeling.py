# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
import itertools
from typing import Optional

import numpy as np
import torch
from fairseq import metrics, utils, search
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from omegaconf import II

import re
from collections import Counter
import torch
import nltk
import copy

import time
from fairseq.models.ssmt.ssmt_config import QuantNoiseConfig


EVAL_BLEU_ORDER = 4
ENCODING = "utf-8"
SPACE_NORMALIZER = re.compile(r"\s+")


def char_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return list(line)


def tokenize_segs(line, max_seg_len, char_segs, non_alpha=False):
    # Split into all possible segments
    segs = []
    for n in range(1, max_seg_len+1):
        if n == 1 and not char_segs:
            continue

        chars = list(line)
        segs_n = nltk.ngrams(chars, n=n)
        segs_n = ["".join(seg) for seg in segs_n]

        if not non_alpha and n > 1:  # Discard segments with non-alphabetical characters
            segs_n = [seg for seg in segs_n if seg.isalpha() and len(seg) == n]
        else:
            segs_n = [seg for seg in segs_n if len(seg) == n]
        segs.extend(segs_n)
    return segs


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


@dataclass
class SubwordSegmentalLanguageModelingConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False,
        metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False,
        metadata={"help": "boolean to pad to fixed batch size"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")

    # Subword segmental params
    max_seg_len: int = field(
        default=5,
        metadata={"help": "maximum segment length"},
    )
    lexicon_max_size: int = field(
        default=0,
        metadata={"help": "size of decoder subword lexicon"},
    )
    lexicon_min_count: int = field(
        default=1,
        metadata={"help": "minimum frequency for inclusion in lexicon"}
    )
    vocabs_path: Optional[str] = field(
        default=None,
        metadata={"help": "directory for storing and load lex and char vocabs"}
    )

    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )

    average_next_scores: bool = field(
        default=False, metadata={"help": "average next segment scores during decoding"}
    )
    normalize_type: Optional[str] = field(
        default="seg-seg", metadata={"help": "seg: /= # segs, char: /= (# segs + # chars)"}
    )
    marginalize: Optional[str] = field(
        default=None, metadata={"help": "none, approx, exact"}
    )
    decoding: Optional[str] = field(
        default="dynamic", metadata={"help": "decoding algorithm: dynamic, beam, separate"}
    )
    line_prompts: bool = field(
        default=False, metadata={"help": "load text line by line"}
    )

    quant_noise: QuantNoiseConfig = field(default=QuantNoiseConfig())


@register_task("subword_segmental_language_modeling", dataclass=SubwordSegmentalLanguageModelingConfig)
class SubwordSegmentalLanguageModelingTask(FairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    cfg: SubwordSegmentalLanguageModelingConfig

    def __init__(self, cfg: SubwordSegmentalLanguageModelingConfig, tgt_dict, tgt_lex):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        self.tgt_lex = tgt_lex
        self.line_prompts = cfg.line_prompts

    @classmethod
    def setup_task(cls, cfg: SubwordSegmentalLanguageModelingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        # load dictionaries
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.txt")
        )
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        vocab_path = cfg.vocabs_path

        try:
            logger.info("Trying to load existing target lexicon..")
            tgt_lex = cls.load_dictionary(
                os.path.join(vocab_path, "lex_dict.{}.txt".format(cfg.target_lang))
            )
            logger.info("Lexicon loaded.")
            logger.info("[{}] lexicon dictionary: {} types".format(cfg.target_lang, len(tgt_lex)))

        except FileNotFoundError:
            logger.info("Target lexicon dictionary file does not exist.")
            logger.info("Creating subword lexicon from word dictionary...")
            start_time = time.time()

            counter = Counter()
            symbols = []

            for index, word in enumerate(tgt_dict.symbols):
                if index < tgt_dict.nspecial:
                    counter[word] = tgt_dict.count[index]
                    symbols.append(word)
                else:
                    subwords = tokenize_segs(word, cfg.max_seg_len, char_segs=True, non_alpha=False)
                    subwords = {subword: tgt_dict.count[index] for subword in subwords}  # Counter(subwords)
                    counter.update(subwords)
            print("--- Finished creating %s seconds ---" % (time.time() - start_time))

            nonspecial_symbols = [subword for subword in counter.keys() if subword not in symbols]
            symbols.extend(nonspecial_symbols)

            # Trim lexicon to maximum size
            assert cfg.lexicon_max_size > 0
            trimmed_counter = Counter()
            trimmed_symbols = []

            subword_counter = Counter()
            index = 0
            for subword in symbols:
                if index < tgt_dict.nspecial:
                    trimmed_counter[subword] = counter[subword]
                    trimmed_symbols.append(subword)
                else:
                    subword_counter[subword] = counter[subword]
                index += 1

            for subword, count in subword_counter.most_common(n=cfg.lexicon_max_size):
                trimmed_counter[subword] = count
                trimmed_symbols.append(subword)

            counter = trimmed_counter
            symbols = trimmed_symbols

            print("--- Finished trimming %s seconds ---" % (time.time() - start_time))

            # Add space to lexicon vocab
            space_word = " "
            counter[space_word] = 1
            symbols.append(space_word)

            tgt_lex = copy.deepcopy(tgt_dict)
            tgt_lex.indices = {subword: index for index, subword in enumerate(symbols)}
            tgt_lex.count = [counter[subword] for subword in symbols]
            tgt_lex.symbols = symbols

            tgt_lex.save(os.path.join(vocab_path, "lex_dict.{}.txt".format(cfg.target_lang)))
            logger.info("Target lexicon dictionary saved to file: ")
            logger.info(os.path.join(vocab_path, "lex_dict.{}.txt".format(cfg.target_lang)))

            print("--- Finished saving %s seconds ---" % (time.time() - start_time))

        # START
        try:
            logger.info("Trying to load existing target char vocab..")
            tgt_dict = cls.load_dictionary(
                os.path.join(vocab_path, "char_dict.{}.txt".format(cfg.target_lang))
            )
            logger.info("Char vocab loaded.")
            logger.info("[{}] char dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

            # Add end-of-morpheme token to char vocab
            eom_word = "<eom>"
            tgt_dict.symbols = tgt_dict.symbols[0: tgt_dict.nspecial] + [eom_word] + tgt_dict.symbols[
                                                                                     tgt_dict.nspecial:]
            tgt_dict.count = tgt_dict.count[0: tgt_dict.nspecial] + [1] + tgt_dict.count[tgt_dict.nspecial:]
            tgt_dict.indices = {char: index for index, char in enumerate(tgt_dict.symbols)}
            tgt_dict.nspecial += 1

        except FileNotFoundError:
            logger.info("Target char dictionary file does not exist.")
            logger.info("Creating char vocab from word dictionary...")

            # CREATE CHARACTER TGT_DICT
            counter = Counter()
            symbols = []
            for index, word in enumerate(tgt_dict.symbols):
                if index < tgt_dict.nspecial:
                    counter[word] = tgt_dict.count[index]
                    symbols.append(word)
                else:
                    chars = {char: word.count(char) for char in word}
                    counter.update(chars)

            nonspecial_symbols = [char for char in counter.keys() if char not in symbols]
            symbols.extend(nonspecial_symbols)

            # Add space to char vocab
            space_word = " "
            counter[space_word] = 1
            symbols.append(space_word)

            # Add end-of-morpheme token to char vocab
            eom_word = "<eom>"
            counter[eom_word] = 1
            symbols = symbols[0: tgt_dict.nspecial] + [eom_word] + symbols[tgt_dict.nspecial:]
            tgt_dict.nspecial += 1

            tgt_dict.indices = {char: index for index, char in enumerate(symbols)}
            tgt_dict.count = [counter[char] for char in symbols]
            tgt_dict.symbols = symbols

            # Sort tgt_dict special characters, then alphabetical characters, then the rest
            special_tokens = []
            alpha_chars = []
            non_alpha_chars = []
            for index, char in enumerate(tgt_dict.symbols):
                if index < tgt_dict.nspecial:
                    special_tokens.append(char)
                elif char.isalpha():
                    alpha_chars.append(char)
                else:
                    non_alpha_chars.append(char)

            tgt_dict.symbols = special_tokens + alpha_chars + non_alpha_chars
            for index, char in enumerate(tgt_dict.symbols):
                tgt_dict.indices[char] = index

            tgt_dict.save(os.path.join(vocab_path, "char_dict.{}.txt".format(cfg.target_lang)))
            logger.info("Target char dictionary saved to file:")
            logger.info(os.path.join(vocab_path, "char_dict.{}.txt".format(cfg.target_lang)))

        return cls(cfg, tgt_dict, tgt_lex)


    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        return model


    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """

        if self.line_prompts:
            return self.load_line_prompts(split, epoch, combine, **kwargs)

        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]

        # Read input sentences.
        tgt_path = data_path[0: data_path.rindex("/") + 1] + split + "." + self.cfg.target_lang
        tgt_lines = []
        with open(tgt_path, encoding=ENCODING) as file:
            for line in file:
                tgt_lines.append(line.strip())

        # Now split this into continuous segments of length tokens_per_sample, but don't split in the middle of a word
        def distribute_lines(lines):
            sequences = []
            current_sequence = ""

            for line in lines:
                words = line.split()  # Split the line into words
                for word in words:
                    if len(current_sequence) + len(word) + 1 > self.cfg.tokens_per_sample:  # +1 for space or newline
                        sequences.append(
                            current_sequence.rstrip())  # Append the full sequence and strip trailing spaces
                        current_sequence = word + " "  # Start a new sequence with the current word
                    else:
                        current_sequence += word + " "  # Add word to the current sequence
                current_sequence = current_sequence.rstrip() + "\n"  # Prepare to start a new line

            if current_sequence.strip():  # Add any remaining content to the sequences
                sequences.append(current_sequence.strip())

            return sequences

        tgt_samples = distribute_lines(tgt_lines)

        tgt_sentences = []
        tgt_lengths = []

        for sample in tgt_samples:
            # Tokenize the sentence, splitting on spaces
            tokens = self.tgt_dict.encode_line(
                sample, line_tokenizer=char_tokenize, add_if_not_exist=False,
            )
            tgt_sentences.append(tokens.type(torch.int64))
            tgt_lengths.append(tokens.numel())

        logger.info(
            "{} {} {} examples".format(
                data_path, split, len(tgt_sentences)
            )
        )

        add_eos_for_other_targets = False
        pad_to_bsz = None
        fixed_pad_length = None

        self.datasets[split] = MonolingualDataset(
            dataset=tgt_sentences,
            sizes=tgt_lengths,
            src_vocab=self.tgt_dict,
            tgt_vocab=self.tgt_dict,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=[],
            add_bos_token=self.cfg.add_bos_token,
            fixed_pad_length=fixed_pad_length,
            pad_to_bsz=pad_to_bsz,
        )


    def load_line_prompts(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]

        # Read input sentences.
        tgt_path = data_path[0: data_path.rindex("/") + 1] + split + "." + self.cfg.target_lang
        tgt_lines = []
        with open(tgt_path, encoding=ENCODING) as file:
            for line in file:
                tgt_lines.append(line.strip())

        tgt_samples = tgt_lines

        tgt_sentences = []
        tgt_lengths = []

        for sample in tgt_samples:
            # Tokenize the sentence, splitting on spaces
            tokens = self.tgt_dict.encode_line(
                sample, line_tokenizer=char_tokenize, add_if_not_exist=False, append_eos=False
            )
            # Add eos to start to be consistent with training
            tokens = torch.cat([tokens, torch.tensor([self.tgt_dict.eos()])])
            tgt_sentences.append(tokens.type(torch.int64))
            tgt_lengths.append(tokens.numel())

        logger.info(
            "{} {} {} examples".format(
                data_path, split, len(tgt_sentences)
            )
        )

        add_eos_for_other_targets = False
        pad_to_bsz = None
        fixed_pad_length = None

        self.datasets[split] = MonolingualDataset(
            dataset=tgt_sentences,
            sizes=tgt_lengths,
            src_vocab=self.tgt_dict,
            tgt_vocab=self.tgt_dict,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=[],
            add_bos_token=self.cfg.add_bos_token,
            fixed_pad_length=fixed_pad_length,
            pad_to_bsz=pad_to_bsz,
        )

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        args, task_args = args  # seperate default generation args from task-specific args

        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.subword_segmental_text_generator import (
            SubwordSegmentalTextGenerator
        )

        from fairseq.subword_segmental_separate_text_generator import (
            SubwordSegmentalSeparateTextGenerator
        )

        from fairseq.subword_segmental_evaluator import (
            SubwordSegmentalEvaluator
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        decoding = getattr(task_args, "decoding")
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if decoding == "dynamic":
           seq_gen_cls = SubwordSegmentalTextGenerator
        elif decoding == "separate":
            seq_gen_cls = SubwordSegmentalSeparateTextGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            average_next_scores=getattr(task_args, "average_next_scores", False),
            normalize_type=getattr(task_args, "normalize_type", None),
            marginalize=getattr(task_args, "marginalize", None),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.target_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.target_dictionary.bos()
                if self.cfg.add_bos_token is False
                else self.target_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.target_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.target_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.target_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if self.cfg.add_bos_token is False:
                bos_token = self.target_dictionary.bos()
            else:
                bos_token = self.target_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the language_modeling task is not supported"
                )

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.cfg.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.target_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        ).next_epoch_itr(shuffle=False)


    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    @property
    def target_lexicon(self):
        """Return the target lexicon :class:`~fairseq.data.Dictionary`."""
        return self.tgt_lex

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        # Print example segmentations
        sample["net_input"]["mode"] = "segment"

        # View current subword segmentation by sampling validation segmentations
        n = 10
        sample["id"] = sample["id"][0: n]
        sample["nsentences"] = n

        sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"][0: n]
        sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"][0: n]
        sample["net_input"]["prev_output_tokens"] = sample["net_input"]["prev_output_tokens"][0: n]
        sample["target"] = sample["target"][0: n]
        sample["ntokens"] = torch.numel(sample["target"])

        split_indices = self.segment(sample, model, criterion)
        split_text = self.split_text(sample, split_indices)
        #print(split_text)

        for i, text in enumerate(split_text):
            print(sample["id"][i].item())
            print("|" + text.replace("</s>", ""))


        return loss, sample_size, logging_output

    def split_text(self, sample, split_indices, num_examples=10):
        target_ids = sample["target"].transpose(0, 1)

        batch_size = target_ids.shape[1]
        seq_len = target_ids.shape[0]
        batch_texts = []
        num_examples = min(batch_size, num_examples)

        eos_ends = []

        for i in range(num_examples):
            eos_ends.append(False)
            seq_text = ""
            for j in range(seq_len):
                if target_ids[j][i] == self.tgt_dict.eos_index:
                    eos_ends[i] = True
                    break
                seq_text += self.tgt_dict.symbols[target_ids[j][i]]

            seq_text = seq_text.replace("</s>", " ")
            batch_texts.append(seq_text)

        split_texts = []
        for i, text in enumerate(batch_texts):
            for counter, index in enumerate(split_indices[i]):
                text = text[:index + counter + 1] + "|" + text[index + counter + 1:]

            if eos_ends[i]:
                text = text[0: -1] + "</s>"

            split_texts.append(text)

        return split_texts

    def segment(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            split_indices = criterion(model, sample)
        return split_indices
