# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
from typing import Dict, List, Optional
import sys

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence

def generate(beam_ids, dict, eoms):
    # Replace pad
    text = "".join([dict.symbols[id] if id != dict.pad() else "~" for id in beam_ids[0, 1:]])
    for counter, index in enumerate(eoms):
        text = text[: index + counter] + "-" + text[index + counter: ]
    print(text)


class SubwordSegmentalSeparateTextGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        max_len=0,
        min_len=1,
        normalize_scores=True,
        average_next_scores=False,
        normalize_type=None,
        marginalize=None,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len or self.model.max_decoder_positions()

        self.normalize_scores = normalize_scores
        self.average_next_scores = average_next_scores
        self.normalize_type = normalize_type
        self.marginalize = marginalize
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(
        self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception(
                "expected src_tokens or source in net input. input keys: "
                + str(net_input.keys())
            )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam


        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, 1).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not


        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        beam_ids = []
        beam_lprobs = []
        beam_eoms = [[[0] for _ in range(beam_size)] for _ in range(bsz)]

        active_batch_indices = [i for i in range(bsz)]
        active_bsz = bsz
        to_be_finalized = [beam_size for _ in range(bsz)]  # number of sentences remaining
        finalized_sent_ids = [[] for _ in range(bsz)]  # completed
        finalized_sent_lprobs = [[] for _ in range(bsz)]
        finalized_sent_eoms = [[] for _ in range(bsz)]

        decode_normalization_type, final_normalization_type = self.normalize_type.split("-")
        beam_marginalize, final_marginalize = self.marginalize.split("-")

        LOGINF = math.inf
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Remove final src tokens (always eos)
        src_tokens = src_tokens[:, 0: src_tokens.size(1) - 1]

        # Make any other eos tokens pads
        src_tokens[src_tokens == self.eos] = self.pad

        # Add eos to start of src tokens
        bos_tokens = torch.tensor([[self.eos]] * bsz).to(device)
        src_tokens = torch.cat([bos_tokens, src_tokens], dim=1)
        prompt_len = src_tokens.size(1)
        beam_eoms = [[[prompt_len - 1] for _ in range(beam_size)] for _ in range(bsz)]

        chars2lex = self.model.single_model.decoder.chars2lex
        lex2chars = {chars_ids: lex_id for lex_id, chars_ids in chars2lex.items()}

        for step in range(max_len + 1):  # 0 is first letter, one extra step for EOS marker

            with torch.autograd.profiler.record_function(
                "EnsembleModel: forward_decoder"
            ):
                if step == 0:
                    init_beam_ids = src_tokens
                    prev_seg_end = [prompt_len - 1 for i in range(bsz)]
                    prev_output_tokens = (init_beam_ids, prev_seg_end)

                    next_lex_lprobs, top_char_segs, top_char_lprobs, history_encodings = self.model.forward_decoder(
                        prev_output_tokens,
                        None,
                        incremental_states,
                        self.temperature,
                    )

                    # Handle min length constraint
                    next_lex_lprobs[:, self.eos] = -LOGINF

                    for batch_num in range(bsz):
                        batch_lprobs = next_lex_lprobs[batch_num]
                        beam_ids.append([])
                        beam_lprobs.append([])

                        prompt_ids = init_beam_ids[batch_num].unsqueeze(0)


                        next_char_ids = []
                        next_seg_lprobs = []
                        for beam_num, index in enumerate(torch.topk(batch_lprobs, k=beam_size).indices):
                            char_ids = list(lex2chars[index.item()])
                            next_char_ids.append(char_ids)
                            next_seg_lprobs.append(batch_lprobs[index])

                        for i, char_ids in enumerate(top_char_segs[batch_num]): # seg_len = i+1
                            next_char_ids.append(char_ids)
                            next_seg_lprobs.append(top_char_lprobs[batch_num, i])

                        next_seg_lprobs = torch.tensor(next_seg_lprobs)

                        for beam_num, index in enumerate(torch.topk(next_seg_lprobs, k=beam_size).indices):
                            char_ids = next_char_ids[index]
                            beam_ids[batch_num].append(torch.cat([prompt_ids, torch.tensor([char_ids]).to(device)], dim=1))
                            beam_lprobs[batch_num].append(torch.tensor([next_seg_lprobs[index].unsqueeze(-1)]).to(device))
                            beam_eoms[batch_num][beam_num].append(prompt_len - 1 + len(char_ids))


                    # Prepare incremental states for multiple beams per batch
                    reorder_state = torch.tensor([[batch_num] * beam_size for batch_num in range(bsz)]).flatten()
                    reorder_state = reorder_state.to(device)
                    history_embeddings = torch.repeat_interleave(history_encodings, beam_size, dim=0)\
                        .view(bsz, beam_size, 1, -1)
                else:

                    if step >= max_len:
                        print("Reached max step without finalizing all sentences.")

                    # Prep con_beams
                    prev_ids = [ids.transpose(0, 1) for batch_ids in beam_ids for ids in batch_ids]
                    prev_ids = pad_sequence(prev_ids, padding_value=self.pad).transpose(0, 1).squeeze(-1)
                    prev_eoms = [beam_eoms[batch_num][beam_num][-1] for batch_num in range(active_bsz)
                                     for beam_num in range(beam_size)]
                    prev_output_tokens = (prev_ids, prev_eoms, None)
                    self.model.reorder_incremental_state(incremental_states, reorder_state)

                    next_lex_lprobs, top_char_segs, top_char_lprobs, history_encodings = self.model.forward_decoder(
                        prev_output_tokens,
                        None,
                        incremental_states,
                        self.temperature,
                    )  # con_con_lprobs, con_end_lprobs should contain lprobs of current segment up to this point

                    # Handle max length constraint
                    if step == max_len:  # Force eos
                        next_lex_lprobs[:, :] = -LOGINF
                        next_lex_lprobs[:, 0: self.eos] = -LOGINF
                        next_lex_lprobs[:, self.eos + 1:] = -LOGINF
                    elif step < self.min_len:
                        next_lex_lprobs[:, self.eos] = -LOGINF

                    next_lex_lprobs = next_lex_lprobs.view(active_bsz, beam_size, -1)

                    new_top_char_segs = []
                    k = 0
                    for i in range(active_bsz):
                        new_top_char_segs.append([])
                        for j in range(beam_size):
                            new_top_char_segs[i].append(top_char_segs[k])
                            k += 1
                    top_char_segs = new_top_char_segs

                    top_char_lprobs = top_char_lprobs.view(active_bsz, beam_size, -1)

                    max_seg_ids = []  # stores char ids
                    max_seg_lprobs = []
                    # max_con_con_lprobs, max_con_end_lprobs contains entire sequence probabilities of beam_size * cand_size candidates
                    for batch_num in range(active_bsz):
                        max_seg_ids.append([])
                        max_seg_lprobs.append([])
                        normalizers = []
                        batch_details = []

                        # for beam_num, prev_ids in enumerate(con_beam_ids):
                        for beam_num in range(beam_size):
                            beam_next_lex_lprobs = next_lex_lprobs[batch_num, beam_num]
                            for index in torch.topk(beam_next_lex_lprobs, k=beam_size).indices:  # beam_size used to be cand-size
                                max_seg_ids[batch_num].append(list(lex2chars[index.item()]))

                                if beam_marginalize == "none":
                                    score = torch.sum(beam_lprobs[batch_num][beam_num][0: -1]) + beam_next_lex_lprobs[
                                        index]
                                    max_seg_lprobs[batch_num].append(score)
                                elif beam_marginalize == "approx":
                                    score = torch.sum(beam_lprobs[batch_num][beam_num][0: -1]) + beam_next_lex_lprobs[index]
                                elif beam_marginalize == "exact":
                                    score = -LOGINF
                                else:
                                    raise ValueError("Invalid marginalization type")

                                if decode_normalization_type == "seg":
                                    normalizers.append(len(beam_eoms[batch_num][beam_num]))
                                elif decode_normalization_type == "char":
                                    normalizers.append(step + len(beam_eoms[batch_num][beam_num]))

                                batch_details.append({"tokens": torch.cat([beam_ids[batch_num][beam_num],
                                                                           torch.tensor([max_seg_ids[batch_num][-1]]).to(device)], dim=-1),
                                                      "score": score})

                            beam_next_char_lprobs = top_char_lprobs[batch_num, beam_num]
                            for i, lprob in enumerate(beam_next_char_lprobs):  # seg_len = i+1
                                char_ids = top_char_segs[batch_num][beam_num][i]
                                max_seg_ids[batch_num].append(char_ids)

                                if beam_marginalize == "none":
                                    score = torch.sum(beam_lprobs[batch_num][beam_num][0: -1]) + beam_next_lex_lprobs[
                                        index]
                                    max_seg_lprobs[batch_num].append(score)
                                elif beam_marginalize == "approx":
                                    score = torch.sum(beam_lprobs[batch_num][beam_num][0: -1]) + beam_next_char_lprobs[i]
                                elif beam_marginalize == "exact":
                                    score = -LOGINF
                                else:
                                    raise ValueError("Invalid marginalization type")



                                if decode_normalization_type == "seg":
                                    normalizers.append(len(beam_eoms[batch_num][beam_num]))
                                elif decode_normalization_type == "char":
                                    normalizers.append(step + len(beam_eoms[batch_num][beam_num]))


                                batch_details.append({"tokens": torch.cat([beam_ids[batch_num][beam_num],
                                                                           torch.tensor([max_seg_ids[batch_num][-1]]).to(device)],
                                                                          dim=-1),
                                                      "score": score})
                        if beam_marginalize == "none":
                            pass
                        elif beam_marginalize == "approx":
                            scores = self.approximate_generations(batch_details)
                            for i in range(len(batch_details)):
                                max_seg_lprobs[batch_num].append(scores[i])
                        elif beam_marginalize == "exact":
                            scores = self.marginalize_generations(batch_details).squeeze(0)
                            for i in range(len(batch_details)):
                                max_seg_lprobs[batch_num].append(scores[i])

                        # Normalize scores
                        for beam_num in range(len(max_seg_lprobs[batch_num])):
                            if final_normalization_type == "seg":
                                max_seg_lprobs[batch_num][beam_num] /= normalizers[beam_num]
                            elif final_normalization_type == "char":
                                max_seg_lprobs[batch_num][beam_num] /= normalizers[beam_num]


                    # Compare new continued segments
                    new_beam_ids = []
                    new_beam_lprobs = []
                    new_beam_eoms = []
                    new_history_embeddings = []
                    new_reorder_states = []

                    for batch_num in range(active_bsz):
                        new_beam_ids.append([])
                        new_beam_lprobs.append([])
                        new_beam_eoms.append([])
                        new_history_embeddings.append([])
                        new_reorder_states.append([])

                        for beam_num in range(cand_size):
                            max_lprob = max(max_seg_lprobs[batch_num])
                            max_index = max_seg_lprobs[batch_num].index(max_lprob)
                            max_beam_num = int(max_index / (beam_size + 5))
                            max_id = max_seg_ids[batch_num][max_index]

                            beam_max_ids = beam_ids[batch_num][max_beam_num]
                            new_char_ids = max_id
                            new_beam_ids[batch_num].append(torch.cat([beam_max_ids, torch.tensor([new_char_ids]).to(device)], dim=-1))

                            beam_max_lprobs = beam_lprobs[batch_num][max_beam_num].detach().clone()
                            beam_max_lprobs = torch.cat(
                                [beam_max_lprobs, torch.tensor([max_seg_lprobs[batch_num][max_index]]).to(device)])
                            new_beam_lprobs[batch_num].append(beam_max_lprobs)

                            beam_max_eoms = beam_eoms[batch_num][max_beam_num].copy()  # necessary?
                            beam_max_eoms.append(beam_max_eoms[-1] + len(new_char_ids))
                            new_beam_eoms[batch_num].append(beam_max_eoms)

                            max_seg_lprobs[batch_num][max_index] = -LOGINF
                            new_reorder_states[batch_num].append(
                                active_bsz * beam_size + batch_num * beam_size + max_beam_num)

                            for cand_num in range(beam_size * (beam_size + 5)):
                                cand_beam_num = int(cand_num / (beam_size + 5))
                                if torch.equal(beam_max_ids, beam_ids[batch_num][cand_beam_num]) and\
                                        new_char_ids == max_seg_ids[batch_num][cand_num] and\
                                        beam_eoms[batch_num][max_beam_num] == beam_eoms[batch_num][cand_beam_num]:
                                    max_seg_lprobs[batch_num][cand_num] = -LOGINF


                    # Collect top beam_size beams and store finished sentences
                    beam_ids = []
                    beam_lprobs = []
                    beam_eoms = []
                    history_embeddings = []
                    reorder_states = []

                    deactivated_batch_nums = []
                    deactivated_batch_indices = []
                    for batch_num in range(active_bsz):
                        beam_ids.append([])
                        beam_lprobs.append([])
                        beam_eoms.append([])
                        history_embeddings.append([])
                        reorder_states.append([])

                        cand_num = 0
                        while len(beam_ids[batch_num]) < beam_size:
                            if new_beam_ids[batch_num][cand_num][0, -1] != self.eos and step < max_len:
                                beam_ids[batch_num].append(new_beam_ids[batch_num][cand_num])
                                beam_lprobs[batch_num].append(new_beam_lprobs[batch_num][cand_num])
                                beam_eoms[batch_num].append(new_beam_eoms[batch_num][cand_num])
                                reorder_states[batch_num].append(new_reorder_states[batch_num][cand_num])
                            elif new_beam_ids[batch_num][cand_num][0, -1] == self.eos or step == max_len: #and cand_num < beam_size:
                                batch_index = active_batch_indices[batch_num]
                                finalized_sent_ids[batch_index].append(new_beam_ids[batch_num][cand_num])
                                finalized_sent_lprobs[batch_index].append(new_beam_lprobs[batch_num][cand_num])
                                finalized_sent_eoms[batch_index].append(new_beam_eoms[batch_num][cand_num])

                                to_be_finalized[batch_index] -= 1
                                if to_be_finalized[batch_index] == 0:
                                    active_bsz -= 1
                                    deactivated_batch_indices.append(batch_index)
                                    deactivated_batch_nums.append(batch_num)
                                    break
                            cand_num += 1


                    if step == max_len or active_bsz == 0:
                        break

                    for batch_index in deactivated_batch_indices:
                        active_batch_indices.remove(batch_index)

                    for batch_num in sorted(deactivated_batch_nums, reverse=True):
                        del beam_ids[batch_num]
                        del beam_lprobs[batch_num]
                        del beam_eoms[batch_num]
                        del history_embeddings[batch_num]
                        del reorder_states[batch_num]


        finalized = []
        for batch_num in range(bsz):
            finalized.append([])
            for beam_num in range(beam_size):
                if beam_num >= len(finalized_sent_eoms[batch_num]):
                    break

                normalizer = 1
                if self.normalize_scores:
                    if final_normalization_type == "seg":
                        normalizer = len(finalized_sent_eoms[batch_num][beam_num]) - 1
                    elif final_normalization_type == "char":
                        normalizer = len(finalized_sent_eoms[batch_num][beam_num]) - 1 + len(finalized_sent_ids[batch_num][beam_num])
                score = finalized_sent_lprobs[batch_num][beam_num][-1] / normalizer  
                

                finalized[batch_num].append({"tokens": finalized_sent_ids[batch_num][beam_num],
                                         "score": score,
                                         "eoms": finalized_sent_eoms[batch_num][beam_num],
                                         "alignment": None,
                                         "positional_scores": finalized_sent_lprobs[batch_num][beam_num]})

        # sort by score descending
        for batch_num in range(bsz):

            if final_marginalize == "none":
                scores = torch.tensor(
                    [float(elem["score"].item()) for elem in finalized[batch_num]]
                )
            elif final_marginalize == "approx":
                scores = self.approximate_generations(finalized[batch_num])
            elif final_marginalize == "exact":
                scores = self.marginalize_generations(finalized[batch_num]).squeeze(0)
            else:
                raise ValueError("Invalid marginalization type")

            for beam_num in range(len(finalized[batch_num])):
                finalized[batch_num][beam_num]["score"] = scores[beam_num]

            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[batch_num] = [finalized[batch_num][ssi] for ssi in sorted_scores_indices]
            finalized[batch_num] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[batch_num]
            )


        return finalized


    def approximate_generations(self, beams):
        # Sum the probabilities (stored in log form) of identical generations
        scores = torch.zeros(len(beams))
        for i, beam in enumerate(beams):
            score = beam["score"]
            for j, other_beam in enumerate(beams):
                if i == j:
                    continue
                if torch.equal(beam["tokens"], other_beam["tokens"]):
                    score = torch.logsumexp(torch.tensor([score, other_beam["score"]]), dim=0)
            scores[i] = score

        return scores


    def marginalize_generations(self, beams):
        self.model.mode = "forward"
        prev_output_tokens = [beams[beam_num]["tokens"] for beam_num in range(len(beams))]
        prev_output_tokens = [beam_ids.transpose(0, 1) for beam_ids in prev_output_tokens]
        prev_output_tokens = pad_sequence(prev_output_tokens, padding_value=self.pad).transpose(0, 1).squeeze(-1)
        incremental_states = None

        self.model.single_model.decoder.generate = False
        log_alpha, log_R, seq_lens = self.model.forward_decoder(
            prev_output_tokens,
            None,
            incremental_states,
            self.temperature,
        )
        self.model.single_model.decoder.generate = True

        index = seq_lens.unsqueeze(dim=-1).transpose(0, 1)
        lprobs = torch.gather(log_alpha, dim=0, index=index)

        return lprobs


    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))


    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min(
            [
                m.max_decoder_positions()
                for m in self.models
                if hasattr(m, "max_decoder_positions")
            ]
            + [sys.maxsize]
        )

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states() and incremental_states is not None:
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                if hasattr(model, "decoder"):
                    decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                else:
                    decoder_out = model.forward(tokens)

        return decoder_out

    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )

