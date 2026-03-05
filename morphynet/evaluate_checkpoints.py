"""
evaluate_checkpoints.py  (morphynet version)
============================================
Evaluates SSLM checkpoints from a Hugging Face repo against the MorphyNet dataset.

Produces:
  - Per-checkpoint CSV with precision/recall/f1 broken down by category
    (inflectional, derivational_suffix, derivational_prefix, derivational_all)
  - A summary CSV over all checkpoints
  - A multi-line plot of score evolution across checkpoints

Usage (see run_evaluate_checkpoints.sh for a ready-made SLURM invocation):
    python3 evaluate_checkpoints.py \
        --repo-id SakethVemula/my-sslm-model \
        --lang hin \
        --lang-long hin \
        --data-dir data \
        --dataset-path /home2/$USER/dataset \
        --token $HF_TOKEN \
        --morph-type all \
        --affix-type all
"""

import argparse
import os
import sys
import re
import gc
import json
import glob
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, HfApi

# ── import MorphyNetScore from this directory ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morphynet_score import MorphyNetScore

# ── configure pytorch memory ─────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("morphynet_evaluation_debug.log"),
    ]
)

# ── fairseq path ──────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FAIRSEQ_PATH = os.path.join(_REPO_ROOT, "fairseq")
if os.path.exists(_FAIRSEQ_PATH) and _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)

from fairseq import checkpoint_utils


# ─────────────────────────────────────────────────────────────────────────────
# Fairseq model helpers (identical to morphscore/evaluate_checkpoints.py)
# ─────────────────────────────────────────────────────────────────────────────

def setup_fairseq_modules():
    import fairseq
    import fairseq.models
    import importlib.util

    custom_models_dir = os.path.join(_REPO_ROOT, "fairseq", "models")

    ssmt_path = os.path.join(custom_models_dir, "ssmt", "__init__.py")
    if os.path.exists(ssmt_path):
        spec = importlib.util.spec_from_file_location("fairseq.models.ssmt", ssmt_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["fairseq.models.ssmt"] = module
        fairseq.models.ssmt = module
        spec.loader.exec_module(module)

    model_file = os.path.join(custom_models_dir, "transformer_sslm.py")
    if os.path.exists(model_file):
        if "transformer_sslm" not in fairseq.models.MODEL_REGISTRY:
            spec = importlib.util.spec_from_file_location("transformer_sslm_custom", model_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


from torch.nn.utils.rnn import pad_sequence

def get_model_segmentation_batch(model, task, texts: list, device):
    """Run SSLM model in segment mode on a batch and return lists of segment strings."""
    tgt_dict = task.target_dictionary

    def char_tokenize(line):
        return list(line.strip())

    eos_token = tgt_dict.eos()
    batch_target_tokens = []
    
    for text in texts:
        tokens = tgt_dict.encode_line(
            text, line_tokenizer=char_tokenize, add_if_not_exist=False, append_eos=False
        ).long()
        target_tokens = torch.cat([tokens, torch.tensor([eos_token])])
        batch_target_tokens.append(target_tokens)

    pad_idx = tgt_dict.pad()
    target_tokens_padded = pad_sequence(batch_target_tokens, batch_first=True, padding_value=pad_idx).to(device)

    sos_tensor = torch.full((len(texts), 1), eos_token, dtype=torch.long, device=device)
    prev_output_tokens = torch.cat([sos_tensor, target_tokens_padded], dim=1)

    with torch.no_grad():
        split_indices_batch = model(prev_output_tokens, mode="segment")

    batch_segments = []
    for i, seq_target in enumerate(batch_target_tokens):
        # Reconstruct raw text from exact unpadded tokens
        raw_text = ""
        target_list = seq_target.tolist()
        valid_indices = target_list[:-1] if target_list[-1] == eos_token else target_list
        for idx in valid_indices:
            raw_text += tgt_dict.symbols[idx]
        raw_text = raw_text.replace("</s>", " ")

        indices = split_indices_batch[i]
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        segments = []
        last_idx = 0
        for end_idx in indices:
            seg = raw_text[last_idx: end_idx + 1]
            segments.append(seg)
            last_idx = end_idx + 1

        segments = [s for s in segments if s]
        batch_segments.append(segments)

    return batch_segments


def get_ckpt_num(filename: str):
    """Parse checkpoint filename into a sortable (epoch, updates) tuple."""
    m = re.search(r'checkpoint_(\d+)_(\d+)\.pt', filename)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'checkpoint(\d+)\.pt', filename)
    if m:
        return (int(m.group(1)), float('inf'))
    return (-1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(model, task, device,
                        infl_entries, deriv_entries,
                        ms: MorphyNetScore,
                        morph_type: str, affix_type: str,
                        batch_size: int = 128):
    """
    Evaluate a single model checkpoint against MorphyNet entries.

    Returns dict:  { category -> {precision, recall, f1, details} }
    """
    def segment_fn(wordforms: list):
        try:
            segs_batch = get_model_segmentation_batch(model, task, wordforms, device)
            return segs_batch
        except Exception as e:
            return [[wf] for wf in wordforms]

    return ms.evaluate_breakdown(
        infl_entries=infl_entries,
        deriv_entries=deriv_entries,
        segment_fn=segment_fn,
        morph_type=morph_type,
        affix_type=affix_type,
        batch_size=batch_size,
    )


def process_epoch(label: str, filename: str, args,
                  infl_entries, deriv_entries,
                  ms: MorphyNetScore, hf_token: str):
    """Download, load, evaluate and cleanup one checkpoint."""
    logging.info(f"Processing {label} ({filename}) ...")

    ckpt_path = model = task = models = None
    try:
        # Download
        try:
            ckpt_path = hf_hub_download(
                repo_id=args.repo_id, filename=filename,
                token=hf_token, local_dir="temp_eval_ckpts"
            )
        except Exception as e:
            logging.error(f"Failed to download {filename}: {e}")
            return None

        # Load model
        data_override = os.path.join(args.dataset_path, args.lang, "bin")
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path],
            arg_overrides={"data": data_override, "vocabs_path": data_override}
        )
        model = models[0]
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        device = next(model.parameters()).device

        # Evaluate
        results = evaluate_checkpoint(
            model, task, device,
            infl_entries, deriv_entries, ms,
            args.morph_type, args.affix_type,
            args.batch_size
        )

        # Log summary
        for cat, res in results.items():
            logging.info(
                f"  {cat}: P={res['precision']:.4f} R={res['recall']:.4f} "
                f"F1={res['f1']:.4f}  (n={res['n_evaluated']}/{res['n_total']})"
            )

        # Save per-checkpoint instance details CSV
        out_dir = f"{args.lang}_morphynet_results"
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(filename).replace('.pt', '')
        for cat, res in results.items():
            df = pd.DataFrame(res['details'])
            csv_path = os.path.join(out_dir, f"{base}_{cat}_segmentations.csv")
            df.to_csv(csv_path, index=False)
            logging.info(f"  Saved instance details to {csv_path}")

        return results

    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

    finally:
        if model is not None:
            model.cpu()
            del model
        if task is not None:
            del task
        if models is not None:
            del models
        torch.cuda.empty_cache()
        gc.collect()

        if ckpt_path and os.path.exists(ckpt_path):
            try:
                os.remove(ckpt_path)
            except OSError as e:
                logging.warning(f"Could not remove checkpoint file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SSLM checkpoints on MorphyNet morphological alignment"
    )
    parser.add_argument("--repo-id",      required=True,  help="HuggingFace repo ID")
    parser.add_argument("--lang",         required=True,  help="ISO lang code used in checkpoint paths (e.g. hin)")
    parser.add_argument("--lang-long",    required=True,  help="Language folder name inside data-dir (e.g. hin or hindi)")
    parser.add_argument("--data-dir",     default="data", help="Root of morphynet data dir (default: data)")
    parser.add_argument("--token",        default=None,   help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--dataset-path", required=True,  help="Path containing fairseq binarised data per language")
    parser.add_argument("--morph-type",   default="all",  choices=["inflectional", "derivational", "all"],
                        help="Which morphology type to evaluate")
    parser.add_argument("--affix-type",   default="all",  choices=["suffix", "prefix", "all"],
                        help="Derivational affix type filter (ignored for inflectional)")
    parser.add_argument("--batch-size",   type=int, default=128,
                        help="Number of wordforms to segment per batch")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN")

    setup_fairseq_modules()

    # ── Load MorphyNet data ──────────────────────────────────────────────────
    lang_data_dir = os.path.join(args.data_dir, args.lang_long)
    ms = MorphyNetScore()

    infl_entries = deriv_entries = None

    if args.morph_type in ("inflectional", "all"):
        infl_path = os.path.join(lang_data_dir, f"{args.lang_long}.inflectional.v1.tsv")
        if not os.path.exists(infl_path):
            logging.warning(f"Inflectional file not found: {infl_path}")
        else:
            infl_entries = ms.load_inflectional(infl_path)
            logging.info(f"Loaded {len(infl_entries)} inflectional entries from {infl_path}")

    if args.morph_type in ("derivational", "all"):
        deriv_path = os.path.join(lang_data_dir, f"{args.lang_long}.derivational.v1.tsv")
        if not os.path.exists(deriv_path):
            logging.warning(f"Derivational file not found: {deriv_path}")
        else:
            deriv_entries = ms.load_derivational(deriv_path)
            logging.info(f"Loaded {len(deriv_entries)} derivational entries from {deriv_path}")

    if not infl_entries and not deriv_entries:
        logging.error("No data loaded. Check --data-dir and --lang-long.")
        sys.exit(1)

    # ── Fetch checkpoint list from HF ────────────────────────────────────────
    logging.info(f"Fetching checkpoint list from {args.repo_id} ...")
    api = HfApi(token=hf_token)
    try:
        all_files = list(api.list_repo_files(repo_id=args.repo_id))
    except Exception as e:
        logging.error(f"Could not list repo files: {e}")
        sys.exit(1)

    ckpt_files = []
    for f in all_files:
        if not (f.startswith(f"{args.lang}/") or f"/{args.lang}/" in f):
            continue
        if not f.endswith(".pt"):
            continue
        num = get_ckpt_num(f)
        if num[0] > 0:
            ckpt_files.append((num, f))

    ckpt_files.sort(key=lambda x: x[0])
    logging.info(f"Found {len(ckpt_files)} checkpoint(s).")

    if not ckpt_files:
        logging.error("No checkpoints found. Check --lang and --repo-id.")
        sys.exit(1)

    # ── Evaluate loop ────────────────────────────────────────────────────────
    # Accumulators: {category -> {x_labels, precisions, recalls, f1s}}
    summary: dict = {}
    checkpoint_labels = []
    checkpoint_filenames = []

    for (epoch, updates), filename in ckpt_files:
        label = f"Epoch {epoch}" if updates == float('inf') else f"Epoch {epoch} (Step {updates})"
        x_val = f"E{epoch}" if updates == float('inf') else f"E{epoch}_S{updates}"

        results = process_epoch(label, filename, args,
                                infl_entries, deriv_entries, ms, hf_token)
        gc.collect()
        torch.cuda.empty_cache()

        if results is None:
            continue

        checkpoint_labels.append(x_val)
        checkpoint_filenames.append(filename)

        for cat, res in results.items():
            if cat not in summary:
                summary[cat] = {'precisions': [], 'recalls': [], 'f1s': []}
            summary[cat]['precisions'].append(res['precision'])
            summary[cat]['recalls'].append(res['recall'])
            summary[cat]['f1s'].append(res['f1'])

    if not checkpoint_labels:
        logging.error("No checkpoints were successfully evaluated.")
        sys.exit(1)

    # ── Save aggregate CSV ───────────────────────────────────────────────────
    rows = []
    for i, (label, fname) in enumerate(zip(checkpoint_labels, checkpoint_filenames)):
        row = {'checkpoint': fname, 'label': label}
        for cat, data in summary.items():
            if i < len(data['precisions']):
                row[f'{cat}_precision'] = data['precisions'][i]
                row[f'{cat}_recall']    = data['recalls'][i]
                row[f'{cat}_f1']        = data['f1s'][i]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_out = f"{args.lang}_morphynet_scores.csv"
    summary_df.to_csv(csv_out, index=False)
    logging.info(f"Summary CSV saved to {csv_out}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10.colors
    cat_names = list(summary.keys())

    for ci, cat in enumerate(cat_names):
        data = summary[cat]
        color = colors[ci % len(colors)]
        xs = checkpoint_labels[:len(data['f1s'])]
        plt.plot(xs, data['f1s'],       label=f'{cat} F1',
                 color=color, marker='o', linewidth=2)
        plt.plot(xs, data['precisions'], label=f'{cat} P',
                 color=color, marker='^', linestyle='--', linewidth=1, alpha=0.7)
        plt.plot(xs, data['recalls'],    label=f'{cat} R',
                 color=color, marker='s', linestyle=':', linewidth=1, alpha=0.7)

    plt.xlabel('Checkpoint')
    plt.ylabel('Score')
    plt.title(f'MorphyNet Score Evolution — {args.lang_long}')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    plot_out = f"{args.lang}_morphynet_evolution.png"
    plt.savefig(plot_out, dpi=150)
    logging.info(f"Plot saved to {plot_out}")


if __name__ == "__main__":
    main()
