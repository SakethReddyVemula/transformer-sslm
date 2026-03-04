#!/bin/bash
#SBATCH --job-name=morphynet-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --output=morphynet_eval_%j.txt
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode078

# ─────────────────────────────────────────────────────────────────────────────
# MorphyNet Evaluation Script
#
# Evaluates SSLM model checkpoints from a Hugging Face repo against the
# MorphyNet dataset. Produces:
#   - Per-checkpoint CSVs with instance-level segmentation details
#   - A summary CSV across all checkpoints
#   - A multi-category score evolution plot
#
# To submit: sbatch run_evaluate_checkpoints.sh
# To run locally (no SLURM): bash run_evaluate_checkpoints.sh
# ─────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=0

# Activate the SSLM virtual environment
source /home2/$USER/sslm-venv/bin/activate

# ── Language Configuration ────────────────────────────────────────────────────
# LANG: ISO code used in checkpoint filenames and HF repo paths
# LANG_LONG: folder name under data/ in the morphynet data directory
#            (must match {LANG_LONG}.inflectional.v1.tsv and .derivational.v1.tsv)
LANG=eng
LANG_LONG=eng

# ── HuggingFace Configuration ─────────────────────────────────────────────────
HF_REPO_ID=""        # e.g. "SakethVemula/my-sslm-model"
HF_TOKEN=""          # or export HF_TOKEN before submitting

# ── Morphology Filter ─────────────────────────────────────────────────────────
# --morph-type: inflectional | derivational | all
# --affix-type: suffix | prefix | all
#   (affix-type only affects derivational evaluation;
#    inflectional always evaluates all segments regardless)
MORPH_TYPE=all
AFFIX_TYPE=all

# ── Dataset for Fairseq Dict ──────────────────────────────────────────────────
DATASET_PATH="/home2/$USER/dataset"

# ── Paths ─────────────────────────────────────────────────────────────────────
# data-dir: root directory containing language subfolders with MorphyNet TSV files
#   e.g. data/eng/eng.inflectional.v1.tsv
#        data/eng/eng.derivational.v1.tsv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

# Create results directory
mkdir -p "${LANG}_morphynet_results"

# ── Run Evaluation ────────────────────────────────────────────────────────────
python3 "${SCRIPT_DIR}/evaluate_checkpoints.py" \
    --repo-id      "$HF_REPO_ID"     \
    --lang         "$LANG"           \
    --lang-long    "$LANG_LONG"      \
    --data-dir     "$DATA_DIR"       \
    --token        "$HF_TOKEN"       \
    --dataset-path "$DATASET_PATH"   \
    --morph-type   "$MORPH_TYPE"     \
    --affix-type   "$AFFIX_TYPE"

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm -rf ~/.cache/huggingface/hub/models--*/  2>/dev/null || true
