#!/bin/bash

set -e

########################################
# GPU setup (single-process training)
########################################
# Choose ONE GPU (change if needed)
export CUDA_VISIBLE_DEVICES=0

########################################
# Activate environment
########################################
source /home/vishnuraj/saketh/sslm-venv/bin/activate

########################################
# Language + paths
########################################
LANG=mal
export PT_DATA_DIR="/home/vishnuraj/saketh/dataset/${LANG}"

rm -rf "$PT_DATA_DIR/bin"
mkdir -p "$PT_DATA_DIR/bin"

########################################
# Preprocess
########################################
python3 /home/vishnuraj/saketh/fairseq/fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $PT_DATA_DIR/train.${LANG} \
    --validpref $PT_DATA_DIR/valid.${LANG} \
    --testpref $PT_DATA_DIR/test.${LANG} \
    --destdir $PT_DATA_DIR/bin

########################################
# Vocab + save directories
########################################
export PT_VOCAB_DIR="/home/vishnuraj/saketh/dataset/${LANG}/bin"

# Local scratch replacement (no SLURM_TMPDIR)
LOCAL_TMP="/home/vishnuraj/saketh/tmp_sslm_${LANG}"
rm -rf "$LOCAL_TMP"
mkdir -p "$LOCAL_TMP"

########################################
# HuggingFace cache
########################################
export HF_HOME="${LOCAL_TMP}/.cache/huggingface"
mkdir -p "$HF_HOME"

########################################
# Save dir
########################################
export PT_SAVE_DIR="${LOCAL_TMP}/${LANG}"
mkdir -p "$PT_SAVE_DIR"
mkdir -p "$PT_VOCAB_DIR"

########################################
# HF + WandB
########################################
export HF_TOKEN=""
export HF_REPO_ID=""
export HF_SUBFOLDER="${LANG}"
export HF_DELETE_LOCAL=1

export WANDB_API_KEY=""
export WANDB_PROJECT="tsslm-pretraining-v2"
export WANDB_NAME="${LANG}_sslm"

########################################
# Model hyperparameters
########################################
DECODER_LAYERS=3
DECODER_EMBED_DIM=128
DECODER_FFN_EMBED_DIM=512
DECODER_ATTENTION_HEADS=4
MAX_SEG_LEN=15

echo "Training with reduced model size:"
echo "Layers: $DECODER_LAYERS"
echo "Embed Dim: $DECODER_EMBED_DIM"
echo "FFN Dim: $DECODER_FFN_EMBED_DIM"
echo "Heads: $DECODER_ATTENTION_HEADS"
echo "Max Seg Len: $MAX_SEG_LEN"

########################################
# SINGLE-GPU training (no torchrun)
########################################
python /home/vishnuraj/saketh/fairseq/fairseq_cli/train.py $PT_DATA_DIR/bin \
    --task subword_segmental_language_modeling \
    --arch transformer_sslm \
    --target-lang ${LANG} \
    --save-interval 1 \
    --save-interval-updates 50 \
    --keep-interval-updates -1 \
    --validate-interval-updates 1000 \
    --criterion subword_segmental_lm_cross_entropy \
    --max-epoch 25 \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 1000 \
    --warmup-init-lr 1e-07 \
    --dropout 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --tokens-per-sample 512 \
    --sample-break-mode none \
    --max-tokens 16384 \
    --vocabs-path $PT_VOCAB_DIR \
    --update-freq 1 \
    --keep-last-epochs -1 \
    --patience 2 \
    --max-seg-len $MAX_SEG_LEN \
    --lexicon-max-size 10000 \
    --log-interval 1 \
    --fp16 \
    --num-workers 1 \
    --wandb-project $WANDB_PROJECT \
    --save-dir $PT_SAVE_DIR \
    --decoder-layers $DECODER_LAYERS \
    --decoder-embed-dim $DECODER_EMBED_DIM \
    --decoder-ffn-embed-dim $DECODER_FFN_EMBED_DIM \
    --decoder-attention-heads $DECODER_ATTENTION_HEADS

########################################
# Cleanup
########################################
rm -rf "$LOCAL_TMP"