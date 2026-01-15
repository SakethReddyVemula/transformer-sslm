#!/bin/bash
#SBATCH --job-name=transformer-sslm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=24:00:00
#SBATCH --output=tsslm_te_10000.txt
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode083

# Ensure each process sees unique GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate SSLM virtual environment
source /home2/$USER/sslm-venv/bin/activate

# Define directories - PLEASE UPDATE THESE PATHS
export PT_DATA_DIR="$HOME/data-bin"
export PT_MODEL_DIR="$HOME/model-bin"

# Create model directory if it doesn't exist
mkdir -p $PT_MODEL_DIR

get_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}

export MASTER_PORT=$(get_free_port)
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR 

# WandB Configuration
export WANDB_API_KEY="c8a7a539cb5fed3df89b21d71956ca6b4befd2a5"
export WANDB_PROJECT="tsslm-te-pretraining"

# Run Training
# Using torchrun for distributed training
torchrun  \
    --master_port $MASTER_PORT \
    --nproc_per_node 4 \
    --nnodes 1 \
    fairseq/fairseq_cli/train.py $PT_DATA_DIR/bin \
    --task subword_segmental_language_modeling \
    --arch transformer_sslm \
    --target-lang te --save-interval 5 \
    --criterion subword_segmental_lm_cross_entropy \
    --max-epoch 40 --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.1 --skip-invalid-size-inputs-valid-test \
    --tokens-per-sample 512 --sample-break-mode none \
    --max-tokens 4096 --vocabs-path $PT_MODEL_DIR --update-freq 64 \
    --keep-best-checkpoints 1  \
    --max-seg-len 5 --lexicon-max-size 10000 \
    --wandb-project $WANDB_PROJECT \
    --save-dir $PT_MODEL_DIR
