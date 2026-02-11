#!/bin/bash
#SBATCH --job-name=eng-fertility
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --output=eng-fertility.txt
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode076

# Ensure each process sees unique GPUs
export CUDA_VISIBLE_DEVICES=0

# Activate SSLM virtual environment
source /home2/$USER/sslm-venv/bin/activate

# Choose the language's ISO code
LANG=eng

# Setup huggingface repo
export HF_TOKEN=""
export HF_REPO_ID=""

get_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}

export MASTER_PORT=$(get_free_port)
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR 

torchrun \
    --master_port $MASTER_PORT \
    --nproc_per_node 1 \
    --nnodes 1 \
    evaluate_fertility.py \
    --repo-id $HF_REPO_ID \
    --lang $LANG \
    --token $HF_TOKEN \
    --save_csv "fertility_results_${LANG}.csv" \
    --save_png "fertility_evolution_${LANG}.png"

echo "Done, successful fertility evaluation"