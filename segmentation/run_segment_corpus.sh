#!/bin/bash
#SBATCH --job-name=seg_mal_tel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --output=seg_mal_tel.out
#SBATCH --error=seg_mal_tel.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode075

# Setup
source /home2/$USER/sslm-venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

# Configuration
LANGS=("mal" "tel")

# Identities
HF_REPO_ID=${HF_REPO_ID:-"SakethVemula/sslm-models"} 
HF_UPLOAD_REPO_ID="SakethVemula/sslm-corpus-segments"
export HF_TOKEN="" 

echo "Starting corpus segmentation evaluation loop..."
echo "Languages: ${LANGS[*]}"
echo "Validation Data Base: /home2/$USER/dataset/"
echo "Source Repo: $HF_REPO_ID"
echo "Target Repo: $HF_UPLOAD_REPO_ID"

get_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

for LANG_CODE in "${LANGS[@]}"; do
    echo "------------------------------------------------"
    echo "Processing $LANG_CODE"
    echo "------------------------------------------------"
    
    DATA_PATH=/home2/$USER/dataset/${LANG_CODE}/test.${LANG_CODE}
    if [ ! -f "$DATA_PATH" ]; then
        DATA_PATH=/home2/$USER/dataset/${LANG_CODE}/valid.${LANG_CODE}
    fi
    # Use corpus_segmentations subdir for the corpus outputs
    OUTPUT_DIR="results/corpus_segmentations/${LANG_CODE}"
    mkdir -p $OUTPUT_DIR
    
    export MASTER_PORT=$(get_free_port)
    
    # We use torchrun to support multiple processes if needed and to match previous scripts
    torchrun \
        --master_port $MASTER_PORT \
        --nproc_per_node 1 \
        --nnodes 1 \
        segment_corpus.py \
        --repo-id "$HF_REPO_ID" \
        --upload-repo-id "$HF_UPLOAD_REPO_ID" \
        --lang "$LANG_CODE" \
        --data-path "$DATA_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 512
        
    echo "Finished $LANG_CODE"
done

echo "Job finished."
