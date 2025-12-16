#!/bin/bash
#SBATCH --job-name=arc_train
#SBATCH --output=/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/outputs/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/outputs/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-80:3  # Request 3 GPUs
#SBATCH -p GPU-shared
#SBATCH -A cis250063p




# --- Setup paths ---
PROJECT_ROOT=/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2
OUTPUT_ROOT=$PROJECT_ROOT/shared/arc/outputs
CACHE_ROOT=$PROJECT_ROOT/shared/arc/cache

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/runs" "$CACHE_ROOT/hf" "$CACHE_ROOT/ds"

cd "$PROJECT_ROOT" || exit 1

# --- Conda env ---
eval "$(conda shell.bash hook)"
conda activate arc-env

# --- Caches (Hugging Face) ---
export HF_HOME="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/ds"


# Set environment variables for better stability
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#Load your site’s CUDA module (example names; use what `module avail` shows)
module avail cuda
module load cuda/12.6.1

#Make sure CUDA_HOME is set (modules usually do this, but be explicit if needed)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6.1}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# # --- Torchrun (single node, 4 GPUs) ---
# MASTER_PORT=${MASTER_PORT:-29501}
# echo "Starting torchrun on $(hostname) with 3 GPUs..."
# torchrun \
#   --nproc_per_node=3 \
#   --master_port=$MASTER_PORT \
#   main/code/arc-trainer/train_v1.py
deepspeed --num_gpus=3 main/code/arc-trainer/train_v1.py