#!/bin/bash
#SBATCH --job-name=llama-3
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250063p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100-80:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/outputs/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2/shared/arc/outputs/logs/%x_%j.err

# --- Paths ---
PROJECT_ROOT=/ocean/projects/cis250063p/jbentley/ARC-AGI-2/Capstone-ARC2
OUTPUT_ROOT=$PROJECT_ROOT/shared/arc/outputs
CACHE_ROOT=$PROJECT_ROOT/shared/arc/cache
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/runs" "$CACHE_ROOT/hf" "$CACHE_ROOT/ds"
cd "$PROJECT_ROOT" || exit 1

# --- Conda env ---
eval "$(conda shell.bash hook)"
conda activate arc-env

# --- Hugging Face caches (no deprecated TRANSFORMERS_CACHE) ---
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/ds"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

# --- Runtime sanity & stability ---
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK>2 ? SLURM_CPUS_PER_TASK-2 : 1 ))
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1

# If deepspeed is installed, keep it inert so it doesn't try to compile CUDA ops
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export DS_BUILD_AIO=0

# --- Info dump (handy for debugging) ---
echo "Host: $(hostname)"
nvidia-smi || true
python - <<'PY'
import torch, os
print("cuda_available:", torch.cuda.is_available(), "num_gpus:", torch.cuda.device_count())
print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"), "RANK:", os.environ.get("RANK"), "WORLD_SIZE:", os.environ.get("WORLD_SIZE"))
PY

# --- Launch training with DDP (ranks = GPUs on node) ---
MASTER_PORT=${MASTER_PORT:-29501}

echo "Launching torchrun with 4 ranks on $(hostname)"
torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  main/code/arc-trainer/train_v9_stage1.py
