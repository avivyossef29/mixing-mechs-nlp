#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=Falcon-H1-3B_inbal
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=studentkillable
#SBATCH --array=0-31
#SBATCH --time=23:50:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

### 1. Setup Environment ###
source /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/miniconda3/etc/profile.d/conda.sh
conda activate /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/envs/zamba-env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

### 2. Define Paths ###
USERNAME="inbalmoryles"
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"

# Model and Directories
MODEL_ID="tiiuae/Falcon-H1-3B-Instruct"
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/mixing-mechs-results/run_100boxes_3000samples/${MODEL_ID}"

# Cache & Setup
export HF_HOME="${VOL_BASE}/hf_cache"
export TRITON_CACHE_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/triton_cache"
export TMPDIR="${VOL_BASE}/tmp"
export PIP_CACHE_DIR="${VOL_BASE}/pip_cache"
export HF_TOKEN=$(cat "${VOL_BASE}/hf_token.txt")

mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR" "$RESULTS_DIR" "$PIP_CACHE_DIR" "$HF_HOME"

### 3. Execution ###
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"
echo "Model: $MODEL_ID"
echo "Saving results to: $RESULTS_DIR"
echo "Running Layer: $SLURM_ARRAY_TASK_ID"

python run_layer_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 100 \
    --num-samples 3000 \
    --hf-token "$HF_TOKEN" \
    --start-layer $SLURM_ARRAY_TASK_ID \
    --end-layer $SLURM_ARRAY_TASK_ID

echo "Job finished at: $(date)"