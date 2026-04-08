#!/bin/bash
#SBATCH --job-name=gemma-2-2b-it_inbal
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# 1. Environment & Path Setup
export USERNAME="inbalmoryles"
export VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
export REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"

# Using explicit python path from the Volume environment
export MY_PYTHON="${VOL_BASE}/envs/transformers-env/bin/python"

# Model & Results config
MODEL_ID="google/gemma-2-2b-it"
RESULTS_DIR="${VOL_BASE}/results/${MODEL_ID}"

# HF Configuration
export HF_HOME="${VOL_BASE}/hf_cache"
export TMPDIR="${VOL_BASE}/tmp"
export HF_TOKEN=$(cat "${VOL_BASE}/hf_token.txt")

mkdir -p "$RESULTS_DIR" "$TMPDIR"

# 2. Execution
cd "$REPO_DIR"

echo "Node: $HOSTNAME | Python: $MY_PYTHON"

# Verify environment visibility
$MY_PYTHON -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Run experiment using absolute python path
$MY_PYTHON run_layer_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 3000 \
    --hf-token "$HF_TOKEN"

echo "Job finished at: $(date)"