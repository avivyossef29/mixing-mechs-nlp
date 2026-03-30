#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=mamba2_inbal
#SBATCH --output=%x-%j.out        # Log file: mamba2_inbal-12345.out
#SBATCH --error=%x-%j.err         # Error file
#SBATCH --partition=studentkillable
#SBATCH --time=20:00:00           # 20 hours max runtime
#SBATCH --gres=gpu:1              # 1 GPU (CRITICAL for Mamba kernels!)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

### 1. Setup Environment ###
source /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/miniconda3/bin/activate
conda activate mixing_pro_2026

### 2. Define Paths ###
USERNAME="inbalmoryles"
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"

MODEL_ID="tiiuae/falcon-7b-instruct"
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/results/${MODEL_ID}"

# Ensure the results directory actually exists
mkdir -p "$RESULTS_DIR"

### 3. Execution ###
# CRITICAL: We move to the code directory so imports work
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"
echo "Base Volume:  $VOL_BASE"
echo "Code Dir:     $REPO_DIR"
echo "Saving results to: $RESULTS_DIR"
echo "Note: Ensure Mamba2 GPU kernels (mamba-ssm) are installed in mixing_env"

export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/hf_cache"
python ../run_layer_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 3000 \
    --start-layer 58

echo "Job finished at: $(date)"
