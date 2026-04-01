#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=Zamba2-7B-Instruct-v2_inbal
#SBATCH --output=%x-%j.out        # Log file: bloomz-3b_inbal-12345.out
#SBATCH --error=%x-%j.err         # Error file
#SBATCH --partition=studentkillable
#SBATCH --time=20:00:00           # 20 hours max runtime
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

### 1. Setup Environment ###
source ~/.bashrc
source /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/miniconda3/bin/activate

# Activate your specific environment
conda activate mixing_pro_2026

### 2. Define Paths ###
USERNAME="inbalmoryles"
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"

# Model and Directories
MODEL_ID="Zyphra/Zamba2-7B-Instruct-v2"
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/results/${MODEL_ID}"

# Ensure the results directory actually exists
export HF_HOME="${VOL_BASE}/hf_cache"
mkdir -p "$HF_HOME"
mkdir -p "$RESULTS_DIR"

### 3. Execution ###
# CRITICAL: We move to the code directory so imports work
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"
echo "Base Volume:  $VOL_BASE"
echo "Code Dir:     $REPO_DIR"
echo "Saving results to: $RESULTS_DIR"

python run_layer_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 1000 \
    --hf-token "$HF_TOKEN"

echo "Job finished at: $(date)"