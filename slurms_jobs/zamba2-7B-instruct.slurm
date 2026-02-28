#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=zamba2-7B-instruct_patch
#SBATCH --output=%x-%j.out        # Log file: zamba2-7B-instruct_patch-12345.out
#SBATCH --error=%x-%j.err         # Error file
#SBATCH --partition=studentkillable
#SBATCH --time=12:00:00           # 12 hours max runtime
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

### 1. Setup Environment ###
source ~/.bashrc

# Activate your specific environment (using the underscore name)
conda activate /vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/venv

### 2. Define Paths ###
USERNAME="avivyossef"

# A. Base Volume Path (The Root)
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"

# B. Derived Paths
# We assume your code folder is named 'mixing-mechs-nlp' inside your volume
MODEL_ID="Zyphra/Zamba2-7B-instruct"
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/${MODEL_ID}"

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

python run_layer_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 1000 

echo "Job finished at: $(date)"
