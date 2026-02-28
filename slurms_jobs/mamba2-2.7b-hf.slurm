#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=mamba2_patch
#SBATCH --output=%x-%j.out        # Log file: mamba2_patch-12345.out
#SBATCH --error=%x-%j.err         # Error file
#SBATCH --partition=studentkillable
#SBATCH --time=23:50:00           # 20 hours max runtime
#SBATCH --gres=gpu:1              # 1 GPU (CRITICAL for Mamba kernels!)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

### 1. Setup Environment ###
source ~/.bashrc

# Activate the Mamba2 GPU environment (where mamba-ssm kernels are installed)
conda activate mamba2-gpu

### 2. Define Paths ###
USERNAME="avivyossef"

# A. Base Volume Path (The Root)
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"

# B. Derived Paths
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/mixing-mechs-results"

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
echo "Using Mamba2 GPU kernels (mamba-ssm installed in mamba2-gpu env)"

python run_layer_experiments.py \
    --model-id "AntonV/mamba2-2.7b-hf" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 1000 \
    --start-layer 58

echo "Job finished at: $(date)"
