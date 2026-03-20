#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=mamba2_ssm_patch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=23:50:00
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

### 1. Setup Environment ###
source ~/.bashrc
conda activate mamba2-gpu

### 2. Define Paths ###
USERNAME="avivyossef"
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/mixing-mechs-results/mamba2_ssm_patch"

mkdir -p "$RESULTS_DIR"

### 3. Execution ###
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"
echo "Code Dir:     $REPO_DIR"
echo "Saving results to: $RESULTS_DIR"
echo "Experiment: SSM state patching (patching h, not residual stream)"
python -c "from transformers.models.mamba2.modeling_mamba2 import is_fast_path_available; print('Fast path (CUDA kernels) available:', is_fast_path_available)"

python run_mamba_layer_experiments.py \
    --model-id "AntonV/mamba2-2.7b-hf" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 100

echo "Job finished at: $(date)"
