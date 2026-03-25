#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=mamba2_patch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=23:50:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

### 1. Setup Environment ###
source /vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/miniconda3/etc/profile.d/conda.sh
conda activate /vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/miniconda3/envs/mamba2-gpu

### 2. Define Paths ###
USERNAME="inbalmoryles"
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/mixing-mechs-results/mamba2"

mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

export HF_HOME="${VOL_BASE}/hf_cache"

### 3. Execution ###
python run_layer_experiments.py \
    --model-id "AntonV/mamba2-2.7b-hf" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 1000 \
    --start-layer 58

echo "Job finished at: $(date)"
