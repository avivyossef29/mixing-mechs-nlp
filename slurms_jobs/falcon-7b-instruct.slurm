#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=falcon-7b-patch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

### 1. Setup Environment ###
source /vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/miniconda3/etc/profile.d/conda.sh
conda activate /vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/venv

### 2. Define Paths ###
USERNAME="inbalmoryles"
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
MODEL_ID="tiiuae/falcon-7b-instruct"
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/mixing-mechs-results/falcon-7b"

mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

export HF_HOME="${VOL_BASE}/hf_cache"

### 3. Execution ###
python run_layer_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 500 

echo "Job finished at: $(date)"
