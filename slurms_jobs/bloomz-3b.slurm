#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=bloomz_patch
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/%x-%j.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

### 1. Define Paths ###
USERNAME="inbalmoryles"

# Code location (your specific folder)
REPO_DIR="/specific/a/home/cc/students/csguests/${USERNAME}/mixing-mechs-nlp"

# Results location (in your Volume)
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
MODEL_ID="bigscience/bloomz-3b"
RESULTS_DIR="${VOL_BASE}/${MODEL_ID}"

mkdir -p "$RESULTS_DIR"

### 2. Execution ###
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"
echo "Code Dir:     $REPO_DIR"
echo "Saving results to: $RESULTS_DIR"

# Using Aviv's Python directly
/vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/venv/bin/python run_layer_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 3000

echo "Job finished at: $(date)"
