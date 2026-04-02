#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=falcon-h1-ablation-sweep
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

### 1. Setup Environment ###
source ~/.bashrc
conda activate zamba-env

### 2. Define Paths ###
USERNAME="avivyossef"
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"

MODEL_ID="tiiuae/Falcon-H1-3B-Instruct"
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/ablation-sweep/${MODEL_ID}"

mkdir -p "$RESULTS_DIR"

### 3. Execution ###
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"
echo "Saving results to: $RESULTS_DIR"

# Full sweep: both components x 5 noise scales x 5 layer ranges x 100 samples
python run_falcon_h1_ablation_baseline.py \
    --model-id "${MODEL_ID}" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 100 \
    --sweep

echo "Job finished at: $(date)"
