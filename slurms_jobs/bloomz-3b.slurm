#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=bloomz_patch
#SBATCH --output=%x-%j.out        # Log file
#SBATCH --error=%x-%j.err         # Error file
#SBATCH --partition=studentkillable
#SBATCH --time=12:00:00           # 12 hours max runtime
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

### 1. Setup Environment ###
# source ~/.bashrc
# conda activate /vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/venv

### 2. Define Paths ###
USERNAME="inbalmoryles"

# הנתיב שבו נמצא הקוד שלך (לפי מה ששלחת)
REPO_DIR="/specific/a/home/cc/students/csguests/${USERNAME}/mixing-mechs-nlp"

# הנתיב שבו יישמרו התוצאות (ב-Volume שלך)
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
MODEL_ID="bigscience/bloomz-3b"
RESULTS_DIR="${VOL_BASE}/${MODEL_ID}"

# Ensure the results directory actually exists
mkdir -p "$RESULTS_DIR"

### 3. Execution ###
# CRITICAL: We move to the code directory so imports work
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"
echo "Code Dir:     $REPO_DIR"
echo "Saving results to: $RESULTS_DIR"

# שימוש בפייתון של אביב להרצת הקוד שלך
/vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/venv/bin/python run_layer_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "$RESULTS_DIR" \
    --num-instances 20 \
    --num-samples 3000 

echo "Job finished at: $(date)"
