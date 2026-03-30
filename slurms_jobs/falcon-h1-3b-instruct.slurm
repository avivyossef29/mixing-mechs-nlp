#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=falcon-h1-3b-instruct_inbal
#SBATCH --output=%x-%j.out        # Log file: falcon-h1-3b-instruct_inbal-12345.out
#SBATCH --error=%x-%j.err         # Error file
#SBATCH --partition=studentkillable
#SBATCH --time=20:00:00           # 20 hours max runtime
#SBATCH --gres=gpu:1              # 1 GPU needed
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

### 1. Setup Environment ###
source /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/miniconda3/bin/activate

# Activate your specific environment
conda activate mixing_env

### 2. Define Paths ###
USERNAME="inbalmoryles"
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"

# Model and Directories - Now pointing directly to your cloned repo!
MODEL_ID="tiiuae/Falcon-H1-3B-Instruct"
REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"
RESULTS_DIR="${VOL_BASE}/results/${MODEL_ID}"

# Ensure the results directory actually exists
mkdir -p "$RESULTS_DIR"

### 3. Execution ###
# CRITICAL: We move to the newly cloned code directory so imports work
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
    --num-samples 3000 

echo "Job finished at: $(date)"
