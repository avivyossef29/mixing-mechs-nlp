#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=falcon-h1-component-patch
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
BASE_RESULTS="${VOL_BASE}/component-patching/${MODEL_ID}"

mkdir -p "$BASE_RESULTS"

### 3. Execution ###
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"

NUM_SAMPLES=200

# Run all three: residual (baseline), mamba, attention
# Each patches the specified component's output with the CF's version
# and classifies the result as no_effect/lexical/positional/reflexive.

echo "=== Residual stream patching (baseline for comparison) ==="
python run_falcon_h1_component_patching.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/residual" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --component residual

echo "=== Mamba component patching ==="
python run_falcon_h1_component_patching.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/mamba" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --component mamba

echo "=== Attention component patching ==="
python run_falcon_h1_component_patching.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/attention" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --component attention

echo "Job finished at: $(date)"
