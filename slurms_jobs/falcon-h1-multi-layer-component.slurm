#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=falcon-h1-multi-layer-comp
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
BASE_RESULTS="${VOL_BASE}/multi-layer-component/${MODEL_ID}"

mkdir -p "$BASE_RESULTS"

### 3. Execution ###
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"

NUM_SAMPLES=200

# Patch mamba at ALL 32 layers simultaneously
echo "=== Mamba patched at ALL layers ==="
python run_falcon_h1_multi_layer_component_patching.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/mamba_all_layers" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --component mamba

# Patch attention at ALL 32 layers simultaneously
echo "=== Attention patched at ALL layers ==="
python run_falcon_h1_multi_layer_component_patching.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/attention_all_layers" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --component attention

echo "Job finished at: $(date)"
