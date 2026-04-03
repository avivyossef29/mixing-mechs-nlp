#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=falcon-h1-attn-ranges
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

# Attention layers 17-23 (where lexical binding dominates in residual patching)
echo "=== Attention patched at layers 17-23 ==="
python run_falcon_h1_multi_layer_component_patching.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/attention_layers_17_23" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --component attention \
    --start-layer 17 \
    --end-layer 23

# Attention layers 24-30 (where reflexive binding dominates in residual patching)
echo "=== Attention patched at layers 24-30 ==="
python run_falcon_h1_multi_layer_component_patching.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/attention_layers_24_30" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --component attention \
    --start-layer 24 \
    --end-layer 30

echo "Job finished at: $(date)"
