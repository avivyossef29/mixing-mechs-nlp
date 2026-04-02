#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=falcon-h1-noise-mechanism
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
BASE_RESULTS="${VOL_BASE}/ablation-mechanism"

mkdir -p "$BASE_RESULTS"

### 3. Execution ###
cd "$REPO_DIR"

echo "Running on node: $HOSTNAME"
echo "Date: $(date)"

NUM_SAMPLES=200
NOISE_SCALE=5

# Targeted experiment: at each transition layer, add noise to ONE component
# and run the full patching experiment to see if the transition survives.
#
# Baseline transitions (from prior experiment):
#   Layer 17: no_effect -> lexical    (binding first appears)
#   Layer 22: lexical -> reflexive    (binding type changes)
#
# We test: does the transition survive when we corrupt mamba vs attention?

# --- Transition 1: Layer 17 (binding appears) ---
# Noise on mamba at layer 17 only, patch sweep around the transition
echo "=== Noise on MAMBA @ layer 17, patching layers 15-19 ==="
python run_falcon_h1_ablation_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/noise_mamba_at_17" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --ablate mamba \
    --ablate-method noise \
    --noise-scale $NOISE_SCALE \
    --ablate-layers "17" \
    --start-layer 15 \
    --end-layer 19

echo "=== Noise on ATTENTION @ layer 17, patching layers 15-19 ==="
python run_falcon_h1_ablation_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/noise_attn_at_17" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --ablate attention \
    --ablate-method noise \
    --noise-scale $NOISE_SCALE \
    --ablate-layers "17" \
    --start-layer 15 \
    --end-layer 19

# --- Transition 2: Layer 22 (lexical -> reflexive) ---
echo "=== Noise on MAMBA @ layer 22, patching layers 20-24 ==="
python run_falcon_h1_ablation_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/noise_mamba_at_22" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --ablate mamba \
    --ablate-method noise \
    --noise-scale $NOISE_SCALE \
    --ablate-layers "22" \
    --start-layer 20 \
    --end-layer 24

echo "=== Noise on ATTENTION @ layer 22, patching layers 20-24 ==="
python run_falcon_h1_ablation_experiments.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${BASE_RESULTS}/noise_attn_at_22" \
    --num-samples $NUM_SAMPLES \
    --num-instances 20 \
    --ablate attention \
    --ablate-method noise \
    --noise-scale $NOISE_SCALE \
    --ablate-layers "22" \
    --start-layer 20 \
    --end-layer 24

# --- Per-layer sweep: noise at each individual layer, patch at that layer ---
echo "=== Per-layer noise sweep ==="
for LAYER in $(seq 15 26); do
    echo "--- Noise mamba @ layer $LAYER, patch @ layer $LAYER ---"
    python run_falcon_h1_ablation_experiments.py \
        --model-id "${MODEL_ID}" \
        --output-dir "${BASE_RESULTS}/perlayer_mamba/noise_at_${LAYER}" \
        --num-samples $NUM_SAMPLES \
        --num-instances 20 \
        --ablate mamba \
        --ablate-method noise \
        --noise-scale $NOISE_SCALE \
        --ablate-layers "$LAYER" \
        --start-layer $LAYER \
        --end-layer $LAYER

    echo "--- Noise attn @ layer $LAYER, patch @ layer $LAYER ---"
    python run_falcon_h1_ablation_experiments.py \
        --model-id "${MODEL_ID}" \
        --output-dir "${BASE_RESULTS}/perlayer_attn/noise_at_${LAYER}" \
        --num-samples $NUM_SAMPLES \
        --num-instances 20 \
        --ablate attention \
        --ablate-method noise \
        --noise-scale $NOISE_SCALE \
        --ablate-layers "$LAYER" \
        --start-layer $LAYER \
        --end-layer $LAYER
done

echo "Job finished at: $(date)"
