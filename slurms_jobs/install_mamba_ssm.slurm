#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=install_deps_inbal
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=1:00:00               # 1 hour is enough
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                    # Increased memory for compilation
#SBATCH --gres=gpu:1                 # GPU is REQUIRED to compile Mamba kernels

### 1. Setup Environment ###
# Points directly to your miniconda
source /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/miniconda3/bin/activate

# Activate your specific environment
conda activate mixing_pro_2026

### 2. Installation ###
echo "Running on node: $HOSTNAME"
echo "Date: $(date)"

# A. Fix the missing transformers error
echo "Installing base NLP libraries (transformers, etc.)..."
pip install transformers torch accelerate datasets

# B. Install Mamba dependencies (needed for Zamba and Mamba models)
echo "Installing mamba-ssm and causal-conv1d from PyPI..."
# We install these directly. Standard pip is usually safer than source install.
pip install causal-conv1d>=1.4.0 mamba-ssm

echo "Installation finished at: $(date)"

# 3. Verification
echo "Verifying installations..."
python -c 'import transformers; print("SUCCESS: transformers is ready!")'
python -c 'import mamba_ssm; print("SUCCESS: mamba_ssm is ready!")' || echo "WARNING: mamba_ssm failed (needed only for Mamba/Zamba)"

echo "Job completed at: $(date)"
