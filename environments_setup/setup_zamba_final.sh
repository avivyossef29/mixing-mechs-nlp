#!/bin/bash
#SBATCH --job-name=setup_zamba
#SBATCH --output=setup_zamba_%j.out
#SBATCH --error=setup_zamba_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Path Definitions
export USERNAME="inbalmoryles"
export VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
export MY_ENV_PATH="${VOL_BASE}/envs/zamba-env"
export REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"

# Quota & Permission Protections
export PYTHONNOUSERSITE=1
export CONDA_ENVS_PATH="${VOL_BASE}/envs"
export CONDA_PKGS_DIRS="${VOL_BASE}/conda_pkgs"
export PIP_CACHE_DIR="${VOL_BASE}/pip_cache"
export HF_HOME="${VOL_BASE}/hf_cache"
export TMPDIR="${VOL_BASE}/tmp"

mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

# Initialize Conda
source "${VOL_BASE}/miniconda3/etc/profile.d/conda.sh"

echo "=== Step 1: Create Environment ==="
conda create --prefix "$MY_ENV_PATH" python=3.10 -y
conda activate "$MY_ENV_PATH"
which pip

echo "=== Step 2: Install Core Build Dependencies ==="
pip install --no-cache-dir numpy==1.26.4 packaging ninja setuptools wheel
pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 3: Compile Mamba Kernels (No Build Isolation) ==="
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
pip install --no-cache-dir --no-build-isolation causal-conv1d==1.6.0 mamba-ssm==2.1.0

echo "=== Step 4: Final Environment Sync ==="
conda env update --prefix "$MY_ENV_PATH" -f "${REPO_DIR}/zamba_environment.yml"

echo "=== Validation ==="
"$MY_ENV_PATH/bin/python" -c "import torch; import mamba_ssm; import transformers; print('SUCCESS: Zamba ready!')"