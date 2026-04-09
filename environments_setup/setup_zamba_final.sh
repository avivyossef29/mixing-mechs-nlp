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
export PIP_CACHE_DIR="${VOL_BASE}/pip_cache"
export HF_HOME="${VOL_BASE}/hf_cache"
export TMPDIR="${VOL_BASE}/tmp"

mkdir -p "$CONDA_ENVS_PATH" "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

# Initialize Conda
source "${VOL_BASE}/miniconda3/etc/profile.d/conda.sh"

echo "=== Step 1: Create Environment ==="
conda create --prefix "$MY_ENV_PATH" python=3.10 -y
conda activate "$MY_ENV_PATH"

echo "=== Step 2: Install Core & Zamba Compatibility ==="
# גרסת ה-transformers הזו קריטית ל-Zamba2
pip install --no-cache-dir numpy==1.26.4 packaging ninja setuptools wheel
pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir transformers==4.49.0 tokenizers==0.21.0

echo "=== Step 3: Graphics & Analysis Stack ==="
pip install --no-cache-dir pandas matplotlib seaborn scipy plotnine tqdm accelerate datasets einops pyvene==0.1.8

echo "=== Step 4: Compile Mamba Kernels ==="
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
pip install --no-cache-dir --no-build-isolation causal-conv1d==1.6.0 mamba-ssm==2.1.0

echo "=== Step 5: Final Environment Sync ==="
conda env update --prefix "$MY_ENV_PATH" -f "${REPO_DIR}/zamba_environment.yml"

echo "=== Validation ==="
"$MY_ENV_PATH/bin/python" -c "import torch; import mamba_ssm; import transformers; print('SUCCESS: Zamba ready with Transformers 4.49.0!')"
