#!/bin/bash
#SBATCH --job-name=setup_mamba2
#SBATCH --output=setup_mamba2_%j.out
#SBATCH --error=setup_mamba2_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

export USERNAME="inbalmoryles"
export VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
export MY_ENV_PATH="${VOL_BASE}/envs/mamba2-gpu"
export REPO_DIR="${VOL_BASE}/mixing-mechs-nlp"

source "${VOL_BASE}/miniconda3/etc/profile.d/conda.sh"

echo "=== Step 1: Create Environment ==="
conda create --prefix "$MY_ENV_PATH" python=3.10 -y
conda activate "$MY_ENV_PATH"

echo "=== Step 2: Core Build & Science Dependencies ==="
pip install --no-cache-dir numpy==1.26.4 packaging ninja setuptools wheel pandas matplotlib seaborn scipy plotnine tqdm
pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 3: Compile Mamba2 Kernels ==="
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
pip install --no-cache-dir --no-build-isolation causal-conv1d==1.6.0 mamba-ssm==2.1.0

echo "=== Step 4: Final Environment Sync ==="
conda env update --prefix "$MY_ENV_PATH" -f "${REPO_DIR}/mamba2-gpu_environment.yml"

echo "=== Validation ==="
"$MY_ENV_PATH/bin/python" -c "import torch; import mamba_ssm; print('SUCCESS: Mamba2 ready with Science Stack!')"
