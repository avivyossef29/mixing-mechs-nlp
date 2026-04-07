#!/bin/bash
#SBATCH --job-name=setup_zamba
#SBATCH --output=setup_zamba_%j.out
#SBATCH --error=setup_zamba_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# 1. הגדרת נתיבים ב-Volume (23TB פנויים)
export USERNAME="inbalmoryles"
export VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
export MY_ENV_PATH="${VOL_BASE}/envs/zamba-env"

# פתרון קריטי לבעיית ה-Quota בתיקיית הבית
export PIP_CACHE_DIR="${VOL_BASE}/pip_cache"
export CONDA_PKGS_DIRS="${VOL_BASE}/conda_pkgs"
export HF_HOME="${VOL_BASE}/hf_cache"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME"

source /vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/miniconda3/etc/profile.d/conda.sh
rm -rf "$MY_ENV_PATH"

echo "=== Step 1: Create Base Env ==="
conda create -p "$MY_ENV_PATH" python=3.10 -y
conda activate "$MY_ENV_PATH"

echo "=== Step 2: Build Tools & Graphics (v2) ==="
# התקנה ישירה ללא Cache בבית
pip install --no-cache-dir numpy==1.26.4 packaging ninja setuptools wheel
pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir pandas matplotlib seaborn scipy plotnine tqdm

echo "=== Step 3: Compiling Kernels (No Isolation) ==="
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH

# שימוש ב-no-build-isolation כדי ש-pip יראה את ה-torch שכבר התקנו
pip install --no-cache-dir --no-build-isolation causal-conv1d==1.6.0 mamba-ssm==2.1.0

echo "=== Step 4: YAML Update (Specific for Zamba) ==="
# עדכון שאר התלויות מהקובץ המקורי (כמו Transformers 4.57.6)
conda env update -p "$MY_ENV_PATH" -f "zamba_environment.yml"

python -c "import torch; import mamba_ssm; import transformers; print(f'SUCCESS: Zamba ready with Transformers {transformers.__version__}')"