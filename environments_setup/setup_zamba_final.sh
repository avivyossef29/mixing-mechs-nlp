#!/bin/bash
#SBATCH --job-name=setup_zamba
#SBATCH --output=setup_zamba_%j.out
#SBATCH --error=setup_zamba_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

### 1. הגדרת נתיבים ב-Volume (23TB פנויים) ###
export USERNAME="inbalmoryles"
export VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
export MY_ENV_PATH="${VOL_BASE}/envs/zamba-env"

# פתרון קריטי לבעיית ה-Quota: חסימת גישה לתיקיית הבית
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${VOL_BASE}/pip_cache"
export CONDA_PKGS_DIRS="${VOL_BASE}/conda_pkgs"
export HF_HOME="${VOL_BASE}/hf_cache"
export TMPDIR="${VOL_BASE}/tmp"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TMPDIR"

# טעינת קונדה מה-Volume שלך
source /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/miniconda3/etc/profile.d/conda.sh
rm -rf "$MY_ENV_PATH"

echo "=== Step 1: Create Base Env ==="
conda create -p "$MY_ENV_PATH" python=3.10 -y
conda activate "$MY_ENV_PATH"

# בדיקה שה-pip הנכון עובד
which pip

echo "=== Step 2: Build Tools & Core Libraries ==="
# התקנה ישירה ללא Cache בבית
pip install --no-cache-dir numpy==1.26.4 packaging ninja setuptools wheel
pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 3: Compiling Kernels (No Isolation) ==="
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH

# שימוש ב-no-build-isolation קריטי כדי ש-pip יראה את ה-torch ב-Volume
pip install --no-cache-dir --no-build-isolation causal-conv1d==1.6.0 mamba-ssm==2.1.0

echo "=== Step 4: Final Package Installation ==="
# במקום conda env update שעלול לקרוס בגלל Quota, נתקין ישירות את הנדרש לזמבה
pip install --no-cache-dir pandas matplotlib seaborn scipy plotnine tqdm transformers accelerate datasets

# בדיקה סופית ואישור הצלחה
"$MY_ENV_PATH/bin/python" -c "import torch; import mamba_ssm; import transformers; print(f'SUCCESS: Zamba ready with Transformers {transformers.__version__}')"