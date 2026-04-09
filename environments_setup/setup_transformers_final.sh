#!/bin/bash
#SBATCH --job-name=setup_trans
#SBATCH --output=setup_trans_%j.out
#SBATCH --error=setup_trans_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00

export USERNAME="inbalmoryles"
export VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
export MY_ENV_PATH="${VOL_BASE}/envs/transformers-env"

source "${VOL_BASE}/miniconda3/etc/profile.d/conda.sh"
rm -rf $MY_ENV_PATH

echo "=== Step 1: Base Env ==="
conda create -p $MY_ENV_PATH python=3.10 -y
conda activate $MY_ENV_PATH

echo "=== Step 2: Core & Graphics Stack ==="
# התקנה מאוחדת של כל חבילות הניתוח
pip install --no-cache-dir numpy==1.26.4 pandas matplotlib seaborn scipy plotnine tqdm

echo "=== Step 3: Heavy Lift (Transformers & Tokenizers fix) ==="
# שילוב הגרסאות שפתר לנו את הבעיה ב-Gemma ו-Falcon
pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir transformers==4.48.0 tokenizers==0.21.0 accelerate==0.34.0 datasets einops pyvene==0.1.8

echo "=== Validation ==="
"$MY_ENV_PATH/bin/python" -c "import pandas; import matplotlib; import transformers; print('SUCCESS: Transformers Env ready with fix for Tokenizers 0.21.0')"
