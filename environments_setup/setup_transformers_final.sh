#!/bin/bash
#SBATCH --job-name=setup_trans
#SBATCH --output=setup_trans_%j.out
#SBATCH --error=setup_trans_%j.err
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00

export CONDA_PKGS_DIRS=/vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/.conda_pkgs
export MY_ENV_PATH=/vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/envs/transformers-env
mkdir -p $CONDA_PKGS_DIRS

source /vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/miniconda3/etc/profile.d/conda.sh
rm -rf $MY_ENV_PATH

echo "=== Step 1: Base Env ==="
conda create -p $MY_ENV_PATH python=3.10 -y
conda activate $MY_ENV_PATH

echo "=== Step 2: Core & Graphics ==="
# התקנת תשתיות ניתוח נתונים וגרפים עבור plotting.py
pip install numpy==1.26.4 pandas matplotlib seaborn scipy plotnine tqdm
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 accelerate==0.34.0 datasets einops pyvene==0.1.8

python -c "import pandas; import matplotlib; print('SUCCESS: Transformers Env with Graphics ready')"
