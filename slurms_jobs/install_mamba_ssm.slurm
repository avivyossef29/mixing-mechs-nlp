#!/bin/bash

### Resource Allocation ###
#SBATCH --job-name=install_mamba
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=1:00:00           # 1 hour should be enough
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1              # GPU needed for CUDA compilation

### 1. Setup Environment ###
source ~/.bashrc

# Activate the Mamba2 GPU environment
conda activate mamba2-gpu
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/.cache/huggingface"
mkdir -p $HF_HOME

### 2. Define Paths ###
USERNAME="avivyossef"
VOL_BASE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USERNAME}"
# Mamba is cloned in home directory
MAMBA_DIR="${HOME}/mamba"

### 3. Installation ###
echo "Running on node: $HOSTNAME"
echo "Date: $(date)"
echo "Installing mamba-ssm from source..."

cd "$MAMBA_DIR"

# Install mamba-ssm with no build isolation
pip install . --no-build-isolation

echo "Installation finished at: $(date)"

# Verify installation
echo "Verifying installation..."
python -c 'import mamba_ssm; print("SUCCESS: mamba_ssm installed successfully!")' || echo "ERROR: Installation verification failed"

echo "Job completed at: $(date)"
