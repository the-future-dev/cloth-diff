#!/bin/bash
#SBATCH -A berzelius-2025-278
#SBATCH -p berzelius
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -J clothdiff
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# Single ClothFold training job with auto-resolved dataset and env_kwargs
# Usage: sbatch train_transformer_lowdim.sh [eps] [vars]
# Defaults: eps=200, vars=25

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

# Parse arguments
EPS=${1:-200}
VARS=${2:-25}

echo "=== ClothFold Training (Refactored) ==="
echo "Episodes: $EPS, Variations: $VARS"
echo "=========================="

python -m ml_framework.core.cli \
  --mode train \
  --model transformer-lowdim \
  --config transformer_policies/config/lowdim-transformer.yaml \
  --eps "$EPS" \
  --vars "$VARS" \
  --exp_name "lowdim-transformer-clothfold-${VARS}vars-${EPS}eps" \
  --dataset_path "./data/ClothFold_vars-${VARS}_eps-${EPS}_img-128.pkl" \
  --wandb