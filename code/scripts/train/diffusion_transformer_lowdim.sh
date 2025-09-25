#!/bin/bash
#SBATCH -A berzelius-2025-278
#SBATCH -p berzelius
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -J clothdiff-diff-t-lowdim
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# Usage: sbatch train_diffusion_transformer_lowdim.sh [eps] [vars]
EPS=${1:-200}
VARS=${2:-25}

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

python -m ml_framework.core.cli \
  --mode train \
  --model diffusion-transformer-lowdim \
  --config diffusion_policy/config/diffusion-transformer-lowdim.yaml \
  --eps "$EPS" \
  --vars "$VARS" \
  --exp_name "diffusion-transformer-lowdim-${VARS}vars-${EPS}eps" \
  --dataset_path "/proj/rep-learning-robotics/users/x_andri/dmfd/data/ClothFold_vars-${VARS}_eps-${EPS}_image_based_trajs.pkl" \
  --wandb
