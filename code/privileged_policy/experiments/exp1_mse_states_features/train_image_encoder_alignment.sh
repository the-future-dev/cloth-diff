#!/bin/bash
#SBATCH -A berzelius-2025-278
#SBATCH -p berzelius
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -J clothdiff-exp1-step2
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# Usage: sbatch train_image_encoder_alignment.sh [eps] [vars] [state_encoder_checkpoint]
EPS=${1:-200}
VARS=${2:-25}
STATE_CHECKPOINT=${3:-"checkpoints/diffusion-transformer-lowdim-${VARS}vars-${EPS}eps"}

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

python privileged_policy/experiments/exp1_mse_states_features/train_image_encoder_alignment.py \
  --eps "$EPS" \
  --vars "$VARS" \
  --state_checkpoint "$STATE_CHECKPOINT" \
  --exp_name "exp1-image-alignment-${VARS}vars-${EPS}eps"