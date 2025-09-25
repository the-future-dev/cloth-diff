#!/bin/bash
#SBATCH -A berzelius-2025-278
#SBATCH -p berzelius
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -J clothdiff-t-image
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# Usage: sbatch train_transformer_image.sh [--single]

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

if [ "$1" == "--single" ]; then
    EPS=200
    VARS=25
    python -m ml_framework.core.cli \
      --mode train \
      --model transformer-image \
      --config foundation_policy/config/transformer-image.yaml \
      --eps "$EPS" \
      --vars "$VARS" \
      --exp_name "transformer-image-${VARS}vars-${EPS}eps" \
      --dataset_path "./data/ClothFold_vars-${VARS}_eps-${EPS}_img-128.pkl" \
      --wandb
else
    combinations=(
        "40 5"
        "200 25"
        "1000 125"
        "4000 500"
        "8000 1000"
    )
    for combo in "${combinations[@]}"; do
        set -- $combo
        eps=$1
        vars=$2
        python -m ml_framework.core.cli \
          --mode train \
          --model transformer-image \
          --config foundation_policy/config/transformer-image.yaml \
          --eps "$eps" \
          --vars "$vars" \
          --exp_name "transformer-image-${vars}vars-${eps}eps" \
          --dataset_path "./data/ClothFold_vars-${vars}_eps-${eps}_img-128.pkl" \
          --wandb &
    done
    wait
fi
