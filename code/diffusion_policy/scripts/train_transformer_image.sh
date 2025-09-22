#!/bin/bash
#SBATCH -A berzelius-2025-278
#SBATCH -p berzelius
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -J clothdiff
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH


DATA=${1:-"./data/ClothFold_vars-25_eps-200_img-128.pkl"}
BATCH=${BATCH:-256}
EPOCHS=${EPOCHS:-50}
SEED=${SEED:-1234}

python -m diffusion_policy.core.cli \
  --config diffusion_policy/config/common.yaml \
  --mode train \
  --dataset_path "$DATA" \
  --model transformer-image \
  --batch_size "$BATCH" \
  --max_epochs "$EPOCHS" \
  --seed "$SEED" \
  --wandb \
  --wandb_project "clothdiff" \
  --checkpoint_every 1 \
  --keep_last 5 \
  --exp_name "tf-image-${SLURM_JOB_NAME:-clothdiff}-${SLURM_JOB_ID:-local}"
# To disable wandb explicitly, add: --no-wandb
