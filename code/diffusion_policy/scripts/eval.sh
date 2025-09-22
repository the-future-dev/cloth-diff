#!/usr/bin/env bash
# Basic evaluation script printing average loss on dataset
set -euo pipefail

DATA=${1:-"./data/ClothFold_vars-25_eps-200_img-128.pkl"}
MODEL=${MODEL:-transformer-lowdim}
BATCH=${BATCH:-256}

python -m diffusion_policy.core.cli \
  --mode eval \
  --dataset_path "$DATA" \
  --model "$MODEL" \
  --batch_size "$BATCH"
