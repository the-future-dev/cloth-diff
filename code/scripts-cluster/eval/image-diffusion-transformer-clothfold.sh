#!/bin/bash
#SBATCH -A berzelius-2025-35
#SBATCH --gpus 1
#SBATCH -t 0-02:00:00
#SBATCH -J clothfold-eval-diff-img

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <num_eval_eps> <num_variations> <checkpoint_path>"
    exit 1
fi

# Required arguments
NUM_EVAL_EPS=$1
NUM_VAR=$2
CKPT_PATH=$3

# Environment setup
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:${PWD}/rlkit/rlkit:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

echo "=========================================="
echo " Evaluating Image‐based Diffusion Transformer Policy"
echo "  Checkpoint        : ${CKPT_PATH}"
echo "  # Variations      : ${NUM_VAR}"
echo "  # Eval Episodes   : ${NUM_EVAL_EPS}"
echo "  5‐seed evaluation : ${EVAL_5_SEEDS}"
echo "=========================================="

python experiments/run_diffusion.py \
    --is_eval=True \
    --test_checkpoint=${CKPT_PATH} \
    --seed=1234 \
    --env_name=ClothFold \
    --env_kwargs_observation_mode=cam_rgb \
    --env_kwargs_num_variations=${NUM_VAR} \
    --is_image_based=True \
    --action_size=8 \
    --use_ema=False \
    \
    --model_type=transformer \
    --visual_encoder=DrQCNN \
    --transformer_n_emb=256 \
    --transformer_n_layer=8 \
    --transformer_n_head=4 \
    --transformer_p_drop_emb=0.0 \
    --transformer_p_drop_attn=0.01 \
    --transformer_causal_attn=True \
    --transformer_time_as_cond=True \
    --transformer_n_cond_layers=0 \
    \
    --horizon=8 \
    --n_obs_steps=2 \
    --n_action_steps=8 \
    --num_inference_steps=100 \
    --obs_as_global_cond=True \
    --pred_action_steps_only=False \
    --crop_shape 0 0 \
    \
    --env_img_size=32 \
    --enable_img_transformations=False \
    --crop_shape 0 0 \
    \
    --eval_videos=True \
    --eval_over_five_seeds=True \
    --eval_gif_size=64 \
    --num_eval_eps=${NUM_EVAL_EPS}
