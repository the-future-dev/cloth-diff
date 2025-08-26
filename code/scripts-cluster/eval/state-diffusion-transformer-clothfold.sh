#!/bin/bash
#SBATCH -A berzelius-2025-35
#SBATCH --gpus 1
#SBATCH -t 0-02:00:00

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <num_eval_eps> <num_variations> <checkpoint_path>"
    exit 1
fi

num_eval_eps=$1
num_var=$2
ckpt_path=$3

# --- Load modules and activate your conda env ---
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

# --- Make sure Python sees SoftGym, rlpyt_cloth, rlkit, etc. ---
export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}/rlpyt_cloth:${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:${PWD}/rlkit/rlkit:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

# --- Run evaluation ---
python experiments/run_diffusion.py \
    --is_eval True \
    --eval_over_five_seeds True \
    --use_ema True \
    --seed 11 \
    --test_checkpoint ${ckpt_path} \
    --env_name ClothFold \
    --env_kwargs_observation_mode key_point \
    --env_kwargs_num_variations ${num_var} \
    --is_image_based False \
    --observation_size 18 \
    --action_size 8 \
    --model_type transformer \
    --transformer_n_emb 256 \
    --transformer_n_layer 8 \
    --transformer_n_head 4 \
    --transformer_p_drop_emb 0.0 \
    --transformer_p_drop_attn 0.01 \
    --transformer_causal_attn True \
    --transformer_time_as_cond True \
    --transformer_n_cond_layers 0 \
    --horizon 8 \
    --n_obs_steps 2 \
    --n_action_steps 8 \
    --num_inference_steps 100 \
    --obs_as_global_cond True \
    --obs_as_local_cond False \
    --pred_action_steps_only False \
    --oa_step_convention True \
    --cond_predict_scale True \
    --eval_videos True \
    --eval_gif_size 256 \
    --num_eval_eps ${num_eval_eps}