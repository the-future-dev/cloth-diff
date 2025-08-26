#!/bin/bash
#SBATCH -A berzelius-2025-35
#SBATCH --gpus 1
#SBATCH -t 0-08:00:00

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <num_episodes> <num_variations> [resume]"
    exit 1
fi

env=ClothFold
seed=11
now=$(date +%m.%d.%H.%M)
eps=$1
num_var=$2

# Optional third argument "resume"
resume_flag=""
if [ "$#" -eq 3 ] && [ "$3" == "resume" ]; then
    resume_flag="--resume"
    echo ">>> Resuming from latest checkpoint!"
fi

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}/rlpyt_cloth:${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:${PWD}/rlkit/rlkit:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

echo "Softgym set up: OK"

# --- Check if dataset exists ---
dataset_file="data/ClothFold_numvariations${num_var}_eps${eps}_image_based_trajs.pkl"
echo "Checking for dataset file: ${dataset_file}"
if [ ! -f "$dataset_file" ]; then
    echo "Error: Dataset file not found: $dataset_file" >&2
    exit 1
fi
echo "Dataset file found."
# --- End Check ---

python experiments/run_diffusion.py \
    $resume_flag \
    --seed=${seed} \
    --name=state-diffusion-transformer-${env}-${num_var}-${eps} \
    --wandb \
    --saved_rollouts=${dataset_file} \
    --env_name=${env} \
    --env_kwargs_observation_mode=key_point \
    --env_kwargs_num_variations=${num_var} \
    --is_image_based=False \
    --observation_size=18 \
    --action_size=8 \
    \
    --model_type=transformer \
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
    --obs_as_local_cond=False \
    --pred_action_steps_only=False \
    --oa_step_convention=True \
    --cond_predict_scale=True \
    \
    --scheduler_num_train_timesteps=100 \
    --scheduler_beta_start=0.0001 \
    --scheduler_beta_end=0.02 \
    --scheduler_beta_schedule=squaredcos_cap_v2 \
    --scheduler_variance_type=fixed_small \
    --scheduler_clip_sample=True \
    --scheduler_prediction_type=epsilon \
    \
    --max_train_steps=300000 \
    --max_train_epochs=1000000 \
    --batch_size=256 \
    --lrate=0.0001 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=1000 \
    \
    --eval_interval=1000 \
    --num_eval_eps=3 \
    --eval_videos=True \
    --eval_gif_size=256 