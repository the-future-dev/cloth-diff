#!/bin/bash
#SBATCH -A berzelius-2025-35
#SBATCH --gpus 1
#SBATCH -t 1-00:00:00

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <num_episodes> <num_variations> [resume|--resume|-r]"
    exit 1
fi

env=ClothFold
seed=11
now=$(date +%m.%d.%H.%M)
eps=$1
num_var=$2

# Optional third argument "resume" (accepts resume, --resume or -r)
resume_flag=""
if [ "$#" -eq 3 ]; then
    case "$3" in
        resume|--resume|-r)
            resume_flag="--resume"
            echo ">>> Resuming from latest checkpoint!"
            ;;
        *)
            echo "Unknown resume option: $3"
            echo "Usage: $0 <num_episodes> <num_variations> [resume|--resume|-r]"
            exit 1
            ;;
    esac
fi

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}/rlpyt_cloth:${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:${PWD}/rlkit/rlkit:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

echo "Softgym set up: OK"

# --- Check if dataset exists ---
dataset_file="/proj/rep-learning-robotics/users/x_andri/dmfd/data/ClothFold_numvariations${num_var}_eps${eps}_image_based_trajs.pkl"
echo "Checking for dataset file: ${dataset_file}"
if [ ! -f "$dataset_file" ]; then
    echo "Error: Dataset file not found: $dataset_file" >&2
    exit 1
fi
echo "Dataset file found."
# --- End Check ---

python diffusion_policy/run_diffusion.py \
    $resume_flag \
    --seed=${seed} \
    --name=image-diffusion-transformer-04-${env}-${num_var}-${eps} \
    --wandb \
    --saved_rollouts=${dataset_file} \
    --env_name=${env} \
    --env_kwargs_observation_mode=cam_rgb \
    --env_kwargs_num_variations=${num_var} \
    --is_image_based=True \
    --action_size=8 \
    --use_ema=True \
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
    --obs_as_local_cond=False \
    --pred_action_steps_only=False \
    \
    --env_img_size=32 \
    --enable_img_transformations=False \
    --crop_shape 0 0 \
    \
    --scheduler_num_train_timesteps=100 \
    --scheduler_beta_start=0.0001 \
    --scheduler_beta_end=0.02 \
    --scheduler_beta_schedule=squaredcos_cap_v2 \
    --scheduler_variance_type=fixed_small \
    --scheduler_clip_sample=True \
    --scheduler_prediction_type=epsilon \
    \
    --max_train_steps=600000 \
    --max_train_epochs=200000 \
    --batch_size=256 \
    --lrate=1e-4 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=1000 \
    --beta1=0.95 \
    --beta2=0.999 \
    --transformer_weight_decay=1e-6 \
    --encoder_weight_decay=1e-4 \
    \
    --eval_interval=1000 \
    --num_eval_eps=3 \
    --eval_videos=True \
    --eval_gif_size=64


# Version 00

# #!/bin/bash
# #SBATCH -A berzelius-2025-35
# #SBATCH --gpus 1
# #SBATCH -t 0-08:00:00

# if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
#     echo "Usage: $0 <num_episodes> <num_variations> [resume|--resume|-r]"
#     exit 1
# fi

# env=ClothFold
# seed=11
# now=$(date +%m.%d.%H.%M)
# eps=$1
# num_var=$2

# # Optional third argument "resume" (accepts resume, --resume or -r)
# resume_flag=""
# if [ "$#" -eq 3 ]; then
#     case "$3" in
#         resume|--resume|-r)
#             resume_flag="--resume"
#             echo ">>> Resuming from latest checkpoint!"
#             ;;
#         *)
#             echo "Unknown resume option: $3"
#             echo "Usage: $0 <num_episodes> <num_variations> [resume|--resume|-r]"
#             exit 1
#             ;;
#     esac
# fi

# module load Mambaforge/23.3.1-1-hpc1-bdist
# mamba activate softgym

# export PYFLEXROOT=${PWD}/softgym/PyFlex
# export PYTHONPATH=${PWD}/rlpyt_cloth:${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:${PWD}/rlkit/rlkit:$PYTHONPATH
# export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

# echo "Softgym set up: OK"

# # --- Check if dataset exists ---
# dataset_file="data/ClothFold_numvariations${num_var}_eps${eps}_image_based_trajs.pkl"
# echo "Checking for dataset file: ${dataset_file}"
# if [ ! -f "$dataset_file" ]; then
#     echo "Error: Dataset file not found: $dataset_file" >&2
#     exit 1
# fi
# echo "Dataset file found."
# # --- End Check ---

# python experiments/run_diffusion.py \
#     $resume_flag \
#     --seed=${seed} \
#     --name=image-diffusion-transformer-${env}-${num_var}-${eps} \
#     --wandb \
#     --saved_rollouts=${dataset_file} \
#     --env_name=${env} \
#     --env_kwargs_observation_mode=cam_rgb \
#     --env_kwargs_num_variations=${num_var} \
#     --is_image_based=True \
#     --action_size=8 \
#     --use_ema=False \
#     \
#     --model_type=transformer \
#     --transformer_n_emb=256 \
#     --transformer_n_layer=8 \
#     --transformer_n_head=4 \
#     --transformer_p_drop_emb=0.0 \
#     --transformer_p_drop_attn=0.3 \
#     --transformer_causal_attn=True \
#     --transformer_time_as_cond=True \
#     --transformer_n_cond_layers=0 \
#     \
#     --horizon=8 \
#     --n_obs_steps=2 \
#     --n_action_steps=8 \
#     --num_inference_steps=100 \
#     --obs_as_global_cond=True \
#     --obs_as_local_cond=False \
#     --pred_action_steps_only=False \
#     --cond_predict_scale=True \
#     --channel_cond=False \
#     \
#     --env_img_size=32 \
#     --enable_img_transformations=False \
#     --crop_shape 0 0 \
#     --cnn_channels 32 64 128 256 \
#     --cnn_kernels 3 3 3 3 \
#     --cnn_strides 2 2 2 2 \
#     --latent_dim=512 \
#     \
#     --scheduler_num_train_timesteps=100 \
#     --scheduler_beta_start=0.0001 \
#     --scheduler_beta_end=0.02 \
#     --scheduler_beta_schedule=squaredcos_cap_v2 \
#     --scheduler_variance_type=fixed_small \
#     --scheduler_clip_sample=True \
#     --scheduler_prediction_type=epsilon \
#     \
#     --max_train_steps=600000 \
#     --max_train_epochs=1000000 \
#     --batch_size=256 \
#     --lrate=1e-4 \
#     --lr_scheduler=cosine \
#     --lr_warmup_steps=1000 \
#     \
#     --eval_interval=1000 \
#     --num_eval_eps=3 \
#     --eval_videos=True \
#     --eval_gif_size=64
