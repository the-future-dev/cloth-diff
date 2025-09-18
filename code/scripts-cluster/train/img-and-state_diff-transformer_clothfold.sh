#!/bin/bash
#SBATCH -A berzelius-2025-278
#SBATCH -p berzelius
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -J clothfold-train-double
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

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
    echo "Note: This script requires a dataset with both state and image trajectories." >&2
    echo "Make sure to generate it with both key_point and cam_rgb observation modes." >&2
    exit 1
fi
echo "Dataset file found."

python diffusion_policy/run_diffusion.py \
    $resume_flag \
    --seed=${seed} \
    --name=img-state-diff-transformer-01-${env}-${num_var}-${eps} \
    --wandb \
    --saved_rollouts=${dataset_file} \
    --env_name=${env} \
    --env_kwargs_observation_mode=cam_rgb \
    --env_kwargs_num_variations=${num_var} \
    --is_image_based=True \
    --action_size=8 \
    --use_ema=True \
    \
    --model_type=double_modality \
    --visual_encoder=DrQCNN \
    --obs_encoder_group_norm=False \
    --eval_fixed_crop=False \
    --crop_shape 0 0 \
    \
    --state_encoder_type=identity \
    --priv_fuse_op=concat \
    --shared_encoder_type=transformer \
    --shared_encoder_kwargs='{"d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512}' \
    \
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
    --env_img_size=512 \
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
