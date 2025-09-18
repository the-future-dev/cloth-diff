#!/bin/bash
#SBATCH -A berzelius-2025-278
#SBATCH -p berzelius
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -J clothfold-priv-train
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

if [ "$#" -lt 0 ] || [ "$#" -gt 1 ]; then
    echo "Usage: $0 [resume|--resume|-r]"
    exit 1
fi

# Optional first argument "resume"
resume_flag=""
if [ "$#" -eq 1 ]; then
    case "$1" in
        resume|--resume|-r)
            resume_flag="--resume"
            echo ">>> Resuming from latest checkpoint!"
            ;;
        *)
            echo "Unknown resume option: $1"
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

# --- Privileged settings ---
priv_fuse_op=concat
priv_state_encoder=mlp
priv_shared_encoder_type=cross_attention
priv_disable_privileged_prob=0.50
image_encoder=DrQCNN #ResNet18Conv

declare -a PARAMS=(
    "8000 1000"
    "4000 500"
    "1000 125"
    "200 25"
    "40 5"
)


for param in "${PARAMS[@]}"; do
    set -- $param
    eps=$1
    num_var=$2
    seed=11
    env=ClothFold
    
    # --- Dataset path ---
    dataset_file="/proj/rep-learning-robotics/users/x_andri/dmfd/data/ClothFold_numvariations${num_var}_eps${eps}_image_based_trajs.pkl"
    echo "Checking for dataset file: ${dataset_file}"
    if [ ! -f "$dataset_file" ]; then
        echo "Error: Dataset file not found: $dataset_file" >&2
        exit 1
    fi
    echo "Dataset numvariations${num_var}_eps${eps} found."

    python diffusion_policy/run_diffusion.py \
    $resume_flag \
    --seed=${seed} \
    --name=priv-no_mask-${priv_shared_encoder_type}-fuse_${priv_fuse_op}-state_${priv_state_encoder}-image_${image_encoder}-${env}-${num_var}-${eps} \
    --wandb \
    --saved_rollouts=${dataset_file} \
    --env_name=${env} \
    --env_kwargs_observation_mode=cam_rgb \
    --env_kwargs_num_variations=${num_var} \
    --is_image_based=True \
    --action_size=8 \
    --use_ema=False \
    \
    --num_workers=8 \
    --pin_memory \
    --persistent_workers \
    --prefetch_factor=10 \
    --model_type=privileged \
    \
    --obs_encoder_group_norm=False \
    --eval_fixed_crop=True \
    --priv_fuse_op=${priv_fuse_op} \
    --disable_privileged_method=zero \
    --disable_privileged_prob=${priv_disable_privileged_prob} \
    --state_encoder_type=${priv_state_encoder} \
    --state_mlp_hidden_dims=50 \
    --state_feat_dim=50 \
    --shared_encoder_type=${priv_shared_encoder_type} \
    --shared_encoder_kwargs='{"d_model": 100, "nhead": 2, "num_layers": 4, "dropout": 0.03, "img_dim": 50, "state_dim": 50}' \
    \
    --visual_encoder=${image_encoder} \
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
    --num_inference_steps=50 \
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
    --max_train_epochs=2000000 \
    --batch_size=256 \
    --lrate=1e-4 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=500 \
    --beta1=0.95 \
    --beta2=0.999 \
    --transformer_weight_decay=1e-6 \
    --encoder_weight_decay=1e-4 \
    \
    --eval_interval=1500 \
    --num_eval_eps=5 \
    --eval_videos=True &

done

wait
echo "ALL completed"
