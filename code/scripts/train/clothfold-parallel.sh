#!/bin/bash

#############################################################################
# Parallel ClothFold Training Script for New diffusion_policy
# 
# Launches 5 parallel SLURM jobs with different eps/vars combinations:
# eps: 40, 200, 1000, 4000, 8000
# vars: 5, 25, 125, 500, 1000
#
# Usage: ./diffusion-policy-clothfold-parallel.sh [resume]
#############################################################################

# Check if resume flag is passed
resume_flag=""
if [ "$#" -eq 1 ] && [ "$1" == "resume" ]; then
    resume_flag="--resume"
    echo ">>> Will resume from latest checkpoints!"
fi

# Job combinations (eps, vars)
combinations=(
    "40 5"
    "200 25" 
    "1000 125"
    "4000 500"
    "8000 1000"
)

# Function to submit a single job
submit_job() {
    local eps=$1
    local vars=$2
    local job_name="clothfold-${vars}vars-${eps}eps"
    
    echo "Submitting job: $job_name (eps=$eps, vars=$vars)"
    
    sbatch --job-name="$job_name" <<EOF
#!/bin/bash
#SBATCH -A berzelius-2025-278
#SBATCH -p berzelius
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# Environment setup
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=\${PWD}/softgym/PyFlex
export PYTHONPATH=\${PWD}:\${PWD}/softgym:\${PYFLEXROOT}/bindings/build:\$PYTHONPATH
export LD_LIBRARY_PATH=\${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:\$LD_LIBRARY_PATH

echo "=== Job: $job_name ==="
echo "Episodes: $eps, Variations: $vars"
echo "Dataset auto-resolve: ClothFold_vars-${vars}_eps-${eps}_img-128.pkl"
echo "========================"

# Run training with auto-resolved dataset and env_kwargs
python -m diffusion_policy.core.cli \\
    --config diffusion_policy/config/lowdim-transformer.yaml \\
    --eps $eps \\
    --vars $vars \\
    --exp_name "lowdim-transformer-clothfold-${vars}vars-${eps}eps" \\
    --wandb \\
    --eval_video \\
    $resume_flag

echo "=== Job $job_name completed ==="
EOF

}

# Submit all jobs
echo "Starting parallel ClothFold training jobs..."
for combo in "${combinations[@]}"; do
    set -- $combo  # Split "eps vars" into $1 and $2
    submit_job $1 $2
    sleep 2  # Small delay between submissions
done

echo ""
echo "All jobs submitted! Check status with: squeue -u \$USER"
echo "Monitor outputs in: slurm-clothfold-*-<jobid>.{out,err}"
echo ""
echo "Jobs launched:"
for combo in "${combinations[@]}"; do
    set -- $combo
    echo "  - eps=$1, vars=$2 -> clothfold-${2}vars-${1}eps"
done