#!/bin/bash
#SBATCH -A berzelius-2025-35
#SBATCH --gpus 1
#SBATCH -t 1-00:00:00

# Usage: ./scripts/train/image-dmfd-clothfold-parameterized.sh <num_episodes> <num_variations>
# Example: ./scripts/train/image-dmfd-clothfold-parameterized.sh 8000 1000
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_episodes> <num_variations>"
    exit 1
fi

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

echo "Softgym set up: OK"

num_eps=$1
num_variations=$2

python core/generate_expert_trajs.py \
--save_observation_img=True \
--num_eps=${num_eps} \
--num_variations=${num_variations} \
--env_img_size=32 \
--env_name=ClothFold \
--save_states_in_folder=True \
--out_filename=ClothFold_numvariations${num_variations}_eps${num_eps}_image_based_trajs_states.pkl
