To compile Pyflex using apptainer gather:
- Path to the softgym folder
- Mamba/Conda environment: with command:
```bash
mamba info --envs
```

Gather both base and softgym path!

RUN:
```bash
apptainer shell \
    --nv \
    -B PATH_TO_SOFTGYM:/workspage/softgym \
    -B PATH_TO_MAMBA:PATH_TO_MAMBA \
    -B PATH_TO_ENV:PATH_TO_ENV \
    -B /tmp/.X11-unix:/tmp/.X11-unix \
    --env DISPLAY=$DISPLAY,QT_X11_NO_MITSHM=1 \
    PATH_TO_SOFTGYM/softgym_latest.sif
```

Example:
```bash
apptainer shell \
  --nv \
  -B /proj/rep-learning-robotics/users/x_andri/cloth-diff/code/softgym:/workspace/softgym \
  -B /software/sse/manual/Mambaforge/23.3.1-1/hpc1-bdist:/software/sse/manual/Mambaforge/23.3.1-1/hpc1-bdist \
  -B /home/x_andri/.conda/envs/softgym:/home/x_andri/.conda/envs/softgym \
  -B /tmp/.X11-unix:/tmp/.X11-unix \
  --env DISPLAY=$DISPLAY,QT_X11_NO_MITSHM=1 \
  /proj/rep-learning-robotics/users/x_andri/cloth-diff/code/softgym/softgym_latest.sif
```

Inside Apptainer: source mamba and activate the softgym env

Example
```bash
export PATH="/software/sse/manual/Mambaforge/23.3.1-1/hpc1-bdist/bin:$PATH"
source /software/sse/manual/Mambaforge/23.3.1-1/hpc1-bdist/etc/profile.d/conda.sh
conda activate softgym
```

Inside /workspage/softgym run:
```bash
. softgym/prepare_1.0.sh
```

Then:
```bash
. softgym/compile_1.0.sh
```

Then you can exit!