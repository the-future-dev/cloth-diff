# Experiment 1: MSE between States and Features

This experiment implements a 3-step privileged policy training recipe:

## Step 1: Train Diffusion Lowdim Policy
Train a diffusion-based transformer policy on low-dimensional state observations.

**Command:**
```bash
sbatch scripts/train/diffusion_transformer_lowdim.sh [eps] [vars]
```

**Example:**
```bash
sbatch scripts/train/diffusion_transformer_lowdim.sh 200 25
```

This will train a policy with 200 episodes and 25 variations, saving checkpoints to `checkpoints/diffusion-transformer-lowdim-25vars-200eps/`.

## Step 2: Train Image Encoder Alignment
Train an image encoder to produce features that match the state encoder from Step 1 using MSE loss.

**Command:**
```bash
sbatch privileged_policy/experiments/exp1_mse_states_features/train_image_encoder_alignment.sh [eps] [vars] [state_checkpoint]
```

**Example:**
```bash
sbatch privileged_policy/experiments/exp1_mse_states_features/train_image_encoder_alignment.sh 200 25 checkpoints/diffusion-transformer-lowdim-25vars-200eps
```

This will:
- Load the trained state encoder from the checkpoint
- Train a ResNet18Conv image encoder to minimize MSE with state features
- Save the aligned image encoder to `checkpoints/exp1-image-alignment-25vars-200eps/`

## Step 3: Train Diffusion Image Policy
Train a diffusion-based image policy using:
- Transformer weights initialized from Step 1
- Image encoder aligned from Step 2

**Requirements:**
- Modify the diffusion image policy config to load pretrained weights
- Update the image encoder initialization to use the aligned encoder from Step 2
- Ensure proper weight loading and feature dimension matching

**Key modifications needed:**
1. Load transformer weights from Step 1 checkpoint
2. Replace default image encoder with aligned encoder from Step 2
3. Verify feature dimensions match between state and image encoders
4. Update config to handle pretrained components

**Config changes:**
- Set `pretrained_transformer_path` to Step 1 checkpoint
- Set `pretrained_image_encoder_path` to Step 2 checkpoint
- Ensure `state_feat_dim` matches `image_feat_dim` after alignment