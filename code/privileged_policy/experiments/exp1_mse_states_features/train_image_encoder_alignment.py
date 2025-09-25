#!/usr/bin/env python3
"""
Train image encoder to align with state encoder features using MSE loss.

This script loads a trained diffusion lowdim policy, extracts its state encoder,
and trains an image encoder to produce the same feature representations.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Import necessary modules
from ml_framework.data.data import make_dataloaders
from diffusion_policy.model.encoder_factory import EncoderFactory
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policies.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion


def load_state_encoder_from_checkpoint(checkpoint_path, device='cuda'):
    """Load normalizer from a trained diffusion lowdim policy checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create normalizer
    normalizer = LinearNormalizer()
    if 'normalizer' in checkpoint:
        normalizer.load_state_dict(checkpoint['normalizer'])
    
    return normalizer


def create_image_encoder(obs_shape_meta, visual_encoder='ResNet18Conv'):
    """Create image encoder with appropriate configuration."""
    # Create image encoder
    image_encoder = EncoderFactory.create_image_encoder(
        encoder_type=visual_encoder,
        obs_shape_meta=obs_shape_meta,
        crop_shape=(76, 76),  # Default crop shape
        obs_encoder_group_norm=False,
        eval_fixed_crop=False
    )

    return image_encoder


def get_image_obs_shape_meta(dataset_path):
    """Get observation shape metadata from dataset."""
    # Load a small sample to get shapes
    import pickle
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # Get first trajectory
    traj = data['trajs'][0]

    # Build shape meta
    obs_shape_meta = {}

    if 'image' in traj:
        # Assume image is [T, H, W, C] or [T, C, H, W]
        img_shape = traj['image'].shape
        if len(img_shape) == 4:
            if img_shape[-1] in [1, 3]:  # HWC
                obs_shape_meta['image'] = {'shape': img_shape[1:], 'type': 'rgb'}
            else:  # CHW
                obs_shape_meta['image'] = {'shape': img_shape[1:], 'type': 'rgb'}

    if 'state' in traj:
        state_shape = traj['state'].shape
        obs_shape_meta['state'] = {'shape': state_shape[1:], 'type': 'low_dim'}

    return obs_shape_meta


def train_image_encoder_alignment(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb
    if args.wandb:
        wandb.init(
            project="clothdiff",
            name=args.exp_name,
            config=vars(args)
        )

    # Get observation shape metadata
    obs_shape_meta = get_image_obs_shape_meta(args.dataset_path)
    print(f"Observation shape meta: {obs_shape_meta}")

    # Load normalizer from checkpoint
    print(f"Loading normalizer from {args.state_checkpoint}")
    normalizer = load_state_encoder_from_checkpoint(args.state_checkpoint, device)
    normalizer.to(device)

    # Create image encoder
    print(f"Creating image encoder: {args.visual_encoder}")
    image_encoder = create_image_encoder(obs_shape_meta, args.visual_encoder)
    image_encoder.to(device)
    image_encoder.train()

    # Create dataloaders (privileged mode to get both state and image)
    train_loader, val_loader = make_dataloaders(
        args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_image_based=True,
        privileged=True,
        horizon=None  # Use full trajectories
    )

    # Get feature dimensions
    with torch.no_grad():
        # Get a sample batch
        sample_batch = next(iter(train_loader))

        # Process state through normalizer (this is the "state encoder" for lowdim)
        state_features = normalizer['state'].normalize(sample_batch['state'].to(device))
        target_dim = state_features.shape[-1]

        # Get image feature dimension
        image_features = image_encoder({'image': sample_batch['image'].to(device)})
        if isinstance(image_features, dict):
            image_feat_dim = image_features['image'].shape[-1]
        else:
            image_feat_dim = image_features.shape[-1]

    print(f"Target feature dimension: {target_dim}")
    print(f"Image feature dimension: {image_feat_dim}")

    # Add projection head if dimensions don't match
    if image_feat_dim != target_dim:
        projection_head = nn.Linear(image_feat_dim, target_dim).to(device)
        print(f"Adding projection head: {image_feat_dim} -> {target_dim}")
    else:
        projection_head = None

    # Setup optimizer
    if projection_head is not None:
        params = list(image_encoder.parameters()) + list(projection_head.parameters())
    else:
        params = image_encoder.parameters()

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.max_epochs):
        # Training
        image_encoder.train()
        if projection_head:
            projection_head.train()

        train_loss = 0.0
        train_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epochs}"):
            state = batch['state'].to(device)
            image = batch['image'].to(device)

            # Get target features from normalized states
            with torch.no_grad():
                target_features = normalizer['state'].normalize(state)

            # Get image features
            image_obs = {'image': image}
            image_features = image_encoder(image_obs)

            if isinstance(image_features, dict):
                image_features = image_features['image']

            # Apply projection if needed
            if projection_head is not None:
                image_features = projection_head(image_features)

            # Compute MSE loss
            loss = nn.functional.mse_loss(image_features, target_features)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * state.shape[0]
            train_samples += state.shape[0]

        train_loss /= train_samples

        # Validation
        image_encoder.eval()
        if projection_head:
            projection_head.eval()

        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(device)
                image = batch['image'].to(device)

                target_features = normalizer['state'].normalize(state)
                image_obs = {'image': image}
                image_features = image_encoder(image_obs)

                if isinstance(image_features, dict):
                    image_features = image_features['image']

                if projection_head is not None:
                    image_features = projection_head(image_features)

                loss = nn.functional.mse_loss(image_features, target_features)

                val_loss += loss.item() * state.shape[0]
                val_samples += state.shape[0]

        val_loss /= val_samples

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Log to wandb
        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_dir = f"checkpoints/{args.exp_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint = {
                'epoch': epoch + 1,
                'image_encoder': image_encoder.state_dict(),
                'projection_head': projection_head.state_dict() if projection_head else None,
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }

            torch.save(checkpoint, f"{checkpoint_dir}/best_model.pth")
            print(f"Saved best model with val_loss = {val_loss:.6f}")

        # Early stopping
        if epoch + 1 >= args.min_epochs and val_loss < args.early_stop_threshold:
            print(f"Early stopping at epoch {epoch+1} with val_loss = {val_loss:.6f}")
            break

    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train image encoder alignment")
    parser.add_argument('--eps', type=int, default=200, help='Number of episodes')
    parser.add_argument('--vars', type=int, default=25, help='Number of variations')
    parser.add_argument('--state_checkpoint', type=str, required=True,
                       help='Path to trained state encoder checkpoint')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--dataset_path', type=str,
                       default="/proj/rep-learning-robotics/users/x_andri/dmfd/data/ClothFold_vars-{vars}_eps-{eps}_image_based_trajs.pkl",
                       help='Path to dataset')
    parser.add_argument('--visual_encoder', type=str, default='ResNet18Conv',
                       help='Type of visual encoder')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--min_epochs', type=int, default=10, help='Minimum epochs before early stopping')
    parser.add_argument('--early_stop_threshold', type=float, default=1e-6,
                       help='Early stopping threshold')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')

    args = parser.parse_args()

    # Format dataset path
    if '{vars}' in args.dataset_path and '{eps}' in args.dataset_path:
        args.dataset_path = args.dataset_path.format(vars=args.vars, eps=args.eps)

    train_image_encoder_alignment(args)


if __name__ == '__main__':
    main()