"""
Unified image processing module for diffusion policies.

Handles tensor format conversions and preprocessing consistently across all model types.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class ImageProcessor:
    """
    Standardized image preprocessing for diffusion policies.
    
    Handles format conversions between different tensor layouts:
    - Standard format: [B, C, H, W] (PyTorch convention)
    - Model format: [B, H, W, C] (Some models expect this)
    """
    
    @staticmethod
    def to_model_format(images: torch.Tensor, model_type: str) -> torch.Tensor:
        """
        Convert images to the format expected by model_type.
        
        Args:
            images: Input tensor, assumed to be [B, C, H, W] or [B, T, C, H, W]
            model_type: Type of model ('privileged', 'transformer', 'unet', etc.)
            
        Returns:
            Tensor in the appropriate format for the model
        """
        if model_type in ['privileged', 'transformer', 'double_modality']:
            # These models expect [B, H, W, C] or [B, T, H, W, C]
            if len(images.shape) == 4:  # [B, C, H, W]
                return images.permute(0, 2, 3, 1).contiguous()
            elif len(images.shape) == 5:  # [B, T, C, H, W]
                return images.permute(0, 1, 3, 4, 2).contiguous()
            else:
                raise ValueError(f"Unexpected image tensor shape: {images.shape}")
        else:
            # Standard PyTorch format [B, C, H, W] or [B, T, C, H, W]
            return images
    
    @staticmethod
    def to_encoder_format(images: torch.Tensor, encoder_type: str) -> torch.Tensor:
        """
        Convert images to the format expected by encoder_type.
        
        Args:
            images: Input tensor in any format
            encoder_type: Type of encoder ('DrQCNN', 'ResNet18Conv', etc.)
            
        Returns:
            Tensor in the format expected by the encoder
        """
        if encoder_type == 'DrQCNN':
            # DrQCNN expects [B, C, H, W]
            if len(images.shape) == 4 and images.shape[-1] in [1, 3]:  # [B, H, W, C]
                return images.permute(0, 3, 1, 2).contiguous()
            elif len(images.shape) == 5 and images.shape[-1] in [1, 3]:  # [B, T, H, W, C]
                B, T, H, W, C = images.shape
                return images.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
            else:
                return images
        else:
            # Robomimic encoders typically expect [B, H, W, C]
            if len(images.shape) == 4 and images.shape[1] in [1, 3]:  # [B, C, H, W]
                return images.permute(0, 2, 3, 1).contiguous()
            elif len(images.shape) == 5 and images.shape[2] in [1, 3]:  # [B, T, C, H, W]
                return images.permute(0, 1, 3, 4, 2).contiguous()
            else:
                return images
    
    @staticmethod
    def validate_image_tensor(images: torch.Tensor, 
                            expected_shape: Optional[Tuple[int, ...]] = None,
                            name: str = "images") -> None:
        """
        Validate image tensor properties.
        
        Args:
            images: Tensor to validate
            expected_shape: Expected shape (None to skip shape check)
            name: Name for error messages
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(images)}")
        
        if len(images.shape) < 3:
            raise ValueError(f"{name} must have at least 3 dimensions, got shape {images.shape}")
        
        if len(images.shape) > 5:
            raise ValueError(f"{name} has too many dimensions, got shape {images.shape}")
        
        if expected_shape is not None:
            if images.shape != expected_shape:
                raise ValueError(f"{name} shape mismatch. Expected {expected_shape}, got {images.shape}")
        
        # Check for common issues
        if images.dtype not in [torch.float32, torch.float16]:
            print(f"Warning: {name} has dtype {images.dtype}, consider using float32 or float16")
        
        if torch.isnan(images).any():
            raise ValueError(f"{name} contains NaN values")
        
        if torch.isinf(images).any():
            raise ValueError(f"{name} contains infinite values")
    
    @staticmethod
    def resize_images(images: torch.Tensor, 
                     target_size: Tuple[int, int],
                     mode: str = 'bilinear') -> torch.Tensor:
        """
        Resize images to target size.
        
        Args:
            images: Input images in any format
            target_size: (height, width)
            mode: Interpolation mode
            
        Returns:
            Resized images in the same format as input
        """
        original_shape = images.shape
        
        # Convert to standard format for resizing
        if len(images.shape) == 4:  # [B, H, W, C] or [B, C, H, W]
            if images.shape[-1] in [1, 3]:  # [B, H, W, C]
                x = images.permute(0, 3, 1, 2)  # -> [B, C, H, W]
                need_permute_back = True
            else:  # [B, C, H, W]
                x = images
                need_permute_back = False
        elif len(images.shape) == 5:  # [B, T, H, W, C] or [B, T, C, H, W]
            if images.shape[-1] in [1, 3]:  # [B, T, H, W, C]
                B, T, H, W, C = images.shape
                x = images.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
                need_permute_back = True
            else:  # [B, T, C, H, W]
                B, T, C, H, W = images.shape
                x = images.view(-1, C, H, W)
                need_permute_back = False
        else:
            raise ValueError(f"Cannot resize tensor with shape {images.shape}")
        
        # Resize
        x = F.interpolate(x, size=target_size, mode=mode, align_corners=False)
        
        # Convert back to original format
        if len(original_shape) == 5:
            if need_permute_back:
                _, C, H, W = x.shape
                x = x.view(B, T, C, H, W).permute(0, 1, 3, 4, 2)
            else:
                _, C, H, W = x.shape
                x = x.view(B, T, C, H, W)
        elif need_permute_back:
            x = x.permute(0, 2, 3, 1)
        
        return x
    
    @staticmethod
    def normalize_images(images: torch.Tensor, 
                        mean: Optional[Tuple[float, ...]] = None,
                        std: Optional[Tuple[float, ...]] = None) -> torch.Tensor:
        """
        Normalize images with mean and std.
        
        Args:
            images: Input images
            mean: Mean values per channel (None for [0.5, 0.5, 0.5])
            std: Std values per channel (None for [0.5, 0.5, 0.5])
            
        Returns:
            Normalized images
        """
        if mean is None:
            mean = (0.5, 0.5, 0.5)
        if std is None:
            std = (0.5, 0.5, 0.5)
        
        # Convert to tensor
        device = images.device
        mean_tensor = torch.tensor(mean, device=device)
        std_tensor = torch.tensor(std, device=device)
        
        # Handle different formats
        if len(images.shape) == 4:
            if images.shape[-1] in [1, 3]:  # [B, H, W, C]
                # Reshape for broadcasting
                mean_tensor = mean_tensor.view(1, 1, 1, -1)
                std_tensor = std_tensor.view(1, 1, 1, -1)
            else:  # [B, C, H, W]
                mean_tensor = mean_tensor.view(1, -1, 1, 1)
                std_tensor = std_tensor.view(1, -1, 1, 1)
        elif len(images.shape) == 5:
            if images.shape[-1] in [1, 3]:  # [B, T, H, W, C]
                mean_tensor = mean_tensor.view(1, 1, 1, 1, -1)
                std_tensor = std_tensor.view(1, 1, 1, 1, -1)
            else:  # [B, T, C, H, W]
                mean_tensor = mean_tensor.view(1, 1, -1, 1, 1)
                std_tensor = std_tensor.view(1, 1, -1, 1, 1)
        
        return (images - mean_tensor) / std_tensor
