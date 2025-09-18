"""
Unified optimizer setup for diffusion policies.

Provides consistent parameter grouping and optimizer creation across all policy types.
"""

import torch
from torch.optim import AdamW, Adam
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class OptimizerGroup:
    """
    Represents a parameter group with specific settings.
    """
    def __init__(self, 
                 name: str,
                 parameters: Union[torch.nn.Module, List[torch.nn.Parameter]],
                 weight_decay: float = 0.0,
                 lr_multiplier: float = 1.0):
        """
        Args:
            name: Name of the parameter group
            parameters: Module or list of parameters
            weight_decay: Weight decay for this group
            lr_multiplier: Learning rate multiplier for this group
        """
        self.name = name
        self.weight_decay = weight_decay
        self.lr_multiplier = lr_multiplier
        
        if isinstance(parameters, torch.nn.Module):
            self.parameters = list(parameters.parameters())
        elif isinstance(parameters, list):
            self.parameters = parameters
        else:
            raise ValueError(f"Parameters must be Module or list, got {type(parameters)}")
        
        # Filter out parameters that don't require gradients
        self.parameters = [p for p in self.parameters if p.requires_grad]
        
        if len(self.parameters) == 0:
            raise ValueError(f"Parameter group '{name}' has no trainable parameters")
    
    def to_dict(self, base_lr: float) -> Dict[str, Any]:
        """Convert to optimizer parameter group dictionary."""
        return {
            'params': self.parameters,
            'weight_decay': self.weight_decay,
            'lr': base_lr * self.lr_multiplier
        }
    
    def num_parameters(self) -> int:
        """Count total parameters in this group."""
        return sum(p.numel() for p in self.parameters)


class BaseOptimizedPolicy(ABC):
    """
    Abstract base class for policies with standardized optimizer setup.
    """
    
    @abstractmethod
    def get_optimizer_groups(self) -> List[OptimizerGroup]:
        """
        Return parameter groups for optimization.
        
        Returns:
            List of OptimizerGroup objects
        """
        pass
    
    def create_optimizer(self,
                        learning_rate: float,
                        betas: tuple = (0.9, 0.999),
                        eps: float = 1e-8,
                        optimizer_type: str = 'adamw') -> torch.optim.Optimizer:
        """
        Create optimizer from parameter groups.
        
        Args:
            learning_rate: Base learning rate
            betas: Adam betas
            eps: Adam epsilon
            optimizer_type: Type of optimizer ('adamw', 'adam')
            
        Returns:
            Configured optimizer
        """
        groups = self.get_optimizer_groups()
        
        if len(groups) == 0:
            raise ValueError("No parameter groups defined")
        
        # Convert to optimizer format
        param_groups = [group.to_dict(learning_rate) for group in groups]
        
        # Create optimizer
        if optimizer_type.lower() == 'adamw':
            optimizer = AdamW(param_groups, betas=betas, eps=eps)
        elif optimizer_type.lower() == 'adam':
            optimizer = Adam(param_groups, betas=betas, eps=eps)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Log parameter counts
        total_params = sum(group.num_parameters() for group in groups)
        print(f"Created {optimizer_type} optimizer with {len(groups)} parameter groups:")
        for group in groups:
            print(f"  {group.name}: {group.num_parameters():,} parameters, "
                  f"wd={group.weight_decay}, lr_mult={group.lr_multiplier}")
        print(f"Total trainable parameters: {total_params:,}")
        
        return optimizer


class DiffusionPolicyOptimizer:
    """
    Factory for creating optimizers for different diffusion policy types.
    """
    
    @staticmethod
    def create_transformer_optimizer(policy,
                                   transformer_weight_decay: float = 1e-6,
                                   encoder_weight_decay: Union[float, Dict[str, float]] = 1e-4,
                                   learning_rate: float = 1e-4,
                                   betas: tuple = (0.95, 0.999)) -> torch.optim.Optimizer:
        """
        Create optimizer for transformer-based policies.
        
        Args:
            policy: Policy object
            transformer_weight_decay: Weight decay for transformer parameters
            encoder_weight_decay: Weight decay for encoder parameters
            learning_rate: Base learning rate
            betas: Adam betas
            
        Returns:
            Configured optimizer
        """
        groups = []

        # Helper to extract encoder-specific weight decay if a dict is provided
        def _enc_wd(key: str, default: float) -> float:
            if isinstance(encoder_weight_decay, dict):
                return float(encoder_weight_decay.get(key, default))
            return float(encoder_weight_decay)
        
        # Transformer parameters (main model)
        if hasattr(policy, 'model') and hasattr(policy.model, 'get_optim_groups'):
            # Use model's own parameter grouping if available
            transformer_groups = policy.model.get_optim_groups(weight_decay=transformer_weight_decay)
            for i, group in enumerate(transformer_groups):
                groups.append({
                    'params': group['params'],
                    'weight_decay': group.get('weight_decay', transformer_weight_decay),
                    'lr': learning_rate
                })
        elif hasattr(policy, 'model'):
            groups.append({
                'params': policy.model.parameters(),
                'weight_decay': transformer_weight_decay,
                'lr': learning_rate
            })
        
        # Image encoder parameters
        if hasattr(policy, 'obs_encoder') and policy.obs_encoder is not None:
            groups.append({
                'params': policy.obs_encoder.parameters(),
                'weight_decay': _enc_wd('obs', encoder_weight_decay if not isinstance(encoder_weight_decay, dict) else 1e-4),
                'lr': learning_rate
            })
        
        # State encoder parameters (for privileged/double_modality)
        if hasattr(policy, 'state_encoder') and policy.state_encoder is not None:
            groups.append({
                'params': policy.state_encoder.parameters(),
                'weight_decay': _enc_wd('state', encoder_weight_decay if not isinstance(encoder_weight_decay, dict) else 1e-4),
                'lr': learning_rate
            })
        
        # Shared encoder parameters
        shared_encoder = None
        if hasattr(policy, 'shared_encoder') and getattr(policy, 'shared_encoder') is not None:
            shared_encoder = policy.shared_encoder
        elif hasattr(policy, 'feature_fusion') and hasattr(policy.feature_fusion, 'shared_encoder'):
            shared_encoder = policy.feature_fusion.shared_encoder
        if shared_encoder is not None:
            groups.append({
                'params': shared_encoder.parameters(),
                'weight_decay': _enc_wd('shared', encoder_weight_decay if not isinstance(encoder_weight_decay, dict) else 1e-4),
                'lr': learning_rate
            })
        
        # Additional components (gating, etc.)
        if hasattr(policy, 'priv_gating_alpha'):
            groups.append({
                'params': [policy.priv_gating_alpha],
                'weight_decay': 0.0,  # No weight decay for gating parameters
                'lr': learning_rate
            })
        
        if len(groups) == 0:
            raise ValueError("No parameter groups found for transformer optimizer")
        
        optimizer = AdamW(groups, betas=betas)
        
        # Log parameter counts
        total_params = sum(sum(p.numel() for p in group['params']) for group in groups)
        trainable_params = sum(sum(p.numel() for p in group['params'] if p.requires_grad) for group in groups)
        print(f"Transformer optimizer created with {len(groups)} parameter groups")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return optimizer
    
    @staticmethod
    def create_unet_optimizer(policy,
                            model_weight_decay: float = 1e-6,
                            encoder_weight_decay: float = 1e-4,
                            learning_rate: float = 1e-4,
                            betas: tuple = (0.95, 0.999)) -> torch.optim.Optimizer:
        """
        Create optimizer for UNet-based policies.
        
        Args:
            policy: Policy object
            model_weight_decay: Weight decay for main model parameters
            encoder_weight_decay: Weight decay for encoder parameters
            learning_rate: Base learning rate
            betas: Adam betas
            
        Returns:
            Configured optimizer
        """
        groups = []
        
        # Main model parameters
        if hasattr(policy, 'nets'):
            # UNet policies typically have 'nets' dictionary
            if 'action_model' in policy.nets:
                groups.append({
                    'params': policy.nets['action_model'].parameters(),
                    'weight_decay': model_weight_decay,
                    'lr': learning_rate
                })
            
            if 'vision_encoder' in policy.nets:
                groups.append({
                    'params': policy.nets['vision_encoder'].parameters(),
                    'weight_decay': encoder_weight_decay,
                    'lr': learning_rate
                })
        elif hasattr(policy, 'model'):
            groups.append({
                'params': policy.model.parameters(),
                'weight_decay': model_weight_decay,
                'lr': learning_rate
            })
        
        if len(groups) == 0:
            raise ValueError("No parameter groups found for UNet optimizer")
        
        optimizer = AdamW(groups, betas=betas)
        
        # Log parameter counts
        total_params = sum(sum(p.numel() for p in group['params']) for group in groups)
        trainable_params = sum(sum(p.numel() for p in group['params'] if p.requires_grad) for group in groups)
        print(f"UNet optimizer created with {len(groups)} parameter groups")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return optimizer
    
    @staticmethod
    def create_generic_optimizer(policy,
                               weight_decay: float = 1e-6,
                               learning_rate: float = 1e-4,
                               betas: tuple = (0.95, 0.999)) -> torch.optim.Optimizer:
        """
        Create optimizer for any policy type (fallback).
        
        Args:
            policy: Policy object
            weight_decay: Weight decay for all parameters
            learning_rate: Learning rate
            betas: Adam betas
            
        Returns:
            Configured optimizer
        """
        # Collect all parameters
        all_params = []
        
        if hasattr(policy, 'parameters'):
            all_params.extend(policy.parameters())
        else:
            # Try to find parameters in common attributes
            for attr_name in ['model', 'nets', 'obs_encoder', 'state_encoder', 'shared_encoder']:
                if hasattr(policy, attr_name):
                    attr = getattr(policy, attr_name)
                    if hasattr(attr, 'parameters'):
                        all_params.extend(attr.parameters())
                    elif isinstance(attr, dict):
                        for sub_attr in attr.values():
                            if hasattr(sub_attr, 'parameters'):
                                all_params.extend(sub_attr.parameters())
        
        # Filter out non-trainable parameters
        trainable_params = [p for p in all_params if p.requires_grad]
        
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found")
        
        optimizer = AdamW(trainable_params, lr=learning_rate, betas=betas, weight_decay=weight_decay)
        
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Generic optimizer created for {total_params:,} parameters")
        
        return optimizer
    
    @staticmethod
    def auto_create_optimizer(policy,
                            transformer_weight_decay: float = 1e-6,
                            encoder_weight_decay: float = 1e-4,
                            weight_decay: float = 1e-6,
                            learning_rate: float = 1e-4,
                            betas: tuple = (0.95, 0.999)) -> torch.optim.Optimizer:
        """
        Automatically create the appropriate optimizer based on policy type.
        
        Args:
            policy: Policy object
            transformer_weight_decay: Weight decay for transformer parameters
            encoder_weight_decay: Weight decay for encoder parameters  
            weight_decay: Fallback weight decay
            learning_rate: Learning rate
            betas: Adam betas
            
        Returns:
            Configured optimizer
        """
        policy_class_name = policy.__class__.__name__.lower()
        
        if 'transformer' in policy_class_name:
            return DiffusionPolicyOptimizer.create_transformer_optimizer(
                policy, transformer_weight_decay, encoder_weight_decay, learning_rate, betas
            )
        elif 'unet' in policy_class_name:
            return DiffusionPolicyOptimizer.create_unet_optimizer(
                policy, weight_decay, encoder_weight_decay, learning_rate, betas
            )
        else:
            return DiffusionPolicyOptimizer.create_generic_optimizer(
                policy, weight_decay, learning_rate, betas
            )
