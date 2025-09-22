"""
Shape validation and logging utilities for diffusion policy training.

This module provides comprehensive debugging and validation of tensor shapes
throughout the diffusion policy pipeline to ensure consistency and catch
dimension mismatches early.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from functools import reduce
import operator


class ShapeValidator:
    """
    Comprehensive shape validation and logging utility for diffusion policy training.
    
    This class provides detailed logging and assertion functionality to validate
    that tensor shapes are consistent between datasets, policies, and training loops.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the shape validator with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
        self.validation_history = []
        
    def log_batch_shapes(self, batch: Dict[str, torch.Tensor], batch_name: str = "batch") -> None:
        """
        Log the shapes of all tensors in a batch with detailed information.
        
        Args:
            batch: Dictionary containing tensor data
            batch_name: Name/description of the batch for logging
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BATCH SHAPES ANALYSIS: {batch_name}")
        self.logger.info(f"{'='*60}")
        
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                shape = list(tensor.shape)
                dtype = tensor.dtype
                device = tensor.device
                numel = tensor.numel()
                memory_mb = tensor.element_size() * numel / (1024 * 1024)
                
                self.logger.info(f"  {key:12s}: shape={shape} dtype={dtype} device={device}")
                self.logger.info(f"  {'':<12s}  numel={numel:,} memory={memory_mb:.2f}MB")
                
                # Additional info for common tensor patterns
                if len(shape) >= 2:
                    batch_size, seq_len = shape[0], shape[1]
                    self.logger.info(f"  {'':<12s}  batch_size={batch_size}, seq_len={seq_len}")
                    if len(shape) > 2:
                        remaining_dims = shape[2:]
                        flat_dim = np.prod(remaining_dims)
                        self.logger.info(f"  {'':<12s}  remaining_dims={remaining_dims}, flattened={flat_dim}")
                        
        self.logger.info(f"{'='*60}\n")
        
    def validate_policy_input_output(self, policy, batch: Dict[str, torch.Tensor], 
                                   model_type: str, device: torch.device) -> Dict[str, Any]:
        """
        Comprehensive validation of policy input/output shapes with assertions.
        
        Args:
            policy: The policy model to validate
            batch: Input batch for testing
            model_type: Type of model (transformer-lowdim, transformer-image, etc.)
            device: Device for computation
            
        Returns:
            Dictionary with validation results and computed dimensions
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"POLICY VALIDATION: {model_type}")
        self.logger.info(f"{'='*60}")
        
        results = {
            'model_type': model_type,
            'input_shapes': {},
            'output_shapes': {},
            'computed_dims': {},
            'validation_passed': True,
            'errors': []
        }
        
        try:
            # Move batch to device
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            
            # Log input shapes
            self.logger.info("INPUT SHAPES:")
            for key, tensor in batch_dev.items():
                results['input_shapes'][key] = list(tensor.shape)
                self.logger.info(f"  {key}: {list(tensor.shape)}")
            
            # Validate based on model type
            if model_type in ("transformer-privileged", "transformer_privileged"):
                results.update(self._validate_privileged_policy(policy, batch_dev))
            elif model_type in ("transformer-image", "transformer_image"):
                results.update(self._validate_image_policy(policy, batch_dev))
            elif model_type in ("transformer-lowdim", "transformer_lowdim"):
                results.update(self._validate_lowdim_policy(policy, batch_dev))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Test prediction
            self.logger.info("\nTESTING POLICY PREDICTION:")
            with torch.no_grad():
                if model_type in ("transformer-privileged", "transformer_privileged"):
                    pred_result = policy.predict_action({"image": batch_dev["image"]})
                else:
                    pred_result = policy.predict_action({"obs": batch_dev["obs"]})
                    
                if isinstance(pred_result, dict) and "action" in pred_result:
                    action_shape = list(pred_result["action"].shape)
                    results['output_shapes']['predicted_action'] = action_shape
                    self.logger.info(f"  predicted_action: {action_shape}")
                else:
                    self.logger.error(f"  Unexpected prediction result type: {type(pred_result)}")
                    results['errors'].append(f"Unexpected prediction result: {type(pred_result)}")
                    results['validation_passed'] = False
            
            # Test loss computation
            self.logger.info("\nTESTING LOSS COMPUTATION:")
            with torch.no_grad():
                try:
                    loss_out = policy.compute_loss(batch_dev)
                    # Handle both tuple (loss, components) and plain tensor returns
                    if isinstance(loss_out, tuple) and len(loss_out) == 2:
                        loss, comp = loss_out
                    else:
                        loss = loss_out
                    loss_val = float(loss)
                    results['initial_loss'] = loss_val
                    self.logger.info(f"  initial_loss: {loss_val:.6f}")
                    
                    # Validate loss is reasonable
                    if np.isnan(loss_val) or np.isinf(loss_val):
                        results['errors'].append(f"Invalid loss value: {loss_val}")
                        results['validation_passed'] = False
                        self.logger.error(f"  ERROR: Invalid loss value: {loss_val}")
                except Exception as e:
                    results['errors'].append(f"Loss computation failed: {str(e)}")
                    results['validation_passed'] = False
                    self.logger.error(f"  ERROR: Loss computation failed: {e}")
            
        except Exception as e:
            results['errors'].append(f"Policy validation failed: {str(e)}")
            results['validation_passed'] = False
            self.logger.error(f"POLICY VALIDATION FAILED: {e}")
            
        # Final validation summary
        status = "PASSED" if results['validation_passed'] else "FAILED"
        self.logger.info(f"\nVALIDATION STATUS: {status}")
        if results['errors']:
            self.logger.error("ERRORS FOUND:")
            for error in results['errors']:
                self.logger.error(f"  - {error}")
        
        self.logger.info(f"{'='*60}\n")
        
        # Store in history
        self.validation_history.append(results)
        
        return results
        
    def _validate_privileged_policy(self, policy, batch_dev: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validate privileged policy dimensions."""
        results = {'computed_dims': {}}
        
        # Expected keys
        expected_keys = {'image', 'state', 'action'}
        missing_keys = expected_keys - set(batch_dev.keys())
        if missing_keys:
            results.setdefault('errors', []).append(f"Missing keys for privileged policy: {missing_keys}")
            
        if 'image' in batch_dev:
            img_shape = list(batch_dev['image'].shape)
            # For privileged: [B, T, H, W, C] or [B, T, flattened]
            if len(img_shape) == 5:  # [B, T, H, W, C]
                B, T, H, W, C = img_shape
                Do_img = H * W * C
            elif len(img_shape) == 3:  # [B, T, flattened]
                Do_img = img_shape[2]
            else:
                raise ValueError(f"Unexpected image shape for privileged policy: {img_shape}")
            results['computed_dims']['Do_img'] = Do_img
            self.logger.info(f"  Computed Do_img: {Do_img} from shape {img_shape}")
            
        if 'state' in batch_dev:
            state_shape = list(batch_dev['state'].shape)
            # For state: [B, T, state_dim]
            if len(state_shape) != 3:
                results.setdefault('errors', []).append(f"Unexpected state shape: {state_shape}, expected [B, T, state_dim]")
            else:
                Do_state = state_shape[2]
                results['computed_dims']['Do_state'] = Do_state
                self.logger.info(f"  Computed Do_state: {Do_state} from shape {state_shape}")
                
        if 'action' in batch_dev:
            action_shape = list(batch_dev['action'].shape)
            Da = action_shape[-1]  # Last dimension
            results['computed_dims']['Da'] = Da
            self.logger.info(f"  Computed Da: {Da} from shape {action_shape}")
            
        return results
        
    def _validate_image_policy(self, policy, batch_dev: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validate image policy dimensions."""
        results = {'computed_dims': {}}
        
        if 'obs' in batch_dev:
            obs_shape = list(batch_dev['obs'].shape)
            # For image obs: flatten from index 2 onward
            if len(obs_shape) > 2:
                Do = np.prod(obs_shape[2:])
            else:
                Do = obs_shape[-1]
            results['computed_dims']['Do'] = Do
            self.logger.info(f"  Computed Do: {Do} from shape {obs_shape}")
            
        if 'action' in batch_dev:
            action_shape = list(batch_dev['action'].shape)
            Da = action_shape[-1]
            results['computed_dims']['Da'] = Da
            self.logger.info(f"  Computed Da: {Da} from shape {action_shape}")
            
        return results
        
    def _validate_lowdim_policy(self, policy, batch_dev: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validate low-dimensional policy dimensions."""
        results = {'computed_dims': {}}
        
        if 'obs' in batch_dev:
            obs_shape = list(batch_dev['obs'].shape)
            # For lowdim obs: use last dimension
            Do = obs_shape[-1]
            results['computed_dims']['Do'] = Do
            self.logger.info(f"  Computed Do: {Do} from shape {obs_shape}")
            
        if 'action' in batch_dev:
            action_shape = list(batch_dev['action'].shape)
            Da = action_shape[-1]
            results['computed_dims']['Da'] = Da
            self.logger.info(f"  Computed Da: {Da} from shape {action_shape}")
            
        return results
        
    def assert_dimensions_match(self, computed_dims: Dict[str, int], 
                              policy_dims: Dict[str, int], 
                              tolerance: float = 1e-6) -> None:
        """
        Assert that computed dimensions match expected policy dimensions.
        
        Args:
            computed_dims: Dimensions computed from data
            policy_dims: Dimensions expected by policy
            tolerance: Tolerance for floating point comparisons
        """
        self.logger.info("\nDIMENSION MATCHING ASSERTIONS:")
        
        for key in computed_dims:
            if key in policy_dims:
                computed = computed_dims[key]
                expected = policy_dims[key]
                
                if computed != expected:
                    error_msg = f"Dimension mismatch for {key}: computed={computed}, expected={expected}"
                    self.logger.error(f"  ASSERTION FAILED: {error_msg}")
                    raise AssertionError(error_msg)
                else:
                    self.logger.info(f"  ✓ {key}: {computed} == {expected}")
                    
        self.logger.info("All dimension assertions passed!\n")
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation runs."""
        if not self.validation_history:
            return {"message": "No validations performed yet"}
            
        total_runs = len(self.validation_history)
        passed_runs = sum(1 for v in self.validation_history if v['validation_passed'])
        
        return {
            "total_validations": total_runs,
            "passed_validations": passed_runs,
            "failed_validations": total_runs - passed_runs,
            "success_rate": passed_runs / total_runs if total_runs > 0 else 0,
            "latest_validation": self.validation_history[-1] if self.validation_history else None
        }


def create_shape_validator(verbose: bool = True) -> ShapeValidator:
    """
    Create a configured shape validator with appropriate logging.
    
    Args:
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured ShapeValidator instance
    """
    # Set up logger
    logger = logging.getLogger("shape_validator")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    # Create console handler if not already exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(name)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return ShapeValidator(logger)


def compute_obs_dim_old_style(shape: List[int], is_image: bool = False) -> int:
    """
    Compute observation dimension using old diffusion policy style.
    
    This function mimics how the old diffusion policy computed observation
    dimensions, which was more context-aware than the current _flatten_obs_dim.
    
    Args:
        shape: Shape of the observation tensor (e.g., [B, T, ...])
        is_image: Whether this is an image observation
        
    Returns:
        Computed observation dimension
    """
    if len(shape) <= 2:
        return shape[-1]
    
    if is_image:
        # For image observations: flatten spatial and channel dimensions
        # [B, T, H, W, C] -> H * W * C
        return np.prod(shape[2:])
    else:
        # For state observations: use last dimension only
        # [B, T, state_dim] -> state_dim
        return shape[-1]


def validate_obs_dim_computation(shape: List[int], current_result: int, 
                               is_image: bool = False) -> bool:
    """
    Validate that observation dimension computation matches expected behavior.
    
    Args:
        shape: Input shape
        current_result: Current computed result
        is_image: Whether this is an image observation
        
    Returns:
        True if computation is correct, False otherwise
    """
    expected = compute_obs_dim_old_style(shape, is_image)
    return current_result == expected