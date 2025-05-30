"""Optimization utilities for Hyena-GLT training."""

from typing import Dict, Any, Optional, List, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings


class AdamWWithScheduler(torch.optim.AdamW):
    """AdamW optimizer with built-in scheduling."""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        scheduler_type: str = "cosine"
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        self.current_step = 0
        self.base_lr = lr
        
    def step(self, closure=None):
        """Perform optimization step with learning rate scheduling."""
        self.current_step += 1
        self._update_lr()
        return super().step(closure)
    
    def _update_lr(self):
        """Update learning rate based on schedule."""
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Schedule after warmup
            if self.scheduler_type == "cosine":
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.scheduler_type == "linear":
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.base_lr * (1 - progress)
            else:
                lr = self.base_lr
        
        for param_group in self.param_groups:
            param_group['lr'] = lr


class LayerWiseDecayOptimizer:
    """Implements layer-wise learning rate decay."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        base_lr: float = 1e-3,
        layer_decay: float = 0.9,
        **optimizer_kwargs
    ):
        self.model = model
        self.base_lr = base_lr
        self.layer_decay = layer_decay
        self.optimizer_kwargs = optimizer_kwargs
        
    def create_optimizer(self) -> torch.optim.AdamW:
        """Create optimizer with layer-wise decay."""
        param_groups = self._get_layer_wise_param_groups()
        return torch.optim.AdamW(param_groups, **self.optimizer_kwargs)
    
    def _get_layer_wise_param_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups with layer-wise learning rates."""
        param_groups = []
        
        # Get layer depths
        layer_depths = self._get_layer_depths()
        max_depth = max(layer_depths.values()) if layer_depths else 0
        
        # Group parameters by layer depth
        depth_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                depth = layer_depths.get(name, max_depth)
                if depth not in depth_params:
                    depth_params[depth] = []
                depth_params[depth].append(param)
        
        # Create parameter groups with scaled learning rates
        for depth, params in depth_params.items():
            lr_scale = self.layer_decay ** (max_depth - depth)
            param_groups.append({
                'params': params,
                'lr': self.base_lr * lr_scale,
                'layer_depth': depth
            })
        
        return param_groups
    
    def _get_layer_depths(self) -> Dict[str, int]:
        """Get depth of each parameter in the model."""
        depths = {}
        
        # Simple heuristic: count dots in parameter names
        for name, _ in self.model.named_parameters():
            # Count the number of blocks/layers
            if 'hyena_blocks' in name:
                # Extract block number
                parts = name.split('.')
                block_idx = None
                for i, part in enumerate(parts):
                    if part == 'hyena_blocks' and i + 1 < len(parts):
                        try:
                            block_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            continue
                depths[name] = block_idx if block_idx is not None else 0
            else:
                # For non-block parameters, assign depth 0
                depths[name] = 0
        
        return depths


class LinearWarmupCosineDecayScheduler(_LRScheduler):
    """Linear warmup followed by cosine decay scheduler."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
                for base_lr in self.base_lrs
            ]


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    layer_wise_decay: Optional[float] = None,
    **kwargs
) -> Optimizer:
    """Create optimizer for model training."""
    
    # Filter parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if layer_wise_decay is not None:
        # Use layer-wise decay
        layer_optimizer = LayerWiseDecayOptimizer(
            model, learning_rate, layer_wise_decay, weight_decay=weight_decay, **kwargs
        )
        return layer_optimizer.create_optimizer()
    
    # Standard optimizer configuration
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    **kwargs
) -> Optional[_LRScheduler]:
    """Create learning rate scheduler."""
    
    if scheduler_type.lower() == "cosine":
        return LinearWarmupCosineDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            **kwargs
        )
    elif scheduler_type.lower() == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=kwargs.get('end_factor', 0.1),
            total_iters=total_steps,
            **{k: v for k, v in kwargs.items() if k != 'end_factor'}
        )
    elif scheduler_type.lower() == "constant":
        return None
    else:
        warnings.warn(f"Unknown scheduler type: {scheduler_type}, using constant LR")
        return None


def get_parameter_names(model: torch.nn.Module, forbidden_layer_types: List[type]) -> List[str]:
    """Get names of parameters that should not have weight decay applied."""
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add parameters of the current module
    result += list(model._parameters.keys())
    return result


def configure_weight_decay(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay_bias: bool = True,
    no_decay_norm: bool = True
) -> List[Dict[str, Any]]:
    """Configure weight decay for different parameter types."""
    
    # Define layer types that should not have weight decay
    no_decay_types = []
    if no_decay_norm:
        no_decay_types.extend([torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d])
    
    # Get parameter names without weight decay
    no_decay_names = set()
    if no_decay_bias:
        no_decay_names.update([n for n, _ in model.named_parameters() if 'bias' in n])
    
    # Create parameter groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should have weight decay
        should_decay = True
        
        # Check by name
        if name in no_decay_names:
            should_decay = False
        
        # Check by parent module type
        for module_name, module in model.named_modules():
            if name.startswith(module_name) and isinstance(module, tuple(no_decay_types)):
                should_decay = False
                break
        
        if should_decay:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
