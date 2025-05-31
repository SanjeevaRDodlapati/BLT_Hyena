"""
Advanced Mixed Precision Training Implementation for BLT_Hyena.

This module provides comprehensive mixed precision training capabilities including:
- FP16/BF16/FP8 support with dynamic scaling
- Kernel-level precision optimization
- Memory-efficient gradient accumulation
- Precision-aware loss scaling strategies
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

try:
    from transformer_engine import pytorch as te
    HAS_TRANSFORMER_ENGINE = True
except ImportError:
    HAS_TRANSFORMER_ENGINE = False
    te = None

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Supported precision modes for training."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    MIXED_FP16 = "mixed_fp16"
    MIXED_BF16 = "mixed_bf16"
    ADAPTIVE = "adaptive"


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    
    # Basic precision settings
    mode: PrecisionMode = PrecisionMode.FP16
    enable_autocast: bool = True
    
    # Gradient scaling
    loss_scale: Optional[float] = None  # None for dynamic scaling
    init_scale: float = 2.0**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True
    
    # Advanced scaling strategies
    dynamic_loss_scale: bool = True
    per_layer_scaling: bool = False
    gradient_clipping: float = 1.0
    
    # FP8 specific settings (if available)
    fp8_format: str = "E4M3"  # E4M3 or E5M2
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "most_recent"
    
    # Kernel precision settings
    kernel_precision: str = "ieee"  # "ieee", "tf32", "tf32x3"
    max_num_imprecise_acc: Optional[int] = None
    
    # Memory optimizations
    cpu_offload: bool = False
    gradient_checkpointing: bool = False
    
    # Monitoring and debugging
    monitor_overflow: bool = True
    log_precision_stats: bool = False
    precision_check_interval: int = 100


class AdvancedGradScaler(GradScaler):
    """Enhanced gradient scaler with additional features."""
    
    def __init__(self, config: MixedPrecisionConfig):
        super().__init__(
            init_scale=config.init_scale,
            growth_factor=config.growth_factor,
            backoff_factor=config.backoff_factor,
            growth_interval=config.growth_interval,
            enabled=config.enabled,
        )
        self.config = config
        self.overflow_count = 0
        self.total_steps = 0
        self.precision_stats = {}
        
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss with monitoring."""
        self.total_steps += 1
        scaled_loss = super().scale(loss)
        
        if self.config.monitor_overflow:
            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                self.overflow_count += 1
                logger.warning(f"Loss overflow detected. Total overflows: {self.overflow_count}")
        
        return scaled_loss
    
    def step(self, optimizer: torch.optim.Optimizer, *args, **kwargs) -> Optional[float]:
        """Enhanced step with precision monitoring."""
        # Get gradients before stepping
        if self.config.log_precision_stats and self.total_steps % self.config.precision_check_interval == 0:
            self._log_gradient_stats(optimizer)
        
        # Apply gradient clipping if enabled
        if self.config.gradient_clipping > 0:
            for param_group in optimizer.param_groups:
                params_with_grad = [p for p in param_group['params'] if p.grad is not None]
                if params_with_grad:
                    torch.nn.utils.clip_grad_norm_(
                        params_with_grad, 
                        self.config.gradient_clipping
                    )
        
        return super().step(optimizer, *args, **kwargs)
    
    def _log_gradient_stats(self, optimizer: torch.optim.Optimizer) -> None:
        """Log gradient statistics for monitoring."""
        total_norm = 0.0
        param_count = 0
        
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(dtype=torch.float32)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        self.precision_stats[self.total_steps] = {
            'gradient_norm': total_norm,
            'param_count': param_count,
            'scale': self.get_scale(),
            'overflow_count': self.overflow_count,
        }
        
        if self.total_steps % (self.config.precision_check_interval * 10) == 0:
            logger.info(f"Precision stats - Step: {self.total_steps}, "
                       f"Grad norm: {total_norm:.6f}, Scale: {self.get_scale()}")


class MixedPrecisionManager:
    """Comprehensive mixed precision training manager."""
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.scaler = None
        self.fp8_recipe = None
        
        # Initialize based on precision mode
        self._setup_precision_mode()
        
    def _setup_precision_mode(self) -> None:
        """Setup precision mode and required components."""
        if self.config.mode in [PrecisionMode.FP16, PrecisionMode.MIXED_FP16]:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to FP32")
                self.config.mode = PrecisionMode.FP32
                return
                
            self.scaler = AdvancedGradScaler(self.config)
            logger.info("Initialized FP16 mixed precision training")
            
        elif self.config.mode in [PrecisionMode.BF16, PrecisionMode.MIXED_BF16]:
            if not torch.cuda.is_bf16_supported():
                logger.warning("BF16 not supported, falling back to FP16")
                self.config.mode = PrecisionMode.FP16
                self._setup_precision_mode()
                return
                
            # BF16 doesn't need gradient scaling
            logger.info("Initialized BF16 mixed precision training")
            
        elif self.config.mode == PrecisionMode.FP8:
            if not HAS_TRANSFORMER_ENGINE:
                logger.warning("Transformer Engine not available, falling back to FP16")
                self.config.mode = PrecisionMode.FP16
                self._setup_precision_mode()
                return
                
            self._setup_fp8()
            logger.info("Initialized FP8 mixed precision training")
            
        elif self.config.mode == PrecisionMode.ADAPTIVE:
            self._setup_adaptive_precision()
            logger.info("Initialized adaptive mixed precision training")
    
    def _setup_fp8(self) -> None:
        """Setup FP8 training with Transformer Engine."""
        if not HAS_TRANSFORMER_ENGINE:
            return
            
        self.fp8_recipe = te.recipe.DelayedScaling(
            margin=self.config.fp8_margin,
            interval=self.config.fp8_interval,
            fp8_format=getattr(te.recipe.Format, self.config.fp8_format),
            amax_history_len=self.config.fp8_amax_history_len,
            amax_compute_algo=self.config.fp8_amax_compute_algo,
        )
    
    def _setup_adaptive_precision(self) -> None:
        """Setup adaptive precision that switches based on conditions."""
        # Start with FP16 and adapt based on training dynamics
        self.config.mode = PrecisionMode.FP16
        self.scaler = AdvancedGradScaler(self.config)
        self.adaptive_mode = True
        self.switch_threshold = 5  # Switch after N consecutive overflows
        
    def get_autocast_context(self, **kwargs) -> Any:
        """Get appropriate autocast context for the precision mode."""
        if self.config.mode == PrecisionMode.FP32:
            return torch.autocast(device_type='cuda', enabled=False)
        elif self.config.mode in [PrecisionMode.FP16, PrecisionMode.MIXED_FP16]:
            return torch.autocast(device_type='cuda', dtype=torch.float16, **kwargs)
        elif self.config.mode in [PrecisionMode.BF16, PrecisionMode.MIXED_BF16]:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16, **kwargs)
        elif self.config.mode == PrecisionMode.FP8:
            # FP8 autocast is handled by Transformer Engine
            return torch.autocast(device_type='cuda', dtype=torch.float16, **kwargs)
        else:
            return torch.autocast(device_type='cuda', dtype=torch.float16, **kwargs)
    
    def get_fp8_context(self):
        """Get FP8 context manager if available."""
        if self.config.mode == PrecisionMode.FP8 and self.fp8_recipe is not None:
            return te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss if gradient scaling is enabled."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> Optional[float]:
        """Step optimizer with appropriate scaling."""
        if self.scaler is not None:
            return self.scaler.step(optimizer)
        else:
            optimizer.step()
            return None
    
    def update_scaler(self) -> None:
        """Update gradient scaler."""
        if self.scaler is not None:
            self.scaler.update()
    
    def optimize_model_for_precision(self, model: nn.Module) -> nn.Module:
        """Optimize model for the selected precision mode."""
        if self.config.mode == PrecisionMode.FP8 and HAS_TRANSFORMER_ENGINE:
            # Replace linear layers with FP8 versions
            model = self._replace_linear_with_fp8(model)
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        
        return model
    
    def _replace_linear_with_fp8(self, model: nn.Module) -> nn.Module:
        """Replace standard linear layers with FP8 versions."""
        if not HAS_TRANSFORMER_ENGINE:
            return model
            
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Replace with Transformer Engine Linear
                fp8_linear = te.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                # Copy weights
                fp8_linear.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    fp8_linear.bias.data.copy_(module.bias.data)
                setattr(model, name, fp8_linear)
            else:
                self._replace_linear_with_fp8(module)
        
        return model
    
    def get_precision_stats(self) -> Dict[str, Any]:
        """Get comprehensive precision statistics."""
        stats = {
            'mode': self.config.mode.value,
            'enabled': self.config.enabled,
        }
        
        if self.scaler is not None:
            stats.update({
                'scale': self.scaler.get_scale(),
                'overflow_count': self.scaler.overflow_count,
                'total_steps': self.scaler.total_steps,
                'overflow_rate': self.scaler.overflow_count / max(1, self.scaler.total_steps),
            })
        
        return stats
    
    def should_skip_step(self) -> bool:
        """Check if optimizer step should be skipped due to overflow."""
        if self.scaler is not None:
            # Check if gradients are finite
            return not self.scaler._check_inf_per_device(self.scaler._get_inf_per_device())
        return False


def create_mixed_precision_manager(
    mode: Union[str, PrecisionMode] = "fp16",
    **kwargs
) -> MixedPrecisionManager:
    """Factory function to create mixed precision manager."""
    if isinstance(mode, str):
        mode = PrecisionMode(mode.lower())
    
    config = MixedPrecisionConfig(mode=mode, **kwargs)
    return MixedPrecisionManager(config)


# Decorator for automatic mixed precision
def mixed_precision_forward(precision_manager: MixedPrecisionManager):
    """Decorator to automatically apply mixed precision to forward pass."""
    def decorator(forward_func):
        def wrapper(*args, **kwargs):
            with precision_manager.get_autocast_context():
                with precision_manager.get_fp8_context():
                    return forward_func(*args, **kwargs)
        return wrapper
    return decorator


# Context manager for training step
class MixedPrecisionTrainingStep:
    """Context manager for a complete mixed precision training step."""
    
    def __init__(self, precision_manager: MixedPrecisionManager, optimizer: torch.optim.Optimizer):
        self.precision_manager = precision_manager
        self.optimizer = optimizer
        self.should_step = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.should_step and not self.precision_manager.should_skip_step():
            self.precision_manager.step_optimizer(self.optimizer)
        self.precision_manager.update_scaler()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss within the training step."""
        return self.precision_manager.scale_loss(loss)
    
    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass with proper scaling."""
        scaled_loss = self.scale_loss(loss)
        scaled_loss.backward()
    
    def skip_step(self) -> None:
        """Mark this step to be skipped."""
        self.should_step = False
