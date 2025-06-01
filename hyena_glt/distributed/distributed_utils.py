"""
Distributed training utilities for cluster deployment.
Based on patterns from BLT and Savanna repositories.
"""

import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from .device_manager import DeviceManager, GPUClusterConfig

logger = logging.getLogger(__name__)


def init_distributed_training(
    backend: str = "nccl", timeout_minutes: int = 30
) -> DeviceManager:
    """
    Initialize distributed training environment.

    Args:
        backend: Distributed backend
        timeout_minutes: Timeout for process group initialization

    Returns:
        DeviceManager instance
    """
    device_manager = DeviceManager()

    if device_manager.is_distributed:
        # Set timeout for large clusters
        if timeout_minutes > 0:
            timedelta(minutes=timeout_minutes)

        device_manager.setup_distributed(backend=backend)

        if device_manager.is_main_process:
            logger.info(
                f"Distributed training initialized: "
                f"world_size={device_manager.world_size}, "
                f"local_rank={device_manager.local_rank}"
            )

    return device_manager


def wrap_model_for_distributed(
    model: torch.nn.Module,
    device_manager: DeviceManager,
    config: GPUClusterConfig,
    use_fsdp: bool = False,
    transformer_layer_cls: type | None = None,
) -> torch.nn.Module:
    """
    Wrap model for distributed training.

    Args:
        model: PyTorch model
        device_manager: Device manager instance
        config: Cluster configuration
        use_fsdp: Use FSDP instead of DDP
        transformer_layer_cls: Transformer layer class for FSDP wrapping

    Returns:
        Wrapped model
    """
    # Move model to device first
    model = device_manager.move_to_device(model)

    if not device_manager.is_distributed:
        return model

    if use_fsdp:
        # Use FSDP for large models
        auto_wrap_policy: Callable[..., Any] | None
        if transformer_layer_cls:
            auto_wrap_policy = transformer_auto_wrap_policy(  # type: ignore[call-arg, assignment]
                transformer_layer_cls={transformer_layer_cls}
            )
        else:
            auto_wrap_policy = None

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=None,  # Handle separately with GradScaler
            device_id=device_manager.local_rank,
        )

        if device_manager.is_main_process:
            logger.info("Model wrapped with FSDP")

    else:
        # Use DDP for standard distributed training
        model = DDP(
            model,
            device_ids=[device_manager.local_rank],
            output_device=device_manager.local_rank,
            find_unused_parameters=config.find_unused_parameters,
        )

        if device_manager.is_main_process:
            logger.info("Model wrapped with DDP")

    return model


def setup_mixed_precision(
    device_manager: DeviceManager, config: GPUClusterConfig
) -> tuple[torch.cuda.amp.GradScaler | None, Any]:
    """
    Setup advanced mixed precision training with multiple precision modes.

    Args:
        device_manager: Device manager instance
        config: Cluster configuration

    Returns:
        Tuple of (GradScaler if using mixed precision, MixedPrecisionManager if available)
    """
    if not config.mixed_precision or device_manager.device.type != "cuda":
        return None, None

    # Try to use advanced mixed precision if available
    try:
        from ..training.mixed_precision import (
            MixedPrecisionConfig,
            MixedPrecisionManager,
            PrecisionMode
        )
        
        # Determine precision mode from config
        precision_mode = PrecisionMode.FP16  # Default
        if hasattr(config, 'precision_mode'):
            try:
                precision_mode = PrecisionMode(config.precision_mode.lower())
            except (ValueError, AttributeError):
                pass
        
        # Create advanced mixed precision config
        mp_config = MixedPrecisionConfig(
            mode=precision_mode,
            dynamic_loss_scale=getattr(config, 'dynamic_loss_scale', True),
            gradient_clipping=getattr(config, 'gradient_clipping', 1.0),
            kernel_precision=getattr(config, 'kernel_precision', 'ieee'),
            monitor_overflow=getattr(config, 'precision_monitoring', True),
        )
        
        precision_manager = MixedPrecisionManager(mp_config)
        
        if device_manager.is_main_process:
            logger.info(f"Advanced mixed precision training enabled: {precision_mode.value}")
            
        # Return both for backward compatibility
        return precision_manager.scaler, precision_manager
        
    except ImportError:
        # Fallback to basic mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        if device_manager.is_main_process:
            logger.info("Basic mixed precision training enabled")
            
        return scaler, None


def all_reduce_tensor(
    tensor: torch.Tensor, device_manager: DeviceManager
) -> torch.Tensor:
    """
    All-reduce a tensor across all processes.

    Args:
        tensor: Input tensor
        device_manager: Device manager instance

    Returns:
        Reduced tensor
    """
    if not device_manager.is_distributed:
        return tensor

    # Clone to avoid in-place operations
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / device_manager.world_size

    return tensor


def gather_tensor(
    tensor: torch.Tensor, device_manager: DeviceManager
) -> torch.Tensor | None:
    """
    Gather tensor from all processes to rank 0.

    Args:
        tensor: Input tensor
        device_manager: Device manager instance

    Returns:
        Gathered tensor on rank 0, None on other ranks
    """
    if not device_manager.is_distributed:
        return tensor

    if device_manager.is_main_process:
        gather_list = [
            torch.zeros_like(tensor) for _ in range(device_manager.world_size)
        ]
        dist.gather(tensor, gather_list, dst=0)
        return torch.cat(gather_list, dim=0)
    else:
        dist.gather(tensor, dst=0)
        return None


def broadcast_object(obj: Any, src: int, device_manager: DeviceManager) -> Any:
    """
    Broadcast object from source rank to all ranks.

    Args:
        obj: Object to broadcast
        src: Source rank
        device_manager: Device manager instance

    Returns:
        Broadcasted object
    """
    if not device_manager.is_distributed:
        return obj

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def save_checkpoint_distributed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    device_manager: DeviceManager,
    scaler: torch.cuda.amp.GradScaler | None = None,
    precision_manager: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save checkpoint in distributed training with mixed precision support.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        device_manager: Device manager instance
        scaler: Gradient scaler for mixed precision
        precision_manager: Advanced precision manager if available
        metadata: Additional metadata
    """
    if not device_manager.is_main_process:
        return

    # Extract model state dict (handle DDP/FSDP wrapping)
    if hasattr(model, "module"):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "world_size": device_manager.world_size,
    }

    # Save scaler state if available
    if scaler:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Save advanced precision manager state if available
    if precision_manager:
        checkpoint["precision_config"] = precision_manager.config.__dict__
        checkpoint["precision_stats"] = precision_manager.get_precision_stats()

    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint_distributed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device_manager: DeviceManager,
    scaler: torch.cuda.amp.GradScaler | None = None,
    precision_manager: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load checkpoint in distributed training with mixed precision support.

    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        filepath: Path to checkpoint
        device_manager: Device manager instance
        scaler: Gradient scaler for mixed precision
        precision_manager: Advanced precision manager if available
        strict: Whether to strictly enforce state dict keys

    Returns:
        Checkpoint metadata
    """
    map_location = device_manager.device
    checkpoint = torch.load(filepath, map_location=map_location)

    # Load model state dict (handle DDP/FSDP wrapping)
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scaler state if available
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Load precision manager state if available
    if precision_manager and "precision_config" in checkpoint:
        # Note: precision_manager state loading would need custom implementation
        # depending on the specific precision manager design
        pass

    if device_manager.is_main_process:
        logger.info(f"Checkpoint loaded from {filepath}")
        
        # Log precision information if available
        if "precision_stats" in checkpoint:
            stats = checkpoint["precision_stats"]
            logger.info(f"Loaded precision stats - Mode: {stats.get('mode', 'unknown')}")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "metadata": checkpoint.get("metadata", {}),
        "precision_stats": checkpoint.get("precision_stats", {}),
    }
