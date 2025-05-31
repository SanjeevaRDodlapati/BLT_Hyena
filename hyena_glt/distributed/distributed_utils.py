"""
Distributed training utilities for cluster deployment.
Based on patterns from BLT and Savanna repositories.
"""

import logging
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
            torch.timedelta(minutes=timeout_minutes)

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

    if use_fsdp and transformer_layer_cls:
        # Use FSDP for large models
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls=transformer_layer_cls
        )

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
) -> torch.cuda.amp.GradScaler | None:
    """
    Setup mixed precision training.

    Args:
        device_manager: Device manager instance
        config: Cluster configuration

    Returns:
        GradScaler if using mixed precision, None otherwise
    """
    if not config.mixed_precision or device_manager.device.type != "cuda":
        return None

    scaler = torch.cuda.amp.GradScaler()

    if device_manager.is_main_process:
        logger.info("Mixed precision training enabled")

    return scaler


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
    metadata: dict[str, Any] | None = None,
):
    """
    Save checkpoint in distributed training.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        device_manager: Device manager instance
        scaler: Gradient scaler for mixed precision
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

    if scaler:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

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
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load checkpoint in distributed training.

    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        filepath: Path to checkpoint
        device_manager: Device manager instance
        scaler: Gradient scaler for mixed precision
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

    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if device_manager.is_main_process:
        logger.info(f"Checkpoint loaded from {filepath}")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "metadata": checkpoint.get("metadata", {}),
    }
