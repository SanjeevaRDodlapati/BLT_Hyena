"""
Model parallelization support for HyenaGLT on GPU clusters.
Based on patterns from BLT and Savanna repositories.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from .device_manager import DeviceManager, GPUClusterConfig
from .distributed_utils import wrap_model_for_distributed

logger = logging.getLogger(__name__)


class DistributedHyenaGLT(nn.Module):
    """
    Distributed wrapper for HyenaGLT model with cluster-ready GPU handling.
    """

    def __init__(
        self,
        model: nn.Module,
        device_manager: DeviceManager,
        config: GPUClusterConfig,
        use_fsdp: bool = False,
        transformer_layer_cls: type | None = None,
    ):
        """
        Initialize distributed HyenaGLT wrapper.

        Args:
            model: Base HyenaGLT model
            device_manager: Device manager instance
            config: Cluster configuration
            use_fsdp: Use FSDP instead of DDP
            transformer_layer_cls: Transformer layer class for FSDP
        """
        super().__init__()

        self.device_manager = device_manager
        self.config = config
        self.use_fsdp = use_fsdp

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing and hasattr(
            model, "gradient_checkpointing_enable"
        ):
            model.gradient_checkpointing_enable()
            if device_manager.is_main_process:
                logger.info("Gradient checkpointing enabled")

        # Wrap model for distributed training
        self.model = wrap_model_for_distributed(
            model=model,
            device_manager=device_manager,
            config=config,
            use_fsdp=use_fsdp,
            transformer_layer_cls=transformer_layer_cls,
        )

        # Store original model for access to attributes
        self._original_model = model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the distributed model."""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Try to get from wrapped model or original model
            if hasattr(self.model, name):
                return getattr(self.model, name)
            elif hasattr(self._original_model, name):
                return getattr(self._original_model, name)
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                ) from None

    def get_model_size(self) -> dict[str, Any]:
        """Get model size information."""
        if hasattr(self._original_model, "num_parameters"):
            total_params = self._original_model.num_parameters()
        else:
            total_params = sum(p.numel() for p in self._original_model.parameters())

        trainable_params = sum(
            p.numel() for p in self._original_model.parameters() if p.requires_grad
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1e6,  # Assuming float32
        }

    def sync_parameters(self) -> None:
        """Synchronize parameters across all processes."""
        if self.device_manager.is_distributed:
            for param in self.model.parameters():
                torch.distributed.all_reduce(
                    param.data, op=torch.distributed.ReduceOp.SUM
                )
                param.data /= self.device_manager.world_size


class ModelParallelHyenaGLT(nn.Module):
    """
    Model parallel version of HyenaGLT for very large models.
    Splits model across multiple GPUs on a single node.
    """

    def __init__(
        self,
        model: nn.Module,
        device_manager: DeviceManager,
        split_layers: list[str] | None = None,
    ):
        """
        Initialize model parallel HyenaGLT.

        Args:
            model: Base HyenaGLT model
            device_manager: Device manager instance
            split_layers: List of layer names to split across GPUs
        """
        super().__init__()

        self.device_manager = device_manager
        self.split_layers = split_layers or []

        if device_manager.device_count < 2:
            logger.warning(
                "Model parallelism requires multiple GPUs, falling back to single GPU"
            )
            self.model = device_manager.move_to_device(model)
            self.is_parallel = False
        else:
            self.is_parallel = True
            self._setup_model_parallel(model)

    def _setup_model_parallel(self, model: nn.Module) -> None:
        """Setup model parallelism across available GPUs."""
        # Simple strategy: split sequential layers across GPUs
        layers = []
        for name, module in model.named_children():
            layers.append((name, module))

        if not layers:
            # Model has no child modules, treat as single block
            self.model = self.device_manager.move_to_device(model)
            self.is_parallel = False
            return

        # Distribute layers across available GPUs
        gpus_to_use = min(self.device_manager.device_count, len(layers))
        layers_per_gpu = len(layers) // gpus_to_use

        self.gpu_modules = nn.ModuleDict()

        for gpu_idx in range(gpus_to_use):
            start_idx = gpu_idx * layers_per_gpu
            if gpu_idx == gpus_to_use - 1:
                # Last GPU gets remaining layers
                end_idx = len(layers)
            else:
                end_idx = (gpu_idx + 1) * layers_per_gpu

            gpu_layers = nn.ModuleDict()
            for i in range(start_idx, end_idx):
                name, module = layers[i]
                gpu_layers[name] = module

            # Move to specific GPU
            device = torch.device(f"cuda:{gpu_idx}")
            gpu_layers = gpu_layers.to(device)
            self.gpu_modules[f"gpu_{gpu_idx}"] = gpu_layers

        if self.device_manager.is_main_process:
            logger.info(f"Model split across {gpus_to_use} GPUs")

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through model parallel layers."""
        if not self.is_parallel:
            return self.model(x, *args, **kwargs)  # type: ignore[no-any-return]

        # Pass input through each GPU sequentially
        current_device = x.device

        for _gpu_name, gpu_module in self.gpu_modules.items():
            # Move input to current GPU
            gpu_device = next(gpu_module.parameters()).device
            x = x.to(gpu_device)

            # Forward through layers on this GPU
            for _layer_name, layer in gpu_module.items():
                x = layer(x)

        # Move final output back to original device if needed
        if x.device != current_device:
            x = x.to(current_device)

        return x


def create_distributed_model(
    model_cls: type,
    model_config: dict[str, Any],
    device_manager: DeviceManager,
    cluster_config: GPUClusterConfig,
    use_fsdp: bool = False,
    use_model_parallel: bool = False,
    transformer_layer_cls: type | None = None,
) -> nn.Module:
    """
    Create a distributed model with appropriate parallelization strategy.

    Args:
        model_cls: Model class to instantiate
        model_config: Model configuration dictionary
        device_manager: Device manager instance
        cluster_config: Cluster configuration
        use_fsdp: Use FSDP for data parallelism
        use_model_parallel: Use model parallelism within nodes
        transformer_layer_cls: Transformer layer class for FSDP

    Returns:
        Distributed model instance
    """
    # Create base model
    model = model_cls(**model_config)

    if device_manager.is_main_process:
        logger.info(f"Created {model_cls.__name__} with config: {model_config}")

    # Choose parallelization strategy
    if use_model_parallel and device_manager.device_count > 1:
        # Use model parallelism within nodes
        model = ModelParallelHyenaGLT(
            model=model,
            device_manager=device_manager,
        )
        if device_manager.is_main_process:
            logger.info("Using model parallelism")

    if device_manager.is_distributed:
        # Wrap with distributed training
        model = DistributedHyenaGLT(
            model=model,
            device_manager=device_manager,
            config=cluster_config,
            use_fsdp=use_fsdp,
            transformer_layer_cls=transformer_layer_cls,
        )
        if device_manager.is_main_process:
            parallelism_type = "FSDP" if use_fsdp else "DDP"
            logger.info(f"Using distributed data parallelism: {parallelism_type}")

    else:
        # Single process training
        model = device_manager.move_to_device(model)

    return model  # type: ignore[no-any-return]
