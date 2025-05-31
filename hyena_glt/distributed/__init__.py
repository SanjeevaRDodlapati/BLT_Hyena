"""
Distributed training module for cluster-ready GPU handling.

This module provides enterprise-level GPU resource management and distributed
training capabilities based on patterns from BLT, Savanna, and Vortex repositories.

Key Components:
- DeviceManager: Advanced GPU device management and cluster setup
- GPUClusterConfig: Configuration for multi-node GPU clusters
- Distributed utilities: Model wrapping, checkpointing, communication
- Model parallelization: Support for large model distributed training

Usage:
    from hyena_glt.distributed import DeviceManager, GPUClusterConfig
    from hyena_glt.distributed import init_distributed_training
    from hyena_glt.distributed import create_distributed_model
"""

from .device_manager import DeviceManager, GPUClusterConfig
from .distributed_utils import (
    init_distributed_training,
    wrap_model_for_distributed,
    setup_mixed_precision,
    all_reduce_tensor,
    gather_tensor,
    broadcast_object,
    save_checkpoint_distributed,
    load_checkpoint_distributed,
)
from .parallel_model import (
    DistributedHyenaGLT,
    ModelParallelHyenaGLT,
    create_distributed_model,
)

__version__ = "1.0.0"
__all__ = [
    # Core components
    "DeviceManager",
    "GPUClusterConfig",
    
    # Distributed utilities
    "init_distributed_training",
    "wrap_model_for_distributed",
    "setup_mixed_precision",
    "all_reduce_tensor",
    "gather_tensor",
    "broadcast_object",
    "save_checkpoint_distributed",
    "load_checkpoint_distributed",
    
    # Model parallelization
    "DistributedHyenaGLT",
    "ModelParallelHyenaGLT", 
    "create_distributed_model",
]
