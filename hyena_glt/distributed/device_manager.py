"""
Device Manager for cluster-ready GPU handling.
Based on patterns from BLT, Savanna, and Vortex repositories.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages GPU devices for distributed training on clusters."""
    
    def __init__(self, local_rank: Optional[int] = None, world_size: Optional[int] = None):
        """
        Initialize device manager.
        
        Args:
            local_rank: Local rank for current process
            world_size: Total number of processes
        """
        self.local_rank = local_rank or int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        
        self._device = None
        self._is_distributed = self.world_size > 1
        self._device_count = 0
        
        self._setup_device()
    
    def _setup_device(self):
        """Setup CUDA device for current process."""
        if torch.cuda.is_available():
            self._device_count = torch.cuda.device_count()
            
            if self._is_distributed:
                # Set device for distributed training
                if self.local_rank >= self._device_count:
                    raise ValueError(
                        f"Local rank {self.local_rank} >= available devices {self._device_count}"
                    )
                torch.cuda.set_device(self.local_rank)
                self._device = torch.device(f'cuda:{self.local_rank}')
            else:
                # Single GPU training
                self._device = torch.device('cuda')
                torch.cuda.set_device(0)
        else:
            self._device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
    
    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self._device
    
    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self._is_distributed
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    @property
    def device_count(self) -> int:
        """Get number of available CUDA devices."""
        return self._device_count
    
    def setup_distributed(self, backend: str = 'nccl'):
        """
        Initialize distributed training.
        
        Args:
            backend: Distributed backend (nccl for GPU, gloo for CPU)
        """
        if not self._is_distributed:
            logger.info("Single process training, skipping distributed setup")
            return
        
        if not dist.is_initialized():
            # Use NCCL for GPU, fallback to gloo for CPU
            if self._device.type == 'cuda' and backend == 'nccl':
                backend = 'nccl'
            else:
                backend = 'gloo'
            
            dist.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size
            )
            
            logger.info(
                f"Initialized distributed training: rank={self.rank}, "
                f"world_size={self.world_size}, backend={backend}"
            )
    
    def cleanup_distributed(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Cleaned up distributed training")
    
    def move_to_device(self, obj: Any) -> Any:
        """Move tensor or model to the appropriate device."""
        if hasattr(obj, 'to'):
            return obj.to(self._device)
        return obj
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if self._device.type != 'cuda':
            return {'device': 'cpu', 'memory': 'N/A'}
        
        allocated = torch.cuda.memory_allocated(self._device)
        reserved = torch.cuda.memory_reserved(self._device)
        max_allocated = torch.cuda.max_memory_allocated(self._device)
        
        return {
            'device': str(self._device),
            'allocated_gb': allocated / 1e9,
            'reserved_gb': reserved / 1e9,
            'max_allocated_gb': max_allocated / 1e9,
        }
    
    def clear_memory(self):
        """Clear GPU memory cache."""
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
            if self.is_main_process:
                logger.info("Cleared CUDA memory cache")
    
    def synchronize(self):
        """Synchronize all processes."""
        if self._is_distributed and dist.is_initialized():
            dist.barrier()
        elif self._device.type == 'cuda':
            torch.cuda.synchronize()


class GPUClusterConfig:
    """Configuration for GPU cluster deployment."""
    
    def __init__(
        self,
        nodes: int = 1,
        gpus_per_node: int = None,
        backend: str = 'nccl',
        mixed_precision: bool = True,
        gradient_checkpointing: bool = False,
        find_unused_parameters: bool = False,
    ):
        """
        Initialize cluster configuration.
        
        Args:
            nodes: Number of nodes
            gpus_per_node: GPUs per node (auto-detect if None)
            backend: Distributed backend
            mixed_precision: Enable mixed precision training
            gradient_checkpointing: Enable gradient checkpointing
            find_unused_parameters: DDP parameter for unused gradients
        """
        self.nodes = nodes
        self.gpus_per_node = gpus_per_node or torch.cuda.device_count()
        self.backend = backend
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.find_unused_parameters = find_unused_parameters
        
        self.world_size = nodes * self.gpus_per_node
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'nodes': self.nodes,
            'gpus_per_node': self.gpus_per_node,
            'backend': self.backend,
            'mixed_precision': self.mixed_precision,
            'gradient_checkpointing': self.gradient_checkpointing,
            'find_unused_parameters': self.find_unused_parameters,
            'world_size': self.world_size,
        }
    
    @classmethod
    def from_env(cls) -> 'GPUClusterConfig':
        """Create config from environment variables."""
        return cls(
            nodes=int(os.environ.get('NNODES', 1)),
            gpus_per_node=int(os.environ.get('NPROC_PER_NODE', torch.cuda.device_count())),
            backend=os.environ.get('DISTRIBUTED_BACKEND', 'nccl'),
            mixed_precision=os.environ.get('MIXED_PRECISION', 'true').lower() == 'true',
            gradient_checkpointing=os.environ.get('GRADIENT_CHECKPOINTING', 'false').lower() == 'true',
        )
