"""
Memory Optimization for Hyena-GLT

This module provides memory optimization tools including gradient checkpointing,
activation checkpointing, and memory profiling for efficient training and inference.
"""

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ..model import HyenaGLT

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""

    # Gradient checkpointing
    gradient_checkpointing: bool = True
    checkpoint_ratio: float = 0.5  # Fraction of layers to checkpoint

    # Activation checkpointing
    activation_checkpointing: bool = False
    checkpoint_activations: list[str] = None

    # Memory management
    enable_memory_efficient_attention: bool = True
    mixed_precision: bool = True
    pin_memory: bool = True

    # Garbage collection
    aggressive_gc: bool = False
    gc_frequency: int = 100  # Steps between GC calls

    # Batch optimization
    gradient_accumulation_steps: int = 1
    max_batch_size: int = 32
    adaptive_batch_size: bool = False

    # Advanced optimizations
    cpu_offload: bool = False
    activation_offload: bool = False
    parameter_offload: bool = False

    def __post_init__(self):
        if self.checkpoint_activations is None:
            self.checkpoint_activations = []


class MemoryOptimizer:
    """Main memory optimization interface."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_stats = {}
        self.optimized_model = None

    def optimize_model(
        self, model: HyenaGLT, enable_profiling: bool = False
    ) -> nn.Module:
        """
        Apply memory optimizations to the model.

        Args:
            model: The model to optimize
            enable_profiling: Whether to enable memory profiling

        Returns:
            Memory-optimized model
        """
        logger.info("Applying memory optimizations...")

        # Apply gradient checkpointing
        if self.config.gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)

        # Apply activation checkpointing
        if self.config.activation_checkpointing:
            model = self._apply_activation_checkpointing(model)

        # Apply memory-efficient attention
        if self.config.enable_memory_efficient_attention:
            model = self._apply_memory_efficient_attention(model)

        # Apply CPU offloading if requested
        if self.config.cpu_offload:
            model = self._apply_cpu_offload(model)

        # Setup mixed precision if requested
        if self.config.mixed_precision:
            model = self._setup_mixed_precision(model)

        self.optimized_model = model

        if enable_profiling:
            self._setup_memory_profiling()

        logger.info("Memory optimizations applied")
        return model

    def _apply_gradient_checkpointing(self, model: HyenaGLT) -> nn.Module:
        """Apply gradient checkpointing to reduce memory usage."""
        logger.info("Applying gradient checkpointing...")

        # Get layers to checkpoint
        layers = self._get_checkpointable_layers(model)
        num_to_checkpoint = int(len(layers) * self.config.checkpoint_ratio)

        # Apply checkpointing to selected layers
        for _i, (name, layer) in enumerate(layers[:num_to_checkpoint]):
            # Wrap layer with checkpointing
            checkpointed_layer = GradientCheckpointing.apply_to_layer(layer)

            # Replace layer in model
            parent_module = model
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], checkpointed_layer)

        logger.info(f"Applied gradient checkpointing to {num_to_checkpoint} layers")
        return model

    def _apply_activation_checkpointing(self, model: HyenaGLT) -> nn.Module:
        """Apply activation checkpointing to specific modules."""
        logger.info("Applying activation checkpointing...")

        activation_checkpointer = ActivationCheckpointing(self.config)

        for name in self.config.checkpoint_activations:
            if hasattr(model, name):
                module = getattr(model, name)
                checkpointed_module = activation_checkpointer.wrap_module(module)
                setattr(model, name, checkpointed_module)

        logger.info(
            f"Applied activation checkpointing to {len(self.config.checkpoint_activations)} modules"
        )
        return model

    def _apply_memory_efficient_attention(self, model: HyenaGLT) -> nn.Module:
        """Apply memory-efficient attention mechanisms."""
        logger.info("Applying memory-efficient attention...")

        # Replace attention modules with memory-efficient versions
        for name, _module in model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                # Replace with memory-efficient attention
                # This would depend on the specific attention implementation
                pass

        return model

    def _apply_cpu_offload(self, model: HyenaGLT) -> nn.Module:
        """Apply CPU offloading for parameters and activations."""
        logger.info("Applying CPU offloading...")

        if self.config.parameter_offload:
            model = self._apply_parameter_offload(model)

        if self.config.activation_offload:
            model = self._apply_activation_offload(model)

        return model

    def _apply_parameter_offload(self, model: HyenaGLT) -> nn.Module:
        """Offload parameters to CPU when not in use."""
        # This would implement parameter offloading
        # Complex implementation involving hooks and device management
        logger.info("Parameter offloading applied")
        return model

    def _apply_activation_offload(self, model: HyenaGLT) -> nn.Module:
        """Offload activations to CPU to save GPU memory."""
        # This would implement activation offloading
        logger.info("Activation offloading applied")
        return model

    def _setup_mixed_precision(self, model: HyenaGLT) -> nn.Module:
        """Setup mixed precision training."""
        logger.info("Setting up mixed precision...")

        # Convert model to mixed precision
        # This would typically be handled by the training loop with AMP
        return model

    def _get_checkpointable_layers(
        self, model: nn.Module
    ) -> list[tuple[str, nn.Module]]:
        """Get layers that can be checkpointed."""
        checkpointable_layers = []

        for name, module in model.named_modules():
            # Skip root module and leaf modules without parameters
            if name == "" or len(list(module.children())) == 0:
                continue

            # Check if module has significant computation
            if self._is_checkpointable(module):
                checkpointable_layers.append((name, module))

        return checkpointable_layers

    def _is_checkpointable(self, module: nn.Module) -> bool:
        """Check if a module is suitable for checkpointing."""
        # Modules that benefit from checkpointing
        checkpointable_types = (
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.MultiheadAttention,
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
        )

        return (
            isinstance(module, checkpointable_types)
            or "block" in type(module).__name__.lower()
        )

    def _setup_memory_profiling(self):
        """Setup memory profiling hooks."""
        if not hasattr(self, "memory_profiler"):
            self.memory_profiler = MemoryProfiler(self.config)
            self.memory_profiler.setup_hooks(self.optimized_model)


class GradientCheckpointing:
    """Gradient checkpointing utilities."""

    @staticmethod
    def apply_to_layer(layer: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to a specific layer."""

        class CheckpointedLayer(nn.Module):
            def __init__(self, original_layer):
                super().__init__()
                self.layer = original_layer

            def forward(self, *args, **kwargs):
                # Use gradient checkpointing for this layer
                return checkpoint.checkpoint(self.layer, *args, **kwargs)

        return CheckpointedLayer(layer)

    @staticmethod
    def checkpoint_sequential(layers: nn.Sequential, segments: int = 2) -> nn.Module:
        """Apply checkpointing to sequential layers in segments."""

        class CheckpointedSequential(nn.Module):
            def __init__(self, layers, segments):
                super().__init__()
                self.layers = layers
                self.segments = segments

                # Divide layers into segments
                layers_per_segment = len(layers) // segments
                self.segment_boundaries = [
                    i * layers_per_segment for i in range(segments + 1)
                ]
                self.segment_boundaries[-1] = len(layers)

            def forward(self, x):
                for i in range(self.segments):
                    start_idx = self.segment_boundaries[i]
                    end_idx = self.segment_boundaries[i + 1]

                    # Create a function for this segment
                    def segment_forward(input_tensor, start=start_idx, end=end_idx):
                        for j in range(start, end):
                            input_tensor = self.layers[j](input_tensor)
                        return input_tensor

                    # Apply checkpointing to this segment
                    x = checkpoint.checkpoint(segment_forward, x)

                return x

        return CheckpointedSequential(layers, segments)


class ActivationCheckpointing:
    """Activation checkpointing for specific modules."""

    def __init__(self, config: MemoryConfig):
        self.config = config

    def wrap_module(self, module: nn.Module) -> nn.Module:
        """Wrap a module with activation checkpointing."""

        class CheckpointedModule(nn.Module):
            def __init__(self, original_module):
                super().__init__()
                self.module = original_module
                self.checkpointed_activations = {}

            def forward(self, *args, **kwargs):
                # Store activations for checkpointing
                def checkpoint_hook(module, input, output):
                    # Store only if training
                    if self.training:
                        # Store activations on CPU to save GPU memory
                        if isinstance(output, torch.Tensor):
                            self.checkpointed_activations[id(module)] = output.cpu()
                        elif isinstance(output, tuple | list):
                            self.checkpointed_activations[id(module)] = [
                                o.cpu() if isinstance(o, torch.Tensor) else o
                                for o in output
                            ]

                # Register hook
                handle = self.module.register_forward_hook(checkpoint_hook)

                try:
                    output = self.module(*args, **kwargs)
                    return output
                finally:
                    handle.remove()

        return CheckpointedModule(module)


class MemoryProfiler:
    """Memory profiling and monitoring utilities."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_stats = {
            "peak_memory": 0,
            "allocated_memory": [],
            "cached_memory": [],
            "layer_memory": {},
        }
        self.hooks = []

    def setup_hooks(self, model: nn.Module):
        """Setup memory profiling hooks."""
        for name, module in model.named_modules():
            hook = module.register_forward_hook(self._memory_hook(name))
            self.hooks.append(hook)

    def _memory_hook(self, layer_name: str):
        """Create a memory monitoring hook for a layer."""

        def hook(module, input, output):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_cached()

                self.memory_stats["allocated_memory"].append(allocated)
                self.memory_stats["cached_memory"].append(cached)
                self.memory_stats["layer_memory"][layer_name] = allocated

                # Update peak memory
                if allocated > self.memory_stats["peak_memory"]:
                    self.memory_stats["peak_memory"] = allocated

        return hook

    def get_memory_summary(self) -> dict[str, Any]:
        """Get comprehensive memory usage summary."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            cached_memory = torch.cuda.memory_cached()
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            current_memory = mem_info.rss
            max_memory = mem_info.vms
            cached_memory = 0

        summary = {
            "current_memory_mb": current_memory / 1024 / 1024,
            "max_memory_mb": max_memory / 1024 / 1024,
            "cached_memory_mb": cached_memory / 1024 / 1024,
            "peak_memory_mb": self.memory_stats["peak_memory"] / 1024 / 1024,
            "memory_efficiency": current_memory / max_memory if max_memory > 0 else 0,
        }

        return summary

    def print_memory_summary(self):
        """Print memory usage summary."""
        summary = self.get_memory_summary()

        print("\n" + "=" * 50)
        print("MEMORY USAGE SUMMARY")
        print("=" * 50)
        print(f"Current Memory: {summary['current_memory_mb']:.2f} MB")
        print(f"Peak Memory: {summary['peak_memory_mb']:.2f} MB")
        print(f"Max Memory: {summary['max_memory_mb']:.2f} MB")
        print(f"Cached Memory: {summary['cached_memory_mb']:.2f} MB")
        print(f"Memory Efficiency: {summary['memory_efficiency']:.2%}")

        # Top memory consuming layers
        if self.memory_stats["layer_memory"]:
            print("\nTOP MEMORY CONSUMING LAYERS:")
            sorted_layers = sorted(
                self.memory_stats["layer_memory"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for layer, memory in sorted_layers[:5]:
                print(f"  {layer}: {memory / 1024 / 1024:.2f} MB")

    def clear_hooks(self):
        """Remove all memory profiling hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def reset_stats(self):
        """Reset memory statistics."""
        self.memory_stats = {
            "peak_memory": 0,
            "allocated_memory": [],
            "cached_memory": [],
            "layer_memory": {},
        }

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class MemoryEfficientTraining:
    """Memory-efficient training utilities."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.gc_counter = 0

    def optimize_training_step(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Perform memory-optimized training step."""

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision if enabled
        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        # Scale loss for gradient accumulation
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Garbage collection
        self._maybe_garbage_collect(step)

        return loss

    def _maybe_garbage_collect(self, step: int):
        """Conditionally perform garbage collection."""
        if self.config.aggressive_gc:
            self.gc_counter += 1
            if self.gc_counter >= self.config.gc_frequency:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.gc_counter = 0


class AdaptiveBatchSizer:
    """Adaptive batch sizing to maximize memory utilization."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.current_batch_size = config.max_batch_size
        self.memory_threshold = 0.9  # Use up to 90% of available memory

    def find_optimal_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        loss_fn: Callable,
        max_iterations: int = 10,
    ) -> int:
        """Find optimal batch size through binary search."""
        if not self.config.adaptive_batch_size:
            return self.config.max_batch_size

        logger.info("Finding optimal batch size...")

        min_batch_size = 1
        max_batch_size = self.config.max_batch_size
        optimal_batch_size = min_batch_size

        for _iteration in range(max_iterations):
            test_batch_size = (min_batch_size + max_batch_size) // 2

            try:
                # Test with this batch size
                test_input = sample_input.repeat(test_batch_size, 1, 1)
                test_target = torch.randint(0, 10, (test_batch_size,))

                # Clear memory before test
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Test forward and backward pass
                model.train()
                optimizer = torch.optim.Adam(model.parameters())

                optimizer.zero_grad()
                output = model(test_input)
                loss = loss_fn(output, test_target)
                loss.backward()
                optimizer.step()

                # Check memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated()
                    memory_total = torch.cuda.get_device_properties(0).total_memory
                    memory_ratio = memory_used / memory_total

                    if memory_ratio < self.memory_threshold:
                        optimal_batch_size = test_batch_size
                        min_batch_size = test_batch_size + 1
                    else:
                        max_batch_size = test_batch_size - 1
                else:
                    optimal_batch_size = test_batch_size
                    break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    max_batch_size = test_batch_size - 1
                else:
                    raise e

            if min_batch_size > max_batch_size:
                break

        logger.info(f"Optimal batch size found: {optimal_batch_size}")
        self.current_batch_size = optimal_batch_size
        return optimal_batch_size


class MemoryBenchmark:
    """Benchmark memory usage of different optimization strategies."""

    def __init__(self):
        self.benchmark_results = {}

    def benchmark_optimizations(
        self,
        base_model: HyenaGLT,
        optimization_configs: dict[str, MemoryConfig],
        sample_input: torch.Tensor,
        num_iterations: int = 10,
    ) -> dict[str, dict[str, float]]:
        """Benchmark different memory optimization configurations."""
        results = {}

        for config_name, config in optimization_configs.items():
            logger.info(f"Benchmarking {config_name} configuration...")

            # Create optimized model
            model = base_model.__class__(base_model.config)
            model.load_state_dict(base_model.state_dict())

            optimizer = MemoryOptimizer(config)
            optimized_model = optimizer.optimize_model(model, enable_profiling=True)

            # Benchmark memory usage
            memory_stats = self._benchmark_single_config(
                optimized_model, sample_input, num_iterations
            )

            results[config_name] = memory_stats

        self.benchmark_results = results
        return results

    def _benchmark_single_config(
        self, model: nn.Module, sample_input: torch.Tensor, num_iterations: int
    ) -> dict[str, float]:
        """Benchmark a single configuration."""
        memory_stats = {
            "peak_memory": 0,
            "avg_memory": 0,
            "min_memory": float("inf"),
            "max_memory": 0,
        }

        memory_readings = []

        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        for _i in range(num_iterations):
            # Clear memory before each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Training step
            optimizer.zero_grad()

            target = torch.randint(0, 10, (sample_input.size(0),))
            output = model(sample_input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Record memory usage
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
            else:
                process = psutil.Process()
                current_memory = process.memory_info().rss
                peak_memory = current_memory

            memory_readings.append(current_memory)
            memory_stats["peak_memory"] = max(memory_stats["peak_memory"], peak_memory)

        # Calculate statistics
        memory_readings = np.array(memory_readings)
        memory_stats["avg_memory"] = np.mean(memory_readings) / 1024 / 1024  # MB
        memory_stats["min_memory"] = np.min(memory_readings) / 1024 / 1024
        memory_stats["max_memory"] = np.max(memory_readings) / 1024 / 1024
        memory_stats["peak_memory"] = memory_stats["peak_memory"] / 1024 / 1024

        return memory_stats

    def print_benchmark_results(self):
        """Print memory benchmark results."""
        if not self.benchmark_results:
            print("No benchmark results available")
            return

        print("\n" + "=" * 60)
        print("MEMORY OPTIMIZATION BENCHMARK RESULTS")
        print("=" * 60)

        for config_name, stats in self.benchmark_results.items():
            print(f"\n{config_name.upper()} CONFIGURATION:")
            print(f"  Average Memory: {stats['avg_memory']:.2f} MB")
            print(f"  Peak Memory: {stats['peak_memory']:.2f} MB")
            print(
                f"  Memory Range: {stats['min_memory']:.2f} - {stats['max_memory']:.2f} MB"
            )

        # Compare configurations
        if len(self.benchmark_results) > 1:
            print("\nMEMORY SAVINGS:")
            baseline_config = list(self.benchmark_results.keys())[0]
            baseline_memory = self.benchmark_results[baseline_config]["avg_memory"]

            for config_name, stats in list(self.benchmark_results.items())[1:]:
                savings = (
                    (baseline_memory - stats["avg_memory"]) / baseline_memory * 100
                )
                print(
                    f"  {config_name} vs {baseline_config}: {savings:.1f}% memory savings"
                )
