"""
Performance monitoring utilities for Hyena-GLT framework.

This module provides tools for profiling, benchmarking, and monitoring
the performance of Hyena-GLT models and operations.
"""

import time
import psutil
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProfilerContext:
    """Context manager for profiling code execution with detailed metrics."""
    
    def __init__(self, name: str = "operation", enable_gpu: bool = True):
        self.name = name
        self.enable_gpu = enable_gpu and TORCH_AVAILABLE
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.start_gpu_memory = None
        self.end_gpu_memory = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if self.enable_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
        logger.info(f"Started profiling: {self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
        self.end_time = time.perf_counter()
        self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = self.end_time - self.start_time
        memory_delta = self.end_memory - self.start_memory
        
        result_msg = f"Completed profiling: {self.name}\n"
        result_msg += f"  Duration: {duration:.4f}s\n"
        result_msg += f"  Memory delta: {memory_delta:+.2f}MB\n"
        
        if self.enable_gpu and self.start_gpu_memory is not None:
            gpu_memory_delta = self.end_gpu_memory - self.start_gpu_memory
            result_msg += f"  GPU memory delta: {gpu_memory_delta:+.2f}MB"
            
        logger.info(result_msg)
        
    def get_metrics(self) -> Dict[str, float]:
        """Get profiling metrics as a dictionary."""
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Profiler context not completed")
            
        metrics = {
            'duration_seconds': self.end_time - self.start_time,
            'memory_delta_mb': self.end_memory - self.start_memory,
            'peak_memory_mb': self.end_memory
        }
        
        if self.enable_gpu and self.start_gpu_memory is not None:
            metrics.update({
                'gpu_memory_delta_mb': self.end_gpu_memory - self.start_gpu_memory,
                'peak_gpu_memory_mb': self.end_gpu_memory
            })
            
        return metrics


def memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary containing memory usage metrics in MB.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    stats = {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }
    
    return stats


def gpu_memory_usage() -> Optional[Dict[str, float]]:
    """Get current GPU memory usage statistics.
    
    Returns:
        Dictionary containing GPU memory metrics in MB, or None if CUDA unavailable.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
        
    stats = {}
    for i in range(torch.cuda.device_count()):
        device_stats = {
            f'device_{i}_allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
            f'device_{i}_cached_mb': torch.cuda.memory_reserved(i) / 1024 / 1024,
            f'device_{i}_max_allocated_mb': torch.cuda.max_memory_allocated(i) / 1024 / 1024
        }
        stats.update(device_stats)
        
    return stats


def benchmark_model(
    model_fn: Callable,
    input_data: Any,
    num_runs: int = 10,
    warmup_runs: int = 2
) -> Dict[str, float]:
    """Benchmark a model function with multiple runs.
    
    Args:
        model_fn: Function to benchmark (should be callable with input_data)
        input_data: Input data to pass to the model function
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (not included in timing)
        
    Returns:
        Dictionary with benchmark statistics.
    """
    logger.info(f"Benchmarking model with {warmup_runs} warmup + {num_runs} runs")
    
    # Warmup runs
    for _ in range(warmup_runs):
        with ProfilerContext("warmup", enable_gpu=TORCH_AVAILABLE):
            _ = model_fn(input_data)
            
    # Benchmark runs
    run_times = []
    memory_deltas = []
    
    for run_idx in range(num_runs):
        with ProfilerContext(f"benchmark_run_{run_idx}", enable_gpu=TORCH_AVAILABLE) as profiler:
            result = model_fn(input_data)
            
        metrics = profiler.get_metrics()
        run_times.append(metrics['duration_seconds'])
        memory_deltas.append(metrics['memory_delta_mb'])
        
    # Calculate statistics
    avg_time = sum(run_times) / len(run_times)
    min_time = min(run_times)
    max_time = max(run_times)
    std_time = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
    
    stats = {
        'avg_time_seconds': avg_time,
        'min_time_seconds': min_time,
        'max_time_seconds': max_time,
        'std_time_seconds': std_time,
        'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
        'num_runs': num_runs
    }
    
    logger.info(f"Benchmark completed: avg={avg_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s")
    return stats


def measure_throughput(
    model_fn: Callable,
    input_data: Any,
    duration_seconds: float = 10.0,
    batch_size: Optional[int] = None
) -> Dict[str, float]:
    """Measure model throughput over a specified duration.
    
    Args:
        model_fn: Function to measure throughput for
        input_data: Input data to pass to the model function
        duration_seconds: How long to run the throughput test
        batch_size: Batch size (if applicable) for throughput calculation
        
    Returns:
        Dictionary with throughput statistics.
    """
    logger.info(f"Measuring throughput for {duration_seconds} seconds")
    
    start_time = time.perf_counter()
    end_time = start_time + duration_seconds
    
    iterations = 0
    total_items = 0
    
    while time.perf_counter() < end_time:
        result = model_fn(input_data)
        iterations += 1
        
        if batch_size is not None:
            total_items += batch_size
        else:
            total_items += 1
            
    actual_duration = time.perf_counter() - start_time
    
    stats = {
        'iterations_per_second': iterations / actual_duration,
        'items_per_second': total_items / actual_duration,
        'total_iterations': iterations,
        'total_items': total_items,
        'actual_duration_seconds': actual_duration
    }
    
    logger.info(f"Throughput: {stats['iterations_per_second']:.2f} iter/s, {stats['items_per_second']:.2f} items/s")
    return stats


@contextmanager
def monitor_resources(interval_seconds: float = 1.0):
    """Context manager for continuous resource monitoring.
    
    Args:
        interval_seconds: Sampling interval for resource monitoring
        
    Yields:
        List that will be populated with resource snapshots
    """
    snapshots = []
    monitoring = True
    
    def monitor_loop():
        while monitoring:
            snapshot = {
                'timestamp': time.time(),
                'memory': memory_usage(),
                'gpu_memory': gpu_memory_usage()
            }
            snapshots.append(snapshot)
            time.sleep(interval_seconds)
            
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    
    try:
        yield snapshots
    finally:
        monitoring = False
        monitor_thread.join(timeout=1.0)
