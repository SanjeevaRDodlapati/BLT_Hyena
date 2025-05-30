#!/usr/bin/env python3
"""
Performance Monitoring Demo for Hyena-GLT Framework

This script demonstrates the new performance monitoring utilities
added to the Hyena-GLT framework.
"""

import time
import numpy as np
from hyena_glt.utils.performance import (
    ProfilerContext,
    memory_usage,
    gpu_memory_usage,
    benchmark_model,
    measure_throughput,
    monitor_resources
)

def dummy_computation(data_size: int = 1000000) -> np.ndarray:
    """Simulate a computational workload."""
    # Create some data and perform operations
    data = np.random.randn(data_size)
    result = np.fft.fft(data)
    result = np.abs(result)
    result = np.sort(result)
    return result

def memory_intensive_task(size: int = 10000000) -> np.ndarray:
    """Simulate a memory-intensive task."""
    # Allocate large arrays
    arrays = []
    for i in range(5):
        arr = np.random.randn(size // 5)
        arrays.append(arr)
        time.sleep(0.1)  # Small delay to see memory growth
    
    # Combine arrays
    combined = np.concatenate(arrays)
    return combined

def main():
    print("🚀 Hyena-GLT Performance Monitoring Demo")
    print("=" * 50)
    
    # 1. Basic memory usage
    print("\n📊 Current Memory Usage:")
    mem_stats = memory_usage()
    for key, value in mem_stats.items():
        if 'mb' in key:
            print(f"  {key}: {value:.2f} MB")
        else:
            print(f"  {key}: {value:.2f}%")
    
    # 2. GPU memory usage (if available)
    print("\n🖥️  GPU Memory Usage:")
    gpu_stats = gpu_memory_usage()
    if gpu_stats:
        for key, value in gpu_stats.items():
            print(f"  {key}: {value:.2f} MB")
    else:
        print("  No GPU detected or PyTorch not available")
    
    # 3. Profiling with context manager
    print("\n⏱️  Profiling Example:")
    with ProfilerContext("dummy_computation", enable_gpu=False) as profiler:
        result = dummy_computation(500000)
        
    metrics = profiler.get_metrics()
    print(f"  Operation completed in {metrics['duration_seconds']:.4f} seconds")
    print(f"  Memory delta: {metrics['memory_delta_mb']:+.2f} MB")
    
    # 4. Benchmarking
    print("\n🏃 Benchmarking Example:")
    benchmark_stats = benchmark_model(
        model_fn=lambda x: dummy_computation(x),
        input_data=100000,
        num_runs=5,
        warmup_runs=2
    )
    
    print(f"  Average time: {benchmark_stats['avg_time_seconds']:.4f}s")
    print(f"  Min time: {benchmark_stats['min_time_seconds']:.4f}s")
    print(f"  Max time: {benchmark_stats['max_time_seconds']:.4f}s")
    print(f"  Std deviation: {benchmark_stats['std_time_seconds']:.4f}s")
    
    # 5. Throughput measurement
    print("\n📈 Throughput Measurement:")
    throughput_stats = measure_throughput(
        model_fn=lambda x: dummy_computation(x),
        input_data=50000,
        duration_seconds=3.0
    )
    
    print(f"  Iterations per second: {throughput_stats['iterations_per_second']:.2f}")
    print(f"  Total iterations: {throughput_stats['total_iterations']}")
    print(f"  Actual duration: {throughput_stats['actual_duration_seconds']:.2f}s")
    
    # 6. Resource monitoring during execution
    print("\n📱 Resource Monitoring Example:")
    print("  Running memory-intensive task with monitoring...")
    
    with monitor_resources(interval_seconds=0.5) as snapshots:
        result = memory_intensive_task(2000000)
        time.sleep(2)  # Let monitoring collect some data
    
    if snapshots:
        print(f"  Collected {len(snapshots)} resource snapshots")
        initial_mem = snapshots[0]['memory']['rss_mb']
        peak_mem = max(snap['memory']['rss_mb'] for snap in snapshots)
        print(f"  Initial memory: {initial_mem:.2f} MB")
        print(f"  Peak memory: {peak_mem:.2f} MB")
        print(f"  Memory increase: {peak_mem - initial_mem:.2f} MB")
    
    print("\n✅ Performance monitoring demo completed!")
    print("\nThese utilities can be used to:")
    print("  - Profile model training and inference")
    print("  - Monitor memory usage during data processing")
    print("  - Benchmark different model configurations")
    print("  - Measure throughput for production deployments")
    print("  - Track resource usage over time")

if __name__ == "__main__":
    main()
