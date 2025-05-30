"""
Hyena-GLT Utilities

This module provides utility functions and helper classes for the Hyena-GLT framework.
"""

# Import only what exists
from .performance import (
    ProfilerContext,
    memory_usage,
    gpu_memory_usage,
    benchmark_model,
    measure_throughput,
    monitor_resources
)

__version__ = "1.0.1"
__author__ = "Hyena-GLT Development Team"

__all__ = [
    # Performance monitoring
    "ProfilerContext",
    "memory_usage",
    "gpu_memory_usage", 
    "benchmark_model",
    "measure_throughput",
    "monitor_resources"
]
