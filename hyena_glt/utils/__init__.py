"""
Hyena-GLT Utilities

This module provides utility functions and helper classes for the Hyena-GLT framework.
"""

from .visualization import (
    plot_attention_maps,
    plot_training_curves,
    plot_sequence_embeddings,
    plot_genomic_features
)

from .model_utils import (
    count_parameters,
    get_model_size,
    validate_model_config,
    load_pretrained_weights
)

from .analysis import (
    analyze_tokenization,
    compute_sequence_statistics,
    analyze_model_predictions,
    generate_attribution_maps
)

from .genomic_utils import (
    reverse_complement,
    translate_dna,
    gc_content,
    find_orfs,
    validate_sequence
)

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

from .performance import (
    ProfilerContext,
    memory_usage,
    gpu_memory_usage,
    benchmark_model,
    measure_throughput
)

__version__ = "1.0.1"
__author__ = "Hyena-GLT Development Team"

__all__ = [
    # Visualization
    "plot_attention_maps",
    "plot_training_curves", 
    "plot_sequence_embeddings",
    "plot_genomic_features",
    
    # Model utilities
    "count_parameters",
    "get_model_size",
    "validate_model_config",
    "load_pretrained_weights",
    
    # Analysis
    "analyze_tokenization",
    "compute_sequence_statistics",
    "analyze_model_predictions",
    "generate_attribution_maps",
    
    # Genomic utilities
    "reverse_complement",
    "translate_dna",
    "gc_content",
    "find_orfs",
    "validate_sequence",
    
    # Performance monitoring
    "ProfilerContext",
    "memory_usage",
    "gpu_memory_usage",
    "benchmark_model",
    "measure_throughput",
    "monitor_resources"
]
