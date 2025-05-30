"""
Shared utilities for Hyena-GLT examples and tutorials.

This module provides common functionality used across multiple examples,
including data generation, visualization, model utilities, and analysis functions.
"""

from .data_utils import (
    generate_synthetic_genomic_data,
    load_genomic_dataset,
    create_balanced_dataset,
    validate_genomic_sequences
)

from .visualization_utils import (
    plot_training_history,
    plot_confusion_matrix,
    plot_attention_heatmap,
    plot_sequence_analysis,
    create_genomic_dashboard
)

from .model_utils import (
    quick_train_model,
    evaluate_model_comprehensive,
    analyze_model_predictions,
    save_model_with_metadata,
    load_model_with_metadata
)

from .genomic_analysis import (
    compute_sequence_features,
    find_genomic_motifs,
    analyze_gc_content,
    compare_sequences,
    generate_sequence_report
)

__all__ = [
    # Data utilities
    'generate_synthetic_genomic_data',
    'load_genomic_dataset',
    'create_balanced_dataset',
    'validate_genomic_sequences',
    
    # Visualization utilities
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_attention_heatmap',
    'plot_sequence_analysis',
    'create_genomic_dashboard',
    
    # Model utilities
    'quick_train_model',
    'evaluate_model_comprehensive',
    'analyze_model_predictions',
    'save_model_with_metadata',
    'load_model_with_metadata',
    
    # Genomic analysis
    'compute_sequence_features',
    'find_genomic_motifs',
    'analyze_gc_content',
    'compare_sequences',
    'generate_sequence_report'
]
