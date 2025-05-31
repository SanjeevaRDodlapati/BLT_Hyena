"""
Shared utilities for Hyena-GLT examples and tutorials.

This module provides common functionality used across multiple examples,
including data generation, visualization, model utilities, and analysis functions.
"""

from .data_utils import (
    create_balanced_dataset,
    generate_synthetic_genomic_data,
    load_genomic_dataset,
    validate_genomic_sequences,
)
from .genomic_analysis import (
    analyze_gc_content,
    compare_sequences,
    compute_sequence_features,
    find_genomic_motifs,
    generate_sequence_report,
)
from .model_utils import (
    analyze_model_predictions,
    evaluate_model_comprehensive,
    load_model_with_metadata,
    quick_train_model,
    save_model_with_metadata,
)
from .visualization_utils import (
    create_genomic_dashboard,
    plot_attention_heatmap,
    plot_confusion_matrix,
    plot_sequence_analysis,
    plot_training_history,
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
