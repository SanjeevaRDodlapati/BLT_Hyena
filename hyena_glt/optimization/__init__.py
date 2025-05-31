"""
Hyena-GLT Optimization Module

This module provides comprehensive optimization tools for the Hyena-GLT model,
including quantization, pruning, knowledge distillation, and deployment optimization.
"""

# Core optimization components
try:
    from .quantization import (
        DynamicQuantizer,
        ModelQuantizer,
        QATTrainer,
        QuantizationCalibrator,
        QuantizationConfig,
    )

    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False

try:
    from .pruning import (
        GradientPruner,
        MagnitudePruner,
        ModelPruner,
        PruningConfig,
        StructuredPruner,
        UnstructuredPruner,
    )

    HAS_PRUNING = True
except ImportError:
    HAS_PRUNING = False

try:
    from .distillation import (
        AttentionDistiller,
        DistillationConfig,
        FeatureDistiller,
        KnowledgeDistiller,
        StudentModelFactory,
    )

    HAS_DISTILLATION = True
except ImportError:
    HAS_DISTILLATION = False

try:
    from .deployment import (
        DeploymentConfig,
        InferenceEngine,
        ModelOptimizer,
        ModelProfiler,
        ONNXExporter,
        TensorRTOptimizer,
    )

    HAS_DEPLOYMENT = True
except ImportError:
    HAS_DEPLOYMENT = False

try:
    from .memory import (
        ActivationCheckpointing,
        GradientCheckpointing,
        MemoryConfig,
        MemoryOptimizer,
        MemoryProfiler,
    )

    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False

# Convenience imports
__all__ = [
    # Availability flags
    "HAS_QUANTIZATION",
    "HAS_PRUNING",
    "HAS_DISTILLATION",
    "HAS_DEPLOYMENT",
    "HAS_MEMORY",
]

# Add available components to __all__
if HAS_QUANTIZATION:
    __all__.extend(
        [
            "QuantizationConfig",
            "ModelQuantizer",
            "DynamicQuantizer",
            "QATTrainer",
            "QuantizationCalibrator",
        ]
    )

if HAS_PRUNING:
    __all__.extend(
        [
            "PruningConfig",
            "ModelPruner",
            "StructuredPruner",
            "UnstructuredPruner",
            "MagnitudePruner",
            "GradientPruner",
        ]
    )

if HAS_DISTILLATION:
    __all__.extend(
        [
            "DistillationConfig",
            "KnowledgeDistiller",
            "FeatureDistiller",
            "AttentionDistiller",
            "StudentModelFactory",
        ]
    )

if HAS_DEPLOYMENT:
    __all__.extend(
        [
            "DeploymentConfig",
            "ModelOptimizer",
            "ONNXExporter",
            "TensorRTOptimizer",
            "InferenceEngine",
            "ModelProfiler",
        ]
    )

if HAS_MEMORY:
    __all__.extend(
        [
            "MemoryConfig",
            "MemoryOptimizer",
            "GradientCheckpointing",
            "ActivationCheckpointing",
            "MemoryProfiler",
        ]
    )


def get_optimization_info() -> dict[str, bool]:
    """Get information about available optimization modules."""
    info = {
        "quantization": HAS_QUANTIZATION,
        "pruning": HAS_PRUNING,
        "distillation": HAS_DISTILLATION,
        "deployment": HAS_DEPLOYMENT,
        "memory": HAS_MEMORY,
    }
    return info


def check_dependencies() -> bool:
    """Check if all optimization dependencies are available."""
    missing = []

    if not HAS_QUANTIZATION:
        missing.append("quantization (torch.quantization)")
    if not HAS_PRUNING:
        missing.append("pruning (torch.nn.utils.prune)")
    if not HAS_DISTILLATION:
        missing.append("distillation")
    if not HAS_DEPLOYMENT:
        missing.append("deployment (onnx, tensorrt)")
    if not HAS_MEMORY:
        missing.append("memory optimization")

    if missing:
        print(f"Missing optimization modules: {', '.join(missing)}")
        print("Install additional dependencies for full optimization support.")
    else:
        print("All optimization modules available!")

    return len(missing) == 0
