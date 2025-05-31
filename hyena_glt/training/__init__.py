"""Training infrastructure for Hyena-GLT models."""

from .checkpointing import CheckpointManager
from .curriculum import CurriculumLearning
from .metrics import GenomicMetrics, MultiTaskMetrics
from .multitask import MultiTaskLoss, TaskConfig, TaskWeightScheduler
from .optimization import AdamWWithScheduler, create_optimizer, create_scheduler
from .trainer import HyenaGLTTrainer, TrainingConfig

# Import fine-tuning utilities
try:
    from .finetuning import (
        FineTuner,
        FinetuningConfig,
        LayerFreezer,
        ModelAdapter,
        TaskSpecificFineTuner,
        finetune_for_generation,
        finetune_for_sequence_classification,
        finetune_for_token_classification,
    )

    _finetuning_available = True
except ImportError:
    _finetuning_available = False

# Import pretrained model utilities
try:
    from .pretrained import (
        ModelConverter,
        ModelDownloader,
        ModelInfo,
        ModelRegistry,
        PretrainedModelManager,
        get_model_info,
        list_pretrained_models,
        load_pretrained_model,
    )

    _pretrained_available = True
except ImportError:
    _pretrained_available = False

__all__ = [
    "HyenaGLTTrainer",
    "TrainingConfig",
    "AdamWWithScheduler",
    "create_optimizer",
    "create_scheduler",
    "CurriculumLearning",
    "MultiTaskLoss",
    "TaskWeightScheduler",
    "TaskConfig",
    "GenomicMetrics",
    "MultiTaskMetrics",
    "CheckpointManager",
]

# Add fine-tuning exports if available
if _finetuning_available:
    __all__.extend(
        [
            "FinetuningConfig",
            "FineTuner",
            "LayerFreezer",
            "ModelAdapter",
            "TaskSpecificFineTuner",
            "finetune_for_sequence_classification",
            "finetune_for_token_classification",
            "finetune_for_generation",
        ]
    )

# Add pretrained model exports if available
if _pretrained_available:
    __all__.extend(
        [
            "ModelInfo",
            "ModelRegistry",
            "ModelDownloader",
            "PretrainedModelManager",
            "ModelConverter",
            "load_pretrained_model",
            "list_pretrained_models",
            "get_model_info",
        ]
    )
