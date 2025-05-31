"""Configuration management for Hyena-GLT models."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from transformers import PretrainedConfig  # type: ignore[attr-defined]


class HyenaGLTConfig(PretrainedConfig):
    """Configuration class for Hyena-GLT model."""

    model_type = "hyena_glt"

    def __init__(
        self,
        # Model architecture
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 32768,
        # BLT specific parameters
        local_encoder_layers: int = 2,
        local_decoder_layers: int = 2,
        patch_size: int = 8,
        min_patch_size: int = 4,
        max_patch_size: int = 16,
        dynamic_patching: bool = True,
        cross_attention_layers: int = 4,
        # Hyena specific parameters
        hyena_order: int = 2,
        hyena_filter_size: int = 512,
        hyena_short_filter_size: int = 32,
        use_bias: bool = True,
        use_glu: bool = True,
        hyena_dropout: float = 0.1,
        # Genomic specific parameters
        sequence_type: str = "dna",  # "dna", "rna", "protein", "mixed"
        genomic_vocab_size: int = 4096,
        enable_reverse_complement: bool = True,
        kmer_size: int = 3,
        overlap_size: int = 1,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        # Multi-task learning
        task_weights: dict[str, float] | None = None,
        # Optimization
        use_gradient_checkpointing: bool = False,
        use_flash_attention: bool = True,
        compile_model: bool = False,
        # Hardware specific
        device: str = "auto",
        precision: str = "float16",  # "float16", "bfloat16", "float32"
        **kwargs: Any,
    ) -> None:
        # Set default task weights
        if task_weights is None:
            task_weights = {
                "sequence_classification": 1.0,
                "token_classification": 1.0,
                "sequence_generation": 1.0,
                "masked_lm": 1.0,
            }

        # Model architecture
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

        # BLT specific parameters
        self.local_encoder_layers = local_encoder_layers
        self.local_decoder_layers = local_decoder_layers
        self.patch_size = patch_size
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.dynamic_patching = dynamic_patching
        self.cross_attention_layers = cross_attention_layers

        # Hyena specific parameters
        self.hyena_order = hyena_order
        self.hyena_filter_size = hyena_filter_size
        self.hyena_short_filter_size = hyena_short_filter_size
        self.use_bias = use_bias
        self.use_glu = use_glu
        self.hyena_dropout = hyena_dropout

        # Genomic specific parameters
        self.sequence_type = sequence_type
        self.genomic_vocab_size = genomic_vocab_size
        self.enable_reverse_complement = enable_reverse_complement
        self.kmer_size = kmer_size
        self.overlap_size = overlap_size

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # Multi-task learning
        self.task_weights = task_weights

        # Optimization
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_flash_attention = use_flash_attention
        self.compile_model = compile_model

        # Hardware specific
        self.device = device
        self.precision = precision

        # Validation
        self._validate_config()

        super().__init__(**kwargs)  # type: ignore[no-untyped-call]

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.sequence_type not in ["dna", "rna", "protein", "mixed"]:
            raise ValueError(f"Invalid sequence_type: {self.sequence_type}")

        if self.hyena_order < 1:
            raise ValueError("hyena_order must be >= 1")

        if (
            self.patch_size < self.min_patch_size
            or self.patch_size > self.max_patch_size
        ):
            raise ValueError(
                f"patch_size must be between {self.min_patch_size} and {self.max_patch_size}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> "HyenaGLTConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "HyenaGLTConfig":
        """Load config from JSON file."""
        with open(json_path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save(self, save_path: str) -> None:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# Predefined configurations for common use cases
GENOMIC_CONFIGS = {
    "hyena-glt-small": HyenaGLTConfig(
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=16384,
        hyena_filter_size=256,
    ),
    "hyena-glt-base": HyenaGLTConfig(
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=32768,
        hyena_filter_size=512,
    ),
    "hyena-glt-large": HyenaGLTConfig(
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=65536,
        hyena_filter_size=1024,
    ),
    "hyena-glt-protein": HyenaGLTConfig(
        sequence_type="protein",
        genomic_vocab_size=8192,
        kmer_size=2,
        enable_reverse_complement=False,
    ),
    "hyena-glt-long": HyenaGLTConfig(
        max_position_embeddings=131072,
        hyena_filter_size=1024,
        patch_size=16,
        use_gradient_checkpointing=True,
    ),
}
