"""
Task-specific fine-tuning scripts for Hyena-GLT models.

This module provides ready-to-use scripts for fine-tuning Hyena-GLT models
on common genomic tasks with best practices and optimizations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from ..data.dataset import (
    SequenceClassificationDataset,
    SequenceGenerationDataset,
    TokenClassificationDataset,
)
from ..data.tokenizer import DNATokenizer, ProteinTokenizer
from .finetuning import (
    TaskSpecificFineTuner,
    finetune_for_generation,
    finetune_for_sequence_classification,
    finetune_for_token_classification,
)
from .mixed_precision import MixedPrecisionConfig, PrecisionMode, create_mixed_precision_manager
from .pretrained import PretrainedModelManager

logger = logging.getLogger(__name__)


def get_optimal_precision_config(task_type: str, model_size: str = "medium", hardware_info: dict = None) -> MixedPrecisionConfig:
    """
    Get optimal mixed precision configuration for specific genomic tasks.
    
    Args:
        task_type: Type of genomic task (genome_annotation, variant_effect, etc.)
        model_size: Size of the model (small, medium, large)
        hardware_info: Dict containing GPU info (compute_capability, memory_gb, etc.)
    
    Returns:
        Optimized MixedPrecisionConfig for the task
    """
    if hardware_info is None:
        hardware_info = {}
        if torch.cuda.is_available():
            hardware_info['compute_capability'] = torch.cuda.get_device_capability()[0]
            hardware_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Base configuration
    base_config = {
        'dynamic_loss_scale': True,
        'monitor_overflow': True,
        'log_precision_stats': True,
        'precision_check_interval': 100,
    }
    
    if task_type == "genome_annotation":
        # Long sequences need careful precision management
        if hardware_info.get('compute_capability', 0) >= 8.0:
            # Use FP8 on H100/A100 for maximum efficiency
            config = MixedPrecisionConfig(
                mode=PrecisionMode.FP8,
                gradient_clipping=1.0,
                gradient_checkpointing=True,
                fp8_format="E4M3",
                **base_config
            )
        else:
            # Adaptive precision for older hardware
            config = MixedPrecisionConfig(
                mode=PrecisionMode.ADAPTIVE,
                gradient_clipping=1.0,
                gradient_checkpointing=True,
                **base_config
            )
    
    elif task_type == "variant_effect":
        # Stability is crucial for variant effect prediction
        config = MixedPrecisionConfig(
            mode=PrecisionMode.BF16,
            gradient_clipping=0.5,
            growth_interval=1000,  # Conservative scaling
            **base_config
        )
    
    elif task_type == "protein_function":
        # Protein sequences can be very long
        if hardware_info.get('memory_gb', 0) > 32 and hardware_info.get('compute_capability', 0) >= 8.0:
            config = MixedPrecisionConfig(
                mode=PrecisionMode.FP8,
                gradient_checkpointing=True,
                cpu_offload=model_size == "large",
                fp8_format="E4M3",
                **base_config
            )
        else:
            config = MixedPrecisionConfig(
                mode=PrecisionMode.BF16,
                gradient_checkpointing=True,
                cpu_offload=model_size == "large",
                **base_config
            )
    
    elif task_type == "generation":
        # Memory optimization is key for generation
        config = MixedPrecisionConfig(
            mode=PrecisionMode.FP16,
            gradient_checkpointing=True,
            gradient_clipping=2.0,  # Higher threshold for generation
            cpu_offload=True,
            growth_interval=500,  # Frequent scaling updates
            **base_config
        )
    
    else:
        # Default configuration
        config = MixedPrecisionConfig(
            mode=PrecisionMode.FP16,
            **base_config
        )
    
    return config


def apply_task_specific_optimizations(config: Any, task_type: str, mixed_precision_config: MixedPrecisionConfig) -> Any:
    """Apply task-specific optimizations to training configuration."""
    
    # Set mixed precision flags based on the precision mode
    config.fp16 = mixed_precision_config.mode in [PrecisionMode.FP16, PrecisionMode.MIXED_FP16, PrecisionMode.ADAPTIVE]
    config.bf16 = mixed_precision_config.mode in [PrecisionMode.BF16, PrecisionMode.MIXED_BF16]
    config.fp8 = mixed_precision_config.mode == PrecisionMode.FP8
    
    # Set other precision-related flags
    config.dynamic_loss_scaling = mixed_precision_config.dynamic_loss_scale
    config.gradient_clipping = mixed_precision_config.gradient_clipping
    config.gradient_checkpointing = mixed_precision_config.gradient_checkpointing
    config.precision_monitoring = mixed_precision_config.monitor_overflow
    
    # Task-specific optimizations
    if task_type == "genome_annotation":
        # Optimize for long sequence processing
        config.dataloader_num_workers = min(8, config.batch_size)
        config.save_strategy = "steps"
        config.save_steps = 500
        config.eval_steps = 500
        config.logging_steps = 50
        
    elif task_type == "variant_effect":
        # Conservative settings for stability
        config.max_grad_norm = 0.5
        config.lr_scheduler_type = "cosine_with_restarts"
        config.eval_strategy = "steps"
        config.eval_steps = 200
        
    elif task_type == "protein_function":
        # Memory-efficient settings for long proteins
        config.dataloader_pin_memory = False  # Save GPU memory
        config.remove_unused_columns = True
        config.prediction_loss_only = False
        
    elif task_type == "generation":
        # Generation-specific optimizations
        config.prediction_loss_only = True
        config.include_inputs_for_metrics = False
        config.metric_for_best_model = "perplexity"
        config.greater_is_better = False
    
    return config


class GenomeAnnotationFineTuner:
    """Fine-tuner for genome annotation tasks (gene finding, regulatory element prediction)."""

    def __init__(self, pretrained_model_path: str, output_dir: str):
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir

        # Genome annotation specific labels
        self.labels = [
            "intergenic",
            "gene",
            "exon",
            "intron",
            "promoter",
            "enhancer",
            "silencer",
            "transcription_start_site",
            "transcription_end_site",
            "splice_site",
        ]

        self.config = TaskSpecificFineTuner.create_token_classification_config(
            pretrained_model_path=pretrained_model_path,
            output_dir=output_dir,
            num_labels=len(self.labels),
            learning_rate=3e-5,
            num_epochs=5,
            batch_size=4,  # Large sequences require smaller batches
            max_length=8192,
            gradient_accumulation_steps=4,
            warmup_ratio=0.15,
            weight_decay=0.01,
            use_layer_wise_decay=True,
            layer_wise_lr_decay=0.9,
        )

        # Get optimal mixed precision configuration
        self.mixed_precision_config = get_optimal_precision_config("genome_annotation")
        self.config = apply_task_specific_optimizations(self.config, "genome_annotation", self.mixed_precision_config)
        
        # Create mixed precision manager
        self.precision_manager = create_mixed_precision_manager(
            mode=self.mixed_precision_config.mode,
            gradient_clipping=self.mixed_precision_config.gradient_clipping,
            dynamic_loss_scale=self.mixed_precision_config.dynamic_loss_scale,
            monitor_overflow=self.mixed_precision_config.monitor_overflow,
            gradient_checkpointing=self.mixed_precision_config.gradient_checkpointing,
        )

    def create_dataset(
        self, data_path: str, tokenizer_type: str = "dna"
    ) -> TokenClassificationDataset:
        """Create dataset for genome annotation."""
        if tokenizer_type == "dna":
            tokenizer = DNATokenizer(k=6)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

        return TokenClassificationDataset(
            data=data_path,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            label_names=self.labels,
        )

    def fine_tune(self, train_data_path: str, eval_data_path: str | None = None) -> Any:
        """Fine-tune model for genome annotation."""
        train_dataset = self.create_dataset(train_data_path)
        eval_dataset = self.create_dataset(eval_data_path) if eval_data_path else None

        return finetune_for_token_classification(
            pretrained_model_path=self.pretrained_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=self.output_dir,
            num_labels=len(self.labels),
        )

    def get_precision_stats(self) -> dict:
        """Get mixed precision training statistics."""
        return self.precision_manager.get_precision_stats()
    
    def optimize_model_for_task(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for genome annotation task with mixed precision."""
        # Apply mixed precision optimizations
        model = self.precision_manager.optimize_model_for_precision(model)
        
        # Task-specific model optimizations
        if hasattr(model, 'config'):
            # Enable attention optimizations for long sequences
            if hasattr(model.config, 'use_flash_attn'):
                model.config.use_flash_attn = True
            if hasattr(model.config, 'attention_dropout'):
                model.config.attention_dropout = 0.0  # Disable for better precision
        
        return model


class VariantEffectFineTuner:
    """Fine-tuner for variant effect prediction tasks."""

    def __init__(self, pretrained_model_path: str, output_dir: str):
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir

        # Variant effect labels
        self.labels = [
            "benign",
            "likely_benign",
            "uncertain_significance",
            "likely_pathogenic",
            "pathogenic",
        ]

        self.config = TaskSpecificFineTuner.create_sequence_classification_config(
            pretrained_model_path=pretrained_model_path,
            output_dir=output_dir,
            num_labels=len(self.labels),
            learning_rate=2e-5,
            num_epochs=8,
            batch_size=16,
            max_length=1024,
            warmup_ratio=0.1,
            weight_decay=0.01,
            label_smoothing=0.1,
            early_stopping_patience=3,
        )

        # Get optimal mixed precision configuration
        self.mixed_precision_config = get_optimal_precision_config("variant_effect")
        self.config = apply_task_specific_optimizations(self.config, "variant_effect", self.mixed_precision_config)
        
        # Create mixed precision manager
        self.precision_manager = create_mixed_precision_manager(
            mode=self.mixed_precision_config.mode,
            gradient_clipping=self.mixed_precision_config.gradient_clipping,
            dynamic_loss_scale=self.mixed_precision_config.dynamic_loss_scale,
            monitor_overflow=self.mixed_precision_config.monitor_overflow,
        )

    def create_dataset(
        self, data_path: str, tokenizer_type: str = "dna"
    ) -> SequenceClassificationDataset:
        """Create dataset for variant effect prediction."""
        if tokenizer_type == "dna":
            tokenizer = DNATokenizer(k=6)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

        return SequenceClassificationDataset(
            data=data_path,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            label_names=self.labels,
        )

    def fine_tune(self, train_data_path: str, eval_data_path: str | None = None) -> Any:
        """Fine-tune model for variant effect prediction."""
        train_dataset = self.create_dataset(train_data_path)
        eval_dataset = self.create_dataset(eval_data_path) if eval_data_path else None

        return finetune_for_sequence_classification(
            pretrained_model_path=self.pretrained_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=self.output_dir,
            num_labels=len(self.labels),
        )

    def get_precision_stats(self) -> dict:
        """Get mixed precision training statistics."""
        return self.precision_manager.get_precision_stats()
    
    def optimize_model_for_task(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for variant effect prediction with mixed precision."""
        # Apply mixed precision optimizations
        model = self.precision_manager.optimize_model_for_precision(model)
        
        # Task-specific model optimizations for variant effect prediction
        if hasattr(model, 'config'):
            # Conservative settings for numerical stability
            if hasattr(model.config, 'hidden_dropout_prob'):
                model.config.hidden_dropout_prob = 0.1  # Moderate dropout for stability
            if hasattr(model.config, 'attention_probs_dropout_prob'):
                model.config.attention_probs_dropout_prob = 0.1
        
        return model


class ProteinFunctionFineTuner:
    """Fine-tuner for protein function prediction tasks."""

    def __init__(
        self, pretrained_model_path: str, output_dir: str, function_type: str = "go"
    ):
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.function_type = function_type

        # Function prediction labels (example for GO terms)
        if function_type == "go":
            self.labels = [
                "molecular_function",
                "biological_process",
                "cellular_component",
                "catalytic_activity",
                "binding",
                "transporter_activity",
                "enzyme_regulator_activity",
            ]
        elif function_type == "ec":
            self.labels = [
                "oxidoreductase",
                "transferase",
                "hydrolase",
                "lyase",
                "isomerase",
                "ligase",
                "translocase",
            ]
        else:
            raise ValueError(f"Unsupported function type: {function_type}")

        self.config = TaskSpecificFineTuner.create_sequence_classification_config(
            pretrained_model_path=pretrained_model_path,
            output_dir=output_dir,
            num_labels=len(self.labels),
            learning_rate=1e-5,
            num_epochs=10,
            batch_size=8,
            max_length=2048,
            warmup_ratio=0.2,
            weight_decay=0.01,
            use_layer_wise_decay=True,
            layer_wise_lr_decay=0.95,
        )

        # Get optimal mixed precision configuration based on hardware
        hardware_info = {}
        if torch.cuda.is_available():
            hardware_info['compute_capability'] = torch.cuda.get_device_capability()[0]
            hardware_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        self.mixed_precision_config = get_optimal_precision_config("protein_function", "medium", hardware_info)
        self.config = apply_task_specific_optimizations(self.config, "protein_function", self.mixed_precision_config)
        
        # Create mixed precision manager
        self.precision_manager = create_mixed_precision_manager(
            mode=self.mixed_precision_config.mode,
            gradient_clipping=self.mixed_precision_config.gradient_clipping,
            dynamic_loss_scale=self.mixed_precision_config.dynamic_loss_scale,
            monitor_overflow=self.mixed_precision_config.monitor_overflow,
            gradient_checkpointing=self.mixed_precision_config.gradient_checkpointing,
            fp8_format=getattr(self.mixed_precision_config, 'fp8_format', 'E4M3'),
        )

    def create_dataset(self, data_path: str) -> SequenceClassificationDataset:
        """Create dataset for protein function prediction."""
        tokenizer = ProteinTokenizer()

        return SequenceClassificationDataset(
            data=data_path,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            label_names=self.labels,
        )

    def fine_tune(self, train_data_path: str, eval_data_path: str | None = None) -> Any:
        """Fine-tune model for protein function prediction."""
        train_dataset = self.create_dataset(train_data_path)
        eval_dataset = self.create_dataset(eval_data_path) if eval_data_path else None

        return finetune_for_sequence_classification(
            pretrained_model_path=self.pretrained_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=self.output_dir,
            num_labels=len(self.labels),
        )

    def get_precision_stats(self) -> dict:
        """Get mixed precision training statistics."""
        return self.precision_manager.get_precision_stats()
    
    def optimize_model_for_task(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for protein function prediction with mixed precision."""
        # Apply mixed precision optimizations
        model = self.precision_manager.optimize_model_for_precision(model)
        
        # Task-specific model optimizations for protein function prediction
        if hasattr(model, 'config'):
            # Optimize for long protein sequences
            if hasattr(model.config, 'use_flash_attn'):
                model.config.use_flash_attn = True
            if hasattr(model.config, 'max_position_embeddings'):
                # Ensure we can handle long protein sequences
                model.config.max_position_embeddings = max(
                    model.config.max_position_embeddings, 2048
                )
            # Memory efficient settings for protein sequences
            if hasattr(model.config, 'gradient_checkpointing'):
                model.config.gradient_checkpointing = True
        
        return model


class GenomeGenerationFineTuner:
    """Fine-tuner for genome sequence generation tasks."""

    def __init__(
        self,
        pretrained_model_path: str,
        output_dir: str,
        generation_type: str = "promoter",
    ):
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.generation_type = generation_type

        self.config = TaskSpecificFineTuner.create_generation_config(
            pretrained_model_path=pretrained_model_path,
            output_dir=output_dir,
            learning_rate=5e-6,
            num_epochs=5,
            batch_size=2,
            max_length=4096,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            weight_decay=0.01,
        )

        # Get optimal mixed precision configuration for generation
        self.mixed_precision_config = get_optimal_precision_config("generation")
        self.config = apply_task_specific_optimizations(self.config, "generation", self.mixed_precision_config)
        
        # Create mixed precision manager  
        self.precision_manager = create_mixed_precision_manager(
            mode=self.mixed_precision_config.mode,
            gradient_clipping=self.mixed_precision_config.gradient_clipping,
            dynamic_loss_scale=self.mixed_precision_config.dynamic_loss_scale,
            monitor_overflow=self.mixed_precision_config.monitor_overflow,
            gradient_checkpointing=self.mixed_precision_config.gradient_checkpointing,
            cpu_offload=self.mixed_precision_config.cpu_offload,
        )

    def create_dataset(
        self, data_path: str, tokenizer_type: str = "dna"
    ) -> SequenceGenerationDataset:
        """Create dataset for sequence generation."""
        if tokenizer_type == "dna":
            tokenizer = DNATokenizer(k=6)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

        return SequenceGenerationDataset(
            data=data_path,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            mask_probability=0.15,
        )

    def fine_tune(self, train_data_path: str, eval_data_path: str | None = None) -> Any:
        """Fine-tune model for sequence generation."""
        train_dataset = self.create_dataset(train_data_path)
        eval_dataset = self.create_dataset(eval_data_path) if eval_data_path else None

        return finetune_for_generation(
            pretrained_model_path=self.pretrained_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=self.output_dir,
        )

    def get_precision_stats(self) -> dict:
        """Get mixed precision training statistics."""
        return self.precision_manager.get_precision_stats()
    
    def optimize_model_for_task(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for genome generation task with mixed precision."""
        # Apply mixed precision optimizations
        model = self.precision_manager.optimize_model_for_precision(model)
        
        # Task-specific model optimizations for generation
        if hasattr(model, 'config'):
            # Enable memory efficient attention for long generation sequences
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True  # Enable KV caching for generation
            if hasattr(model.config, 'use_flash_attn'):
                model.config.use_flash_attn = True
            # Optimize for generation tasks
            if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
                model.config.pad_token_id = model.config.eos_token_id
        
        return model


class DomainAdaptationFineTuner:
    """Fine-tuner for domain adaptation across species or sequence types."""

    def __init__(
        self,
        pretrained_model_path: str,
        output_dir: str,
        source_domain: str,
        target_domain: str,
    ):
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.source_domain = source_domain
        self.target_domain = target_domain

        self.config = TaskSpecificFineTuner.create_domain_adaptation_config(
            pretrained_model_path=pretrained_model_path,
            output_dir=output_dir,
            task_type="sequence_classification",
            learning_rate=5e-6,
            num_epochs=15,
            batch_size=8,
            warmup_ratio=0.3,
            weight_decay=0.01,
            use_layer_wise_decay=True,
            layer_wise_lr_decay=0.98,
        )
        
        # Adaptive mixed precision for domain adaptation
        self.mixed_precision_config = MixedPrecisionConfig(
            mode=PrecisionMode.ADAPTIVE,
            gradient_clipping=0.8,
            dynamic_loss_scale=True,
            monitor_overflow=True,
            gradient_checkpointing=False,  # Keep gradients for adaptation analysis
            growth_interval=1500,  # Conservative for domain shifts
        )
        
        self.config = apply_task_specific_optimizations(self.config, "domain_adaptation", self.mixed_precision_config)
        
        # Create mixed precision manager
        self.precision_manager = create_mixed_precision_manager(
            mode=self.mixed_precision_config.mode,
            gradient_clipping=self.mixed_precision_config.gradient_clipping,
            dynamic_loss_scale=self.mixed_precision_config.dynamic_loss_scale,
            monitor_overflow=self.mixed_precision_config.monitor_overflow,
        )

    def fine_tune_gradual(
        self, target_data_path: str, eval_data_path: str | None = None
    ) -> None:
        """Perform gradual domain adaptation."""
        # This would implement gradual unfreezing and learning rate scheduling
        # for domain adaptation
        pass

    def get_precision_stats(self) -> dict:
        """Get mixed precision training statistics."""
        return self.precision_manager.get_precision_stats()
    
    def optimize_model_for_task(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for domain adaptation with mixed precision."""
        # Apply mixed precision optimizations
        model = self.precision_manager.optimize_model_for_precision(model)
        
        # Task-specific model optimizations for domain adaptation
        if hasattr(model, 'config'):
            # Conservative settings for domain adaptation stability
            if hasattr(model.config, 'hidden_dropout_prob'):
                model.config.hidden_dropout_prob = 0.2  # Higher dropout for domain adaptation
            if hasattr(model.config, 'attention_probs_dropout_prob'):
                model.config.attention_probs_dropout_prob = 0.1
            # Disable aggressive optimizations that might hurt adaptation
            if hasattr(model.config, 'use_flash_attn'):
                model.config.use_flash_attn = False  # More stable for domain shifts
        
        return model


def main() -> None:
    """Main function for command-line fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Hyena-GLT models for genomic tasks"
    )

    # Required arguments
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "genome_annotation",
            "variant_effect",
            "protein_function",
            "generation",
            "domain_adaptation",
        ],
        help="Type of fine-tuning task",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Path to pre-trained model or model name",
    )
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for fine-tuned model",
    )

    # Optional arguments
    parser.add_argument(
        "--eval_data", type=str, default=None, help="Path to evaluation data"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="dna",
        choices=["dna", "rna", "protein"],
        help="Type of tokenizer to use",
    )
    parser.add_argument(
        "--function_type",
        type=str,
        default="go",
        choices=["go", "ec"],
        help="Type of protein function prediction (for protein_function task)",
    )
    parser.add_argument(
        "--generation_type",
        type=str,
        default="promoter",
        help="Type of sequence generation (for generation task)",
    )
    parser.add_argument(
        "--source_domain",
        type=str,
        default=None,
        help="Source domain for domain adaptation",
    )
    parser.add_argument(
        "--target_domain",
        type=str,
        default=None,
        help="Target domain for domain adaptation",
    )

    # Training parameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (if not specified, task-specific default will be used)",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--max_length", type=int, default=None, help="Maximum sequence length"
    )

    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Check if pretrained_model is a path or model name
        if Path(args.pretrained_model).exists():
            pretrained_model_path = args.pretrained_model
        else:
            # Try to load from model registry
            manager = PretrainedModelManager()
            pretrained_model_path = str(manager.download_model(args.pretrained_model))

        # Create appropriate fine-tuner based on task
        finetuner: Any
        if args.task == "genome_annotation":
            finetuner = GenomeAnnotationFineTuner(
                pretrained_model_path, args.output_dir
            )
        elif args.task == "variant_effect":
            finetuner = VariantEffectFineTuner(pretrained_model_path, args.output_dir)
        elif args.task == "protein_function":
            finetuner = ProteinFunctionFineTuner(
                pretrained_model_path, args.output_dir, args.function_type
            )
        elif args.task == "generation":
            finetuner = GenomeGenerationFineTuner(
                pretrained_model_path, args.output_dir, args.generation_type
            )
        elif args.task == "domain_adaptation":
            if not args.source_domain or not args.target_domain:
                raise ValueError(
                    "Domain adaptation requires --source_domain and --target_domain"
                )
            finetuner = DomainAdaptationFineTuner(
                pretrained_model_path,
                args.output_dir,
                args.source_domain,
                args.target_domain,
            )
        else:
            raise ValueError(f"Unknown task: {args.task}")

        # Override default parameters if specified
        if args.learning_rate is not None:
            finetuner.config.learning_rate = args.learning_rate
        if args.batch_size is not None:
            finetuner.config.batch_size = args.batch_size
        if args.num_epochs is not None:
            finetuner.config.num_epochs = args.num_epochs
        if args.max_length is not None:
            finetuner.config.max_length = args.max_length

        # Start fine-tuning
        logger.info(f"Starting fine-tuning for task: {args.task}")
        logger.info(f"Pretrained model: {pretrained_model_path}")
        logger.info(f"Training data: {args.train_data}")
        logger.info(f"Output directory: {args.output_dir}")

        trainer = finetuner.fine_tune(args.train_data, args.eval_data)

        # Save configuration
        config_path = Path(args.output_dir) / "finetuning_args.json"
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

        logger.info("Fine-tuning completed successfully!")

        # Evaluate if eval data provided
        if args.eval_data and hasattr(trainer, "evaluate"):
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()

            results_path = Path(args.output_dir) / "eval_results.json"
            with open(results_path, "w") as f:
                json.dump(eval_results, f, indent=2)

            logger.info(f"Evaluation results saved to {results_path}")
            for metric, value in eval_results.items():
                logger.info(f"{metric}: {value}")

    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
