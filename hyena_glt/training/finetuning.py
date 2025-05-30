"""
Fine-tuning utilities for Hyena-GLT models.

This module provides utilities for fine-tuning pre-trained Hyena-GLT models
on specific genomic tasks with task-specific optimizations and data handling.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import copy
from transformers import get_scheduler

from ..config import HyenaGLTConfig
from ..model.hyena_glt import (
    HyenaGLT,
    HyenaGLTForSequenceClassification,
    HyenaGLTForTokenClassification,
    HyenaGLTForSequenceGeneration,
    HyenaGLTForMultiTask
)
from .trainer import HyenaGLTTrainer, TrainingConfig
from .optimization import AdamWWithScheduler, LayerWiseDecayOptimizer
from .metrics import GenomicMetrics
from ..data.dataset import GenomicDataset
from ..evaluation.metrics import MultiTaskEvaluator

logger = logging.getLogger(__name__)


@dataclass
class FinetuningConfig:
    """Configuration for fine-tuning Hyena-GLT models."""
    
    # Model and checkpoint paths
    pretrained_model_path: str
    output_dir: str
    
    # Task configuration
    task_type: str = "sequence_classification"  # sequence_classification, token_classification, sequence_generation, multitask
    num_labels: Optional[int] = None
    label_names: Optional[List[str]] = None
    
    # Fine-tuning strategy
    freeze_backbone: bool = False
    freeze_layers: Optional[List[str]] = None  # Layer patterns to freeze
    unfreeze_layers: Optional[List[str]] = None  # Layer patterns to unfreeze
    layer_wise_lr_decay: float = 0.9  # Learning rate decay for lower layers
    
    # Training parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data parameters
    max_length: int = 2048
    batch_size: int = 8
    dataloader_num_workers: int = 4
    
    # Optimization strategy
    use_layer_wise_decay: bool = True
    discriminative_learning: bool = True
    adaptive_learning_rate: bool = False
    
    # Regularization
    dropout_rate: Optional[float] = None
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Evaluation
    eval_strategy: str = "epoch"  # steps, epoch
    eval_steps: int = 500
    save_strategy: str = "epoch"
    save_steps: int = 500
    
    # Logging
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Miscellaneous
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True


class LayerFreezer:
    """Utility class for freezing and unfreezing model layers."""
    
    @staticmethod
    def freeze_parameters(model: nn.Module, layer_patterns: List[str]) -> int:
        """Freeze parameters matching the given layer patterns."""
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in layer_patterns):
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"Frozen {frozen_count} parameters matching patterns: {layer_patterns}")
        return frozen_count
    
    @staticmethod
    def unfreeze_parameters(model: nn.Module, layer_patterns: List[str]) -> int:
        """Unfreeze parameters matching the given layer patterns."""
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in layer_patterns):
                param.requires_grad = True
                unfrozen_count += 1
        logger.info(f"Unfrozen {unfrozen_count} parameters matching patterns: {layer_patterns}")
        return unfrozen_count
    
    @staticmethod
    def freeze_backbone(model: nn.Module) -> int:
        """Freeze the backbone while keeping task heads trainable."""
        backbone_patterns = [
            "hyena_glt.encoder",
            "hyena_glt.hyena_blocks",
            "hyena_glt.token_merger",
            "hyena_glt.decoder"
        ]
        return LayerFreezer.freeze_parameters(model, backbone_patterns)
    
    @staticmethod
    def get_layer_groups(model: nn.Module) -> Dict[str, List[str]]:
        """Get layer groups for discriminative learning rates."""
        layer_groups = {
            "embeddings": [],
            "encoder": [],
            "hyena_blocks": [],
            "token_merger": [],
            "decoder": [],
            "heads": []
        }
        
        for name, _ in model.named_parameters():
            if "embedding" in name:
                layer_groups["embeddings"].append(name)
            elif "encoder" in name:
                layer_groups["encoder"].append(name)
            elif "hyena_blocks" in name:
                layer_groups["hyena_blocks"].append(name)
            elif "token_merger" in name:
                layer_groups["token_merger"].append(name)
            elif "decoder" in name:
                layer_groups["decoder"].append(name)
            else:
                layer_groups["heads"].append(name)
        
        return layer_groups


class ModelAdapter:
    """Utility class for adapting pre-trained models to new tasks."""
    
    @staticmethod
    def adapt_model_for_task(
        model: HyenaGLT,
        task_type: str,
        num_labels: Optional[int] = None,
        config: Optional[HyenaGLTConfig] = None
    ) -> nn.Module:
        """Adapt a pre-trained model for a specific task."""
        
        if config is None:
            config = model.config
        
        # Update config for new task
        if num_labels is not None:
            config.num_labels = num_labels
        
        if task_type == "sequence_classification":
            adapted_model = HyenaGLTForSequenceClassification(config)
            adapted_model.hyena_glt = model
        elif task_type == "token_classification":
            adapted_model = HyenaGLTForTokenClassification(config)
            adapted_model.hyena_glt = model
        elif task_type == "sequence_generation":
            adapted_model = HyenaGLTForSequenceGeneration(config)
            adapted_model.hyena_glt = model
        elif task_type == "multitask":
            adapted_model = HyenaGLTForMultiTask(config)
            adapted_model.hyena_glt = model
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        logger.info(f"Adapted model for task: {task_type}")
        return adapted_model
    
    @staticmethod
    def resize_token_embeddings(model: nn.Module, new_vocab_size: int):
        """Resize token embeddings for new vocabulary."""
        if hasattr(model, 'hyena_glt'):
            base_model = model.hyena_glt
        else:
            base_model = model
        
        old_embeddings = base_model.embeddings.word_embeddings
        old_vocab_size, embedding_dim = old_embeddings.weight.shape
        
        if new_vocab_size == old_vocab_size:
            return
        
        # Create new embeddings
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        
        # Copy old weights
        if new_vocab_size > old_vocab_size:
            new_embeddings.weight.data[:old_vocab_size] = old_embeddings.weight.data
            # Initialize new tokens with mean of existing embeddings
            new_embeddings.weight.data[old_vocab_size:] = old_embeddings.weight.data.mean(0)
        else:
            new_embeddings.weight.data = old_embeddings.weight.data[:new_vocab_size]
        
        base_model.embeddings.word_embeddings = new_embeddings
        logger.info(f"Resized token embeddings from {old_vocab_size} to {new_vocab_size}")


class FineTuner:
    """Main class for fine-tuning Hyena-GLT models."""
    
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.model = None
        self.trainer = None
        
    def load_pretrained_model(self) -> nn.Module:
        """Load pre-trained model."""
        logger.info(f"Loading pre-trained model from {self.config.pretrained_model_path}")
        
        # Load the base model
        checkpoint = torch.load(self.config.pretrained_model_path, map_location="cpu")
        
        if "config" in checkpoint:
            model_config = HyenaGLTConfig(**checkpoint["config"])
        else:
            # Try to load config from separate file
            config_path = Path(self.config.pretrained_model_path).parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    model_config = HyenaGLTConfig(**json.load(f))
            else:
                raise ValueError("Could not find model configuration")
        
        # Create base model
        base_model = HyenaGLT(model_config)
        
        # Load weights
        if "model_state_dict" in checkpoint:
            base_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            base_model.load_state_dict(checkpoint)
        
        # Adapt for task
        self.model = ModelAdapter.adapt_model_for_task(
            base_model,
            self.config.task_type,
            self.config.num_labels,
            model_config
        )
        
        return self.model
    
    def setup_model_for_finetuning(self):
        """Setup model for fine-tuning with freezing and other configurations."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained_model() first.")
        
        # Apply freezing strategy
        if self.config.freeze_backbone:
            LayerFreezer.freeze_backbone(self.model)
        
        if self.config.freeze_layers:
            LayerFreezer.freeze_parameters(self.model, self.config.freeze_layers)
        
        if self.config.unfreeze_layers:
            LayerFreezer.unfreeze_parameters(self.model, self.config.unfreeze_layers)
        
        # Apply dropout if specified
        if self.config.dropout_rate is not None:
            self._set_dropout_rate(self.config.dropout_rate)
        
        logger.info("Model setup for fine-tuning completed")
    
    def _set_dropout_rate(self, dropout_rate: float):
        """Set dropout rate for all dropout layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
    
    def create_optimizer(self) -> Optimizer:
        """Create optimizer with optional layer-wise learning rate decay."""
        if not self.config.use_layer_wise_decay:
            # Standard optimizer
            return AdamWWithScheduler(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Layer-wise decay optimizer
        layer_groups = LayerFreezer.get_layer_groups(self.model)
        
        return LayerWiseDecayOptimizer(
            self.model,
            base_lr=self.config.learning_rate,
            layer_decay=self.config.layer_wise_lr_decay,
            weight_decay=self.config.weight_decay,
            layer_groups=layer_groups
        )
    
    def create_training_config(self, train_dataset: GenomicDataset, eval_dataset: Optional[GenomicDataset] = None) -> TrainingConfig:
        """Create training configuration for the trainer."""
        total_steps = len(train_dataset) // self.config.batch_size * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        return TrainingConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            evaluation_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_threshold=self.config.early_stopping_threshold,
            label_smoothing_factor=self.config.label_smoothing,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to=self.config.report_to,
            seed=self.config.seed
        )
    
    def fine_tune(
        self,
        train_dataset: GenomicDataset,
        eval_dataset: Optional[GenomicDataset] = None,
        compute_metrics: Optional[Callable] = None
    ):
        """Fine-tune the model on the given dataset."""
        
        # Load and setup model
        if self.model is None:
            self.load_pretrained_model()
        self.setup_model_for_finetuning()
        
        # Create optimizer
        optimizer = self.create_optimizer()
        
        # Create training config
        training_config = self.create_training_config(train_dataset, eval_dataset)
        
        # Create trainer
        self.trainer = HyenaGLTTrainer(
            model=self.model,
            config=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            compute_metrics=compute_metrics
        )
        
        # Start training
        logger.info("Starting fine-tuning...")
        self.trainer.train()
        
        # Save final model
        self.trainer.save_model()
        logger.info(f"Fine-tuning completed. Model saved to {self.config.output_dir}")
        
        return self.trainer
    
    def evaluate(self, eval_dataset: GenomicDataset) -> Dict[str, float]:
        """Evaluate the fine-tuned model."""
        if self.trainer is None:
            raise ValueError("Model not fine-tuned. Call fine_tune() first.")
        
        return self.trainer.evaluate(eval_dataset)
    
    def save_config(self):
        """Save fine-tuning configuration."""
        config_path = Path(self.config.output_dir) / "finetuning_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Fine-tuning config saved to {config_path}")


class TaskSpecificFineTuner:
    """Task-specific fine-tuning utilities."""
    
    @staticmethod
    def create_sequence_classification_config(
        pretrained_model_path: str,
        output_dir: str,
        num_labels: int,
        **kwargs
    ) -> FinetuningConfig:
        """Create configuration for sequence classification fine-tuning."""
        return FinetuningConfig(
            pretrained_model_path=pretrained_model_path,
            output_dir=output_dir,
            task_type="sequence_classification",
            num_labels=num_labels,
            learning_rate=2e-5,
            num_epochs=5,
            batch_size=16,
            **kwargs
        )
    
    @staticmethod
    def create_token_classification_config(
        pretrained_model_path: str,
        output_dir: str,
        num_labels: int,
        **kwargs
    ) -> FinetuningConfig:
        """Create configuration for token classification fine-tuning."""
        return FinetuningConfig(
            pretrained_model_path=pretrained_model_path,
            output_dir=output_dir,
            task_type="token_classification",
            num_labels=num_labels,
            learning_rate=3e-5,
            num_epochs=3,
            batch_size=8,
            use_layer_wise_decay=True,
            layer_wise_lr_decay=0.9,
            **kwargs
        )
    
    @staticmethod
    def create_generation_config(
        pretrained_model_path: str,
        output_dir: str,
        **kwargs
    ) -> FinetuningConfig:
        """Create configuration for sequence generation fine-tuning."""
        return FinetuningConfig(
            pretrained_model_path=pretrained_model_path,
            output_dir=output_dir,
            task_type="sequence_generation",
            learning_rate=1e-5,
            num_epochs=3,
            batch_size=4,
            gradient_accumulation_steps=4,
            **kwargs
        )
    
    @staticmethod
    def create_domain_adaptation_config(
        pretrained_model_path: str,
        output_dir: str,
        task_type: str = "sequence_classification",
        **kwargs
    ) -> FinetuningConfig:
        """Create configuration for domain adaptation."""
        return FinetuningConfig(
            pretrained_model_path=pretrained_model_path,
            output_dir=output_dir,
            task_type=task_type,
            freeze_backbone=False,
            use_layer_wise_decay=True,
            layer_wise_lr_decay=0.95,
            learning_rate=1e-5,
            num_epochs=10,
            warmup_ratio=0.2,
            **kwargs
        )


# Convenience functions for common fine-tuning scenarios
def finetune_for_sequence_classification(
    pretrained_model_path: str,
    train_dataset: GenomicDataset,
    eval_dataset: Optional[GenomicDataset],
    output_dir: str,
    num_labels: int,
    **kwargs
):
    """Fine-tune model for sequence classification."""
    config = TaskSpecificFineTuner.create_sequence_classification_config(
        pretrained_model_path, output_dir, num_labels, **kwargs
    )
    
    finetuner = FineTuner(config)
    
    # Create metrics function
    def compute_metrics(eval_pred):
        metrics = GenomicMetrics()
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        return metrics.compute_classification_metrics(predictions, labels)
    
    return finetuner.fine_tune(train_dataset, eval_dataset, compute_metrics)


def finetune_for_token_classification(
    pretrained_model_path: str,
    train_dataset: GenomicDataset,
    eval_dataset: Optional[GenomicDataset],
    output_dir: str,
    num_labels: int,
    **kwargs
):
    """Fine-tune model for token classification."""
    config = TaskSpecificFineTuner.create_token_classification_config(
        pretrained_model_path, output_dir, num_labels, **kwargs
    )
    
    finetuner = FineTuner(config)
    
    # Create metrics function for token classification
    def compute_metrics(eval_pred):
        metrics = GenomicMetrics()
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        # Flatten for token-level metrics
        predictions = predictions.reshape(-1)
        labels = labels.reshape(-1)
        
        # Remove ignored tokens (usually -100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        return metrics.compute_classification_metrics(predictions, labels)
    
    return finetuner.fine_tune(train_dataset, eval_dataset, compute_metrics)


def finetune_for_generation(
    pretrained_model_path: str,
    train_dataset: GenomicDataset,
    eval_dataset: Optional[GenomicDataset],
    output_dir: str,
    **kwargs
):
    """Fine-tune model for sequence generation."""
    config = TaskSpecificFineTuner.create_generation_config(
        pretrained_model_path, output_dir, **kwargs
    )
    
    finetuner = FineTuner(config)
    
    # Create metrics function for generation
    def compute_metrics(eval_pred):
        metrics = GenomicMetrics()
        predictions, labels = eval_pred
        return {"perplexity": metrics.compute_perplexity(predictions, labels)}
    
    return finetuner.fine_tune(train_dataset, eval_dataset, compute_metrics)
