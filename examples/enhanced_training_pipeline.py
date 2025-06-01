#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Hyena-GLT Models

This example demonstrates advanced training workflows including:
- Multi-modal genomic learning (DNA, RNA, protein integration)
- Interpretability tools and attention analysis
- Advanced curriculum learning strategies
- Real-time monitoring and visualization
- Production-ready training workflows

Author: Hyena-GLT Development Team
Version: 1.1.0
"""

import argparse
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings('ignore')

# Hyena-GLT core imports
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data import (
    DNATokenizer,
    ProteinTokenizer,
    RNATokenizer,
)
from hyena_glt.model import (
    HyenaGLTForSequenceClassification,
)
from hyena_glt.training import (
    CurriculumLearning,
    HyenaGLTTrainer,
    TrainingConfig,
)
from hyena_glt.utils.performance import ProfilerContext

# Note: Using matplotlib/seaborn directly instead of custom visualization utils
# from examples.utils.visualization_utils import (
#     plot_training_history,
#     plot_attention_heatmap,
#     plot_confusion_matrix
# )


@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration with advanced features."""

    # Basic training
    output_dir: str = "./enhanced_training_outputs"
    experiment_name: str = "enhanced_hyena_glt"
    seed: int = 42

    # Training parameters
    num_epochs: int = 1
    max_steps: int = None
    batch_size: int = 2

    # Multi-modal configuration
    use_multimodal: bool = True
    modalities: list[str] = None  # ['dna', 'rna', 'protein']
    modality_weights: dict[str, float] = None

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_strategy: str = "length_based"  # length_based, complexity_based, difficulty_based
    curriculum_stages: int = 4

    # Interpretability
    enable_interpretability: bool = True
    attention_analysis: bool = True
    gradient_analysis: bool = True

    # Advanced monitoring
    real_time_plotting: bool = True
    profile_performance: bool = True
    save_attention_maps: bool = True

    # Model selection
    model_size: str = "base"  # small, base, large

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ['dna', 'rna', 'protein']
        if self.modality_weights is None:
            self.modality_weights = {'dna': 0.5, 'rna': 0.3, 'protein': 0.2}


class MultiModalGenomicDataset(Dataset):
    """Enhanced dataset for multi-modal genomic data."""

    def __init__(
        self,
        dna_sequences: list[str],
        rna_sequences: list[str] | None = None,
        protein_sequences: list[str] | None = None,
        labels: list[int] = None,
        tokenizers: dict[str, Any] = None,
        max_lengths: dict[str, int] = None
    ):
        self.dna_sequences = dna_sequences
        self.rna_sequences = rna_sequences or dna_sequences
        self.protein_sequences = protein_sequences or dna_sequences
        self.labels = labels or [0] * len(dna_sequences)
        self.tokenizers = tokenizers
        self.max_lengths = max_lengths or {'dna': 512, 'rna': 512, 'protein': 512}

        assert len(self.dna_sequences) == len(self.labels)

    def __len__(self):
        return len(self.dna_sequences)

    def __getitem__(self, idx):
        item = {
            'dna': self.dna_sequences[idx],
            'rna': self.rna_sequences[idx],
            'protein': self.protein_sequences[idx],
            'label': self.labels[idx]
        }

        # Tokenize if tokenizers provided
        if self.tokenizers:
            # Create a copy of items to avoid "dictionary changed size during iteration" error
            for modality, sequence in list(item.items()):
                if modality in self.tokenizers and modality != 'label':
                    tokens = self.tokenizers[modality].encode_plus(
                        sequence,
                        max_length=self.max_lengths[modality],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    item[f'{modality}_input_ids'] = tokens['input_ids'].squeeze(0)
                    item[f'{modality}_attention_mask'] = tokens['attention_mask'].squeeze(0)

        return item


class EnhancedTrainingPipeline:
    """Enhanced training pipeline with advanced features."""

    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        self.set_random_seeds()

        # Initialize components
        self.tokenizers = {}
        self.models = {}
        self.trainers = {}
        self.metrics_history = {}

        # Performance monitoring
        if config.profile_performance:
            self.profiler = ProfilerContext()

        self.logger.info(f"Enhanced Training Pipeline initialized: {config.experiment_name}")

    def setup_logging(self):
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.output_dir}/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create output directories."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(f'{self.config.output_dir}/models', exist_ok=True)
        os.makedirs(f'{self.config.output_dir}/plots', exist_ok=True)
        os.makedirs(f'{self.config.output_dir}/attention_maps', exist_ok=True)
        os.makedirs(f'{self.config.output_dir}/logs', exist_ok=True)

    def set_random_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

    def initialize_tokenizers(self) -> dict[str, Any]:
        """Initialize tokenizers for different modalities."""
        self.logger.info("Initializing tokenizers...")

        tokenizers = {}
        for modality in self.config.modalities:
            if modality == 'dna':
                tokenizers['dna'] = DNATokenizer(k=3)
            elif modality == 'rna':
                tokenizers['rna'] = RNATokenizer(k=3)
            elif modality == 'protein':
                tokenizers['protein'] = ProteinTokenizer()
            else:
                raise ValueError(f"Unknown modality: {modality}")

        self.tokenizers = tokenizers
        return tokenizers

    def create_model_config(self, task_type: str = "classification") -> HyenaGLTConfig:
        """Create model configuration based on size and task."""
        size_configs = {
            'small': {
                'hidden_size': 256,
                'num_hyena_layers': 6,
                'num_attention_heads': 8,
                'intermediate_size': 1024
            },
            'base': {
                'hidden_size': 512,
                'num_hyena_layers': 12,
                'num_attention_heads': 16,
                'intermediate_size': 2048
            },
            'large': {
                'hidden_size': 768,
                'num_hyena_layers': 24,
                'num_attention_heads': 32,
                'intermediate_size': 3072
            }
        }

        config_params = size_configs[self.config.model_size]

        config = HyenaGLTConfig(
            # Architecture
            **config_params,

            # Genomic-specific
            genomic_vocab_size=4096,
            max_position_embeddings=2048,

            # Task-specific
            num_labels=2 if task_type == "classification" else None,
            task_type=task_type,

            # BLT-specific
            local_encoder_layers=2,
            local_decoder_layers=2,
            patch_size=4,

            # Hyena-specific
            hyena_order=3,
            hyena_filter_size=64,
            use_conv_bias=True,

            # Training efficiency
            gradient_checkpointing=True,

            # Multi-modal support
            multi_modal=self.config.use_multimodal,
            modalities=self.config.modalities if self.config.use_multimodal else None
        )

        return config

    def create_synthetic_data(self, num_samples: int = 1000) -> tuple[list[str], list[str], list[str], list[int]]:
        """Create synthetic multi-modal genomic data."""
        self.logger.info(f"Generating {num_samples} synthetic multi-modal samples...")

        nucleotides = ['A', 'T', 'G', 'C']
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

        dna_sequences = []
        rna_sequences = []
        protein_sequences = []
        labels = []

        for _i in range(num_samples):
            # Generate DNA sequence
            dna_seq = ''.join(np.random.choice(nucleotides, size=np.random.randint(100, 500)))
            dna_sequences.append(dna_seq)

            # Generate corresponding RNA sequence (DNA -> RNA transcription)
            rna_seq = dna_seq.replace('T', 'U')
            rna_sequences.append(rna_seq)

            # Generate protein sequence
            protein_seq = ''.join(np.random.choice(amino_acids, size=len(dna_seq)//3))
            protein_sequences.append(protein_seq)

            # Generate label based on GC content and sequence features
            gc_content = (dna_seq.count('G') + dna_seq.count('C')) / len(dna_seq)
            has_start_codon = 'ATG' in dna_seq
            label = 1 if (gc_content > 0.5 and has_start_codon) else 0
            labels.append(label)

        return dna_sequences, rna_sequences, protein_sequences, labels

    def setup_curriculum_learning(self, dataset: Dataset) -> CurriculumLearning:
        """Setup curriculum learning strategy."""
        if not self.config.use_curriculum:
            return None

        self.logger.info(f"Setting up {self.config.curriculum_strategy} curriculum learning...")

        # Import difficulty measures
        from hyena_glt.training.curriculum import (
            SequenceLengthDifficulty, 
            GenomicComplexityDifficulty
        )

        # Create difficulty measures based on strategy
        if self.config.curriculum_strategy == "length_based":
            difficulty_measures = [SequenceLengthDifficulty(normalize=True, max_length=1024)]
        elif self.config.curriculum_strategy == "complexity_based":
            difficulty_measures = [GenomicComplexityDifficulty()]
        else:
            # Use both for difficulty-based strategy
            difficulty_measures = [
                SequenceLengthDifficulty(normalize=True, max_length=1024),
                GenomicComplexityDifficulty()
            ]

        # Create curriculum with proper parameters
        curriculum = CurriculumLearning(
            difficulty_measures=difficulty_measures,
            curriculum_strategy="linear",  # Use linear progression
            start_difficulty=0.1,
            end_difficulty=1.0,
            curriculum_steps=50,  # Minimal steps for very fast testing
        )

        return curriculum

    def create_training_config(self) -> TrainingConfig:
        """Create comprehensive training configuration."""
        return TrainingConfig(
            # Basic parameters - Now configurable via command line
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,

            # Optimization
            learning_rate=1e-4,
            weight_decay=0.01,
            optimizer_type="adamw",
            scheduler_type="cosine",
            warmup_steps=50,  # Reduced from 200

            # Logging and evaluation - More frequent for shorter runs
            eval_steps=20,  # Reduced from 50
            save_steps=50,  # Reduced from 100
            logging_steps=10,  # Reduced from 25
            log_level="INFO",

            # Checkpointing
            output_dir=f"{self.config.output_dir}/models",
            save_total_limit=3,
            save_best_only=True,

            # Mixed precision
            fp16=torch.cuda.is_available(),

            # Multi-task learning
            multi_task=self.config.use_multimodal,

            # Curriculum learning
            curriculum_learning=self.config.use_curriculum,

            # Experiment tracking
            use_wandb=False,

            # Early stopping
            early_stopping=True,
            early_stopping_patience=3,
            early_stopping_metric="eval_loss"
        )

    def analyze_attention_patterns(self, model: nn.Module, sample_data: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Analyze attention patterns for interpretability."""
        if not self.config.attention_analysis:
            return {}

        self.logger.info("Analyzing attention patterns...")

        try:
            # Import Hyena-specific attention analyzer
            from hyena_glt.interpretability.attention_analysis import HyenaAttentionAnalyzer
            
            # Create analyzer for Hyena models
            analyzer = HyenaAttentionAnalyzer(model)
            
            # Extract input_ids from sample_data
            input_ids = sample_data.get('input_ids')
            if input_ids is None:
                self.logger.warning("No input_ids found in sample_data for attention analysis")
                return {}
            
            # Extract Hyena convolution patterns (attention-like patterns)
            patterns = analyzer.extract_hyena_patterns(input_ids)
            
            if patterns:
                # Analyze positional patterns 
                positional_analysis = analyzer.analyze_positional_patterns(patterns)
                
                # Create summary analysis
                analysis = {
                    'num_patterns': len(patterns),
                    'pattern_names': list(patterns.keys()),
                    'positional_analysis': positional_analysis,
                    'avg_local_attention': np.mean([stats.get('local_attention', 0.0) 
                                                   for stats in positional_analysis.values()]),
                    'avg_long_range_attention': np.mean([stats.get('long_range_attention', 0.0) 
                                                        for stats in positional_analysis.values()]),
                    'avg_periodicity': np.mean([stats.get('periodicity', 0.0) 
                                               for stats in positional_analysis.values()])
                }
                
                self.logger.info(f"Successfully analyzed {len(patterns)} Hyena convolution patterns")
                return analysis
            else:
                self.logger.warning("No Hyena patterns found - model may not have convolution layers")
                return {}
                
        except ImportError as e:
            self.logger.warning(f"Could not import HyenaAttentionAnalyzer: {e}")
            return {}
        except Exception as e:
            self.logger.warning(f"Attention analysis failed: {e}")
            return {}

    def visualize_training_progress(self, metrics_history: dict[str, list[float]]):
        """Create real-time training visualization."""
        if not self.config.real_time_plotting:
            return

        self.logger.info("Creating training visualizations...")

        # Plot training metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        if 'train_loss' in metrics_history and 'eval_loss' in metrics_history:
            axes[0, 0].plot(metrics_history['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(metrics_history['eval_loss'], label='Eval Loss', color='red')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # Accuracy curves
        if 'train_accuracy' in metrics_history and 'eval_accuracy' in metrics_history:
            axes[0, 1].plot(metrics_history['train_accuracy'], label='Train Accuracy', color='green')
            axes[0, 1].plot(metrics_history['eval_accuracy'], label='Eval Accuracy', color='orange')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # Learning rate
        if 'learning_rate' in metrics_history:
            axes[1, 0].plot(metrics_history['learning_rate'], color='purple')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)

        # Memory usage (if available)
        if 'memory_usage' in metrics_history:
            axes[1, 1].plot(metrics_history['memory_usage'], color='brown')
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.config.output_dir}/plots/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()

    def train_with_step_control(self, trainer: HyenaGLTTrainer) -> dict[str, Any]:
        """Custom training method that respects max_steps parameter."""
        if self.config.max_steps is None:
            # Use default trainer if no max_steps specified
            return trainer.train()
        
        self.logger.info(f"Training with max_steps control: {self.config.max_steps} steps")
        
        # Start training with step control
        trainer.logger.info("Starting training...")

        if not trainer.train_dataloader:
            raise ValueError("No training dataloader provided")

        trainer.model.train()

        # Training state tracking
        global_step = 0
        total_loss = 0.0
        
        # Progress tracking
        from tqdm.auto import tqdm
        progress_bar = tqdm(total=self.config.max_steps, desc="Training (Step Control)")

        trainer.global_step = 0
        trainer.epoch = 0

        # Training loop with step control
        for epoch in range(trainer.config.num_epochs):
            trainer.epoch = epoch
            epoch_loss = 0.0

            for step, batch in enumerate(trainer.train_dataloader):
                # Check if we've reached max_steps
                if global_step >= self.config.max_steps:
                    self.logger.info(f"Reached max_steps ({self.config.max_steps}). Stopping training.")
                    break

                # Forward pass using trainer's method
                loss, metrics = trainer._training_step(batch)
                epoch_loss += loss.item()
                total_loss += loss.item()

                # Backward pass
                if trainer.config.fp16 and trainer.scaler:
                    trainer.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (step + 1) % trainer.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if trainer.config.max_grad_norm > 0:
                        if trainer.config.fp16 and trainer.scaler:
                            trainer.scaler.unscale_(trainer.optimizer)

                        torch.nn.utils.clip_grad_norm_(
                            trainer.model.parameters(), trainer.config.max_grad_norm
                        )

                    # Optimizer step
                    if trainer.config.fp16 and trainer.scaler:
                        trainer.scaler.step(trainer.optimizer)
                        trainer.scaler.update()
                    else:
                        trainer.optimizer.step()

                    if trainer.scheduler:
                        trainer.scheduler.step()

                    trainer.optimizer.zero_grad()
                    global_step += 1
                    trainer.global_step = global_step

                # Logging
                if global_step % trainer.config.logging_steps == 0:
                    avg_loss = total_loss / global_step if global_step > 0 else 0.0
                    trainer._log_metrics({"train_loss": loss.item(), "avg_loss": avg_loss, **metrics})

                # Evaluation
                if (
                    global_step % trainer.config.eval_steps == 0
                    and trainer.eval_dataloader
                ):
                    eval_metrics = trainer.evaluate()
                    trainer._log_metrics(eval_metrics, prefix="eval")

                    # Early stopping check
                    if trainer.config.early_stopping:
                        trainer._check_early_stopping(eval_metrics)

                    trainer.model.train()  # Return to training mode

                # Checkpointing
                if global_step % trainer.config.save_steps == 0:
                    trainer._save_checkpoint()

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item(), "step": global_step})

                # Early stopping check
                if hasattr(trainer, 'early_stopping_counter') and trainer.early_stopping_counter >= trainer.config.early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break

            # Check if we've reached max_steps (break outer loop)
            if global_step >= self.config.max_steps:
                break

            # End of epoch
            avg_epoch_loss = epoch_loss / len(trainer.train_dataloader)
            trainer.logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")

            if hasattr(trainer, 'early_stopping_counter') and trainer.early_stopping_counter >= trainer.config.early_stopping_patience:
                break

        progress_bar.close()

        # Final evaluation
        if trainer.eval_dataloader:
            final_metrics = trainer.evaluate()
            trainer.logger.info(f"Final evaluation metrics: {final_metrics}")

        # Save final checkpoint
        trainer._save_checkpoint()

        trainer.logger.info(f"Training completed! Stopped at step {global_step}")

        return {
            "final_step": global_step,
            "final_epoch": trainer.epoch,
            "total_loss": total_loss,
            "avg_loss": total_loss / global_step if global_step > 0 else 0.0,
        }

    def run_enhanced_training(self):
        """Run the complete enhanced training pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING ENHANCED HYENA-GLT TRAINING PIPELINE")
        self.logger.info("=" * 60)

        # Store performance metrics separately
        performance_metrics = {}
        attention_analysis = {}

        try:
            # Performance monitoring context
            with self.profiler if self.config.profile_performance else nullcontext():

                # 1. Initialize tokenizers
                self.initialize_tokenizers()

                # 2. Create synthetic data - Reduced size for faster testing
                dna_seqs, rna_seqs, protein_seqs, labels = self.create_synthetic_data(200)  # Reduced from 1000

                # 3. Create dataset
                dataset = MultiModalGenomicDataset(
                    dna_sequences=dna_seqs,
                    rna_sequences=rna_seqs,
                    protein_sequences=protein_seqs,
                    labels=labels,
                    tokenizers=self.tokenizers
                )

                # 4. Split dataset
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

                # 5. Create model and config
                model_config = self.create_model_config("classification")
                model = HyenaGLTForSequenceClassification(model_config)

                # 6. Setup curriculum learning
                self.setup_curriculum_learning(train_dataset)

                # 7. Create training configuration with configurable parameters
                training_config = self.create_training_config()

                # 8. Create custom data loaders with key mapping and configurable batch size
                def collate_fn(batch):
                    """Custom collate function to map dataset keys to model input format."""
                    # Use DNA as primary modality for input_ids
                    collated = {}
                    
                    # Map dna_input_ids to input_ids (primary input)
                    if 'dna_input_ids' in batch[0]:
                        collated['input_ids'] = torch.stack([item['dna_input_ids'] for item in batch])
                        collated['attention_mask'] = torch.stack([item['dna_attention_mask'] for item in batch])
                    
                    # Include labels
                    if 'label' in batch[0]:
                        collated['labels'] = torch.tensor([item['label'] for item in batch])
                    
                    return collated

                train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=self.config.batch_size, 
                    shuffle=True, 
                    collate_fn=collate_fn
                )
                val_dataloader = DataLoader(
                    val_dataset, 
                    batch_size=self.config.batch_size, 
                    shuffle=False, 
                    collate_fn=collate_fn
                )

                # 9. Initialize trainer
                trainer = HyenaGLTTrainer(
                    model=model,
                    config=training_config,
                    train_dataloader=train_dataloader,
                    eval_dataloader=val_dataloader,
                    tokenizer=self.tokenizers.get('dna'),  # Primary tokenizer
                )

                # 10. Train model with enhanced monitoring
                self.logger.info("Starting enhanced training...")

                # Use custom training method that supports max_steps
                training_results = self.train_with_step_control(trainer)

                # 11. Analyze model interpretability
                if self.config.enable_interpretability and val_dataset:
                    # Create a sample with correct key mapping and move to device
                    sample_batch = collate_fn([val_dataset[0]])
                    device = next(model.parameters()).device
                    sample_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample_batch.items()}
                    attention_analysis = self.analyze_attention_patterns(model, sample_batch)
                    self.logger.info(f"Attention analysis: {attention_analysis}")

                # 12. Save final model and results
                model.save_pretrained(f"{self.config.output_dir}/models/final_model")

            # Get performance metrics after profiler context has completed
            if self.config.profile_performance and hasattr(self, 'profiler'):
                try:
                    performance_metrics = self.profiler.get_metrics()
                except RuntimeError as e:
                    self.logger.warning(f"Could not get profiler metrics: {e}")
                    performance_metrics = {"error": "Profiler context not completed properly"}

            # 13. Generate comprehensive report
            self.generate_training_report(
                training_results, 
                attention_analysis,
                performance_metrics
            )

            self.logger.info("Enhanced training pipeline completed successfully!")

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise

    def generate_training_report(self, training_results: dict[str, Any], attention_analysis: dict[str, Any], performance_metrics: dict[str, Any] = None):
        """Generate comprehensive training report."""
        self.logger.info("Generating training report...")

        # Use provided performance metrics or empty dict
        if performance_metrics is None:
            performance_metrics = {}

        report = {
            'experiment_name': self.config.experiment_name,
            'model_size': self.config.model_size,
            'config': asdict(self.config),
            'training_results': training_results,
            'attention_analysis': attention_analysis,
            'performance_profile': performance_metrics
        }

        # Save report
        import json
        with open(f"{self.config.output_dir}/training_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Create markdown report
        self.create_markdown_report(report)

    def create_markdown_report(self, report: dict[str, Any]):
        """Create markdown training report."""
        markdown_content = f"""# Enhanced Training Report: {report['experiment_name']}

## Configuration
- **Model Size**: {report['model_size']}
- **Multi-modal**: {report['config']['use_multimodal']}
- **Curriculum Learning**: {report['config']['use_curriculum']}
- **Interpretability**: {report['config']['enable_interpretability']}

## Training Results
- **Final Loss**: {report['training_results'].get('final_loss', 'N/A')}
- **Best Accuracy**: {report['training_results'].get('best_accuracy', 'N/A')}
- **Training Time**: {report['training_results'].get('training_time', 'N/A')}

## Attention Analysis
{f"- **Number of Layers**: {report['attention_analysis'].get('num_layers', 'N/A')}" if report['attention_analysis'] else "Not performed"}
{f"- **Number of Heads**: {report['attention_analysis'].get('num_heads', 'N/A')}" if report['attention_analysis'] else ""}
{f"- **Average Attention Entropy**: {np.mean(report['attention_analysis'].get('attention_entropy', [])) if report['attention_analysis'].get('attention_entropy') else 'N/A'}" if report['attention_analysis'] else ""}

## Performance Profile
{f"- **Peak Memory Usage**: {report['performance_profile'].get('peak_memory', 'N/A')}" if report.get('performance_profile') else "Not profiled"}
{f"- **Average GPU Utilization**: {report['performance_profile'].get('avg_gpu_util', 'N/A')}" if report.get('performance_profile') else ""}

Generated: {pd.Timestamp.now()}
"""

        with open(f"{self.config.output_dir}/training_report.md", 'w') as f:
            f.write(markdown_content)


def nullcontext():
    """Null context manager for when profiling is disabled."""
    from contextlib import nullcontext as _nullcontext
    return _nullcontext()


def main():
    """Main function to run enhanced training pipeline."""
    parser = argparse.ArgumentParser(description="Enhanced Hyena-GLT Training Pipeline")
    parser.add_argument("--output-dir", default="./enhanced_training_outputs", help="Output directory")
    parser.add_argument("--experiment-name", default="enhanced_hyena_glt", help="Experiment name")
    parser.add_argument("--model-size", choices=['small', 'base', 'large'], default='base', help="Model size")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of training steps (overrides epochs)")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--disable-multimodal", action='store_true', help="Disable multi-modal training")
    parser.add_argument("--disable-curriculum", action='store_true', help="Disable curriculum learning")
    parser.add_argument("--disable-interpretability", action='store_true', help="Disable interpretability analysis")
    parser.add_argument("--curriculum-strategy", choices=['length_based', 'complexity_based', 'difficulty_based'],
                        default='length_based', help="Curriculum learning strategy")

    args = parser.parse_args()

    # Create enhanced training configuration
    config = EnhancedTrainingConfig(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        model_size=args.model_size,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        use_multimodal=not args.disable_multimodal,
        use_curriculum=not args.disable_curriculum,
        enable_interpretability=not args.disable_interpretability,
        curriculum_strategy=args.curriculum_strategy
    )

    # Initialize and run pipeline
    pipeline = EnhancedTrainingPipeline(config)
    pipeline.run_enhanced_training()


if __name__ == "__main__":
    main()
