"""
Task-specific fine-tuning scripts for Hyena-GLT models.

This module provides ready-to-use scripts for fine-tuning Hyena-GLT models
on common genomic tasks with best practices and optimizations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import sys
from dataclasses import asdict

from .finetuning import (
    FineTuner,
    FinetuningConfig,
    TaskSpecificFineTuner,
    finetune_for_sequence_classification,
    finetune_for_token_classification,
    finetune_for_generation
)
from .pretrained import PretrainedModelManager, load_pretrained_model
from ..data.dataset import (
    GenomicDataset,
    SequenceClassificationDataset,
    TokenClassificationDataset,
    SequenceGenerationDataset
)
from ..data.tokenizer import DNATokenizer, RNATokenizer, ProteinTokenizer
from ..evaluation.metrics import MultiTaskEvaluator
from ..config import HyenaGLTConfig

logger = logging.getLogger(__name__)


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
            "splice_site"
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
            layer_wise_lr_decay=0.9
        )
    
    def create_dataset(self, data_path: str, tokenizer_type: str = "dna") -> TokenClassificationDataset:
        """Create dataset for genome annotation."""
        if tokenizer_type == "dna":
            tokenizer = DNATokenizer(k=6)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        
        return TokenClassificationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            label_names=self.labels
        )
    
    def fine_tune(self, train_data_path: str, eval_data_path: Optional[str] = None):
        """Fine-tune model for genome annotation."""
        train_dataset = self.create_dataset(train_data_path)
        eval_dataset = self.create_dataset(eval_data_path) if eval_data_path else None
        
        return finetune_for_token_classification(
            pretrained_model_path=self.pretrained_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=self.output_dir,
            num_labels=len(self.labels)
        )


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
            "pathogenic"
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
            early_stopping_patience=3
        )
    
    def create_dataset(self, data_path: str, tokenizer_type: str = "dna") -> SequenceClassificationDataset:
        """Create dataset for variant effect prediction."""
        if tokenizer_type == "dna":
            tokenizer = DNATokenizer(k=6)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        
        return SequenceClassificationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            label_names=self.labels
        )
    
    def fine_tune(self, train_data_path: str, eval_data_path: Optional[str] = None):
        """Fine-tune model for variant effect prediction."""
        train_dataset = self.create_dataset(train_data_path)
        eval_dataset = self.create_dataset(eval_data_path) if eval_data_path else None
        
        return finetune_for_sequence_classification(
            pretrained_model_path=self.pretrained_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=self.output_dir,
            num_labels=len(self.labels)
        )


class ProteinFunctionFineTuner:
    """Fine-tuner for protein function prediction tasks."""
    
    def __init__(self, pretrained_model_path: str, output_dir: str, function_type: str = "go"):
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
                "enzyme_regulator_activity"
            ]
        elif function_type == "ec":
            self.labels = [
                "oxidoreductase",
                "transferase",
                "hydrolase", 
                "lyase",
                "isomerase",
                "ligase",
                "translocase"
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
            layer_wise_lr_decay=0.95
        )
    
    def create_dataset(self, data_path: str) -> SequenceClassificationDataset:
        """Create dataset for protein function prediction."""
        tokenizer = ProteinTokenizer()
        
        return SequenceClassificationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            label_names=self.labels
        )
    
    def fine_tune(self, train_data_path: str, eval_data_path: Optional[str] = None):
        """Fine-tune model for protein function prediction."""
        train_dataset = self.create_dataset(train_data_path)
        eval_dataset = self.create_dataset(eval_data_path) if eval_data_path else None
        
        return finetune_for_sequence_classification(
            pretrained_model_path=self.pretrained_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=self.output_dir,
            num_labels=len(self.labels)
        )


class GenomeGenerationFineTuner:
    """Fine-tuner for genome sequence generation tasks."""
    
    def __init__(self, pretrained_model_path: str, output_dir: str, generation_type: str = "promoter"):
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
            weight_decay=0.01
        )
    
    def create_dataset(self, data_path: str, tokenizer_type: str = "dna") -> SequenceGenerationDataset:
        """Create dataset for sequence generation."""
        if tokenizer_type == "dna":
            tokenizer = DNATokenizer(k=6)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        
        return SequenceGenerationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            mask_probability=0.15
        )
    
    def fine_tune(self, train_data_path: str, eval_data_path: Optional[str] = None):
        """Fine-tune model for sequence generation."""
        train_dataset = self.create_dataset(train_data_path)
        eval_dataset = self.create_dataset(eval_data_path) if eval_data_path else None
        
        return finetune_for_generation(
            pretrained_model_path=self.pretrained_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=self.output_dir
        )


class DomainAdaptationFineTuner:
    """Fine-tuner for domain adaptation across species or sequence types."""
    
    def __init__(self, pretrained_model_path: str, output_dir: str, source_domain: str, target_domain: str):
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
            layer_wise_lr_decay=0.98
        )
    
    def fine_tune_gradual(self, target_data_path: str, eval_data_path: Optional[str] = None):
        """Perform gradual domain adaptation."""
        # This would implement gradual unfreezing and learning rate scheduling
        # for domain adaptation
        pass


def main():
    """Main function for command-line fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Hyena-GLT models for genomic tasks")
    
    # Required arguments
    parser.add_argument("--task", type=str, required=True,
                       choices=["genome_annotation", "variant_effect", "protein_function", "generation", "domain_adaptation"],
                       help="Type of fine-tuning task")
    parser.add_argument("--pretrained_model", type=str, required=True,
                       help="Path to pre-trained model or model name")
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for fine-tuned model")
    
    # Optional arguments
    parser.add_argument("--eval_data", type=str, default=None,
                       help="Path to evaluation data")
    parser.add_argument("--tokenizer_type", type=str, default="dna",
                       choices=["dna", "rna", "protein"],
                       help="Type of tokenizer to use")
    parser.add_argument("--function_type", type=str, default="go",
                       choices=["go", "ec"],
                       help="Type of protein function prediction (for protein_function task)")
    parser.add_argument("--generation_type", type=str, default="promoter",
                       help="Type of sequence generation (for generation task)")
    parser.add_argument("--source_domain", type=str, default=None,
                       help="Source domain for domain adaptation")
    parser.add_argument("--target_domain", type=str, default=None,
                       help="Target domain for domain adaptation")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (if not specified, task-specific default will be used)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Maximum sequence length")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        if args.task == "genome_annotation":
            finetuner = GenomeAnnotationFineTuner(pretrained_model_path, args.output_dir)
        elif args.task == "variant_effect":
            finetuner = VariantEffectFineTuner(pretrained_model_path, args.output_dir)
        elif args.task == "protein_function":
            finetuner = ProteinFunctionFineTuner(pretrained_model_path, args.output_dir, args.function_type)
        elif args.task == "generation":
            finetuner = GenomeGenerationFineTuner(pretrained_model_path, args.output_dir, args.generation_type)
        elif args.task == "domain_adaptation":
            if not args.source_domain or not args.target_domain:
                raise ValueError("Domain adaptation requires --source_domain and --target_domain")
            finetuner = DomainAdaptationFineTuner(
                pretrained_model_path, args.output_dir, args.source_domain, args.target_domain
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
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        logger.info("Fine-tuning completed successfully!")
        
        # Evaluate if eval data provided
        if args.eval_data and hasattr(trainer, 'evaluate'):
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            
            results_path = Path(args.output_dir) / "eval_results.json"
            with open(results_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {results_path}")
            for metric, value in eval_results.items():
                logger.info(f"{metric}: {value}")
    
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
