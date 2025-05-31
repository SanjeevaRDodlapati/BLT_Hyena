#!/usr/bin/env python3
"""
General fine-tuning script for Hyena-GLT models.

This script provides a unified interface for fine-tuning Hyena-GLT models
on various genomic tasks with command-line configuration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from hyena_glt.data.dataset import (
    SequenceClassificationDataset,
    SequenceGenerationDataset,
    TokenClassificationDataset,
)
from hyena_glt.data.tokenizer import DNATokenizer, ProteinTokenizer, RNATokenizer
from hyena_glt.training.finetuning import (
    FineTuner,
    FinetuningConfig,
)
from hyena_glt.training.metrics import GenomicMetrics
from hyena_glt.training.pretrained import PretrainedModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_tokenizer(tokenizer_type: str, k: int = 6):
    """Create tokenizer based on type."""
    if tokenizer_type == "dna":
        return DNATokenizer(k=k)
    elif tokenizer_type == "rna":
        return RNATokenizer(k=k)
    elif tokenizer_type == "protein":
        return ProteinTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def create_dataset(
    data_path: str, task_type: str, tokenizer, max_length: int, label_names=None
):
    """Create dataset based on task type."""
    if task_type == "sequence_classification":
        return SequenceClassificationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            label_names=label_names,
        )
    elif task_type == "token_classification":
        return TokenClassificationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            label_names=label_names,
        )
    elif task_type == "sequence_generation":
        return SequenceGenerationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            mask_probability=0.15,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Hyena-GLT models")

    # Required arguments
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=[
            "sequence_classification",
            "token_classification",
            "sequence_generation",
        ],
        help="Type of fine-tuning task",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Pre-trained model name or path",
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
        "--k", type=int, default=6, help="K-mer size for DNA/RNA tokenizers"
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=None,
        help="Number of labels for classification tasks",
    )
    parser.add_argument(
        "--label_names",
        type=str,
        nargs="+",
        default=None,
        help="Names of labels for classification tasks",
    )

    # Training parameters
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm"
    )

    # Fine-tuning strategy
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone weights"
    )
    parser.add_argument(
        "--use_layer_wise_decay",
        action="store_true",
        help="Use layer-wise learning rate decay",
    )
    parser.add_argument(
        "--layer_wise_lr_decay",
        type=float,
        default=0.9,
        help="Layer-wise learning rate decay factor",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="Label smoothing factor"
    )

    # Early stopping
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.001,
        help="Early stopping threshold",
    )

    # Evaluation
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch"],
        help="Evaluation strategy",
    )
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Determine number of labels if not provided
    if args.num_labels is None and args.task_type in [
        "sequence_classification",
        "token_classification",
    ]:
        if args.label_names:
            args.num_labels = len(args.label_names)
        else:
            raise ValueError(
                f"Must provide either --num_labels or --label_names for {args.task_type}"
            )

    # Download pretrained model if needed
    if not Path(args.pretrained_model).exists():
        logger.info(f"Downloading pre-trained model: {args.pretrained_model}")
        manager = PretrainedModelManager()
        pretrained_model_path = str(manager.download_model(args.pretrained_model))
    else:
        pretrained_model_path = args.pretrained_model

    # Create fine-tuning configuration
    config = FinetuningConfig(
        pretrained_model_path=pretrained_model_path,
        output_dir=args.output_dir,
        task_type=args.task_type,
        num_labels=args.num_labels,
        label_names=args.label_names,
        freeze_backbone=args.freeze_backbone,
        use_layer_wise_decay=args.use_layer_wise_decay,
        layer_wise_lr_decay=args.layer_wise_lr_decay,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        fp16=args.fp16,
    )

    # Create tokenizer
    tokenizer = create_tokenizer(args.tokenizer_type, args.k)

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = create_dataset(
        args.train_data, args.task_type, tokenizer, args.max_length, args.label_names
    )

    eval_dataset = None
    if args.eval_data:
        eval_dataset = create_dataset(
            args.eval_data, args.task_type, tokenizer, args.max_length, args.label_names
        )

    # Create metrics function
    def compute_metrics(eval_pred):
        metrics = GenomicMetrics()
        predictions, labels = eval_pred

        if args.task_type == "sequence_classification":
            predictions = predictions.argmax(axis=-1)
            return metrics.compute_classification_metrics(predictions, labels)
        elif args.task_type == "token_classification":
            predictions = predictions.argmax(axis=-1)
            # Flatten for token-level metrics
            predictions = predictions.reshape(-1)
            labels = labels.reshape(-1)
            # Remove ignored tokens
            mask = labels != -100
            predictions = predictions[mask]
            labels = labels[mask]
            return metrics.compute_classification_metrics(predictions, labels)
        elif args.task_type == "sequence_generation":
            return {"perplexity": metrics.compute_perplexity(predictions, labels)}
        else:
            return {}

    # Create fine-tuner
    logger.info("Setting up fine-tuner...")
    finetuner = FineTuner(config)

    # Start fine-tuning
    logger.info("Starting fine-tuning...")
    trainer = finetuner.fine_tune(train_dataset, eval_dataset, compute_metrics)

    # Save configuration and results
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Fine-tuning completed! Model saved to {args.output_dir}")

    # Run final evaluation if eval data provided
    if args.eval_data:
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate(eval_dataset)

        # Save evaluation results
        results_path = Path(args.output_dir) / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        logger.info("Evaluation results:")
        for metric, value in eval_results.items():
            if isinstance(value, int | float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
