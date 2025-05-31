#!/usr/bin/env python3
"""
Example script for fine-tuning Hyena-GLT for genome annotation.

This script demonstrates how to fine-tune a pre-trained Hyena-GLT model
for genome annotation tasks like gene finding and regulatory element prediction.
"""

import argparse
import json
import logging

# Add the project root to Python path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from hyena_glt.training.pretrained import PretrainedModelManager
from hyena_glt.training.task_specific import GenomeAnnotationFineTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Hyena-GLT for genome annotation")

    parser.add_argument("--pretrained_model", type=str, default="hyena-glt-base",
                       help="Pre-trained model name or path")
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data (JSONL format)")
    parser.add_argument("--eval_data", type=str, default=None,
                       help="Path to evaluation data (JSONL format)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for fine-tuned model")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=8192,
                       help="Maximum sequence length")

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Check if pretrained model is available
    if not Path(args.pretrained_model).exists():
        logger.info(f"Downloading pre-trained model: {args.pretrained_model}")
        manager = PretrainedModelManager()
        pretrained_path = str(manager.download_model(args.pretrained_model))
    else:
        pretrained_path = args.pretrained_model

    # Create fine-tuner
    logger.info("Setting up genome annotation fine-tuner...")
    finetuner = GenomeAnnotationFineTuner(pretrained_path, args.output_dir)

    # Override default parameters
    finetuner.config.learning_rate = args.learning_rate
    finetuner.config.batch_size = args.batch_size
    finetuner.config.num_epochs = args.num_epochs
    finetuner.config.max_length = args.max_length

    # Start fine-tuning
    logger.info("Starting fine-tuning...")
    trainer = finetuner.fine_tune(args.train_data, args.eval_data)

    # Save configuration
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "pretrained_model": args.pretrained_model,
            "train_data": args.train_data,
            "eval_data": args.eval_data,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "max_length": args.max_length,
            "labels": finetuner.labels
        }, f, indent=2)

    logger.info(f"Fine-tuning completed! Model saved to {args.output_dir}")

    # Run final evaluation if eval data provided
    if args.eval_data:
        logger.info("Running final evaluation...")
        eval_dataset = finetuner.create_dataset(args.eval_data)
        eval_results = trainer.evaluate(eval_dataset)

        # Save evaluation results
        results_path = Path(args.output_dir) / "eval_results.json"
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)

        logger.info("Evaluation results:")
        for metric, value in eval_results.items():
            logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
