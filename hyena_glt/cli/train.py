#!/usr/bin/env python3
"""
Command-line interface for training Hyena-GLT models.

Usage:
    hyena-glt-train --config config.json
    hyena-glt-train --model tiny --data path/to/data --output results/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import core training modules
try:
    from ..training.trainer import HyenaGLTTrainer
    from ..config.model_config import ModelConfig
    from ..data.data_loader import GenomicDataLoader
    from ..utils.logging_utils import setup_logging
except ImportError as e:
    print(f"Error importing Hyena-GLT modules: {e}")
    print("Please ensure the package is properly installed.")
    sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for training CLI."""
    parser = argparse.ArgumentParser(
        description="Train Hyena-GLT models for genomic sequence analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  hyena-glt-train --config configs/tiny_model.json
  
  # Quick training with minimal parameters
  hyena-glt-train --model tiny --data data/genome_seqs.fa --epochs 10
  
  # Resume from checkpoint
  hyena-glt-train --config configs/model.json --resume checkpoints/latest.pt
  
  # Training with custom parameters
  hyena-glt-train --model small --data data/ --batch-size 32 --lr 1e-4
        """
    )
    
    # Primary arguments
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to training configuration JSON file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["tiny", "small", "medium", "large"],
        default="tiny",
        help="Model size preset (default: tiny)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (FASTA file or directory)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./training_output",
        help="Output directory for checkpoints and logs (default: ./training_output)"
    )
    
    # Training parameters
    training_group = parser.add_argument_group("training parameters")
    training_group.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    training_group.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    training_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    training_group.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    
    # System parameters
    system_group = parser.add_argument_group("system parameters")
    system_group.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    system_group.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    system_group.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    
    # Resume and checkpoint options
    checkpoint_group = parser.add_argument_group("checkpoint options")
    checkpoint_group.add_argument("--resume", type=str, help="Resume from checkpoint")
    checkpoint_group.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    
    # Logging and monitoring
    logging_group = parser.add_argument_group("logging options")
    logging_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    logging_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    logging_group.add_argument("--wandb-project", type=str, default="hyena-glt", help="W&B project name")
    
    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check if data path exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data path does not exist: {args.data}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Validate config file if provided
    if args.config and not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")


def setup_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Setup training configuration from arguments."""
    # Load base config if provided
    if args.config:
        config = load_config(args.config)
    else:
        # Create default config based on model size
        config = {
            "model": {
                "type": "hyena-glt",
                "size": args.model,
            },
            "training": {},
            "data": {},
            "system": {}
        }
    
    # Override with command line arguments
    config["training"].update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_sequence_length": args.max_length,
        "save_every": args.save_every,
        "mixed_precision": args.mixed_precision,
    })
    
    config["data"].update({
        "data_path": args.data,
        "num_workers": args.num_workers,
    })
    
    config["system"].update({
        "device": args.device,
        "output_dir": args.output,
    })
    
    if args.resume:
        config["training"]["resume_from"] = args.resume
    
    if args.wandb:
        config["logging"] = {
            "wandb": True,
            "wandb_project": args.wandb_project,
        }
    
    return config


def main():
    """Main training function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate arguments
        validate_arguments(args)
        logger.info("Arguments validated successfully")
        
        # Setup configuration
        config = setup_training_config(args)
        logger.info(f"Training configuration loaded for {config['model']['size']} model")
        
        # Initialize trainer
        trainer = HyenaGLTTrainer(config)
        logger.info("Trainer initialized")
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
