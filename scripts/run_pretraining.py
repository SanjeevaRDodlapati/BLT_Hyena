#!/usr/bin/env python3
"""
Main script for running Hyena-GLT pretraining.

This script provides a command-line interface for pretraining Hyena-GLT models
on genomic datasets using various pretraining strategies.

Usage:
    python scripts/run_pretraining.py --config configs/pretraining/base_config.yaml
    python scripts/run_pretraining.py --strategy mlm --model_size small --max_steps 10000
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hyena_glt.training.pretraining import HyenaGLTPretrainer
from hyena_glt.training.pretraining_config import (
    HyenaGLTPretrainingConfig,
    OpenGenomeConfigBuilder,
    create_pretraining_configs
)
from hyena_glt.training.data_utils import create_genomic_dataloaders


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Hyena-GLT pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to YAML configuration file"
    )
    
    # Quick setup options (alternative to config file)
    parser.add_argument(
        "--strategy",
        choices=["ar", "mlm", "oadm", "span"],
        help="Pretraining strategy to use"
    )
    parser.add_argument(
        "--model_size",
        choices=["tiny", "small", "base", "large"],
        help="Model size preset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing genomic data files"
    )
    parser.add_argument(
        "--opengenome_config",
        type=str,
        help="Path to OpenGenome dataset configuration file"
    )
    
    # Training parameters
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        help="Maximum sequence length"
    )
    
    # Hardware options
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for training"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for optimization"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pretraining_output",
        help="Directory to save training outputs"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name for the experiment (used in logging)"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to log file (optional)"
    )
    
    # Checkpoint options
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        help="Save checkpoint every N steps"
    )
    
    # Validation options
    parser.add_argument(
        "--validate_every",
        type=int,
        help="Run validation every N steps"
    )
    parser.add_argument(
        "--validation_data_dir",
        type=str,
        help="Directory containing validation data"
    )
    
    return parser.parse_args()


def load_config_from_file(config_path: str) -> HyenaGLTPretrainingConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return HyenaGLTPretrainingConfig.from_dict(config_dict)


def create_config_from_args(args) -> HyenaGLTPretrainingConfig:
    """Create configuration from command line arguments."""
    if args.config:
        # Load from file and override with CLI args
        config = load_config_from_file(args.config)
        
        # Override with CLI arguments if provided
        if args.strategy:
            config.strategy.name = args.strategy
        if args.max_steps:
            config.optimization.max_steps = args.max_steps
        if args.batch_size:
            config.optimization.batch_size = args.batch_size
        if args.learning_rate:
            config.optimization.learning_rate = args.learning_rate
        if args.sequence_length:
            config.data.max_sequence_length = args.sequence_length
            
    elif args.opengenome_config:
        # Create from OpenGenome configuration
        builder = OpenGenomeConfigBuilder(args.opengenome_config)
        config = builder.build_config(
            strategy=args.strategy or "mlm",
            model_size=args.model_size or "base"
        )
        
        # Override with CLI arguments
        if args.max_steps:
            config.optimization.max_steps = args.max_steps
        if args.batch_size:
            config.optimization.batch_size = args.batch_size
        if args.learning_rate:
            config.optimization.learning_rate = args.learning_rate
            
    else:
        # Create from scratch using presets
        configs = create_pretraining_configs()
        strategy = args.strategy or "mlm"
        model_size = args.model_size or "base"
        
        config_key = f"{strategy}_{model_size}"
        if config_key not in configs:
            raise ValueError(f"No preset configuration for {config_key}")
        config = configs[config_key]
        
        # Set data directory if provided
        if args.data_dir:
            config.data.data_paths = [args.data_dir]
        
        # Override with CLI arguments
        if args.max_steps:
            config.optimization.max_steps = args.max_steps
        if args.batch_size:
            config.optimization.batch_size = args.batch_size
        if args.learning_rate:
            config.optimization.learning_rate = args.learning_rate
        if args.sequence_length:
            config.data.max_sequence_length = args.sequence_length
    
    # Apply remaining CLI overrides
    if args.output_dir:
        config.logging.output_dir = args.output_dir
    if args.experiment_name:
        config.logging.experiment_name = args.experiment_name
    if args.device and args.device != "auto":
        config.hardware.device = args.device
    if args.mixed_precision:
        config.hardware.mixed_precision = True
    if args.compile:
        config.hardware.compile_model = True
    if args.resume_from:
        config.logging.resume_from_checkpoint = args.resume_from
    if args.save_every:
        config.logging.save_every_n_steps = args.save_every
    if args.validate_every:
        config.logging.validate_every_n_steps = args.validate_every
    if args.validation_data_dir:
        config.data.validation_data_paths = [args.validation_data_dir]
    
    return config


def validate_config(config: HyenaGLTPretrainingConfig):
    """Validate the configuration and check for common issues."""
    logger = logging.getLogger(__name__)
    
    # Check data paths exist
    for path in config.data.data_paths:
        if not os.path.exists(path):
            logger.warning(f"Data path does not exist: {path}")
    
    # Check output directory is writable
    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {output_dir}")
    
    # Check GPU availability if CUDA requested
    if config.hardware.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        config.hardware.device = "cpu"
    
    # Check memory requirements
    if config.hardware.device == "cpu" and config.optimization.batch_size > 32:
        logger.warning("Large batch size on CPU may cause memory issues")
    
    logger.info(f"Configuration validated successfully")
    logger.info(f"Strategy: {config.strategy.name}")
    logger.info(f"Model size: {config.model.d_model} dims, {config.model.n_layer} layers")
    logger.info(f"Batch size: {config.optimization.batch_size}")
    logger.info(f"Max steps: {config.optimization.max_steps}")
    logger.info(f"Device: {config.hardware.device}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Hyena-GLT pretraining")
    logger.info(f"Command line arguments: {vars(args)}")
    
    try:
        # Create configuration
        logger.info("Loading configuration...")
        config = create_config_from_args(args)
        
        # Validate configuration
        validate_config(config)
        
        # Save configuration
        config_save_path = Path(config.logging.output_dir) / "config.yaml"
        config.save_to_file(str(config_save_path))
        logger.info(f"Configuration saved to: {config_save_path}")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_genomic_dataloaders(config)
        logger.info(f"Created data loaders with {len(train_loader)} training batches")
        
        # Initialize pretrainer
        logger.info("Initializing pretrainer...")
        pretrainer = HyenaGLTPretrainer(config)
        
        # Start pretraining
        logger.info("Starting pretraining...")
        pretrainer.pretrain(train_loader, val_loader)
        
        logger.info("Pretraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Pretraining failed with error: {e}")
        if args.log_level == "DEBUG":
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
