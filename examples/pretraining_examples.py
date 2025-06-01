"""
Example scripts for running Hyena-GLT pretraining.

This module contains example functions that demonstrate how to use
the pretraining system with different configurations and datasets.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hyena_glt.training.pretraining import HyenaGLTPretrainer
from hyena_glt.training.pretraining_config import (
    HyenaGLTPretrainingConfig,
    OpenGenomeConfigBuilder,
    create_pretraining_configs
)
from hyena_glt.training.data_utils import create_genomic_dataloaders
from hyena_glt.training.evaluation import run_comprehensive_evaluation


def example_quick_start():
    """Quick start example with minimal configuration."""
    print("=== Quick Start Example ===")
    
    # Get pre-built configurations
    configs = create_pretraining_configs()
    config = configs["mlm_small"]
    
    # Update paths for your data
    config.data.data_paths = ["/path/to/your/genomic/data"]
    config.logging.output_dir = "./quick_start_output"
    config.optimization.max_steps = 1000  # Short run for testing
    
    # Create data loaders
    train_loader, val_loader = create_genomic_dataloaders(config)
    
    # Initialize and run pretrainer
    pretrainer = HyenaGLTPretrainer(config)
    pretrainer.pretrain(train_loader, val_loader)
    
    print("Quick start pretraining completed!")


def example_opengenome_integration():
    """Example using OpenGenome dataset configuration."""
    print("=== OpenGenome Integration Example ===")
    
    # Path to OpenGenome config (from savanna repository)
    opengenome_config_path = "/home/sdodl001/savanna/configs/data/opengenome2.yml"
    
    if not os.path.exists(opengenome_config_path):
        print(f"OpenGenome config not found at {opengenome_config_path}")
        print("Please update the path or copy the config file")
        return
    
    # Build configuration from OpenGenome data
    builder = OpenGenomeConfigBuilder(opengenome_config_path)
    config = builder.build_config(strategy="mlm", model_size="base")
    
    # Customize for your setup
    config.optimization.max_steps = 50000
    config.logging.output_dir = "./opengenome_pretraining"
    config.logging.experiment_name = "hyena_glt_opengenome"
    
    # Enable wandb logging
    config.logging.wandb.enable = True
    config.logging.wandb.project = "hyena-glt-opengenome"
    
    # Create data loaders
    train_loader, val_loader = create_genomic_dataloaders(config)
    
    # Initialize and run pretrainer
    pretrainer = HyenaGLTPretrainer(config)
    pretrainer.pretrain(train_loader, val_loader)
    
    print("OpenGenome pretraining completed!")


def example_custom_configuration():
    """Example with custom configuration for specific use case."""
    print("=== Custom Configuration Example ===")
    
    # Start with base config
    configs = create_pretraining_configs()
    config = configs["mlm_base"]
    
    # Customize model architecture
    config.model.d_model = 768
    config.model.n_layer = 16
    config.model.n_head = 12
    config.model.max_position_embeddings = 12288
    
    # Customize data processing
    config.data.max_sequence_length = 12288
    config.data.preprocessing.reverse_complement_augmentation = True
    config.data.preprocessing.min_sequence_length = 500
    
    # Use span masking strategy
    config.strategy.span_masking.enable = True
    config.strategy.span_masking.mean_span_length = 4.0
    config.strategy.span_masking.max_span_length = 15
    
    # Custom optimization settings
    config.optimization.learning_rate = 8e-5
    config.optimization.batch_size = 16
    config.optimization.gradient_accumulation_steps = 4
    config.optimization.warmup_steps = 10000
    config.optimization.max_steps = 200000
    
    # Enable label smoothing
    config.loss.label_smoothing = 0.1
    
    # Hardware optimizations
    config.hardware.mixed_precision = True
    config.hardware.compile_model = True
    
    # Comprehensive logging
    config.logging.output_dir = "./custom_pretraining"
    config.logging.experiment_name = "hyena_glt_custom"
    config.logging.log_every_n_steps = 50
    config.logging.save_every_n_steps = 5000
    config.logging.validate_every_n_steps = 1000
    
    print("Configuration created. Starting pretraining...")
    
    # Create data loaders
    train_loader, val_loader = create_genomic_dataloaders(config)
    
    # Initialize and run pretrainer
    pretrainer = HyenaGLTPretrainer(config)
    pretrainer.pretrain(train_loader, val_loader)
    
    print("Custom pretraining completed!")


def example_multi_strategy_comparison():
    """Example comparing different pretraining strategies."""
    print("=== Multi-Strategy Comparison Example ===")
    
    strategies = ["ar", "mlm", "span"]
    base_output_dir = "./strategy_comparison"
    
    for strategy in strategies:
        print(f"\n--- Running {strategy.upper()} pretraining ---")
        
        # Get configuration for this strategy
        configs = create_pretraining_configs()
        config_key = f"{strategy}_base"
        
        if config_key not in configs:
            print(f"Configuration {config_key} not available, skipping...")
            continue
            
        config = configs[config_key]
        
        # Customize for comparison
        config.optimization.max_steps = 10000  # Short runs for comparison
        config.logging.output_dir = f"{base_output_dir}/{strategy}"
        config.logging.experiment_name = f"hyena_glt_{strategy}_comparison"
        
        # Enable wandb with strategy tag
        config.logging.wandb.enable = True
        config.logging.wandb.project = "hyena-glt-strategy-comparison"
        config.logging.wandb.tags.append(strategy)
        
        # Create data loaders
        train_loader, val_loader = create_genomic_dataloaders(config)
        
        # Initialize and run pretrainer
        pretrainer = HyenaGLTPretrainer(config)
        pretrainer.pretrain(train_loader, val_loader)
        
        # Run evaluation
        print(f"Evaluating {strategy} model...")
        model = pretrainer.model
        tokenizer = pretrainer.tokenizer
        device = pretrainer.device
        
        eval_results = run_comprehensive_evaluation(
            model, tokenizer, val_loader, device, max_batches=10
        )
        
        print(f"{strategy.upper()} Results:")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nStrategy comparison completed!")


def example_evaluation_only():
    """Example of running evaluation on a pretrained model."""
    print("=== Evaluation Only Example ===")
    
    # Path to pretrained model checkpoint
    checkpoint_path = "./pretraining_output/checkpoints/checkpoint_10000.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please update the path to your pretrained model")
        return
    
    # Load configuration and model
    config_path = "./pretraining_output/config.yaml"
    config = HyenaGLTPretrainingConfig.load_from_file(config_path)
    
    # Create data loader for evaluation
    _, val_loader = create_genomic_dataloaders(config)
    
    # Initialize pretrainer and load checkpoint
    pretrainer = HyenaGLTPretrainer(config)
    pretrainer.load_checkpoint(checkpoint_path)
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    eval_results = run_comprehensive_evaluation(
        pretrainer.model,
        pretrainer.tokenizer,
        val_loader,
        pretrainer.device,
        max_batches=50
    )
    
    print("Evaluation Results:")
    print("=" * 50)
    for metric, value in eval_results.items():
        print(f"{metric:25s}: {value:.4f}")
    
    print("Evaluation completed!")


def example_resume_training():
    """Example of resuming training from a checkpoint."""
    print("=== Resume Training Example ===")
    
    # Path to checkpoint to resume from
    checkpoint_path = "./pretraining_output/checkpoints/checkpoint_5000.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please update the path to your checkpoint")
        return
    
    # Load configuration
    config_path = "./pretraining_output/config.yaml"
    config = HyenaGLTPretrainingConfig.load_from_file(config_path)
    
    # Update configuration for resumed training
    config.logging.resume_from_checkpoint = checkpoint_path
    config.optimization.max_steps = 20000  # Continue to more steps
    config.logging.experiment_name = "hyena_glt_resumed"
    
    # Create data loaders
    train_loader, val_loader = create_genomic_dataloaders(config)
    
    # Initialize pretrainer (will automatically load checkpoint)
    pretrainer = HyenaGLTPretrainer(config)
    
    # Resume training
    print(f"Resuming training from step {pretrainer.global_step}")
    pretrainer.pretrain(train_loader, val_loader)
    
    print("Resumed training completed!")


if __name__ == "__main__":
    """Run examples based on command line argument."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Hyena-GLT pretraining examples")
    parser.add_argument(
        "example",
        choices=[
            "quick_start",
            "opengenome",
            "custom",
            "comparison",
            "evaluation",
            "resume"
        ],
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "quick_start":
        example_quick_start()
    elif args.example == "opengenome":
        example_opengenome_integration()
    elif args.example == "custom":
        example_custom_configuration()
    elif args.example == "comparison":
        example_multi_strategy_comparison()
    elif args.example == "evaluation":
        example_evaluation_only()
    elif args.example == "resume":
        example_resume_training()
    else:
        print(f"Unknown example: {args.example}")
        sys.exit(1)
