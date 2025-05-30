#!/usr/bin/env python3
"""
Command-line interface for evaluating Hyena-GLT models.

Usage:
    hyena-glt-eval --model path/to/model --data path/to/test_data
    hyena-glt-eval --checkpoint model.pt --task variant_effect --data variants.vcf
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import core evaluation modules
try:
    from ..evaluation.evaluator import HyenaGLTEvaluator
    from ..model.model import HyenaGLT
    from ..data.data_loader import GenomicDataLoader
    from ..utils.logging_utils import setup_logging
    from ..utils.metrics import compute_genomic_metrics
except ImportError as e:
    print(f"Error importing Hyena-GLT modules: {e}")
    print("Please ensure the package is properly installed.")
    sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate Hyena-GLT models on genomic tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic model evaluation
  hyena-glt-eval --model models/hyena_glt.pt --data test_data.fa
  
  # Task-specific evaluation
  hyena-glt-eval --checkpoint model.pt --task variant_effect --data variants.vcf
  
  # Comprehensive evaluation with metrics
  hyena-glt-eval --model model.pt --data test/ --metrics accuracy,f1,auc
  
  # Evaluation with custom config
  hyena-glt-eval --config eval_config.json --output results/
        """
    )
    
    # Model specification
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", 
        type=str,
        help="Path to saved model file (.pt)"
    )
    model_group.add_argument(
        "--checkpoint",
        type=str, 
        help="Path to training checkpoint"
    )
    
    # Data and task specification
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to evaluation data (FASTA, VCF, or directory)"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["sequence_classification", "variant_effect", "gene_expression", 
                "regulatory_elements", "comparative_genomics", "auto"],
        default="auto",
        help="Evaluation task type (default: auto-detect)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to evaluation configuration JSON file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_results",
        help="Output directory for results (default: ./evaluation_results)"
    )
    
    # Evaluation parameters
    eval_group = parser.add_argument_group("evaluation parameters")
    eval_group.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    eval_group.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    eval_group.add_argument("--metrics", type=str, help="Comma-separated list of metrics to compute")
    
    # System parameters
    system_group = parser.add_argument_group("system parameters")
    system_group.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    system_group.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    
    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument("--save-predictions", action="store_true", help="Save model predictions")
    output_group.add_argument("--save-embeddings", action="store_true", help="Save sequence embeddings")
    output_group.add_argument("--format", choices=["json", "csv", "txt"], default="json", help="Output format")
    
    # Logging options
    logging_group = parser.add_argument_group("logging options")
    logging_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    logging_group.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser


def load_model(model_path: str, device: str = "auto") -> HyenaGLT:
    """Load model from checkpoint or saved file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Determine device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        model = HyenaGLT.load_from_checkpoint(model_path, map_location=device)
        model.eval()
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def detect_task_type(data_path: str) -> str:
    """Auto-detect task type based on data format."""
    data_path = Path(data_path)
    
    if data_path.suffix.lower() == '.vcf':
        return "variant_effect"
    elif data_path.suffix.lower() in ['.fa', '.fasta']:
        return "sequence_classification"
    elif data_path.suffix.lower() in ['.bed', '.gtf', '.gff']:
        return "regulatory_elements"
    elif data_path.is_dir():
        # Check for specific file patterns in directory
        files = list(data_path.glob("*"))
        if any(f.suffix.lower() == '.vcf' for f in files):
            return "variant_effect"
        elif any(f.suffix.lower() in ['.fa', '.fasta'] for f in files):
            return "sequence_classification"
        else:
            return "sequence_classification"
    else:
        return "sequence_classification"


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check model file exists
    model_path = args.model or args.checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    # Check data path exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data path does not exist: {args.data}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Validate config file if provided
    if args.config and not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")


def setup_evaluation_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Setup evaluation configuration from arguments."""
    # Load base config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "evaluation": {},
            "data": {},
            "system": {},
            "output": {}
        }
    
    # Auto-detect task if needed
    task = args.task
    if task == "auto":
        task = detect_task_type(args.data)
    
    # Update config with command line arguments
    config["evaluation"].update({
        "task": task,
        "batch_size": args.batch_size,
        "max_sequence_length": args.max_length,
    })
    
    config["data"].update({
        "data_path": args.data,
        "num_workers": args.num_workers,
    })
    
    config["system"].update({
        "device": args.device,
    })
    
    config["output"].update({
        "output_dir": args.output,
        "save_predictions": args.save_predictions,
        "save_embeddings": args.save_embeddings,
        "format": args.format,
    })
    
    # Parse metrics if provided
    if args.metrics:
        config["evaluation"]["metrics"] = [m.strip() for m in args.metrics.split(",")]
    
    return config


def run_evaluation(model: HyenaGLT, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluation and return results."""
    logger = logging.getLogger(__name__)
    
    # Initialize evaluator
    evaluator = HyenaGLTEvaluator(model, config)
    
    # Load evaluation data
    data_loader = GenomicDataLoader(config["data"])
    eval_dataset = data_loader.load_evaluation_data()
    
    logger.info(f"Loaded {len(eval_dataset)} samples for evaluation")
    
    # Run evaluation
    logger.info(f"Running {config['evaluation']['task']} evaluation...")
    results = evaluator.evaluate(eval_dataset)
    
    return results


def save_results(results: Dict[str, Any], output_dir: str, format: str = "json") -> None:
    """Save evaluation results to file."""
    output_path = Path(output_dir)
    
    if format == "json":
        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == "csv":
        import pandas as pd
        results_file = output_path / "evaluation_results.csv"
        # Convert nested dict to flat structure for CSV
        flat_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_results[f"{key}_{subkey}"] = subvalue
            else:
                flat_results[key] = value
        pd.DataFrame([flat_results]).to_csv(results_file, index=False)
    else:  # txt format
        results_file = output_path / "evaluation_results.txt"
        with open(results_file, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
    
    print(f"Results saved to: {results_file}")


def main():
    """Main evaluation function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate arguments
        validate_arguments(args)
        logger.info("Arguments validated successfully")
        
        # Load model
        model_path = args.model or args.checkpoint
        logger.info(f"Loading model from: {model_path}")
        model = load_model(model_path, args.device)
        logger.info("Model loaded successfully")
        
        # Setup evaluation configuration
        config = setup_evaluation_config(args)
        logger.info(f"Running {config['evaluation']['task']} evaluation")
        
        # Run evaluation
        results = run_evaluation(model, config)
        
        # Print summary results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"\n{key.upper()}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
        
        # Save detailed results
        save_results(results, args.output, args.format)
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
