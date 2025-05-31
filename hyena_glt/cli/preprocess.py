#!/usr/bin/env python3
"""
Command-line interface for preprocessing genomic data for Hyena-GLT.

Usage:
    hyena-glt-preprocess --input genome.fa --output processed/ --task sequence_classification
    hyena-glt-preprocess --input variants.vcf --output variant_data/ --task variant_effect
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Import preprocessing modules
try:
    from ..data.preprocessing import GenomicPreprocessor
    from ..utils.file_utils import detect_file_format, validate_genomic_file
    from ..utils.logging_utils import setup_logging
except ImportError as e:
    print(f"Error importing Hyena-GLT modules: {e}")
    print("Please ensure the package is properly installed.")
    sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="Preprocess genomic data for Hyena-GLT training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sequence preprocessing
  hyena-glt-preprocess --input genome.fa --output processed_data/

  # Variant effect preprocessing
  hyena-glt-preprocess --input variants.vcf --reference hg38.fa --output variant_data/

  # Custom preprocessing with specific parameters
  hyena-glt-preprocess --input data/ --task sequence_classification --max-length 2048

  # Preprocessing with tokenization
  hyena-glt-preprocess --input genome.fa --tokenizer custom_tokenizer.json --output tokens/
        """,
    )

    # Input/Output specification
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory (FASTA, VCF, BED, etc.)",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for processed data"
    )

    parser.add_argument(
        "--reference", type=str, help="Reference genome file (required for some tasks)"
    )

    # Task specification
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "sequence_classification",
            "variant_effect",
            "gene_expression",
            "regulatory_elements",
            "comparative_genomics",
            "auto",
        ],
        default="auto",
        help="Preprocessing task type (default: auto-detect)",
    )

    parser.add_argument(
        "--config", type=str, help="Path to preprocessing configuration JSON file"
    )

    # Sequence parameters
    sequence_group = parser.add_argument_group("sequence parameters")
    sequence_group.add_argument(
        "--max-length", type=int, default=1024, help="Maximum sequence length"
    )
    sequence_group.add_argument(
        "--min-length", type=int, default=50, help="Minimum sequence length"
    )
    sequence_group.add_argument(
        "--overlap", type=int, default=0, help="Overlap between sequence chunks"
    )
    sequence_group.add_argument(
        "--stride",
        type=int,
        help="Stride for sequence chunking (default: max_length - overlap)",
    )

    # Tokenization parameters
    tokenization_group = parser.add_argument_group("tokenization parameters")
    tokenization_group.add_argument(
        "--tokenizer", type=str, help="Path to custom tokenizer"
    )
    tokenization_group.add_argument(
        "--vocab-size", type=int, default=4096, help="Vocabulary size for new tokenizer"
    )
    tokenization_group.add_argument(
        "--kmer-size", type=int, default=6, help="K-mer size for tokenization"
    )

    # Data filtering and quality control
    filter_group = parser.add_argument_group("filtering parameters")
    filter_group.add_argument(
        "--filter-n", action="store_true", help="Filter sequences containing N bases"
    )
    filter_group.add_argument(
        "--filter-repeats", action="store_true", help="Filter repetitive sequences"
    )
    filter_group.add_argument(
        "--min-quality", type=float, default=0.0, help="Minimum sequence quality score"
    )
    filter_group.add_argument(
        "--max-ambiguous",
        type=float,
        default=0.1,
        help="Maximum fraction of ambiguous bases",
    )

    # Data splitting
    split_group = parser.add_argument_group("data splitting")
    split_group.add_argument(
        "--train-split", type=float, default=0.8, help="Training set fraction"
    )
    split_group.add_argument(
        "--val-split", type=float, default=0.1, help="Validation set fraction"
    )
    split_group.add_argument(
        "--test-split", type=float, default=0.1, help="Test set fraction"
    )
    split_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )

    # Output format options
    format_group = parser.add_argument_group("output format")
    format_group.add_argument(
        "--format",
        choices=["hdf5", "parquet", "json", "fasta"],
        default="hdf5",
        help="Output format",
    )
    format_group.add_argument(
        "--compress", action="store_true", help="Compress output files"
    )
    format_group.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of sequences per output chunk",
    )

    # System parameters
    system_group = parser.add_argument_group("system parameters")
    system_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of processing workers"
    )
    system_group.add_argument(
        "--memory-limit", type=str, default="8GB", help="Memory limit for processing"
    )

    # Logging options
    logging_group = parser.add_argument_group("logging options")
    logging_group.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    logging_group.add_argument(
        "--progress", action="store_true", help="Show progress bar"
    )

    return parser


def detect_task_type(input_path: str) -> str:
    """Auto-detect preprocessing task based on input format."""
    input_path_obj = Path(input_path)

    if input_path_obj.is_file():
        file_format = detect_file_format(str(input_path_obj))
        if file_format == "vcf":
            return "variant_effect"
        elif file_format in ["fasta", "fa"]:
            return "sequence_classification"
        elif file_format in ["bed", "gtf", "gff"]:
            return "regulatory_elements"
        else:
            return "sequence_classification"
    else:
        # Check directory contents
        files = list(input_path_obj.glob("*"))
        if any(detect_file_format(str(f)) == "vcf" for f in files):
            return "variant_effect"
        elif any(detect_file_format(str(f)) in ["fasta", "fa"] for f in files):
            return "sequence_classification"
        else:
            return "sequence_classification"


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check input exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path does not exist: {args.input}")

    # Validate genomic file format
    if os.path.isfile(args.input):
        if not validate_genomic_file(args.input):
            raise ValueError(f"Invalid genomic file format: {args.input}")

    # Check reference genome if required
    if args.reference and not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference genome file not found: {args.reference}")

    # Validate data splits sum to 1.0
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Data splits must sum to 1.0, got {total_split}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Validate config file if provided
    if args.config and not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")


def setup_preprocessing_config(args: argparse.Namespace) -> dict[str, Any]:
    """Setup preprocessing configuration from arguments."""
    # Load base config if provided
    if args.config:
        with open(args.config) as f:
            config: dict[str, Any] = json.load(f)
    else:
        config = {
            "preprocessing": {},
            "tokenization": {},
            "filtering": {},
            "splitting": {},
            "output": {},
            "system": {},
        }

    # Auto-detect task if needed
    task = args.task
    if task == "auto":
        task = detect_task_type(args.input)

    # Update config with command line arguments
    config["preprocessing"].update(
        {
            "task": task,
            "input_path": args.input,
            "reference_path": args.reference,
            "max_length": args.max_length,
            "min_length": args.min_length,
            "overlap": args.overlap,
            "stride": args.stride or (args.max_length - args.overlap),
        }
    )

    config["tokenization"].update(
        {
            "tokenizer_path": args.tokenizer,
            "vocab_size": args.vocab_size,
            "kmer_size": args.kmer_size,
        }
    )

    config["filtering"].update(
        {
            "filter_n_bases": args.filter_n,
            "filter_repeats": args.filter_repeats,
            "min_quality": args.min_quality,
            "max_ambiguous_fraction": args.max_ambiguous,
        }
    )

    config["splitting"].update(
        {
            "train_fraction": args.train_split,
            "val_fraction": args.val_split,
            "test_fraction": args.test_split,
            "random_seed": args.seed,
        }
    )

    config["output"].update(
        {
            "output_dir": args.output,
            "format": args.format,
            "compress": args.compress,
            "chunk_size": args.chunk_size,
        }
    )

    config["system"].update(
        {
            "num_workers": args.num_workers,
            "memory_limit": args.memory_limit,
            "show_progress": args.progress,
        }
    )

    return config


def run_preprocessing(config: dict[str, Any]) -> dict[str, Any]:
    """Run preprocessing pipeline and return statistics."""
    logger = logging.getLogger(__name__)

    # Extract preprocessing parameters from config
    preprocessing_config = config.get("preprocessing", {})

    # Initialize preprocessor with proper parameters
    preprocessor = GenomicPreprocessor(
        sequence_type=preprocessing_config.get("sequence_type", "dna"),
        min_length=preprocessing_config.get("min_length", 50),
        max_length=preprocessing_config.get("max_length", 10000),
        quality_threshold=preprocessing_config.get("quality_threshold", 20.0),
        max_n_ratio=preprocessing_config.get("max_n_ratio", 0.1),
        remove_duplicates=preprocessing_config.get("remove_duplicates", True),
        normalize_case=preprocessing_config.get("normalize_case", True),
        filter_non_standard=preprocessing_config.get("filter_non_standard", True),
    )

    logger.info(
        f"Starting {preprocessing_config.get('task', 'unknown')} preprocessing..."
    )
    logger.info(f"Input: {preprocessing_config.get('input_path', 'unknown')}")
    logger.info(f"Output: {config.get('output', {}).get('output_dir', 'unknown')}")

    # Run preprocessing pipeline
    input_path = preprocessing_config.get("input_path")
    output_dir = config.get("output", {}).get("output_dir")

    if not input_path:
        raise ValueError("Input path not specified in config")

    output_path = Path(output_dir) / "processed_sequences.fasta" if output_dir else None

    stats: dict[str, Any] = preprocessor.preprocess_file(
        input_path=input_path, output_path=output_path, file_format="auto"
    )

    # Save processing statistics
    if output_dir:
        stats_file = Path(output_dir) / "preprocessing_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Preprocessing completed. Statistics saved to: {stats_file}")

    return stats


def print_summary(stats: dict[str, Any]) -> None:
    """Print preprocessing summary."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    if "input_stats" in stats:
        print("\nInput Statistics:")
        for key, value in stats["input_stats"].items():
            print(f"  {key}: {value}")

    if "output_stats" in stats:
        print("\nOutput Statistics:")
        for key, value in stats["output_stats"].items():
            print(f"  {key}: {value}")

    if "filtering_stats" in stats:
        print("\nFiltering Statistics:")
        for key, value in stats["filtering_stats"].items():
            print(f"  {key}: {value}")

    if "processing_time" in stats:
        print(f"\nProcessing Time: {stats['processing_time']:.2f} seconds")


def main() -> None:
    """Main preprocessing function."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Validate arguments
        validate_arguments(args)
        logger.info("Arguments validated successfully")

        # Setup preprocessing configuration
        config = setup_preprocessing_config(args)
        logger.info(
            f"Preprocessing configuration setup for {config['preprocessing']['task']} task"
        )

        # Run preprocessing
        stats = run_preprocessing(config)

        # Print summary
        print_summary(stats)
        logger.info("Preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
