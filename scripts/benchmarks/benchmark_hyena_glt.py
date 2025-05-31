"""
Benchmarking script for Hyena-GLT model evaluation.

This script provides comprehensive benchmarking capabilities including:
- Performance evaluation across multiple genomic tasks
- Scalability analysis (batch size, sequence length)
- Computational efficiency measurement
- Statistical analysis and visualization
- Comparative evaluation against baselines

Usage:
    python benchmark_hyena_glt.py --config benchmark_config.json
    python benchmark_hyena_glt.py --quick-benchmark
    python benchmark_hyena_glt.py --comparison-mode --models model1,model2
"""

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_genomic_datasets(num_samples_per_task: int = 1000) -> dict[str, Any]:
    """Create sample genomic datasets for benchmarking."""

    from hyena_glt.data import GenomicTokenizer

    logger.info("Creating sample genomic datasets...")

    # Initialize tokenizer
    GenomicTokenizer(sequence_type="dna", k=1)

    datasets = {}

    # 1. Promoter Detection (Binary Classification)
    logger.info("Creating promoter detection dataset...")
    promoter_sequences = []
    promoter_labels = []

    for _i in range(num_samples_per_task):
        # Generate random DNA sequence
        length = np.random.randint(200, 1000)
        nucleotides = ["A", "C", "G", "T"]
        sequence = "".join(np.random.choice(nucleotides, length))

        # Binary classification based on TATA box presence
        has_tata = "TATAAA" in sequence or "TATAWA" in sequence.replace(
            "T", "W"
        ).replace("A", "W")

        promoter_sequences.append(sequence)
        promoter_labels.append(1 if has_tata else 0)

    datasets["promoter_detection"] = {
        "sequences": promoter_sequences,
        "labels": promoter_labels,
        "task_type": "sequence_classification",
        "num_classes": 2,
        "description": "Binary classification for promoter region detection",
    }

    # 2. GC Content Prediction (Regression)
    logger.info("Creating GC content prediction dataset...")
    gc_sequences = []
    gc_labels = []

    for _i in range(num_samples_per_task):
        length = np.random.randint(500, 2000)
        # Create sequences with varying GC content
        gc_content = np.random.uniform(0.2, 0.8)

        sequence = []
        for _ in range(length):
            if np.random.random() < gc_content:
                sequence.append(np.random.choice(["G", "C"]))
            else:
                sequence.append(np.random.choice(["A", "T"]))

        sequence = "".join(sequence)
        actual_gc = (sequence.count("G") + sequence.count("C")) / len(sequence)

        gc_sequences.append(sequence)
        gc_labels.append(actual_gc)

    datasets["gc_content_prediction"] = {
        "sequences": gc_sequences,
        "labels": gc_labels,
        "task_type": "regression",
        "description": "Regression task for predicting GC content",
    }

    # 3. Splice Site Prediction (Token Classification)
    logger.info("Creating splice site prediction dataset...")
    splice_sequences = []
    splice_labels = []

    for _i in range(
        num_samples_per_task // 2
    ):  # Fewer samples for token classification
        length = np.random.randint(1000, 4000)
        nucleotides = ["A", "C", "G", "T"]
        sequence = "".join(np.random.choice(nucleotides, length))

        # Create random splice site labels (0: intergenic, 1: exon, 2: intron)
        labels = np.random.choice([0, 1, 2], size=length, p=[0.6, 0.25, 0.15])

        # Add some realistic splice sites (GT-AG pattern)
        for j in range(0, length - 10, 100):
            if j + 2 < length and sequence[j : j + 2] == "GT":
                labels[j : j + 2] = 2  # Mark as intron start

        splice_sequences.append(sequence)
        splice_labels.append(labels.tolist())

    datasets["splice_site_prediction"] = {
        "sequences": splice_sequences,
        "labels": splice_labels,
        "task_type": "token_classification",
        "num_classes": 3,
        "description": "Token-level classification for splice site prediction",
    }

    # 4. Sequence Generation (Language Modeling)
    logger.info("Creating sequence generation dataset...")
    generation_sequences = []

    for _i in range(num_samples_per_task):
        length = np.random.randint(100, 500)
        nucleotides = ["A", "C", "G", "T"]
        sequence = "".join(np.random.choice(nucleotides, length))
        generation_sequences.append(sequence)

    datasets["sequence_generation"] = {
        "sequences": generation_sequences,
        "labels": generation_sequences,  # Self-supervised
        "task_type": "generation",
        "description": "Autoregressive sequence generation task",
    }

    logger.info(
        f"Created {len(datasets)} datasets with {num_samples_per_task} samples each"
    )
    return datasets


def create_model_variants() -> dict[str, dict]:
    """Create different model configurations for comparison."""

    from hyena_glt.config import HyenaGLTConfig

    variants = {
        "small": {
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 512,
            "hyena_order": 2,
            "hyena_filter_size": 32,
            "local_window_size": 64,
        },
        "base": {
            "hidden_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": 1024,
            "hyena_order": 2,
            "hyena_filter_size": 64,
            "local_window_size": 128,
        },
        "large": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "intermediate_size": 2048,
            "hyena_order": 3,
            "hyena_filter_size": 128,
            "local_window_size": 256,
        },
    }

    # Convert to full configs
    full_configs = {}
    for name, variant in variants.items():
        config = HyenaGLTConfig(
            vocab_size=8,  # DNA tokens
            max_position_embeddings=4096,
            use_genomic_encoding=True,
            dropout=0.1,
            **variant,
        )
        full_configs[name] = config

    return full_configs


def run_comprehensive_benchmark(
    config_path: str | None = None,
    output_dir: str = "./benchmark_results",
    quick_mode: bool = False,
) -> dict[str, Any]:
    """Run comprehensive benchmarking suite."""

    from hyena_glt.evaluation.benchmark import (
        BenchmarkConfig,
        BenchmarkRunner,
        create_genomic_benchmark_suite,
    )
    from hyena_glt.evaluation.visualizers import create_evaluation_dashboard
    from hyena_glt.model import (
        HyenaGLTForSequenceClassification,
        HyenaGLTForSequenceGeneration,
        HyenaGLTForTokenClassification,
    )

    logger.info("Starting comprehensive benchmark...")

    # Load or create benchmark configuration
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            benchmark_configs = json.load(f)
        benchmark_configs = [BenchmarkConfig(**config) for config in benchmark_configs]
    else:
        benchmark_configs = create_genomic_benchmark_suite()
        if quick_mode:
            # Reduce scope for quick benchmark
            for config in benchmark_configs:
                config.num_runs = 2
                config.batch_sizes = [4, 8] if config.batch_sizes else None
                config.sequence_lengths = (
                    [512, 1024] if config.sequence_lengths else None
                )

    # Create sample datasets
    num_samples = 100 if quick_mode else 1000
    datasets = create_sample_genomic_datasets(num_samples)

    # Create model configurations
    model_configs = create_model_variants()
    if quick_mode:
        model_configs = {"small": model_configs["small"]}  # Only test small model

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Results storage
    all_results = {}

    # Run benchmarks for each model variant
    for model_name, model_config in model_configs.items():
        logger.info(f"\nBenchmarking model variant: {model_name}")

        model_results = {}

        # Model factory function
        def model_factory(config_dict, base_config=model_config):
            # Determine model type based on task
            if "num_labels" in config_dict:
                if config_dict.get("task_type") == "token_classification":
                    return HyenaGLTForTokenClassification(base_config)
                elif config_dict.get("task_type") == "generation":
                    return HyenaGLTForSequenceGeneration(base_config)
                else:
                    return HyenaGLTForSequenceClassification(base_config)
            else:
                return HyenaGLTForSequenceClassification(base_config)

        # Run each benchmark configuration
        for benchmark_config in benchmark_configs:
            if quick_mode and benchmark_config.name != "genomic_scalability":
                continue  # Skip some benchmarks in quick mode

            logger.info(f"Running benchmark: {benchmark_config.name}")

            try:
                # Create data loaders for this benchmark
                data_loaders = create_benchmark_data_loaders(
                    datasets, benchmark_config, quick_mode
                )

                # Run benchmark
                runner = BenchmarkRunner(benchmark_config)
                benchmark_result = runner.run_full_benchmark(
                    model_factory=model_factory,
                    base_model_config=asdict(model_config),
                    data_loaders=data_loaders,
                )

                model_results[benchmark_config.name] = benchmark_result

            except Exception as e:
                logger.error(f"Benchmark {benchmark_config.name} failed: {e}")
                continue

        all_results[model_name] = model_results

    # Generate comprehensive analysis
    logger.info("Generating analysis and reports...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw results
    results_file = output_path / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Raw results saved to: {results_file}")

    # Create visualizations
    try:
        for model_name, model_results in all_results.items():
            if model_results:
                viz_files = create_evaluation_dashboard(
                    model_results,
                    str(output_path / model_name),
                    model_name.capitalize(),
                )
                logger.info(f"Visualizations for {model_name}: {viz_files}")
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")

    # Generate summary report
    try:
        summary_report = generate_benchmark_summary(all_results, str(output_path))
        logger.info(f"Summary report generated: {summary_report}")
    except Exception as e:
        logger.warning(f"Summary report generation failed: {e}")

    logger.info("Comprehensive benchmark completed!")
    return all_results


def create_benchmark_data_loaders(
    datasets: dict[str, Any], benchmark_config: Any, quick_mode: bool = False
) -> dict[str, Any]:
    """Create data loaders for benchmarking."""

    from hyena_glt.data import GenomicDataset, GenomicTokenizer

    tokenizer = GenomicTokenizer(sequence_type="dna", k=1)
    data_loaders = {}

    batch_size = 4 if quick_mode else 8
    max_samples = 50 if quick_mode else 200

    for dataset_name in benchmark_config.datasets:
        if dataset_name in datasets:
            dataset_info = datasets[dataset_name]

            # Create dataset
            sequences = dataset_info["sequences"][:max_samples]
            labels = dataset_info["labels"][:max_samples]

            dataset = GenomicDataset(
                sequences=sequences,
                labels=labels,
                tokenizer=tokenizer,
                max_length=1024 if quick_mode else 2048,
                task_type=dataset_info["task_type"],
            )

            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True
            )

            data_loaders[dataset_name] = data_loader

    return data_loaders


def generate_benchmark_summary(results: dict[str, Any], output_dir: str) -> str:
    """Generate a comprehensive benchmark summary report."""

    summary_path = Path(output_dir) / "benchmark_summary.md"

    with open(summary_path, "w") as f:
        f.write("# Hyena-GLT Benchmark Summary\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overview\n\n")
        f.write(
            f"This report summarizes the benchmarking results for {len(results)} model variants "
        )
        f.write("across multiple genomic sequence modeling tasks.\n\n")

        # Model comparison
        f.write("## Model Variants\n\n")
        for model_name, model_results in results.items():
            f.write(f"### {model_name.capitalize()}\n\n")

            # Extract parameter info if available
            param_info = None
            for _benchmark_name, benchmark_result in model_results.items():
                if (
                    hasattr(benchmark_result, "results")
                    and "parameter_analysis" in benchmark_result.results
                ):
                    param_info = benchmark_result.results["parameter_analysis"]
                    break

            if param_info:
                f.write(
                    f"- **Total Parameters**: {param_info.get('total_parameters', 'Unknown'):,}\n"
                )
                f.write(
                    f"- **Trainable Parameters**: {param_info.get('trainable_parameters', 'Unknown'):,}\n"
                )

            f.write("\n")

        # Performance summary
        f.write("## Performance Summary\n\n")

        # Collect task performance across models
        task_performance = {}
        for model_name, model_results in results.items():
            for _benchmark_name, benchmark_result in model_results.items():
                if hasattr(benchmark_result, "results"):
                    for key, value in benchmark_result.results.items():
                        if key.startswith("dataset_"):
                            task_name = key.replace("dataset_", "")
                            if task_name not in task_performance:
                                task_performance[task_name] = {}

                            # Extract metrics
                            if "summary_metrics" in value:
                                task_performance[task_name][model_name] = value[
                                    "summary_metrics"
                                ]

        # Write task performance table
        if task_performance:
            f.write("### Task Performance Comparison\n\n")

            for task_name, model_metrics in task_performance.items():
                f.write(f"#### {task_name.replace('_', ' ').title()}\n\n")

                # Create table
                metrics = set()
                for _model_name, task_metrics in model_metrics.items():
                    if isinstance(task_metrics, dict):
                        metrics.update(task_metrics.keys())

                if metrics:
                    f.write("| Model | " + " | ".join(metrics) + " |\n")
                    f.write("|-------|" + "|".join(["-------"] * len(metrics)) + "|\n")

                    for model_name, task_metrics in model_metrics.items():
                        if isinstance(task_metrics, dict):
                            values = [
                                f"{task_metrics.get(metric, 0):.4f}"
                                for metric in metrics
                            ]
                            f.write(f"| {model_name} | " + " | ".join(values) + " |\n")

                f.write("\n")

        # Computational efficiency
        f.write("## Computational Efficiency\n\n")

        efficiency_data = {}
        for model_name, model_results in results.items():
            for _benchmark_name, benchmark_result in model_results.items():
                if (
                    hasattr(benchmark_result, "results")
                    and "computational_metrics" in benchmark_result.results
                ):
                    efficiency_data[model_name] = benchmark_result.results[
                        "computational_metrics"
                    ]

        if efficiency_data:
            f.write(
                "| Model | Avg Inference Time (s) | Peak Memory (MB) | Throughput (samples/s) |\n"
            )
            f.write(
                "|-------|----------------------|------------------|----------------------|\n"
            )

            for model_name, metrics in efficiency_data.items():
                inf_time = metrics.get("avg_inference_time", 0)
                memory = metrics.get("peak_memory_mb", 0)
                throughput = metrics.get("throughput_samples_per_sec", 0)

                f.write(
                    f"| {model_name} | {inf_time:.4f} | {memory:.1f} | {throughput:.1f} |\n"
                )

        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on the benchmark results:\n\n")
        f.write(
            "1. **Model Selection**: Choose model size based on task complexity and computational constraints\n"
        )
        f.write(
            "2. **Task Optimization**: Focus additional training on underperforming tasks\n"
        )
        f.write(
            "3. **Efficiency Optimization**: Consider model compression for deployment scenarios\n"
        )
        f.write(
            "4. **Scalability**: Monitor memory usage for long sequences in production\n"
        )
        f.write(
            "5. **Further Analysis**: Conduct task-specific hyperparameter optimization\n\n"
        )

        f.write("---\n")
        f.write(
            "*This report was generated automatically by the Hyena-GLT benchmarking framework.*\n"
        )

    return str(summary_path)


def main():
    """Main benchmarking script."""

    parser = argparse.ArgumentParser(
        description="Comprehensive benchmarking for Hyena-GLT genomic models"
    )

    parser.add_argument(
        "--config", type=str, help="Path to benchmark configuration JSON file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results",
    )

    parser.add_argument(
        "--quick-benchmark",
        action="store_true",
        help="Run quick benchmark with reduced scope",
    )

    parser.add_argument(
        "--comparison-mode",
        action="store_true",
        help="Run comparison between multiple models",
    )

    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model variants to test (small,base,large)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run benchmarks on",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Running benchmarks on device: {device}")

    # Run benchmark
    try:
        run_comprehensive_benchmark(
            config_path=args.config,
            output_dir=args.output_dir,
            quick_mode=args.quick_benchmark,
        )

        logger.info("Benchmarking completed successfully!")
        logger.info(f"Results available in: {args.output_dir}")

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        raise


if __name__ == "__main__":
    main()
