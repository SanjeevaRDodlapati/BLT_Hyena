#!/usr/bin/env python3
"""
Performance Benchmark for Hyena-GLT Models

This script benchmarks Hyena-GLT models across different configurations
and datasets to measure training speed, memory usage, and accuracy.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

# Import Hyena-GLT components
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data import DNATokenizer, SequenceClassificationDataset
from hyena_glt.model import HyenaGLTForSequenceClassification
from hyena_glt.training import HyenaGLTTrainer, TrainingConfig


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    model_sizes: list[str] = None
    sequence_lengths: list[int] = None
    batch_sizes: list[int] = None
    dataset_sizes: list[int] = None
    num_epochs: int = 1
    device: str = "auto"
    memory_limit_gb: float = 16.0
    output_dir: str = "./benchmark_results"

    def __post_init__(self):
        if self.model_sizes is None:
            self.model_sizes = ["tiny", "small", "medium"]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512, 1024]
        if self.batch_sizes is None:
            self.batch_sizes = [4, 8, 16, 32]
        if self.dataset_sizes is None:
            self.dataset_sizes = [100, 500, 1000]


class PerformanceBenchmark:
    """Comprehensive performance benchmark for Hyena-GLT models."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = DNATokenizer(k=3)

    def get_model_config(self, size: str, seq_length: int) -> HyenaGLTConfig:
        """Get model configuration for different sizes."""
        configs = {
            "tiny": {
                "hidden_size": 128,
                "num_hyena_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 256,
                "local_encoder_layers": 1,
                "local_decoder_layers": 1,
            },
            "small": {
                "hidden_size": 256,
                "num_hyena_layers": 4,
                "num_attention_heads": 8,
                "intermediate_size": 512,
                "local_encoder_layers": 2,
                "local_decoder_layers": 2,
            },
            "medium": {
                "hidden_size": 512,
                "num_hyena_layers": 6,
                "num_attention_heads": 16,
                "intermediate_size": 1024,
                "local_encoder_layers": 3,
                "local_decoder_layers": 3,
            },
            "large": {
                "hidden_size": 768,
                "num_hyena_layers": 8,
                "num_attention_heads": 24,
                "intermediate_size": 2048,
                "local_encoder_layers": 4,
                "local_decoder_layers": 4,
            },
        }

        base_config = configs[size]
        return HyenaGLTConfig(
            **base_config,
            max_position_embeddings=seq_length,
            num_labels=2,
            task_type="sequence_classification",
            genomic_vocab_size=4096,
            patch_size=4,
            hyena_order=3,
            hyena_filter_size=64,
            gradient_checkpointing=True,
        )

    def generate_synthetic_data(
        self, num_samples: int, seq_length: int
    ) -> tuple[list[str], list[int]]:
        """Generate synthetic genomic data for benchmarking."""
        nucleotides = ["A", "T", "G", "C"]
        sequences = []
        labels = []

        for _ in range(num_samples):
            # Generate random sequence
            sequence = "".join(np.random.choice(nucleotides, size=seq_length))
            sequences.append(sequence)

            # Simple binary classification based on GC content
            gc_content = (sequence.count("G") + sequence.count("C")) / len(sequence)
            label = 1 if gc_content > 0.5 else 0
            labels.append(label)

        return sequences, labels

    def measure_memory_usage(self) -> dict[str, float]:
        """Measure current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()

        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                gpu_memory[f"gpu_{i}_allocated"] = allocated
                gpu_memory[f"gpu_{i}_cached"] = cached

        return {
            "ram_gb": memory_info.rss / 1024**3,
            "ram_peak_gb": (
                memory_info.peak_wss / 1024**3
                if hasattr(memory_info, "peak_wss")
                else None
            ),
            **gpu_memory,
        }

    def benchmark_training(
        self, model_size: str, seq_length: int, batch_size: int, dataset_size: int
    ) -> dict[str, Any]:
        """Benchmark training performance for specific configuration."""

        self.logger.info(
            f"Benchmarking: {model_size}, seq_len={seq_length}, batch={batch_size}, data={dataset_size}"
        )

        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Generate data
            sequences, labels = self.generate_synthetic_data(dataset_size, seq_length)

            # Create model and config
            model_config = self.get_model_config(model_size, seq_length)
            model = HyenaGLTForSequenceClassification(model_config)
            model = model.to(self.device)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            # Create dataset and dataloader
            dataset = SequenceClassificationDataset(
                sequences=sequences,
                labels=labels,
                tokenizer=self.tokenizer,
                max_length=seq_length,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Avoid multiprocessing issues in benchmarks
            )

            # Training configuration
            training_config = TrainingConfig(
                num_epochs=self.config.num_epochs,
                batch_size=batch_size,
                learning_rate=1e-4,
                output_dir=f"{self.config.output_dir}/tmp_{model_size}_{seq_length}_{batch_size}",
                logging_steps=10,
                eval_steps=None,  # Skip evaluation for speed
                save_steps=None,  # Skip saving for speed
                fp16=torch.cuda.is_available(),
                gradient_checkpointing=True,
            )

            # Create trainer
            trainer = HyenaGLTTrainer(
                model=model,
                config=training_config,
                train_dataloader=dataloader,
                eval_dataloader=None,
                tokenizer=self.tokenizer,
            )

            # Measure memory before training
            memory_before = self.measure_memory_usage()

            # Benchmark training
            start_time = time.time()
            training_results = trainer.train()
            end_time = time.time()

            # Measure memory after training
            memory_after = self.measure_memory_usage()

            # Calculate metrics
            training_time = end_time - start_time
            samples_per_second = (dataset_size * self.config.num_epochs) / training_time

            # Extract final loss
            final_loss = training_results.get("train_loss", None)

            # Calculate memory delta
            memory_delta = {
                key: memory_after.get(key, 0) - memory_before.get(key, 0)
                for key in memory_before.keys()
            }

            result = {
                "model_size": model_size,
                "sequence_length": seq_length,
                "batch_size": batch_size,
                "dataset_size": dataset_size,
                "num_parameters": num_params,
                "trainable_parameters": trainable_params,
                "training_time_seconds": training_time,
                "samples_per_second": samples_per_second,
                "final_loss": final_loss,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_delta": memory_delta,
                "success": True,
                "error": None,
                "device": str(self.device),
            }

            # Cleanup
            del model, trainer, dataset, dataloader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return {
                "model_size": model_size,
                "sequence_length": seq_length,
                "batch_size": batch_size,
                "dataset_size": dataset_size,
                "success": False,
                "error": str(e),
                "device": str(self.device),
            }

    def run_benchmark_suite(self) -> pd.DataFrame:
        """Run complete benchmark suite."""
        self.logger.info("Starting comprehensive benchmark suite...")

        total_runs = (
            len(self.config.model_sizes)
            * len(self.config.sequence_lengths)
            * len(self.config.batch_sizes)
            * len(self.config.dataset_sizes)
        )

        self.logger.info(f"Total benchmark runs: {total_runs}")

        run_count = 0
        for model_size in self.config.model_sizes:
            for seq_length in self.config.sequence_lengths:
                for batch_size in self.config.batch_sizes:
                    for dataset_size in self.config.dataset_sizes:
                        run_count += 1
                        self.logger.info(f"Run {run_count}/{total_runs}")

                        # Skip combinations that might cause OOM
                        estimated_memory = self._estimate_memory_usage(
                            model_size, seq_length, batch_size
                        )

                        if estimated_memory > self.config.memory_limit_gb:
                            self.logger.warning(
                                f"Skipping {model_size}, {seq_length}, {batch_size} "
                                f"(estimated {estimated_memory:.1f}GB > {self.config.memory_limit_gb}GB)"
                            )
                            continue

                        result = self.benchmark_training(
                            model_size, seq_length, batch_size, dataset_size
                        )
                        self.results.append(result)

                        # Save intermediate results
                        if run_count % 10 == 0:
                            self.save_results()

        # Create DataFrame
        results_df = pd.DataFrame(self.results)

        # Save final results
        self.save_results()
        self.generate_report(results_df)

        return results_df

    def _estimate_memory_usage(
        self, model_size: str, seq_length: int, batch_size: int
    ) -> float:
        """Estimate memory usage in GB."""
        # Simple heuristic based on model size and sequence length
        size_multipliers = {"tiny": 0.1, "small": 0.5, "medium": 2.0, "large": 8.0}
        base_memory = size_multipliers.get(model_size, 1.0)

        # Factor in sequence length and batch size
        memory_estimate = base_memory * (seq_length / 512) * (batch_size / 8)

        return memory_estimate

    def save_results(self):
        """Save benchmark results to JSON."""
        results_file = os.path.join(self.config.output_dir, "benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {results_file}")

    def generate_report(self, results_df: pd.DataFrame):
        """Generate benchmark report."""
        if results_df.empty:
            self.logger.warning("No results to report")
            return

        # Filter successful runs
        successful_runs = results_df[results_df["success"]]

        if successful_runs.empty:
            self.logger.warning("No successful runs to report")
            return

        # Generate summary statistics
        report = {
            "summary": {
                "total_runs": len(results_df),
                "successful_runs": len(successful_runs),
                "failed_runs": len(results_df) - len(successful_runs),
                "device": str(self.device),
            },
            "performance_stats": {
                "avg_samples_per_second": successful_runs["samples_per_second"].mean(),
                "max_samples_per_second": successful_runs["samples_per_second"].max(),
                "min_samples_per_second": successful_runs["samples_per_second"].min(),
                "avg_training_time": successful_runs["training_time_seconds"].mean(),
                "max_memory_usage_gb": successful_runs.apply(
                    lambda row: (
                        max(row["memory_after"].values()) if row["memory_after"] else 0
                    ),
                    axis=1,
                ).max(),
            },
            "best_configurations": {
                "fastest_training": successful_runs.loc[
                    successful_runs["samples_per_second"].idxmax()
                ][
                    [
                        "model_size",
                        "sequence_length",
                        "batch_size",
                        "samples_per_second",
                    ]
                ].to_dict(),
                "most_memory_efficient": successful_runs.loc[
                    successful_runs.apply(
                        lambda row: (
                            max(row["memory_after"].values())
                            if row["memory_after"]
                            else float("inf")
                        ),
                        axis=1,
                    ).idxmin()
                ][["model_size", "sequence_length", "batch_size"]].to_dict(),
            },
        }

        # Save report
        report_file = os.path.join(self.config.output_dir, "benchmark_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Benchmark report saved to {report_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("HYENA-GLT PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total runs: {report['summary']['total_runs']}")
        print(f"Successful runs: {report['summary']['successful_runs']}")
        print(f"Failed runs: {report['summary']['failed_runs']}")
        print(f"Device: {report['summary']['device']}")
        print(
            f"\nAverage samples/second: {report['performance_stats']['avg_samples_per_second']:.2f}"
        )
        print(
            f"Max samples/second: {report['performance_stats']['max_samples_per_second']:.2f}"
        )
        print(
            f"Max memory usage: {report['performance_stats']['max_memory_usage_gb']:.2f} GB"
        )

        if "fastest_training" in report["best_configurations"]:
            fastest = report["best_configurations"]["fastest_training"]
            print("\nFastest configuration:")
            print(f"  Model: {fastest['model_size']}")
            print(f"  Sequence length: {fastest['sequence_length']}")
            print(f"  Batch size: {fastest['batch_size']}")
            print(f"  Speed: {fastest['samples_per_second']:.2f} samples/sec")


def main():
    """Main benchmark script."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Hyena-GLT performance")
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        default=["tiny", "small"],
        help="Model sizes to benchmark",
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[128, 256, 512],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[4, 8, 16],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--dataset-sizes",
        nargs="+",
        type=int,
        default=[100, 500],
        help="Dataset sizes to test",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--memory-limit", type=float, default=16.0, help="Memory limit in GB"
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create benchmark configuration
    config = BenchmarkConfig(
        model_sizes=args.model_sizes,
        sequence_lengths=args.sequence_lengths,
        batch_sizes=args.batch_sizes,
        dataset_sizes=args.dataset_sizes,
        num_epochs=args.epochs,
        device=args.device,
        memory_limit_gb=args.memory_limit,
        output_dir=args.output_dir,
    )

    # Run benchmark
    benchmark = PerformanceBenchmark(config)
    results_df = benchmark.run_benchmark_suite()

    print(f"\nBenchmark complete! Results saved to {config.output_dir}")
    print("Results summary:")
    print(results_df[results_df["success"]].describe())


if __name__ == "__main__":
    main()
