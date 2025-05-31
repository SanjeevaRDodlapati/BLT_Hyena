"""
Benchmarking utilities for genomic sequence models.
"""

import gc
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from .metrics import BenchmarkEvaluator


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""

    name: str
    description: str
    datasets: list[str]
    metrics: list[str]
    batch_sizes: list[int] = None
    sequence_lengths: list[int] = None
    num_runs: int = 3
    warmup_runs: int = 1
    device: str = "cuda"
    precision: str = "fp32"  # fp32, fp16, bf16
    output_dir: str = "./benchmark_results"


@dataclass
class BenchmarkResult:
    """Results from a benchmarking experiment."""

    config: BenchmarkConfig
    model_name: str
    model_params: int
    results: dict[str, Any]
    system_info: dict[str, Any]
    timestamp: str


class ModelProfiler:
    """Profiler for model computational characteristics."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }

    def measure_memory_usage(
        self, input_shapes: dict[str, tuple[int, ...]]
    ) -> dict[str, float]:
        """Measure model memory usage."""
        if not torch.cuda.is_available():
            return {}

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Baseline memory
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        # Create sample inputs
        inputs = {}
        for key, shape in input_shapes.items():
            if "input_ids" in key or "labels" in key:
                inputs[key] = torch.randint(0, 1000, shape, device=self.device)
            else:
                inputs[key] = torch.randn(shape, device=self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(**inputs)

        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        model_memory = peak_memory - baseline_memory

        return {
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "model_memory_mb": model_memory,
        }

    def measure_flops(
        self, input_shapes: dict[str, tuple[int, ...]], num_samples: int = 10
    ) -> dict[str, float]:
        """Estimate FLOPs for model inference."""
        try:
            from fvcore.nn import FlopCountMode, flop_count

            # Create sample inputs
            inputs = {}
            for key, shape in input_shapes.items():
                if "input_ids" in key or "labels" in key:
                    inputs[key] = torch.randint(0, 1000, shape, device=self.device)
                else:
                    inputs[key] = torch.randn(shape, device=self.device)

            # Count FLOPs
            flop_dict, _ = flop_count(
                self.model,
                inputs,
                supported_ops={torch.nn.Conv1d, torch.nn.Linear, torch.nn.LayerNorm},
            )

            total_flops = sum(flop_dict.values())

            return {
                "total_flops": total_flops,
                "flops_per_param": total_flops
                / self.count_parameters()["total_parameters"],
            }

        except ImportError:
            print("fvcore not available for FLOP counting")
            return {}

    def profile_inference_speed(
        self,
        input_shapes: dict[str, tuple[int, ...]],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> dict[str, float]:
        """Profile model inference speed."""
        # Create sample inputs
        inputs = {}
        for key, shape in input_shapes.items():
            if "input_ids" in key or "labels" in key:
                inputs[key] = torch.randint(0, 1000, shape, device=self.device)
            else:
                inputs[key] = torch.randn(shape, device=self.device)

        self.model.eval()

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(**inputs)

        # Synchronize if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Time inference
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()

            with torch.no_grad():
                _ = self.model(**inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        times = np.array(times)

        return {
            "mean_inference_time": np.mean(times),
            "std_inference_time": np.std(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "median_inference_time": np.median(times),
            "throughput_samples_per_sec": input_shapes["input_ids"][0] / np.mean(times),
        }


class ScalabilityBenchmark:
    """Benchmark model scalability across different configurations."""

    def __init__(self, model_factory: Callable, config: BenchmarkConfig):
        self.model_factory = model_factory
        self.config = config

    def run_batch_size_scaling(
        self, base_config: dict, batch_sizes: list[int]
    ) -> dict[str, list[dict]]:
        """Benchmark across different batch sizes."""
        results = {"batch_sizes": batch_sizes, "metrics": []}

        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")

            # Create model
            model = self.model_factory(base_config)
            profiler = ModelProfiler(model)

            # Input shape with current batch size
            input_shapes = {
                "input_ids": (batch_size, base_config.get("max_length", 512))
            }

            try:
                # Profile this configuration
                memory_metrics = profiler.measure_memory_usage(input_shapes)
                speed_metrics = profiler.profile_inference_speed(input_shapes)

                batch_results = {
                    "batch_size": batch_size,
                    "memory_metrics": memory_metrics,
                    "speed_metrics": speed_metrics,
                }

                results["metrics"].append(batch_results)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch size {batch_size}")
                    results["metrics"].append(
                        {"batch_size": batch_size, "error": "OOM"}
                    )
                    break
                else:
                    raise e

        return results

    def run_sequence_length_scaling(
        self, base_config: dict, sequence_lengths: list[int]
    ) -> dict[str, list[dict]]:
        """Benchmark across different sequence lengths."""
        results = {"sequence_lengths": sequence_lengths, "metrics": []}

        for seq_len in sequence_lengths:
            print(f"Testing sequence length: {seq_len}")

            # Update config for this sequence length
            config_copy = base_config.copy()
            config_copy["max_length"] = seq_len

            # Create model
            model = self.model_factory(config_copy)
            profiler = ModelProfiler(model)

            # Input shape with current sequence length
            input_shapes = {"input_ids": (base_config.get("batch_size", 8), seq_len)}

            try:
                # Profile this configuration
                memory_metrics = profiler.measure_memory_usage(input_shapes)
                speed_metrics = profiler.profile_inference_speed(input_shapes)
                param_metrics = profiler.count_parameters()

                seq_results = {
                    "sequence_length": seq_len,
                    "parameter_metrics": param_metrics,
                    "memory_metrics": memory_metrics,
                    "speed_metrics": speed_metrics,
                }

                results["metrics"].append(seq_results)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at sequence length {seq_len}")
                    results["metrics"].append(
                        {"sequence_length": seq_len, "error": "OOM"}
                    )
                    break
                else:
                    raise e

        return results


class ComparisonBenchmark:
    """Compare multiple models on the same tasks."""

    def __init__(self, models: dict[str, torch.nn.Module], tasks: dict[str, Any]):
        self.models = models
        self.tasks = tasks

    def run_comparison(
        self, data_loaders: dict[str, Any], device: str = "cuda"
    ) -> dict[str, dict]:
        """Run comprehensive comparison across all models and tasks."""
        results = {}

        for model_name, model in self.models.items():
            print(f"Evaluating model: {model_name}")
            model_results = {}

            # Move model to device
            model = model.to(device)

            # Profile model characteristics
            profiler = ModelProfiler(model)
            model_results["parameters"] = profiler.count_parameters()

            # Evaluate on each task
            for task_name, task_config in self.tasks.items():
                if task_name in data_loaders:
                    print(f"  Task: {task_name}")

                    # Create evaluator for this task
                    evaluator = BenchmarkEvaluator({"tasks": {task_name: task_config}})

                    # Run evaluation
                    task_results = evaluator.evaluate_model(
                        model, data_loaders[task_name], device
                    )

                    model_results[task_name] = task_results

            results[model_name] = model_results

        return results


class SystemProfiler:
    """Profile system characteristics for benchmarking."""

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "cpu": {
                "model": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
                "cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
            },
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            "pytorch_version": torch.__version__,
        }

        # GPU information
        if torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory
                / (1024**3),
                "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
            }

        return info


class BenchmarkRunner:
    """Main benchmark runner orchestrating all benchmarking experiments."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_benchmark(
        self,
        model_factory: Callable,
        base_model_config: dict,
        data_loaders: dict | None = None,
    ) -> BenchmarkResult:
        """Run comprehensive benchmark suite."""
        print(f"Running benchmark: {self.config.name}")

        # Get system information
        system_info = SystemProfiler.get_system_info()

        # Create base model for parameter counting
        base_model = model_factory(base_model_config)
        profiler = ModelProfiler(base_model)
        model_params = profiler.count_parameters()["total_parameters"]

        results = {
            "parameter_analysis": profiler.count_parameters(),
            "system_info": system_info,
        }

        # Scalability benchmarks
        if self.config.batch_sizes:
            scalability = ScalabilityBenchmark(model_factory, self.config)
            results["batch_size_scaling"] = scalability.run_batch_size_scaling(
                base_model_config, self.config.batch_sizes
            )

        if self.config.sequence_lengths:
            scalability = ScalabilityBenchmark(model_factory, self.config)
            results["sequence_length_scaling"] = (
                scalability.run_sequence_length_scaling(
                    base_model_config, self.config.sequence_lengths
                )
            )

        # Task evaluation benchmarks
        if data_loaders:
            task_configs = {
                dataset: {"type": "classification", "num_classes": 2}
                for dataset in self.config.datasets
            }

            evaluator = BenchmarkEvaluator({"tasks": task_configs})

            for dataset_name, data_loader in data_loaders.items():
                if dataset_name in self.config.datasets:
                    model = model_factory(base_model_config)
                    dataset_results = evaluator.evaluate_model(
                        model, data_loader, self.config.device
                    )
                    results[f"dataset_{dataset_name}"] = dataset_results

        # Create benchmark result
        benchmark_result = BenchmarkResult(
            config=self.config,
            model_name=base_model_config.get("model_name", "HyenaGLT"),
            model_params=model_params,
            results=results,
            system_info=system_info,
            timestamp=time.strftime("%Y-%m-%d_%H-%M-%S"),
        )

        # Save results
        self.save_results(benchmark_result)

        return benchmark_result

    def save_results(self, benchmark_result: BenchmarkResult):
        """Save benchmark results to disk."""
        # Convert to dictionary for JSON serialization
        result_dict = asdict(benchmark_result)

        # Save detailed results
        output_file = (
            self.output_dir / f"{self.config.name}_{benchmark_result.timestamp}.json"
        )
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"Benchmark results saved to: {output_file}")

        # Save summary
        summary_file = self.output_dir / f"{self.config.name}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Benchmark: {self.config.name}\n")
            f.write(f"Description: {self.config.description}\n")
            f.write(f"Timestamp: {benchmark_result.timestamp}\n")
            f.write(f"Model: {benchmark_result.model_name}\n")
            f.write(f"Parameters: {benchmark_result.model_params:,}\n\n")

            f.write("System Information:\n")
            for key, value in benchmark_result.system_info.items():
                f.write(f"  {key}: {value}\n")

        print(f"Benchmark summary saved to: {summary_file}")


def create_genomic_benchmark_suite() -> list[BenchmarkConfig]:
    """Create predefined benchmark configurations for genomic tasks."""

    benchmarks = []

    # Speed and memory benchmark
    benchmarks.append(
        BenchmarkConfig(
            name="genomic_scalability",
            description="Scalability analysis across batch sizes and sequence lengths",
            datasets=["dna_classification", "protein_classification"],
            metrics=["inference_time", "memory_usage", "throughput"],
            batch_sizes=[1, 4, 8, 16, 32],
            sequence_lengths=[512, 1024, 2048, 4096, 8192],
            num_runs=10,
            warmup_runs=3,
        )
    )

    # Accuracy benchmark
    benchmarks.append(
        BenchmarkConfig(
            name="genomic_accuracy",
            description="Accuracy evaluation on genomic classification tasks",
            datasets=["promoter_detection", "splice_site_prediction", "protein_family"],
            metrics=["accuracy", "f1", "auc_roc", "mcc"],
            num_runs=3,
        )
    )

    # Long sequence benchmark
    benchmarks.append(
        BenchmarkConfig(
            name="long_sequence_performance",
            description="Performance on very long genomic sequences",
            datasets=["genome_annotation", "long_noncoding_rna"],
            metrics=["accuracy", "inference_time", "memory_usage"],
            sequence_lengths=[8192, 16384, 32768],
            batch_sizes=[1, 2, 4],
            num_runs=5,
        )
    )

    return benchmarks
