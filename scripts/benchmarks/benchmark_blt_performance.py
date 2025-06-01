#!/usr/bin/env python3
"""
Performance Benchmark for BLT-Hyena Position Embedding System
================================================================

This script benchmarks the performance of the new BLT position embedding system
compared to the baseline approach, measuring:
- Forward pass latency
- Memory usage
- Throughput
- Computational efficiency across different sequence lengths
"""

import gc
import time
import tracemalloc

import numpy as np
import torch
import torch.nn as nn

# Import our models
from hyena_glt.model.hyena_glt import HyenaGLT


class OldGenomicPositionalEncoding(nn.Module):
    """Baseline implementation for comparison"""

    def __init__(self, d_model: int = 256, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class BaselineModel(nn.Module):
    """Simplified baseline model for comparison"""

    def __init__(self, d_model: int = 256, vocab_size: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = OldGenomicPositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True), num_layers=2
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.output_proj(x)


class PerformanceBenchmark:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Test configurations
        self.seq_lengths = [64, 128, 256, 512, 1024]
        self.batch_sizes = [1, 4, 8, 16]
        self.num_warmup = 5
        self.num_iterations = 20

        # Results storage
        self.results = {"blt_model": {}, "baseline_model": {}}

    def create_models(self, d_model: int = 256, vocab_size: int = 1000):
        """Create BLT and baseline models for comparison"""
        # Import the configuration
        from hyena_glt.config import HyenaGLTConfig

        # BLT Model Configuration
        blt_config = HyenaGLTConfig(
            genomic_vocab_size=vocab_size,
            hidden_size=d_model,
            num_layers=2,
            num_attention_heads=4,
            max_position_embeddings=2048,
        )
        blt_model = HyenaGLT(blt_config).to(self.device)

        # Baseline Model
        baseline_model = BaselineModel(d_model=d_model, vocab_size=vocab_size).to(
            self.device
        )

        return blt_model, baseline_model

    def measure_memory_usage(
        self, model: nn.Module, input_data: torch.Tensor
    ) -> dict[str, float]:
        """Measure memory usage during forward pass"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        tracemalloc.start()

        # Forward pass
        with torch.no_grad():
            _ = model(input_data)

        # Get memory statistics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_stats = {
            "cpu_current_mb": current / 1024 / 1024,
            "cpu_peak_mb": peak / 1024 / 1024,
        }

        if torch.cuda.is_available():
            memory_stats.update(
                {
                    "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                }
            )

        return memory_stats

    def measure_latency(
        self, model: nn.Module, input_data: torch.Tensor
    ) -> dict[str, float]:
        """Measure forward pass latency"""
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup):
                _ = model(input_data)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Actual measurement
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(self.num_iterations):
                _ = model(input_data)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        avg_latency = (end_time - start_time) / self.num_iterations

        return {
            "avg_latency_ms": avg_latency * 1000,
            "throughput_samples_per_sec": input_data.size(0) / avg_latency,
        }

    def run_single_benchmark(self, seq_len: int, batch_size: int) -> dict:
        """Run benchmark for a single configuration"""
        print(f"\nBenchmarking seq_len={seq_len}, batch_size={batch_size}")

        # Create models
        blt_model, baseline_model = self.create_models()

        # Create input data
        input_data = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)

        results = {}

        # Benchmark BLT model
        print("  Testing BLT model...")
        blt_memory = self.measure_memory_usage(blt_model, input_data)
        blt_latency = self.measure_latency(blt_model, input_data)

        results["blt"] = {**blt_memory, **blt_latency}

        # Benchmark baseline model
        print("  Testing baseline model...")
        baseline_memory = self.measure_memory_usage(baseline_model, input_data)
        baseline_latency = self.measure_latency(baseline_model, input_data)

        results["baseline"] = {**baseline_memory, **baseline_latency}

        # Calculate relative performance
        results["relative"] = {
            "latency_ratio": blt_latency["avg_latency_ms"]
            / baseline_latency["avg_latency_ms"],
            "memory_ratio": blt_memory["cpu_peak_mb"] / baseline_memory["cpu_peak_mb"],
            "throughput_ratio": blt_latency["throughput_samples_per_sec"]
            / baseline_latency["throughput_samples_per_sec"],
        }

        if torch.cuda.is_available():
            results["relative"]["gpu_memory_ratio"] = (
                blt_memory["gpu_peak_mb"] / baseline_memory["gpu_peak_mb"]
            )

        return results

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across all configurations"""
        print("=" * 60)
        print("BLT-Hyena Performance Benchmark")
        print("=" * 60)

        all_results = []

        for seq_len in self.seq_lengths:
            for batch_size in self.batch_sizes:
                try:
                    results = self.run_single_benchmark(seq_len, batch_size)
                    results["config"] = {"seq_len": seq_len, "batch_size": batch_size}
                    all_results.append(results)
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    continue

        return all_results

    def analyze_results(self, results: list[dict]):
        """Analyze and visualize benchmark results"""
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Extract data for analysis
        seq_lens = []
        batch_sizes = []
        latency_ratios = []
        memory_ratios = []
        throughput_ratios = []

        for result in results:
            config = result["config"]
            relative = result["relative"]

            seq_lens.append(config["seq_len"])
            batch_sizes.append(config["batch_size"])
            latency_ratios.append(relative["latency_ratio"])
            memory_ratios.append(relative["memory_ratio"])
            throughput_ratios.append(relative["throughput_ratio"])

        # Summary statistics
        print("\n📊 SUMMARY STATISTICS:")
        print(
            f"   Latency Ratio (BLT/Baseline): {np.mean(latency_ratios):.3f} ± {np.std(latency_ratios):.3f}"
        )
        print(
            f"   Memory Ratio (BLT/Baseline):  {np.mean(memory_ratios):.3f} ± {np.std(memory_ratios):.3f}"
        )
        print(
            f"   Throughput Ratio (BLT/Baseline): {np.mean(throughput_ratios):.3f} ± {np.std(throughput_ratios):.3f}"
        )

        # Detailed results table
        print("\n📋 DETAILED RESULTS:")
        print(
            f"{'Seq Len':<8} {'Batch':<6} {'Latency Ratio':<14} {'Memory Ratio':<13} {'Throughput Ratio':<16}"
        )
        print("-" * 70)

        for result in results:
            config = result["config"]
            relative = result["relative"]
            print(
                f"{config['seq_len']:<8} {config['batch_size']:<6} "
                f"{relative['latency_ratio']:<14.3f} {relative['memory_ratio']:<13.3f} "
                f"{relative['throughput_ratio']:<16.3f}"
            )

        # Performance insights
        print("\n🔍 PERFORMANCE INSIGHTS:")

        if np.mean(latency_ratios) < 1.1:
            print("   ✅ BLT latency overhead is minimal (<10%)")
        elif np.mean(latency_ratios) < 1.5:
            print("   ⚠️  BLT has moderate latency overhead (10-50%)")
        else:
            print("   ❌ BLT has significant latency overhead (>50%)")

        if np.mean(memory_ratios) < 1.2:
            print("   ✅ BLT memory overhead is acceptable (<20%)")
        elif np.mean(memory_ratios) < 2.0:
            print("   ⚠️  BLT has moderate memory overhead (20-100%)")
        else:
            print("   ❌ BLT has high memory overhead (>100%)")

        # Scaling analysis
        large_seq_results = [r for r in results if r["config"]["seq_len"] >= 512]
        if large_seq_results:
            large_seq_latency = np.mean(
                [r["relative"]["latency_ratio"] for r in large_seq_results]
            )
            print(f"   📈 Large sequence (≥512) latency ratio: {large_seq_latency:.3f}")

        return {
            "summary": {
                "avg_latency_ratio": np.mean(latency_ratios),
                "avg_memory_ratio": np.mean(memory_ratios),
                "avg_throughput_ratio": np.mean(throughput_ratios),
                "std_latency_ratio": np.std(latency_ratios),
                "std_memory_ratio": np.std(memory_ratios),
                "std_throughput_ratio": np.std(throughput_ratios),
            },
            "detailed_results": results,
        }


def main():
    """Run the complete performance benchmark"""
    benchmark = PerformanceBenchmark()

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()

    if not results:
        print("❌ No benchmark results obtained!")
        return

    # Analyze results
    analysis = benchmark.analyze_results(results)

    # Save results
    import os
    results_dir = os.path.join(os.getcwd(), "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "blt_performance_benchmark.pt")
    
    torch.save(
        {
            "results": results,
            "analysis": analysis,
            "benchmark_config": {
                "seq_lengths": benchmark.seq_lengths,
                "batch_sizes": benchmark.batch_sizes,
                "device": str(benchmark.device),
                "num_iterations": benchmark.num_iterations,
            },
        },
        results_file,
    )

    print(f"\n💾 Results saved to {results_file}")
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
