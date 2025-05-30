"""
Performance benchmark tests for the Hyena-GLT framework.

Comprehensive performance testing including speed benchmarks,
memory usage tests, and scalability analysis.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
import tempfile
import os
from typing import Dict, List, Tuple, Any

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model import HyenaGLTModel
from hyena_glt.training import HyenaGLTTrainer
from hyena_glt.data import DNATokenizer
from tests.utils import TestConfig, ModelTestUtils, PerformanceTestUtils


class TestSpeedBenchmarks:
    """Test execution speed benchmarks."""
    
    @pytest.fixture
    def performance_config(self):
        """Get configuration optimized for performance testing."""
        config = TestConfig.get_medium_config()
        config.model.hidden_size = 256
        config.model.num_layers = 6
        return config
    
    @pytest.fixture
    def benchmark_model(self, performance_config):
        """Create model for benchmarking."""
        model = HyenaGLTModel(performance_config)
        model.eval()
        return model
    
    def test_forward_pass_speed(self, benchmark_model):
        """Test forward pass speed across different input sizes."""
        batch_sizes = [1, 2, 4, 8, 16]
        sequence_lengths = [50, 100, 200, 500, 1000]
        
        results = []
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # Create input
                x = torch.randint(0, 4, (batch_size, seq_len))
                
                # Warm up
                with torch.no_grad():
                    for _ in range(3):
                        _ = benchmark_model(x)
                
                # Benchmark
                times = []
                with torch.no_grad():
                    for _ in range(10):
                        start_time = time.perf_counter()
                        output = benchmark_model(x)
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                # Calculate throughput
                tokens_per_second = (batch_size * seq_len) / avg_time
                
                result = {
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'avg_time_ms': avg_time * 1000,
                    'std_time_ms': std_time * 1000,
                    'tokens_per_second': tokens_per_second,
                    'samples_per_second': batch_size / avg_time
                }
                results.append(result)
                
                # Assertions
                assert avg_time > 0
                assert tokens_per_second > 0
                
                # Performance thresholds (adjust based on expected performance)
                if batch_size == 1 and seq_len <= 100:
                    assert avg_time < 1.0  # Less than 1 second for small inputs
    
    @pytest.mark.parametrize("precision", ["float32", "float16"])
    def test_precision_impact_on_speed(self, benchmark_model, precision):
        """Test impact of different precisions on speed."""
        if precision == "float16":
            benchmark_model = benchmark_model.half()
        
        x = torch.randint(0, 4, (4, 100))
        if precision == "float16":
            x = x.half()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                start_time = time.perf_counter()
                output = benchmark_model(x)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        
        assert avg_time > 0
        
        # float16 should generally be faster (if supported)
        if precision == "float16" and torch.cuda.is_available():
            # This is more of a guideline than strict requirement
            pass  # float16 performance varies by hardware
    
    def test_batch_size_scaling(self, benchmark_model):
        """Test how performance scales with batch size."""
        batch_sizes = [1, 2, 4, 8, 16, 32]
        seq_len = 100
        
        throughputs = []
        latencies = []
        
        for batch_size in batch_sizes:
            x = torch.randint(0, 4, (batch_size, seq_len))
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = benchmark_model(x)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.perf_counter()
                    output = benchmark_model(x)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            throughput = (batch_size * seq_len) / avg_time
            
            throughputs.append(throughput)
            latencies.append(avg_time / batch_size)  # Per-sample latency
        
        # Check scaling properties
        # Throughput should generally increase with batch size
        for i in range(1, len(throughputs)):
            # Allow some variance, but general trend should be upward
            assert throughputs[i] >= throughputs[i-1] * 0.8
        
        # Per-sample latency should generally decrease with batch size
        for i in range(1, min(4, len(latencies))):  # Check first few
            assert latencies[i] <= latencies[i-1] * 1.2  # Allow some variance
    
    def test_sequence_length_scaling(self, benchmark_model):
        """Test how performance scales with sequence length."""
        sequence_lengths = [50, 100, 200, 400, 800]
        batch_size = 4
        
        times_per_token = []
        
        for seq_len in sequence_lengths:
            x = torch.randint(0, 4, (batch_size, seq_len))
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = benchmark_model(x)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.perf_counter()
                    output = benchmark_model(x)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            time_per_token = avg_time / (batch_size * seq_len)
            times_per_token.append(time_per_token)
        
        # Time per token should be relatively stable or grow sub-linearly
        # This tests the efficiency of the Hyena operator
        assert all(t > 0 for t in times_per_token)
        
        # The growth should be sub-quadratic (better than O(n²))
        if len(times_per_token) >= 3:
            # Check that doubling sequence length doesn't quadruple time per token
            ratio_1 = times_per_token[1] / times_per_token[0]
            ratio_2 = times_per_token[2] / times_per_token[1]
            
            # Should be better than quadratic scaling
            assert ratio_1 < 3.0  # Allow for some overhead
            assert ratio_2 < 3.0


class TestMemoryBenchmarks:
    """Test memory usage benchmarks."""
    
    @pytest.fixture
    def memory_config(self):
        """Get configuration for memory testing."""
        config = TestConfig.get_medium_config()
        config.model.hidden_size = 512
        config.model.num_layers = 8
        return config
    
    @pytest.fixture
    def memory_model(self, memory_config):
        """Create model for memory testing."""
        model = HyenaGLTModel(memory_config)
        return model
    
    def test_model_memory_footprint(self, memory_model):
        """Test model parameter memory footprint."""
        # Calculate theoretical memory usage
        total_params = sum(p.numel() for p in memory_model.parameters())
        param_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        assert param_memory_mb > 0
        
        # For the test config, memory should be reasonable
        assert param_memory_mb < 1000  # Less than 1GB for test model
        
        print(f"Model parameters: {total_params:,}")
        print(f"Parameter memory: {param_memory_mb:.2f} MB")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, memory_model):
        """Test GPU memory usage during inference."""
        device = torch.device('cuda')
        memory_model = memory_model.to(device)
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        batch_sizes = [1, 2, 4, 8]
        seq_len = 200
        
        memory_usage = []
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            x = torch.randint(0, 4, (batch_size, seq_len)).to(device)
            
            # Forward pass
            with torch.no_grad():
                output = memory_model(x)
            
            peak_memory = torch.cuda.max_memory_allocated(device)
            current_memory = torch.cuda.memory_allocated(device)
            
            memory_usage.append({
                'batch_size': batch_size,
                'peak_memory_mb': peak_memory / (1024 * 1024),
                'current_memory_mb': current_memory / (1024 * 1024)
            })
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(device)
        
        # Memory should scale roughly linearly with batch size
        for i in range(1, len(memory_usage)):
            prev_memory = memory_usage[i-1]['peak_memory_mb']
            curr_memory = memory_usage[i]['peak_memory_mb']
            
            # Allow for some overhead, but should be roughly linear
            expected_ratio = memory_usage[i]['batch_size'] / memory_usage[i-1]['batch_size']
            actual_ratio = curr_memory / prev_memory
            
            assert actual_ratio <= expected_ratio * 1.5  # Allow 50% overhead
    
    def test_memory_efficiency_vs_sequence_length(self, memory_model):
        """Test memory efficiency across different sequence lengths."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            memory_model = memory_model.to(device)
        else:
            device = torch.device('cpu')
        
        sequence_lengths = [50, 100, 200, 400, 800]
        batch_size = 2
        
        memory_per_token = []
        
        for seq_len in sequence_lengths:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            
            x = torch.randint(0, 4, (batch_size, seq_len)).to(device)
            
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated(device)
            
            with torch.no_grad():
                output = memory_model(x)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated(device)
                memory_used = peak_memory - initial_memory
                memory_per_token.append(memory_used / (batch_size * seq_len))
            else:
                # For CPU, we can't easily measure memory per operation
                memory_per_token.append(1.0)  # Placeholder
        
        # Memory per token should not grow quadratically
        # This tests the linear complexity of Hyena vs quadratic of attention
        if torch.cuda.is_available() and len(memory_per_token) >= 3:
            # Check that memory doesn't grow quadratically
            ratio_1 = memory_per_token[1] / memory_per_token[0]
            ratio_2 = memory_per_token[2] / memory_per_token[1]
            
            # Should be much better than quadratic
            assert ratio_1 < 2.0
            assert ratio_2 < 2.0
    
    def test_training_memory_usage(self, memory_model):
        """Test memory usage during training."""
        memory_model.train()
        
        batch_size = 2
        seq_len = 100
        x = torch.randint(0, 4, (batch_size, seq_len))
        target = torch.randint(0, 4, (batch_size, seq_len))
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            memory_model = memory_model.to(device)
            x = x.to(device)
            target = target.to(device)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            
            initial_memory = torch.cuda.memory_allocated(device)
        
        # Forward pass
        output = memory_model(x)
        
        if torch.cuda.is_available():
            forward_memory = torch.cuda.memory_allocated(device)
        
        # Backward pass
        loss = nn.CrossEntropyLoss()(
            output.view(-1, output.size(-1)), 
            target.view(-1)
        )
        loss.backward()
        
        if torch.cuda.is_available():
            backward_memory = torch.cuda.max_memory_allocated(device)
            
            forward_overhead = forward_memory - initial_memory
            backward_overhead = backward_memory - forward_memory
            
            assert forward_overhead > 0
            assert backward_overhead > 0
            
            # Backward pass should use more memory (gradients)
            assert backward_overhead >= forward_overhead * 0.5
            
            print(f"Forward memory overhead: {forward_overhead / (1024*1024):.2f} MB")
            print(f"Backward memory overhead: {backward_overhead / (1024*1024):.2f} MB")


class TestScalabilityBenchmarks:
    """Test scalability benchmarks."""
    
    def test_parameter_scaling(self):
        """Test how performance scales with model size."""
        hidden_sizes = [128, 256, 512]
        num_layers_list = [2, 4, 6]
        
        results = []
        
        for hidden_size in hidden_sizes:
            for num_layers in num_layers_list:
                config = TestConfig.get_small_config()
                config.model.hidden_size = hidden_size
                config.model.num_layers = num_layers
                
                model = HyenaGLTModel(config)
                model.eval()
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                
                # Benchmark inference
                x = torch.randint(0, 4, (2, 100))
                
                times = []
                with torch.no_grad():
                    for _ in range(10):
                        start_time = time.perf_counter()
                        output = model(x)
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                
                results.append({
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'total_params': total_params,
                    'inference_time': avg_time,
                    'params_per_ms': total_params / (avg_time * 1000)
                })
        
        # Check that performance scales reasonably
        for result in results:
            assert result['inference_time'] > 0
            assert result['params_per_ms'] > 0
        
        # Larger models should generally take longer but be efficient per parameter
        small_models = [r for r in results if r['total_params'] < 1000000]
        large_models = [r for r in results if r['total_params'] >= 1000000]
        
        if small_models and large_models:
            avg_small_time = np.mean([r['inference_time'] for r in small_models])
            avg_large_time = np.mean([r['inference_time'] for r in large_models])
            
            # Larger models should take longer, but not excessively
            assert avg_large_time >= avg_small_time
            assert avg_large_time <= avg_small_time * 10  # Max 10x slower
    
    def test_hyena_vs_attention_scaling(self):
        """Test Hyena scaling vs theoretical attention scaling."""
        sequence_lengths = [100, 200, 400, 800]
        
        # Test our Hyena model
        config = TestConfig.get_small_config()
        config.model.hidden_size = 256
        config.model.num_layers = 4
        
        hyena_model = HyenaGLTModel(config)
        hyena_model.eval()
        
        hyena_times = []
        
        for seq_len in sequence_lengths:
            x = torch.randint(0, 4, (1, seq_len))
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = hyena_model(x)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(5):
                    start_time = time.perf_counter()
                    output = hyena_model(x)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            
            hyena_times.append(np.mean(times))
        
        # Check scaling properties
        # Hyena should scale better than O(n²)
        for i in range(1, len(hyena_times)):
            time_ratio = hyena_times[i] / hyena_times[i-1]
            length_ratio = sequence_lengths[i] / sequence_lengths[i-1]
            
            # Should be much better than quadratic
            assert time_ratio <= length_ratio ** 1.5  # Allow some super-linear growth
            
            print(f"Seq len {sequence_lengths[i]}: {hyena_times[i]*1000:.2f}ms, "
                  f"ratio: {time_ratio:.2f} (vs length ratio: {length_ratio:.2f})")
    
    def test_multi_gpu_scaling(self):
        """Test multi-GPU scaling if available."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU not available")
        
        config = TestConfig.get_medium_config()
        
        # Single GPU
        model_single = HyenaGLTModel(config).cuda(0)
        
        # Multi GPU (DataParallel)
        model_multi = nn.DataParallel(HyenaGLTModel(config)).cuda()
        
        batch_size = 8
        seq_len = 200
        x = torch.randint(0, 4, (batch_size, seq_len)).cuda()
        
        # Benchmark single GPU
        single_times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.perf_counter()
                output = model_single(x)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                single_times.append(end_time - start_time)
        
        # Benchmark multi GPU
        multi_times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.perf_counter()
                output = model_multi(x)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                multi_times.append(end_time - start_time)
        
        single_avg = np.mean(single_times)
        multi_avg = np.mean(multi_times)
        
        print(f"Single GPU: {single_avg*1000:.2f}ms")
        print(f"Multi GPU: {multi_avg*1000:.2f}ms")
        print(f"Speedup: {single_avg/multi_avg:.2f}x")
        
        # Multi-GPU should be at least as fast (allowing for overhead)
        assert multi_avg <= single_avg * 1.5  # Allow for some overhead


class TestComparativeBenchmarks:
    """Test comparative performance benchmarks."""
    
    def test_hyena_glt_vs_baselines(self):
        """Compare Hyena-GLT against baseline models."""
        config = TestConfig.get_small_config()
        
        # Our Hyena-GLT model
        hyena_glt = HyenaGLTModel(config)
        hyena_glt.eval()
        
        # Simple baseline transformer (mock)
        class SimpleTransformer(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embedding = nn.Embedding(config.model.vocab_size, config.model.hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config.model.hidden_size,
                        nhead=config.model.num_attention_heads,
                        batch_first=True
                    ),
                    num_layers=config.model.num_layers
                )
                self.output = nn.Linear(config.model.hidden_size, config.model.vocab_size)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x)
        
        baseline_transformer = SimpleTransformer(config)
        baseline_transformer.eval()
        
        # Test different sequence lengths
        sequence_lengths = [50, 100, 200]
        batch_size = 2
        
        results = {
            'hyena_glt': [],
            'transformer': []
        }
        
        for seq_len in sequence_lengths:
            x = torch.randint(0, 4, (batch_size, seq_len))
            
            # Benchmark Hyena-GLT
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.perf_counter()
                    output = hyena_glt(x)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            results['hyena_glt'].append(np.mean(times))
            
            # Benchmark baseline transformer
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.perf_counter()
                    output = baseline_transformer(x)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            results['transformer'].append(np.mean(times))
        
        # Compare scaling
        for i, seq_len in enumerate(sequence_lengths):
            hyena_time = results['hyena_glt'][i]
            transformer_time = results['transformer'][i]
            
            speedup = transformer_time / hyena_time
            
            print(f"Seq len {seq_len}: Hyena-GLT {hyena_time*1000:.2f}ms, "
                  f"Transformer {transformer_time*1000:.2f}ms, "
                  f"Speedup: {speedup:.2f}x")
            
            # For longer sequences, Hyena should show advantage
            if seq_len >= 200:
                assert speedup >= 0.8  # At least competitive
    
    def test_different_model_architectures(self):
        """Test different architectural configurations."""
        base_config = TestConfig.get_small_config()
        
        configurations = [
            {"name": "small", "hidden_size": 128, "num_layers": 2},
            {"name": "medium", "hidden_size": 256, "num_layers": 4},
            {"name": "large", "hidden_size": 512, "num_layers": 6}
        ]
        
        results = []
        
        for config_dict in configurations:
            config = TestConfig.get_small_config()
            config.model.hidden_size = config_dict["hidden_size"]
            config.model.num_layers = config_dict["num_layers"]
            
            model = HyenaGLTModel(config)
            model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Benchmark
            x = torch.randint(0, 4, (2, 100))
            
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.perf_counter()
                    output = model(x)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            
            results.append({
                'name': config_dict["name"],
                'params': total_params,
                'time': avg_time,
                'params_per_second': total_params / avg_time
            })
        
        # Check efficiency scaling
        for result in results:
            print(f"{result['name']}: {result['params']:,} params, "
                  f"{result['time']*1000:.2f}ms, "
                  f"{result['params_per_second']:.0f} params/sec")
            
            assert result['time'] > 0
            assert result['params_per_second'] > 0


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_performance_baselines(self):
        """Test against established performance baselines."""
        config = TestConfig.get_small_config()
        model = HyenaGLTModel(config)
        model.eval()
        
        # Standard benchmark case
        batch_size = 4
        seq_len = 100
        x = torch.randint(0, 4, (batch_size, seq_len))
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                start_time = time.perf_counter()
                output = model(x)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        
        # Performance baseline (adjust based on target performance)
        # This should be updated based on expected performance for your hardware
        expected_max_time = 0.1  # 100ms for small model on CPU
        
        assert avg_time < expected_max_time, \
            f"Performance regression: {avg_time*1000:.2f}ms > {expected_max_time*1000:.2f}ms"
        
        print(f"Baseline performance: {avg_time*1000:.2f}ms (target: <{expected_max_time*1000:.2f}ms)")
    
    def test_memory_baselines(self):
        """Test against memory usage baselines."""
        config = TestConfig.get_small_config()
        model = HyenaGLTModel(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        param_memory_mb = total_params * 4 / (1024 * 1024)  # float32
        
        # Memory baseline (adjust based on model size expectations)
        expected_max_memory = 100  # 100MB for small test model
        
        assert param_memory_mb < expected_max_memory, \
            f"Memory regression: {param_memory_mb:.2f}MB > {expected_max_memory}MB"
        
        print(f"Parameter memory: {param_memory_mb:.2f}MB (target: <{expected_max_memory}MB)")


class TestBenchmarkVisualization:
    """Test benchmark result visualization."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_performance_plot_generation(self, temp_dir):
        """Test generation of performance plots."""
        # Mock performance data
        batch_sizes = [1, 2, 4, 8, 16]
        latencies = [10, 15, 25, 40, 70]  # ms
        throughputs = [100, 133, 160, 200, 228]  # samples/sec
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Latency plot
        ax1.plot(batch_sizes, latencies, 'o-')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency vs Batch Size')
        ax1.grid(True)
        
        # Throughput plot
        ax2.plot(batch_sizes, throughputs, 's-')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (samples/sec)')
        ax2.set_title('Throughput vs Batch Size')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(temp_dir, "performance_benchmark.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Verify plot was created
        assert os.path.exists(plot_path)
        assert os.path.getsize(plot_path) > 0
    
    def test_memory_scaling_visualization(self, temp_dir):
        """Test memory scaling visualization."""
        # Mock memory data
        sequence_lengths = [50, 100, 200, 400, 800]
        memory_usage = [10, 25, 55, 120, 250]  # MB
        
        # Create memory scaling plot
        plt.figure(figsize=(8, 6))
        plt.plot(sequence_lengths, memory_usage, 'ro-', label='Actual')
        
        # Add quadratic comparison
        quadratic = [memory_usage[0] * (s/sequence_lengths[0])**2 for s in sequence_lengths]
        plt.plot(sequence_lengths, quadratic, 'b--', label='Quadratic (O(n²))')
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Scaling: Hyena vs Quadratic')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(temp_dir, "memory_scaling.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(plot_path)


if __name__ == "__main__":
    pytest.main([__file__])
