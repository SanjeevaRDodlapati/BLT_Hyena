"""
Performance benchmark tests for Hyena-GLT models.
"""

import gc
import time
from typing import Any

import psutil
import pytest
import torch

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data.dataset import GenomicDataset
from hyena_glt.data.tokenizer import DNATokenizer
from hyena_glt.model.hyena_glt import HyenaGLT, HyenaGLTForSequenceClassification
from tests.utils import DataGenerator, TestConfig


@pytest.mark.benchmark
class TestInferenceBenchmarks:
    """Benchmark inference performance."""

    def test_sequence_classification_speed(self):
        """Benchmark sequence classification inference speed."""
        model_config = HyenaGLTConfig(**TestConfig.MEDIUM_CONFIG)
        model_config.num_labels = 10

        model = HyenaGLTForSequenceClassification(model_config)
        model.eval()

        tokenizer = DNATokenizer()

        # Generate test sequences
        batch_sizes = [1, 4, 8, 16]
        sequence_lengths = [128, 256, 512, 1024]

        results = {}

        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # Generate batch
                sequences = []
                for _ in range(batch_size):
                    seq = DataGenerator.generate_dna_sequence(seq_len)
                    seq_str = "".join(["ATCG"[x] for x in seq])
                    sequences.append(seq_str)

                # Tokenize
                encoded = tokenizer.encode_batch(sequences)

                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(encoded)

                # Benchmark
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()

                with torch.no_grad():
                    for _ in range(10):
                        model(encoded)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()

                avg_time = (end_time - start_time) / 10
                sequences_per_second = batch_size / avg_time

                key = f"batch_{batch_size}_len_{seq_len}"
                results[key] = {
                    "time_per_batch": avg_time,
                    "sequences_per_second": sequences_per_second,
                    "tokens_per_second": (batch_size * seq_len) / avg_time,
                }

                # Memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                    results[key]["gpu_memory_mb"] = memory_mb

                # Performance assertions
                assert sequences_per_second > 0
                if seq_len <= 256:
                    assert sequences_per_second > 1  # At least 1 sequence/second

        # Print results for reference
        print("\nInference Benchmark Results:")
        for key, metrics in results.items():
            print(
                f"{key}: {metrics['sequences_per_second']:.2f} seq/s, "
                f"{metrics['tokens_per_second']:.0f} tokens/s"
            )

    def test_memory_efficiency(self):
        """Test memory efficiency across different model sizes."""
        configs = [
            ("small", TestConfig.SMALL_CONFIG),
            ("medium", TestConfig.MEDIUM_CONFIG),
        ]

        memory_results = {}

        for config_name, config_dict in configs:
            model_config = HyenaGLTConfig(**config_dict)
            model = HyenaGLT(model_config)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())

            # Estimate model size in MB
            model_size_mb = total_params * 4 / 1024**2  # 4 bytes per float32

            # Test inference memory
            batch_size, seq_len = 4, 256
            input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))

            if torch.cuda.is_available():
                model = model.cuda()
                input_ids = input_ids.cuda()
                torch.cuda.reset_peak_memory_stats()

            model.eval()
            with torch.no_grad():
                model(input_ids)

            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            else:
                # Use psutil for CPU memory (approximate)
                process = psutil.Process()
                peak_memory_mb = process.memory_info().rss / 1024**2

            memory_results[config_name] = {
                "parameters": total_params,
                "model_size_mb": model_size_mb,
                "peak_inference_memory_mb": peak_memory_mb,
                "memory_efficiency": (
                    total_params / peak_memory_mb if peak_memory_mb > 0 else 0
                ),
            }

        # Print memory results
        print("\nMemory Efficiency Results:")
        for config_name, metrics in memory_results.items():
            print(
                f"{config_name}: {metrics['parameters']:,} params, "
                f"{metrics['model_size_mb']:.1f}MB model, "
                f"{metrics['peak_inference_memory_mb']:.1f}MB peak"
            )

        # Basic assertions
        assert (
            memory_results["small"]["model_size_mb"]
            < memory_results["medium"]["model_size_mb"]
        )
        assert (
            memory_results["small"]["parameters"]
            < memory_results["medium"]["parameters"]
        )

    @pytest.mark.gpu
    def test_gpu_utilization(self):
        """Test GPU utilization efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model_config = HyenaGLTConfig(**TestConfig.MEDIUM_CONFIG)
        model = HyenaGLT(model_config).cuda()

        tokenizer = DNATokenizer()

        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16, 32]
        seq_len = 512

        utilization_results = {}

        for batch_size in batch_sizes:
            # Generate data
            sequences = []
            for _ in range(batch_size):
                seq = DataGenerator.generate_dna_sequence(seq_len)
                seq_str = "".join(["ATCG"[x] for x in seq])
                sequences.append(seq_str)

            encoded = tokenizer.encode_batch(sequences).cuda()

            # Warmup
            model.eval()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(encoded)

            # Benchmark with timing
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                for _ in range(20):
                    model(encoded)

            torch.cuda.synchronize()
            end_time = time.time()

            avg_time = (end_time - start_time) / 20
            throughput = batch_size / avg_time

            utilization_results[batch_size] = {
                "time_per_batch": avg_time,
                "throughput": throughput,
                "memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }

        # Print GPU utilization results
        print("\nGPU Utilization Results:")
        for batch_size, metrics in utilization_results.items():
            print(
                f"Batch {batch_size}: {metrics['throughput']:.1f} seq/s, "
                f"{metrics['memory_mb']:.0f}MB"
            )

        # Throughput should generally increase with batch size
        # (up to memory limits)
        throughputs = [
            metrics["throughput"] for metrics in utilization_results.values()
        ]
        assert max(throughputs) > min(throughputs)


@pytest.mark.benchmark
class TestTrainingBenchmarks:
    """Benchmark training performance."""

    def test_training_speed(self):
        """Test training speed across different configurations."""
        from hyena_glt.training.config import TrainingConfig
        from hyena_glt.training.trainer import HyenaGLTTrainer

        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model_config.num_labels = 3

        training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=4,
            num_epochs=1,
            eval_steps=100,  # Disable evaluation for pure speed test
            logging_steps=10,
        )

        model = HyenaGLTForSequenceClassification(model_config)
        tokenizer = DNATokenizer()

        # Generate training data
        sequences = []
        labels = []
        for i in range(40):
            seq_length = 200
            seq = DataGenerator.generate_dna_sequence(seq_length)
            seq_str = "".join(["ATCG"[x] for x in seq])
            sequences.append(seq_str)
            labels.append(i % 3)

        dataset = GenomicDataset(
            sequences=sequences, labels=labels, tokenizer=tokenizer, max_length=256
        )

        # Benchmark training
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.output_dir = temp_dir
            training_config.save_steps = 1000  # Disable saving for speed

            trainer = HyenaGLTTrainer(
                model=model,
                config=training_config,
                train_dataset=dataset,
                tokenizer=tokenizer,
            )

            start_time = time.time()
            trainer.train(max_steps=20)  # Train for 20 steps
            end_time = time.time()

            total_time = end_time - start_time
            steps_per_second = 20 / total_time
            samples_per_second = steps_per_second * training_config.batch_size

            print(
                f"\nTraining Speed: {steps_per_second:.2f} steps/s, "
                f"{samples_per_second:.2f} samples/s"
            )

            # Should complete training in reasonable time
            assert steps_per_second > 0.1  # At least 0.1 steps per second

    def test_gradient_accumulation_performance(self):
        """Test performance with gradient accumulation."""
        from hyena_glt.training.config import TrainingConfig
        from hyena_glt.training.trainer import HyenaGLTTrainer

        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(model_config)
        tokenizer = DNATokenizer()

        # Small dataset
        sequences = ["ATCG" * 50 for _ in range(16)]
        dataset = GenomicDataset(
            sequences=sequences,
            tokenizer=tokenizer,
            max_length=256,
            task_type="generation",
        )

        # Test different accumulation settings
        accumulation_steps = [1, 2, 4]
        results = {}

        for accum_steps in accumulation_steps:
            training_config = TrainingConfig(
                learning_rate=1e-4,
                batch_size=2,
                gradient_accumulation_steps=accum_steps,
                logging_steps=5,
            )

            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                training_config.output_dir = temp_dir

                trainer = HyenaGLTTrainer(
                    model=model,
                    config=training_config,
                    train_dataset=dataset,
                    tokenizer=tokenizer,
                )

                start_time = time.time()
                trainer.train(max_steps=10)
                end_time = time.time()

                results[accum_steps] = end_time - start_time

        print("\nGradient Accumulation Performance:")
        for steps, time_taken in results.items():
            print(f"Accumulation {steps}: {time_taken:.2f}s")

        # All should complete successfully
        assert all(time_taken > 0 for time_taken in results.values())


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Test model scalability."""

    def test_sequence_length_scaling(self):
        """Test performance scaling with sequence length."""
        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(model_config)
        model.eval()

        tokenizer = DNATokenizer()
        batch_size = 2
        sequence_lengths = [64, 128, 256, 512, 1024]

        scaling_results = {}

        for seq_len in sequence_lengths:
            # Generate sequences
            sequences = []
            for _ in range(batch_size):
                seq = DataGenerator.generate_dna_sequence(seq_len)
                seq_str = "".join(["ATCG"[x] for x in seq])
                sequences.append(seq_str)

            encoded = tokenizer.encode_batch(sequences)

            # Measure inference time
            start_time = time.time()

            with torch.no_grad():
                for _ in range(5):
                    model(encoded)

            end_time = time.time()

            avg_time = (end_time - start_time) / 5
            tokens_per_second = (batch_size * seq_len) / avg_time

            scaling_results[seq_len] = {
                "time_per_batch": avg_time,
                "tokens_per_second": tokens_per_second,
            }

        print("\nSequence Length Scaling:")
        for seq_len, metrics in scaling_results.items():
            print(f"Length {seq_len}: {metrics['tokens_per_second']:.0f} tokens/s")

        # Check that performance doesn't degrade too severely
        # (Hyena should scale better than quadratic attention)
        shortest_len = min(sequence_lengths)
        longest_len = max(sequence_lengths)

        shortest_speed = scaling_results[shortest_len]["tokens_per_second"]
        longest_speed = scaling_results[longest_len]["tokens_per_second"]

        # Speed shouldn't drop by more than 10x for sequence length increase
        length_ratio = longest_len / shortest_len
        speed_ratio = shortest_speed / longest_speed

        assert speed_ratio < length_ratio  # Should scale better than linear

    def test_batch_size_scaling(self):
        """Test performance scaling with batch size."""
        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(model_config)
        model.eval()

        tokenizer = DNATokenizer()
        seq_len = 256
        batch_sizes = [1, 2, 4, 8, 16]

        batch_scaling_results = {}

        for batch_size in batch_sizes:
            # Generate batch
            sequences = []
            for _ in range(batch_size):
                seq = DataGenerator.generate_dna_sequence(seq_len)
                seq_str = "".join(["ATCG"[x] for x in seq])
                sequences.append(seq_str)

            encoded = tokenizer.encode_batch(sequences)

            # Measure time
            start_time = time.time()

            with torch.no_grad():
                for _ in range(10):
                    model(encoded)

            end_time = time.time()

            avg_time = (end_time - start_time) / 10
            sequences_per_second = batch_size / avg_time

            batch_scaling_results[batch_size] = {
                "time_per_batch": avg_time,
                "sequences_per_second": sequences_per_second,
                "efficiency": sequences_per_second / batch_size,
            }

        print("\nBatch Size Scaling:")
        for batch_size, metrics in batch_scaling_results.items():
            print(
                f"Batch {batch_size}: {metrics['sequences_per_second']:.1f} seq/s, "
                f"efficiency {metrics['efficiency']:.3f}"
            )

        # Throughput should generally increase with batch size
        throughputs = [
            metrics["sequences_per_second"]
            for metrics in batch_scaling_results.values()
        ]

        assert max(throughputs) > min(throughputs)


@pytest.mark.benchmark
@pytest.mark.memory_intensive
class TestMemoryBenchmarks:
    """Test memory usage patterns."""

    def test_memory_growth_with_sequence_length(self):
        """Test memory growth patterns."""
        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(model_config)

        if torch.cuda.is_available():
            model = model.cuda()

        sequence_lengths = [128, 256, 512, 1024]
        batch_size = 4
        memory_usage = {}

        for seq_len in sequence_lengths:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Generate input
            input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # Forward pass
            model.eval()
            with torch.no_grad():
                model(input_ids)

            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            else:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024**2

            memory_usage[seq_len] = memory_mb

        print("\nMemory Usage by Sequence Length:")
        for seq_len, memory_mb in memory_usage.items():
            print(f"Length {seq_len}: {memory_mb:.1f}MB")

        # Memory should grow sub-quadratically (better than attention)
        lengths = list(memory_usage.keys())
        memories = list(memory_usage.values())

        # Check that memory growth is reasonable
        for i in range(1, len(lengths)):
            length_ratio = lengths[i] / lengths[i - 1]
            memory_ratio = memories[i] / memories[i - 1]

            # Memory growth should be less than quadratic
            assert memory_ratio < length_ratio**1.5

    def test_training_memory_stability(self):
        """Test memory stability during training."""
        from hyena_glt.training.config import TrainingConfig
        from hyena_glt.training.trainer import HyenaGLTTrainer

        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(model_config)

        training_config = TrainingConfig(
            learning_rate=1e-4, batch_size=2, logging_steps=5
        )

        tokenizer = DNATokenizer()

        # Small dataset
        sequences = ["ATCG" * 100 for _ in range(20)]
        dataset = GenomicDataset(
            sequences=sequences,
            tokenizer=tokenizer,
            max_length=512,
            task_type="generation",
        )

        if torch.cuda.is_available():
            model = model.cuda()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.output_dir = temp_dir
            training_config.save_steps = 1000  # Disable saving

            trainer = HyenaGLTTrainer(
                model=model,
                config=training_config,
                train_dataset=dataset,
                tokenizer=tokenizer,
            )

            # Train for several steps and monitor memory
            initial_memory = 0
            peak_memory = 0

            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() / 1024**2

            trainer.train(max_steps=15)

            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2

            print(
                f"\nTraining Memory: Initial {initial_memory:.1f}MB, "
                f"Peak {peak_memory:.1f}MB"
            )

            # Memory usage should be reasonable
            if torch.cuda.is_available():
                assert peak_memory < 2000  # Less than 2GB for small model
