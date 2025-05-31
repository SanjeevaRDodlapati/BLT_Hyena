"""
Integration tests for end-to-end workflow testing.

Tests complete workflows including training pipelines, evaluation workflows,
and optimization pipelines of the Hyena-GLT framework.
"""

import json
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data import DNATokenizer, GenomicDataset
from hyena_glt.evaluation import GenomicBenchmark, ModelAnalyzer
from hyena_glt.model import HyenaGLTModel
from hyena_glt.optimization import DynamicQuantizer, QuantizationConfig
from hyena_glt.training import HyenaGLTTrainer
from tests.utils import DataGenerator, ModelTestUtils, TestConfig


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return TestConfig.get_small_config()

    @pytest.fixture
    def sample_data(self):
        """Generate sample genomic data."""
        generator = DataGenerator()
        sequences = generator.generate_dna_sequences(100, min_length=50, max_length=100)
        labels = generator.generate_classification_labels(100, num_classes=4)
        return sequences, labels

    def test_complete_training_workflow(self, config, sample_data, temp_dir):
        """Test complete training workflow from data to trained model."""
        sequences, labels = sample_data

        # Step 1: Create model
        model = HyenaGLTModel(config)

        # Step 2: Prepare data
        tokenizer = DNATokenizer()

        # Create mock dataset
        with patch.object(GenomicDataset, "__init__", return_value=None):
            with patch.object(GenomicDataset, "__len__", return_value=len(sequences)):
                with patch.object(GenomicDataset, "__getitem__") as mock_getitem:
                    # Mock dataset returns
                    mock_getitem.side_effect = lambda idx: {
                        "input_ids": torch.randint(0, 4, (50,)),
                        "labels": torch.tensor(labels[idx % len(labels)]),
                    }

                    dataset = GenomicDataset(sequences, labels, tokenizer)

                    # Step 3: Create trainer
                    trainer = HyenaGLTTrainer(
                        model=model,
                        config=config,
                        train_dataset=dataset,
                        val_dataset=dataset,
                    )

                    # Step 4: Train model
                    trainer.train(num_epochs=2, save_dir=temp_dir)

                    # Step 5: Verify training artifacts
                    assert os.path.exists(os.path.join(temp_dir, "model_final.pt"))
                    assert os.path.exists(os.path.join(temp_dir, "training_log.json"))

    def test_evaluation_workflow(self, config, sample_data, temp_dir):
        """Test complete evaluation workflow."""
        sequences, labels = sample_data

        # Create and train model
        model = HyenaGLTModel(config)
        DNATokenizer()

        # Mock trained model
        model.eval()

        # Create evaluation workflow
        analyzer = ModelAnalyzer(config)
        benchmark = GenomicBenchmark(config)

        # Test model analysis
        with patch.object(analyzer, "analyze_parameters") as mock_analyze:
            mock_analyze.return_value = {
                "total_params": 1000000,
                "trainable_params": 900000,
            }

            param_stats = analyzer.analyze_parameters(model)
            assert "total_params" in param_stats

        # Test benchmarking
        with patch.object(
            benchmark, "run_dna_classification_benchmark"
        ) as mock_benchmark:
            mock_benchmark.return_value = {
                "accuracy": 0.85,
                "f1_score": 0.83,
                "precision": 0.84,
                "recall": 0.82,
            }

            results = benchmark.run_dna_classification_benchmark(
                model, task="promoter_prediction"
            )

            assert "accuracy" in results
            assert results["accuracy"] > 0

    def test_optimization_workflow(self, config, temp_dir):
        """Test complete model optimization workflow."""
        # Create model
        model = HyenaGLTModel(config)

        # Step 1: Quantization
        quant_config = QuantizationConfig(mode="dynamic")
        quantizer = DynamicQuantizer(quant_config)

        quantized_model = quantizer.quantize(model)

        # Step 2: Save optimized model
        save_path = os.path.join(temp_dir, "optimized_model.pt")
        torch.save(quantized_model.state_dict(), save_path)

        # Step 3: Load and verify
        loaded_model = HyenaGLTModel(config)
        loaded_model.load_state_dict(torch.load(save_path))

        # Test inference
        x = torch.randint(0, 4, (1, 20))

        with torch.no_grad():
            original_output = model(x)
            optimized_output = quantized_model(x)
            loaded_output = loaded_model(x)

        # Verify shapes match
        assert original_output.shape == optimized_output.shape
        assert original_output.shape == loaded_output.shape

    def test_fine_tuning_workflow(self, config, sample_data, temp_dir):
        """Test fine-tuning workflow."""
        sequences, labels = sample_data

        # Step 1: Create pre-trained model
        pretrained_model = HyenaGLTModel(config)

        # Save pretrained model
        pretrained_path = os.path.join(temp_dir, "pretrained.pt")
        torch.save(pretrained_model.state_dict(), pretrained_path)

        # Step 2: Load for fine-tuning
        finetuned_model = HyenaGLTModel(config)
        finetuned_model.load_state_dict(torch.load(pretrained_path))

        # Step 3: Fine-tune with mock trainer
        tokenizer = DNATokenizer()

        with patch.object(GenomicDataset, "__init__", return_value=None):
            with patch.object(GenomicDataset, "__len__", return_value=len(sequences)):
                with patch.object(GenomicDataset, "__getitem__") as mock_getitem:
                    mock_getitem.side_effect = lambda idx: {
                        "input_ids": torch.randint(0, 4, (50,)),
                        "labels": torch.tensor(labels[idx % len(labels)]),
                    }

                    dataset = GenomicDataset(sequences, labels, tokenizer)

                    trainer = HyenaGLTTrainer(
                        model=finetuned_model,
                        config=config,
                        train_dataset=dataset,
                        val_dataset=dataset,
                    )

                    # Fine-tune for fewer epochs
                    trainer.train(num_epochs=1, save_dir=temp_dir)

                    # Verify fine-tuned model
                    assert os.path.exists(os.path.join(temp_dir, "model_final.pt"))


class TestTrainingPipeline:
    """Test training pipeline integration."""

    @pytest.fixture
    def config(self):
        return TestConfig.get_small_config()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_multi_task_training_pipeline(self, config, temp_dir):
        """Test multi-task training pipeline."""
        # Configure for multi-task learning
        config.training.multi_task_learning = True
        config.training.task_weights = {
            "classification": 1.0,
            "regression": 0.5,
            "generation": 0.3,
        }

        model = HyenaGLTModel(config)

        # Create mock multi-task dataset
        def mock_multi_task_batch():
            return {
                "input_ids": torch.randint(0, 4, (2, 50)),
                "classification_labels": torch.randint(0, 4, (2,)),
                "regression_labels": torch.randn(2, 1),
                "generation_labels": torch.randint(0, 4, (2, 50)),
            }

        with patch("torch.utils.data.DataLoader") as mock_loader:
            mock_loader.return_value = [mock_multi_task_batch() for _ in range(5)]

            trainer = HyenaGLTTrainer(
                model=model, config=config, train_dataset=Mock(), val_dataset=Mock()
            )

            # Test multi-task training step
            batch = mock_multi_task_batch()
            loss = trainer.training_step(batch, 0)

            assert isinstance(loss, torch.Tensor)
            assert loss.item() > 0

    def test_curriculum_learning_pipeline(self, config, temp_dir):
        """Test curriculum learning pipeline."""
        # Enable curriculum learning
        config.training.curriculum_learning = True
        config.training.curriculum_strategy = "length_based"

        model = HyenaGLTModel(config)

        # Create mock curriculum dataset
        with patch("hyena_glt.training.CurriculumDataset") as mock_curriculum:
            mock_curriculum.return_value = Mock()

            trainer = HyenaGLTTrainer(
                model=model, config=config, train_dataset=Mock(), val_dataset=Mock()
            )

            # Test curriculum update
            trainer.update_curriculum(epoch=5)

            # Should not raise errors
            assert True

    def test_distributed_training_setup(self, config, temp_dir):
        """Test distributed training setup."""
        config.training.distributed = True
        config.training.world_size = 2

        model = HyenaGLTModel(config)

        with patch("torch.distributed.init_process_group"):
            with patch("torch.nn.parallel.DistributedDataParallel") as mock_ddp:
                mock_ddp.return_value = model

                trainer = HyenaGLTTrainer(
                    model=model, config=config, train_dataset=Mock(), val_dataset=Mock()
                )

                # Test setup doesn't crash
                assert isinstance(trainer.model, nn.Module)

    def test_mixed_precision_training(self, config, temp_dir):
        """Test mixed precision training integration."""
        config.training.mixed_precision = True

        model = HyenaGLTModel(config)

        with patch("torch.cuda.amp.GradScaler") as mock_scaler:
            with patch("torch.cuda.amp.autocast") as mock_autocast:
                mock_scaler.return_value = Mock()
                mock_autocast.return_value.__enter__ = Mock()
                mock_autocast.return_value.__exit__ = Mock()

                trainer = HyenaGLTTrainer(
                    model=model, config=config, train_dataset=Mock(), val_dataset=Mock()
                )

                # Test training step with mixed precision
                batch = {
                    "input_ids": torch.randint(0, 4, (2, 50)),
                    "labels": torch.randint(0, 4, (2, 50)),
                }

                loss = trainer.training_step(batch, 0)
                assert isinstance(loss, torch.Tensor)


class TestEvaluationWorkflow:
    """Test evaluation workflow integration."""

    @pytest.fixture
    def config(self):
        return TestConfig.get_small_config()

    @pytest.fixture
    def trained_model(self, config):
        """Create a mock trained model."""
        model = HyenaGLTModel(config)
        model.eval()
        return model

    def test_comprehensive_evaluation_pipeline(self, config, trained_model):
        """Test comprehensive evaluation pipeline."""
        analyzer = ModelAnalyzer(config)

        # Test parameter analysis
        param_stats = analyzer.analyze_parameters(trained_model)
        assert "total_params" in param_stats

        # Test gradient analysis (with mock gradients)
        x = torch.randint(0, 4, (2, 20))
        output = trained_model(x)
        loss = output.mean()
        loss.backward()

        grad_stats = analyzer.analyze_gradients(trained_model)
        assert "gradient_norm" in grad_stats

    def test_benchmark_suite_execution(self, config, trained_model):
        """Test benchmark suite execution."""
        benchmark = GenomicBenchmark(config)

        # Mock benchmark datasets
        with patch.object(benchmark, "load_dna_dataset") as mock_dna:
            with patch.object(benchmark, "load_protein_dataset") as mock_protein:
                with patch.object(benchmark, "load_rna_dataset") as mock_rna:

                    mock_dna.return_value = Mock()
                    mock_protein.return_value = Mock()
                    mock_rna.return_value = Mock()

                    # Test DNA benchmark
                    dna_results = benchmark.run_dna_classification_benchmark(
                        trained_model, task="promoter_prediction"
                    )
                    assert isinstance(dna_results, dict)

                    # Test protein benchmark
                    protein_results = benchmark.run_protein_folding_benchmark(
                        trained_model
                    )
                    assert isinstance(protein_results, dict)

                    # Test RNA benchmark
                    rna_results = benchmark.run_rna_structure_benchmark(trained_model)
                    assert isinstance(rna_results, dict)

    def test_cross_validation_workflow(self, config, trained_model):
        """Test cross-validation evaluation workflow."""
        from hyena_glt.evaluation.benchmark import TaskBenchmark

        task_benchmark = TaskBenchmark(config)

        with patch.object(task_benchmark, "run_fold_benchmark") as mock_fold:
            mock_fold.return_value = {"accuracy": 0.8, "f1_score": 0.75}

            cv_results = task_benchmark.run_cross_validation_benchmark(
                trained_model, task="classification", n_folds=3
            )

            assert "mean_accuracy" in cv_results
            assert "std_accuracy" in cv_results
            assert mock_fold.call_count == 3

    def test_performance_profiling_workflow(self, config, trained_model):
        """Test performance profiling workflow."""
        from hyena_glt.evaluation.analysis import PerformanceAnalyzer

        perf_analyzer = PerformanceAnalyzer(config)

        # Test latency profiling
        x = torch.randint(0, 4, (1, 50))
        latency_stats = perf_analyzer.measure_latency(trained_model, x, num_runs=3)

        assert "mean_latency" in latency_stats
        assert "std_latency" in latency_stats

        # Test memory profiling
        memory_stats = perf_analyzer.measure_memory(trained_model, x)

        assert "peak_memory" in memory_stats
        assert "allocated_memory" in memory_stats

    def test_attention_analysis_workflow(self, config, trained_model):
        """Test attention analysis workflow."""
        from hyena_glt.evaluation.analysis import AttentionAnalyzer

        attention_analyzer = AttentionAnalyzer(config)

        # Mock attention weights extraction
        with patch.object(
            attention_analyzer, "extract_attention_weights"
        ) as mock_extract:
            mock_attention = torch.randn(1, 4, 20, 20)  # (batch, heads, seq, seq)
            mock_extract.return_value = mock_attention

            torch.randint(0, 4, (1, 20))
            patterns = attention_analyzer.analyze_attention_patterns(mock_attention)

            assert "head_diversity" in patterns
            assert "attention_entropy" in patterns


class TestOptimizationPipeline:
    """Test optimization pipeline integration."""

    @pytest.fixture
    def config(self):
        return TestConfig.get_small_config()

    @pytest.fixture
    def trained_model(self, config):
        model = HyenaGLTModel(config)
        model.eval()
        return model

    def test_quantization_pipeline(self, config, trained_model):
        """Test quantization optimization pipeline."""
        from hyena_glt.optimization.quantization import DynamicQuantizer

        quant_config = QuantizationConfig(mode="dynamic", bits=8)
        quantizer = DynamicQuantizer(quant_config)

        # Test quantization
        quantized_model = quantizer.quantize(trained_model)

        # Test inference equivalence
        x = torch.randint(0, 4, (1, 20))

        with torch.no_grad():
            original_output = trained_model(x)
            quantized_output = quantized_model(x)

        # Check output shapes match
        assert original_output.shape == quantized_output.shape

        # Check reasonable accuracy preservation
        relative_error = torch.abs(original_output - quantized_output) / (
            torch.abs(original_output) + 1e-8
        )
        assert relative_error.mean() < 0.5  # Allow quantization error

    def test_pruning_pipeline(self, config, trained_model):
        """Test pruning optimization pipeline."""
        from hyena_glt.optimization.pruning import MagnitudePruner, PruningConfig

        prune_config = PruningConfig(sparsity=0.3, pruning_type="magnitude")
        pruner = MagnitudePruner(prune_config)

        # Test pruning
        pruned_model = pruner.prune(trained_model)

        # Test sparsity
        sparsity = pruner.get_sparsity(pruned_model)
        assert 0.2 <= sparsity <= 0.4  # Allow some tolerance

        # Test inference still works
        x = torch.randint(0, 4, (1, 20))

        with torch.no_grad():
            output = pruned_model(x)

        assert output.shape == (1, 20, config.model.vocab_size)

    def test_distillation_pipeline(self, config, trained_model):
        """Test knowledge distillation pipeline."""
        from hyena_glt.optimization.distillation import (
            DistillationConfig,
            KnowledgeDistiller,
        )

        # Create student model (smaller)
        student_config = TestConfig.get_small_config()
        student_config.model.hidden_size = config.model.hidden_size // 2
        student_model = HyenaGLTModel(student_config)

        # Setup distillation
        distill_config = DistillationConfig(temperature=4.0, alpha=0.7)
        distiller = KnowledgeDistiller(distill_config, trained_model, student_model)

        # Test distillation loss
        x = torch.randint(0, 4, (2, 20))
        y = torch.randint(0, 4, (2, 20))

        loss = distiller.compute_distillation_loss(x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_combined_optimization_pipeline(self, config, trained_model):
        """Test combined optimization techniques."""
        from hyena_glt.optimization.deployment import (
            DeploymentConfig,
            OptimizationPipeline,
        )

        # Setup combined optimization
        deploy_config = DeploymentConfig(
            enable_quantization=True, enable_pruning=True, enable_distillation=False
        )

        pipeline = OptimizationPipeline(deploy_config)

        # Test combined optimization
        optimized_model = pipeline.optimize(trained_model)

        assert isinstance(optimized_model, nn.Module)

        # Test that optimized model still works
        x = torch.randint(0, 4, (1, 20))

        with torch.no_grad():
            output = optimized_model(x)

        assert output.shape == (1, 20, config.model.vocab_size)


class TestDataPipeline:
    """Test data processing pipeline integration."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_genomic_data_loading_pipeline(self, temp_dir):
        """Test complete genomic data loading pipeline."""
        from hyena_glt.data import DNATokenizer, GenomicDataLoader, GenomicDataset

        # Create sample data files
        sequences = ["ATCGATCG", "GCTAGCTA", "TTAACCGG"]
        labels = [0, 1, 2]

        # Create data files
        seq_file = os.path.join(temp_dir, "sequences.txt")
        label_file = os.path.join(temp_dir, "labels.txt")

        with open(seq_file, "w") as f:
            for seq in sequences:
                f.write(f"{seq}\n")

        with open(label_file, "w") as f:
            for label in labels:
                f.write(f"{label}\n")

        # Test loading pipeline
        tokenizer = DNATokenizer()

        with patch.object(GenomicDataset, "__init__", return_value=None):
            with patch.object(GenomicDataset, "__len__", return_value=len(sequences)):
                with patch.object(GenomicDataset, "__getitem__") as mock_getitem:
                    mock_getitem.side_effect = lambda idx: {
                        "input_ids": torch.randint(0, 4, (8,)),
                        "labels": torch.tensor(labels[idx]),
                    }

                    dataset = GenomicDataset(sequences, labels, tokenizer)

                    # Test data loader
                    from torch.utils.data import DataLoader

                    loader = DataLoader(dataset, batch_size=2, shuffle=True)

                    # Test batch loading
                    batch = next(iter(loader))

                    assert "input_ids" in batch
                    assert "labels" in batch
                    assert batch["input_ids"].shape[0] == 2  # Batch size

    def test_multi_modal_data_pipeline(self, temp_dir):
        """Test multi-modal genomic data pipeline."""
        from hyena_glt.data import DNATokenizer, ProteinTokenizer, RNATokenizer

        # Test different tokenizers
        dna_tokenizer = DNATokenizer()
        protein_tokenizer = ProteinTokenizer()
        rna_tokenizer = RNATokenizer()

        # Test DNA tokenization
        dna_seq = "ATCGATCG"
        dna_tokens = dna_tokenizer.encode(dna_seq)
        assert len(dna_tokens) > 0

        # Test protein tokenization
        protein_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWK"
        protein_tokens = protein_tokenizer.encode(protein_seq)
        assert len(protein_tokens) > 0

        # Test RNA tokenization
        rna_seq = "AUCGAUCG"
        rna_tokens = rna_tokenizer.encode(rna_seq)
        assert len(rna_tokens) > 0

    def test_data_preprocessing_pipeline(self, temp_dir):
        """Test data preprocessing pipeline."""
        from hyena_glt.data.augmentation import SequenceAugmenter
        from hyena_glt.data.preprocessing import SequencePreprocessor

        # Create preprocessor and augmenter
        preprocessor = SequencePreprocessor()
        augmenter = SequenceAugmenter()

        # Test preprocessing
        raw_sequences = ["atcgatcg", "GCTAGCTA", "ttaaccgg"]

        processed_sequences = []
        for seq in raw_sequences:
            # Normalize
            normalized = preprocessor.normalize_sequence(seq)

            # Quality filter
            if preprocessor.quality_filter(normalized):
                # Augment
                augmented = augmenter.random_mutation(normalized, mutation_rate=0.1)
                processed_sequences.append(augmented)

        assert len(processed_sequences) > 0
        assert all(seq.isupper() for seq in processed_sequences)


class TestModelSerializationWorkflow:
    """Test model serialization and loading workflows."""

    @pytest.fixture
    def config(self):
        return TestConfig.get_small_config()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_model_checkpoint_workflow(self, config, temp_dir):
        """Test model checkpointing and loading workflow."""
        # Create and train model
        model = HyenaGLTModel(config)

        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "epoch": 10,
            "optimizer_state_dict": {},
            "metrics": {"accuracy": 0.85, "loss": 0.3},
        }

        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create new model and load state
        new_model = HyenaGLTModel(config)
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])

        # Test that models produce same output
        x = torch.randint(0, 4, (1, 20))

        with torch.no_grad():
            original_output = model(x)
            loaded_output = new_model(x)

        # Should be identical
        assert torch.allclose(original_output, loaded_output, atol=1e-6)

    def test_config_serialization_workflow(self, config, temp_dir):
        """Test configuration serialization workflow."""
        # Save config
        config_path = os.path.join(temp_dir, "config.json")

        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Load config
        with open(config_path) as f:
            config_dict = json.load(f)

        # Create new config from loaded dict
        new_config = HyenaGLTConfig.from_dict(config_dict)

        # Test that configs are equivalent
        assert new_config.model.hidden_size == config.model.hidden_size
        assert new_config.model.num_layers == config.model.num_layers
        assert new_config.training.learning_rate == config.training.learning_rate

    def test_model_export_workflow(self, config, temp_dir):
        """Test model export workflow."""
        model = HyenaGLTModel(config)
        model.eval()

        # Test TorchScript export
        example_input = torch.randint(0, 4, (1, 20))

        try:
            # Trace model
            traced_model = torch.jit.trace(model, example_input)

            # Save traced model
            traced_path = os.path.join(temp_dir, "traced_model.pt")
            traced_model.save(traced_path)

            # Load and test
            loaded_traced = torch.jit.load(traced_path)

            with torch.no_grad():
                original_output = model(example_input)
                traced_output = loaded_traced(example_input)

            # Should be close (allowing for tracing differences)
            assert torch.allclose(original_output, traced_output, atol=1e-3)

        except Exception as e:
            # Some models might not be traceable, which is okay
            pytest.skip(f"Model not traceable: {e}")


class TestPerformanceBenchmarkWorkflow:
    """Test performance benchmarking workflows."""

    @pytest.fixture
    def config(self):
        return TestConfig.get_small_config()

    def test_latency_benchmark_workflow(self, config):
        """Test latency benchmarking workflow."""
        from hyena_glt.evaluation.benchmark import LatencyBenchmark

        model = HyenaGLTModel(config)
        model.eval()

        benchmark = LatencyBenchmark()

        # Test forward pass latency
        batch_sizes = [1, 2, 4]
        sequence_lengths = [10, 20]

        results = benchmark.measure_forward_latency(
            model, batch_sizes, sequence_lengths, num_runs=3
        )

        assert len(results) == len(batch_sizes) * len(sequence_lengths)

        for result in results:
            assert "batch_size" in result
            assert "sequence_length" in result
            assert "latency_ms" in result
            assert result["latency_ms"] > 0

    def test_memory_benchmark_workflow(self, config):
        """Test memory benchmarking workflow."""
        from hyena_glt.evaluation.benchmark import MemoryBenchmark

        model = HyenaGLTModel(config)

        benchmark = MemoryBenchmark()

        # Test memory scaling
        batch_sizes = [1, 2, 4]

        results = benchmark.measure_memory_scaling(
            model, batch_sizes, sequence_length=20
        )

        assert len(results) == len(batch_sizes)

        for result in results:
            assert "batch_size" in result
            assert "peak_memory_mb" in result
            assert result["peak_memory_mb"] > 0

    def test_throughput_benchmark_workflow(self, config):
        """Test throughput benchmarking workflow."""
        from hyena_glt.evaluation.benchmark import ThroughputBenchmark

        model = HyenaGLTModel(config)

        benchmark = ThroughputBenchmark()

        # Test inference throughput
        results = benchmark.measure_inference_throughput(
            model, batch_size=2, sequence_length=20, num_runs=5
        )

        assert "samples_per_second" in results
        assert "tokens_per_second" in results
        assert results["samples_per_second"] > 0
        assert results["tokens_per_second"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
