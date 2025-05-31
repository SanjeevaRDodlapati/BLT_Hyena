"""
Unit tests for evaluation module.

Tests the evaluation metrics, analysis tools, benchmarking utilities,
and visualization components of the Hyena-GLT framework.
"""

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.evaluation.analysis import (
    AttentionAnalyzer,
    ModelAnalyzer,
    PerformanceAnalyzer,
    RepresentationAnalyzer,
)
from hyena_glt.evaluation.benchmark import (
    GenomicBenchmark,
    LatencyBenchmark,
    MemoryBenchmark,
    TaskBenchmark,
    ThroughputBenchmark,
)
from hyena_glt.evaluation.metrics import (
    ClassificationMetrics,
    GenomicMetrics,
    MetricsAggregator,
    RegressionMetrics,
    SequenceMetrics,
)
from hyena_glt.evaluation.visualizers import (
    AttentionVisualizer,
    MetricsVisualizer,
    PerformanceVisualizer,
    SequenceVisualizer,
)
from tests.utils import DataGenerator, ModelTestUtils, TestConfig


class TestGenomicMetrics:
    """Test genomic-specific metrics."""

    def test_gc_content_accuracy(self):
        """Test GC content calculation accuracy."""
        metrics = GenomicMetrics()

        # Test cases
        sequences = ["ATCG", "GGCC", "AAAA", "GCGC"]
        expected_gc = [0.5, 1.0, 0.0, 1.0]

        for seq, expected in zip(sequences, expected_gc, strict=False):
            gc_content = metrics.calculate_gc_content(seq)
            assert abs(gc_content - expected) < 1e-6

    def test_motif_detection_metrics(self):
        """Test motif detection performance metrics."""
        metrics = GenomicMetrics()

        # Mock motif predictions and ground truth
        pred_motifs = [[(10, 15, "TATA"), (20, 25, "CAAT")]]
        true_motifs = [[(9, 14, "TATA"), (21, 26, "CAAT")]]

        precision, recall, f1 = metrics.motif_detection_metrics(
            pred_motifs, true_motifs, overlap_threshold=0.5
        )

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    def test_variant_calling_metrics(self):
        """Test variant calling performance metrics."""
        metrics = GenomicMetrics()

        # Mock variant predictions and ground truth
        pred_variants = [[(100, "A", "G"), (200, "T", "C")]]
        true_variants = [[(100, "A", "G"), (150, "G", "A")]]

        precision, recall, f1 = metrics.variant_calling_metrics(
            pred_variants, true_variants
        )

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    def test_phylogenetic_distance_metrics(self):
        """Test phylogenetic distance calculations."""
        metrics = GenomicMetrics()

        # Mock phylogenetic trees or distance matrices
        tree1 = Mock()
        tree2 = Mock()

        with patch.object(metrics, "robinson_foulds_distance") as mock_rf:
            mock_rf.return_value = 0.5
            distance = metrics.phylogenetic_distance(tree1, tree2)
            assert distance == 0.5


class TestSequenceMetrics:
    """Test sequence-level metrics."""

    def test_sequence_accuracy(self):
        """Test sequence-level accuracy calculation."""
        metrics = SequenceMetrics()

        pred = torch.tensor([[0, 1, 2, 3], [1, 1, 2, 0]])
        target = torch.tensor([[0, 1, 2, 3], [1, 0, 2, 3]])

        accuracy = metrics.sequence_accuracy(pred, target)
        expected_accuracy = 0.5  # One sequence matches exactly

        assert abs(accuracy - expected_accuracy) < 1e-6

    def test_edit_distance(self):
        """Test edit distance calculation."""
        metrics = SequenceMetrics()

        seq1 = [0, 1, 2, 3]
        seq2 = [0, 1, 3, 2]

        distance = metrics.edit_distance(seq1, seq2)
        assert distance == 2  # Two substitutions needed

    def test_bleu_score(self):
        """Test BLEU score calculation for sequences."""
        metrics = SequenceMetrics()

        pred_sequences = [[0, 1, 2, 3]]
        target_sequences = [[0, 1, 2, 3]]

        bleu = metrics.bleu_score(pred_sequences, target_sequences)
        assert bleu == 1.0  # Perfect match

    @pytest.mark.parametrize("mask_ratio", [0.0, 0.1, 0.5])
    def test_masked_accuracy(self, mask_ratio):
        """Test masked sequence accuracy."""
        metrics = SequenceMetrics()

        pred = torch.randn(10, 100, 4)  # (batch, seq_len, vocab_size)
        target = torch.randint(0, 4, (10, 100))
        mask = torch.rand(10, 100) > mask_ratio

        accuracy = metrics.masked_accuracy(pred, target, mask)
        assert 0 <= accuracy <= 1


class TestClassificationMetrics:
    """Test classification metrics."""

    def test_multi_class_metrics(self):
        """Test multi-class classification metrics."""
        metrics = ClassificationMetrics()

        pred = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
        target = torch.tensor([0, 1, 2])

        accuracy = metrics.accuracy(pred, target)
        precision = metrics.precision(pred, target, average="macro")
        recall = metrics.recall(pred, target, average="macro")
        f1 = metrics.f1_score(pred, target, average="macro")

        assert accuracy == 1.0  # All predictions correct
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_binary_classification_metrics(self):
        """Test binary classification metrics."""
        metrics = ClassificationMetrics()

        pred = torch.tensor([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2]])
        target = torch.tensor([0, 1, 0])

        auc = metrics.auc_roc(pred, target)
        assert 0 <= auc <= 1

    def test_confusion_matrix(self):
        """Test confusion matrix generation."""
        metrics = ClassificationMetrics()

        pred = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
        target = torch.tensor([0, 1])

        cm = metrics.confusion_matrix(pred, target, num_classes=3)
        assert cm.shape == (3, 3)
        assert cm.sum() == 2  # Two samples


class TestRegressionMetrics:
    """Test regression metrics."""

    def test_mse_mae_metrics(self):
        """Test MSE and MAE calculations."""
        metrics = RegressionMetrics()

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 1.9, 3.2])

        mse = metrics.mse(pred, target)
        mae = metrics.mae(pred, target)
        rmse = metrics.rmse(pred, target)

        assert mse > 0
        assert mae > 0
        assert rmse > 0
        assert rmse == torch.sqrt(mse)

    def test_r2_score(self):
        """Test R² score calculation."""
        metrics = RegressionMetrics()

        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])

        r2 = metrics.r2_score(pred, target)
        assert abs(r2 - 1.0) < 1e-6  # Perfect correlation


class TestMetricsAggregator:
    """Test metrics aggregation functionality."""

    def test_metric_aggregation(self):
        """Test aggregation of multiple metrics."""
        aggregator = MetricsAggregator()

        # Add sample metrics
        aggregator.add_metric("accuracy", 0.8)
        aggregator.add_metric("accuracy", 0.9)
        aggregator.add_metric("loss", 0.5)
        aggregator.add_metric("loss", 0.3)

        # Test aggregation
        avg_metrics = aggregator.get_average_metrics()
        assert avg_metrics["accuracy"] == 0.85
        assert avg_metrics["loss"] == 0.4

    def test_metric_history(self):
        """Test metric history tracking."""
        aggregator = MetricsAggregator()

        for i in range(5):
            aggregator.add_metric("loss", 1.0 - i * 0.1)

        history = aggregator.get_metric_history("loss")
        assert len(history) == 5
        assert history[0] == 1.0
        assert history[-1] == 0.6


class TestModelAnalyzer:
    """Test model analysis functionality."""

    @pytest.fixture
    def model_analyzer(self):
        config = TestConfig.get_small_config()
        return ModelAnalyzer(config)

    def test_parameter_analysis(self, model_analyzer):
        """Test parameter count and distribution analysis."""
        model = ModelTestUtils.create_test_model()

        param_stats = model_analyzer.analyze_parameters(model)

        assert "total_params" in param_stats
        assert "trainable_params" in param_stats
        assert "param_distribution" in param_stats
        assert param_stats["total_params"] > 0

    def test_gradient_analysis(self, model_analyzer):
        """Test gradient flow analysis."""
        model = ModelTestUtils.create_test_model()

        # Create dummy loss
        x = torch.randn(2, 10, 4)
        output = model(x)
        loss = output.mean()
        loss.backward()

        grad_stats = model_analyzer.analyze_gradients(model)

        assert "gradient_norm" in grad_stats
        assert "gradient_distribution" in grad_stats

    def test_activation_analysis(self, model_analyzer):
        """Test activation statistics analysis."""
        model = ModelTestUtils.create_test_model()
        x = torch.randn(2, 10, 4)

        with patch.object(model_analyzer, "register_hooks") as mock_hooks:
            model_analyzer.analyze_activations(model, x)
            mock_hooks.assert_called_once()


class TestAttentionAnalyzer:
    """Test attention analysis functionality."""

    @pytest.fixture
    def attention_analyzer(self):
        config = TestConfig.get_small_config()
        return AttentionAnalyzer(config)

    def test_attention_pattern_analysis(self, attention_analyzer):
        """Test attention pattern analysis."""
        # Mock attention weights
        attention_weights = torch.randn(1, 4, 10, 10)  # (batch, heads, seq, seq)

        patterns = attention_analyzer.analyze_attention_patterns(attention_weights)

        assert "head_diversity" in patterns
        assert "attention_entropy" in patterns
        assert "positional_bias" in patterns

    def test_attention_head_analysis(self, attention_analyzer):
        """Test individual attention head analysis."""
        attention_weights = torch.randn(1, 4, 10, 10)

        head_analysis = attention_analyzer.analyze_attention_heads(attention_weights)

        assert len(head_analysis) == 4  # Number of heads
        for head_stats in head_analysis:
            assert "mean_attention" in head_stats
            assert "max_attention" in head_stats


class TestRepresentationAnalyzer:
    """Test representation analysis functionality."""

    @pytest.fixture
    def repr_analyzer(self):
        config = TestConfig.get_small_config()
        return RepresentationAnalyzer(config)

    def test_representation_similarity(self, repr_analyzer):
        """Test representation similarity analysis."""
        repr1 = torch.randn(100, 128)  # (samples, features)
        repr2 = torch.randn(100, 128)

        similarity = repr_analyzer.compute_similarity(repr1, repr2)

        assert "cosine_similarity" in similarity
        assert "euclidean_distance" in similarity

    def test_dimensionality_analysis(self, repr_analyzer):
        """Test dimensionality and clustering analysis."""
        representations = torch.randn(100, 128)

        dim_analysis = repr_analyzer.analyze_dimensionality(representations)

        assert "intrinsic_dimension" in dim_analysis
        assert "pca_explained_variance" in dim_analysis


class TestPerformanceAnalyzer:
    """Test performance analysis functionality."""

    @pytest.fixture
    def perf_analyzer(self):
        config = TestConfig.get_small_config()
        return PerformanceAnalyzer(config)

    def test_latency_analysis(self, perf_analyzer):
        """Test latency analysis."""
        model = ModelTestUtils.create_test_model()
        x = torch.randn(1, 10, 4)

        latency_stats = perf_analyzer.measure_latency(model, x, num_runs=5)

        assert "mean_latency" in latency_stats
        assert "std_latency" in latency_stats
        assert "min_latency" in latency_stats
        assert "max_latency" in latency_stats

    def test_memory_analysis(self, perf_analyzer):
        """Test memory usage analysis."""
        model = ModelTestUtils.create_test_model()
        x = torch.randn(1, 10, 4)

        memory_stats = perf_analyzer.measure_memory(model, x)

        assert "peak_memory" in memory_stats
        assert "allocated_memory" in memory_stats

    def test_throughput_analysis(self, perf_analyzer):
        """Test throughput analysis."""
        model = ModelTestUtils.create_test_model()
        batch_sizes = [1, 2, 4]

        throughput_stats = perf_analyzer.measure_throughput(
            model, batch_sizes, sequence_length=10
        )

        assert len(throughput_stats) == len(batch_sizes)
        for stats in throughput_stats:
            assert "batch_size" in stats
            assert "throughput" in stats


class TestGenomicBenchmark:
    """Test genomic benchmarking functionality."""

    @pytest.fixture
    def benchmark(self):
        config = TestConfig.get_small_config()
        return GenomicBenchmark(config)

    def test_dna_classification_benchmark(self, benchmark):
        """Test DNA classification benchmark."""
        model = ModelTestUtils.create_test_model()

        # Mock dataset
        with patch.object(benchmark, "load_dna_dataset") as mock_load:
            mock_dataset = Mock()
            mock_load.return_value = mock_dataset

            results = benchmark.run_dna_classification_benchmark(
                model, task="promoter_prediction"
            )

            assert "accuracy" in results
            assert "f1_score" in results

    def test_protein_folding_benchmark(self, benchmark):
        """Test protein folding benchmark."""
        model = ModelTestUtils.create_test_model()

        with patch.object(benchmark, "load_protein_dataset") as mock_load:
            mock_dataset = Mock()
            mock_load.return_value = mock_dataset

            results = benchmark.run_protein_folding_benchmark(model)

            assert "contact_accuracy" in results
            assert "distance_mae" in results

    def test_rna_structure_benchmark(self, benchmark):
        """Test RNA structure benchmark."""
        model = ModelTestUtils.create_test_model()

        with patch.object(benchmark, "load_rna_dataset") as mock_load:
            mock_dataset = Mock()
            mock_load.return_value = mock_dataset

            results = benchmark.run_rna_structure_benchmark(model)

            assert "base_pair_accuracy" in results
            assert "structure_similarity" in results


class TestTaskBenchmark:
    """Test task-specific benchmarking."""

    @pytest.fixture
    def task_benchmark(self):
        config = TestConfig.get_small_config()
        return TaskBenchmark(config)

    def test_multi_task_benchmark(self, task_benchmark):
        """Test multi-task benchmark execution."""
        model = ModelTestUtils.create_test_model()

        tasks = ["classification", "regression", "generation"]

        with patch.object(task_benchmark, "run_single_task_benchmark") as mock_run:
            mock_run.return_value = {"accuracy": 0.8}

            results = task_benchmark.run_multi_task_benchmark(model, tasks)

            assert len(results) == len(tasks)
            assert mock_run.call_count == len(tasks)

    def test_cross_validation_benchmark(self, task_benchmark):
        """Test cross-validation benchmark."""
        model = ModelTestUtils.create_test_model()

        with patch.object(task_benchmark, "run_fold_benchmark") as mock_fold:
            mock_fold.return_value = {"accuracy": 0.8}

            results = task_benchmark.run_cross_validation_benchmark(
                model, task="classification", n_folds=3
            )

            assert "mean_accuracy" in results
            assert "std_accuracy" in results
            assert mock_fold.call_count == 3


class TestLatencyBenchmark:
    """Test latency benchmarking."""

    @pytest.fixture
    def latency_benchmark(self):
        return LatencyBenchmark()

    def test_forward_pass_latency(self, latency_benchmark):
        """Test forward pass latency measurement."""
        model = ModelTestUtils.create_test_model()
        batch_sizes = [1, 2, 4]
        sequence_lengths = [10, 50]

        results = latency_benchmark.measure_forward_latency(
            model, batch_sizes, sequence_lengths, num_runs=3
        )

        assert len(results) == len(batch_sizes) * len(sequence_lengths)
        for result in results:
            assert "batch_size" in result
            assert "sequence_length" in result
            assert "latency_ms" in result

    def test_generation_latency(self, latency_benchmark):
        """Test generation latency measurement."""
        model = ModelTestUtils.create_test_model()

        with patch.object(model, "generate") as mock_generate:
            mock_generate.return_value = torch.randint(0, 4, (1, 20))

            results = latency_benchmark.measure_generation_latency(
                model, input_length=10, generation_length=10
            )

            assert "total_latency" in results
            assert "tokens_per_second" in results


class TestMemoryBenchmark:
    """Test memory benchmarking."""

    @pytest.fixture
    def memory_benchmark(self):
        return MemoryBenchmark()

    def test_memory_usage_scaling(self, memory_benchmark):
        """Test memory usage scaling with batch size."""
        model = ModelTestUtils.create_test_model()
        batch_sizes = [1, 2, 4, 8]

        results = memory_benchmark.measure_memory_scaling(
            model, batch_sizes, sequence_length=10
        )

        assert len(results) == len(batch_sizes)
        for result in results:
            assert "batch_size" in result
            assert "peak_memory_mb" in result

    def test_gradient_memory(self, memory_benchmark):
        """Test gradient computation memory usage."""
        model = ModelTestUtils.create_test_model()

        results = memory_benchmark.measure_gradient_memory(
            model, batch_size=2, sequence_length=10
        )

        assert "forward_memory" in results
        assert "backward_memory" in results
        assert "total_memory" in results


class TestThroughputBenchmark:
    """Test throughput benchmarking."""

    @pytest.fixture
    def throughput_benchmark(self):
        return ThroughputBenchmark()

    def test_training_throughput(self, throughput_benchmark):
        """Test training throughput measurement."""
        model = ModelTestUtils.create_test_model()

        with patch("torch.cuda.is_available", return_value=False):
            results = throughput_benchmark.measure_training_throughput(
                model, batch_size=2, sequence_length=10, num_steps=5
            )

        assert "samples_per_second" in results
        assert "tokens_per_second" in results
        assert "steps_per_second" in results

    def test_inference_throughput(self, throughput_benchmark):
        """Test inference throughput measurement."""
        model = ModelTestUtils.create_test_model()

        results = throughput_benchmark.measure_inference_throughput(
            model, batch_size=4, sequence_length=10, num_runs=10
        )

        assert "samples_per_second" in results
        assert "tokens_per_second" in results


class TestMetricsVisualizer:
    """Test metrics visualization functionality."""

    @pytest.fixture
    def visualizer(self):
        return MetricsVisualizer()

    def test_training_curves(self, visualizer):
        """Test training curves visualization."""
        metrics_history = {
            "train_loss": [1.0, 0.8, 0.6, 0.4],
            "val_loss": [1.1, 0.9, 0.7, 0.5],
            "train_accuracy": [0.5, 0.6, 0.7, 0.8],
            "val_accuracy": [0.4, 0.5, 0.6, 0.7],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "training_curves.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                visualizer.plot_training_curves(metrics_history, save_path)
                mock_save.assert_called_once()

    def test_confusion_matrix_plot(self, visualizer):
        """Test confusion matrix visualization."""
        cm = np.array([[10, 2], [1, 8]])
        class_names = ["Class A", "Class B"]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "confusion_matrix.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                visualizer.plot_confusion_matrix(cm, class_names, save_path)
                mock_save.assert_called_once()

    def test_performance_comparison(self, visualizer):
        """Test performance comparison visualization."""
        models_performance = {
            "Model A": {"accuracy": 0.8, "f1": 0.75},
            "Model B": {"accuracy": 0.85, "f1": 0.8},
            "Model C": {"accuracy": 0.9, "f1": 0.88},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "performance_comparison.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                visualizer.plot_performance_comparison(models_performance, save_path)
                mock_save.assert_called_once()


class TestAttentionVisualizer:
    """Test attention visualization functionality."""

    @pytest.fixture
    def attention_visualizer(self):
        return AttentionVisualizer()

    def test_attention_heatmap(self, attention_visualizer):
        """Test attention heatmap visualization."""
        attention_weights = torch.randn(1, 4, 10, 10)
        tokens = ["A", "T", "C", "G"] * 2 + ["A", "T"]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "attention_heatmap.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                attention_visualizer.plot_attention_heatmap(
                    attention_weights[0, 0], tokens, save_path
                )
                mock_save.assert_called_once()

    def test_attention_patterns(self, attention_visualizer):
        """Test attention patterns visualization."""
        attention_weights = torch.randn(5, 4, 10, 10)  # Multiple layers

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "attention_patterns.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                attention_visualizer.plot_attention_patterns(
                    attention_weights, save_path
                )
                mock_save.assert_called_once()


class TestSequenceVisualizer:
    """Test sequence visualization functionality."""

    @pytest.fixture
    def sequence_visualizer(self):
        return SequenceVisualizer()

    def test_sequence_alignment(self, sequence_visualizer):
        """Test sequence alignment visualization."""
        sequences = ["ATCGATCG", "ATCGATCG", "ACCGATCG"]
        labels = ["Reference", "Prediction", "Variant"]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "sequence_alignment.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                sequence_visualizer.plot_sequence_alignment(
                    sequences, labels, save_path
                )
                mock_save.assert_called_once()

    def test_motif_visualization(self, sequence_visualizer):
        """Test motif visualization."""
        motif_matrix = np.random.rand(4, 8)  # 4 nucleotides, 8 positions

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "motif.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                sequence_visualizer.plot_motif(motif_matrix, save_path)
                mock_save.assert_called_once()


class TestPerformanceVisualizer:
    """Test performance visualization functionality."""

    @pytest.fixture
    def performance_visualizer(self):
        return PerformanceVisualizer()

    def test_latency_scaling(self, performance_visualizer):
        """Test latency scaling visualization."""
        batch_sizes = [1, 2, 4, 8, 16]
        latencies = [10, 15, 25, 40, 70]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "latency_scaling.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                performance_visualizer.plot_latency_scaling(
                    batch_sizes, latencies, save_path
                )
                mock_save.assert_called_once()

    def test_memory_usage(self, performance_visualizer):
        """Test memory usage visualization."""
        sequence_lengths = [10, 50, 100, 200]
        memory_usage = [100, 400, 800, 1600]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "memory_usage.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                performance_visualizer.plot_memory_usage(
                    sequence_lengths, memory_usage, save_path
                )
                mock_save.assert_called_once()

    def test_throughput_comparison(self, performance_visualizer):
        """Test throughput comparison visualization."""
        models = ["Model A", "Model B", "Model C"]
        throughputs = [100, 150, 200]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "throughput_comparison.png")

            with patch("matplotlib.pyplot.savefig") as mock_save:
                performance_visualizer.plot_throughput_comparison(
                    models, throughputs, save_path
                )
                mock_save.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
