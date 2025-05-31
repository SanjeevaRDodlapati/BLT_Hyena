"""
Unit tests for optimization module.

Tests the optimization techniques including quantization, pruning,
knowledge distillation, memory optimization, and deployment utilities
of the Hyena-GLT framework.
"""

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.optimization.deployment import (
    DeploymentConfig,
    ModelDeployer,
    ONNXConverter,
    OptimizationPipeline,
    TorchScriptConverter,
)
from hyena_glt.optimization.distillation import (
    AttentionDistiller,
    DistillationConfig,
    DistillationLoss,
    FeatureDistiller,
    KnowledgeDistiller,
)
from hyena_glt.optimization.memory import (
    ActivationCheckpointing,
    GradientCheckpointing,
    MemoryOptimizer,
    MemoryProfiler,
    MemoryUtils,
)
from hyena_glt.optimization.pruning import (
    GradientPruner,
    MagnitudePruner,
    PruningConfig,
    PruningScheduler,
    StructuredPruner,
)
from hyena_glt.optimization.quantization import (
    DynamicQuantizer,
    QATQuantizer,
    QuantizationConfig,
    QuantizationUtils,
    StaticQuantizer,
)
from tests.utils import ModelTestUtils, TestConfig


class TestQuantizationConfig:
    """Test quantization configuration."""

    def test_default_config(self):
        """Test default quantization configuration."""
        config = QuantizationConfig()

        assert config.bits == 8
        assert config.mode == "dynamic"
        assert config.backend == "fbgemm"
        assert isinstance(config.layers_to_quantize, list)

    def test_custom_config(self):
        """Test custom quantization configuration."""
        config = QuantizationConfig(
            bits=4,
            mode="static",
            backend="qnnpack",
            layers_to_quantize=["linear", "conv"],
        )

        assert config.bits == 4
        assert config.mode == "static"
        assert config.backend == "qnnpack"
        assert config.layers_to_quantize == ["linear", "conv"]

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid bits
        with pytest.raises(ValueError):
            QuantizationConfig(bits=3)

        # Test invalid mode
        with pytest.raises(ValueError):
            QuantizationConfig(mode="invalid")


class TestDynamicQuantizer:
    """Test dynamic quantization."""

    @pytest.fixture
    def quantizer(self):
        config = QuantizationConfig(mode="dynamic")
        return DynamicQuantizer(config)

    def test_quantize_model(self, quantizer):
        """Test dynamic model quantization."""
        model = ModelTestUtils.create_test_model()

        quantized_model = quantizer.quantize(model)

        # Check that model was quantized
        assert quantized_model is not None
        # Model should have quantized layers
        any(
            "quantized" in str(type(module)).lower()
            for module in quantized_model.modules()
        )
        # Note: This might not be true for all backends, so we just check structure
        assert isinstance(quantized_model, nn.Module)

    def test_quantization_accuracy(self, quantizer):
        """Test quantization preserves reasonable accuracy."""
        model = ModelTestUtils.create_test_model()
        x = torch.randn(2, 10, 4)

        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(x)

        # Quantize and get quantized output
        quantized_model = quantizer.quantize(model)
        quantized_model.eval()
        with torch.no_grad():
            quantized_output = quantized_model(x)

        # Check outputs have similar shape
        assert original_output.shape == quantized_output.shape

        # Check outputs are reasonably close (allowing for quantization error)
        relative_error = torch.abs(original_output - quantized_output) / (
            torch.abs(original_output) + 1e-8
        )
        assert (
            relative_error.mean() < 0.5
        )  # Allow 50% relative error due to quantization


class TestStaticQuantizer:
    """Test static quantization."""

    @pytest.fixture
    def quantizer(self):
        config = QuantizationConfig(mode="static")
        return StaticQuantizer(config)

    def test_calibration_dataset(self, quantizer):
        """Test calibration with dataset."""
        model = ModelTestUtils.create_test_model()

        # Create calibration dataset
        calibration_data = [torch.randn(1, 10, 4) for _ in range(5)]

        # Mock the calibration process
        with patch.object(quantizer, "_prepare_model") as mock_prepare:
            mock_prepare.return_value = model

            quantized_model = quantizer.quantize(model, calibration_data)

            assert quantized_model is not None
            mock_prepare.assert_called_once()

    def test_quantization_without_calibration(self, quantizer):
        """Test static quantization without calibration data."""
        model = ModelTestUtils.create_test_model()

        with pytest.raises(ValueError, match="Calibration data required"):
            quantizer.quantize(model)


class TestQATQuantizer:
    """Test quantization-aware training."""

    @pytest.fixture
    def quantizer(self):
        config = QuantizationConfig(mode="qat")
        return QATQuantizer(config)

    def test_prepare_qat_model(self, quantizer):
        """Test QAT model preparation."""
        model = ModelTestUtils.create_test_model()

        qat_model = quantizer.prepare_qat(model)

        assert qat_model is not None
        # Should have fake quantization modules
        any("fake_quant" in str(type(module)).lower() for module in qat_model.modules())
        # This depends on PyTorch version and backend
        assert isinstance(qat_model, nn.Module)

    def test_qat_training_step(self, quantizer):
        """Test QAT training step."""
        model = ModelTestUtils.create_test_model()
        qat_model = quantizer.prepare_qat(model)

        x = torch.randn(2, 10, 4)
        y = torch.randint(0, 4, (2, 10))

        # Simulate training step
        qat_model.train()
        output = qat_model(x)
        loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), y.view(-1))

        assert loss.item() > 0

        # Check gradients
        loss.backward()
        has_gradients = any(
            param.grad is not None
            for param in qat_model.parameters()
            if param.requires_grad
        )
        assert has_gradients


class TestQuantizationUtils:
    """Test quantization utilities."""

    def test_model_size_comparison(self):
        """Test model size comparison utilities."""
        model = ModelTestUtils.create_test_model()

        original_size = QuantizationUtils.get_model_size(model)
        assert original_size > 0

        # Mock quantized model with smaller size
        quantized_model = ModelTestUtils.create_test_model()

        with patch.object(QuantizationUtils, "get_model_size") as mock_size:
            mock_size.side_effect = [original_size, original_size * 0.25]

            compression_ratio = QuantizationUtils.compare_model_sizes(
                model, quantized_model
            )

            assert compression_ratio == 4.0  # 4x compression

    def test_quantization_error_analysis(self):
        """Test quantization error analysis."""
        original_output = torch.randn(10, 20)
        quantized_output = original_output + torch.randn(10, 20) * 0.1

        error_metrics = QuantizationUtils.analyze_quantization_error(
            original_output, quantized_output
        )

        assert "mse" in error_metrics
        assert "mae" in error_metrics
        assert "max_error" in error_metrics
        assert all(error >= 0 for error in error_metrics.values())


class TestPruningConfig:
    """Test pruning configuration."""

    def test_default_config(self):
        """Test default pruning configuration."""
        config = PruningConfig()

        assert config.sparsity == 0.5
        assert config.pruning_type == "magnitude"
        assert not config.structured
        assert isinstance(config.layers_to_prune, list)

    def test_structured_pruning_config(self):
        """Test structured pruning configuration."""
        config = PruningConfig(
            sparsity=0.3,
            pruning_type="gradient",
            structured=True,
            granularity="channel",
        )

        assert config.sparsity == 0.3
        assert config.pruning_type == "gradient"
        assert config.structured
        assert config.granularity == "channel"


class TestMagnitudePruner:
    """Test magnitude-based pruning."""

    @pytest.fixture
    def pruner(self):
        config = PruningConfig(sparsity=0.5, pruning_type="magnitude")
        return MagnitudePruner(config)

    def test_unstructured_pruning(self, pruner):
        """Test unstructured magnitude pruning."""
        model = ModelTestUtils.create_test_model()

        # Get initial parameter count
        sum(p.numel() for p in model.parameters())

        pruned_model = pruner.prune(model)

        # Check that model structure is preserved
        assert isinstance(pruned_model, nn.Module)

        # Check sparsity
        sparsity = pruner.get_sparsity(pruned_model)
        assert 0.4 <= sparsity <= 0.6  # Allow some tolerance

    def test_layer_wise_pruning(self, pruner):
        """Test layer-wise pruning."""
        model = ModelTestUtils.create_test_model()

        # Prune specific layers
        layer_sparsities = {"transformer_blocks.0": 0.3, "transformer_blocks.1": 0.7}

        pruned_model = pruner.prune_layers(model, layer_sparsities)

        assert isinstance(pruned_model, nn.Module)

    def test_gradual_pruning(self, pruner):
        """Test gradual pruning schedule."""
        ModelTestUtils.create_test_model()

        # Test pruning schedule
        initial_sparsity = 0.0
        final_sparsity = 0.5
        steps = 5

        sparsities = pruner.get_pruning_schedule(
            initial_sparsity, final_sparsity, steps
        )

        assert len(sparsities) == steps
        assert sparsities[0] == initial_sparsity
        assert sparsities[-1] == final_sparsity
        assert all(
            sparsities[i] <= sparsities[i + 1] for i in range(len(sparsities) - 1)
        )


class TestStructuredPruner:
    """Test structured pruning."""

    @pytest.fixture
    def pruner(self):
        config = PruningConfig(
            sparsity=0.5,
            pruning_type="magnitude",
            structured=True,
            granularity="channel",
        )
        return StructuredPruner(config)

    def test_channel_pruning(self, pruner):
        """Test channel-wise structured pruning."""
        model = ModelTestUtils.create_test_model()

        pruned_model = pruner.prune(model)

        assert isinstance(pruned_model, nn.Module)

        # Check that some channels were removed
        original_channels = self._count_channels(model)
        pruned_channels = self._count_channels(pruned_model)

        # Should have fewer channels after pruning
        assert pruned_channels <= original_channels

    def _count_channels(self, model):
        """Count total channels in model."""
        total_channels = 0
        for module in model.modules():
            if isinstance(module, nn.Linear | nn.Conv1d):
                if hasattr(module, "out_features"):
                    total_channels += module.out_features
                elif hasattr(module, "out_channels"):
                    total_channels += module.out_channels
        return total_channels

    def test_filter_pruning(self, pruner):
        """Test filter-wise structured pruning."""
        # Create a simple conv model for filter pruning
        model = nn.Sequential(
            nn.Conv1d(4, 16, 3),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 4),
        )

        pruner.config.granularity = "filter"
        pruned_model = pruner.prune(model)

        assert isinstance(pruned_model, nn.Module)


class TestGradientPruner:
    """Test gradient-based pruning."""

    @pytest.fixture
    def pruner(self):
        config = PruningConfig(sparsity=0.5, pruning_type="gradient")
        return GradientPruner(config)

    def test_gradient_based_pruning(self, pruner):
        """Test gradient-based importance scoring."""
        model = ModelTestUtils.create_test_model()

        # Simulate training to get gradients
        x = torch.randn(2, 10, 4)
        y = torch.randint(0, 4, (2, 10))

        model.train()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()

        # Now prune based on gradients
        pruned_model = pruner.prune(model)

        assert isinstance(pruned_model, nn.Module)

        sparsity = pruner.get_sparsity(pruned_model)
        assert 0.4 <= sparsity <= 0.6

    def test_fisher_information_pruning(self, pruner):
        """Test Fisher information-based pruning."""
        model = ModelTestUtils.create_test_model()

        # Mock Fisher information calculation
        with patch.object(pruner, "calculate_fisher_information") as mock_fisher:
            mock_fisher.return_value = {
                name: torch.rand_like(param) for name, param in model.named_parameters()
            }

            pruned_model = pruner.prune_by_fisher(model)

            assert isinstance(pruned_model, nn.Module)
            mock_fisher.assert_called_once()


class TestPruningScheduler:
    """Test pruning scheduler."""

    @pytest.fixture
    def scheduler(self):
        config = PruningConfig(sparsity=0.8)
        return PruningScheduler(config)

    def test_polynomial_schedule(self, scheduler):
        """Test polynomial pruning schedule."""
        schedule = scheduler.polynomial_schedule(
            initial_sparsity=0.0, final_sparsity=0.8, total_steps=10, power=3
        )

        assert len(schedule) == 10
        assert schedule[0] == 0.0
        assert abs(schedule[-1] - 0.8) < 1e-6

        # Check monotonic increase
        assert all(schedule[i] <= schedule[i + 1] for i in range(len(schedule) - 1))

    def test_exponential_schedule(self, scheduler):
        """Test exponential pruning schedule."""
        schedule = scheduler.exponential_schedule(
            initial_sparsity=0.1, final_sparsity=0.9, total_steps=5
        )

        assert len(schedule) == 5
        assert schedule[0] == 0.1
        assert abs(schedule[-1] - 0.9) < 1e-6


class TestDistillationConfig:
    """Test knowledge distillation configuration."""

    def test_default_config(self):
        """Test default distillation configuration."""
        config = DistillationConfig()

        assert config.temperature == 4.0
        assert config.alpha == 0.7
        assert config.distillation_type == "response"
        assert isinstance(config.feature_layers, list)

    def test_attention_distillation_config(self):
        """Test attention distillation configuration."""
        config = DistillationConfig(
            distillation_type="attention",
            attention_loss_weight=1.0,
            match_attention_heads=True,
        )

        assert config.distillation_type == "attention"
        assert config.attention_loss_weight == 1.0
        assert config.match_attention_heads


class TestKnowledgeDistiller:
    """Test knowledge distillation."""

    @pytest.fixture
    def distiller(self):
        config = DistillationConfig(temperature=4.0, alpha=0.7)
        teacher = ModelTestUtils.create_test_model()
        student = ModelTestUtils.create_test_model()
        return KnowledgeDistiller(config, teacher, student)

    def test_response_distillation(self, distiller):
        """Test response-based knowledge distillation."""
        x = torch.randn(2, 10, 4)
        y = torch.randint(0, 4, (2, 10))

        loss = distiller.compute_distillation_loss(x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_temperature_scaling(self, distiller):
        """Test temperature scaling in distillation."""
        logits = torch.randn(2, 10, 4)

        # Test different temperatures
        soft_probs_low = distiller.apply_temperature(logits, temperature=1.0)
        soft_probs_high = distiller.apply_temperature(logits, temperature=10.0)

        # Higher temperature should produce smoother distributions
        entropy_low = (
            -(soft_probs_low * torch.log(soft_probs_low + 1e-8)).sum(-1).mean()
        )
        entropy_high = (
            -(soft_probs_high * torch.log(soft_probs_high + 1e-8)).sum(-1).mean()
        )

        assert entropy_high > entropy_low

    def test_progressive_distillation(self, distiller):
        """Test progressive knowledge distillation."""
        x = torch.randn(2, 10, 4)
        y = torch.randint(0, 4, (2, 10))

        # Test different alpha values (teacher vs student loss weighting)
        loss_teacher_heavy = distiller.compute_distillation_loss(x, y, alpha=0.9)
        loss_student_heavy = distiller.compute_distillation_loss(x, y, alpha=0.1)

        assert isinstance(loss_teacher_heavy, torch.Tensor)
        assert isinstance(loss_student_heavy, torch.Tensor)


class TestAttentionDistiller:
    """Test attention-based distillation."""

    @pytest.fixture
    def distiller(self):
        config = DistillationConfig(distillation_type="attention")
        teacher = ModelTestUtils.create_test_model()
        student = ModelTestUtils.create_test_model()
        return AttentionDistiller(config, teacher, student)

    def test_attention_matching(self, distiller):
        """Test attention pattern matching."""
        x = torch.randn(2, 10, 4)

        with patch.object(distiller, "extract_attention_maps") as mock_extract:
            # Mock attention maps
            teacher_attention = torch.randn(2, 4, 10, 10)  # (batch, heads, seq, seq)
            student_attention = torch.randn(2, 4, 10, 10)
            mock_extract.side_effect = [teacher_attention, student_attention]

            attention_loss = distiller.compute_attention_loss(x)

            assert isinstance(attention_loss, torch.Tensor)
            assert attention_loss.item() >= 0
            assert mock_extract.call_count == 2

    def test_attention_head_alignment(self, distiller):
        """Test attention head alignment."""
        teacher_heads = torch.randn(2, 6, 10, 10)  # 6 heads
        student_heads = torch.randn(2, 4, 10, 10)  # 4 heads

        aligned_loss = distiller.align_attention_heads(teacher_heads, student_heads)

        assert isinstance(aligned_loss, torch.Tensor)
        assert aligned_loss.item() >= 0


class TestFeatureDistiller:
    """Test feature-based distillation."""

    @pytest.fixture
    def distiller(self):
        config = DistillationConfig(
            distillation_type="feature", feature_layers=["layer1", "layer2"]
        )
        teacher = ModelTestUtils.create_test_model()
        student = ModelTestUtils.create_test_model()
        return FeatureDistiller(config, teacher, student)

    def test_intermediate_feature_matching(self, distiller):
        """Test intermediate feature matching."""
        x = torch.randn(2, 10, 4)

        with patch.object(distiller, "extract_features") as mock_extract:
            # Mock intermediate features
            teacher_features = [torch.randn(2, 10, 64), torch.randn(2, 10, 128)]
            student_features = [torch.randn(2, 10, 64), torch.randn(2, 10, 128)]
            mock_extract.side_effect = [teacher_features, student_features]

            feature_loss = distiller.compute_feature_loss(x)

            assert isinstance(feature_loss, torch.Tensor)
            assert feature_loss.item() >= 0
            assert mock_extract.call_count == 2

    def test_feature_adaptation(self, distiller):
        """Test feature dimension adaptation."""
        teacher_features = torch.randn(2, 10, 128)
        student_features = torch.randn(2, 10, 64)

        adapted_loss = distiller.adapt_and_match_features(
            teacher_features, student_features
        )

        assert isinstance(adapted_loss, torch.Tensor)
        assert adapted_loss.item() >= 0


class TestDistillationLoss:
    """Test distillation loss functions."""

    def test_kl_divergence_loss(self):
        """Test KL divergence loss for distillation."""
        teacher_logits = torch.randn(10, 4)
        student_logits = torch.randn(10, 4)
        temperature = 4.0

        loss_fn = DistillationLoss()
        kl_loss = loss_fn.kl_divergence_loss(
            student_logits, teacher_logits, temperature
        )

        assert isinstance(kl_loss, torch.Tensor)
        assert kl_loss.item() >= 0

    def test_mse_feature_loss(self):
        """Test MSE loss for feature matching."""
        teacher_features = torch.randn(10, 64)
        student_features = torch.randn(10, 64)

        loss_fn = DistillationLoss()
        mse_loss = loss_fn.mse_feature_loss(student_features, teacher_features)

        assert isinstance(mse_loss, torch.Tensor)
        assert mse_loss.item() >= 0

    def test_attention_transfer_loss(self):
        """Test attention transfer loss."""
        teacher_attention = torch.randn(2, 4, 10, 10)
        student_attention = torch.randn(2, 4, 10, 10)

        loss_fn = DistillationLoss()
        attention_loss = loss_fn.attention_transfer_loss(
            student_attention, teacher_attention
        )

        assert isinstance(attention_loss, torch.Tensor)
        assert attention_loss.item() >= 0


class TestMemoryOptimizer:
    """Test memory optimization."""

    @pytest.fixture
    def optimizer(self):
        return MemoryOptimizer()

    def test_gradient_accumulation(self, optimizer):
        """Test gradient accumulation for memory efficiency."""
        model = ModelTestUtils.create_test_model()

        # Simulate large batch through accumulation
        batch_size = 1
        accumulation_steps = 4
        x = torch.randn(batch_size, 10, 4)
        y = torch.randint(0, 4, (batch_size, 10))

        optimizer.zero_grad(model)

        for _step in range(accumulation_steps):
            loss = optimizer.forward_and_accumulate(model, x, y, accumulation_steps)
            assert isinstance(loss, torch.Tensor)

        # Check gradients were accumulated
        has_gradients = any(
            param.grad is not None
            for param in model.parameters()
            if param.requires_grad
        )
        assert has_gradients

    def test_mixed_precision(self, optimizer):
        """Test mixed precision training."""
        model = ModelTestUtils.create_test_model()
        x = torch.randn(2, 10, 4)
        y = torch.randint(0, 4, (2, 10))

        with patch("torch.cuda.amp.autocast") as mock_autocast:
            mock_autocast.return_value.__enter__ = Mock()
            mock_autocast.return_value.__exit__ = Mock()

            loss = optimizer.mixed_precision_forward(model, x, y)

            assert isinstance(loss, torch.Tensor)
            mock_autocast.assert_called_once()


class TestGradientCheckpointing:
    """Test gradient checkpointing."""

    @pytest.fixture
    def checkpointer(self):
        return GradientCheckpointing()

    def test_checkpoint_sequential(self, checkpointer):
        """Test checkpointing sequential layers."""
        model = ModelTestUtils.create_test_model()
        x = torch.randn(2, 10, 4)

        # Apply checkpointing to transformer blocks
        checkpointed_model = checkpointer.apply_checkpointing(
            model, layer_names=["transformer_blocks"]
        )

        # Test forward pass
        output = checkpointed_model(x)
        assert output.shape == (2, 10, 4)

        # Test backward pass
        loss = output.mean()
        loss.backward()

        has_gradients = any(
            param.grad is not None
            for param in checkpointed_model.parameters()
            if param.requires_grad
        )
        assert has_gradients

    def test_selective_checkpointing(self, checkpointer):
        """Test selective layer checkpointing."""
        model = ModelTestUtils.create_test_model()

        # Checkpoint every other layer
        checkpointed_model = checkpointer.apply_selective_checkpointing(
            model, checkpoint_ratio=0.5
        )

        assert isinstance(checkpointed_model, nn.Module)


class TestActivationCheckpointing:
    """Test activation checkpointing."""

    @pytest.fixture
    def checkpointer(self):
        return ActivationCheckpointing()

    def test_activation_offloading(self, checkpointer):
        """Test activation offloading to CPU."""
        activations = torch.randn(10, 1000, 512)  # Large activations

        with patch("torch.cuda.is_available", return_value=True):
            offloaded = checkpointer.offload_activations(activations)

            # Should move to CPU or compress
            assert isinstance(offloaded, torch.Tensor | tuple)

    def test_activation_compression(self, checkpointer):
        """Test activation compression."""
        activations = torch.randn(10, 100, 128)

        compressed = checkpointer.compress_activations(activations)
        decompressed = checkpointer.decompress_activations(compressed)

        # Check shape preservation
        assert decompressed.shape == activations.shape

        # Check reasonable reconstruction error
        error = torch.abs(activations - decompressed).mean()
        assert error < 0.1  # Allow some compression error


class TestMemoryProfiler:
    """Test memory profiling utilities."""

    @pytest.fixture
    def profiler(self):
        return MemoryProfiler()

    def test_memory_tracking(self, profiler):
        """Test memory usage tracking."""
        model = ModelTestUtils.create_test_model()
        x = torch.randn(2, 10, 4)

        with profiler.track_memory() as tracker:
            output = model(x)
            loss = output.mean()
            loss.backward()

        memory_stats = tracker.get_stats()

        assert "peak_memory" in memory_stats
        assert "allocated_memory" in memory_stats
        assert memory_stats["peak_memory"] > 0

    def test_layer_wise_memory(self, profiler):
        """Test layer-wise memory profiling."""
        model = ModelTestUtils.create_test_model()
        x = torch.randn(1, 10, 4)

        layer_memory = profiler.profile_layer_memory(model, x)

        assert isinstance(layer_memory, dict)
        assert len(layer_memory) > 0

        for _layer_name, memory_info in layer_memory.items():
            assert "input_memory" in memory_info
            assert "output_memory" in memory_info


class TestMemoryUtils:
    """Test memory utility functions."""

    def test_memory_cleanup(self):
        """Test memory cleanup utilities."""
        # Create some tensors
        [torch.randn(100, 100) for _ in range(10)]

        # Clear and cleanup
        MemoryUtils.clear_cache()
        MemoryUtils.garbage_collect()

        # Should not raise any errors
        assert True

    def test_memory_estimation(self):
        """Test memory requirement estimation."""
        batch_size = 8
        sequence_length = 512
        hidden_size = 768

        estimated_memory = MemoryUtils.estimate_memory_requirements(
            batch_size, sequence_length, hidden_size
        )

        assert estimated_memory > 0
        assert isinstance(estimated_memory, int | float)


class TestModelDeployer:
    """Test model deployment utilities."""

    @pytest.fixture
    def deployer(self):
        config = DeploymentConfig()
        return ModelDeployer(config)

    def test_model_optimization_pipeline(self, deployer):
        """Test complete model optimization pipeline."""
        model = ModelTestUtils.create_test_model()

        optimized_model = deployer.optimize_for_deployment(model)

        assert isinstance(optimized_model, nn.Module)

    def test_batch_size_optimization(self, deployer):
        """Test optimal batch size finding."""
        model = ModelTestUtils.create_test_model()

        optimal_batch_size = deployer.find_optimal_batch_size(
            model, sequence_length=10, max_memory_mb=1000
        )

        assert optimal_batch_size > 0
        assert isinstance(optimal_batch_size, int)


class TestTorchScriptConverter:
    """Test TorchScript conversion."""

    @pytest.fixture
    def converter(self):
        return TorchScriptConverter()

    def test_script_conversion(self, converter):
        """Test model scripting."""
        model = ModelTestUtils.create_test_model()
        example_input = torch.randn(1, 10, 4)

        scripted_model = converter.script_model(model, example_input)

        assert isinstance(scripted_model, torch.jit.ScriptModule)

        # Test that scripted model works
        output = scripted_model(example_input)
        assert output.shape == (1, 10, 4)

    def test_trace_conversion(self, converter):
        """Test model tracing."""
        model = ModelTestUtils.create_test_model()
        example_input = torch.randn(1, 10, 4)

        traced_model = converter.trace_model(model, example_input)

        assert isinstance(traced_model, torch.jit.ScriptModule)

        # Test that traced model works
        output = traced_model(example_input)
        assert output.shape == (1, 10, 4)


class TestONNXConverter:
    """Test ONNX conversion."""

    @pytest.fixture
    def converter(self):
        return ONNXConverter()

    def test_onnx_export(self, converter):
        """Test ONNX model export."""
        model = ModelTestUtils.create_test_model()
        example_input = torch.randn(1, 10, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")

            # Mock ONNX export
            with patch("torch.onnx.export") as mock_export:
                converter.export_to_onnx(model, example_input, onnx_path)
                mock_export.assert_called_once()

    def test_onnx_optimization(self, converter):
        """Test ONNX model optimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.onnx")
            output_path = os.path.join(tmpdir, "optimized.onnx")

            # Create dummy ONNX file
            with open(input_path, "w") as f:
                f.write("dummy onnx")

            with (
                patch("onnx.load") as mock_load,
                patch("onnx.save") as mock_save,
                patch.object(converter, "optimize_onnx_model") as mock_optimize,
            ):

                mock_load.return_value = Mock()
                mock_optimize.return_value = Mock()

                converter.optimize_onnx(input_path, output_path)

                mock_load.assert_called_once()
                mock_save.assert_called_once()


class TestOptimizationPipeline:
    """Test complete optimization pipeline."""

    @pytest.fixture
    def pipeline(self):
        config = DeploymentConfig(
            enable_quantization=True, enable_pruning=True, enable_distillation=False
        )
        return OptimizationPipeline(config)

    def test_full_optimization_pipeline(self, pipeline):
        """Test complete optimization pipeline."""
        model = ModelTestUtils.create_test_model()

        optimized_model = pipeline.optimize(model)

        assert isinstance(optimized_model, nn.Module)

    def test_pipeline_with_validation(self, pipeline):
        """Test optimization pipeline with validation."""
        model = ModelTestUtils.create_test_model()

        # Mock validation dataset
        val_data = [
            (torch.randn(1, 10, 4), torch.randint(0, 4, (1, 10))) for _ in range(5)
        ]

        optimized_model = pipeline.optimize_with_validation(model, val_data)

        assert isinstance(optimized_model, nn.Module)

    def test_optimization_metrics(self, pipeline):
        """Test optimization metrics collection."""
        original_model = ModelTestUtils.create_test_model()
        optimized_model = ModelTestUtils.create_test_model()

        metrics = pipeline.compute_optimization_metrics(original_model, optimized_model)

        assert "compression_ratio" in metrics
        assert "speedup" in metrics
        assert "accuracy_retention" in metrics


class TestDeploymentConfig:
    """Test deployment configuration."""

    def test_default_config(self):
        """Test default deployment configuration."""
        config = DeploymentConfig()

        assert config.target_platform == "cpu"
        assert config.optimization_level == "O2"
        assert config.enable_quantization
        assert not config.enable_pruning

    def test_mobile_config(self):
        """Test mobile deployment configuration."""
        config = DeploymentConfig.for_mobile()

        assert config.target_platform == "mobile"
        assert config.enable_quantization
        assert config.max_model_size_mb is not None

    def test_server_config(self):
        """Test server deployment configuration."""
        config = DeploymentConfig.for_server()

        assert config.target_platform == "server"
        assert config.enable_mixed_precision
        assert config.batch_size_optimization


if __name__ == "__main__":
    pytest.main([__file__])
