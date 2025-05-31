"""
Unit tests for Hyena-GLT model architecture components.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model.heads import (
    MultiTaskHead,
    SequenceClassificationHead,
    SequenceGenerationHead,
    TokenClassificationHead,
)
from hyena_glt.model.hyena_glt import (
    HyenaGLT,
    HyenaGLTForSequenceClassification,
    HyenaGLTForSequenceGeneration,
    HyenaGLTForTokenClassification,
)
from hyena_glt.model.layers import AdaptiveTokenMerger, HybridLayer, HyenaGLTBlock
from hyena_glt.model.operators import (
    AdaptiveKernelConvolution,
    GenomicPositionalEncoding,
    GeometricAttention,
    HyenaOperator,
)
from tests.utils import DataGenerator, ModelTestUtils, TestConfig, skip_if_no_cuda


class TestHyenaOperator:
    """Test core Hyena operator."""

    def test_hyena_operator_creation(self):
        """Test creating Hyena operator."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        operator = HyenaOperator(
            hidden_size=config.hidden_size,
            kernel_size=config.hyena_kernel_size,
            order=config.hyena_order,
        )

        assert isinstance(operator, nn.Module)
        assert operator.hidden_size == config.hidden_size
        assert operator.kernel_size == config.hyena_kernel_size
        assert operator.order == config.hyena_order

    def test_hyena_operator_forward(self):
        """Test Hyena operator forward pass."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        batch_size, seq_len = 2, 32

        operator = HyenaOperator(
            hidden_size=config.hidden_size,
            kernel_size=config.hyena_kernel_size,
            order=config.hyena_order,
        )

        # Create input tensor
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # Forward pass
        output = operator(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_hyena_operator_gradient_flow(self):
        """Test gradient flow through Hyena operator."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        batch_size, seq_len = 2, 16

        operator = HyenaOperator(
            hidden_size=config.hidden_size,
            kernel_size=config.hyena_kernel_size,
            order=config.hyena_order,
        )

        x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)

        output = operator(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check parameter gradients
        for param in operator.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @pytest.mark.parametrize("kernel_size", [3, 7, 15, 31])
    def test_hyena_different_kernel_sizes(self, kernel_size):
        """Test Hyena operator with different kernel sizes."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        operator = HyenaOperator(
            hidden_size=config.hidden_size,
            kernel_size=kernel_size,
            order=config.hyena_order,
        )

        x = torch.randn(2, 32, config.hidden_size)
        output = operator(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_hyena_different_orders(self, order):
        """Test Hyena operator with different orders."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        operator = HyenaOperator(
            hidden_size=config.hidden_size,
            kernel_size=config.hyena_kernel_size,
            order=order,
        )

        x = torch.randn(2, 32, config.hidden_size)
        output = operator(x)

        assert output.shape == x.shape


class TestGenomicPositionalEncoding:
    """Test genomic positional encoding."""

    def test_positional_encoding_creation(self):
        """Test creating positional encoding."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        pos_enc = GenomicPositionalEncoding(
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
        )

        assert isinstance(pos_enc, nn.Module)

    def test_positional_encoding_forward(self):
        """Test positional encoding forward pass."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        batch_size, seq_len = 2, 32

        pos_enc = GenomicPositionalEncoding(
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
        )

        x = torch.randn(batch_size, seq_len, config.hidden_size)
        encoded = pos_enc(x)

        assert encoded.shape == x.shape
        assert not torch.equal(encoded, x)  # Should modify input

    def test_positional_encoding_consistency(self):
        """Test positional encoding consistency."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        pos_enc = GenomicPositionalEncoding(
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
        )

        x = torch.randn(2, 32, config.hidden_size)

        # Multiple forward passes should be identical
        encoded1 = pos_enc(x)
        encoded2 = pos_enc(x)

        assert torch.allclose(encoded1, encoded2)


class TestAdaptiveTokenMerger:
    """Test adaptive token merging component."""

    def test_token_merger_creation(self):
        """Test creating token merger."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        merger = AdaptiveTokenMerger(
            hidden_size=config.hidden_size,
            merge_ratio=config.token_merge_ratio,
            strategy=config.merge_strategy,
        )

        assert isinstance(merger, nn.Module)

    def test_token_merger_forward(self):
        """Test token merger forward pass."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        batch_size, seq_len = 2, 32

        merger = AdaptiveTokenMerger(
            hidden_size=config.hidden_size,
            merge_ratio=config.token_merge_ratio,
            strategy="mean",
        )

        x = torch.randn(batch_size, seq_len, config.hidden_size)
        merged, merge_info = merger(x)

        expected_len = int(seq_len * (1 - config.token_merge_ratio))
        assert merged.shape[1] <= expected_len  # Should reduce sequence length
        assert merged.shape[0] == batch_size
        assert merged.shape[2] == config.hidden_size

        assert merge_info is not None

    def test_token_merger_unmerge(self):
        """Test token unmerging functionality."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        batch_size, seq_len = 2, 32

        merger = AdaptiveTokenMerger(
            hidden_size=config.hidden_size,
            merge_ratio=config.token_merge_ratio,
            strategy="mean",
        )

        x = torch.randn(batch_size, seq_len, config.hidden_size)
        merged, merge_info = merger(x)
        unmerged = merger.unmerge(merged, merge_info, original_length=seq_len)

        assert unmerged.shape == x.shape

    @pytest.mark.parametrize("merge_ratio", [0.1, 0.3, 0.5, 0.7])
    def test_different_merge_ratios(self, merge_ratio):
        """Test different merge ratios."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        batch_size, seq_len = 2, 32

        merger = AdaptiveTokenMerger(
            hidden_size=config.hidden_size, merge_ratio=merge_ratio, strategy="mean"
        )

        x = torch.randn(batch_size, seq_len, config.hidden_size)
        merged, _ = merger(x)

        expected_len = int(seq_len * (1 - merge_ratio))
        assert abs(merged.shape[1] - expected_len) <= 1  # Allow for rounding


class TestHyenaGLTBlock:
    """Test Hyena-GLT transformer block."""

    def test_block_creation(self):
        """Test creating Hyena-GLT block."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        block = HyenaGLTBlock(config, layer_idx=0)

        assert isinstance(block, nn.Module)
        assert hasattr(block, "hyena_operator")
        assert hasattr(block, "feed_forward")

    def test_block_forward(self):
        """Test block forward pass."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        batch_size, seq_len = 2, 32

        block = HyenaGLTBlock(config, layer_idx=0)

        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = block(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_block_with_token_merging(self):
        """Test block with token merging enabled."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        config.blt_layers = [0]  # Enable merging for layer 0
        batch_size, seq_len = 2, 32

        block = HyenaGLTBlock(config, layer_idx=0)

        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = block(x)

        # Output might have different sequence length due to merging
        assert output.shape[0] == batch_size
        assert output.shape[2] == config.hidden_size

    def test_block_gradient_flow(self):
        """Test gradient flow through block."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        block = HyenaGLTBlock(config, layer_idx=0)

        x = torch.randn(2, 16, config.hidden_size, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

        # Check parameter gradients
        for param in block.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestTaskHeads:
    """Test task-specific heads."""

    def test_sequence_classification_head(self):
        """Test sequence classification head."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        num_classes = 5

        head = SequenceClassificationHead(
            config=config, num_classes=num_classes, pooling_strategy="cls"
        )

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        logits = head(x)

        assert logits.shape == (batch_size, num_classes)
        assert not torch.isnan(logits).any()

    def test_token_classification_head(self):
        """Test token classification head."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        num_classes = 5

        head = TokenClassificationHead(config=config, num_classes=num_classes)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        logits = head(x)

        assert logits.shape == (batch_size, seq_len, num_classes)
        assert not torch.isnan(logits).any()

    def test_sequence_generation_head(self):
        """Test sequence generation head."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        head = SequenceGenerationHead(config=config)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        logits = head(x)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_multi_task_head(self):
        """Test multi-task head."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        task_configs = {
            "classification": {"type": "sequence_classification", "num_classes": 3},
            "annotation": {"type": "token_classification", "num_classes": 5},
        }

        head = MultiTaskHead(config=config, task_configs=task_configs)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        outputs = head(x)

        assert "classification" in outputs
        assert "annotation" in outputs
        assert outputs["classification"].shape == (batch_size, 3)
        assert outputs["annotation"].shape == (batch_size, seq_len, 5)


class TestHyenaGLTModel:
    """Test main Hyena-GLT model."""

    def test_model_creation(self):
        """Test creating Hyena-GLT model."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)

        assert isinstance(model, nn.Module)
        assert model.config == config

    def test_model_forward(self):
        """Test model forward pass."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids)

        assert hasattr(outputs, "last_hidden_state")
        assert outputs.last_hidden_state.shape == (
            batch_size,
            seq_len,
            config.hidden_size,
        )

    def test_model_with_attention_mask(self):
        """Test model with attention mask."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 20:] = 0  # Mask part of first sequence

        outputs = model(input_ids, attention_mask=attention_mask)

        assert outputs.last_hidden_state.shape == (
            batch_size,
            seq_len,
            config.hidden_size,
        )

    def test_model_parameter_count(self):
        """Test model parameter count is reasonable."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)

        param_count = ModelTestUtils.count_parameters(model)

        # Should have reasonable number of parameters
        assert param_count > 1000
        assert param_count < 1_000_000  # Less than 1M for small config

    def test_model_gradient_flow(self):
        """Test gradient flow through model."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        gradient_flow = ModelTestUtils.check_gradient_flow(model, input_ids)

        # Most parameters should have gradients
        params_with_grads = sum(gradient_flow.values())
        total_params = len(gradient_flow)

        assert params_with_grads > total_params * 0.5  # At least 50% have gradients

    @skip_if_no_cuda()
    def test_model_device_movement(self):
        """Test moving model between devices."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)

        # Test moving to CUDA
        model = model.cuda()
        assert ModelTestUtils.check_model_device(model, "cuda:0")

        # Test moving back to CPU
        model = model.cpu()
        assert ModelTestUtils.check_model_device(model, "cpu")


class TestSpecializedModels:
    """Test specialized model variants."""

    def test_sequence_classification_model(self):
        """Test sequence classification model."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        num_classes = 5

        model = HyenaGLTForSequenceClassification(config, num_classes=num_classes)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, num_classes, (batch_size,))

        outputs = model(input_ids, labels=labels)

        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, num_classes)
        assert outputs.loss.item() > 0

    def test_token_classification_model(self):
        """Test token classification model."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        num_classes = 5

        model = HyenaGLTForTokenClassification(config, num_classes=num_classes)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, num_classes, (batch_size, seq_len))

        outputs = model(input_ids, labels=labels)

        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, seq_len, num_classes)
        assert outputs.loss.item() > 0

    def test_sequence_generation_model(self):
        """Test sequence generation model."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        model = HyenaGLTForSequenceGeneration(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids, labels=labels)

        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
        assert outputs.loss.item() > 0


class TestModelSerialization:
    """Test model saving and loading."""

    def test_model_state_dict(self):
        """Test model state dict operations."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)

        # Get state dict
        state_dict = model.state_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Create new model and load state dict
        new_model = HyenaGLT(config)
        new_model.load_state_dict(state_dict)

        # Check parameters are identical
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters(), strict=False
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_model_save_load(self, temp_dir):
        """Test model saving and loading to disk."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)

        # Save model
        save_path = temp_dir / "model"
        model.save_pretrained(str(save_path))

        # Check files exist
        assert (save_path / "config.json").exists()
        assert (save_path / "pytorch_model.bin").exists()

        # Load model
        loaded_model = HyenaGLT.from_pretrained(str(save_path))

        assert loaded_model.config.hidden_size == config.hidden_size
        assert loaded_model.config.num_layers == config.num_layers
