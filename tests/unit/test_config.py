"""
Unit tests for Hyena-GLT configuration system.
"""

from dataclasses import fields

import pytest
import torch

from hyena_glt.config import HyenaGLTConfig
from tests.utils import TestConfig


class TestHyenaGLTConfig:
    """Test configuration system."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = HyenaGLTConfig()

        # Check essential attributes exist
        assert hasattr(config, "hidden_size")
        assert hasattr(config, "num_layers")
        assert hasattr(config, "num_heads")
        assert hasattr(config, "vocab_size")
        assert hasattr(config, "max_position_embeddings")

        # Check default values are reasonable
        assert config.hidden_size > 0
        assert config.num_layers > 0
        assert config.num_heads > 0
        assert config.vocab_size > 0
        assert config.max_position_embeddings > 0

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        assert config.hidden_size == 64
        assert config.num_layers == 2
        assert config.num_heads == 2
        assert config.vocab_size == 32
        assert config.max_position_embeddings == 256
        assert config.hyena_order == 2
        assert config.hyena_kernel_size == 7

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid hidden_size
        with pytest.raises((ValueError, AssertionError)):
            HyenaGLTConfig(hidden_size=0)

        # Test invalid num_layers
        with pytest.raises((ValueError, AssertionError)):
            HyenaGLTConfig(num_layers=0)

        # Test invalid num_heads
        with pytest.raises((ValueError, AssertionError)):
            HyenaGLTConfig(num_heads=0)

        # Test heads must divide hidden_size
        with pytest.raises((ValueError, AssertionError)):
            HyenaGLTConfig(hidden_size=64, num_heads=5)

    def test_hyena_specific_config(self):
        """Test Hyena-specific configuration parameters."""
        config = HyenaGLTConfig(
            hyena_order=3, hyena_kernel_size=15, hyena_use_bias=True, hyena_dropout=0.1
        )

        assert config.hyena_order == 3
        assert config.hyena_kernel_size == 15
        assert config.hyena_use_bias is True
        assert config.hyena_dropout == 0.1

    def test_blt_specific_config(self):
        """Test BLT-specific configuration parameters."""
        config = HyenaGLTConfig(
            token_merge_ratio=0.5,
            blt_layers=[0, 2],
            merge_strategy="mean",
            unmerge_strategy="expand",
        )

        assert config.token_merge_ratio == 0.5
        assert config.blt_layers == [0, 2]
        assert config.merge_strategy == "mean"
        assert config.unmerge_strategy == "expand"

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "hidden_size" in config_dict
        assert config_dict["hidden_size"] == 64

        # Test from_dict (if available)
        if hasattr(HyenaGLTConfig, "from_dict"):
            restored_config = HyenaGLTConfig.from_dict(config_dict)
            assert restored_config.hidden_size == config.hidden_size
            assert restored_config.num_layers == config.num_layers

    def test_config_inheritance(self):
        """Test configuration inheritance and updates."""
        base_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        # Create new config with updated values
        updated_config = HyenaGLTConfig(
            **TestConfig.SMALL_CONFIG, hidden_size=128, num_layers=4
        )

        assert updated_config.hidden_size == 128
        assert updated_config.num_layers == 4
        assert updated_config.vocab_size == base_config.vocab_size  # Unchanged

    def test_genomic_task_configs(self):
        """Test genomic task-specific configurations."""
        # DNA classification config
        dna_config = HyenaGLTConfig(
            vocab_size=4,  # A, T, C, G
            num_labels=2,
            task_type="sequence_classification",
        )
        assert dna_config.vocab_size == 4

        # Protein function config
        protein_config = HyenaGLTConfig(
            vocab_size=20,  # 20 amino acids
            num_labels=10,
            task_type="sequence_classification",
        )
        assert protein_config.vocab_size == 20

        # Gene annotation config
        annotation_config = HyenaGLTConfig(
            vocab_size=4,
            num_labels=5,  # Different annotation types
            task_type="token_classification",
        )
        assert annotation_config.num_labels == 5

    def test_config_device_compatibility(self):
        """Test configuration works with different devices."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        # Should work regardless of device availability
        if torch.cuda.is_available():
            # Test with CUDA device
            assert config.hidden_size == 64  # Basic functionality

        # Test with CPU
        assert config.num_layers == 2

    def test_config_repr(self):
        """Test configuration string representation."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        repr_str = repr(config)
        assert isinstance(repr_str, str)
        assert "HyenaGLTConfig" in repr_str

        str_repr = str(config)
        assert isinstance(str_repr, str)

    def test_config_equality(self):
        """Test configuration equality comparison."""
        config1 = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        config2 = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        config3 = HyenaGLTConfig(**TestConfig.MEDIUM_CONFIG)

        assert config1 == config2  # Same parameters
        assert config1 != config3  # Different parameters

    def test_config_field_types(self):
        """Test configuration field types."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)

        # Check integer fields
        assert isinstance(config.hidden_size, int)
        assert isinstance(config.num_layers, int)
        assert isinstance(config.num_heads, int)
        assert isinstance(config.vocab_size, int)

        # Check float fields
        assert isinstance(config.token_merge_ratio, float)
        assert isinstance(config.hidden_dropout_prob, float)
        assert isinstance(config.attention_probs_dropout_prob, float)

        # Check boolean fields
        assert isinstance(config.use_position_encoding, bool)

        # Check list fields
        assert isinstance(config.blt_layers, list)

    @pytest.mark.parametrize(
        "hidden_size,num_heads", [(64, 2), (128, 4), (256, 8), (512, 16)]
    )
    def test_head_size_calculation(self, hidden_size, num_heads):
        """Test head size calculation for different configurations."""
        config = HyenaGLTConfig(hidden_size=hidden_size, num_heads=num_heads)

        expected_head_size = hidden_size // num_heads
        # If config has head_size attribute, check it
        if hasattr(config, "head_size"):
            assert config.head_size == expected_head_size

        # Check divisibility
        assert hidden_size % num_heads == 0

    @pytest.mark.parametrize("sequence_length", [128, 256, 512, 1024])
    def test_position_embedding_limits(self, sequence_length):
        """Test position embedding configuration."""
        config = HyenaGLTConfig(
            max_position_embeddings=sequence_length, sequence_length=sequence_length
        )

        assert config.max_position_embeddings >= sequence_length

    def test_task_type_validation(self):
        """Test task type validation."""
        valid_task_types = [
            "sequence_classification",
            "token_classification",
            "sequence_generation",
            "multi_task",
        ]

        for task_type in valid_task_types:
            config = HyenaGLTConfig(task_type=task_type)
            assert config.task_type == task_type

    def test_config_json_serialization(self, temp_dir):
        """Test JSON serialization of configuration."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        config_path = temp_dir / "config.json"

        # Save config
        config.save_pretrained(str(config_path.parent))

        # Load config
        loaded_config = HyenaGLTConfig.from_pretrained(str(config_path.parent))

        # Check key attributes match
        assert loaded_config.hidden_size == config.hidden_size
        assert loaded_config.num_layers == config.num_layers
        assert loaded_config.vocab_size == config.vocab_size
