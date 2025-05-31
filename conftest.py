"""
Shared test fixtures and configuration for Hyena-GLT tests.

This module provides common fixtures, test utilities, and configuration
that can be used across all test modules.
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
import torch

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data.dataset import GenomicDataset
from hyena_glt.data.tokenizer import DNATokenizer, ProteinTokenizer, RNATokenizer
from hyena_glt.model.hyena_glt import (
    HyenaGLT,
    HyenaGLTForSequenceClassification,
    HyenaGLTForSequenceGeneration,
    HyenaGLTForTokenClassification,
)
from hyena_glt.training.config import TrainingConfig
from tests.utils import DataGenerator, TestConfig


# Configure test environment
def pytest_configure(config):
    """Configure pytest environment."""
    # Set torch to deterministic mode for reproducible tests
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for testing
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to tests with 'slow' in name or taking long time
        if 'slow' in item.name.lower() or 'benchmark' in item.name.lower():
            item.add_marker(pytest.mark.slow)

        # Add gpu marker to tests with 'gpu' or 'cuda' in name
        if 'gpu' in item.name.lower() or 'cuda' in item.name.lower():
            item.add_marker(pytest.mark.gpu)

        # Add memory intensive marker
        if 'memory' in item.name.lower() or 'large' in item.name.lower():
            item.add_marker(pytest.mark.memory_intensive)


# Fixtures for test configuration
@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TestConfig()


@pytest.fixture(scope="session")
def small_model_config():
    """Provide small model configuration for fast testing."""
    return HyenaGLTConfig(**TestConfig.SMALL_CONFIG)


@pytest.fixture(scope="session")
def medium_model_config():
    """Provide medium model configuration for thorough testing."""
    return HyenaGLTConfig(**TestConfig.MEDIUM_CONFIG)


# Device fixtures
@pytest.fixture(scope="session")
def device():
    """Provide device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Force CPU device for CPU-specific tests."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """Provide CUDA device if available, skip test otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# Model fixtures
@pytest.fixture
def small_hyena_model(small_model_config, device):
    """Provide small Hyena-GLT model for testing."""
    model = HyenaGLT(small_model_config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def small_classification_model(small_model_config, device):
    """Provide small classification model for testing."""
    config = small_model_config
    config.num_labels = 3
    model = HyenaGLTForSequenceClassification(config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def small_token_classification_model(small_model_config, device):
    """Provide small token classification model for testing."""
    config = small_model_config
    config.num_labels = 5
    model = HyenaGLTForTokenClassification(config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def small_generation_model(small_model_config, device):
    """Provide small generation model for testing."""
    model = HyenaGLTForSequenceGeneration(small_model_config)
    model.to(device)
    model.eval()
    return model


# Tokenizer fixtures
@pytest.fixture
def dna_tokenizer():
    """Provide DNA tokenizer for testing."""
    return DNATokenizer()


@pytest.fixture
def protein_tokenizer():
    """Provide protein tokenizer for testing."""
    return ProteinTokenizer()


@pytest.fixture
def rna_tokenizer():
    """Provide RNA tokenizer for testing."""
    return RNATokenizer()


# Data fixtures
@pytest.fixture
def sample_dna_sequences():
    """Provide sample DNA sequences for testing."""
    sequences = []
    for _i in range(10):
        length = torch.randint(50, 200, (1,)).item()
        seq = DataGenerator.generate_dna_sequence(length)
        seq_str = ''.join(['ATCG'[x] for x in seq])
        sequences.append(seq_str)
    return sequences


@pytest.fixture
def sample_protein_sequences():
    """Provide sample protein sequences for testing."""
    sequences = []
    aa_chars = "ACDEFGHIKLMNPQRSTVWY"
    for _i in range(10):
        length = torch.randint(30, 100, (1,)).item()
        seq = DataGenerator.generate_protein_sequence(length, vocab_size=20)
        seq_str = ''.join([aa_chars[x] for x in seq])
        sequences.append(seq_str)
    return sequences


@pytest.fixture
def sample_labels():
    """Provide sample labels for classification tasks."""
    return [i % 3 for i in range(10)]  # 3 classes


@pytest.fixture
def sample_token_labels():
    """Provide sample token-level labels."""
    labels = []
    for _i in range(10):
        length = torch.randint(50, 200, (1,)).item()
        token_labels = torch.randint(0, 5, (length,)).tolist()
        labels.append(token_labels)
    return labels


# Dataset fixtures
@pytest.fixture
def dna_classification_dataset(sample_dna_sequences, sample_labels, dna_tokenizer):
    """Provide DNA classification dataset for testing."""
    return GenomicDataset(
        sequences=sample_dna_sequences,
        labels=sample_labels,
        tokenizer=dna_tokenizer,
        max_length=256,
        task_type='classification'
    )


@pytest.fixture
def protein_generation_dataset(sample_protein_sequences, protein_tokenizer):
    """Provide protein generation dataset for testing."""
    return GenomicDataset(
        sequences=sample_protein_sequences,
        tokenizer=protein_tokenizer,
        max_length=256,
        task_type='generation'
    )


@pytest.fixture
def dna_token_classification_dataset(sample_dna_sequences, sample_token_labels, dna_tokenizer):
    """Provide DNA token classification dataset for testing."""
    return GenomicDataset(
        sequences=sample_dna_sequences,
        token_labels=sample_token_labels,
        tokenizer=dna_tokenizer,
        max_length=256,
        task_type='token_classification'
    )


# Training configuration fixtures
@pytest.fixture
def training_config():
    """Provide training configuration for testing."""
    return TrainingConfig(
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=1,
        eval_steps=5,
        save_steps=10,
        logging_steps=2,
        warmup_steps=2,
        max_grad_norm=1.0
    )


# Temporary directory fixtures
@pytest.fixture
def temp_dir():
    """Provide temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def temp_model_dir():
    """Provide temporary directory for model saving/loading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = os.path.join(tmp_dir, "model")
        os.makedirs(model_dir)
        yield model_dir


# Mock fixtures
@pytest.fixture
def mock_wandb():
    """Mock wandb for testing without actual logging."""
    import sys

    # Create mock wandb module
    mock_wandb_module = MagicMock()
    sys.modules['wandb'] = mock_wandb_module

    # Mock key functions
    mock_wandb_module.init.return_value = MagicMock()
    mock_wandb_module.log = MagicMock()
    mock_wandb_module.finish = MagicMock()

    yield mock_wandb_module

    # Clean up
    if 'wandb' in sys.modules:
        del sys.modules['wandb']


@pytest.fixture
def mock_tensorboard():
    """Mock tensorboard for testing without actual logging."""
    import sys

    # Create mock tensorboard module
    mock_tb = MagicMock()
    sys.modules['torch.utils.tensorboard'] = mock_tb

    yield mock_tb

    # Clean up
    if 'torch.utils.tensorboard' in sys.modules:
        del sys.modules['torch.utils.tensorboard']


# Performance testing fixtures
@pytest.fixture
def benchmark_runner():
    """Provide benchmark runner for performance tests."""
    class BenchmarkRunner:
        def __init__(self):
            self.results = {}

        def run_benchmark(self, name: str, func, *args, **kwargs):
            """Run a benchmark and record results."""
            import time

            # Warmup
            for _ in range(3):
                func(*args, **kwargs)

            # Actual benchmark
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            self.results[name] = {
                'time': end_time - start_time,
                'result': result
            }

            return result

        def get_results(self):
            return self.results

    return BenchmarkRunner()


# Error handling fixtures
@pytest.fixture
def capture_warnings():
    """Capture warnings during tests."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Automatically cleanup CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test for reproducibility."""
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    import random
    random.seed(42)
    yield


# Skip conditions
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )


def skip_if_no_internet():
    """Skip test if internet is not available."""
    import urllib.request
    try:
        urllib.request.urlopen('http://google.com', timeout=1)
        return False
    except Exception:
        return pytest.mark.skip(reason="Internet not available")


def skip_if_insufficient_memory(min_memory_gb=8):
    """Skip test if insufficient memory available."""
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    return pytest.mark.skipif(
        available_memory_gb < min_memory_gb,
        reason=f"Insufficient memory: {available_memory_gb:.1f}GB < {min_memory_gb}GB"
    )


# Parameterized test helpers
@pytest.fixture(params=['cpu', 'cuda'])
def all_devices(request):
    """Parameterize tests across CPU and CUDA devices."""
    device_name = request.param
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(device_name)


@pytest.fixture(params=[1, 2, 4])
def batch_sizes(request):
    """Parameterize tests across different batch sizes."""
    return request.param


@pytest.fixture(params=[64, 128, 256])
def sequence_lengths(request):
    """Parameterize tests across different sequence lengths."""
    return request.param


# Test data generators
@pytest.fixture
def data_generator():
    """Provide data generator for creating test data."""
    return DataGenerator()


# Custom assertions
def assert_model_output_shape(output, expected_batch_size, expected_seq_len=None, expected_features=None):
    """Assert model output has expected shape."""
    if hasattr(output, 'logits'):
        logits = output.logits
    else:
        logits = output

    assert logits.shape[0] == expected_batch_size
    if expected_seq_len is not None:
        assert logits.shape[1] == expected_seq_len
    if expected_features is not None:
        assert logits.shape[-1] == expected_features


def assert_tensor_close(tensor1, tensor2, rtol=1e-5, atol=1e-6):
    """Assert two tensors are close within tolerance."""
    torch.testing.assert_close(tensor1, tensor2, rtol=rtol, atol=atol)


def assert_gradients_exist(model):
    """Assert model has gradients (useful for training tests)."""
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    assert has_gradients, "Model should have gradients after backward pass"


def assert_no_gradients(model):
    """Assert model has no gradients (useful for inference tests)."""
    for param in model.parameters():
        assert param.grad is None, "Model should not have gradients in eval mode"
