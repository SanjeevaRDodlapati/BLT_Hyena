"""
Utilities for testing Hyena-GLT models.

Provides common fixtures, data generators, and testing helpers.
"""

import os
import tempfile
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pytest

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data.tokenizer import DNATokenizer, ProteinTokenizer
from hyena_glt.model.hyena_glt import HyenaGLT


class TestConfig:
    """Configuration for testing."""
    
    # Small model for fast testing
    SMALL_CONFIG = {
        'hidden_size': 64,
        'num_layers': 2,
        'num_heads': 2,
        'intermediate_size': 128,
        'vocab_size': 32,
        'max_position_embeddings': 256,
        'hyena_order': 2,
        'hyena_kernel_size': 7,
        'token_merge_ratio': 0.5,
        'blt_layers': [0],
        'use_position_encoding': True,
        'layer_norm_eps': 1e-5,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'initializer_range': 0.02,
        'sequence_length': 128
    }
    
    # Medium model for more thorough testing
    MEDIUM_CONFIG = {
        'hidden_size': 256,
        'num_layers': 4,
        'num_heads': 4,
        'intermediate_size': 512,
        'vocab_size': 64,
        'max_position_embeddings': 1024,
        'hyena_order': 3,
        'hyena_kernel_size': 15,
        'token_merge_ratio': 0.3,
        'blt_layers': [0, 2],
        'use_position_encoding': True,
        'layer_norm_eps': 1e-5,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'initializer_range': 0.02,
        'sequence_length': 512
    }

    # Test tolerances
    RTOL = 1e-4
    ATOL = 1e-6
    RTOL_HALF = 1e-3
    ATOL_HALF = 1e-4


class DataGenerator:
    """Generates test data for various genomic tasks."""
    
    @staticmethod
    def generate_dna_sequence(length: int, 
                            vocab_size: int = 4,
                            seed: Optional[int] = None) -> torch.Tensor:
        """Generate random DNA sequences."""
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randint(0, vocab_size, (length,))
    
    @staticmethod
    def generate_protein_sequence(length: int,
                                vocab_size: int = 20,
                                seed: Optional[int] = None) -> torch.Tensor:
        """Generate random protein sequences."""
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randint(0, vocab_size, (length,))
    
    @staticmethod
    def generate_batch(batch_size: int,
                      seq_length: int,
                      vocab_size: int,
                      seed: Optional[int] = None) -> torch.Tensor:
        """Generate batch of sequences."""
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randint(0, vocab_size, (batch_size, seq_length))
    
    @staticmethod
    def generate_classification_data(batch_size: int,
                                   seq_length: int,
                                   vocab_size: int,
                                   num_classes: int,
                                   seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data for sequence classification."""
        if seed is not None:
            torch.manual_seed(seed)
        
        sequences = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = torch.randint(0, num_classes, (batch_size,))
        return sequences, labels
    
    @staticmethod
    def generate_token_classification_data(batch_size: int,
                                         seq_length: int,
                                         vocab_size: int,
                                         num_classes: int,
                                         seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data for token classification."""
        if seed is not None:
            torch.manual_seed(seed)
        
        sequences = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = torch.randint(0, num_classes, (batch_size, seq_length))
        return sequences, labels
    
    @staticmethod
    def generate_genomic_data(task: str,
                            batch_size: int = 4,
                            seq_length: int = 128,
                            **kwargs) -> Dict[str, torch.Tensor]:
        """Generate task-specific genomic data."""
        
        if task == "dna_classification":
            sequences, labels = DataGenerator.generate_classification_data(
                batch_size, seq_length, 4, kwargs.get('num_classes', 2)
            )
            return {"input_ids": sequences, "labels": labels}
        
        elif task == "protein_function":
            sequences, labels = DataGenerator.generate_classification_data(
                batch_size, seq_length, 20, kwargs.get('num_classes', 10)
            )
            return {"input_ids": sequences, "labels": labels}
        
        elif task == "gene_annotation":
            sequences, labels = DataGenerator.generate_token_classification_data(
                batch_size, seq_length, 4, kwargs.get('num_classes', 5)
            )
            return {"input_ids": sequences, "labels": labels}
        
        elif task == "variant_effect":
            sequences, labels = DataGenerator.generate_classification_data(
                batch_size, seq_length, 4, kwargs.get('num_classes', 3)
            )
            return {"input_ids": sequences, "labels": labels}
        
        else:
            raise ValueError(f"Unknown task: {task}")


class ModelTestUtils:
    """Utilities for testing models."""
    
    @staticmethod
    def create_test_model(config_dict: Optional[Dict[str, Any]] = None) -> HyenaGLT:
        """Create a small model for testing."""
        if config_dict is None:
            config_dict = TestConfig.SMALL_CONFIG
        
        config = HyenaGLTConfig(**config_dict)
        model = HyenaGLT(config)
        model.eval()
        return model
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def check_gradient_flow(model: torch.nn.Module, 
                          input_ids: torch.Tensor,
                          labels: torch.Tensor = None) -> Dict[str, bool]:
        """Check if gradients flow through model."""
        model.train()
        
        if labels is not None:
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
        else:
            outputs = model(input_ids)
            loss = outputs.last_hidden_state.sum()
        
        loss.backward()
        
        gradient_flow = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_flow[name] = (param.grad.abs().sum() > 0).item()
            else:
                gradient_flow[name] = False
        
        return gradient_flow
    
    @staticmethod
    def check_output_shapes(model: torch.nn.Module,
                          input_ids: torch.Tensor,
                          expected_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, bool]:
        """Check model output shapes."""
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        shape_checks = {}
        for attr_name, expected_shape in expected_shapes.items():
            if hasattr(outputs, attr_name):
                actual_shape = getattr(outputs, attr_name).shape
                shape_checks[attr_name] = actual_shape == expected_shape
            else:
                shape_checks[attr_name] = False
        
        return shape_checks
    
    @staticmethod
    def check_model_device(model: torch.nn.Module, expected_device: str) -> bool:
        """Check if model is on expected device."""
        for param in model.parameters():
            if str(param.device) != expected_device:
                return False
        return True


class PerformanceTestUtils:
    """Utilities for performance testing."""
    
    @staticmethod
    def measure_forward_time(model: torch.nn.Module,
                           input_ids: torch.Tensor,
                           num_runs: int = 10,
                           warmup_runs: int = 3) -> Dict[str, float]:
        """Measure forward pass time."""
        model.eval()
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_ids)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(num_runs):
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                
                with torch.no_grad():
                    _ = model(input_ids)
                
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                import time
                start = time.time()
                
                with torch.no_grad():
                    _ = model(input_ids)
                
                end = time.time()
                times.append((end - start) * 1000)  # Convert to milliseconds
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times)
        }
    
    @staticmethod
    def measure_memory_usage(model: torch.nn.Module,
                           input_ids: torch.Tensor) -> Dict[str, float]:
        """Measure memory usage during forward pass."""
        device = next(model.parameters()).device
        
        if device.type != 'cuda':
            return {'memory_mb': 0.0}
        
        input_ids = input_ids.to(device)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        with torch.no_grad():
            _ = model(input_ids)
        
        memory_bytes = torch.cuda.max_memory_allocated()
        memory_mb = memory_bytes / (1024 * 1024)
        
        return {'memory_mb': memory_mb}


@pytest.fixture
def temp_dir():
    """Provide temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def small_config():
    """Provide small model configuration for fast testing."""
    return HyenaGLTConfig(**TestConfig.SMALL_CONFIG)


@pytest.fixture
def medium_config():
    """Provide medium model configuration for thorough testing."""
    return HyenaGLTConfig(**TestConfig.MEDIUM_CONFIG)


@pytest.fixture
def sample_dna_data():
    """Provide sample DNA data."""
    return DataGenerator.generate_genomic_data("dna_classification")


@pytest.fixture
def sample_protein_data():
    """Provide sample protein data."""
    return DataGenerator.generate_genomic_data("protein_function")


@pytest.fixture
def device():
    """Provide computing device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )


def skip_if_no_gpu_memory(min_memory_gb: float = 2.0):
    """Skip test if insufficient GPU memory."""
    if not torch.cuda.is_available():
        return pytest.mark.skipif(True, reason="CUDA not available")
    
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return pytest.mark.skipif(
        memory_gb < min_memory_gb,
        reason=f"Insufficient GPU memory: {memory_gb:.1f}GB < {min_memory_gb}GB"
    )
