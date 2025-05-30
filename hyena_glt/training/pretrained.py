"""
Pretrained model management and adaptation utilities.

This module provides utilities for managing pre-trained Hyena-GLT models,
including model downloads, caching, and adaptation for different tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import logging
import requests
from dataclasses import dataclass
import hashlib
import shutil
from urllib.parse import urlparse
import tempfile

from ..config import HyenaGLTConfig
from ..model.hyena_glt import (
    HyenaGLT,
    HyenaGLTForSequenceClassification,
    HyenaGLTForTokenClassification,
    HyenaGLTForSequenceGeneration,
    HyenaGLTForMultiTask
)

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a pre-trained model."""
    name: str
    description: str
    url: str
    config: Dict[str, Any]
    size_mb: float
    tasks: List[str]
    species: List[str]
    sequence_types: List[str]  # DNA, RNA, protein
    checksum: str
    version: str = "1.0"


class ModelRegistry:
    """Registry of available pre-trained models."""
    
    def __init__(self):
        self.models = self._initialize_models()
    
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize the model registry with available models."""
        models = {}
        
        # Base models
        models["hyena-glt-small"] = ModelInfo(
            name="hyena-glt-small",
            description="Small Hyena-GLT model for general genomic tasks",
            url="https://example.com/models/hyena-glt-small.pth",
            config={
                "hidden_size": 256,
                "num_hyena_layers": 6,
                "num_attention_heads": 8,
                "intermediate_size": 1024,
                "max_position_embeddings": 8192
            },
            size_mb=150.0,
            tasks=["classification", "generation", "token_classification"],
            species=["human", "mouse", "general"],
            sequence_types=["DNA", "RNA"],
            checksum="abc123def456",
            version="1.0"
        )
        
        models["hyena-glt-base"] = ModelInfo(
            name="hyena-glt-base",
            description="Base Hyena-GLT model for genomic sequence modeling",
            url="https://example.com/models/hyena-glt-base.pth",
            config={
                "hidden_size": 512,
                "num_hyena_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 2048,
                "max_position_embeddings": 16384
            },
            size_mb=500.0,
            tasks=["classification", "generation", "token_classification"],
            species=["human", "mouse", "general"],
            sequence_types=["DNA", "RNA"],
            checksum="def456ghi789",
            version="1.0"
        )
        
        models["hyena-glt-large"] = ModelInfo(
            name="hyena-glt-large",
            description="Large Hyena-GLT model for complex genomic tasks",
            url="https://example.com/models/hyena-glt-large.pth",
            config={
                "hidden_size": 768,
                "num_hyena_layers": 24,
                "num_attention_heads": 24,
                "intermediate_size": 3072,
                "max_position_embeddings": 32768
            },
            size_mb=1200.0,
            tasks=["classification", "generation", "token_classification"],
            species=["human", "mouse", "general"],
            sequence_types=["DNA", "RNA"],
            checksum="ghi789jkl012",
            version="1.0"
        )
        
        # Specialized models
        models["hyena-glt-protein"] = ModelInfo(
            name="hyena-glt-protein",
            description="Hyena-GLT model specialized for protein sequences",
            url="https://example.com/models/hyena-glt-protein.pth",
            config={
                "hidden_size": 512,
                "num_hyena_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 2048,
                "max_position_embeddings": 8192,
                "vocab_size": 25  # 20 amino acids + special tokens
            },
            size_mb=450.0,
            tasks=["classification", "generation", "structure_prediction"],
            species=["human", "mouse", "general"],
            sequence_types=["protein"],
            checksum="jkl012mno345",
            version="1.0"
        )
        
        models["hyena-glt-human-genome"] = ModelInfo(
            name="hyena-glt-human-genome",
            description="Hyena-GLT model specialized for human genomic sequences",
            url="https://example.com/models/hyena-glt-human-genome.pth",
            config={
                "hidden_size": 768,
                "num_hyena_layers": 18,
                "num_attention_heads": 24,
                "intermediate_size": 3072,
                "max_position_embeddings": 65536
            },
            size_mb=900.0,
            tasks=["classification", "generation", "variant_calling"],
            species=["human"],
            sequence_types=["DNA"],
            checksum="mno345pqr678",
            version="1.0"
        )
        
        return models
    
    def list_models(self, task: Optional[str] = None, species: Optional[str] = None, 
                   sequence_type: Optional[str] = None) -> List[ModelInfo]:
        """List available models with optional filtering."""
        filtered_models = []
        
        for model_info in self.models.values():
            if task and task not in model_info.tasks:
                continue
            if species and species not in model_info.species:
                continue
            if sequence_type and sequence_type not in model_info.sequence_types:
                continue
            
            filtered_models.append(model_info)
        
        return filtered_models
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.models.get(model_name)
    
    def register_model(self, model_info: ModelInfo):
        """Register a new model."""
        self.models[model_info.name] = model_info
        logger.info(f"Registered model: {model_info.name}")


class ModelDownloader:
    """Utility for downloading and caching pre-trained models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "hyena_glt"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def download_model(self, model_info: ModelInfo, force_download: bool = False) -> Path:
        """Download a model and return the local path."""
        model_cache_dir = self.cache_dir / model_info.name
        model_path = model_cache_dir / "model.pth"
        config_path = model_cache_dir / "config.json"
        
        # Check if model already exists
        if model_path.exists() and config_path.exists() and not force_download:
            if self._verify_checksum(model_path, model_info.checksum):
                logger.info(f"Model {model_info.name} already cached")
                return model_path
            else:
                logger.warning(f"Checksum mismatch for {model_info.name}, re-downloading")
        
        # Create cache directory
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model
        logger.info(f"Downloading model {model_info.name} from {model_info.url}")
        self._download_file(model_info.url, model_path)
        
        # Verify checksum
        if not self._verify_checksum(model_path, model_info.checksum):
            model_path.unlink()
            raise ValueError(f"Downloaded model checksum verification failed for {model_info.name}")
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(model_info.config, f, indent=2)
        
        logger.info(f"Model {model_info.name} downloaded successfully")
        return model_path
    
    def _download_file(self, url: str, destination: Path):
        """Download a file from URL to destination."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = downloaded_size / total_size * 100
                            print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
                
                print()  # New line after progress
                
                # Move temporary file to destination
                shutil.move(tmp_file.name, destination)
                
            except Exception as e:
                # Clean up temporary file on error
                Path(tmp_file.name).unlink(missing_ok=True)
                raise e
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_checksum = sha256_hash.hexdigest()
        return actual_checksum == expected_checksum
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache."""
        if model_name:
            model_cache_dir = self.cache_dir / model_name
            if model_cache_dir.exists():
                shutil.rmtree(model_cache_dir)
                logger.info(f"Cleared cache for model: {model_name}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all model cache")


class PretrainedModelManager:
    """Main class for managing pre-trained models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.registry = ModelRegistry()
        self.downloader = ModelDownloader(cache_dir)
    
    def load_model(
        self,
        model_name: str,
        task: Optional[str] = None,
        num_labels: Optional[int] = None,
        force_download: bool = False,
        device: Optional[str] = None
    ) -> nn.Module:
        """Load a pre-trained model."""
        
        # Get model info
        model_info = self.registry.get_model_info(model_name)
        if model_info is None:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Download model if needed
        model_path = self.downloader.download_model(model_info, force_download)
        
        # Load config
        config_path = model_path.parent / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        
        config = HyenaGLTConfig(**config_dict)
        
        # Update config for task if specified
        if num_labels is not None:
            config.num_labels = num_labels
        
        # Create and load base model
        base_model = HyenaGLT(config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            base_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            base_model.load_state_dict(checkpoint)
        
        # Adapt for specific task if specified
        if task:
            model = self._adapt_for_task(base_model, task, config)
        else:
            model = base_model
        
        # Move to device
        if device:
            model = model.to(device)
        
        logger.info(f"Loaded model {model_name} for task: {task or 'base'}")
        return model
    
    def _adapt_for_task(self, base_model: HyenaGLT, task: str, config: HyenaGLTConfig) -> nn.Module:
        """Adapt base model for specific task."""
        if task == "sequence_classification":
            model = HyenaGLTForSequenceClassification(config)
            model.hyena_glt = base_model
        elif task == "token_classification":
            model = HyenaGLTForTokenClassification(config)
            model.hyena_glt = base_model
        elif task == "sequence_generation":
            model = HyenaGLTForSequenceGeneration(config)
            model.hyena_glt = base_model
        elif task == "multitask":
            model = HyenaGLTForMultiTask(config)
            model.hyena_glt = base_model
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return model
    
    def list_available_models(self, **filters) -> List[ModelInfo]:
        """List available models with optional filtering."""
        return self.registry.list_models(**filters)
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return self.registry.get_model_info(model_name)
    
    def download_model(self, model_name: str, force_download: bool = False) -> Path:
        """Download a model without loading it."""
        model_info = self.registry.get_model_info(model_name)
        if model_info is None:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return self.downloader.download_model(model_info, force_download)
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache."""
        self.downloader.clear_cache(model_name)


# Convenience functions
def load_pretrained_model(
    model_name: str,
    task: Optional[str] = None,
    num_labels: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    device: Optional[str] = None
) -> nn.Module:
    """Load a pre-trained Hyena-GLT model."""
    manager = PretrainedModelManager(cache_dir)
    return manager.load_model(
        model_name=model_name,
        task=task,
        num_labels=num_labels,
        force_download=force_download,
        device=device
    )


def list_pretrained_models(**filters) -> List[ModelInfo]:
    """List available pre-trained models."""
    manager = PretrainedModelManager()
    return manager.list_available_models(**filters)


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get information about a pre-trained model."""
    manager = PretrainedModelManager()
    return manager.get_model_info(model_name)


class ModelConverter:
    """Utility for converting models between different formats."""
    
    @staticmethod
    def convert_to_onnx(
        model: nn.Module,
        output_path: str,
        input_shape: tuple = (1, 512),
        opset_version: int = 11
    ):
        """Convert model to ONNX format."""
        try:
            import onnx
            import torch.onnx
        except ImportError:
            raise ImportError("ONNX not installed. Install with: pip install onnx")
        
        model.eval()
        dummy_input = torch.randint(0, 1000, input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model converted to ONNX: {output_path}")
    
    @staticmethod
    def convert_to_torchscript(model: nn.Module, output_path: str, input_shape: tuple = (1, 512)):
        """Convert model to TorchScript format."""
        model.eval()
        dummy_input = torch.randint(0, 1000, input_shape)
        
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
        
        logger.info(f"Model converted to TorchScript: {output_path}")
    
    @staticmethod
    def quantize_model(model: nn.Module, quantization_type: str = "dynamic") -> nn.Module:
        """Quantize model for faster inference."""
        if quantization_type == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        logger.info(f"Model quantized with {quantization_type} quantization")
        return quantized_model
