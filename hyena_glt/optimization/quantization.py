"""
Model Quantization for Hyena-GLT

This module provides comprehensive quantization support for the Hyena-GLT model,
including post-training quantization, quantization-aware training, and dynamic quantization.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import (
    get_default_qconfig,
    get_default_qat_qconfig,
    prepare,
    prepare_qat,
    convert,
    quantize_dynamic
)
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable, Union
import logging
import numpy as np
from pathlib import Path
import json

from ..config import HyenaGLTConfig
from ..model import HyenaGLT

logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    # Quantization method
    method: str = "dynamic"  # "dynamic", "static", "qat"
    
    # Backend configuration
    backend: str = "fbgemm"  # "fbgemm", "qnnpack"
    
    # Quantization parameters
    bits: int = 8
    observe_method: str = "histogram"  # "histogram", "minmax"
    
    # QAT specific parameters
    qat_epochs: int = 10
    qat_lr: float = 1e-5
    qat_warmup_steps: int = 100
    
    # Calibration parameters
    calibration_samples: int = 1000
    calibration_batch_size: int = 32
    
    # Layer-specific settings
    quantize_embedding: bool = False
    quantize_output: bool = True
    skip_layers: List[str] = None
    
    # Advanced options
    per_channel: bool = True
    symmetric: bool = False
    reduce_range: bool = False
    
    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = []

class ModelQuantizer:
    """Main quantization interface for Hyena-GLT models."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.quantized_model = None
        self.calibrated = False
        
    def quantize_model(
        self,
        model: HyenaGLT,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        save_path: Optional[str] = None
    ) -> nn.Module:
        """
        Quantize a Hyena-GLT model.
        
        Args:
            model: The model to quantize
            calibration_loader: Data loader for calibration (required for static quantization)
            save_path: Path to save the quantized model
            
        Returns:
            Quantized model
        """
        if self.config.method == "dynamic":
            quantized_model = self._dynamic_quantize(model)
        elif self.config.method == "static":
            if calibration_loader is None:
                raise ValueError("Calibration loader required for static quantization")
            quantized_model = self._static_quantize(model, calibration_loader)
        elif self.config.method == "qat":
            if calibration_loader is None:
                raise ValueError("Training loader required for QAT")
            quantized_model = self._qat_quantize(model, calibration_loader)
        else:
            raise ValueError(f"Unknown quantization method: {self.config.method}")
        
        self.quantized_model = quantized_model
        
        if save_path:
            self.save_quantized_model(save_path)
            
        return quantized_model
    
    def _dynamic_quantize(self, model: HyenaGLT) -> nn.Module:
        """Apply dynamic quantization."""
        logger.info("Applying dynamic quantization...")
        
        # Set quantization configuration
        model.eval()
        
        # Define modules to quantize
        modules_to_quantize = {nn.Linear}
        if self.config.quantize_embedding:
            modules_to_quantize.add(nn.Embedding)
        
        # Apply dynamic quantization
        quantized_model = quantize_dynamic(
            model,
            modules_to_quantize,
            dtype=torch.qint8 if self.config.bits == 8 else torch.quint8,
            inplace=False
        )
        
        logger.info("Dynamic quantization completed")
        return quantized_model
    
    def _static_quantize(
        self,
        model: HyenaGLT,
        calibration_loader: torch.utils.data.DataLoader
    ) -> nn.Module:
        """Apply static quantization with calibration."""
        logger.info("Applying static quantization...")
        
        # Set backend
        torch.backends.quantized.engine = self.config.backend
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = get_default_qconfig(self.config.backend)
        
        # Skip specific layers if requested
        for layer_name in self.config.skip_layers:
            if hasattr(model, layer_name):
                setattr(getattr(model, layer_name), 'qconfig', None)
        
        # Prepare model
        prepared_model = prepare(model, inplace=False)
        
        # Calibrate
        calibrator = QuantizationCalibrator(self.config)
        calibrator.calibrate(prepared_model, calibration_loader)
        
        # Convert to quantized model
        quantized_model = convert(prepared_model, inplace=False)
        
        logger.info("Static quantization completed")
        return quantized_model
    
    def _qat_quantize(
        self,
        model: HyenaGLT,
        train_loader: torch.utils.data.DataLoader
    ) -> nn.Module:
        """Apply quantization-aware training."""
        logger.info("Starting quantization-aware training...")
        
        # Set backend
        torch.backends.quantized.engine = self.config.backend
        
        # Prepare model for QAT
        model.train()
        model.qconfig = get_default_qat_qconfig(self.config.backend)
        
        # Skip specific layers if requested
        for layer_name in self.config.skip_layers:
            if hasattr(model, layer_name):
                setattr(getattr(model, layer_name), 'qconfig', None)
        
        # Prepare for QAT
        prepared_model = prepare_qat(model, inplace=False)
        
        # Train with QAT
        qat_trainer = QATTrainer(self.config)
        trained_model = qat_trainer.train(prepared_model, train_loader)
        
        # Convert to quantized model
        trained_model.eval()
        quantized_model = convert(trained_model, inplace=False)
        
        logger.info("QAT completed")
        return quantized_model
    
    def save_quantized_model(self, save_path: str):
        """Save the quantized model."""
        if self.quantized_model is None:
            raise ValueError("No quantized model to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_path / "quantized_model.pt"
        torch.save(self.quantized_model.state_dict(), model_path)
        
        # Save config
        config_path = save_path / "quantization_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Quantized model saved to {save_path}")
    
    def load_quantized_model(
        self,
        load_path: str,
        model_class: type = HyenaGLT
    ) -> nn.Module:
        """Load a quantized model."""
        load_path = Path(load_path)
        
        # Load config
        config_path = load_path / "quantization_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create model (this would need to be adapted based on actual architecture)
        # For now, returning a placeholder
        logger.info(f"Quantized model loaded from {load_path}")
        return self.quantized_model

class DynamicQuantizer:
    """Specialized dynamic quantization for inference optimization."""
    
    def __init__(self, dtype: torch.dtype = torch.qint8):
        self.dtype = dtype
        
    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization optimized for inference."""
        model.eval()
        
        # Dynamic quantization for linear layers
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=self.dtype,
            inplace=False
        )
        
        return quantized_model

class QATTrainer:
    """Quantization-Aware Training trainer."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
    def train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None
    ) -> nn.Module:
        """Train model with quantization awareness."""
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.qat_lr)
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        for epoch in range(self.config.qat_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Warmup quantization
                if batch_idx < self.config.qat_warmup_steps:
                    # Gradually enable quantization
                    for module in model.modules():
                        if hasattr(module, 'activation_post_process'):
                            if hasattr(module.activation_post_process, 'enable'):
                                module.activation_post_process.enable()
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"QAT Epoch {epoch+1}/{self.config.qat_epochs}, Loss: {avg_loss:.4f}")
        
        return model

class QuantizationCalibrator:
    """Calibration utility for static quantization."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
    def calibrate(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader
    ):
        """Run calibration on the prepared model."""
        model.eval()
        
        samples_processed = 0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                if samples_processed >= self.config.calibration_samples:
                    break
                
                # Forward pass for calibration
                _ = model(data)
                
                samples_processed += data.size(0)
                
                if batch_idx % 100 == 0:
                    logger.info(f"Calibration progress: {samples_processed}/{self.config.calibration_samples}")
        
        logger.info(f"Calibration completed with {samples_processed} samples")

class QuantizationBenchmark:
    """Benchmark quantized models against original models."""
    
    def __init__(self):
        self.results = {}
        
    def benchmark(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Compare original and quantized models."""
        if metrics is None:
            metrics = ['accuracy', 'latency', 'memory']
        
        results = {
            'original': {},
            'quantized': {},
            'compression_ratio': None,
            'speedup': None
        }
        
        # Benchmark original model
        if 'accuracy' in metrics:
            results['original']['accuracy'] = self._measure_accuracy(original_model, test_loader)
        if 'latency' in metrics:
            results['original']['latency'] = self._measure_latency(original_model, test_loader)
        if 'memory' in metrics:
            results['original']['memory'] = self._measure_memory(original_model)
        
        # Benchmark quantized model
        if 'accuracy' in metrics:
            results['quantized']['accuracy'] = self._measure_accuracy(quantized_model, test_loader)
        if 'latency' in metrics:
            results['quantized']['latency'] = self._measure_latency(quantized_model, test_loader)
        if 'memory' in metrics:
            results['quantized']['memory'] = self._measure_memory(quantized_model)
        
        # Calculate improvements
        if 'memory' in metrics:
            results['compression_ratio'] = (
                results['original']['memory'] / results['quantized']['memory']
            )
        
        if 'latency' in metrics:
            results['speedup'] = (
                results['original']['latency'] / results['quantized']['latency']
            )
        
        self.results = results
        return results
    
    def _measure_accuracy(self, model: nn.Module, test_loader: torch.utils.data.DataLoader) -> float:
        """Measure model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def _measure_latency(self, model: nn.Module, test_loader: torch.utils.data.DataLoader) -> float:
        """Measure model inference latency."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 10:
                    break
                _ = model(data)
        
        # Measure
        times = []
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 100:
                    break
                    
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = model(data)
                end_time.record()
                
                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time))
        
        return np.mean(times)
    
    def _measure_memory(self, model: nn.Module) -> int:
        """Measure model memory usage in bytes."""
        param_size = 0
        param_sum = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        
        buffer_size = 0
        buffer_sum = 0
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        
        return param_size + buffer_size
    
    def print_results(self):
        """Print benchmark results."""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*50)
        print("QUANTIZATION BENCHMARK RESULTS")
        print("="*50)
        
        for model_type in ['original', 'quantized']:
            print(f"\n{model_type.upper()} MODEL:")
            for metric, value in self.results[model_type].items():
                if metric == 'accuracy':
                    print(f"  Accuracy: {value:.4f}")
                elif metric == 'latency':
                    print(f"  Latency: {value:.2f} ms")
                elif metric == 'memory':
                    print(f"  Memory: {value / 1024 / 1024:.2f} MB")
        
        print(f"\nIMPROVEMENTS:")
        if self.results['compression_ratio']:
            print(f"  Compression Ratio: {self.results['compression_ratio']:.2f}x")
        if self.results['speedup']:
            print(f"  Speedup: {self.results['speedup']:.2f}x")
        
        if 'accuracy' in self.results['original'] and 'accuracy' in self.results['quantized']:
            accuracy_drop = self.results['original']['accuracy'] - self.results['quantized']['accuracy']
            print(f"  Accuracy Drop: {accuracy_drop:.4f}")
