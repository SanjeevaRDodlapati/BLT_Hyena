"""
Knowledge Distillation for Hyena-GLT

This module provides comprehensive knowledge distillation support for creating
smaller, more efficient Hyena-GLT models while maintaining performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
import logging
import numpy as np
from pathlib import Path
import json

from ..config import HyenaGLTConfig
from ..model import HyenaGLT

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    
    # Temperature for softmax
    temperature: float = 4.0
    
    # Loss weights
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for student loss
    
    # Training parameters
    epochs: int = 20
    learning_rate: float = 1e-4
    batch_size: int = 32
    
    # Distillation type
    distillation_type: str = "response"  # "response", "feature", "attention", "comprehensive"
    
    # Feature distillation parameters
    feature_layers: List[str] = None
    feature_loss_weight: float = 0.1
    
    # Attention distillation parameters
    attention_layers: List[str] = None
    attention_loss_weight: float = 0.1
    
    # Advanced options
    gradual_unfreezing: bool = False
    layer_wise_lr: bool = False
    adaptive_temperature: bool = False
    
    def __post_init__(self):
        if self.feature_layers is None:
            self.feature_layers = []
        if self.attention_layers is None:
            self.attention_layers = []

class KnowledgeDistiller:
    """Main knowledge distillation interface."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.teacher_model = None
        self.student_model = None
        self.distilled_model = None
        
    def distill(
        self,
        teacher_model: HyenaGLT,
        student_model: HyenaGLT,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        save_path: Optional[str] = None
    ) -> nn.Module:
        """
        Perform knowledge distillation.
        
        Args:
            teacher_model: The large, pre-trained teacher model
            student_model: The smaller student model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save the distilled model
            
        Returns:
            Distilled student model
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Setup distillation based on type
        if self.config.distillation_type == "response":
            distilled_model = self._response_distillation(train_loader, val_loader)
        elif self.config.distillation_type == "feature":
            distilled_model = self._feature_distillation(train_loader, val_loader)
        elif self.config.distillation_type == "attention":
            distilled_model = self._attention_distillation(train_loader, val_loader)
        elif self.config.distillation_type == "comprehensive":
            distilled_model = self._comprehensive_distillation(train_loader, val_loader)
        else:
            raise ValueError(f"Unknown distillation type: {self.config.distillation_type}")
        
        self.distilled_model = distilled_model
        
        if save_path:
            self.save_distilled_model(save_path)
        
        return distilled_model
    
    def _response_distillation(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """Perform response-based knowledge distillation."""
        logger.info("Starting response-based knowledge distillation...")
        
        optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.epochs):
            self.student_model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Get teacher and student outputs
                with torch.no_grad():
                    teacher_output = self.teacher_model(data)
                
                student_output = self.student_model(data)
                
                # Calculate losses
                distillation_loss = self._distillation_loss(
                    student_output, teacher_output, self.config.temperature
                )
                student_loss = criterion(student_output, target)
                
                # Combined loss
                total_loss = (
                    self.config.alpha * distillation_loss +
                    self.config.beta * student_loss
                )
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            scheduler.step()
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}")
            
            # Validation
            if val_loader:
                val_acc = self._validate(val_loader)
                logger.info(f"Validation Accuracy: {val_acc:.4f}")
        
        return self.student_model
    
    def _feature_distillation(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """Perform feature-based knowledge distillation."""
        logger.info("Starting feature-based knowledge distillation...")
        
        # Setup feature extractors
        feature_distiller = FeatureDistiller(self.config)
        
        optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.epochs):
            self.student_model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Get outputs and features
                teacher_features = {}
                student_features = {}
                
                def teacher_hook(name):
                    def hook(module, input, output):
                        teacher_features[name] = output
                    return hook
                
                def student_hook(name):
                    def hook(module, input, output):
                        student_features[name] = output
                    return hook
                
                # Register hooks
                teacher_hooks = []
                student_hooks = []
                
                for layer_name in self.config.feature_layers:
                    if hasattr(self.teacher_model, layer_name):
                        teacher_layer = getattr(self.teacher_model, layer_name)
                        teacher_hooks.append(
                            teacher_layer.register_forward_hook(teacher_hook(layer_name))
                        )
                    
                    if hasattr(self.student_model, layer_name):
                        student_layer = getattr(self.student_model, layer_name)
                        student_hooks.append(
                            student_layer.register_forward_hook(student_hook(layer_name))
                        )
                
                # Forward pass
                with torch.no_grad():
                    teacher_output = self.teacher_model(data)
                
                student_output = self.student_model(data)
                
                # Calculate losses
                distillation_loss = self._distillation_loss(
                    student_output, teacher_output, self.config.temperature
                )
                student_loss = criterion(student_output, target)
                
                # Feature distillation loss
                feature_loss = feature_distiller.compute_feature_loss(
                    teacher_features, student_features
                )
                
                # Combined loss
                total_loss = (
                    self.config.alpha * distillation_loss +
                    self.config.beta * student_loss +
                    self.config.feature_loss_weight * feature_loss
                )
                
                total_loss.backward()
                optimizer.step()
                
                # Remove hooks
                for hook in teacher_hooks + student_hooks:
                    hook.remove()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}")
        
        return self.student_model
    
    def _attention_distillation(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """Perform attention-based knowledge distillation."""
        logger.info("Starting attention-based knowledge distillation...")
        
        attention_distiller = AttentionDistiller(self.config)
        
        optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.epochs):
            self.student_model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Get attention maps
                teacher_attentions = attention_distiller.extract_attention_maps(
                    self.teacher_model, data
                )
                student_attentions = attention_distiller.extract_attention_maps(
                    self.student_model, data
                )
                
                # Get outputs
                with torch.no_grad():
                    teacher_output = self.teacher_model(data)
                
                student_output = self.student_model(data)
                
                # Calculate losses
                distillation_loss = self._distillation_loss(
                    student_output, teacher_output, self.config.temperature
                )
                student_loss = criterion(student_output, target)
                
                # Attention distillation loss
                attention_loss = attention_distiller.compute_attention_loss(
                    teacher_attentions, student_attentions
                )
                
                # Combined loss
                total_loss = (
                    self.config.alpha * distillation_loss +
                    self.config.beta * student_loss +
                    self.config.attention_loss_weight * attention_loss
                )
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}")
        
        return self.student_model
    
    def _comprehensive_distillation(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """Perform comprehensive knowledge distillation."""
        logger.info("Starting comprehensive knowledge distillation...")
        
        # Combine all distillation methods
        feature_distiller = FeatureDistiller(self.config)
        attention_distiller = AttentionDistiller(self.config)
        
        optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.epochs):
            self.student_model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Get all outputs and intermediate representations
                teacher_features = {}
                student_features = {}
                
                # Feature extraction (similar to feature distillation)
                # ... (implementation would be similar to _feature_distillation)
                
                # Get attention maps
                teacher_attentions = attention_distiller.extract_attention_maps(
                    self.teacher_model, data
                )
                student_attentions = attention_distiller.extract_attention_maps(
                    self.student_model, data
                )
                
                # Get final outputs
                with torch.no_grad():
                    teacher_output = self.teacher_model(data)
                
                student_output = self.student_model(data)
                
                # Calculate all losses
                distillation_loss = self._distillation_loss(
                    student_output, teacher_output, self.config.temperature
                )
                student_loss = criterion(student_output, target)
                
                # Additional losses would be computed here
                feature_loss = torch.tensor(0.0, device=data.device)  # Placeholder
                attention_loss = attention_distiller.compute_attention_loss(
                    teacher_attentions, student_attentions
                )
                
                # Combined loss
                total_loss = (
                    self.config.alpha * distillation_loss +
                    self.config.beta * student_loss +
                    self.config.feature_loss_weight * feature_loss +
                    self.config.attention_loss_weight * attention_loss
                )
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}")
        
        return self.student_model
    
    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute distillation loss."""
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
    
    def _validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Validate the student model."""
        self.student_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.student_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def save_distilled_model(self, save_path: str):
        """Save the distilled model."""
        if self.distilled_model is None:
            raise ValueError("No distilled model to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_path / "distilled_model.pt"
        torch.save(self.distilled_model.state_dict(), model_path)
        
        # Save config
        config_path = save_path / "distillation_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Distilled model saved to {save_path}")

class FeatureDistiller:
    """Feature-based knowledge distillation."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
    
    def compute_feature_loss(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature-based distillation loss."""
        total_loss = 0.0
        
        for layer_name in self.config.feature_layers:
            if layer_name in teacher_features and layer_name in student_features:
                teacher_feat = teacher_features[layer_name]
                student_feat = student_features[layer_name]
                
                # Adapt dimensions if necessary
                if teacher_feat.shape != student_feat.shape:
                    student_feat = self._adapt_features(student_feat, teacher_feat.shape)
                
                # MSE loss between features
                loss = F.mse_loss(student_feat, teacher_feat)
                total_loss += loss
        
        return total_loss
    
    def _adapt_features(
        self,
        student_feat: torch.Tensor,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """Adapt student features to match teacher dimensions."""
        # Simple linear projection for dimension matching
        if len(student_feat.shape) == 3 and len(target_shape) == 3:
            # Sequence models: [batch, seq_len, hidden_dim]
            if student_feat.size(-1) != target_shape[-1]:
                adapter = nn.Linear(student_feat.size(-1), target_shape[-1])
                student_feat = adapter(student_feat)
        
        return student_feat

class AttentionDistiller:
    """Attention-based knowledge distillation."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
    
    def extract_attention_maps(
        self,
        model: nn.Module,
        input_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract attention maps from the model."""
        attention_maps = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                # Assuming attention output includes attention weights
                if isinstance(output, tuple) and len(output) > 1:
                    attention_maps[name] = output[1]  # Attention weights
                else:
                    attention_maps[name] = output
            return hook
        
        hooks = []
        
        # Register hooks for attention layers
        for layer_name in self.config.attention_layers:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                hooks.append(layer.register_forward_hook(attention_hook(layer_name)))
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def compute_attention_loss(
        self,
        teacher_attentions: Dict[str, torch.Tensor],
        student_attentions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention-based distillation loss."""
        total_loss = 0.0
        
        for layer_name in self.config.attention_layers:
            if layer_name in teacher_attentions and layer_name in student_attentions:
                teacher_attn = teacher_attentions[layer_name]
                student_attn = student_attentions[layer_name]
                
                # Adapt dimensions if necessary
                if teacher_attn.shape != student_attn.shape:
                    student_attn = self._adapt_attention(student_attn, teacher_attn.shape)
                
                # MSE loss between attention maps
                loss = F.mse_loss(student_attn, teacher_attn)
                total_loss += loss
        
        return total_loss
    
    def _adapt_attention(
        self,
        student_attn: torch.Tensor,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """Adapt student attention to match teacher dimensions."""
        # Handle different attention dimensions
        if student_attn.dim() == 4 and len(target_shape) == 4:
            # Multi-head attention: [batch, heads, seq_len, seq_len]
            if student_attn.size(1) != target_shape[1]:
                # Average over heads if different number of heads
                student_attn = student_attn.mean(dim=1, keepdim=True)
                student_attn = student_attn.repeat(1, target_shape[1], 1, 1)
        
        return student_attn

class StudentModelFactory:
    """Factory for creating student models with different architectures."""
    
    @staticmethod
    def create_compressed_model(
        teacher_config: HyenaGLTConfig,
        compression_ratio: float = 0.5
    ) -> HyenaGLT:
        """Create a compressed version of the teacher model."""
        student_config = StudentModelFactory._compress_config(teacher_config, compression_ratio)
        return HyenaGLT(student_config)
    
    @staticmethod
    def create_depth_compressed_model(
        teacher_config: HyenaGLTConfig,
        layer_ratio: float = 0.5
    ) -> HyenaGLT:
        """Create a student model with fewer layers."""
        student_config = copy.deepcopy(teacher_config)
        student_config.num_layers = int(teacher_config.num_layers * layer_ratio)
        return HyenaGLT(student_config)
    
    @staticmethod
    def create_width_compressed_model(
        teacher_config: HyenaGLTConfig,
        width_ratio: float = 0.5
    ) -> HyenaGLT:
        """Create a student model with smaller hidden dimensions."""
        student_config = copy.deepcopy(teacher_config)
        student_config.hidden_dim = int(teacher_config.hidden_dim * width_ratio)
        student_config.intermediate_dim = int(teacher_config.intermediate_dim * width_ratio)
        return HyenaGLT(student_config)
    
    @staticmethod
    def _compress_config(
        teacher_config: HyenaGLTConfig,
        compression_ratio: float
    ) -> HyenaGLTConfig:
        """Compress the model configuration."""
        import copy
        student_config = copy.deepcopy(teacher_config)
        
        # Compress various dimensions
        student_config.hidden_dim = int(teacher_config.hidden_dim * compression_ratio)
        student_config.intermediate_dim = int(teacher_config.intermediate_dim * compression_ratio)
        student_config.num_layers = max(1, int(teacher_config.num_layers * compression_ratio))
        
        # Adjust other parameters proportionally
        if hasattr(student_config, 'num_heads'):
            student_config.num_heads = max(1, int(teacher_config.num_heads * compression_ratio))
        
        return student_config

class DistillationBenchmark:
    """Benchmark distilled models against teacher and student baselines."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        distilled_model: nn.Module,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Compare teacher, student, and distilled models."""
        results = {
            'teacher': self._evaluate_model(teacher_model, test_loader),
            'student_baseline': self._evaluate_model(student_model, test_loader),
            'distilled': self._evaluate_model(distilled_model, test_loader)
        }
        
        # Calculate improvements
        results['distillation_gain'] = (
            results['distilled']['accuracy'] - results['student_baseline']['accuracy']
        )
        
        results['knowledge_retention'] = (
            results['distilled']['accuracy'] / results['teacher']['accuracy']
        )
        
        self.results = results
        return results
    
    def _evaluate_model(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate a single model."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                total_loss += loss.item()
        
        return {
            'accuracy': correct / total,
            'loss': total_loss / len(test_loader)
        }
    
    def print_results(self):
        """Print benchmark results."""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*50)
        print("KNOWLEDGE DISTILLATION BENCHMARK RESULTS")
        print("="*50)
        
        for model_type in ['teacher', 'student_baseline', 'distilled']:
            print(f"\n{model_type.upper().replace('_', ' ')} MODEL:")
            print(f"  Accuracy: {self.results[model_type]['accuracy']:.4f}")
            print(f"  Loss: {self.results[model_type]['loss']:.4f}")
        
        print(f"\nDISTILLATION METRICS:")
        print(f"  Distillation Gain: {self.results['distillation_gain']:.4f}")
        print(f"  Knowledge Retention: {self.results['knowledge_retention']:.4f}")
