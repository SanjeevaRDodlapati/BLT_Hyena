"""
Model Pruning for Hyena-GLT

This module provides comprehensive pruning support for the Hyena-GLT model,
including magnitude-based pruning, gradient-based pruning, and structured pruning.
"""

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from ..model import HyenaGLT

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration for model pruning."""

    # Pruning method
    method: str = "magnitude"  # "magnitude", "gradient", "random", "structured"

    # Pruning parameters
    sparsity: float = 0.5  # Target sparsity level
    global_pruning: bool = True  # Global vs layer-wise pruning

    # Structured pruning parameters
    structured_type: str = "filter"  # "filter", "channel", "head"
    structured_dim: int = 0  # Dimension to prune

    # Gradual pruning parameters
    gradual_pruning: bool = False
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.5
    pruning_frequency: int = 100  # Steps between pruning updates

    # Fine-tuning parameters
    finetune_epochs: int = 5
    finetune_lr: float = 1e-5

    # Layer-specific settings
    skip_layers: list[str] = None
    layer_sparsities: dict[str, float] = None

    # Advanced options
    importance_score: str = "magnitude"  # "magnitude", "gradient", "taylor"
    normalize_scores: bool = True

    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = []
        if self.layer_sparsities is None:
            self.layer_sparsities = {}


class ModelPruner:
    """Main pruning interface for Hyena-GLT models."""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.pruned_model = None
        self.pruning_masks = {}

    def prune_model(
        self,
        model: HyenaGLT,
        train_loader: torch.utils.data.DataLoader | None = None,
        save_path: str | None = None,
    ) -> nn.Module:
        """
        Prune a Hyena-GLT model.

        Args:
            model: The model to prune
            train_loader: Data loader for gradient-based pruning and fine-tuning
            save_path: Path to save the pruned model

        Returns:
            Pruned model
        """
        if self.config.method == "magnitude":
            pruned_model = self._magnitude_prune(model)
        elif self.config.method == "gradient":
            if train_loader is None:
                raise ValueError("Training loader required for gradient-based pruning")
            pruned_model = self._gradient_prune(model, train_loader)
        elif self.config.method == "structured":
            pruned_model = self._structured_prune(model)
        elif self.config.method == "random":
            pruned_model = self._random_prune(model)
        else:
            raise ValueError(f"Unknown pruning method: {self.config.method}")

        # Fine-tune if requested
        if train_loader and self.config.finetune_epochs > 0:
            pruned_model = self._finetune_pruned_model(pruned_model, train_loader)

        self.pruned_model = pruned_model

        if save_path:
            self.save_pruned_model(save_path)

        return pruned_model

    def _magnitude_prune(self, model: HyenaGLT) -> nn.Module:
        """Apply magnitude-based pruning."""
        logger.info("Applying magnitude-based pruning...")

        model_copy = copy.deepcopy(model)

        if self.config.global_pruning:
            # Global magnitude pruning
            parameters_to_prune = []
            for name, module in model_copy.named_modules():
                if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                    if name not in self.config.skip_layers:
                        parameters_to_prune.append((module, "weight"))

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config.sparsity,
            )
        else:
            # Layer-wise magnitude pruning
            for name, module in model_copy.named_modules():
                if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                    if name not in self.config.skip_layers:
                        sparsity = self.config.layer_sparsities.get(
                            name, self.config.sparsity
                        )
                        prune.l1_unstructured(module, name="weight", amount=sparsity)

        # Store masks
        self._extract_masks(model_copy)

        logger.info("Magnitude-based pruning completed")
        return model_copy

    def _gradient_prune(
        self, model: HyenaGLT, train_loader: torch.utils.data.DataLoader
    ) -> nn.Module:
        """Apply gradient-based pruning."""
        logger.info("Applying gradient-based pruning...")

        # Calculate gradient-based importance scores
        gradient_pruner = GradientPruner(self.config)
        importance_scores = gradient_pruner.compute_importance_scores(
            model, train_loader
        )

        model_copy = copy.deepcopy(model)

        # Apply pruning based on importance scores
        if self.config.global_pruning:
            # Global pruning using importance scores
            all_scores = []
            param_mapping = []

            for name, module in model_copy.named_modules():
                if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                    if (
                        name not in self.config.skip_layers
                        and name in importance_scores
                    ):
                        scores = importance_scores[name].flatten()
                        all_scores.extend(scores.tolist())
                        param_mapping.extend(
                            [(module, "weight", i) for i in range(len(scores))]
                        )

            # Find threshold for global sparsity
            all_scores = np.array(all_scores)
            threshold_idx = int(len(all_scores) * self.config.sparsity)
            threshold = np.partition(all_scores, threshold_idx)[threshold_idx]

            # Apply masks
            for name, module in model_copy.named_modules():
                if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                    if (
                        name not in self.config.skip_layers
                        and name in importance_scores
                    ):
                        mask = importance_scores[name] > threshold
                        prune.custom_from_mask(module, name="weight", mask=mask)

        # Store masks
        self._extract_masks(model_copy)

        logger.info("Gradient-based pruning completed")
        return model_copy

    def _structured_prune(self, model: HyenaGLT) -> nn.Module:
        """Apply structured pruning."""
        logger.info("Applying structured pruning...")

        model_copy = copy.deepcopy(model)
        structured_pruner = StructuredPruner(self.config)

        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                if name not in self.config.skip_layers:
                    structured_pruner.prune_module(module, name)

        logger.info("Structured pruning completed")
        return model_copy

    def _random_prune(self, model: HyenaGLT) -> nn.Module:
        """Apply random pruning (baseline)."""
        logger.info("Applying random pruning...")

        model_copy = copy.deepcopy(model)

        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                if name not in self.config.skip_layers:
                    prune.random_unstructured(
                        module, name="weight", amount=self.config.sparsity
                    )

        # Store masks
        self._extract_masks(model_copy)

        logger.info("Random pruning completed")
        return model_copy

    def _finetune_pruned_model(
        self, model: nn.Module, train_loader: torch.utils.data.DataLoader
    ) -> nn.Module:
        """Fine-tune the pruned model."""
        logger.info("Fine-tuning pruned model...")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.finetune_lr)
        criterion = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(self.config.finetune_epochs):
            epoch_loss = 0.0

            for _batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                # Forward pass
                output = model(data)
                loss = criterion(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            logger.info(
                f"Fine-tuning Epoch {epoch+1}/{self.config.finetune_epochs}, Loss: {avg_loss:.4f}"
            )

        return model

    def _extract_masks(self, model: nn.Module):
        """Extract pruning masks from the model."""
        self.pruning_masks = {}

        for name, module in model.named_modules():
            if hasattr(module, "weight_mask"):
                self.pruning_masks[name] = module.weight_mask.clone()

    def get_sparsity_info(self, model: nn.Module) -> dict[str, Any]:
        """Get detailed sparsity information."""
        sparsity_info = {
            "overall_sparsity": 0.0,
            "layer_sparsities": {},
            "total_params": 0,
            "pruned_params": 0,
        }

        total_params = 0
        pruned_params = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                weight = module.weight
                layer_total = weight.numel()
                layer_pruned = (weight == 0).sum().item()

                total_params += layer_total
                pruned_params += layer_pruned

                layer_sparsity = layer_pruned / layer_total
                sparsity_info["layer_sparsities"][name] = layer_sparsity

        sparsity_info["total_params"] = total_params
        sparsity_info["pruned_params"] = pruned_params
        sparsity_info["overall_sparsity"] = (
            pruned_params / total_params if total_params > 0 else 0.0
        )

        return sparsity_info

    def save_pruned_model(self, save_path: str):
        """Save the pruned model and masks."""
        if self.pruned_model is None:
            raise ValueError("No pruned model to save")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = save_path / "pruned_model.pt"
        torch.save(self.pruned_model.state_dict(), model_path)

        # Save masks
        if self.pruning_masks:
            masks_path = save_path / "pruning_masks.pt"
            torch.save(self.pruning_masks, masks_path)

        # Save config
        config_path = save_path / "pruning_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Save sparsity info
        sparsity_info = self.get_sparsity_info(self.pruned_model)
        sparsity_path = save_path / "sparsity_info.json"
        with open(sparsity_path, "w") as f:
            json.dump(sparsity_info, f, indent=2)

        logger.info(f"Pruned model saved to {save_path}")


class StructuredPruner:
    """Structured pruning implementation."""

    def __init__(self, config: PruningConfig):
        self.config = config

    def prune_module(self, module: nn.Module, name: str):
        """Apply structured pruning to a module."""
        if isinstance(module, nn.Linear):
            self._prune_linear(module)
        elif isinstance(module, nn.Conv1d | nn.Conv2d):
            self._prune_conv(module)

    def _prune_linear(self, module: nn.Linear):
        """Prune linear layer."""
        if self.config.structured_type == "channel":
            # Prune input channels
            self._prune_linear_channels(module, dim=1)
        elif self.config.structured_type == "filter":
            # Prune output channels
            self._prune_linear_channels(module, dim=0)

    def _prune_conv(self, module: nn.Conv1d | nn.Conv2d):
        """Prune convolutional layer."""
        if self.config.structured_type == "filter":
            # Prune output filters
            self._prune_conv_filters(module)
        elif self.config.structured_type == "channel":
            # Prune input channels
            self._prune_conv_channels(module)

    def _prune_linear_channels(self, module: nn.Linear, dim: int):
        """Prune channels in linear layer."""
        weight = module.weight

        # Calculate importance scores for each channel
        if dim == 0:  # Output channels
            importance = torch.norm(weight, dim=1)
        else:  # Input channels
            importance = torch.norm(weight, dim=0)

        # Determine channels to prune
        num_channels = importance.size(0)
        num_to_prune = int(num_channels * self.config.sparsity)

        _, indices_to_prune = torch.topk(importance, num_to_prune, largest=False)

        # Create mask
        mask = torch.ones_like(importance, dtype=torch.bool)
        mask[indices_to_prune] = False

        # Apply structured pruning (this is a simplified version)
        if dim == 0:
            prune.custom_from_mask(module, name="weight", mask=mask.unsqueeze(1))
        else:
            prune.custom_from_mask(module, name="weight", mask=mask.unsqueeze(0))

    def _prune_conv_filters(self, module: nn.Conv1d | nn.Conv2d):
        """Prune filters in convolutional layer."""
        weight = module.weight

        # Calculate importance scores for each filter
        importance = torch.norm(weight.view(weight.size(0), -1), dim=1)

        # Determine filters to prune
        num_filters = importance.size(0)
        num_to_prune = int(num_filters * self.config.sparsity)

        _, indices_to_prune = torch.topk(importance, num_to_prune, largest=False)

        # Create mask
        mask = torch.ones_like(importance, dtype=torch.bool)
        mask[indices_to_prune] = False

        # Apply structured pruning
        mask_shape = [1] * weight.dim()
        mask_shape[0] = weight.size(0)
        mask = mask.view(mask_shape).expand_as(weight)

        prune.custom_from_mask(module, name="weight", mask=mask)

    def _prune_conv_channels(self, module: nn.Conv1d | nn.Conv2d):
        """Prune input channels in convolutional layer."""
        weight = module.weight

        # Calculate importance scores for each input channel
        importance = torch.norm(
            weight.view(weight.size(0), weight.size(1), -1), dim=(0, 2)
        )

        # Determine channels to prune
        num_channels = importance.size(0)
        num_to_prune = int(num_channels * self.config.sparsity)

        _, indices_to_prune = torch.topk(importance, num_to_prune, largest=False)

        # Create mask
        mask = torch.ones_like(importance, dtype=torch.bool)
        mask[indices_to_prune] = False

        # Apply structured pruning
        mask_shape = [1] * weight.dim()
        mask_shape[1] = weight.size(1)
        mask = mask.view(mask_shape).expand_as(weight)

        prune.custom_from_mask(module, name="weight", mask=mask)


class UnstructuredPruner:
    """Unstructured pruning implementation."""

    def __init__(self, config: PruningConfig):
        self.config = config

    def prune_model(self, model: nn.Module) -> nn.Module:
        """Apply unstructured pruning to the entire model."""
        if self.config.global_pruning:
            return self._global_unstructured_prune(model)
        else:
            return self._layerwise_unstructured_prune(model)

    def _global_unstructured_prune(self, model: nn.Module) -> nn.Module:
        """Apply global unstructured pruning."""
        parameters_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                if name not in self.config.skip_layers:
                    parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.sparsity,
        )

        return model

    def _layerwise_unstructured_prune(self, model: nn.Module) -> nn.Module:
        """Apply layer-wise unstructured pruning."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                if name not in self.config.skip_layers:
                    sparsity = self.config.layer_sparsities.get(
                        name, self.config.sparsity
                    )
                    prune.l1_unstructured(module, name="weight", amount=sparsity)

        return model


class MagnitudePruner:
    """Magnitude-based pruning implementation."""

    def __init__(self, config: PruningConfig):
        self.config = config

    def compute_importance_scores(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Compute magnitude-based importance scores."""
        importance_scores = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                importance_scores[name] = torch.abs(module.weight)

        return importance_scores


class GradientPruner:
    """Gradient-based pruning implementation."""

    def __init__(self, config: PruningConfig):
        self.config = config

    def compute_importance_scores(
        self, model: nn.Module, train_loader: torch.utils.data.DataLoader
    ) -> dict[str, torch.Tensor]:
        """Compute gradient-based importance scores."""
        importance_scores = {}

        # Initialize accumulator
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                importance_scores[name] = torch.zeros_like(module.weight)

        model.train()
        criterion = nn.CrossEntropyLoss()

        # Accumulate gradients
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 100:  # Limit samples for efficiency
                break

            model.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Accumulate gradient magnitudes
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                    if module.weight.grad is not None:
                        importance_scores[name] += torch.abs(module.weight.grad)

        # Normalize by number of samples
        num_samples = min(100, len(train_loader))
        for name in importance_scores:
            importance_scores[name] /= num_samples

        return importance_scores


class PruningScheduler:
    """Scheduler for gradual pruning."""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.current_step = 0

    def get_current_sparsity(self) -> float:
        """Get current sparsity level based on schedule."""
        if not self.config.gradual_pruning:
            return self.config.final_sparsity

        # Linear schedule
        progress = min(1.0, self.current_step / (self.config.pruning_frequency * 10))
        sparsity = self.config.initial_sparsity + progress * (
            self.config.final_sparsity - self.config.initial_sparsity
        )

        return sparsity

    def step(self):
        """Advance the scheduler."""
        self.current_step += 1

    def should_prune(self) -> bool:
        """Check if pruning should be applied at current step."""
        return self.current_step % self.config.pruning_frequency == 0
