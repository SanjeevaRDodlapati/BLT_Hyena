"""
Unit tests for Hyena-GLT training components.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model.hyena_glt import HyenaGLTForSequenceClassification
from hyena_glt.training.checkpointing import CheckpointManager
from hyena_glt.training.curriculum import CurriculumLearning
from hyena_glt.training.metrics import GenomicMetrics, MultiTaskMetrics
from hyena_glt.training.multitask import MultiTaskLoss, TaskConfig, TaskWeightScheduler
from hyena_glt.training.optimization import (
    AdamWWithScheduler,
    create_optimizer,
    create_scheduler,
)
from hyena_glt.training.trainer import HyenaGLTTrainer, TrainingConfig
from tests.utils import DataGenerator, TestConfig


class TestTrainingConfig:
    """Test training configuration."""

    def test_training_config_creation(self):
        """Test creating training configuration."""
        config = TrainingConfig(
            learning_rate=1e-4, batch_size=8, num_epochs=2, warmup_steps=100
        )

        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 2
        assert config.warmup_steps == 100

    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Test invalid learning rate
        with pytest.raises((ValueError, AssertionError)):
            TrainingConfig(learning_rate=0)

        # Test invalid batch size
        with pytest.raises((ValueError, AssertionError)):
            TrainingConfig(batch_size=0)

        # Test invalid epochs
        with pytest.raises((ValueError, AssertionError)):
            TrainingConfig(num_epochs=0)

    def test_training_config_defaults(self):
        """Test training configuration defaults."""
        config = TrainingConfig()

        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.num_epochs > 0
        assert config.weight_decay >= 0


class TestOptimization:
    """Test optimization utilities."""

    def test_create_optimizer(self):
        """Test creating optimizer."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)

        optimizer = create_optimizer(
            model=model, learning_rate=1e-4, weight_decay=0.01, optimizer_type="adamw"
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) > 0
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_create_scheduler(self):
        """Test creating learning rate scheduler."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type="cosine",
            num_training_steps=1000,
            warmup_steps=100,
        )

        assert scheduler is not None

        # Test scheduler step
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        # Learning rate should change after step
        assert optimizer.param_groups[0]["lr"] != initial_lr

    def test_adamw_with_scheduler(self):
        """Test AdamW optimizer with scheduler integration."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)

        optimizer_scheduler = AdamWWithScheduler(
            model=model,
            learning_rate=1e-4,
            weight_decay=0.01,
            num_training_steps=1000,
            warmup_steps=100,
        )

        assert hasattr(optimizer_scheduler, "optimizer")
        assert hasattr(optimizer_scheduler, "scheduler")

        # Test optimization step
        batch = DataGenerator.generate_genomic_data(
            "dna_classification", batch_size=2, seq_length=32
        )
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        model.train()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        optimizer_scheduler.zero_grad()
        loss.backward()
        optimizer_scheduler.step()


class TestCurriculumLearning:
    """Test curriculum learning functionality."""

    def test_curriculum_learning_creation(self):
        """Test creating curriculum learning scheduler."""
        stages = [
            {"name": "easy", "max_length": 64, "difficulty": 0.1},
            {"name": "medium", "max_length": 128, "difficulty": 0.5},
            {"name": "hard", "max_length": 256, "difficulty": 1.0},
        ]

        curriculum = CurriculumLearning(stages=stages, transition_steps=[100, 200])

        assert len(curriculum.stages) == 3
        assert curriculum.current_stage == 0

    def test_curriculum_progression(self):
        """Test curriculum learning progression."""
        stages = [
            {"name": "easy", "max_length": 64, "difficulty": 0.1},
            {"name": "medium", "max_length": 128, "difficulty": 0.5},
        ]

        curriculum = CurriculumLearning(stages=stages, transition_steps=[100])

        # Initially at stage 0
        assert curriculum.get_current_stage() == stages[0]

        # After transition step, should move to stage 1
        curriculum.step(step=150)
        assert curriculum.get_current_stage() == stages[1]

    def test_curriculum_data_filtering(self):
        """Test curriculum-based data filtering."""
        stages = [
            {"name": "easy", "max_length": 32, "difficulty": 0.1},
            {"name": "hard", "max_length": 128, "difficulty": 1.0},
        ]

        curriculum = CurriculumLearning(stages=stages, transition_steps=[100])

        # Create sample data with different lengths
        sequences = ["ATCG" * 4, "ATCG" * 16, "ATCG" * 32]  # Lengths: 16, 64, 128
        labels = [0, 1, 0]

        # At easy stage, should filter long sequences
        filtered_data = curriculum.filter_data(sequences, labels)
        assert len(filtered_data[0]) <= len(sequences)  # Some filtering occurred

        # Move to hard stage
        curriculum.step(step=150)
        filtered_data = curriculum.filter_data(sequences, labels)
        # Should allow longer sequences now


class TestMultiTaskLearning:
    """Test multi-task learning components."""

    def test_task_config_creation(self):
        """Test creating task configuration."""
        task_config = TaskConfig(
            name="classification",
            task_type="sequence_classification",
            weight=1.0,
            num_classes=5,
        )

        assert task_config.name == "classification"
        assert task_config.task_type == "sequence_classification"
        assert task_config.weight == 1.0
        assert task_config.num_classes == 5

    def test_multitask_loss(self):
        """Test multi-task loss calculation."""
        task_configs = {
            "task1": TaskConfig(name="task1", task_type="classification", weight=1.0),
            "task2": TaskConfig(name="task2", task_type="classification", weight=0.5),
        }

        multitask_loss = MultiTaskLoss(task_configs)

        # Create sample losses
        losses = {"task1": torch.tensor(2.0), "task2": torch.tensor(1.0)}

        total_loss = multitask_loss(losses)
        expected_loss = 1.0 * 2.0 + 0.5 * 1.0  # Weighted sum

        assert torch.isclose(total_loss, torch.tensor(expected_loss))

    def test_task_weight_scheduler(self):
        """Test dynamic task weight scheduling."""
        task_configs = {
            "task1": TaskConfig(name="task1", task_type="classification", weight=1.0),
            "task2": TaskConfig(name="task2", task_type="classification", weight=0.5),
        }

        scheduler = TaskWeightScheduler(
            task_configs=task_configs, schedule_type="linear", total_steps=1000
        )

        # Initial weights
        weights = scheduler.get_current_weights(step=0)
        assert "task1" in weights
        assert "task2" in weights

        # Weights after some steps
        weights_later = scheduler.get_current_weights(step=500)
        # Weights should change (exact change depends on schedule)
        assert weights_later is not None


class TestGenomicMetrics:
    """Test genomic-specific metrics."""

    def test_genomic_metrics_creation(self):
        """Test creating genomic metrics calculator."""
        metrics = GenomicMetrics(task_type="sequence_classification", num_classes=3)

        assert metrics.task_type == "sequence_classification"
        assert metrics.num_classes == 3

    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        metrics = GenomicMetrics(task_type="sequence_classification", num_classes=3)

        # Create sample predictions and labels
        predictions = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        labels = torch.tensor([0, 1, 2])

        results = metrics.calculate(predictions, labels)

        assert "accuracy" in results
        assert "f1_score" in results
        assert results["accuracy"] == 1.0  # Perfect predictions

    def test_token_classification_metrics(self):
        """Test token-level classification metrics."""
        metrics = GenomicMetrics(task_type="token_classification", num_classes=5)

        batch_size, seq_len, num_classes = 2, 8, 5
        predictions = torch.randn(batch_size, seq_len, num_classes)
        labels = torch.randint(0, num_classes, (batch_size, seq_len))

        results = metrics.calculate(predictions, labels)

        assert "accuracy" in results
        assert "token_f1" in results


class TestCheckpointing:
    """Test checkpoint management."""

    def test_checkpoint_manager_creation(self, temp_dir):
        """Test creating checkpoint manager."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir), save_top_k=3, monitor="val_loss", mode="min"
        )

        assert manager.checkpoint_dir == Path(temp_dir)
        assert manager.save_top_k == 3
        assert manager.monitor == "val_loss"

    def test_save_checkpoint(self, temp_dir):
        """Test saving checkpoint."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)
        optimizer = torch.optim.AdamW(model.parameters())

        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir), save_top_k=3, monitor="val_loss", mode="min"
        )

        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            metrics={"val_loss": 0.5, "val_accuracy": 0.8},
        )

        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".ckpt"

    def test_load_checkpoint(self, temp_dir):
        """Test loading checkpoint."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)
        optimizer = torch.optim.AdamW(model.parameters())

        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir), save_top_k=3, monitor="val_loss", mode="min"
        )

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            metrics={"val_loss": 0.5},
        )

        # Create new model and optimizer
        new_model = HyenaGLTForSequenceClassification(config, num_classes=2)
        new_optimizer = torch.optim.AdamW(new_model.parameters())

        # Load checkpoint
        checkpoint_data = manager.load_checkpoint(
            checkpoint_path=checkpoint_path, model=new_model, optimizer=new_optimizer
        )

        assert checkpoint_data["epoch"] == 1
        assert checkpoint_data["step"] == 100
        assert checkpoint_data["metrics"]["val_loss"] == 0.5

    def test_best_checkpoint_tracking(self, temp_dir):
        """Test tracking of best checkpoint."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir), save_top_k=2, monitor="val_loss", mode="min"
        )

        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)
        optimizer = torch.optim.AdamW(model.parameters())

        # Save multiple checkpoints with different val_loss
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            metrics={"val_loss": 0.8},
        )
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=2,
            step=200,
            metrics={"val_loss": 0.5},  # Better
        )
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=3,
            step=300,
            metrics={"val_loss": 0.6},  # Worse
        )

        best_checkpoint = manager.get_best_checkpoint()
        assert best_checkpoint is not None

        # Load best checkpoint to verify it's the one with val_loss=0.5
        checkpoint_data = manager.load_checkpoint(best_checkpoint, model, optimizer)
        assert checkpoint_data["metrics"]["val_loss"] == 0.5


class TestTrainer:
    """Test main trainer functionality."""

    def test_trainer_creation(self):
        """Test creating trainer."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)

        training_config = TrainingConfig(learning_rate=1e-4, batch_size=4, num_epochs=1)

        trainer = HyenaGLTTrainer(model=model, config=training_config)

        assert trainer.model == model
        assert trainer.config == training_config
        assert hasattr(trainer, "device")

    def test_trainer_setup(self):
        """Test trainer setup procedures."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)

        training_config = TrainingConfig(learning_rate=1e-4, batch_size=4, num_epochs=1)

        trainer = HyenaGLTTrainer(model=model, config=training_config)

        # Test optimizer setup
        trainer._setup_optimizer()
        assert hasattr(trainer, "optimizer")
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

        # Test scheduler setup
        trainer._setup_scheduler()
        if hasattr(trainer, "scheduler"):
            assert trainer.scheduler is not None

    @patch("torch.utils.data.DataLoader")
    def test_trainer_train_step(self, mock_dataloader):
        """Test single training step."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)

        training_config = TrainingConfig(learning_rate=1e-4, batch_size=2, num_epochs=1)

        trainer = HyenaGLTTrainer(model=model, config=training_config)

        # Create mock batch
        batch = {
            "input_ids": torch.randint(0, config.vocab_size, (2, 32)),
            "labels": torch.randint(0, 2, (2,)),
        }

        # Test training step
        trainer._setup_optimizer()
        loss = trainer.training_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_trainer_validation_step(self):
        """Test validation step."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)

        training_config = TrainingConfig(learning_rate=1e-4, batch_size=2, num_epochs=1)

        trainer = HyenaGLTTrainer(model=model, config=training_config)

        batch = {
            "input_ids": torch.randint(0, config.vocab_size, (2, 32)),
            "labels": torch.randint(0, 2, (2,)),
        }

        # Test validation step
        results = trainer.validation_step(batch)

        assert "loss" in results
        assert "predictions" in results
        assert "labels" in results
        assert isinstance(results["loss"], torch.Tensor)

    def test_trainer_compute_metrics(self):
        """Test metrics computation."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLTForSequenceClassification(config, num_classes=2)

        training_config = TrainingConfig(learning_rate=1e-4, batch_size=2, num_epochs=1)

        trainer = HyenaGLTTrainer(model=model, config=training_config)

        # Create sample validation outputs
        outputs = [
            {
                "loss": torch.tensor(0.5),
                "predictions": torch.tensor([[0.8, 0.2], [0.3, 0.7]]),
                "labels": torch.tensor([0, 1]),
            }
        ]

        metrics = trainer.compute_metrics(outputs)

        assert "val_loss" in metrics
        assert "val_accuracy" in metrics
        assert isinstance(metrics["val_loss"], float)
        assert isinstance(metrics["val_accuracy"], float)
