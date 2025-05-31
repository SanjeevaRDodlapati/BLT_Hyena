"""Main trainer class for Hyena-GLT models."""

import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optional imports with fallbacks
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Fallback progress bar
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable or range(total or 0)
            self.desc = desc or ""

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            pass


from ..config import HyenaGLTConfig
from ..model import (
    HyenaGLT,
    HyenaGLTForSequenceClassification,
    HyenaGLTForSequenceGeneration,
    HyenaGLTForTokenClassification,
)
from .checkpointing import CheckpointManager
from .curriculum import CurriculumLearning
from .metrics import GenomicMetrics, MultiTaskMetrics
from .multitask import MultiTaskLoss, TaskConfig, TaskWeightScheduler
from .optimization import configure_weight_decay, create_optimizer, create_scheduler


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic training parameters
    num_epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    layer_wise_decay: float | None = None

    # Logging and evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    log_level: str = "INFO"

    # Multi-task learning
    multi_task: bool = False
    task_weights: dict[str, float] | None = None
    task_weighting_strategy: str = "fixed"

    # Curriculum learning
    curriculum_learning: bool = False
    curriculum_strategy: str = "linear"
    curriculum_steps: int = 5000

    # Checkpointing
    output_dir: str = "./outputs"
    save_total_limit: int = 5
    save_best_only: bool = False

    # Mixed precision and efficiency
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0

    # Distributed training
    local_rank: int = -1

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: str = "loss"
    early_stopping_min_delta: float = 0.001


class HyenaGLTTrainer:
    """Comprehensive trainer for Hyena-GLT models."""

    def __init__(
        self,
        model: HyenaGLT | nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader | None = None,
        eval_dataloader: DataLoader | None = None,
        test_dataloader: DataLoader | None = None,
        tokenizer: Any | None = None,
        callbacks: list[Callable] | None = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer
        self.callbacks = callbacks or []

        # Setup logging
        self._setup_logging()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.local_rank != -1:
            torch.cuda.set_device(config.local_rank)
            self.device = torch.device(f"cuda:{config.local_rank}")

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup distributed training
        self.is_distributed = config.local_rank != -1
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[config.local_rank]
            )

        # Setup mixed precision
        self.scaler = None
        if config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        # Setup optimization
        self.optimizer = None
        self.scheduler = None
        self._setup_optimization()

        # Setup multi-task learning
        self.multi_task_loss = None
        self.task_scheduler = None
        if config.multi_task:
            self._setup_multitask()

        # Setup curriculum learning
        self.curriculum = None
        if config.curriculum_learning:
            self._setup_curriculum()

        # Setup metrics
        self.metrics = GenomicMetrics()
        if config.multi_task:
            task_names = (
                list(config.task_weights.keys()) if config.task_weights else ["main"]
            )
            self.metrics = MultiTaskMetrics(task_names)

        # Setup checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(config.output_dir, "checkpoints"),
            max_checkpoints=config.save_total_limit,
            save_best=not config.save_best_only,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")
        self.early_stopping_counter = 0

        # Setup experiment tracking
        if config.use_wandb:
            self._setup_wandb()

        self.logger.info(f"Trainer initialized with device: {self.device}")
        self.logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _setup_optimization(self):
        """Setup optimizer and scheduler."""
        # Configure weight decay
        if self.config.weight_decay > 0:
            configure_weight_decay(self.model, weight_decay=self.config.weight_decay)
        else:
            self.model.parameters()

        # Create optimizer
        self.optimizer = create_optimizer(
            self.model,
            optimizer_type=self.config.optimizer_type,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            layer_wise_decay=self.config.layer_wise_decay,
        )

        # Calculate total steps
        if self.train_dataloader:
            steps_per_epoch = (
                len(self.train_dataloader) // self.config.gradient_accumulation_steps
            )
            total_steps = steps_per_epoch * self.config.num_epochs
        else:
            total_steps = 10000  # Default fallback

        # Create scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=self.config.scheduler_type,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
        )

        self.logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        self.logger.info(f"Scheduler: {type(self.scheduler).__name__}")

    def _setup_multitask(self):
        """Setup multi-task learning components."""
        if not self.config.task_weights:
            self.logger.warning("Multi-task enabled but no task weights provided")
            return

        # Create task configurations
        tasks = []
        for task_name, weight in self.config.task_weights.items():
            task_config = TaskConfig(name=task_name, weight=weight)
            tasks.append(task_config)

        # Create multi-task loss
        self.multi_task_loss = MultiTaskLoss(
            tasks=tasks, weighting_strategy=self.config.task_weighting_strategy
        )

        # Create task scheduler if needed
        if self.config.curriculum_learning:
            self.task_scheduler = TaskWeightScheduler(
                tasks=tasks, schedule_type=self.config.curriculum_strategy
            )

        self.logger.info(f"Multi-task learning enabled with {len(tasks)} tasks")

    def _setup_curriculum(self):
        """Setup curriculum learning."""
        from .curriculum import GenomicComplexityDifficulty, SequenceLengthDifficulty

        # Create difficulty measures
        difficulty_measures = [
            SequenceLengthDifficulty(normalize=True, max_length=1024),
            GenomicComplexityDifficulty(),
        ]

        # Create curriculum
        self.curriculum = CurriculumLearning(
            difficulty_measures=difficulty_measures,
            curriculum_strategy=self.config.curriculum_strategy,
            curriculum_steps=self.config.curriculum_steps,
        )

        self.logger.info("Curriculum learning enabled")

    def _setup_wandb(self):
        """Setup Weights & Biases tracking."""
        if not HAS_WANDB:
            self.logger.warning("wandb not available, disabling experiment tracking")
            self.config.use_wandb = False
            return

        try:
            wandb.init(
                project=self.config.wandb_project or "hyena-glt",
                name=self.config.wandb_run_name,
                config=asdict(self.config),
            )
            self.logger.info("Wandb initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False

    def train(self) -> dict[str, Any]:
        """Main training loop."""
        self.logger.info("Starting training...")

        if not self.train_dataloader:
            raise ValueError("No training dataloader provided")

        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.train()

        # Training loop
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        progress_bar = tqdm(total=total_steps, desc="Training")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0

            for step, batch in enumerate(self.train_dataloader):
                # Forward pass
                loss, metrics = self._training_step(batch)
                epoch_loss += loss.item()

                # Backward pass
                if self.config.fp16 and self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        if self.config.fp16 and self.scaler:
                            self.scaler.unscale_(self.optimizer)

                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    # Optimizer step
                    if self.config.fp16 and self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if self.scheduler:
                        self.scheduler.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics({"train_loss": loss.item(), **metrics})

                # Evaluation
                if (
                    self.global_step % self.config.eval_steps == 0
                    and self.eval_dataloader
                ):
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, prefix="eval")

                    # Early stopping check
                    if self.config.early_stopping:
                        self._check_early_stopping(eval_metrics)

                    self.model.train()  # Return to training mode

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

                # Early stopping
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break

            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            self.logger.info(
                f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}"
            )

            if self.early_stopping_counter >= self.config.early_stopping_patience:
                break

        progress_bar.close()

        # Final evaluation
        if self.eval_dataloader:
            final_metrics = self.evaluate()
            self.logger.info(f"Final evaluation metrics: {final_metrics}")

        # Save final checkpoint
        self._save_checkpoint()

        self.logger.info("Training completed!")

        return {
            "final_step": self.global_step,
            "final_epoch": self.epoch,
            "best_metric": self.best_metric,
        }

    def _training_step(self, batch: dict[str, torch.Tensor]) -> tuple:
        """Single training step."""
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Curriculum learning
        if self.curriculum:
            self.curriculum.update_curriculum(self.global_step)

        # Forward pass
        if self.config.fp16:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
        else:
            outputs = self.model(**batch)

        # Compute loss
        if self.multi_task_loss and isinstance(outputs, dict):
            # Multi-task loss
            predictions = {k: v for k, v in outputs.items() if k.endswith("_logits")}
            targets = {
                k.replace("_logits", ""): batch[k.replace("_logits", "_labels")]
                for k in predictions.keys()
                if k.replace("_logits", "_labels") in batch
            }

            loss_dict = self.multi_task_loss(predictions, targets, self.global_step)
            loss = loss_dict["total_loss"]
            metrics = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()
                if k != "total_loss"
            }
        else:
            # Single task loss
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            metrics = {}

        return loss, metrics

    def evaluate(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        """Evaluate the model."""
        if dataloader is None:
            dataloader = self.eval_dataloader

        if not dataloader:
            return {}

        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                if self.config.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                # Extract loss and predictions
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                    predictions = (
                        outputs.logits
                        if hasattr(outputs, "logits")
                        else outputs.prediction_scores
                    )
                else:
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    predictions = outputs.get(
                        "logits", outputs.get("prediction_scores")
                    )

                targets = batch.get("labels", batch.get("input_ids"))

                total_loss += loss.item()
                num_batches += 1

                # Update metrics
                if predictions is not None and targets is not None:
                    self.metrics.update(predictions, targets)

        # Compute metrics
        eval_metrics = self.metrics.compute()
        eval_metrics["eval_loss"] = total_loss / num_batches if num_batches > 0 else 0.0

        return eval_metrics

    def _log_metrics(self, metrics: dict[str, float], prefix: str = ""):
        """Log metrics to console and wandb."""
        # Add prefix
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

        # Add step and epoch
        metrics.update(
            {
                "step": self.global_step,
                "epoch": self.epoch,
                "learning_rate": (
                    self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
                ),
            }
        )

        # Log to console
        metric_str = " | ".join(
            [f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, int | float)]
        )
        self.logger.info(f"Step {self.global_step} | {metric_str}")

        # Log to wandb
        if self.config.use_wandb:
            wandb.log(metrics, step=self.global_step)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        # Get current metrics
        metrics = {"step": self.global_step, "epoch": self.epoch}

        if self.eval_dataloader:
            eval_metrics = self.evaluate()
            metrics.update(eval_metrics)
            self.model.train()  # Return to training mode

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model.module if self.is_distributed else self.model,
            step=self.global_step,
            epoch=self.epoch,
            metrics=metrics,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def _check_early_stopping(self, metrics: dict[str, float]):
        """Check early stopping criteria."""
        metric_value = metrics.get(self.config.early_stopping_metric)
        if metric_value is None:
            return

        # Check if metric improved
        if metric_value < self.best_metric - self.config.early_stopping_min_delta:
            self.best_metric = metric_value
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        self.logger.info(
            f"Early stopping: {self.early_stopping_counter}/{self.config.early_stopping_patience} "
            f"(best {self.config.early_stopping_metric}: {self.best_metric:.4f})"
        )

    def predict(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Generate predictions on a dataset."""
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                outputs = self.model(**batch)

                # Extract predictions
                if hasattr(outputs, "logits"):
                    predictions = outputs.logits
                elif hasattr(outputs, "prediction_scores"):
                    predictions = outputs.prediction_scores
                else:
                    predictions = outputs

                all_predictions.append(predictions.cpu())

                if "labels" in batch:
                    all_targets.append(batch["labels"].cpu())

        result = {"predictions": torch.cat(all_predictions, dim=0)}
        if all_targets:
            result["targets"] = torch.cat(all_targets, dim=0)

        return result

    def save_model(self, save_directory: str):
        """Save model and tokenizer."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_to_save = self.model.module if self.is_distributed else self.model
        model_to_save.save_pretrained(save_path)

        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)

        # Save training config
        config_path = save_path / "training_config.json"
        with open(config_path, "w") as f:
            import json

            json.dump(asdict(self.config), f, indent=2)

        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, load_directory: str):
        """Load model from directory."""
        self.checkpoint_manager.load_model_from_checkpoint(
            self.model.module if self.is_distributed else self.model, load_directory
        )
        self.logger.info(f"Model loaded from {load_directory}")

    @classmethod
    def from_config(
        cls,
        model_config: HyenaGLTConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        **kwargs,
    ) -> "HyenaGLTTrainer":
        """Create trainer from configurations."""

        # Create model based on task
        if hasattr(model_config, "task_type"):
            if model_config.task_type == "sequence_classification":
                model = HyenaGLTForSequenceClassification(model_config)
            elif model_config.task_type == "token_classification":
                model = HyenaGLTForTokenClassification(model_config)
            elif model_config.task_type == "sequence_generation":
                model = HyenaGLTForSequenceGeneration(model_config)
            else:
                model = HyenaGLT(model_config)
        else:
            model = HyenaGLT(model_config)

        return cls(
            model=model,
            config=training_config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            **kwargs,
        )
