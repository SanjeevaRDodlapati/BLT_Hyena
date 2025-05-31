"""Multi-task learning utilities for Hyena-GLT."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TaskConfig:
    """Configuration for a single task."""

    name: str
    weight: float = 1.0
    loss_fn: Callable | None = None
    metric_names: list[str] = None
    curriculum_schedule: dict | None = None


class MultiTaskLoss(nn.Module):
    """Multi-task loss with various weighting strategies."""

    def __init__(
        self,
        tasks: list[TaskConfig],
        weighting_strategy: str = "fixed",
        temperature: float = 2.0,
        loss_balancing: bool = True,
    ):
        super().__init__()
        self.tasks = {task.name: task for task in tasks}
        self.weighting_strategy = weighting_strategy
        self.temperature = temperature
        self.loss_balancing = loss_balancing

        # Initialize task weights
        self.register_buffer(
            "task_weights",
            torch.tensor([task.weight for task in tasks], dtype=torch.float32),
        )

        # For uncertainty weighting
        if weighting_strategy == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(len(tasks)))

        # For gradient normalization
        if weighting_strategy == "gradnorm":
            self.register_buffer("initial_losses", torch.zeros(len(tasks)))
            self.register_buffer(
                "loss_history", torch.zeros(len(tasks), 10)
            )  # Keep last 10
            self.history_idx = 0
            self.alpha = 0.12  # GradNorm alpha parameter

        # Track task performance
        self.register_buffer("task_performance", torch.ones(len(tasks)))

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        step: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss."""

        task_losses = {}
        task_names = list(self.tasks.keys())

        # Compute individual task losses
        for i, (task_name, task_config) in enumerate(self.tasks.items()):
            if task_name in predictions and task_name in targets:
                if task_config.loss_fn is not None:
                    loss = task_config.loss_fn(
                        predictions[task_name], targets[task_name]
                    )
                else:
                    loss = F.cross_entropy(predictions[task_name], targets[task_name])
                task_losses[task_name] = loss

        if not task_losses:
            return {"total_loss": torch.tensor(0.0, requires_grad=True)}

        # Convert to tensor for easier manipulation
        losses = torch.stack(
            [task_losses[name] for name in task_names if name in task_losses]
        )
        available_tasks = [name for name in task_names if name in task_losses]

        # Update loss history for GradNorm
        if self.weighting_strategy == "gradnorm" and step is not None:
            with torch.no_grad():
                if step == 0:
                    self.initial_losses[: len(losses)] = losses.detach()

                # Update rolling history
                self.loss_history[: len(losses), self.history_idx % 10] = (
                    losses.detach()
                )
                self.history_idx += 1

        # Compute task weights
        weights = self._compute_task_weights(losses, available_tasks, step)

        # Weighted combination
        if self.loss_balancing:
            # Loss balancing: normalize by average loss magnitude
            loss_scale = losses.detach().mean()
            normalized_losses = losses / (loss_scale + 1e-8)
            total_loss = torch.sum(weights * normalized_losses)
        else:
            total_loss = torch.sum(weights * losses)

        # Prepare output
        result = {"total_loss": total_loss}
        for i, task_name in enumerate(available_tasks):
            result[f"{task_name}_loss"] = task_losses[task_name]
            result[f"{task_name}_weight"] = weights[i]

        return result

    def _compute_task_weights(
        self, losses: torch.Tensor, task_names: list[str], step: int | None = None
    ) -> torch.Tensor:
        """Compute task weights based on the weighting strategy."""

        if self.weighting_strategy == "fixed":
            # Use predefined fixed weights
            task_indices = [list(self.tasks.keys()).index(name) for name in task_names]
            return self.task_weights[task_indices]

        elif self.weighting_strategy == "uncertainty":
            # Uncertainty weighting (Multi-Task Learning Using Uncertainty to Weigh Losses)
            task_indices = [list(self.tasks.keys()).index(name) for name in task_names]
            log_vars = self.log_vars[task_indices]
            precision = torch.exp(-log_vars)
            weights = precision / precision.sum()
            return weights

        elif self.weighting_strategy == "gradnorm":
            # GradNorm weighting
            return self._gradnorm_weights(losses, task_names, step)

        elif self.weighting_strategy == "dwa":
            # Dynamic Weight Average
            return self._dwa_weights(losses, task_names)

        elif self.weighting_strategy == "adaptive":
            # Simple adaptive weighting based on relative performance
            return self._adaptive_weights(losses, task_names)

        else:
            # Default to equal weighting
            return torch.ones(len(losses)) / len(losses)

    def _gradnorm_weights(
        self, losses: torch.Tensor, task_names: list[str], step: int | None = None
    ) -> torch.Tensor:
        """Compute GradNorm weights."""
        if step is None or step < 10:
            return torch.ones(len(losses)) / len(losses)

        # Compute relative loss decrease
        current_losses = losses.detach()
        initial_losses = self.initial_losses[: len(losses)]

        # Avoid division by zero
        loss_ratios = current_losses / (initial_losses + 1e-8)

        # Compute average loss ratio
        avg_loss_ratio = loss_ratios.mean()

        # Compute relative task training rates
        r_i = loss_ratios / (avg_loss_ratio + 1e-8)

        # Target weights (inverse of relative training rate raised to alpha)
        target_weights = r_i**self.alpha
        target_weights = target_weights / target_weights.sum()

        return target_weights

    def _dwa_weights(self, losses: torch.Tensor, task_names: list[str]) -> torch.Tensor:
        """Dynamic Weight Average weighting."""
        if self.history_idx < 2:
            return torch.ones(len(losses)) / len(losses)

        # Get previous losses
        prev_idx = (self.history_idx - 1) % 10
        prev_losses = self.loss_history[: len(losses), prev_idx]

        # Compute loss ratios
        loss_ratios = losses.detach() / (prev_losses + 1e-8)

        # Apply temperature scaling and softmax
        weights = F.softmax(loss_ratios / self.temperature, dim=0)

        return weights * len(weights)  # Rescale to maintain magnitude

    def _adaptive_weights(
        self, losses: torch.Tensor, task_names: list[str]
    ) -> torch.Tensor:
        """Simple adaptive weighting based on loss magnitude."""
        # Higher weight for tasks with higher loss (need more attention)
        loss_magnitudes = losses.detach()
        weights = loss_magnitudes / (loss_magnitudes.sum() + 1e-8)
        return weights

    def update_task_performance(self, metrics: dict[str, float]):
        """Update task performance tracking."""
        for i, (task_name, _) in enumerate(self.tasks.items()):
            if task_name in metrics:
                # Use exponential moving average
                self.task_performance[i] = (
                    0.9 * self.task_performance[i] + 0.1 * metrics[task_name]
                )


class TaskWeightScheduler:
    """Scheduler for task weights during curriculum learning."""

    def __init__(
        self,
        tasks: list[TaskConfig],
        schedule_type: str = "linear",
        total_steps: int = 10000,
        warmup_steps: int = 1000,
    ):
        self.tasks = {task.name: task for task in tasks}
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        # Initialize schedules from task configs
        self.schedules = {}
        for task in tasks:
            if task.curriculum_schedule:
                self.schedules[task.name] = task.curriculum_schedule

    def get_weights(self, step: int) -> dict[str, float]:
        """Get task weights for current step."""
        weights = {}

        for task_name, task_config in self.tasks.items():
            if task_name in self.schedules:
                weights[task_name] = self._compute_scheduled_weight(
                    task_name, step, task_config.weight
                )
            else:
                weights[task_name] = task_config.weight

        return weights

    def _compute_scheduled_weight(
        self, task_name: str, step: int, base_weight: float
    ) -> float:
        """Compute scheduled weight for a task."""
        schedule = self.schedules[task_name]

        if self.schedule_type == "linear":
            return self._linear_schedule(schedule, step, base_weight)
        elif self.schedule_type == "cosine":
            return self._cosine_schedule(schedule, step, base_weight)
        elif self.schedule_type == "step":
            return self._step_schedule(schedule, step, base_weight)
        else:
            return base_weight

    def _linear_schedule(self, schedule: dict, step: int, base_weight: float) -> float:
        """Linear weight scheduling."""
        start_weight = schedule.get("start_weight", base_weight)
        end_weight = schedule.get("end_weight", base_weight)
        start_step = schedule.get("start_step", 0)
        end_step = schedule.get("end_step", self.total_steps)

        if step <= start_step:
            return start_weight
        elif step >= end_step:
            return end_weight
        else:
            progress = (step - start_step) / (end_step - start_step)
            return start_weight + progress * (end_weight - start_weight)

    def _cosine_schedule(self, schedule: dict, step: int, base_weight: float) -> float:
        """Cosine weight scheduling."""
        start_weight = schedule.get("start_weight", base_weight)
        end_weight = schedule.get("end_weight", base_weight)
        start_step = schedule.get("start_step", 0)
        end_step = schedule.get("end_step", self.total_steps)

        if step <= start_step:
            return start_weight
        elif step >= end_step:
            return end_weight
        else:
            progress = (step - start_step) / (end_step - start_step)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return end_weight + (start_weight - end_weight) * cosine_factor

    def _step_schedule(self, schedule: dict, step: int, base_weight: float) -> float:
        """Step-wise weight scheduling."""
        milestones = schedule.get("milestones", [])
        weights = schedule.get("weights", [base_weight])

        if not milestones:
            return base_weight

        # Find current weight based on milestones
        current_weight = weights[0] if weights else base_weight
        for i, milestone in enumerate(milestones):
            if step >= milestone and i + 1 < len(weights):
                current_weight = weights[i + 1]

        return current_weight


class BalancedSampler:
    """Balanced sampling across multiple tasks."""

    def __init__(
        self,
        task_datasets: dict[str, torch.utils.data.Dataset],
        sampling_strategy: str = "proportional",
        temperature: float = 1.0,
    ):
        self.task_datasets = task_datasets
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature

        # Compute dataset sizes
        self.dataset_sizes = {
            name: len(dataset) for name, dataset in task_datasets.items()
        }
        self.total_size = sum(self.dataset_sizes.values())

        # Compute sampling probabilities
        self.task_probs = self._compute_task_probabilities()

    def _compute_task_probabilities(self) -> dict[str, float]:
        """Compute sampling probabilities for each task."""
        if self.sampling_strategy == "proportional":
            # Proportional to dataset size
            probs = {
                name: size / self.total_size
                for name, size in self.dataset_sizes.items()
            }
        elif self.sampling_strategy == "uniform":
            # Equal probability for each task
            num_tasks = len(self.task_datasets)
            probs = dict.fromkeys(self.task_datasets.keys(), 1.0 / num_tasks)
        elif self.sampling_strategy == "sqrt":
            # Square root of proportional (helps smaller datasets)
            sqrt_sizes = {
                name: np.sqrt(size) for name, size in self.dataset_sizes.items()
            }
            total_sqrt = sum(sqrt_sizes.values())
            probs = {
                name: sqrt_size / total_sqrt for name, sqrt_size in sqrt_sizes.items()
            }
        else:
            # Default to proportional
            probs = {
                name: size / self.total_size
                for name, size in self.dataset_sizes.items()
            }

        # Apply temperature scaling
        if self.temperature != 1.0:
            log_probs = {
                name: np.log(prob + 1e-8) / self.temperature
                for name, prob in probs.items()
            }
            max_log_prob = max(log_probs.values())
            exp_probs = {
                name: np.exp(log_prob - max_log_prob)
                for name, log_prob in log_probs.items()
            }
            total_exp = sum(exp_probs.values())
            probs = {name: exp_prob / total_exp for name, exp_prob in exp_probs.items()}

        return probs

    def sample_task(self) -> str:
        """Sample a task based on the sampling strategy."""
        tasks = list(self.task_datasets.keys())
        probabilities = [self.task_probs[task] for task in tasks]
        return np.random.choice(tasks, p=probabilities)

    def get_batch_iterator(self, batch_size: int, num_batches: int):
        """Get iterator for balanced multi-task batches."""
        for _ in range(num_batches):
            # Sample task for this batch
            task_name = self.sample_task()
            dataset = self.task_datasets[task_name]

            # Sample batch from selected task
            indices = torch.randint(0, len(dataset), (batch_size,))
            batch = [dataset[idx] for idx in indices]

            yield task_name, batch
