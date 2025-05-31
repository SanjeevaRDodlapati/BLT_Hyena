"""Task-specific heads for genomic sequence analysis."""

from typing import Any

import torch
import torch.nn as nn

from ..config import HyenaGLTConfig


class SequenceClassificationHead(nn.Module):
    """Head for sequence-level classification tasks (e.g., species classification, sequence type)."""

    def __init__(
        self,
        config: HyenaGLTConfig,
        num_classes: int,
        pooling_strategy: str = "cls",  # "cls", "mean", "max", "attention"
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        self.hidden_size = config.hidden_size

        # Pooling layers
        if pooling_strategy == "attention":
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_query = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size // 4, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, num_classes)
        """
        # Pool sequence representations
        pooled = self._pool_sequence(hidden_states, attention_mask)

        # Apply classifier
        logits = self.classifier(pooled)

        return logits

    def _pool_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool sequence into a single representation."""

        if self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0]

        elif self.pooling_strategy == "mean":
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(
                    hidden_states.size()
                )
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(
                    attention_mask.sum(dim=1, keepdim=True), min=1e-9
                )
                return sum_embeddings / sum_mask
            else:
                return hidden_states.mean(dim=1)

        elif self.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                hidden_states = hidden_states.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, -1e9
                )
            return hidden_states.max(dim=1)[0]

        elif self.pooling_strategy == "attention":
            # Attention-based pooling
            batch_size = hidden_states.size(0)
            query = self.attention_query.expand(batch_size, -1, -1)

            pooled, _ = self.attention_pooling(
                query=query,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=(
                    attention_mask == 0 if attention_mask is not None else None
                ),
            )

            return pooled.squeeze(1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")


class TokenClassificationHead(nn.Module):
    """Head for token-level classification tasks (e.g., gene annotation, functional annotation)."""

    def __init__(
        self,
        config: HyenaGLTConfig,
        num_classes: int,
        dropout: float = 0.1,
        use_crf: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.use_crf = use_crf

        # Token classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size // 2, num_classes),
        )

        # Optional CRF layer for sequence labeling
        if use_crf:
            try:
                from torchcrf import CRF

                self.crf = CRF(num_classes, batch_first=True)
            except ImportError:
                print("Warning: torchcrf not installed, using standard classification")
                self.crf = None
                self.use_crf = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            labels: (batch, seq_len) ground truth labels (for training)
            attention_mask: (batch, seq_len)

        Returns:
            Dictionary with logits, loss (if labels provided), and predictions
        """
        # Get token logits
        logits = self.classifier(hidden_states)  # (batch, seq_len, num_classes)

        outputs = {"logits": logits}

        if self.use_crf and self.crf is not None:
            # Use CRF for structured prediction
            if labels is not None:
                # Compute CRF loss
                crf_mask = attention_mask.bool() if attention_mask is not None else None
                loss = -self.crf(logits, labels, mask=crf_mask, reduction="mean")
                outputs["loss"] = loss

                # Get CRF predictions
                predictions = self.crf.decode(logits, mask=crf_mask)
                outputs["predictions"] = predictions
            else:
                # Just get predictions
                crf_mask = attention_mask.bool() if attention_mask is not None else None
                predictions = self.crf.decode(logits, mask=crf_mask)
                outputs["predictions"] = predictions

        else:
            # Standard classification
            if labels is not None:
                # Compute cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

                # Flatten for loss computation
                active_loss = (
                    attention_mask.view(-1) == 1 if attention_mask is not None else None
                )
                if active_loss is not None:
                    active_logits = logits.view(-1, self.num_classes)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

                outputs["loss"] = loss

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            outputs["predictions"] = predictions

        return outputs


class SequenceGenerationHead(nn.Module):
    """Head for sequence generation tasks (e.g., sequence completion, motif generation)."""

    def __init__(
        self,
        config: HyenaGLTConfig,
        vocab_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Generation head
        self.generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, vocab_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize generator weights."""
        for module in self.generator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            labels: (batch, seq_len) target token IDs (for training)
            attention_mask: (batch, seq_len)

        Returns:
            Dictionary with logits, loss (if labels provided)
        """
        # Get generation logits
        logits = self.generator(hidden_states)  # (batch, seq_len, vocab_size)

        outputs = {"logits": logits}

        if labels is not None:
            # Compute language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            # Apply attention mask if provided
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                active_loss = shift_mask.view(-1) == 1
                active_logits = shift_logits.view(-1, self.vocab_size)[active_loss]
                active_labels = shift_labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    shift_logits.view(-1, self.vocab_size), shift_labels.view(-1)
                )

            outputs["loss"] = loss

        return outputs


class MultiTaskHead(nn.Module):
    """Multi-task head that combines multiple genomic tasks."""

    def __init__(
        self,
        config: HyenaGLTConfig,
        task_configs: dict[str, dict[str, Any]],
        task_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.config = config
        self.task_configs = task_configs
        self.task_weights = task_weights or dict.fromkeys(task_configs.keys(), 1.0)

        # Create task-specific heads
        self.task_heads = nn.ModuleDict()

        for task_name, task_config in task_configs.items():
            task_type = task_config["type"]

            if task_type == "sequence_classification":
                head = SequenceClassificationHead(
                    config=config,
                    num_classes=task_config["num_classes"],
                    pooling_strategy=task_config.get("pooling_strategy", "cls"),
                    dropout=task_config.get("dropout", 0.1),
                )

            elif task_type == "token_classification":
                head = TokenClassificationHead(
                    config=config,
                    num_classes=task_config["num_classes"],
                    dropout=task_config.get("dropout", 0.1),
                    use_crf=task_config.get("use_crf", False),
                )

            elif task_type == "sequence_generation":
                head = SequenceGenerationHead(
                    config=config,
                    vocab_size=task_config["vocab_size"],
                    dropout=task_config.get("dropout", 0.1),
                )

            else:
                raise ValueError(f"Unknown task type: {task_type}")

            self.task_heads[task_name] = head

    def forward(
        self,
        hidden_states: torch.Tensor,
        task: str,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            task: Name of the task to perform
            labels: Ground truth labels (if training)
            attention_mask: Attention mask

        Returns:
            Task-specific outputs
        """
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}")

        # Forward through task-specific head
        outputs = self.task_heads[task](
            hidden_states=hidden_states,
            labels=labels,
            attention_mask=attention_mask,
        )

        # Add task information
        outputs["task"] = task

        # Weight the loss if training
        if "loss" in outputs and task in self.task_weights:
            outputs["loss"] = outputs["loss"] * self.task_weights[task]

        return outputs

    def compute_multitask_loss(
        self, task_outputs: dict[str, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute combined loss across multiple tasks."""
        total_loss = 0.0
        num_tasks = 0

        for task_name, outputs in task_outputs.items():
            if "loss" in outputs:
                weight = self.task_weights.get(task_name, 1.0)
                total_loss += weight * outputs["loss"]
                num_tasks += 1

        if num_tasks > 0:
            return total_loss / num_tasks
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)
