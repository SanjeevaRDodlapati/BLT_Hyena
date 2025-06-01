"""
Pretraining utilities for Hyena-GLT models adapted from savanna.

This module provides comprehensive pretraining functionality including:
- Self-supervised pretraining strategies (AR, MLM, OADM, Span masking)
- Genomic-specific loss functions
- Data loading and batch processing for genomic sequences
- Training loop with evaluation and checkpointing
"""

import gc
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..config import HyenaGLTConfig
from ..data.dataset import GenomicDataset
from ..data.tokenizer import DNATokenizer, RNATokenizer, ProteinTokenizer
from ..model.hyena_glt import HyenaGLT, HyenaGLTForSequenceClassification
from .trainer import HyenaGLTTrainer, TrainingConfig
from .optimization import create_optimizer, create_scheduler
from .metrics import GenomicMetrics
from .checkpointing import CheckpointManager

logger = logging.getLogger(__name__)


class PretrainingConfig:
    """Configuration for pretraining Hyena-GLT models."""
    
    def __init__(
        self,
        # Model configuration
        model_config: Optional[HyenaGLTConfig] = None,
        
        # Training configuration
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        
        # Pretraining strategy
        pretraining_strategy: str = "MLM",  # "AR", "MLM", "OADM", "SPAN"
        mask_probability: float = 0.15,
        span_mask_probability: float = 0.2,
        max_span_length: int = 10,
        oadm_order_probability: float = 0.5,
        
        # Genomic-specific settings
        sequence_type: str = "DNA",  # "DNA", "RNA", "protein"
        max_sequence_length: int = 1024,
        enforce_sample_length: bool = True,
        
        # Data configuration
        train_data_paths: Optional[List[str]] = None,
        valid_data_paths: Optional[List[str]] = None,
        data_weights: Optional[List[float]] = None,
        
        # Loss configuration
        loss_function: str = "cross_entropy",  # "cross_entropy", "focal", "label_smoothing"
        label_smoothing: float = 0.1,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        
        # Logging and evaluation
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        output_dir: str = "./pretraining_outputs",
        
        # Experiment tracking
        use_wandb: bool = False,
        wandb_project: str = "hyena-glt-pretraining",
        wandb_run_name: Optional[str] = None,
        
        # Hardware optimization
        fp16: bool = False,
        gradient_checkpointing: bool = False,
        dataloader_num_workers: int = 4,
        
        # Early stopping
        early_stopping: bool = False,
        early_stopping_patience: int = 5,
        early_stopping_metric: str = "loss",
    ):
        self.model_config = model_config or HyenaGLTConfig()
        
        # Training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        
        # Pretraining strategy
        self.pretraining_strategy = pretraining_strategy.upper()
        self.mask_probability = mask_probability
        self.span_mask_probability = span_mask_probability
        self.max_span_length = max_span_length
        self.oadm_order_probability = oadm_order_probability
        
        # Genomic settings
        self.sequence_type = sequence_type.upper()
        self.max_sequence_length = max_sequence_length
        self.enforce_sample_length = enforce_sample_length
        
        # Data configuration
        self.train_data_paths = train_data_paths or []
        self.valid_data_paths = valid_data_paths or []
        self.data_weights = data_weights
        
        # Loss configuration
        self.loss_function = loss_function
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Logging
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        
        # Experiment tracking
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        
        # Hardware optimization
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing
        self.dataloader_num_workers = dataloader_num_workers
        
        # Early stopping
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        valid_strategies = ["AR", "MLM", "OADM", "SPAN"]
        if self.pretraining_strategy not in valid_strategies:
            raise ValueError(f"pretraining_strategy must be one of {valid_strategies}")
        
        valid_sequence_types = ["DNA", "RNA", "PROTEIN"]
        if self.sequence_type not in valid_sequence_types:
            raise ValueError(f"sequence_type must be one of {valid_sequence_types}")
        
        valid_loss_functions = ["cross_entropy", "focal", "label_smoothing"]
        if self.loss_function not in valid_loss_functions:
            raise ValueError(f"loss_function must be one of {valid_loss_functions}")
        
        if not self.train_data_paths:
            raise ValueError("train_data_paths cannot be empty")


class GenomicMaskingUtils:
    """Utilities for creating masks for different pretraining strategies."""
    
    @staticmethod
    def get_mlm_masks(
        tokens: torch.Tensor,
        mask_prob: float = 0.15,
        mask_token_id: int = 1,
        vocab_size: int = 8,
        special_token_ids: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masks for Masked Language Modeling (MLM).
        
        Args:
            tokens: Input token tensor [batch_size, seq_len]
            mask_prob: Probability of masking tokens
            mask_token_id: Token ID for mask token
            vocab_size: Size of vocabulary
            special_token_ids: List of special token IDs to avoid masking
        
        Returns:
            Tuple of (masked_tokens, loss_mask)
        """
        special_token_ids = special_token_ids or [0]  # Default: avoid padding
        
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Create probability mask
        prob_mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
        
        # Don't mask special tokens
        for special_id in special_token_ids:
            prob_mask = prob_mask & (tokens != special_id)
        
        # Create loss mask (where we calculate loss)
        loss_mask = prob_mask.float()
        
        # Create masked tokens
        masked_tokens = tokens.clone()
        
        # 80% of the time, replace with mask token
        mask_indices = prob_mask & (torch.rand_like(prob_mask.float()) < 0.8)
        masked_tokens[mask_indices] = mask_token_id
        
        # 10% of the time, replace with random token
        random_indices = prob_mask & ~mask_indices & (torch.rand_like(prob_mask.float()) < 0.5)
        random_tokens = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
        masked_tokens[random_indices] = random_tokens[random_indices]
        
        # 10% of the time, keep original token (no change needed)
        
        return masked_tokens, loss_mask
    
    @staticmethod
    def get_span_masks(
        tokens: torch.Tensor,
        span_prob: float = 0.2,
        max_span_length: int = 10,
        mask_token_id: int = 1,
        special_token_ids: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create span masks for span-based pretraining.
        
        Args:
            tokens: Input token tensor [batch_size, seq_len]
            span_prob: Probability of starting a span mask
            max_span_length: Maximum length of spans
            mask_token_id: Token ID for mask token
            special_token_ids: List of special token IDs to avoid masking
        
        Returns:
            Tuple of (masked_tokens, loss_mask)
        """
        special_token_ids = special_token_ids or [0]
        
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        masked_tokens = tokens.clone()
        loss_mask = torch.zeros_like(tokens, dtype=torch.float)
        
        for b in range(batch_size):
            i = 0
            while i < seq_len:
                # Skip special tokens
                if tokens[b, i].item() in special_token_ids:
                    i += 1
                    continue
                
                # Decide whether to start a span
                if torch.rand(1).item() < span_prob:
                    # Determine span length
                    span_length = torch.randint(1, max_span_length + 1, (1,)).item()
                    span_end = min(i + span_length, seq_len)
                    
                    # Mask the span
                    masked_tokens[b, i:span_end] = mask_token_id
                    loss_mask[b, i:span_end] = 1.0
                    
                    i = span_end
                else:
                    i += 1
        
        return masked_tokens, loss_mask
    
    @staticmethod
    def get_oadm_masks(
        tokens: torch.Tensor,
        order_prob: float = 0.5,
        mask_token_id: int = 1,
        special_token_ids: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masks for Order-Agnostic Autoregressive Diffusion Modeling (OADM).
        
        Args:
            tokens: Input token tensor [batch_size, seq_len]
            order_prob: Probability of including token in random order
            mask_token_id: Token ID for mask token
            special_token_ids: List of special token IDs to avoid masking
        
        Returns:
            Tuple of (masked_tokens, loss_mask, position_ids)
        """
        special_token_ids = special_token_ids or [0]
        
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        masked_tokens = tokens.clone()
        loss_mask = torch.zeros_like(tokens, dtype=torch.float)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        for b in range(batch_size):
            # Get non-special token positions
            valid_positions = []
            for i in range(seq_len):
                if tokens[b, i].item() not in special_token_ids:
                    valid_positions.append(i)
            
            if not valid_positions:
                continue
            
            # Randomly shuffle valid positions
            shuffled_positions = torch.randperm(len(valid_positions))
            
            # Determine how many tokens to predict
            num_predict = int(len(valid_positions) * order_prob)
            if num_predict == 0:
                num_predict = 1
            
            # Select positions to predict
            predict_indices = shuffled_positions[:num_predict]
            predict_positions = [valid_positions[i] for i in predict_indices]
            
            # Mask selected positions
            for pos in predict_positions:
                masked_tokens[b, pos] = mask_token_id
                loss_mask[b, pos] = 1.0
            
            # Update position IDs to reflect prediction order
            for order_idx, pos in enumerate(predict_positions):
                position_ids[b, pos] = order_idx
        
        return masked_tokens, loss_mask, position_ids


class GenomicLossFunctions:
    """Loss functions adapted for genomic pretraining."""
    
    @staticmethod
    def cross_entropy_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with optional masking and label smoothing.
        
        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            loss_mask: Mask indicating where to compute loss [batch_size, seq_len]
            label_smoothing: Label smoothing factor
        
        Returns:
            Scalar loss tensor
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        if label_smoothing > 0:
            # Apply label smoothing
            confidence = 1.0 - label_smoothing
            smooth_positives = confidence
            smooth_negatives = label_smoothing / (vocab_size - 1)
            
            # Create smooth targets
            targets_smooth = torch.zeros_like(logits_flat)
            targets_smooth.fill_(smooth_negatives)
            targets_smooth.scatter_(1, targets_flat.unsqueeze(1), smooth_positives)
            
            # Compute loss
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = -torch.sum(targets_smooth * log_probs, dim=-1)
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Apply mask if provided
        if loss_mask is not None:
            mask_flat = loss_mask.view(-1)
            loss = loss * mask_flat
            loss = loss.sum() / (mask_flat.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    @staticmethod
    def focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.
        
        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            loss_mask: Mask indicating where to compute loss [batch_size, seq_len]
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        
        Returns:
            Scalar loss tensor
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Compute probabilities
        probs = F.softmax(logits_flat, dim=-1)
        
        # Get probabilities for target classes
        targets_one_hot = F.one_hot(targets_flat, num_classes=vocab_size).float()
        p_t = torch.sum(probs * targets_one_hot, dim=-1)
        
        # Compute focal loss
        focal_weight = alpha * (1 - p_t) ** gamma
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = focal_weight * ce_loss
        
        # Apply mask if provided
        if loss_mask is not None:
            mask_flat = loss_mask.view(-1)
            loss = loss * mask_flat
            loss = loss.sum() / (mask_flat.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss


class HyenaGLTPretrainer:
    """Main pretrainer class for Hyena-GLT models."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup output directory
        os.makedirs(self.config.logging.output_dir, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize tokenizer
        self.tokenizer = self._create_tokenizer()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize datasets and dataloaders
        self.train_dataloader, self.valid_dataloader = self._create_dataloaders()
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._create_optimizer_and_scheduler()
        
        # Initialize metrics and checkpoint manager
        self.metrics = GenomicMetrics()
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.config.logging.output_dir,
            save_total_limit=5
        )
        
        # Initialize experiment tracking
        if self.config.logging.use_wandb:
            self._init_wandb()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        self.early_stopping_counter = 0
        
        logger.info(f"HyenaGLTPretrainer initialized")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training sequences (estimated): {sum(self.train_dataloader.dataset.file_num_sequences):,}")
        if self.valid_dataloader:
            logger.info(f"Validation sequences (estimated): {sum(self.valid_dataloader.dataset.file_num_sequences):,}")
        else:
            logger.info(f"Validation sequences: 0")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.logging.output_dir, 'pretraining.log')),
                logging.StreamHandler()
            ]
        )
    
    def _create_tokenizer(self):
        """Create appropriate tokenizer based on sequence type."""
        sequence_type = self.config.data.sequence_type.upper()
        if sequence_type == "DNA":
            return DNATokenizer()
        elif sequence_type == "RNA":
            return RNATokenizer()
        elif sequence_type == "PROTEIN":
            return ProteinTokenizer()
        else:
            raise ValueError(f"Unsupported sequence type: {self.config.data.sequence_type}")
    
    def _create_model(self):
        """Create and configure the model."""
        # Update model config with tokenizer info
        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.config.model.max_position_embeddings = self.config.data.max_sequence_length
        
        model = HyenaGLT(self.config.model)
        model = model.to(self.device)
        
        if self.config.hardware.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model
    
    def _create_dataloaders(self):
        """Create training and validation dataloaders."""
        # Import the pretraining dataset and dataloader classes
        from .data_utils import GenomicPretrainingDataset, GenomicDataLoader
        
        # Create training dataset
        train_dataset = GenomicPretrainingDataset(
            data_paths=self.config.data.data_paths,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_sequence_length,
            sequence_type=self.config.data.sequence_type
        )
        
        train_dataloader = GenomicDataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=getattr(self.config.data, 'num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        # Create validation dataset if paths provided
        valid_dataloader = None
        if getattr(self.config.data, 'validation_data_paths', None):
            valid_dataset = GenomicPretrainingDataset(
                data_paths=self.config.data.validation_data_paths,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_sequence_length,
                sequence_type=self.config.data.sequence_type
            )
            
            valid_dataloader = GenomicDataLoader(
                dataset=valid_dataset,
                batch_size=self.config.batch_size,
                num_workers=getattr(self.config.data, 'num_workers', 4),
                pin_memory=True,
                drop_last=False
            )
        
        return train_dataloader, valid_dataloader
    
    def _create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler."""
        optimizer = create_optimizer(
            model=self.model,
            optimizer_type="adamw",
            learning_rate=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay
        )
        
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type="cosine",
            num_warmup_steps=self.config.optimization.warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            wandb.init(
                project=self.config.logging.wandb_project,
                name=getattr(self.config.logging, 'wandb_run_name', None),
                config=self.config.__dict__
            )
        except ImportError:
            logger.warning("wandb not installed, skipping experiment tracking")
            self.config.logging.use_wandb = False
    
    def _get_batch_for_strategy(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare batch based on pretraining strategy.
        
        Returns:
            Tuple of (input_tokens, target_tokens, loss_mask, position_ids)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        if self.config.strategy.strategy == "AR":
            # Autoregressive: predict next token
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            loss_mask = attention_mask[:, 1:].float()
            position_ids = None
            
        elif self.config.strategy.strategy == "MLM":
            # Masked Language Modeling
            input_tokens, loss_mask = GenomicMaskingUtils.get_mlm_masks(
                tokens=tokens,
                mask_prob=self.config.strategy.mask_probability,
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=self.tokenizer.vocab_size
            )
            target_tokens = tokens
            position_ids = None
            
        elif self.config.strategy.strategy == "SPAN":
            # Span masking
            input_tokens, loss_mask = GenomicMaskingUtils.get_span_masks(
                tokens=tokens,
                span_prob=getattr(self.config.strategy, 'span_mask_probability', 0.2),
                max_span_length=getattr(self.config.strategy, 'max_span_length', 10),
                mask_token_id=self.tokenizer.mask_token_id
            )
            target_tokens = tokens
            position_ids = None
            
        elif self.config.strategy.strategy == "OADM":
            # Order-Agnostic Autoregressive Diffusion
            input_tokens, loss_mask, position_ids = GenomicMaskingUtils.get_oadm_masks(
                tokens=tokens,
                order_prob=getattr(self.config.strategy, 'oadm_order_probability', 0.5),
                mask_token_id=self.tokenizer.mask_token_id
            )
            target_tokens = tokens
            
        else:
            raise ValueError(f"Unknown pretraining strategy: {self.config.strategy.strategy}")
        
        return input_tokens, target_tokens, loss_mask, position_ids
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss based on configured loss function."""
        if self.config.loss.loss_function == "cross_entropy":
            return GenomicLossFunctions.cross_entropy_loss(
                logits=logits,
                targets=targets,
                loss_mask=loss_mask,
                label_smoothing=self.config.loss.label_smoothing
            )
        elif self.config.loss.loss_function == "focal":
            return GenomicLossFunctions.focal_loss(
                logits=logits,
                targets=targets,
                loss_mask=loss_mask,
                alpha=getattr(self.config.loss, 'focal_alpha', 1.0),
                gamma=getattr(self.config.loss, 'focal_gamma', 2.0)
            )
        elif self.config.loss.loss_function == "label_smoothing":
            return GenomicLossFunctions.cross_entropy_loss(
                logits=logits,
                targets=targets,
                loss_mask=loss_mask,
                label_smoothing=self.config.loss.label_smoothing
            )
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss.loss_function}")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute a single training step."""
        # Move batch to device
        tokens = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Prepare batch for strategy
        input_tokens, target_tokens, loss_mask, position_ids = self._get_batch_for_strategy(
            tokens, attention_mask
        )
        
        # Forward pass
        kwargs = {"attention_mask": attention_mask[:, :input_tokens.size(1)]}
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        
        outputs = self.model(input_tokens, **kwargs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Compute loss
        loss = self._compute_loss(logits, target_tokens, loss_mask)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.hardware.gradient_accumulation_steps
        
        # Compute metrics
        metrics = {}
        if loss_mask is not None:
            # Accuracy on masked tokens only
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == target_tokens) & (loss_mask > 0)
            metrics["accuracy"] = correct.sum().float() / (loss_mask.sum() + 1e-8)
        else:
            # Standard accuracy
            predictions = torch.argmax(logits, dim=-1)
            metrics["accuracy"] = (predictions == target_tokens).float().mean()
        
        metrics["perplexity"] = torch.exp(loss * self.config.hardware.gradient_accumulation_steps)
        
        return loss, metrics
    
    def _validation_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute a single validation step."""
        with torch.no_grad():
            return self._training_step(batch)
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting pretraining...")
        
        self.model.train()
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        progress_bar = tqdm(total=total_steps, desc="Pretraining")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_metrics = {"accuracy": 0.0, "perplexity": 0.0}
            
            for step, batch in enumerate(self.train_dataloader):
                # Training step
                loss, metrics = self._training_step(batch)
                
                # Backward pass
                if self.config.hardware.fp16:
                    from torch.cuda.amp import autocast, GradScaler
                    if not hasattr(self, 'scaler'):
                        self.scaler = GradScaler()
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Accumulate metrics
                epoch_loss += loss.item()
                for key, value in metrics.items():
                    epoch_metrics[key] += value.item() if isinstance(value, torch.Tensor) else value
                
                # Gradient accumulation and optimization
                if (step + 1) % self.config.hardware.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.optimization.max_grad_norm > 0:
                        if self.config.hardware.fp16:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.optimization.max_grad_norm
                        )
                    
                    # Optimizer step
                    if self.config.hardware.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging.log_steps == 0:
                    self._log_metrics({
                        "train_loss": loss.item() * self.config.hardware.gradient_accumulation_steps,
                        "train_accuracy": metrics["accuracy"],
                        "train_perplexity": metrics["perplexity"],
                        "learning_rate": self.scheduler.get_last_lr()[0] if self.scheduler else self.config.optimization.learning_rate
                    })
                
                # Evaluation
                if (self.global_step % self.config.logging.eval_steps == 0 and 
                    self.valid_dataloader is not None):
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, prefix="eval")
                    
                    # Early stopping check
                    if self.config.logging.early_stopping:
                        self._check_early_stopping(eval_metrics)
                    
                    self.model.train()
                
                # Checkpointing
                if self.global_step % self.config.logging.save_steps == 0:
                    self._save_checkpoint()
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{metrics['accuracy']:.4f}"
                })
                
                # Early stopping
                if (self.config.logging.early_stopping and 
                    self.early_stopping_counter >= self.config.logging.early_stopping_patience):
                    logger.info("Early stopping triggered")
                    break
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            avg_epoch_metrics = {k: v / len(self.train_dataloader) for k, v in epoch_metrics.items()}
            
            logger.info(
                f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}, "
                f"Average accuracy: {avg_epoch_metrics['accuracy']:.4f}"
            )
            
            if (self.config.logging.early_stopping and 
                self.early_stopping_counter >= self.config.logging.early_stopping_patience):
                break
        
        progress_bar.close()
        
        # Final evaluation
        if self.valid_dataloader:
            final_metrics = self.evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
        
        # Save final checkpoint
        self._save_checkpoint()
        
        logger.info("Pretraining completed!")
        
        return {
            "final_step": self.global_step,
            "final_epoch": self.epoch,
            "best_metric": self.best_metric,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set."""
        if self.valid_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_metrics = {"accuracy": 0.0, "perplexity": 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, desc="Evaluating"):
                loss, metrics = self._validation_step(batch)
                
                total_loss += loss.item() * self.config.hardware.gradient_accumulation_steps
                for key, value in metrics.items():
                    total_metrics[key] += value.item() if isinstance(value, torch.Tensor) else value
                num_batches += 1
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {f"eval_{k}": v / num_batches for k, v in total_metrics.items()}
        avg_metrics["eval_loss"] = avg_loss
        
        return avg_metrics
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics to console and wandb."""
        if prefix:
            metrics = {f"{prefix}_{k}" if not k.startswith(prefix) else k: v for k, v in metrics.items()}
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {self.global_step} - {metrics_str}")
        
        # Log to wandb
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=self.global_step)
            except ImportError:
                pass
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.__dict__,
            "best_metric": self.best_metric,
        }
        
        self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            step=self.global_step,
            is_best=(self.best_metric == float('inf'))  # First checkpoint is best by default
        )
        
        logger.info(f"Checkpoint saved at step {self.global_step}")
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> None:
        """Check early stopping criteria."""
        metric_value = metrics.get(f"eval_{self.config.early_stopping_metric}", float('inf'))
        
        if metric_value < self.best_metric:
            self.best_metric = metric_value
            self.early_stopping_counter = 0
            # Save best checkpoint
            checkpoint_data = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "config": self.config.__dict__,
                "best_metric": self.best_metric,
            }
            self.checkpoint_manager.save_checkpoint(checkpoint_data, step=self.global_step, is_best=True)
        else:
            self.early_stopping_counter += 1
        
        logger.info(
            f"Early stopping: {self.early_stopping_counter}/{self.config.early_stopping_patience}, "
            f"Best {self.config.early_stopping_metric}: {self.best_metric:.4f}"
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_metric = checkpoint.get("best_metric", float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


def main():
    """Example usage of the pretrainer."""
    # Configuration
    config = PretrainingConfig(
        # Model configuration
        model_config=HyenaGLTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=1024,
        ),
        
        # Training configuration
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        
        # Pretraining strategy
        pretraining_strategy="MLM",
        mask_probability=0.15,
        
        # Genomic settings
        sequence_type="DNA",
        max_sequence_length=1024,
        
        # Data paths (example)
        train_data_paths=[
            "/path/to/genomic/train/data1.txt",
            "/path/to/genomic/train/data2.txt"
        ],
        valid_data_paths=[
            "/path/to/genomic/valid/data.txt"
        ],
        
        # Output
        output_dir="./hyena_glt_pretraining_outputs",
        
        # Experiment tracking
        use_wandb=True,
        wandb_project="hyena-glt-pretraining",
        wandb_run_name="dna-mlm-experiment",
    )
    
    # Create pretrainer and start training
    pretrainer = HyenaGLTPretrainer(config)
    results = pretrainer.train()
    
    print(f"Training completed: {results}")


if __name__ == "__main__":
    main()
