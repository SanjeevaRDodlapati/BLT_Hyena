"""
Basic training configuration for Hyena-GLT models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Basic training configuration for testing and simple training scenarios."""
    
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 3
    eval_steps: Optional[int] = 100
    save_steps: Optional[int] = 500
    logging_steps: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = 1
    eval_batch_size: Optional[int] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
