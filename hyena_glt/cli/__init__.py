"""Command-line interface for Hyena-GLT."""

from .train import main as train_main
from .eval import main as eval_main
from .preprocess import main as preprocess_main

__all__ = ["train_main", "eval_main", "preprocess_main"]
