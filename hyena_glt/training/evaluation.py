"""
Utilities for evaluating pretrained Hyena-GLT models.

This module provides functions for evaluating model performance during
and after pretraining, including perplexity calculation, downstream
task evaluation, and genomic-specific metrics.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.hyena_glt import HyenaGLTModel
from ..data.tokenizer import DNATokenizer, RNATokenizer, ProteinTokenizer


logger = logging.getLogger(__name__)


class PretrainingEvaluator:
    """Evaluator for pretraining metrics and downstream tasks."""
    
    def __init__(
        self,
        model: HyenaGLTModel,
        tokenizer: Union[DNATokenizer, RNATokenizer, ProteinTokenizer],
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate_perplexity(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> float:
        """
        Calculate perplexity on validation data.
        
        Args:
            dataloader: DataLoader for validation data
            max_batches: Maximum number of batches to evaluate (for speed)
            
        Returns:
            Perplexity score
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating perplexity")):
                if max_batches and i >= max_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # For language modeling
                )
                
                loss = outputs.loss
                
                # Calculate number of tokens (excluding padding)
                if attention_mask is not None:
                    num_tokens = attention_mask.sum().item()
                else:
                    num_tokens = input_ids.numel()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def evaluate_masked_language_modeling(
        self,
        dataloader: DataLoader,
        mask_probability: float = 0.15,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate masked language modeling performance.
        
        Args:
            dataloader: DataLoader for validation data
            mask_probability: Probability of masking tokens
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary with MLM metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        
        mask_token_id = self.tokenizer.mask_token_id
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating MLM")):
                if max_batches and i >= max_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Create random masks
                masked_input_ids, labels = self._create_mlm_masks(
                    input_ids, mask_probability, attention_mask
                )
                
                # Forward pass
                outputs = self.model(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Calculate accuracy on masked tokens
                mask_positions = (labels != -100)
                if mask_positions.any():
                    masked_logits = logits[mask_positions]
                    masked_labels = labels[mask_positions]
                    
                    predictions = masked_logits.argmax(dim=-1)
                    correct = (predictions == masked_labels).sum().item()
                    
                    total_loss += loss.item()
                    total_correct += correct
                    total_masked += mask_positions.sum().item()
        
        if total_masked == 0:
            return {"mlm_loss": float('inf'), "mlm_accuracy": 0.0}
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_masked
        
        return {
            "mlm_loss": avg_loss,
            "mlm_accuracy": accuracy,
            "mlm_perplexity": math.exp(avg_loss)
        }
    
    def _create_mlm_masks(
        self,
        input_ids: torch.Tensor,
        mask_probability: float,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create MLM masks for evaluation."""
        batch_size, seq_len = input_ids.shape
        
        # Create labels (copy of input_ids)
        labels = input_ids.clone()
        
        # Determine which tokens to mask
        probability_matrix = torch.full(labels.shape, mask_probability)
        if attention_mask is not None:
            probability_matrix.masked_fill_(~attention_mask.bool(), 0.0)
        
        # Don't mask special tokens
        special_tokens_mask = (
            (input_ids == self.tokenizer.cls_token_id) |
            (input_ids == self.tokenizer.sep_token_id) |
            (input_ids == self.tokenizer.pad_token_id) |
            (input_ids == self.tokenizer.unk_token_id)
        )
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # Create masked input
        masked_input_ids = input_ids.clone()
        
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        masked_input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        masked_input_ids[indices_random] = random_words[indices_random]
        
        # 10% of the time, keep original
        
        return masked_input_ids, labels
    
    def evaluate_nucleotide_prediction_accuracy(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate per-nucleotide prediction accuracy.
        
        Args:
            dataloader: DataLoader for validation data
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary with nucleotide-specific accuracies
        """
        self.model.eval()
        
        # Track predictions per nucleotide
        nucleotide_correct = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        nucleotide_total = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        total_correct = 0
        total_predictions = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating nucleotide accuracy")):
                if max_batches and i >= max_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Shift for next token prediction
                input_ids_shifted = input_ids[:, :-1]
                labels = input_ids[:, 1:]
                
                if attention_mask is not None:
                    attention_mask_shifted = attention_mask[:, :-1]
                    label_mask = attention_mask[:, 1:]
                else:
                    attention_mask_shifted = None
                    label_mask = torch.ones_like(labels)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids_shifted,
                    attention_mask=attention_mask_shifted
                )
                
                logits = outputs.logits
                predictions = logits.argmax(dim=-1)
                
                # Calculate accuracies
                valid_positions = label_mask.bool()
                valid_predictions = predictions[valid_positions]
                valid_labels = labels[valid_positions]
                
                correct = (valid_predictions == valid_labels)
                total_correct += correct.sum().item()
                total_predictions += valid_predictions.numel()
                
                # Per-nucleotide accuracy
                for nuc, token_id in [('A', 0), ('T', 1), ('G', 2), ('C', 3)]:
                    nuc_positions = (valid_labels == token_id)
                    if nuc_positions.any():
                        nuc_correct = correct[nuc_positions].sum().item()
                        nuc_total = nuc_positions.sum().item()
                        
                        nucleotide_correct[nuc] += nuc_correct
                        nucleotide_total[nuc] += nuc_total
        
        # Calculate final accuracies
        results = {
            "overall_accuracy": total_correct / total_predictions if total_predictions > 0 else 0.0
        }
        
        for nuc in ['A', 'T', 'G', 'C']:
            if nucleotide_total[nuc] > 0:
                results[f"{nuc}_accuracy"] = nucleotide_correct[nuc] / nucleotide_total[nuc]
            else:
                results[f"{nuc}_accuracy"] = 0.0
        
        return results
    
    def evaluate_gc_content_preservation(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate how well the model preserves GC content in generated sequences.
        
        Args:
            dataloader: DataLoader for validation data
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary with GC content metrics
        """
        self.model.eval()
        
        original_gc_contents = []
        predicted_gc_contents = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating GC content")):
                if max_batches and i >= max_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Generate predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = outputs.logits.argmax(dim=-1)
                
                # Calculate GC content for original and predicted sequences
                for j in range(input_ids.size(0)):
                    if attention_mask is not None:
                        seq_len = attention_mask[j].sum().item()
                        orig_seq = input_ids[j, :seq_len]
                        pred_seq = predictions[j, :seq_len]
                    else:
                        orig_seq = input_ids[j]
                        pred_seq = predictions[j]
                    
                    orig_gc = self._calculate_gc_content(orig_seq)
                    pred_gc = self._calculate_gc_content(pred_seq)
                    
                    original_gc_contents.append(orig_gc)
                    predicted_gc_contents.append(pred_gc)
        
        if not original_gc_contents:
            return {"gc_content_mae": 0.0, "gc_content_correlation": 0.0}
        
        # Calculate metrics
        original_gc = torch.tensor(original_gc_contents)
        predicted_gc = torch.tensor(predicted_gc_contents)
        
        mae = torch.abs(original_gc - predicted_gc).mean().item()
        correlation = torch.corrcoef(torch.stack([original_gc, predicted_gc]))[0, 1].item()
        
        return {
            "gc_content_mae": mae,
            "gc_content_correlation": correlation,
            "mean_original_gc": original_gc.mean().item(),
            "mean_predicted_gc": predicted_gc.mean().item()
        }
    
    def _calculate_gc_content(self, sequence: torch.Tensor) -> float:
        """Calculate GC content of a sequence."""
        # Assuming token IDs: A=0, T=1, G=2, C=3
        g_count = (sequence == 2).sum().item()
        c_count = (sequence == 3).sum().item()
        total_count = sequence.numel()
        
        if total_count == 0:
            return 0.0
        
        return (g_count + c_count) / total_count


def run_comprehensive_evaluation(
    model: HyenaGLTModel,
    tokenizer: Union[DNATokenizer, RNATokenizer, ProteinTokenizer],
    val_dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Run comprehensive evaluation of a pretrained model.
    
    Args:
        model: Pretrained HyenaGLT model
        tokenizer: Tokenizer used for the model
        val_dataloader: Validation data loader
        device: Device to run evaluation on
        max_batches: Maximum number of batches for each evaluation
        
    Returns:
        Dictionary with all evaluation metrics
    """
    evaluator = PretrainingEvaluator(model, tokenizer, device)
    
    results = {}
    
    # Perplexity
    logger.info("Calculating perplexity...")
    perplexity = evaluator.evaluate_perplexity(val_dataloader, max_batches)
    results["perplexity"] = perplexity
    
    # MLM performance
    logger.info("Evaluating MLM performance...")
    mlm_results = evaluator.evaluate_masked_language_modeling(val_dataloader, max_batches=max_batches)
    results.update(mlm_results)
    
    # Nucleotide prediction accuracy
    logger.info("Evaluating nucleotide prediction accuracy...")
    nuc_results = evaluator.evaluate_nucleotide_prediction_accuracy(val_dataloader, max_batches)
    results.update(nuc_results)
    
    # GC content preservation
    logger.info("Evaluating GC content preservation...")
    gc_results = evaluator.evaluate_gc_content_preservation(val_dataloader, max_batches)
    results.update(gc_results)
    
    return results
