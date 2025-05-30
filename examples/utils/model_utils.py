"""
Model utilities for training, evaluation, and analysis of Hyena-GLT models.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import time
import warnings
from collections import defaultdict

try:
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report, roc_auc_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available. Some functionality will be limited.")


def quick_train_model(
    sequences: List[str],
    labels: List[int],
    sequence_type: str = "dna",
    task_type: str = "classification",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    validation_split: float = 0.2,
    device: str = "auto",
    verbose: bool = True,
    save_path: Optional[str] = None
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Quickly train a Hyena-GLT model with minimal setup.
    
    Args:
        sequences: List of genomic sequences
        labels: List of corresponding labels
        sequence_type: Type of sequence ('dna', 'rna', 'protein')
        task_type: Type of task ('classification', 'regression')
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        validation_split: Fraction of data for validation
        device: Device to use ('auto', 'cuda', 'cpu')
        verbose: Whether to print progress
        save_path: Path to save the trained model
        
    Returns:
        Tuple of (model, tokenizer, training_history)
    """
    # Import required modules (assumed to be available)
    try:
        from hyena_glt.config import HyenaGLTConfig
        from hyena_glt.model import HyenaGLT
        from hyena_glt.data import DNATokenizer, RNATokenizer, ProteinTokenizer, GenomicDataset
        from hyena_glt.training import HyenaGLTTrainer, TrainingConfig
    except ImportError as e:
        raise ImportError(f"Required Hyena-GLT modules not available: {e}")
    
    if verbose:
        print(f"🚀 Quick training setup for {sequence_type} {task_type}")
        print(f"📊 Data: {len(sequences)} sequences, {len(set(labels))} classes")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize tokenizer
    tokenizers = {
        'dna': DNATokenizer,
        'rna': RNATokenizer,
        'protein': ProteinTokenizer
    }
    
    if sequence_type not in tokenizers:
        raise ValueError(f"Unsupported sequence type: {sequence_type}")
    
    tokenizer = tokenizers[sequence_type]()
    
    # Create model configuration
    num_classes = len(set(labels))
    
    if sequence_type == 'dna':
        config = HyenaGLTConfig.for_dna_classification(
            num_classes=num_classes,
            max_length=min(1024, max(len(seq) for seq in sequences) + 50)
        )
    elif sequence_type == 'rna':
        config = HyenaGLTConfig.for_rna_structure(
            max_length=min(512, max(len(seq) for seq in sequences) + 50)
        )
    else:  # protein
        config = HyenaGLTConfig.for_protein_function(
            num_functions=num_classes,
            max_length=min(1024, max(len(seq) for seq in sequences) + 50)
        )
    
    # Create dataset
    dataset = GenomicDataset(
        sequences=sequences,
        labels=labels,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    # Split dataset
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = HyenaGLT(config)
    model.to(device)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧠 Model: {total_params:,} parameters")
        print(f"💾 Device: {device}")
    
    # Configure training
    training_config = TrainingConfig(
        num_epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=min(100, len(train_loader) // 2),
        save_steps=len(train_loader),
        eval_steps=len(train_loader),
        logging_steps=max(1, len(train_loader) // 10)
    )
    
    # Initialize trainer
    output_dir = save_path or "./quick_train_output"
    trainer = HyenaGLTTrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        eval_loader=val_loader,
        output_dir=output_dir
    )
    
    # Train model
    if verbose:
        print(f"🏃 Training for {epochs} epochs...")
    
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time
    
    if verbose:
        print(f"✅ Training completed in {training_time:.2f} seconds")
        if 'eval_accuracy' in history and history['eval_accuracy']:
            print(f"📈 Final validation accuracy: {history['eval_accuracy'][-1]:.4f}")
    
    # Save model if requested
    if save_path:
        save_model_with_metadata(
            model, tokenizer, config, history,
            save_path, sequence_type, task_type
        )
        if verbose:
            print(f"💾 Model saved to: {save_path}")
    
    return model, tokenizer, history


def evaluate_model_comprehensive(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: Optional[List[str]] = None,
    device: str = "auto",
    return_predictions: bool = True
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of a model.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        class_names: Optional class names
        device: Device to use
        return_predictions: Whether to return predictions and probabilities
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    if not HAS_SKLEARN:
        raise ImportError("Scikit-learn is required for comprehensive evaluation")
    
    # Determine device
    if device == "auto":
        device = next(model.parameters()).device
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                true_labels = batch['labels'].to(device)
            else:
                input_ids, true_labels = batch
                input_ids = input_ids.to(device)
                true_labels = true_labels.to(device)
                attention_mask = None
            
            # Forward pass
            if attention_mask is not None:
                outputs = model(input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids)
            
            # Get predictions
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels, all_predictions, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_true_labels, all_predictions, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Classification report
    report = classification_report(
        all_true_labels, all_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    # Prediction confidence statistics
    max_probs = np.max(all_probabilities, axis=1)
    confidence_stats = {
        'mean': np.mean(max_probs),
        'std': np.std(max_probs),
        'min': np.min(max_probs),
        'max': np.max(max_probs),
        'median': np.median(max_probs)
    }
    
    # Prepare results
    results = {
        'overall_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(np.sum(support))
        },
        'per_class_metrics': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist(),
            'support': support_per_class.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'confidence_statistics': confidence_stats
    }
    
    # Add predictions if requested
    if return_predictions:
        results['predictions'] = {
            'predicted_labels': all_predictions.tolist(),
            'true_labels': all_true_labels.tolist(),
            'probabilities': all_probabilities.tolist()
        }
    
    # Calculate ROC AUC for multiclass if possible
    try:
        if len(np.unique(all_true_labels)) > 2:
            auc_scores = []
            for i in range(all_probabilities.shape[1]):
                binary_true = (all_true_labels == i).astype(int)
                auc = roc_auc_score(binary_true, all_probabilities[:, i])
                auc_scores.append(auc)
            results['roc_auc_per_class'] = auc_scores
            results['overall_metrics']['macro_auc'] = np.mean(auc_scores)
        else:
            auc = roc_auc_score(all_true_labels, all_probabilities[:, 1])
            results['overall_metrics']['roc_auc'] = auc
    except Exception as e:
        # ROC AUC calculation failed, skip it
        pass
    
    return results


def analyze_model_predictions(
    model: nn.Module,
    sequences: List[str],
    tokenizer: Any,
    class_names: List[str],
    device: str = "auto",
    return_attention: bool = True,
    max_sequences: int = 100
) -> List[Dict[str, Any]]:
    """
    Analyze model predictions for individual sequences.
    
    Args:
        model: Trained model
        sequences: List of sequences to analyze
        tokenizer: Tokenizer for the sequences
        class_names: List of class names
        device: Device to use
        return_attention: Whether to return attention weights
        max_sequences: Maximum number of sequences to analyze
        
    Returns:
        List of analysis results for each sequence
    """
    if device == "auto":
        device = next(model.parameters()).device
    
    model.eval()
    results = []
    
    # Limit number of sequences for performance
    sequences = sequences[:max_sequences]
    
    for i, seq in enumerate(sequences):
        try:
            # Tokenize sequence
            tokens = tokenizer.encode(seq)
            input_ids = torch.tensor([tokens]).to(device)
            
            # Get model output
            with torch.no_grad():
                if return_attention:
                    outputs = model(input_ids, output_attentions=True)
                    attention_weights = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()
                else:
                    outputs = model(input_ids)
                    attention_weights = None
            
            # Get predictions
            if hasattr(outputs, 'logits'):
                logits = outputs.logits[0]
            else:
                logits = outputs[0]
            
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            predicted_class = torch.argmax(logits).item()
            
            # Calculate sequence features
            gc_content = (seq.upper().count('G') + seq.upper().count('C')) / len(seq)
            
            # Analyze prediction
            analysis = {
                'sequence_index': i,
                'sequence': seq,
                'sequence_length': len(seq),
                'gc_content': gc_content,
                'predicted_class': predicted_class,
                'predicted_class_name': class_names[predicted_class] if predicted_class < len(class_names) else f'Class {predicted_class}',
                'prediction_confidence': probabilities[predicted_class],
                'all_probabilities': probabilities.tolist(),
                'class_probabilities': {
                    class_names[j] if j < len(class_names) else f'Class {j}': prob
                    for j, prob in enumerate(probabilities)
                }
            }
            
            if attention_weights is not None:
                analysis['attention_weights'] = attention_weights.tolist()
                # Find positions with highest attention
                avg_attention = attention_weights.mean(axis=0)
                top_positions = np.argsort(avg_attention)[-5:]  # Top 5 positions
                analysis['top_attention_positions'] = top_positions.tolist()
            
            results.append(analysis)
            
        except Exception as e:
            print(f"Warning: Failed to analyze sequence {i}: {e}")
            continue
    
    return results


def save_model_with_metadata(
    model: nn.Module,
    tokenizer: Any,
    config: Any,
    training_history: Dict[str, Any],
    save_path: str,
    sequence_type: str,
    task_type: str,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model with comprehensive metadata.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer used
        config: Model configuration
        training_history: Training history
        save_path: Path to save the model
        sequence_type: Type of sequence
        task_type: Type of task
        additional_metadata: Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Prepare metadata
    metadata = {
        'model_info': {
            'model_type': 'hyena-glt',
            'sequence_type': sequence_type,
            'task_type': task_type,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        'config': config.__dict__ if hasattr(config, '__dict__') else str(config),
        'training_info': {
            'final_train_loss': training_history.get('train_loss', [])[-1] if training_history.get('train_loss') else None,
            'final_eval_loss': training_history.get('eval_loss', [])[-1] if training_history.get('eval_loss') else None,
            'final_train_accuracy': training_history.get('train_accuracy', [])[-1] if training_history.get('train_accuracy') else None,
            'final_eval_accuracy': training_history.get('eval_accuracy', [])[-1] if training_history.get('eval_accuracy') else None,
            'total_steps': len(training_history.get('train_loss', [])),
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pytorch_version': torch.__version__
    }
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    # Save metadata
    with open(save_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save full training history
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2, default=str)


def load_model_with_metadata(
    model_path: str,
    device: str = "auto"
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """
    Load model with metadata.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer, metadata)
    """
    try:
        from hyena_glt.model import HyenaGLT
        from hyena_glt.data import DNATokenizer, RNATokenizer, ProteinTokenizer
    except ImportError as e:
        raise ImportError(f"Required Hyena-GLT modules not available: {e}")
    
    model_path = Path(model_path)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load metadata
    metadata_file = model_path / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Determine tokenizer type
    sequence_type = metadata.get('model_info', {}).get('sequence_type', 'dna')
    
    tokenizers = {
        'dna': DNATokenizer,
        'rna': RNATokenizer,
        'protein': ProteinTokenizer
    }
    
    # Load model and tokenizer
    model = HyenaGLT.from_pretrained(model_path, map_location=device)
    tokenizer_class = tokenizers.get(sequence_type, DNATokenizer)
    tokenizer = tokenizer_class.from_pretrained(model_path)
    
    # Load training history if available
    history_file = model_path / 'training_history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            training_history = json.load(f)
        metadata['training_history'] = training_history
    
    return model, tokenizer, metadata


def compute_model_efficiency_metrics(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: str = "auto",
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Compute efficiency metrics for a model.
    
    Args:
        model: Model to evaluate
        sample_input: Sample input tensor
        device: Device to use
        num_runs: Number of runs for timing
        
    Returns:
        Dictionary with efficiency metrics
    """
    if device == "auto":
        device = next(model.parameters()).device
    
    model.eval()
    sample_input = sample_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    
    # Timing
    torch.cuda.synchronize() if device.startswith('cuda') else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input)
    
    torch.cuda.synchronize() if device.startswith('cuda') else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    throughput = sample_input.size(0) / avg_time  # sequences per second
    
    # Memory usage
    if device.startswith('cuda'):
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
    else:
        memory_allocated = None
        memory_reserved = None
    
    # Model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB
    
    return {
        'timing': {
            'average_inference_time': avg_time,
            'throughput_sequences_per_second': throughput,
            'total_time': total_time,
            'num_runs': num_runs
        },
        'memory': {
            'allocated_mb': memory_allocated,
            'reserved_mb': memory_reserved,
            'model_size_mb': param_size
        },
        'model_stats': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
