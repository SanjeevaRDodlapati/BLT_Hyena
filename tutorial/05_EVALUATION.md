# 05 - Model Evaluation

**Estimated Time:** 45 minutes  
**Prerequisites:** [04 - Training](04_TRAINING.md)  
**Next:** [06 - Production](06_PRODUCTION.md)

## Overview

This tutorial covers comprehensive evaluation strategies for BLT_Hyena models across different genomic tasks. You'll learn to implement robust evaluation pipelines, interpret results, and benchmark against existing methods.

## What You'll Learn

- Setting up evaluation environments and metrics
- Task-specific evaluation strategies (classification, generation, variant calling)
- Statistical significance testing and confidence intervals
- Benchmarking against baseline models
- Visualization and interpretation of results
- Error analysis and model debugging techniques

## Evaluation Framework Setup

### Basic Evaluation Configuration

```python
from hyena_glt import HyenaGLT, HyenaGLTConfig
from hyena_glt.evaluation import EvaluationConfig, ModelEvaluator
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationConfig:
    """Configuration for model evaluation"""
    def __init__(self):
        # Data settings
        self.test_data_path = "data/test_sequences.fa"
        self.batch_size = 32
        self.max_seq_length = 4096
        
        # Evaluation settings
        self.metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
        self.compute_confidence_intervals = True
        self.n_bootstrap_samples = 1000
        
        # Output settings
        self.output_dir = "evaluation_results"
        self.save_predictions = True
        self.save_visualizations = True
        
        # Comparison settings
        self.baseline_models = ['random', 'lstm', 'transformer']
        self.significance_level = 0.05

eval_config = EvaluationConfig()
```

### Model Loading and Setup

```python
def load_trained_model(checkpoint_path, device='cuda'):
    """Load a trained BLT_Hyena model from checkpoint"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configurations
    model_config = checkpoint['model_config']
    
    # Create model
    model = HyenaGLT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Training step: {checkpoint['global_step']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, model_config

# Load your trained model
model, model_config = load_trained_model("checkpoints/best_model.pt")
```

## Comprehensive Evaluation Pipeline

### Base Evaluator Class

```python
class GenomicModelEvaluator:
    """Comprehensive evaluator for genomic models"""
    
    def __init__(self, model, tokenizer, config, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Results storage
        self.results = {}
        self.predictions = []
        self.true_labels = []
        
    def evaluate_dataset(self, test_loader, task_type="classification"):
        """Evaluate model on a dataset"""
        
        self.model.eval()
        all_predictions = []
        all_true_labels = []
        all_logits = []
        inference_times = []
        
        print(f"Evaluating {len(test_loader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Time inference
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                end_time.record()
                torch.cuda.synchronize()
                
                # Record timing
                batch_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                inference_times.append(batch_time)
                
                # Extract predictions
                if task_type == "classification":
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_logits.extend(logits.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
                    all_true_labels.extend(labels.cpu().numpy())
                
                elif task_type == "generation":
                    # For generation tasks, evaluate perplexity
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # Store for perplexity calculation
                    all_predictions.extend(losses.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{len(test_loader)} batches")
        
        # Store results
        self.predictions = all_predictions
        self.true_labels = all_true_labels
        self.logits = all_logits if task_type == "classification" else None
        self.inference_times = inference_times
        
        # Compute metrics
        if task_type == "classification":
            metrics = self.compute_classification_metrics()
        elif task_type == "generation":
            metrics = self.compute_generation_metrics()
        
        self.results[task_type] = metrics
        return metrics
    
    def compute_classification_metrics(self):
        """Compute comprehensive classification metrics"""
        
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.logits)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC-AUC and PR-AUC (for binary/multiclass)
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            # Binary classification
            auc_roc = roc_auc_score(y_true, y_prob[:, 1])
            auc_pr = average_precision_score(y_true, y_prob[:, 1])
        else:
            # Multiclass classification
            try:
                auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                auc_pr = average_precision_score(y_true, y_prob, average='weighted')
            except:
                auc_roc = None
                auc_pr = None
        
        # Inference speed
        avg_inference_time = np.mean(self.inference_times)
        throughput = len(y_true) / sum(self.inference_times)  # samples per second
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'support': support.tolist(),
            'n_samples': len(y_true),
            'n_classes': n_classes,
            'avg_inference_time': avg_inference_time,
            'throughput': throughput
        }
        
        return metrics
    
    def compute_generation_metrics(self):
        """Compute generation-specific metrics"""
        
        # Perplexity
        losses = np.array(self.predictions)
        perplexity = np.exp(np.mean(losses))
        
        # Inference speed
        avg_inference_time = np.mean(self.inference_times)
        
        metrics = {
            'perplexity': perplexity,
            'avg_loss': np.mean(losses),
            'avg_inference_time': avg_inference_time,
            'n_samples': len(losses)
        }
        
        return metrics
    
    def compute_confidence_intervals(self, metric_name='accuracy', confidence_level=0.95):
        """Compute bootstrap confidence intervals"""
        
        if not self.predictions or not self.true_labels:
            raise ValueError("No predictions available. Run evaluation first.")
        
        def bootstrap_metric(y_true, y_pred, metric_func):
            """Bootstrap a single metric"""
            n_samples = len(y_true)
            bootstrap_scores = []
            
            for _ in range(self.config.n_bootstrap_samples):
                # Sample with replacement
                indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_true = y_true[indices]
                bootstrap_pred = y_pred[indices]
                
                # Compute metric
                score = metric_func(bootstrap_true, bootstrap_pred)
                bootstrap_scores.append(score)
            
            return np.array(bootstrap_scores)
        
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        
        # Define metric functions
        metric_functions = {
            'accuracy': accuracy_score,
            'f1': lambda y_t, y_p: precision_recall_fscore_support(y_t, y_p, average='weighted')[2],
            'precision': lambda y_t, y_p: precision_recall_fscore_support(y_t, y_p, average='weighted')[0],
            'recall': lambda y_t, y_p: precision_recall_fscore_support(y_t, y_p, average='weighted')[1]
        }
        
        if metric_name not in metric_functions:
            raise ValueError(f"Metric {metric_name} not supported for confidence intervals")
        
        # Bootstrap the metric
        bootstrap_scores = bootstrap_metric(y_true, y_pred, metric_functions[metric_name])
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        
        return {
            'metric': metric_name,
            'mean': np.mean(bootstrap_scores),
            'std': np.std(bootstrap_scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level
        }

# Initialize evaluator
evaluator = GenomicModelEvaluator(model, tokenizer, eval_config)
```

## Task-Specific Evaluation

### Classification Evaluation

```python
class ClassificationEvaluator(GenomicModelEvaluator):
    """Specialized evaluator for genomic classification tasks"""
    
    def __init__(self, model, tokenizer, config, class_names=None):
        super().__init__(model, tokenizer, config)
        self.class_names = class_names or [f"Class_{i}" for i in range(model.config.num_classes)]
    
    def detailed_classification_report(self):
        """Generate detailed classification report"""
        
        if not self.predictions:
            raise ValueError("Run evaluation first")
        
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        
        # Classification report
        from sklearn.metrics import classification_report
        report_dict = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame for better visualization
        report_df = pd.DataFrame(report_dict).transpose()
        
        return report_df
    
    def plot_confusion_matrix(self, normalize=True, figsize=(10, 8)):
        """Plot confusion matrix heatmap"""
        
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if self.config.save_visualizations:
            plt.savefig(f"{self.config.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, figsize=(12, 8)):
        """Plot ROC curves for each class"""
        
        if self.logits is None:
            print("No logits available for ROC curves")
            return
        
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        y_true = np.array(self.true_labels)
        y_score = np.array(self.logits)
        
        # Binarize labels for multiclass ROC
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=figsize)
        
        # Compute ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if self.config.save_visualizations:
            plt.savefig(f"{self.config.output_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_errors(self, top_k=10):
        """Analyze the most common classification errors"""
        
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.logits)
        
        # Find misclassified examples
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        # Get confidence scores for misclassified examples
        misclassified_confidences = []
        for idx in misclassified_indices:
            predicted_class = y_pred[idx]
            confidence = y_prob[idx][predicted_class]
            misclassified_confidences.append(confidence)
        
        # Sort by confidence (high confidence errors are more concerning)
        sorted_indices = np.argsort(misclassified_confidences)[::-1]
        
        # Analyze top errors
        error_analysis = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = misclassified_indices[sorted_indices[i]]
            error_analysis.append({
                'sample_index': idx,
                'true_class': self.class_names[y_true[idx]],
                'predicted_class': self.class_names[y_pred[idx]],
                'confidence': misclassified_confidences[sorted_indices[i]],
                'true_class_prob': y_prob[idx][y_true[idx]]
            })
        
        return pd.DataFrame(error_analysis)

# Example usage for variant calling classification
class_names = ['Normal', 'SNV', 'Indel', 'CNV']
classification_evaluator = ClassificationEvaluator(model, tokenizer, eval_config, class_names)

# Run evaluation
test_metrics = classification_evaluator.evaluate_dataset(test_loader, task_type="classification")
print("Classification Metrics:")
for metric, value in test_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.4f}")

# Generate detailed report
report = classification_evaluator.detailed_classification_report()
print("\nDetailed Classification Report:")
print(report)

# Plot visualizations
classification_evaluator.plot_confusion_matrix()
classification_evaluator.plot_roc_curves()

# Analyze errors
error_analysis = classification_evaluator.analyze_errors()
print("\nTop Classification Errors:")
print(error_analysis)
```

### Generation Evaluation

```python
class GenerationEvaluator(GenomicModelEvaluator):
    """Specialized evaluator for genomic sequence generation"""
    
    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)
    
    def evaluate_generation_quality(self, test_sequences, max_length=200, num_samples=100):
        """Evaluate generation quality with multiple metrics"""
        
        results = {
            'bleu_scores': [],
            'edit_distances': [],
            'gc_contents': [],
            'sequence_diversities': [],
            'biological_validity': []
        }
        
        generated_sequences = []
        
        print(f"Generating {num_samples} sequences for evaluation...")
        
        for i in range(num_samples):
            # Random prompt from test set
            prompt_idx = np.random.randint(0, len(test_sequences))
            prompt_seq = test_sequences[prompt_idx][:50]  # Use first 50 nucleotides as prompt
            
            # Generate sequence
            generated_seq = self.generate_sequence(
                prompt_seq, 
                max_length=max_length,
                temperature=0.8,
                do_sample=True
            )
            
            generated_sequences.append(generated_seq)
            
            # Compute metrics
            if i < len(test_sequences):
                target_seq = test_sequences[i]
                
                # BLEU score (treating nucleotides as tokens)
                bleu = self.compute_sequence_bleu(generated_seq, target_seq)
                results['bleu_scores'].append(bleu)
                
                # Edit distance
                edit_dist = self.compute_edit_distance(generated_seq, target_seq)
                results['edit_distances'].append(edit_dist)
            
            # GC content
            gc_content = self.compute_gc_content(generated_seq)
            results['gc_contents'].append(gc_content)
            
            # Biological validity
            validity_score = self.assess_biological_validity(generated_seq)
            results['biological_validity'].append(validity_score)
        
        # Sequence diversity
        diversity_score = self.compute_sequence_diversity(generated_sequences)
        results['sequence_diversity'] = diversity_score
        
        # Aggregate results
        aggregated_results = {
            'avg_bleu': np.mean(results['bleu_scores']) if results['bleu_scores'] else None,
            'avg_edit_distance': np.mean(results['edit_distances']) if results['edit_distances'] else None,
            'avg_gc_content': np.mean(results['gc_contents']),
            'gc_content_std': np.std(results['gc_contents']),
            'avg_biological_validity': np.mean(results['biological_validity']),
            'sequence_diversity': diversity_score,
            'generated_sequences': generated_sequences[:10]  # Store sample sequences
        }
        
        return aggregated_results
    
    def generate_sequence(self, prompt, max_length=200, temperature=1.0, do_sample=True):
        """Generate a sequence from a prompt"""
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        
        with torch.no_grad():
            # Generate
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode
            generated_sequence = self.tokenizer.decode(outputs[0])
        
        return generated_sequence
    
    def compute_sequence_bleu(self, generated, target):
        """Compute BLEU score for sequences"""
        from nltk.translate.bleu_score import sentence_bleu
        
        # Treat each nucleotide as a token
        generated_tokens = list(generated)
        target_tokens = list(target)
        
        # Compute BLEU with different n-gram sizes
        bleu_score = sentence_bleu(
            [target_tokens],
            generated_tokens,
            weights=(0.25, 0.25, 0.25, 0.25)  # 1-4 gram weights
        )
        
        return bleu_score
    
    def compute_edit_distance(self, seq1, seq2):
        """Compute normalized edit distance between sequences"""
        import editdistance
        
        distance = editdistance.eval(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        
        return distance / max_len if max_len > 0 else 0
    
    def compute_gc_content(self, sequence):
        """Compute GC content of a sequence"""
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        total_count = len(sequence)
        
        return gc_count / total_count if total_count > 0 else 0
    
    def compute_sequence_diversity(self, sequences):
        """Compute diversity score across generated sequences"""
        
        if len(sequences) < 2:
            return 0
        
        # Compute pairwise distances
        distances = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                dist = self.compute_edit_distance(sequences[i], sequences[j])
                distances.append(dist)
        
        return np.mean(distances)
    
    def assess_biological_validity(self, sequence):
        """Assess biological validity of generated sequence"""
        
        sequence = sequence.upper()
        validity_score = 0.0
        
        # Check for valid nucleotides only
        valid_nucleotides = set('ATGC')
        invalid_count = sum(1 for char in sequence if char not in valid_nucleotides)
        nucleotide_validity = 1 - (invalid_count / len(sequence)) if len(sequence) > 0 else 0
        validity_score += nucleotide_validity * 0.4
        
        # Check for reasonable GC content (typical range: 0.3-0.7)
        gc_content = self.compute_gc_content(sequence)
        gc_validity = 1 - abs(gc_content - 0.5) * 2  # Penalize extreme GC content
        gc_validity = max(0, gc_validity)
        validity_score += gc_validity * 0.3
        
        # Check for absence of long repeats (longer than 10)
        max_repeat_length = self.find_max_repeat_length(sequence)
        repeat_penalty = min(max_repeat_length / 20, 1)  # Penalize long repeats
        repeat_validity = 1 - repeat_penalty
        validity_score += repeat_validity * 0.3
        
        return validity_score
    
    def find_max_repeat_length(self, sequence):
        """Find the maximum length of repeated subsequences"""
        max_repeat = 0
        
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence)):
                # Find common prefix
                k = 0
                while (j + k < len(sequence) and 
                       sequence[i + k] == sequence[j + k]):
                    k += 1
                
                if k > max_repeat:
                    max_repeat = k
        
        return max_repeat

# Example usage
generation_evaluator = GenerationEvaluator(model, tokenizer, eval_config)

# Load test sequences for comparison
test_sequences = []  # Load your test genomic sequences here

# Evaluate generation quality
generation_metrics = generation_evaluator.evaluate_generation_quality(
    test_sequences, 
    max_length=200, 
    num_samples=100
)

print("Generation Quality Metrics:")
for metric, value in generation_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.4f}")
```

## Benchmark Comparison

### Baseline Models Implementation

```python
class BaselineComparison:
    """Compare BLT_Hyena against baseline models"""
    
    def __init__(self, test_loader, task_type="classification"):
        self.test_loader = test_loader
        self.task_type = task_type
        self.results = {}
    
    def evaluate_random_baseline(self, num_classes):
        """Evaluate random prediction baseline"""
        
        all_true = []
        all_pred = []
        
        for batch in self.test_loader:
            labels = batch['labels'].numpy()
            batch_size = len(labels)
            
            # Random predictions
            random_preds = np.random.randint(0, num_classes, size=batch_size)
            
            all_true.extend(labels)
            all_pred.extend(random_preds)
        
        # Compute metrics
        accuracy = accuracy_score(all_true, all_pred)
        f1 = precision_recall_fscore_support(all_true, all_pred, average='weighted')[2]
        
        self.results['random'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'model_type': 'random'
        }
        
        return self.results['random']
    
    def evaluate_lstm_baseline(self, vocab_size, num_classes, embed_dim=128, hidden_dim=256):
        """Evaluate LSTM baseline"""
        
        import torch.nn as nn
        from torch.utils.data import DataLoader
        
        class LSTMBaseline(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.classifier = nn.Linear(hidden_dim * 2, num_classes)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                lstm_out, _ = self.lstm(x)
                # Use last hidden state
                pooled = lstm_out[:, -1, :]
                pooled = self.dropout(pooled)
                logits = self.classifier(pooled)
                return logits
        
        # Create and train LSTM model (simplified training)
        lstm_model = LSTMBaseline(vocab_size, embed_dim, hidden_dim, num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lstm_model.to(device)
        
        # Quick training loop (you might want to train properly)
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        lstm_model.train()
        for epoch in range(5):  # Quick training
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                logits = lstm_model(input_ids)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate LSTM
        lstm_model.eval()
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels']
                
                logits = lstm_model(input_ids)
                predictions = torch.argmax(logits, dim=-1)
                
                all_true.extend(labels.numpy())
                all_pred.extend(predictions.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_true, all_pred)
        f1 = precision_recall_fscore_support(all_true, all_pred, average='weighted')[2]
        
        self.results['lstm'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'model_type': 'lstm'
        }
        
        return self.results['lstm']
    
    def compare_models(self, hyena_results):
        """Compare all models"""
        
        # Add Hyena results
        self.results['hyena'] = hyena_results
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'F1 Score': results.get('f1_score', 0),
                'Type': results.get('model_type', model_name)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        """Plot model comparison"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        axes[0].bar(comparison_df['Model'], comparison_df['Accuracy'])
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        
        # F1 Score comparison
        axes[1].bar(comparison_df['Model'], comparison_df['F1 Score'])
        axes[1].set_title('Model F1 Score Comparison')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if self.config.save_visualizations:
            plt.savefig(f"{self.config.output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.show()

# Run baseline comparison
baseline_comparison = BaselineComparison(test_loader, task_type="classification")

# Evaluate baselines
random_results = baseline_comparison.evaluate_random_baseline(num_classes=4)
lstm_results = baseline_comparison.evaluate_lstm_baseline(
    vocab_size=tokenizer.vocab_size, 
    num_classes=4
)

# Compare with Hyena results
comparison_df = baseline_comparison.compare_models(test_metrics)
print("Model Comparison:")
print(comparison_df)

baseline_comparison.plot_model_comparison(comparison_df)
```

## Statistical Significance Testing

### Significance Tests Implementation

```python
class StatisticalTester:
    """Statistical significance testing for model comparisons"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def mcnemar_test(self, model1_predictions, model2_predictions, true_labels):
        """McNemar's test for comparing two models"""
        from statsmodels.stats.contingency_tables import mcnemar
        
        y_true = np.array(true_labels)
        pred1 = np.array(model1_predictions)
        pred2 = np.array(model2_predictions)
        
        # Create contingency table
        # [Model1 correct, Model2 correct]
        # [Model1 correct, Model2 wrong]  
        # [Model1 wrong, Model2 correct]
        # [Model1 wrong, Model2 wrong]
        
        correct1 = (pred1 == y_true)
        correct2 = (pred2 == y_true)
        
        both_correct = np.sum(correct1 & correct2)
        model1_only = np.sum(correct1 & ~correct2)
        model2_only = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        # Contingency table for McNemar's test
        table = np.array([[both_correct, model1_only],
                         [model2_only, both_wrong]])
        
        result = mcnemar(table, exact=True)
        
        return {
            'statistic': result.statistic,
            'pvalue': result.pvalue,
            'significant': result.pvalue < self.alpha,
            'contingency_table': table.tolist()
        }
    
    def bootstrap_difference_test(self, metric1_samples, metric2_samples, n_bootstrap=10000):
        """Bootstrap test for difference in metrics"""
        
        def bootstrap_difference(samples1, samples2, n_bootstrap):
            """Bootstrap the difference between two samples"""
            n1, n2 = len(samples1), len(samples2)
            differences = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample from each group
                boot1 = np.random.choice(samples1, n1, replace=True)
                boot2 = np.random.choice(samples2, n2, replace=True)
                
                # Compute difference in means
                diff = np.mean(boot1) - np.mean(boot2)
                differences.append(diff)
            
            return np.array(differences)
        
        # Bootstrap the difference
        bootstrap_diffs = bootstrap_difference(metric1_samples, metric2_samples, n_bootstrap)
        
        # Compute confidence interval for the difference
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Test if zero is in the confidence interval
        significant = not (ci_lower <= 0 <= ci_upper)
        
        return {
            'mean_difference': np.mean(bootstrap_diffs),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant,
            'p_value_approx': np.mean(bootstrap_diffs <= 0) * 2  # Two-tailed test approximation
        }
    
    def paired_t_test(self, metric1_values, metric2_values):
        """Paired t-test for comparing metrics"""
        from scipy.stats import ttest_rel
        
        statistic, pvalue = ttest_rel(metric1_values, metric2_values)
        
        return {
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha,
            'mean_difference': np.mean(metric1_values) - np.mean(metric2_values)
        }

# Example usage
statistical_tester = StatisticalTester(alpha=0.05)

# Compare Hyena vs LSTM predictions
hyena_predictions = evaluator.predictions
lstm_predictions = baseline_comparison.results['lstm']['predictions'] if 'predictions' in baseline_comparison.results['lstm'] else []

if lstm_predictions:
    mcnemar_result = statistical_tester.mcnemar_test(
        hyena_predictions, lstm_predictions, evaluator.true_labels
    )
    
    print("McNemar's Test Results (Hyena vs LSTM):")
    print(f"  Statistic: {mcnemar_result['statistic']:.4f}")
    print(f"  P-value: {mcnemar_result['pvalue']:.4f}")
    print(f"  Significant: {mcnemar_result['significant']}")
```

## Comprehensive Results Report

### Report Generation

```python
class EvaluationReportGenerator:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self, evaluator, config):
        self.evaluator = evaluator
        self.config = config
        self.report_data = {}
    
    def generate_full_report(self, baseline_results=None, statistical_tests=None):
        """Generate comprehensive evaluation report"""
        
        # Collect all results
        self.report_data = {
            'model_info': self.get_model_info(),
            'evaluation_config': self.get_evaluation_config(),
            'performance_metrics': self.evaluator.results,
            'baseline_comparisons': baseline_results,
            'statistical_tests': statistical_tests,
            'confidence_intervals': self.compute_all_confidence_intervals(),
            'error_analysis': self.get_error_analysis()
        }
        
        # Generate report sections
        report_sections = [
            self.generate_executive_summary(),
            self.generate_methodology_section(),
            self.generate_results_section(),
            self.generate_comparison_section(),
            self.generate_error_analysis_section(),
            self.generate_conclusions_section()
        ]
        
        # Combine into full report
        full_report = "\n\n".join(report_sections)
        
        # Save report
        report_path = f"{self.config.output_dir}/evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(full_report)
        
        print(f"Full evaluation report saved to: {report_path}")
        return full_report
    
    def generate_executive_summary(self):
        """Generate executive summary"""
        
        main_metrics = self.evaluator.results.get('classification', {})
        
        summary = f"""# BLT_Hyena Model Evaluation Report

## Executive Summary

This report presents a comprehensive evaluation of the BLT_Hyena model for genomic sequence analysis.

### Key Findings:
- **Overall Accuracy**: {main_metrics.get('accuracy', 0):.3f}
- **F1 Score**: {main_metrics.get('f1_score', 0):.3f}
- **Precision**: {main_metrics.get('precision', 0):.3f}
- **Recall**: {main_metrics.get('recall', 0):.3f}
- **Inference Speed**: {main_metrics.get('throughput', 0):.1f} samples/second

### Model Configuration:
- **Architecture**: BLT_Hyena with {self.evaluator.model.config.num_layers} layers
- **Hidden Size**: {self.evaluator.model.config.hidden_size}
- **Hyena Order**: {getattr(self.evaluator.model.config, 'hyena_order', 'N/A')}
- **Parameters**: {sum(p.numel() for p in self.evaluator.model.parameters()):,}
"""
        
        return summary
    
    def get_model_info(self):
        """Get model information"""
        
        model_info = {
            'architecture': 'BLT_Hyena',
            'num_parameters': sum(p.numel() for p in self.evaluator.model.parameters()),
            'num_layers': self.evaluator.model.config.num_layers,
            'hidden_size': self.evaluator.model.config.hidden_size,
            'vocab_size': self.evaluator.model.config.vocab_size,
            'max_seq_length': getattr(self.evaluator.model.config, 'max_seq_len', 'N/A'),
            'use_hyena': getattr(self.evaluator.model.config, 'use_hyena', False),
            'hyena_order': getattr(self.evaluator.model.config, 'hyena_order', 'N/A')
        }
        
        return model_info
    
    def compute_all_confidence_intervals(self):
        """Compute confidence intervals for all metrics"""
        
        if not self.evaluator.predictions:
            return {}
        
        ci_results = {}
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        for metric in metrics:
            try:
                ci = self.evaluator.compute_confidence_intervals(metric)
                ci_results[metric] = ci
            except Exception as e:
                print(f"Could not compute CI for {metric}: {e}")
        
        return ci_results

# Generate comprehensive report
report_generator = EvaluationReportGenerator(evaluator, eval_config)
full_report = report_generator.generate_full_report(
    baseline_results=baseline_comparison.results,
    statistical_tests={}  # Add your statistical test results here
)

print("Evaluation Report Generated!")
```

## Key Takeaways

1. **Comprehensive Metrics**: Use multiple evaluation metrics beyond accuracy
2. **Statistical Rigor**: Apply confidence intervals and significance tests
3. **Error Analysis**: Understand failure modes and edge cases
4. **Baseline Comparison**: Compare against reasonable baselines
5. **Task-Specific Evaluation**: Tailor evaluation to genomic applications
6. **Reproducibility**: Document evaluation procedures thoroughly

## Troubleshooting

### Common Evaluation Issues

1. **Memory Issues During Evaluation**
   - Reduce batch size
   - Use gradient checkpointing
   - Process in smaller chunks

2. **Inconsistent Results**
   - Set random seeds for reproducibility
   - Use stratified sampling for small datasets
   - Check for data leakage

3. **Slow Evaluation**
   - Optimize data loading
   - Use GPU acceleration
   - Implement parallel processing

## Next Steps

Continue to [06 - Production](06_PRODUCTION.md) to learn about deploying BLT_Hyena models in production environments, or explore [07 - Advanced Topics](07_ADVANCED.md) for research applications.

## Additional Resources

- [Evaluation Best Practices](../docs/EVALUATION_BEST_PRACTICES.md)
- [Statistical Testing Guide](../docs/STATISTICAL_TESTING.md)
- [Benchmark Datasets](../docs/BENCHMARK_DATASETS.md)
