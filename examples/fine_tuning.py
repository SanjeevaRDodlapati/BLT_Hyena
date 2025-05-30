#!/usr/bin/env python3
"""
Fine-tuning Example for Hyena-GLT Framework

This example demonstrates how to fine-tune a pre-trained Hyena-GLT model 
for downstream genomic tasks like sequence classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Hyena-GLT imports
from hyena_glt import HyenaGLT, HyenaGLTConfig
from hyena_glt.data import GenomicTokenizer, GenomicDataset
from hyena_glt.training import HyenaGLTTrainer
from hyena_glt.evaluation import evaluate_classification
from hyena_glt.utils import plot_training_curves

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenomicClassificationDataset(GenomicDataset):
    """Dataset for genomic sequence classification tasks."""
    
    def __init__(self, sequences, labels, tokenizer, max_length=1024):
        super().__init__(sequences, tokenizer, max_length)
        self.labels = labels
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def create_synthetic_data(num_samples=1000, seq_length=500, num_classes=3):
    """Create synthetic genomic data for demonstration."""
    np.random.seed(42)
    sequences = []
    labels = []
    
    bases = ['A', 'T', 'G', 'C']
    
    for i in range(num_samples):
        # Create sequence with class-specific patterns
        sequence = []
        label = i % num_classes
        
        for j in range(seq_length):
            if label == 0:  # AT-rich sequences
                base = np.random.choice(['A', 'T'], p=[0.6, 0.4])
            elif label == 1:  # GC-rich sequences  
                base = np.random.choice(['G', 'C'], p=[0.5, 0.5])
            else:  # Balanced sequences
                base = np.random.choice(bases)
            sequence.append(base)
        
        sequences.append(''.join(sequence))
        labels.append(label)
    
    return sequences, labels

def main():
    print("🧬 Hyena-GLT Fine-tuning Example")
    print("=" * 50)
    
    # 1. Configuration
    print("1. Setting up configuration...")
    config = HyenaGLTConfig(
        vocab_size=4096,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        sequence_length=1024,
        dropout=0.1,
        num_classes=3  # For classification
    )
    print(f"   ✓ Config: {config.num_layers} layers, {config.hidden_size} hidden size")
    
    # 2. Create synthetic dataset
    print("\n2. Creating synthetic genomic dataset...")
    sequences, labels = create_synthetic_data(num_samples=1000, num_classes=3)
    
    # Split data
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]
    val_sequences = sequences[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"   ✓ Training samples: {len(train_sequences)}")
    print(f"   ✓ Validation samples: {len(val_sequences)}")
    print(f"   ✓ Number of classes: {len(set(labels))}")
    
    # 3. Initialize tokenizer
    print("\n3. Initializing tokenizer...")
    tokenizer = GenomicTokenizer(
        sequence_type="dna",
        vocab_size=config.vocab_size,
        max_length=config.sequence_length
    )
    print(f"   ✓ Tokenizer ready for {tokenizer.sequence_type.upper()} sequences")
    
    # 4. Create datasets
    print("\n4. Creating datasets...")
    train_dataset = GenomicClassificationDataset(
        train_sequences, train_labels, tokenizer, max_length=512
    )
    val_dataset = GenomicClassificationDataset(
        val_sequences, val_labels, tokenizer, max_length=512
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"   ✓ Train loader: {len(train_loader)} batches")
    print(f"   ✓ Val loader: {len(val_loader)} batches")
    
    # 5. Initialize model
    print("\n5. Creating model...")
    model = HyenaGLT(config)
    
    # Add classification head
    model.classifier = nn.Linear(config.hidden_size, config.num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model with {total_params:,} parameters")
    
    # 6. Setup training
    print("\n6. Setting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"   ✓ Training on: {device}")
    
    # 7. Training loop
    print("\n7. Starting fine-tuning...")
    num_epochs = 5
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            # Classification
            sequence_repr = outputs.hidden_states[-1].mean(dim=1)  # Mean pooling
            logits = model.classifier(sequence_repr)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids)
                sequence_repr = outputs.hidden_states[-1].mean(dim=1)
                logits = model.classifier(sequence_repr)
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        scheduler.step()
        
        print(f"   Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_accuracy:.4f}")
    
    # 8. Final evaluation
    print("\n8. Final evaluation...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            sequence_repr = outputs.hidden_states[-1].mean(dim=1)
            logits = model.classifier(sequence_repr)
            
            _, predicted = torch.max(logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    final_accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"   ✓ Final validation accuracy: {final_accuracy:.4f}")
    
    # Class-wise accuracy
    for class_id in range(config.num_classes):
        class_mask = np.array(all_labels) == class_id
        if class_mask.sum() > 0:
            class_acc = np.mean(
                np.array(all_predictions)[class_mask] == np.array(all_labels)[class_mask]
            )
            print(f"   ✓ Class {class_id} accuracy: {class_acc:.4f}")
    
    # 9. Save model
    print("\n9. Saving fine-tuned model...")
    output_dir = Path("./fine_tuned_model")
    output_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'final_accuracy': final_accuracy
    }, output_dir / "model.pt")
    
    print(f"   ✓ Model saved to: {output_dir}")
    
    print("\n" + "=" * 50)
    print("✅ Fine-tuning completed successfully!")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
