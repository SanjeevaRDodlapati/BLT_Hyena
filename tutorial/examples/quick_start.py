#!/usr/bin/env python3
"""
BLT_Hyena Quick Start Example

Description: Complete 5-minute demo showing basic BLT_Hyena functionality
This example demonstrates model creation, training, and inference.

Requirements:
- torch>=1.12.0
- transformers>=4.20.0
- hyena_glt

Usage:
    python quick_start.py

Author: BLT_Hyena Team
Date: 2024-01-01
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List

# Import BLT_Hyena components
try:
    from hyena_glt import HyenaGLT, HyenaGLTConfig
    from hyena_glt.training import HyenaGLTTrainer
    from hyena_glt.data import GenomicTokenizer
except ImportError:
    print("BLT_Hyena not installed. Please install with: pip install -e .")
    exit(1)

def create_sample_data(num_samples: int = 1000, seq_length: int = 128) -> tuple:
    """Create sample genomic data for demonstration."""
    print("Creating sample genomic data...")
    
    # Create random DNA sequences (A=0, T=1, G=2, C=3)
    sequences = torch.randint(0, 4, (num_samples, seq_length))
    
    # Create random binary labels (e.g., coding vs non-coding)
    labels = torch.randint(0, 2, (num_samples,))
    
    print(f"Created {num_samples} sequences of length {seq_length}")
    return sequences, labels

def setup_model() -> tuple:
    """Set up BLT_Hyena model configuration."""
    print("Setting up BLT_Hyena model...")
    
    # Configure model
    config = HyenaGLTConfig(
        vocab_size=4,  # A, T, G, C
        hidden_size=256,
        num_hidden_layers=6,
        hyena_order=2,
        max_position_embeddings=512,
        num_labels=2  # Binary classification
    )
    
    # Create model
    model = HyenaGLT(config)
    
    # Add classification head
    model.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config

def train_model(model: nn.Module, train_data: tuple, val_data: tuple, epochs: int = 3):
    """Train the model with sample data."""
    print("Starting model training...")
    
    train_sequences, train_labels = train_data
    val_sequences, val_labels = val_data
    
    # Create datasets
    train_dataset = TensorDataset(train_sequences, train_labels)
    val_dataset = TensorDataset(val_sequences, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_sequences, batch_labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_sequences)
            logits = model.classifier(outputs.last_hidden_state.mean(dim=1))
            
            # Compute loss
            loss = criterion(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                outputs = model(batch_sequences)
                logits = model.classifier(outputs.last_hidden_state.mean(dim=1))
                
                val_loss += criterion(logits, batch_labels).item()
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        model.train()
    
    print("Training completed!")
    return model

def test_inference(model: nn.Module, test_sequences: torch.Tensor):
    """Test model inference on sample sequences."""
    print("Testing model inference...")
    
    model.eval()
    with torch.no_grad():
        # Single sequence inference
        single_seq = test_sequences[0:1]  # Shape: (1, seq_length)
        
        outputs = model(single_seq)
        features = outputs.last_hidden_state
        logits = model.classifier(features.mean(dim=1))
        probabilities = torch.softmax(logits, dim=1)
        
        print(f"Single sequence inference:")
        print(f"  Input shape: {single_seq.shape}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Prediction probabilities: {probabilities[0].tolist()}")
        print(f"  Predicted class: {torch.argmax(probabilities, dim=1).item()}")
        
        # Batch inference
        batch_seq = test_sequences[:5]  # Shape: (5, seq_length)
        
        batch_outputs = model(batch_seq)
        batch_features = batch_outputs.last_hidden_state
        batch_logits = model.classifier(batch_features.mean(dim=1))
        batch_probs = torch.softmax(batch_logits, dim=1)
        
        print(f"\nBatch inference:")
        print(f"  Input shape: {batch_seq.shape}")
        print(f"  Predictions: {torch.argmax(batch_probs, dim=1).tolist()}")
    
    print("Inference testing completed!")

def save_and_load_model(model: nn.Module, config: HyenaGLTConfig):
    """Demonstrate saving and loading model."""
    print("Testing model save/load...")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'blt_hyena_demo.pt')
    
    # Create new model and load weights
    new_model = HyenaGLT(config)
    new_model.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    checkpoint = torch.load('blt_hyena_demo.pt')
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model saved and loaded successfully!")
    return new_model

def main():
    """Main demonstration function."""
    print("🧬 BLT_Hyena Quick Start Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create sample data
    sequences, labels = create_sample_data(num_samples=1000, seq_length=128)
    
    # Split data
    train_size = int(0.7 * len(sequences))
    val_size = int(0.15 * len(sequences))
    
    train_data = (sequences[:train_size], labels[:train_size])
    val_data = (sequences[train_size:train_size+val_size], labels[train_size:train_size+val_size])
    test_data = (sequences[train_size+val_size:], labels[train_size+val_size:])
    
    print(f"Data split: {len(train_data[0])} train, {len(val_data[0])} val, {len(test_data[0])} test")
    
    # Setup model
    model, config = setup_model()
    
    # Train model
    trained_model = train_model(model, train_data, val_data, epochs=3)
    
    # Test inference
    test_inference(trained_model, test_data[0])
    
    # Save and load model
    loaded_model = save_and_load_model(trained_model, config)
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Check out tutorial/01_FUNDAMENTALS.md for detailed concepts")
    print("2. Explore tutorial/02_HYENA_INTEGRATION.md for advanced features")
    print("3. Try training on real genomic data with tutorial/03_DATA_PIPELINE.md")

if __name__ == "__main__":
    main()
