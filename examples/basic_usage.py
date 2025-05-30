#!/usr/bin/env python3
"""
Basic Usage Example for Hyena-GLT Framework

This example demonstrates the fundamental usage patterns of Hyena-GLT for genomic sequence modeling.
"""

import torch
import numpy as np
from pathlib import Path

# Hyena-GLT imports
from hyena_glt import HyenaGLT, HyenaGLTConfig
from hyena_glt.data import GenomicTokenizer, GenomicDataset
from hyena_glt.utils import analyze_tokenization, plot_sequence_embeddings

def main():
    print("🧬 Hyena-GLT Basic Usage Example")
    print("=" * 50)
    
    # 1. Initialize configuration
    print("1. Setting up configuration...")
    config = HyenaGLTConfig(
        vocab_size=4096,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        sequence_length=2048,
        dropout=0.1
    )
    print(f"   ✓ Config created with {config.num_layers} layers, {config.hidden_size} hidden size")
    
    # 2. Initialize tokenizer
    print("\n2. Initializing genomic tokenizer...")
    tokenizer = GenomicTokenizer(
        sequence_type="dna",
        vocab_size=config.vocab_size,
        max_length=config.sequence_length
    )
    print(f"   ✓ Tokenizer ready for {tokenizer.sequence_type.upper()} sequences")
    
    # 3. Initialize model
    print("\n3. Creating Hyena-GLT model...")
    model = HyenaGLT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {total_params:,} parameters")
    
    # 4. Example DNA sequence
    print("\n4. Processing example DNA sequence...")
    dna_sequence = (
        "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA"
        "TCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT"
        "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
    )
    print(f"   ✓ DNA sequence: {dna_sequence[:50]}... ({len(dna_sequence)} bp)")
    
    # 5. Tokenization
    print("\n5. Tokenizing sequence...")
    tokens = tokenizer.encode(dna_sequence)
    print(f"   ✓ Tokenized to {len(tokens)} tokens")
    print(f"   ✓ Token sample: {tokens[:10]}...")
    
    # 6. Model inference
    print("\n6. Running model inference...")
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokens])
        outputs = model(input_ids)
        
    print(f"   ✓ Output shape: {outputs.logits.shape}")
    print(f"   ✓ Output statistics: mean={outputs.logits.mean():.4f}, std={outputs.logits.std():.4f}")
    
    # 7. Analysis
    print("\n7. Analyzing results...")
    
    # Tokenization analysis
    token_stats = analyze_tokenization(tokenizer, [dna_sequence])
    print(f"   ✓ Average tokens per sequence: {token_stats['avg_tokens']:.1f}")
    print(f"   ✓ Compression ratio: {token_stats['compression_ratio']:.2f}")
    
    # Embedding analysis
    embeddings = outputs.hidden_states[-1]  # Last layer embeddings
    print(f"   ✓ Embedding shape: {embeddings.shape}")
    print(f"   ✓ Embedding norm: {torch.norm(embeddings).item():.4f}")
    
    # 8. Generation example (simple next-token prediction)
    print("\n8. Next-token prediction example...")
    with torch.no_grad():
        # Get logits for last position
        last_logits = outputs.logits[0, -1, :]
        predicted_token = torch.argmax(last_logits).item()
        predicted_prob = torch.softmax(last_logits, dim=-1)[predicted_token].item()
        
    print(f"   ✓ Predicted next token: {predicted_token}")
    print(f"   ✓ Prediction confidence: {predicted_prob:.4f}")
    
    # 9. Sequence classification example
    print("\n9. Sequence classification example...")
    # Add classification head for demo
    classifier = torch.nn.Linear(config.hidden_size, 5)  # 5 classes
    
    with torch.no_grad():
        # Use CLS token or mean pooling
        sequence_repr = embeddings.mean(dim=1)  # [batch_size, hidden_size]
        class_logits = classifier(sequence_repr)
        predicted_class = torch.argmax(class_logits, dim=-1).item()
        class_probs = torch.softmax(class_logits, dim=-1)[0]
        
    print(f"   ✓ Predicted class: {predicted_class}")
    print(f"   ✓ Class probabilities: {class_probs.tolist()[:3]}...")
    
    print("\n" + "=" * 50)
    print("✅ Basic usage example completed successfully!")
    print("\nNext steps:")
    print("- Explore fine_tuning.py for task-specific training")
    print("- Check evaluation.py for model assessment")
    print("- Try generation.py for sequence generation")
    print("=" * 50)

if __name__ == "__main__":
    main()
