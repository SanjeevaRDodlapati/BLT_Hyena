#!/usr/bin/env python3
"""
Generation Example for Hyena-GLT Framework

This example demonstrates genomic sequence generation using Hyena-GLT models
including conditional generation, sequence completion, and analysis.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Hyena-GLT imports
from hyena_glt import HyenaGLT, HyenaGLTConfig
from hyena_glt.data import GenomicTokenizer
from hyena_glt.utils import (
    compute_sequence_statistics,
    analyze_tokenization,
    validate_sequence
)

def load_model_for_generation(checkpoint_path):
    """Load model from checkpoint for generation."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    model = HyenaGLT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = checkpoint['tokenizer']
    
    return model, config, tokenizer

def generate_sequence(model, tokenizer, prompt="", max_length=200, 
                     temperature=1.0, top_k=50, top_p=0.9, device='cpu'):
    """Generate genomic sequence with various sampling strategies."""
    model.eval()
    
    # Encode prompt
    if prompt:
        tokens = tokenizer.encode(prompt)
    else:
        # Start with a random token or special start token
        tokens = [tokenizer.vocab.get('<START>', 1)]
    
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            # Prepare input
            input_ids = torch.tensor([generated_tokens]).to(device)
            
            # Get predictions
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last position logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for stop conditions
            if next_token == tokenizer.vocab.get('<END>', -1):
                break
            
            generated_tokens.append(next_token)
    
    # Decode generated sequence
    generated_sequence = tokenizer.decode(generated_tokens)
    return generated_sequence, generated_tokens

def sequence_completion(model, tokenizer, partial_sequence, max_length=500, device='cpu'):
    """Complete a partial genomic sequence."""
    # Encode partial sequence
    tokens = tokenizer.encode(partial_sequence)
    
    model.eval()
    with torch.no_grad():
        completed_tokens = tokens.copy()
        
        for _ in range(max_length - len(tokens)):
            input_ids = torch.tensor([completed_tokens]).to(device)
            outputs = model(input_ids)
            
            # Get next token prediction
            logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(logits).item()
            
            # Stop if we hit an end token or invalid token
            if next_token == tokenizer.vocab.get('<END>', -1):
                break
            
            completed_tokens.append(next_token)
    
    completed_sequence = tokenizer.decode(completed_tokens)
    return completed_sequence

def conditional_generation(model, tokenizer, condition_type="gc_rich", 
                         length=300, device='cpu'):
    """Generate sequences with specific properties."""
    # Different starting prompts for different conditions
    prompts = {
        "gc_rich": "GCGCGCGC",
        "at_rich": "ATATATATAT", 
        "balanced": "ATGC",
        "protein_coding": "ATG",  # Start codon
    }
    
    prompt = prompts.get(condition_type, "")
    generated_seq, tokens = generate_sequence(
        model, tokenizer, prompt=prompt, max_length=length, 
        temperature=0.8, device=device
    )
    
    return generated_seq

def analyze_generated_sequences(sequences):
    """Analyze properties of generated sequences."""
    stats = compute_sequence_statistics(sequences)
    
    # Additional analysis
    valid_sequences = []
    gc_contents = []
    lengths = []
    
    for seq in sequences:
        if validate_sequence(seq, sequence_type='dna'):
            valid_sequences.append(seq)
            
            # Compute GC content
            gc_count = seq.count('G') + seq.count('C')
            gc_content = gc_count / len(seq) if len(seq) > 0 else 0
            gc_contents.append(gc_content)
            lengths.append(len(seq))
    
    return {
        'total_sequences': len(sequences),
        'valid_sequences': len(valid_sequences),
        'validity_rate': len(valid_sequences) / len(sequences) if sequences else 0,
        'avg_length': np.mean(lengths) if lengths else 0,
        'avg_gc_content': np.mean(gc_contents) if gc_contents else 0,
        'gc_content_std': np.std(gc_contents) if gc_contents else 0,
        'length_std': np.std(lengths) if lengths else 0,
        'base_stats': stats
    }

def plot_generation_analysis(analysis_results, output_dir):
    """Create plots for generation analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot GC content distribution
    plt.figure(figsize=(12, 8))
    
    # Summary statistics
    plt.subplot(2, 2, 1)
    metrics = ['validity_rate', 'avg_gc_content']
    values = [analysis_results[metric] for metric in metrics]
    bars = plt.bar(metrics, values, color=['lightblue', 'lightgreen'])
    plt.title('Generation Quality Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Length distribution (placeholder)
    plt.subplot(2, 2, 2)
    plt.hist([analysis_results['avg_length']] * 10, bins=5, alpha=0.7, color='orange')
    plt.title('Sequence Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    
    # GC content analysis
    plt.subplot(2, 2, 3)
    gc_mean = analysis_results['avg_gc_content']
    gc_std = analysis_results['gc_content_std']
    plt.bar(['Mean GC', 'Std GC'], [gc_mean, gc_std], 
            color=['green', 'red'], alpha=0.7)
    plt.title('GC Content Statistics')
    plt.ylabel('Value')
    
    # Summary table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f"""
    Generation Summary:
    
    Total Sequences: {analysis_results['total_sequences']}
    Valid Sequences: {analysis_results['valid_sequences']}
    Validity Rate: {analysis_results['validity_rate']:.3f}
    
    Avg Length: {analysis_results['avg_length']:.1f}
    Avg GC Content: {analysis_results['avg_gc_content']:.3f}
    """
    plt.text(0.1, 0.5, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("🧬 Hyena-GLT Generation Example")
    print("=" * 50)
    
    # 1. Load model
    checkpoint_path = "./fine_tuned_model/model.pt"
    if not Path(checkpoint_path).exists():
        print("❌ No fine-tuned model found. Please run fine_tuning.py first.")
        print("💡 Creating a demo model for generation...")
        
        # Create demo configuration
        config = HyenaGLTConfig(
            vocab_size=4096,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            sequence_length=1024
        )
        
        model = HyenaGLT(config)
        tokenizer = GenomicTokenizer(sequence_type="dna", vocab_size=config.vocab_size)
        
        print("   ✓ Demo model created (not trained)")
    else:
        print("1. Loading fine-tuned model...")
        model, config, tokenizer = load_model_for_generation(checkpoint_path)
        print("   ✓ Model loaded successfully")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"   ✓ Using device: {device}")
    
    # 2. Unconditional generation
    print("\n2. Unconditional sequence generation...")
    
    generated_sequences = []
    for i in tqdm(range(10), desc="Generating sequences"):
        seq, tokens = generate_sequence(
            model, tokenizer,
            max_length=200,
            temperature=0.8,
            top_k=50,
            device=device
        )
        generated_sequences.append(seq)
    
    print(f"   ✓ Generated {len(generated_sequences)} sequences")
    print(f"   ✓ Sample: {generated_sequences[0][:60]}...")
    
    # 3. Conditional generation
    print("\n3. Conditional sequence generation...")
    
    conditions = ["gc_rich", "at_rich", "balanced", "protein_coding"]
    conditional_sequences = {}
    
    for condition in conditions:
        sequences = []
        for _ in range(5):
            seq = conditional_generation(
                model, tokenizer, 
                condition_type=condition,
                length=150,
                device=device
            )
            sequences.append(seq)
        conditional_sequences[condition] = sequences
        print(f"   ✓ {condition}: {len(sequences)} sequences")
    
    # 4. Sequence completion
    print("\n4. Sequence completion...")
    
    partial_sequences = [
        "ATGCGATCGATCG",
        "GCGCGCGCGC",
        "ATATATAT",
        "ATGAAACGT"
    ]
    
    completed_sequences = []
    for partial in partial_sequences:
        completed = sequence_completion(
            model, tokenizer, partial, 
            max_length=100, device=device
        )
        completed_sequences.append(completed)
        print(f"   ✓ {partial} → {completed[:50]}...")
    
    # 5. Analysis of generated sequences
    print("\n5. Analyzing generated sequences...")
    
    all_sequences = generated_sequences.copy()
    for condition_seqs in conditional_sequences.values():
        all_sequences.extend(condition_seqs)
    all_sequences.extend(completed_sequences)
    
    analysis = analyze_generated_sequences(all_sequences)
    
    print(f"   ✓ Total sequences analyzed: {analysis['total_sequences']}")
    print(f"   ✓ Valid sequences: {analysis['valid_sequences']}")
    print(f"   ✓ Validity rate: {analysis['validity_rate']:.3f}")
    print(f"   ✓ Average length: {analysis['avg_length']:.1f}")
    print(f"   ✓ Average GC content: {analysis['avg_gc_content']:.3f}")
    
    # 6. Conditional analysis
    print("\n6. Conditional generation analysis...")
    
    for condition, sequences in conditional_sequences.items():
        cond_analysis = analyze_generated_sequences(sequences)
        print(f"   {condition}:")
        print(f"     - GC content: {cond_analysis['avg_gc_content']:.3f}")
        print(f"     - Avg length: {cond_analysis['avg_length']:.1f}")
        print(f"     - Validity: {cond_analysis['validity_rate']:.3f}")
    
    # 7. Save results
    print("\n7. Saving generation results...")
    
    output_dir = Path("./generation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save sequences
    results = {
        'unconditional': generated_sequences,
        'conditional': conditional_sequences,
        'completions': {
            'partial': partial_sequences,
            'completed': completed_sequences
        },
        'analysis': analysis,
        'model_config': {
            'num_layers': config.num_layers,
            'hidden_size': config.hidden_size,
            'vocab_size': config.vocab_size
        }
    }
    
    with open(output_dir / 'generation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot analysis
    plot_generation_analysis(analysis, output_dir)
    
    print(f"   ✓ Results saved to: {output_dir}")
    print(f"   ✓ Sequences: generation_results.json")
    print(f"   ✓ Analysis: generation_analysis.png")
    
    # 8. Export sequences to FASTA
    print("\n8. Exporting to FASTA format...")
    
    fasta_file = output_dir / 'generated_sequences.fasta'
    with open(fasta_file, 'w') as f:
        # Unconditional sequences
        for i, seq in enumerate(generated_sequences):
            f.write(f">generated_unconditional_{i+1}\n{seq}\n")
        
        # Conditional sequences
        for condition, sequences in conditional_sequences.items():
            for i, seq in enumerate(sequences):
                f.write(f">generated_{condition}_{i+1}\n{seq}\n")
    
    print(f"   ✓ FASTA exported: {fasta_file}")
    
    print("\n" + "=" * 50)
    print("✅ Generation example completed successfully!")
    print(f"Generated {len(all_sequences)} total sequences")
    print(f"Validity rate: {analysis['validity_rate']:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
