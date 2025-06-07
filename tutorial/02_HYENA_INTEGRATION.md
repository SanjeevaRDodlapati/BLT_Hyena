# 02 - Hyena Integration Deep Dive

**Estimated Time:** 45 minutes  
**Prerequisites:** [01 - Fundamentals](01_FUNDAMENTALS.md)  
**Next:** [03 - Data Pipeline](03_DATA_PIPELINE.md)

## Overview

This tutorial explores the Hyena operator integration in BLT_Hyena, focusing on long-range sequence modeling, attention alternatives, and the mathematical foundations that make Hyena efficient for genomic sequences.

## What You'll Learn

- Understanding Hyena operators and their advantages over attention
- Configuring Hyena parameters for different sequence types
- Working with long-range dependencies in genomic data
- Performance optimization techniques
- Integration patterns with transformer blocks

## Hyena Architecture Fundamentals

### The Attention Problem

Traditional transformers face quadratic complexity O(n²) with sequence length, making them impractical for long genomic sequences:

```python
# Traditional attention complexity
sequence_length = 10000  # Typical gene sequence
attention_ops = sequence_length ** 2  # 100M operations
print(f"Attention operations: {attention_ops:,}")
```

### Hyena Solution

Hyena achieves subquadratic complexity O(n log n) through:

1. **Implicit convolutions** - Replace attention with learned convolutions
2. **FFT-based computation** - Efficient frequency domain operations  
3. **Gating mechanisms** - Control information flow

## Practical Implementation

### Basic Hyena Configuration

```python
from hyena_glt import HyenaGLTConfig, HyenaGLT

# Configure Hyena-specific parameters
config = HyenaGLTConfig(
    vocab_size=4,  # DNA: A, T, G, C
    hidden_size=768,
    num_layers=12,
    
    # Hyena-specific settings
    use_hyena=True,
    hyena_order=2,  # Number of implicit convolutions
    hyena_filter_size=128,  # Convolution filter size
    max_seq_len=8192,  # Maximum sequence length
    
    # Performance optimizations
    use_fft_conv=True,  # Enable FFT convolutions
    hyena_training_additions=True
)

model = HyenaGLT(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Understanding Hyena Order

The `hyena_order` parameter controls the number of implicit convolutions:

```python
# Test different Hyena orders
import torch
from hyena_glt.models.hyena_block import HyenaBlock

def compare_hyena_orders():
    configs = [
        {"order": 1, "description": "Simple convolution"},
        {"order": 2, "description": "Two-stage processing"},
        {"order": 3, "description": "Complex interactions"}
    ]
    
    for config in configs:
        # Create test input
        batch_size, seq_len, hidden_size = 2, 1024, 768
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Configure Hyena block
        hyena_config = HyenaGLTConfig(
            hidden_size=hidden_size,
            hyena_order=config["order"],
            use_hyena=True
        )
        
        block = HyenaBlock(hyena_config)
        
        # Measure performance
        import time
        start_time = time.time()
        with torch.no_grad():
            output = block(x)
        end_time = time.time()
        
        print(f"Order {config['order']}: {config['description']}")
        print(f"  Processing time: {end_time - start_time:.4f}s")
        print(f"  Output shape: {output.shape}")
        print()

compare_hyena_orders()
```

## Working with Long Sequences

### Sequence Length Optimization

```python
def optimize_for_sequence_length(target_length):
    """
    Configure Hyena for optimal performance at target sequence length
    """
    # Calculate optimal filter size
    filter_size = min(128, target_length // 64)
    filter_size = max(32, filter_size)  # Minimum filter size
    
    # Adjust FFT settings
    use_fft = target_length > 512
    
    config = HyenaGLTConfig(
        vocab_size=4,
        hidden_size=768,
        num_layers=12,
        
        # Length-optimized settings
        max_seq_len=target_length,
        hyena_filter_size=filter_size,
        use_fft_conv=use_fft,
        
        # Memory optimization
        gradient_checkpointing=target_length > 4096,
        use_hyena=True,
        hyena_order=2
    )
    
    return config

# Test with different sequence lengths
for length in [1024, 4096, 8192, 16384]:
    config = optimize_for_sequence_length(length)
    print(f"Length {length}: filter_size={config.hyena_filter_size}, "
          f"use_fft={config.use_fft_conv}")
```

### Memory-Efficient Processing

```python
import torch
from torch.utils.checkpoint import checkpoint

def process_long_sequence_efficiently(model, sequence, chunk_size=2048):
    """
    Process very long sequences using chunking and checkpointing
    """
    if len(sequence) <= chunk_size:
        return model(sequence)
    
    # Split into overlapping chunks
    chunks = []
    overlap = chunk_size // 4
    
    for i in range(0, len(sequence), chunk_size - overlap):
        end_idx = min(i + chunk_size, len(sequence))
        chunk = sequence[i:end_idx]
        chunks.append(chunk)
    
    # Process chunks with gradient checkpointing
    outputs = []
    for chunk in chunks:
        # Use checkpointing to save memory
        output = checkpoint(model, chunk, use_reentrant=False)
        outputs.append(output)
    
    # Combine outputs (handle overlaps)
    return combine_chunk_outputs(outputs, overlap)

def combine_chunk_outputs(outputs, overlap):
    """Combine overlapping chunk outputs"""
    if len(outputs) == 1:
        return outputs[0]
    
    combined = outputs[0]
    for i in range(1, len(outputs)):
        # Remove overlap and concatenate
        start_idx = overlap if i > 0 else 0
        combined = torch.cat([combined, outputs[i][start_idx:]], dim=1)
    
    return combined
```

## Advanced Hyena Features

### Custom Convolution Patterns

```python
from hyena_glt.models.hyena_filter import HyenaFilter

class GenomicHyenaFilter(HyenaFilter):
    """Custom Hyena filter optimized for genomic patterns"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Add genomic-specific initialization
        self.init_genomic_patterns()
    
    def init_genomic_patterns(self):
        """Initialize filters with genomic sequence patterns"""
        # Common genomic motifs (simplified)
        motifs = {
            'cg_dinucleotide': [1, -1, 1, -1],  # CG pattern
            'tata_box': [1, 1, -1, 1],  # TATA pattern
            'palindrome': [1, -1, -1, 1]  # Palindromic pattern
        }
        
        # Initialize some filters with these patterns
        with torch.no_grad():
            for i, (name, pattern) in enumerate(motifs.items()):
                if i < self.filter_size:
                    # Embed pattern into filter
                    pattern_tensor = torch.tensor(pattern, dtype=torch.float32)
                    # Repeat pattern to match filter size
                    repeats = self.filter_size // len(pattern) + 1
                    extended_pattern = pattern_tensor.repeat(repeats)[:self.filter_size]
                    
                    # Apply to filter weights
                    self.filters[i] = extended_pattern

# Use custom filter
config = HyenaGLTConfig(
    vocab_size=4,
    hidden_size=768,
    use_hyena=True,
    custom_hyena_filter=GenomicHyenaFilter
)
```

### Performance Monitoring

```python
class HyenaPerformanceMonitor:
    """Monitor Hyena performance and efficiency"""
    
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        self.forward_times = []
        self.memory_usage = []
        self.sequence_lengths = []
    
    def monitor_forward_pass(self, model, input_tensor):
        """Monitor a single forward pass"""
        import time
        import psutil
        import os
        
        # Record input stats
        batch_size, seq_len = input_tensor.shape[:2]
        self.sequence_lengths.append(seq_len)
        
        # Monitor memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time forward pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Monitor memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Record stats
        self.forward_times.append(end_time - start_time)
        self.memory_usage.append(memory_after - memory_before)
        
        return output
    
    def print_stats(self):
        """Print performance statistics"""
        if not self.forward_times:
            print("No performance data recorded")
            return
        
        avg_time = sum(self.forward_times) / len(self.forward_times)
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)
        avg_seq_len = sum(self.sequence_lengths) / len(self.sequence_lengths)
        
        print(f"Performance Statistics:")
        print(f"  Average sequence length: {avg_seq_len:.0f}")
        print(f"  Average forward time: {avg_time:.4f}s")
        print(f"  Average memory usage: {avg_memory:.2f}MB")
        print(f"  Throughput: {avg_seq_len/avg_time:.0f} tokens/second")

# Usage example
monitor = HyenaPerformanceMonitor()
model = HyenaGLT(config)

for seq_len in [512, 1024, 2048, 4096]:
    test_input = torch.randint(0, 4, (1, seq_len))
    monitor.monitor_forward_pass(model, test_input)

monitor.print_stats()
```

## Integration Patterns

### Hybrid Attention-Hyena Models

```python
class HybridAttentionHyenaConfig(HyenaGLTConfig):
    """Configuration for hybrid attention-Hyena models"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Define which layers use attention vs Hyena
        self.attention_layers = [0, 3, 6, 9]  # Use attention for these layers
        self.hyena_layers = [1, 2, 4, 5, 7, 8, 10, 11]  # Use Hyena for these
        
        # Attention settings for specific layers
        self.attention_window_size = 512  # Local attention window
        self.use_sparse_attention = True

def create_hybrid_model():
    """Create a model that uses both attention and Hyena"""
    config = HybridAttentionHyenaConfig(
        vocab_size=4,
        hidden_size=768,
        num_layers=12,
        use_hyena=True,
        use_attention=True
    )
    
    return HyenaGLT(config)
```

### Fine-tuning Hyena Parameters

```python
def fine_tune_hyena_for_task(model, task_type="classification"):
    """
    Fine-tune Hyena parameters for specific tasks
    """
    if task_type == "classification":
        # For classification, focus on global patterns
        for layer in model.layers:
            if hasattr(layer, 'hyena_block'):
                # Increase filter size for global context
                layer.hyena_block.filter_size = min(256, layer.hyena_block.filter_size * 2)
                
    elif task_type == "generation":
        # For generation, focus on local patterns
        for layer in model.layers:
            if hasattr(layer, 'hyena_block'):
                # Optimize for causal generation
                layer.hyena_block.causal = True
                
    elif task_type == "variant_calling":
        # For variant calling, focus on local mutations
        for layer in model.layers:
            if hasattr(layer, 'hyena_block'):
                # Smaller filters for fine-grained patterns
                layer.hyena_block.filter_size = max(32, layer.hyena_block.filter_size // 2)
    
    return model
```

## Practical Exercise

### Genomic Sequence Analysis with Hyena

```python
def analyze_genomic_sequence_with_hyena():
    """Complete example: analyze a genomic sequence using Hyena"""
    
    # 1. Load and preprocess genomic data
    from hyena_glt.data.genomic_tokenizer import GenomicTokenizer
    
    tokenizer = GenomicTokenizer()
    
    # Sample genomic sequence (chromosome segment)
    sequence = "ATGCGTACGTAGCTAGCGATCGTAGCTAGC" * 100  # 3000bp sequence
    
    # Tokenize
    tokens = tokenizer.encode(sequence)
    input_tensor = torch.tensor([tokens])
    
    # 2. Configure Hyena for genomic analysis
    config = HyenaGLTConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_layers=6,
        
        # Optimized for genomic sequences
        use_hyena=True,
        hyena_order=2,
        hyena_filter_size=64,
        max_seq_len=len(tokens),
        
        # Task-specific settings
        num_classes=2,  # Binary classification example
        task_type="classification"
    )
    
    # 3. Create and configure model
    model = HyenaGLT(config)
    model = fine_tune_hyena_for_task(model, "classification")
    
    # 4. Run analysis
    monitor = HyenaPerformanceMonitor()
    
    with torch.no_grad():
        output = monitor.monitor_forward_pass(model, input_tensor)
        
        # Extract features
        sequence_features = output.last_hidden_state
        classification_logits = output.logits
        
        print(f"Sequence length: {len(tokens)}")
        print(f"Feature shape: {sequence_features.shape}")
        print(f"Classification logits: {classification_logits}")
        
    monitor.print_stats()
    
    return {
        'features': sequence_features,
        'predictions': classification_logits,
        'performance': monitor
    }

# Run the analysis
results = analyze_genomic_sequence_with_hyena()
```

## Key Takeaways

1. **Efficiency**: Hyena provides O(n log n) complexity vs O(n²) for attention
2. **Long Sequences**: Optimized for genomic sequences up to 16K+ tokens
3. **Flexibility**: Configurable order and filter sizes for different tasks
4. **Memory**: Gradient checkpointing enables very long sequence processing
5. **Performance**: FFT convolutions provide significant speedup

## Troubleshooting

### Common Issues

1. **Memory errors with long sequences**
   - Enable gradient checkpointing
   - Reduce batch size
   - Use chunked processing

2. **Slow performance**
   - Enable FFT convolutions (`use_fft_conv=True`)
   - Optimize filter size for your sequence length
   - Consider hybrid attention-Hyena approach

3. **Poor convergence**
   - Adjust learning rate for Hyena parameters
   - Initialize filters with domain knowledge
   - Use appropriate sequence length during training

## Next Steps

Now that you understand Hyena integration, proceed to [03 - Data Pipeline](03_DATA_PIPELINE.md) to learn about genomic data processing, or explore [04 - Training](04_TRAINING.md) for end-to-end model training with Hyena.

## Additional Resources

- [Hyena Paper](https://arxiv.org/abs/2302.10866) - Original Hyena research
- [BLT_Hyena Architecture](../docs/ARCHITECTURE.md) - Detailed architecture documentation
- [Performance Benchmarks](../docs/PERFORMANCE.md) - Hyena vs Attention comparisons
