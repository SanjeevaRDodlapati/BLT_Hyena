# 📚 BLT_Hyena Fundamentals

**Goal**: Understand the core concepts, architecture, and unique features of BLT_Hyena.

**Time**: ~45 minutes  
**Prerequisites**: [Quick Start](00_QUICK_START.md) completed

---

## 🧠 What is BLT_Hyena?

BLT_Hyena combines two powerful innovations:

1. **BLT (Byte Latent Transformer)**: Dynamic patching for efficient sequence processing
2. **Hyena Operators**: Subquadratic attention alternatives for long sequences

### The Big Picture

```
Raw Genomic Sequence
         ↓
   [Tokenization]
         ↓
   [Local Encoding] ← BLT-inspired
         ↓
  [Dynamic Patching] ← Adaptive merging
         ↓
   [Hyena Layers] ← Efficient long-range modeling
         ↓
  [Cross-Attention] ← Local-global interaction
         ↓
   [Local Decoding] ← BLT-inspired
         ↓
    Final Output
```

---

## 🔧 Core Components

### 1. Dynamic Patching System

The heart of BLT_Hyena's efficiency:

```python
from hyena_glt import HyenaGLTConfig

# Configure dynamic patching
config = HyenaGLTConfig(
    # Patching parameters
    dynamic_patching=True,
    patch_size=8,              # Default patch size
    min_patch_size=4,          # Minimum merge size
    max_patch_size=16,         # Maximum merge size
    
    # Local processing layers
    local_encoder_layers=2,    # Pre-patching processing
    local_decoder_layers=2,    # Post-processing refinement
)

print("🔧 Dynamic patching enabled")
```

**What it does**: Dynamically merges similar tokens to reduce sequence length while preserving information.

### 2. Hyena Operators

Efficient alternatives to attention for long sequences:

```python
config = HyenaGLTConfig(
    # Hyena-specific parameters
    hyena_order=2,                    # Convolution order
    hyena_filter_size=512,           # Filter dimensions
    hyena_short_filter_size=32,      # Short convolution size
    use_bias=True,
    use_glu=True,                    # Gated Linear Units
    hyena_dropout=0.1
)
```

**Benefits**: 
- 📈 Subquadratic complexity O(N log N) vs O(N²) attention
- 🚀 Better scaling for long genomic sequences
- 💾 Lower memory usage

### 3. Cross-Attention Mechanism

Bridges local and global representations:

```python
config = HyenaGLTConfig(
    cross_attention_layers=4,     # Number of cross-attention layers
    num_attention_heads=12,       # Attention heads
    attention_dropout=0.1
)
```

---

## 🧬 Working with Genomic Data

### Sequence Types

BLT_Hyena supports multiple genomic sequence types:

```python
from hyena_glt import HyenaGLTConfig

# DNA sequences
dna_config = HyenaGLTConfig(
    sequence_type="dna",
    genomic_vocab_size=4096,
    enable_reverse_complement=True,  # Include reverse complement
    kmer_size=3                      # 3-mer tokenization
)

# RNA sequences  
rna_config = HyenaGLTConfig(
    sequence_type="rna",
    genomic_vocab_size=4096,
    enable_reverse_complement=False,  # RNA is single-strand
    kmer_size=3
)

# Protein sequences
protein_config = HyenaGLTConfig(
    sequence_type="protein",
    genomic_vocab_size=8192,         # Larger vocab for amino acids
    enable_reverse_complement=False,
    kmer_size=2                      # Smaller k-mers for proteins
)
```

### Tokenization Strategy

```python
from hyena_glt import GenomicTokenizer

# Create tokenizer for DNA
tokenizer = GenomicTokenizer(
    sequence_type="dna",
    vocab_size=4096,
    kmer_size=3,                  # Use 3-mers (e.g., "ATG", "GCA")
    overlap_size=1,               # Overlapping k-mers
    max_length=1024
)

# Example tokenization
sequence = "ATCGATCGATCGATCG"
tokens = tokenizer.tokenize(sequence)
print(f"Sequence: {sequence}")
print(f"Tokens: {tokens}")

# Convert to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs: {token_ids}")
```

---

## 🎯 Understanding the Forward Pass

Let's trace through what happens when you run a sequence through BLT_Hyena:

```python
import torch
from hyena_glt import HyenaGLT, HyenaGLTConfig, GenomicTokenizer

# Setup
config = HyenaGLTConfig(
    hidden_size=256,
    num_layers=4,
    sequence_type="dna",
    dynamic_patching=True,
    local_encoder_layers=2,
    local_decoder_layers=2
)

model = HyenaGLT(config)
tokenizer = GenomicTokenizer(sequence_type="dna", vocab_size=config.genomic_vocab_size)

# Sample sequence
sequence = "ATCGATCGATCGATCGATCGATCGATCGATCG"
print(f"Input sequence length: {len(sequence)}")

# Tokenize
tokens = tokenizer(sequence, return_tensors="pt", padding=True)
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

print(f"Tokenized length: {input_ids.shape[1]}")

# Forward pass with detailed outputs
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_merge_info=True,
        return_dict=True
    )

print(f"Output shape: {outputs['last_hidden_state'].shape}")
print(f"Hidden states: {len(outputs['hidden_states'])} layers")

if 'merge_info' in outputs:
    print(f"Merge operations: {len(outputs['merge_info'])}")
    for i, merge in enumerate(outputs['merge_info']):
        if merge:
            print(f"  Layer {i}: Merged tokens")
```

### What Happens Inside:

1. **Embedding**: Sequence → token embeddings
2. **Position Encoding**: Add positional information  
3. **Local Encoding**: Initial local context building
4. **Dynamic Patching**: Merge similar tokens adaptively
5. **Hyena Processing**: Main sequence modeling with efficient operators
6. **Cross-Attention**: Local-global information exchange
7. **Local Decoding**: Final refinement
8. **Output**: Rich sequence representations

---

## 🎛️ Key Configuration Options

### Model Size Presets

```python
# Small model (fast, less memory)
small_config = HyenaGLTConfig(
    hidden_size=256,
    num_layers=6,
    num_attention_heads=8,
    intermediate_size=1024
)

# Base model (balanced)
base_config = HyenaGLTConfig(
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)

# Large model (high performance)
large_config = HyenaGLTConfig(
    hidden_size=1024,
    num_layers=24,
    num_attention_heads=16,
    intermediate_size=4096
)
```

### Memory Optimization

```python
# For limited memory
memory_efficient_config = HyenaGLTConfig(
    use_gradient_checkpointing=True,   # Trade compute for memory
    hyena_dropout=0.1,                 # Regularization
    attention_dropout=0.1,
    dropout=0.1,
    
    # Reduce model size
    hidden_size=512,
    num_layers=8,
    
    # Efficient patching
    dynamic_patching=True,
    max_patch_size=32                  # Larger patches = less memory
)
```

### Long Sequence Handling

```python
# For very long sequences (>10k tokens)
long_sequence_config = HyenaGLTConfig(
    max_position_embeddings=32768,     # Support long sequences
    hyena_filter_size=1024,            # Larger filters for long-range
    
    # Aggressive patching for efficiency
    dynamic_patching=True,
    patch_size=16,
    max_patch_size=64,
    
    # Use gradient checkpointing
    use_gradient_checkpointing=True
)
```

---

## 🧪 Hands-On Experiment

Let's see how dynamic patching affects sequence length:

```python
import torch
from hyena_glt import HyenaGLT, HyenaGLTConfig, GenomicTokenizer

# Create two configs: with and without patching
config_with_patching = HyenaGLTConfig(
    hidden_size=256,
    num_layers=4,
    dynamic_patching=True,
    min_patch_size=2,
    max_patch_size=8
)

config_without_patching = HyenaGLTConfig(
    hidden_size=256, 
    num_layers=4,
    dynamic_patching=False
)

# Create models
model_with = HyenaGLT(config_with_patching)
model_without = HyenaGLT(config_without_patching)

# Test sequence (repetitive = good for merging)
test_sequence = "ATGC" * 50  # 200 characters of repetitive DNA
tokenizer = GenomicTokenizer(sequence_type="dna", vocab_size=4096)

tokens = tokenizer(test_sequence, return_tensors="pt")
input_ids = tokens['input_ids']

print(f"Original sequence length: {len(test_sequence)}")
print(f"Tokenized length: {input_ids.shape[1]}")

# Test both models
with torch.no_grad():
    # Without patching
    output_without = model_without(input_ids, output_merge_info=True)
    
    # With patching  
    output_with = model_with(input_ids, output_merge_info=True)

print(f"\nWithout patching - Output length: {output_without['last_hidden_state'].shape[1]}")
print(f"With patching - Output length: {output_with['last_hidden_state'].shape[1]}")

compression_ratio = output_without['last_hidden_state'].shape[1] / output_with['last_hidden_state'].shape[1]
print(f"Compression ratio: {compression_ratio:.2f}x")
```

---

## 🎯 Key Takeaways

After completing this tutorial, you understand:

✅ **BLT_Hyena Architecture**: How dynamic patching + Hyena operators work together  
✅ **Configuration Options**: Key parameters for different use cases  
✅ **Genomic Data Handling**: DNA/RNA/protein sequence processing  
✅ **Forward Pass Flow**: What happens inside the model  
✅ **Memory & Performance**: How to optimize for your constraints  

---

## 🚀 What's Next?

Now that you understand the fundamentals, choose your path:

### 🔗 **Dive Deeper into Hyena**
→ [02_HYENA_INTEGRATION.md](02_HYENA_INTEGRATION.md) - Learn about Hyena operators, convolutions, and long-range modeling

### 📊 **Build Data Pipelines**  
→ [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md) - Process real genomic datasets, handle FASTA files, and scale up

### 🏋️ **Start Training**
→ [04_TRAINING.md](04_TRAINING.md) - Complete training workflows with checkpointing, evaluation, and optimization

### 🔧 **Advanced Topics**
→ [07_ADVANCED.md](07_ADVANCED.md) - Multi-task learning, custom architectures, and research applications

---

## 🤔 Common Questions

**Q: When should I use dynamic patching?**  
A: Always for long sequences (>1000 tokens) or when memory is limited. It provides significant efficiency gains with minimal accuracy loss.

**Q: How do I choose the right model size?**  
A: Start with base config. Use small for prototyping or resource constraints, large for maximum performance on important tasks.

**Q: What's the maximum sequence length?**  
A: Theoretically unlimited with dynamic patching. Practically tested up to 32K tokens. Memory and compute scale subquadratically.

---

*Next up: [Understanding Hyena Integration →](02_HYENA_INTEGRATION.md)*
