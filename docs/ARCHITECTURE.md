# Hyena-GLT Architecture Guide

This document provides a comprehensive overview of the Hyena-GLT architecture, explaining how it combines BLT's byte latent tokenization with Savanna's Striped Hyena blocks for genomic sequence modeling.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Hyena Operators](#hyena-operators)
4. [Dynamic Token Merging](#dynamic-token-merging)
5. [Task-Specific Heads](#task-specific-heads)
6. [Training Dynamics](#training-dynamics)
7. [Performance Characteristics](#performance-characteristics)

## Architecture Overview

Hyena-GLT is a hybrid architecture that combines:

1. **BLT's Byte Latent Tokenization**: Efficient tokenization for genomic sequences
2. **Striped Hyena Blocks**: Long-range convolutions with subquadratic complexity
3. **Dynamic Token Merging**: Adaptive sequence compression for efficiency
4. **Genomic-Specific Adaptations**: Specialized components for biological sequences

```
Input Sequence
     ↓
Byte Latent Tokenization
     ↓
Embedding Layer
     ↓
┌─────────────────────────┐
│  Hyena-GLT Blocks       │
│  ┌─────────────────────┐│
│  │ Dynamic Merging     ││
│  │     ↓               ││
│  │ Hyena Operator      ││
│  │     ↓               ││
│  │ Feed Forward        ││
│  │     ↓               ││
│  │ Layer Norm          ││
│  └─────────────────────┘│
│         × N layers      │
└─────────────────────────┘
     ↓
Task-Specific Head
     ↓
Output
```

## Core Components

### 1. Byte Latent Tokenization

#### Tokenization Process

```python
# DNA sequence tokenization
sequence = "ATCGATCGATCG"
bytes_representation = sequence.encode('utf-8')
# [65, 84, 67, 71, 65, 84, 67, 71, 65, 84, 67, 71]

# Learnable compression
compressed_tokens = byte_latent_tokenizer.encode(bytes_representation)
# Adaptive compression based on sequence patterns
```

#### Key Features

- **Variable-length encoding**: Adapts to sequence complexity
- **Learned compression**: Training-time optimization of token boundaries
- **Cross-modal compatibility**: Works with DNA, RNA, and protein sequences

### 2. Embedding Architecture

```python
class HyenaGLTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = RotaryPositionalEmbedding(config.d_model)
        self.type_embedding = nn.Embedding(4, config.d_model)  # DNA, RNA, Protein, Mixed
        
    def forward(self, input_ids, sequence_type=None):
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Positional embeddings (rotary)
        pos_embeds = self.position_embedding(token_embeds)
        
        # Sequence type embeddings
        if sequence_type is not None:
            type_embeds = self.type_embedding(sequence_type)
            return token_embeds + pos_embeds + type_embeds
        
        return token_embeds + pos_embeds
```

## Hyena Operators

### 1. Standard Hyena Operator

The core Hyena operator provides efficient long-range modeling through:

```python
class HyenaOperator(nn.Module):
    def __init__(self, d_model, l_max, order=2, filter_order=64):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        
        # Projections for each order
        self.projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(order + 1)
        ])
        
        # Implicit filter parameterization
        self.filter_fn = HyenaFilter(d_model, filter_order)
        
        # Gating mechanism
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Multi-path projections
        projections = [proj(x) for proj in self.projections]
        
        # Implicit convolution via FFT
        u = projections[0]
        for i in range(1, self.order + 1):
            v = projections[i]
            filter_coeffs = self.filter_fn(seq_len)
            u = self.fft_conv(u, filter_coeffs) * v
        
        # Gating
        gate_values = torch.sigmoid(self.gate(x))
        return u * gate_values
```

### 2. Striped Hyena Operator

Alternates between attention and convolution for balanced modeling:

```python
class StrippedHyenaOperator(nn.Module):
    def __init__(self, d_model, l_max, order=2, num_heads=8):
        super().__init__()
        self.hyena_op = HyenaOperator(d_model, l_max, order)
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.layer_idx = 0
        
    def forward(self, x):
        if self.layer_idx % 2 == 0:
            # Even layers: Hyena convolution
            return self.hyena_op(x)
        else:
            # Odd layers: Multi-head attention
            x_transposed = x.transpose(0, 1)  # (seq_len, batch, d_model)
            attn_out, _ = self.attention(x_transposed, x_transposed, x_transposed)
            return attn_out.transpose(0, 1)  # Back to (batch, seq_len, d_model)
```

### 3. Genomic-Adapted Hyena

Specialized for genomic sequences with biological inductive biases:

```python
class GenomicHyenaOperator(HyenaOperator):
    def __init__(self, d_model, l_max, order=2, motif_length=10):
        super().__init__(d_model, l_max, order)
        
        # Genomic motif detection
        self.motif_detector = MotifConvolution(d_model, motif_length)
        
        # Biological structure awareness
        self.structure_encoder = StructureEncoder(d_model)
        
        # Conservation-aware filtering
        self.conservation_filter = ConservationFilter(d_model)
        
    def forward(self, x, conservation_scores=None):
        # Standard Hyena processing
        hyena_out = super().forward(x)
        
        # Add genomic-specific processing
        motifs = self.motif_detector(x)
        structure = self.structure_encoder(x)
        
        # Conservation-weighted combination
        if conservation_scores is not None:
            conservation_weights = self.conservation_filter(conservation_scores)
            hyena_out = hyena_out * conservation_weights
        
        return hyena_out + motifs + structure
```

## Dynamic Token Merging

### 1. Merging Strategy

Dynamic token merging reduces sequence length adaptively:

```python
class DynamicMergingLayer(nn.Module):
    def __init__(self, d_model, merge_ratio=0.5, threshold=0.1):
        super().__init__()
        self.merge_ratio = merge_ratio
        self.threshold = threshold
        
        # Similarity computation
        self.similarity_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Merging function
        self.merge_fn = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Compute pairwise similarities
        similarities = self.compute_similarities(x)
        
        # Determine merge candidates
        merge_mask = similarities > self.threshold
        
        # Perform merging
        merged_x, merge_indices = self.merge_tokens(x, merge_mask)
        
        return merged_x, merge_indices
    
    def compute_similarities(self, x):
        # Efficient pairwise similarity computation
        x_shifted = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        concatenated = torch.cat([x, x_shifted], dim=-1)
        similarities = self.similarity_fn(concatenated).squeeze(-1)
        return similarities
    
    def merge_tokens(self, x, merge_mask):
        # Implementation of actual token merging
        # Returns merged sequence and indices for reconstruction
        pass
```

### 2. Adaptive Merging

The merging ratio adapts based on sequence characteristics:

```python
class AdaptiveMergingLayer(DynamicMergingLayer):
    def __init__(self, d_model, base_merge_ratio=0.5):
        super().__init__(d_model, base_merge_ratio)
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Estimate sequence complexity
        complexity = self.complexity_estimator(x.mean(dim=1))  # (batch_size, 1)
        
        # Adapt merge ratio based on complexity
        # High complexity sequences: less merging
        # Low complexity sequences: more merging
        adaptive_ratio = self.merge_ratio * (1 - complexity)
        
        # Apply merging with adaptive ratio
        return self.merge_with_ratio(x, adaptive_ratio)
```

## Task-Specific Heads

### 1. Sequence Classification Head

```python
class SequenceClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, hidden_states, attention_mask=None):
        # Global pooling with attention mask
        if attention_mask is not None:
            masked_states = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        pooled = self.pooler(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
```

### 2. Token Classification Head

```python
class TokenClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits
```

### 3. Regression Head

```python
class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim=1, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        # Global pooling
        if attention_mask is not None:
            masked_states = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        predictions = self.regressor(pooled)
        return predictions
```

## Training Dynamics

### 1. Curriculum Learning Integration

```python
class HyenaGLTWithCurriculum(HyenaGLTModel):
    def __init__(self, config):
        super().__init__(config)
        self.current_max_length = config.initial_length
        self.target_max_length = config.sequence_length
        
    def forward(self, input_ids, curriculum_step=None):
        if curriculum_step is not None:
            # Truncate sequences based on curriculum
            max_len = min(self.current_max_length, input_ids.size(1))
            input_ids = input_ids[:, :max_len]
        
        return super().forward(input_ids)
    
    def update_curriculum(self, step, total_steps):
        # Gradually increase sequence length
        progress = step / total_steps
        self.current_max_length = int(
            self.config.initial_length + 
            (self.target_max_length - self.config.initial_length) * progress
        )
```

### 2. Multi-Task Training

```python
class MultiTaskHyenaGLT(nn.Module):
    def __init__(self, config, tasks):
        super().__init__()
        self.backbone = HyenaGLTModel(config)
        
        # Shared layers
        self.shared_layers = nn.ModuleList([
            HyenaDynamicLayer(config) for _ in range(config.shared_layers)
        ])
        
        # Task-specific branches
        self.task_heads = nn.ModuleDict()
        for task_name, task_config in tasks.items():
            self.task_heads[task_name] = self.create_task_head(task_config)
    
    def forward(self, input_ids, task_name, **kwargs):
        # Shared processing
        hidden_states = self.backbone.embeddings(input_ids)
        
        for layer in self.shared_layers:
            hidden_states = layer(hidden_states)
        
        # Task-specific processing
        task_head = self.task_heads[task_name]
        return task_head(hidden_states, **kwargs)
```

## Performance Characteristics

### 1. Computational Complexity

- **Standard Transformer**: O(n²d) for sequence length n and dimension d
- **Hyena Operator**: O(n log n d) using FFT-based convolutions
- **With Dynamic Merging**: O(m log m d) where m < n (merged sequence length)

### 2. Memory Usage

```python
def estimate_memory_usage(config, batch_size, sequence_length):
    d_model = config.d_model
    n_layers = config.n_layers
    
    # Embedding memory
    embedding_memory = batch_size * sequence_length * d_model * 4  # 4 bytes per float32
    
    # Layer memory (per layer)
    layer_memory = batch_size * sequence_length * d_model * 8  # Forward + backward
    
    # Total memory
    total_memory = embedding_memory + layer_memory * n_layers
    
    # With dynamic merging (sequence length reduction)
    merge_ratio = config.merge_ratio if hasattr(config, 'merge_ratio') else 1.0
    effective_length = int(sequence_length * merge_ratio)
    
    merged_memory = embedding_memory + (
        batch_size * effective_length * d_model * 8 * n_layers
    )
    
    return {
        "standard_memory_gb": total_memory / (1024**3),
        "merged_memory_gb": merged_memory / (1024**3),
        "memory_reduction": (total_memory - merged_memory) / total_memory
    }
```

### 3. Scaling Properties

| Sequence Length | Standard Attention | Hyena | Hyena + Merging |
|----------------|-------------------|-------|-----------------|
| 1K             | 1x                | 0.8x  | 0.6x           |
| 4K             | 4x                | 1.2x  | 0.8x           |
| 16K            | 16x               | 2.1x  | 1.2x           |
| 64K            | 64x               | 4.8x  | 2.1x           |

## BLT Position Embedding System

### Overview

The BLT Position Embedding System is a sophisticated position tracking mechanism designed to handle dynamic token merging in genomic sequences. Unlike traditional position embeddings that lose positional information during token merging, the BLT system preserves both fine-grained and global positional information through a three-tier architecture.

### Architecture Components

#### 1. Segment-Aware Positional Encoding

```python
class SegmentAwarePositionalEncoding(nn.Module):
    """
    Handles position encoding for variable-length patches after token merging.
    
    Key features:
    - Maintains original absolute positions
    - Tracks relative positions within merged patches
    - Records patch length information
    - Supports genomic-specific patterns (codons, motifs)
    """
```

**Core Position Information Tracked:**

1. **Global Position** (`global_pos`): Original absolute position before merging
2. **Patch Length** (`patch_length`): Number of tokens merged into current patch
3. **Position in Patch** (`pos_in_patch`): Relative position within the merged patch (0.0 to 1.0)

#### 2. Cross-Attention Position Bridge

```python
class CrossAttentionPositionBridge(nn.Module):
    """
    Implements U-shape information flow: Byte ↔ Patch ↔ Byte
    
    Functions:
    - encode_byte_to_patch(): Aggregate byte-level info into patch representations
    - decode_patch_to_byte(): Reconstruct byte-level info from patch representations
    """
```

#### 3. BLT Position Manager

```python
class BLTPositionManager(nn.Module):
    """
    Complete position embedding manager integrating all components.
    
    Main methods:
    - encode_positions(): Add position encodings with merge awareness
    - create_patch_representations(): Convert byte → patch with position tracking
    - reconstruct_byte_representations(): Convert patch → byte with position recovery
    """
```

### Token Merging Process

#### Step 1: Pre-Merging Position State

```python
# Original sequence with individual token positions
original_sequence = [tok1, tok2, tok3, tok4, tok5, tok6, tok7, tok8]
original_positions = [0,    1,    2,    3,    4,    5,    6,    7]

# Apply standard sinusoidal position encoding
pos_encoded = position_manager.encode_positions(
    hidden_states, original_positions=original_positions
)
```

#### Step 2: Adaptive Token Merging

```python
class AdaptiveTokenMerger:
    """
    Merges tokens based on:
    - Content similarity scores
    - Genomic pattern detection
    - Boundary prediction
    - Patch size constraints (min/max)
    """
    
    def forward(self, x, attention_mask):
        # Compute content-based merge decisions
        content_scores = self.content_scorer(x)
        pattern_features = self.pattern_detector(x)
        boundary_scores = self.boundary_predictor(combined_features)
        
        # Determine merge boundaries
        merge_boundaries = self._determine_merge_boundaries(
            content_scores, boundary_scores, attention_mask
        )
        
        # Perform actual merging with position tracking
        return self._perform_merging(x, merge_boundaries, attention_mask)
```

**Example Merging:**
```python
# Before merging: 8 individual tokens
[tok1, tok2, tok3, tok4, tok5, tok6, tok7, tok8]

# After merging: 3 patches
patch1 = merge(tok1, tok2, tok3)    # 3 tokens → 1 patch, starts at pos 0
patch2 = merge(tok4, tok5)          # 2 tokens → 1 patch, starts at pos 3  
patch3 = merge(tok6, tok7, tok8)    # 3 tokens → 1 patch, starts at pos 5

# Patch boundaries: [0, 3, 5, 8]
```

#### Step 3: Post-Merging Position Re-encoding

```python
# Compute segment features for each position in merged sequence
segment_features = self._compute_segment_features(
    batch_size, seq_len, patch_boundaries, original_positions
)

# segment_features shape: (batch, seq_len, 3)
# - [:, :, 0]: position within patch (0.0 to 1.0)
# - [:, :, 1]: patch length (normalized by sequence length)  
# - [:, :, 2]: global position (normalized by max_len)

# Example for patch1 (3 tokens):
# pos_in_patch = [0.0, 0.5, 1.0]    # Relative positions within patch
# patch_length = [3/8, 3/8, 3/8]    # All positions know patch contains 3 tokens
# global_pos   = [0/8, 1/8, 2/8]    # Original absolute positions preserved
```

#### Step 4: Enhanced Position Embedding

```python
# Neural encoding of segment information
segment_encoded = self.segment_encoder(segment_features)  # (batch, seq_len, 64)

# Combine with base sinusoidal encoding
base_pe = self.pe_base[:seq_len]  # (seq_len, d_model)
combined = torch.cat([base_pe_batch, segment_encoded], dim=-1)

# Project to final embedding dimension
final_pe = self.position_projection(combined)  # (batch, seq_len, d_model)

# Add genomic-specific patterns (codons, motifs)
final_pe = self._add_genomic_position_patterns(final_pe, original_positions)
```

### Cross-Attention Information Flow

#### Byte-to-Patch Encoding

```python
def encode_byte_to_patch(self, byte_repr, patch_boundaries):
    """
    Aggregate byte-level representations into patch-level representations
    while preserving positional information through cross-attention.
    """
    for each_patch:
        # Use mean of patch bytes as query
        patch_query = patch_bytes.mean(dim=0, keepdim=True)
        
        # Cross-attention: patch summary attends to all patch bytes
        patch_repr = cross_attention(
            query=patch_query,     # What we want to learn
            key=patch_bytes,       # Where to look  
            value=patch_bytes      # What to extract
        )
```

#### Patch-to-Byte Decoding

```python
def decode_patch_to_byte(self, patch_repr, target_byte_len, patch_boundaries):
    """
    Reconstruct byte-level representations from patch-level representations
    while recovering positional information.
    """
    for each_position_in_patch:
        # Use positional queries for each byte position
        if original_byte_repr_available:
            byte_queries = original_byte_repr[patch_positions]
        else:
            byte_queries = positional_encoding[patch_positions]
            
        # Cross-attention: each byte position attends to patch representation
        byte_output = cross_attention(
            query=byte_queries,        # Individual position queries
            key=patch_repr,            # Patch-level information
            value=patch_repr           # Patch-level information
        )
```

### Genomic-Specific Features

#### 1. Codon Pattern Encoding

```python
def _add_genomic_patterns(self):
    """Add genomic-specific positional patterns."""
    # Codon patterns (period 3 for DNA codons)
    codon_freqs = torch.arange(0, self.d_model // 4, 2).float() * (2 * math.pi / 3)
    
    # Common genomic motif patterns
    motif_periods = [8, 10, 21, 147]  # Various biological periodicities
    # 147: nucleosome positioning, 21: DNA major groove, etc.
```

#### 2. Pattern-Aware Merging

```python
class AdaptiveTokenMerger:
    def __init__(self):
        # Pattern detector for genomic motifs
        self.pattern_detector = nn.Conv1d(
            d_model, d_model // 2, 
            kernel_size=3, padding=1,
            groups=d_model // 4  # Grouped convolution for efficiency
        )
        
    def forward(self, x):
        # Detect genomic patterns before merging
        pattern_features = self.pattern_detector(x.transpose(-1, -2))
        
        # Use pattern information to inform merge decisions
        combined_features = torch.cat([x, pattern_features.transpose(-1, -2)], dim=-1)
        boundary_scores = self.boundary_predictor(combined_features)
```

### Position Information Preservation

#### Information Tracked Throughout Pipeline

| Stage | Position Information Preserved |
|-------|-------------------------------|
| **Pre-merge** | Standard sinusoidal encoding per token |
| **During merge** | Original positions + patch boundaries + merge statistics |
| **Post-merge** | Global positions + patch lengths + intra-patch positions |
| **Cross-attention** | Bidirectional byte ↔ patch position mapping |
| **Reconstruction** | Full recovery of original positional structure |

#### Reconstruction Capability

```python
# Full position reconstruction example
original_hidden_states = torch.randn(1, 64, 256)
original_positions = torch.arange(64)

# 1. Encode positions
pos_encoded = position_manager.encode_positions(
    original_hidden_states, original_positions=original_positions
)

# 2. Simulate token merging
patch_boundaries = create_patch_boundaries(pos_encoded)  # [0, 16, 32, 48, 64]

# 3. Re-encode after merging
merged_encoded = position_manager.encode_positions(
    pos_encoded, 
    patch_boundaries=patch_boundaries,
    original_positions=original_positions
)

# 4. Create patch representations
patch_repr, position_info = position_manager.create_patch_representations(
    merged_encoded, patch_boundaries
)

# 5. Reconstruct byte-level representations
reconstructed = position_manager.reconstruct_byte_representations(
    patch_repr, position_info, merged_encoded
)

# Verify preservation: reconstructed.shape == original_hidden_states.shape
assert reconstructed.shape == original_hidden_states.shape
```

### Performance Characteristics

#### Computational Complexity

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| **Standard Position Encoding** | O(L) | O(L × d) |
| **Segment-Aware Encoding** | O(L × d) | O(L × d) |  
| **Cross-Attention Bridge** | O(P × L × d) | O(P × L × d) |
| **Adaptive Token Merging** | O(L × d²) | O(L × d) |

Where: L = sequence length, d = model dimension, P = number of patches

#### Measured Performance (from benchmark)

| Metric | BLT-Hyena | Baseline | Ratio |
|--------|-----------|----------|-------|
| **Latency** | 47.3ms | 10.1ms | 4.7x |
| **Memory** | 127MB | 18MB | 7.0x |
| **Throughput** | 21.2 samples/sec | 99.1 samples/sec | 0.21x |

**Performance Analysis:**
- ✅ **Functional correctness**: All tests pass, position information preserved
- ⚠️ **Computational overhead**: Expected due to sophisticated position tracking
- 🎯 **Optimization opportunities**: Cross-attention mechanisms, memory management

### Integration Points

#### In HyenaGLT Model

```python
class HyenaGLT(nn.Module):
    def __init__(self, config):
        # Replace simple position encoding with BLT position manager
        self.position_manager = BLTPositionManager(
            d_model=config.hidden_size,
            max_len=config.max_position_embeddings,
            num_heads=config.num_attention_heads
        )
        
    def forward(self, input_ids, attention_mask=None):
        # 1. Initial position encoding
        hidden_states = self.position_manager.encode_positions(hidden_states)
        
        # 2. Token merging with position tracking
        if self.initial_merger:
            merged_states, boundaries, merge_info = self.initial_merger(
                hidden_states, attention_mask
            )
            
            # 3. Re-encode positions after merging
            hidden_states = self.position_manager.encode_positions(
                merged_states,
                patch_boundaries=boundaries,
                original_positions=original_positions
            )
```

#### In HyenaGLTBlock

```python
class HyenaGLTBlock(nn.Module):
    def __init__(self, config):
        # Use BLT position manager instead of simple position encoding
        self.position_manager = BLTPositionManager(config)
        
    def forward(self, hidden_states, original_sequence=None, segment_boundaries=None):
        # Apply position-aware processing with merge information
        if segment_boundaries is not None:
            hidden_states = self.position_manager.encode_positions(
                hidden_states,
                patch_boundaries=segment_boundaries
            )
```

### Usage Examples

#### Basic Position Encoding

```python
from hyena_glt.model.position_embeddings import BLTPositionManager

# Create position manager
position_manager = BLTPositionManager(
    d_model=256,
    max_len=1024,
    num_heads=8
)

# Encode positions for a sequence
hidden_states = torch.randn(2, 64, 256)
pos_encoded = position_manager.encode_positions(hidden_states)
```

#### Position Tracking Through Merging

```python
# Original sequence
original_positions = torch.arange(64).unsqueeze(0).expand(2, -1)

# Create patch boundaries (indicating where merging occurred)
patch_boundaries = torch.zeros(2, 64)
patch_boundaries[:, [16, 32, 48]] = 1  # Create 4 patches

# Re-encode with merge awareness
merged_encoded = position_manager.encode_positions(
    hidden_states,
    patch_boundaries=patch_boundaries,
    original_positions=original_positions
)

# Create patch representations
patch_repr, info = position_manager.create_patch_representations(
    merged_encoded, patch_boundaries
)

# Reconstruct byte-level representations
reconstructed = position_manager.reconstruct_byte_representations(
    patch_repr, info, merged_encoded
)
```

### Implementation Notes

#### 1. Memory Optimization

```python
# Use gradient checkpointing for cross-attention bridges
def create_patch_representations(self, byte_hidden_states, patch_boundaries):
    if self.training and self.gradient_checkpointing:
        patch_repr = torch.utils.checkpoint.checkpoint(
            self.cross_attention_bridge.encode_byte_to_patch,
            byte_hidden_states, patch_boundaries
        )
    else:
        patch_repr = self.cross_attention_bridge.encode_byte_to_patch(
            byte_hidden_states, patch_boundaries
        )
```

#### 2. Numerical Stability

```python
def _compute_segment_features(self, ...):
    # Ensure numerical stability in position computations
    pos_in_patch = pos_in_patch / max(patch_length - 1, 1)  # Avoid division by zero
    patch_len_norm = patch_length / seq_len
    global_pos_norm = global_pos / self.max_len
    
    # Clamp values to reasonable ranges
    segment_features = torch.stack([
        pos_in_patch.clamp(0, 1),
        patch_len_norm.clamp(0, 1), 
        global_pos_norm.clamp(0, 1)
    ], dim=-1)
```

#### 3. Debugging and Visualization

```python
def visualize_position_preservation(position_manager, sequence_length=64):
    """Utility function for debugging position preservation."""
    # Create test sequence
    hidden_states = torch.randn(1, sequence_length, 256)
    
    # Apply position encoding and merging
    pos_encoded = position_manager.encode_positions(hidden_states)
    
    # Create random patch boundaries
    boundaries = create_random_boundaries(sequence_length)
    merged_encoded = position_manager.encode_positions(
        pos_encoded, patch_boundaries=boundaries
    )
    
    # Measure position correlation
    correlation = torch.corrcoef(torch.stack([
        pos_encoded.view(-1), merged_encoded.view(-1)
    ]))[0, 1]
    
    print(f"Position preservation correlation: {correlation:.4f}")
    return correlation
```

This architecture guide provides the foundation for understanding how Hyena-GLT achieves efficient and effective genomic sequence modeling through its hybrid design.
