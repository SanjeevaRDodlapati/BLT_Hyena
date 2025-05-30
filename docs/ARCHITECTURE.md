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

## Implementation Notes

### 1. Numerical Stability

```python
class StableHyenaOperator(HyenaOperator):
    def fft_conv(self, u, filter_coeffs):
        # Ensure numerical stability in FFT operations
        u_fft = torch.fft.rfft(u, n=self.l_max, dim=1)
        filter_fft = torch.fft.rfft(filter_coeffs, n=self.l_max, dim=-1)
        
        # Avoid overflow in multiplication
        result_fft = u_fft * filter_fft.clamp(min=-1e6, max=1e6)
        
        # Convert back to time domain
        result = torch.fft.irfft(result_fft, n=self.l_max, dim=1)
        return result[:, :u.size(1)]  # Trim to original length
```

### 2. Gradient Flow

```python
class ResidualHyenaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hyena_op = HyenaOperator(config.d_model, config.sequence_length)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # Pre-norm residual connections for better gradient flow
        residual = x
        x = self.norm1(x)
        x = self.hyena_op(x)
        x = self.dropout(x) + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual
        
        return x
```

This architecture guide provides the foundation for understanding how Hyena-GLT achieves efficient and effective genomic sequence modeling through its hybrid design.
