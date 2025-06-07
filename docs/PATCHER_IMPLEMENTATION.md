# Patcher Implementation Guide

**Complete Tutorial for External Patcher Integration with BLT_Hyena**

[![Tutorial Level](https://img.shields.io/badge/level-intermediate-yellow.svg)](docs/README.md#recommended-reading-path)
[![Prerequisites](https://img.shields.io/badge/prereq-BLT_basics-blue.svg)](docs/BLT_POSITION_EMBEDDINGS.md)
[![Integration](https://img.shields.io/badge/integration-comprehensive-green.svg)](docs/INTEGRATION_GUIDE.md)

---

## 📚 Tutorial Navigation

**Prerequisites:**
- 📖 [Technical Guide](TECHNICAL_GUIDE.md) - Basic BLT_Hyena understanding
- 🎯 [Position Embeddings](BLT_POSITION_EMBEDDINGS.md) - Position system fundamentals
- 🏛️ [Architecture Guide](ARCHITECTURE.md) - Overall system architecture

**Related Guides:**
- 🚀 [Integration Guide](INTEGRATION_GUIDE.md) - Complete integration patterns
- 📊 [Performance Analysis](PERFORMANCE_ANALYSIS.md) - Benchmarking and optimization
- 🔗 [API Reference](API_REFERENCE.md) - Complete function documentation

**Learning Path:**
1. **Start Here** → This guide for patcher implementation details
2. **Next** → [Integration Guide](INTEGRATION_GUIDE.md) for practical usage patterns
3. **Advanced** → [Performance Analysis](PERFORMANCE_ANALYSIS.md) for optimization

---

## Overview

This tutorial provides a comprehensive walkthrough of the Patcher class implementation, teaching you how to integrate sophisticated entropy-based patching algorithms with BLT_Hyena components. 

### What You'll Learn

🎯 **6 Patching Modes**: Master entropy, BPE, space, static, byte, and BPE-patcher modes  
🧮 **Algorithm Details**: Understand entropy calculation, monotonicity, and dual-threshold systems  
🔧 **Implementation**: Build real patchers with actual parameter values  
⚡ **Optimization**: Apply performance tuning and memory management  
🧬 **Integration**: Connect patchers with BLT position embedding system  

### Tutorial Structure

**Part 1:** [Patcher Architecture](#patcher-class-architecture) - Learn the foundational components  
#### **2. Six Patching Modes** (Part 2): Master each patching algorithm

**Cross-References:**
- [Entropy Mode Details](PERFORMANCE_ANALYSIS.md#entropy-mode-benchmarks) - Performance comparison
- [BPE Mode Integration](INTEGRATION_GUIDE.md#bpe-patcher-setup) - Setup instructions  
- [Space Mode Applications](API_REFERENCE.md#space-mode-api) - API documentation
- [Static Mode Optimization](PERFORMANCE_ANALYSIS.md#static-mode-performance) - Speed analysis

> **💡 Learning Checkpoint**: After completing this section, you'll understand all six patching strategies. Consider testing each mode with the [Complete Integration Examples](INTEGRATION_GUIDE.md#complete-examples).

#### **3. Advanced Features** (Part 3): Explore constraints and batch processing

**Related Concepts:**
- [Memory Management](PERFORMANCE_ANALYSIS.md#memory-optimization) - Detailed optimization strategies
- [Batch Processing](INTEGRATION_GUIDE.md#batch-processing-patterns) - Production patterns
- [Validation Systems](API_REFERENCE.md#validation-classes) - Error checking reference

#### **4. BLT Integration** (Part 4): Connect with position embeddings  

**Essential Reading:**
- [Position Embedding System](BLT_POSITION_EMBEDDINGS.md) - Core BLT concepts ***(REQUIRED)***
- [Architecture Overview](ARCHITECTURE.md#blt-position-system) - System design
- [Cross-Attention Bridge](TECHNICAL_GUIDE.md#cross-attention-mechanisms) - Connection details

#### **5. Practical Usage** (Part 5): Apply in real scenarios

**Hands-On Resources:**
- [Integration Guide: Complete Examples](INTEGRATION_GUIDE.md#complete-examples) - Full implementations
- [Performance Tuning](PERFORMANCE_ANALYSIS.md#optimization-guide) - Production optimization
- [Troubleshooting Guide](INTEGRATION_GUIDE.md#troubleshooting) - Common issues and solutions

> **💡 Tutorial Tip**: Each section builds on the previous one. Code examples are designed to be run sequentially for hands-on learning.

### External Implementation Context

The Patcher class implements dynamic sequence segmentation using multiple strategies. While BLT_Hyena focuses on position embeddings and architectural components, the actual patching logic is implemented in external modules like the `bytelatent.data.patcher` module.

> **📋 See Also**: 
> - [Technical Guide: Advanced Patcher Integration](TECHNICAL_GUIDE.md#advanced-patcher-integration) for conceptual overview
> - [Architecture Guide: External Patcher Integration](ARCHITECTURE.md#external-patcher-integration) for system design
> - [Integration Guide](INTEGRATION_GUIDE.md) for complete implementation patterns
> - [Performance Analysis](PERFORMANCE_ANALYSIS.md#patcher-benchmarks) for performance comparison

## Patcher Class Architecture

### Core Components

```python
class PatcherArgs(BaseModel):
    patching_mode: PatchingModeEnum = PatchingModeEnum.entropy
    patching_device: str = "cuda"
    entropy_model_checkpoint_dir: str | None = None
    realtime_patching: bool = False
    threshold: float = 1.335442066192627  # Empirically determined threshold
    threshold_add: float | None = None    # Additional threshold for dual-threshold mode
    max_patch_length: int | None = None
    patch_size: float = 4.5              # Target patch size for static mode
    patching_batch_size: int = 1
    device: str = "cuda"
    monotonicity: bool = False           # Enable monotonicity constraints
    log_time: bool = False
```

## Patching Modes

The Patcher supports 6 distinct patching modes, each optimized for different use cases:

> **📖 Quick Reference**: For API details on each mode, see [API Reference: Patching Modes](API_REFERENCE.md#patching-modes)

### 1. Entropy Mode (`entropy`)

**Most sophisticated mode** - Uses neural network entropy calculations to determine patch boundaries.

> **🔗 Deep Dive**: For mathematical foundations, see [Position Embeddings: Entropy Analysis](BLT_POSITION_EMBEDDINGS.md#entropy-analysis)

**Key Parameters:**

- `threshold`: 1.335442066192627 (empirically optimized)
- `threshold_add`: Optional additional threshold for dual-threshold mode
- `monotonicity`: Enforces monotonic entropy increases

**Algorithm Flow:**

1. Calculate token-level entropy using entropy model
2. Apply threshold-based or monotonicity-based patch detection
3. Generate patch start masks
4. Convert masks to patch length tensors

> **⚡ Performance Note**: Entropy mode provides the best accuracy but requires GPU acceleration. See [Performance Analysis: Entropy Benchmarks](PERFORMANCE_ANALYSIS.md#entropy-mode-benchmarks) for detailed metrics.

**Entropy Calculation:**
```python
def entropy(scores):
    """
    Computes natural log entropy for each token
    scores: [bs, seq_len, vocab] -> returns [bs, seq_len]
    """
    log_probs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy
```

**Monotonicity Algorithm:**
```python
def patch_start_mask_from_entropy_with_monotonicity(entropies, t):
    """
    Creates patch boundaries only when entropy increases beyond threshold
    """
    differences = entropies[:, 1:] - entropies[:, :-1]
    condition = differences > t
    # First token always starts a patch
    mask[:, 0] = True
    mask[:, 1:] = condition
    return mask
```

**Dual-Threshold Algorithm:**
```python
def patch_start_mask_global_and_monotonicity(entropies, t, t_add=0):
    """
    Combines global threshold with monotonicity constraints
    - Global threshold: entropies > t
    - Monotonicity: differences > t_add
    """
    differences = entropies[:, 1:] - entropies[:, :-1]
    condition = (differences > t_add) & (entropies[:, 1:] > t) & (~mask[:, :-1])
    return mask
```

### 2. BPE Mode (`bpe`)
Uses BPE delimiter tokens to define patch boundaries.

**Implementation:**
```python
def find_bpe_delim_patch_start_ids(tokens, delim):
    """
    Finds patch starts at BPE delimiter positions
    delim: BPE_ID constant from tokenizer
    """
    ids = (tokens[:, :-1] == delim).nonzero(as_tuple=False)
    # Always start with [0, 1] for each batch
    out = [[0, 1] for _ in range(tokens.shape[0])]
    for x, y in ids:
        out[x.item()].append(y.item() + 1)  # Start after delimiter
```

### 3. BPE Patcher Mode (`bpe_patcher`)
Uses a trained neural network to predict BPE boundaries.

**Key Features:**
- Requires separate BPE prediction model
- Applies sliding window prediction
- Handles variable-length sequences

### 4. Space Mode (`space`)
Creates patches based on whitespace and punctuation boundaries.

**Character Classification:**
- Alphanumeric: `0-9`, `A-Z`, `a-z`
- Unicode: Handles UTF-8 byte boundaries
- Punctuation and whitespace trigger patch boundaries

**Implementation Logic:**
```python
def find_space_patch_start_ids(tokens):
    """
    Creates patches at word/token boundaries based on character types
    """
    tokens_no_offset = tokens - OFFSET
    patch_end_mask = (
        (tokens_no_offset < ord("0")) |
        ((ord("9") < tokens_no_offset) & (tokens_no_offset < ord("A"))) |
        ((ord("Z") < tokens_no_offset) & (tokens_no_offset < ord("a"))) |
        ((ord("z") < tokens_no_offset) & (tokens_no_offset < 0b1000_0000)) |
        (0b1100_0000 <= tokens_no_offset)
    )
```

### 5. Static Mode (`static`)
Fixed-size patches based on `patch_size` parameter.

**Characteristics:**
- Uniform patch lengths (except possibly the last patch)
- No entropy calculation required
- Fastest execution mode

### 6. Byte Mode (`byte`)
Single-token patches (every token is its own patch).

**Use Cases:**
- Character-level processing
- Maximum granularity requirements
- Debugging and analysis

## Advanced Features

### Patch Length Constraints

**Maximum Patch Length:**
```python
if self.max_patch_length is not None:
    patch_lengths = [
        split_large_numbers(pl, self.max_patch_length)
        for pl in patch_lengths.tolist()
    ]
```

The `split_large_numbers` function recursively splits patches exceeding the maximum length:
```python
def split_large_numbers(lst, m):
    """
    Splits patch lengths > m into multiple patches of size m
    with remainder as final patch
    """
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    return new_lst
```

### Batch Processing

**Entropy Model Batching:**
- Splits sequences into chunks based on `max_length` (default: 8192)
- Handles padding for incomplete batches
- Supports both CPU and GPU processing

**Memory Management:**
```python
batch_numel = max_length * patching_batch_size
splits = torch.split(tokens.flatten(), batch_numel)
for split in splits:
    pad_size = (max_length - (split.numel() % max_length)) % max_length
    # Process split with padding...
```

### Validation and Error Checking

**Critical Assertions:**
1. **Non-negative patch lengths:** `torch.all(patch_lengths >= 0)`
2. **No gaps in sequences:** `not check_non_zero_after_zero(patch_lengths)`
3. **Total length preservation:** `torch.sum(patch_lengths) == expected_total`

**Gap Detection:**
```python
def check_non_zero_after_zero(tensor):
    """
    Ensures no non-zero values follow zero values in patch length tensors
    This would indicate gaps in sequence coverage
    """
    zero_mask = tensor == 0
    shifted_mask = torch.cat([
        torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
        zero_mask[:, :-1],
    ], dim=1)
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()
```

## Integration with BLT_Hyena

The Patcher's output directly feeds into BLT_Hyena's position embedding system:

> **📋 Essential Background**: Before proceeding, ensure you understand [BLT Position Embeddings](BLT_POSITION_EMBEDDINGS.md) fundamentals.

### Position Embedding Integration

1. **Patch Lengths → Position Embeddings**
   - `patch_lengths` tensor provides segmentation information
   - BLT position embeddings adapt to variable patch sizes
   - Cross-attention mechanisms bridge patch boundaries

> **🔗 Implementation Details**: See [Architecture Guide: Position System Integration](ARCHITECTURE.md#blt-position-system) for complete architectural overview.

2. **Entropy Scores → Attention Weights**
   - Optional entropy scores can inform attention mechanisms
   - Higher entropy regions may receive different positional treatment

> **📊 Research Note**: For entropy's impact on attention patterns, see [Performance Analysis: Entropy-Attention Correlation](PERFORMANCE_ANALYSIS.md#entropy-attention-analysis).

3. **Dynamic Sequence Processing**
   - BLT_Hyena's `AdaptiveTokenMerger` can incorporate patch information
   - `CrossAttentionPositionBridge` handles position transitions between patches

> **💻 Code Examples**: Complete integration patterns available in [Integration Guide: BLT-Patcher Workflows](INTEGRATION_GUIDE.md#blt-patcher-workflows).

### Performance Considerations

**Timing Instrumentation:**

```python
if self.log_time:
    self.log["calculate_entropies"] += time.time() - s
    self.log["find_entropy_patch_start_ids"] += time.time() - s
    self.log["patch_lengths_from_start_ids"] += time.time() - s
    self.log["postprocessing_patch_lengths"] += time.time() - s
```

**Optimization Strategies:**

- Entropy model caching for repeated sequences
- Batch size tuning for memory/speed tradeoffs
- Device placement optimization (CPU vs GPU)

> **⚡ Performance Deep Dive**: For comprehensive optimization strategies, see [Performance Analysis: Integration Optimization](PERFORMANCE_ANALYSIS.md#integration-optimization).

## Usage Examples

> **🚀 Quick Start**: New to patchers? Start with [Integration Guide: Basic Setup](INTEGRATION_GUIDE.md#basic-setup) before trying these examples.

### Basic Entropy Patching

```python
patcher_args = PatcherArgs(
    patching_mode=PatchingModeEnum.entropy,
    threshold=1.335442066192627,
    device="cuda"
)
patcher = patcher_args.build()
patch_lengths, entropy_scores = patcher.patch(tokens)
```

> **📊 Performance Context**: This configuration achieves ~8.2x speedup. See [Performance Analysis: Entropy Mode Benchmarks](PERFORMANCE_ANALYSIS.md#entropy-mode-benchmarks) for detailed metrics.

### Monotonic Entropy Patching

```python
patcher_args = PatcherArgs(
    patching_mode=PatchingModeEnum.entropy,
    threshold=1.2,
    monotonicity=True,
    device="cuda"
)
patcher = patcher_args.build()
patch_lengths, _ = patcher.patch(tokens)
```

> **🔬 Algorithm Deep Dive**: For monotonicity constraint details, see [API Reference: Monotonicity Algorithm](API_REFERENCE.md#monotonicity-algorithm).

### Dual-Threshold Patching

```python
patcher_args = PatcherArgs(
    patching_mode=PatchingModeEnum.entropy,
    threshold=1.0,
    threshold_add=0.5,
    device="cuda"
)
patcher = patcher_args.build()
patch_lengths, _ = patcher.patch(tokens)
```

> **⚡ Advanced Usage**: Complete dual-threshold workflows available in [Integration Guide: Advanced Patterns](INTEGRATION_GUIDE.md#advanced-patterns).

### Production-Ready Configuration

```python
# Optimized for production use
patcher_args = PatcherArgs(
    patching_mode=PatchingModeEnum.entropy,
    threshold=1.335442066192627,
    patching_batch_size=4,
    max_patch_length=512,
    log_time=True,
    device="cuda"
)
patcher = patcher_args.build()

# Process with monitoring
with patcher.timing_context():
    patch_lengths, scores = patcher.patch(tokens)
    
# Log performance metrics
print(f"Processing time: {patcher.log}")
```

> **📈 Production Guide**: For complete production deployment patterns, see [Integration Guide: Production Guidelines](INTEGRATION_GUIDE.md#production-guidelines) and [Performance Analysis: Production Optimization](PERFORMANCE_ANALYSIS.md#production-optimization).

## Troubleshooting

> **🔧 Quick Help**: For immediate solutions, see [Integration Guide: Troubleshooting](INTEGRATION_GUIDE.md#troubleshooting) or [API Reference: Error Codes](API_REFERENCE.md#error-handling).

### Common Issues

1. **Memory Errors with Large Sequences**
   - Reduce `patching_batch_size`
   - Use CPU device for entropy model
   - Implement gradient checkpointing

> **📊 Memory Analysis**: For detailed memory usage patterns, see [Performance Analysis: Memory Optimization](PERFORMANCE_ANALYSIS.md#memory-management).

2. **Inconsistent Patch Lengths**
   - Verify `include_next_token` parameter
   - Check sequence padding alignment
   - Validate entropy model outputs

> **🔗 Validation Details**: Complete validation procedures in [API Reference: Validation Classes](API_REFERENCE.md#validation-classes).

3. **Performance Bottlenecks**
   - Profile entropy calculation vs other modes
   - Consider static mode for uniform sequences
   - Optimize device placement

> **⚡ Performance Tuning**: Comprehensive optimization guide available in [Performance Analysis: Optimization Strategies](PERFORMANCE_ANALYSIS.md#optimization-guide).

### Debugging Tools

```python
# Enable timing logs
patcher_args.log_time = True
patch_lengths, scores = patcher.patch(tokens)
print(patcher.log)  # View timing breakdown

# Validate patch coverage
total_tokens = tokens.numel()
total_patches = patch_lengths.sum().item()
assert total_tokens == total_patches, f"Coverage mismatch: {total_tokens} vs {total_patches}"
```

> **🛠️ Advanced Debugging**: For production debugging techniques, see [Integration Guide: Debugging Workflows](INTEGRATION_GUIDE.md#debugging-workflows).

## Future Enhancements

> **🚀 Development Roadmap**: For planned features and contribution opportunities, see [Integration Guide: Development Roadmap](INTEGRATION_GUIDE.md#development-roadmap).

### Potential Improvements

1. **Adaptive Thresholding**
   - Dynamic threshold adjustment based on sequence statistics
   - Per-domain threshold optimization

> **📋 Research Context**: For theoretical foundations, see [Performance Analysis: Adaptive Algorithms](PERFORMANCE_ANALYSIS.md#adaptive-algorithms).

2. **Hierarchical Patching**
   - Multi-level patch boundaries
   - Coarse-to-fine processing strategies

> **🏛️ Architecture Implications**: System design considerations in [Architecture Guide: Hierarchical Processing](ARCHITECTURE.md#hierarchical-processing).

3. **Attention-Guided Patching**
   - Use attention patterns to inform patch boundaries
   - Feedback loop between model attention and patching decisions

> **🔬 Advanced Concepts**: For attention-patcher integration, see [Technical Guide: Attention Systems](TECHNICAL_GUIDE.md#attention-mechanisms).

4. **Streaming Patching**
   - Online patching for real-time applications
   - Incremental patch boundary updates

> **⚡ Real-time Applications**: Implementation patterns in [Integration Guide: Streaming Processing](INTEGRATION_GUIDE.md#streaming-processing).

---

## 📚 Learning Path Completion

Congratulations! You've completed the comprehensive Patcher Implementation tutorial. Here's what you've learned:

✅ **Architecture Understanding** - Patcher class structure and 6 patching modes  
✅ **Algorithm Mastery** - Entropy, BPE, space, static, byte, and BPE-patcher modes  
✅ **Advanced Features** - Constraints, batch processing, and validation systems  
✅ **BLT Integration** - Position embedding system connection  
✅ **Practical Application** - Real-world usage patterns and optimization  

### Next Steps

**For Implementation:**
1. 🚀 **Apply Knowledge** → [Integration Guide: Complete Examples](INTEGRATION_GUIDE.md#complete-examples)
2. ⚡ **Optimize Performance** → [Performance Analysis: Optimization Guide](PERFORMANCE_ANALYSIS.md#optimization-guide)
3. 🔧 **Troubleshoot Issues** → [Integration Guide: Debugging Workflows](INTEGRATION_GUIDE.md#debugging-workflows)

**For Advanced Learning:**
1. 🏛️ **Architecture Deep Dive** → [Architecture Guide](ARCHITECTURE.md)
2. 📊 **Performance Analysis** → [Performance Analysis: Complete Guide](PERFORMANCE_ANALYSIS.md)
3. 🔗 **API Mastery** → [API Reference: Complete Documentation](API_REFERENCE.md)

**For Production Use:**
1. 📋 **Best Practices** → [Integration Guide: Production Guidelines](INTEGRATION_GUIDE.md#production-guidelines)
2. 🔍 **Monitoring Setup** → [Performance Analysis: Monitoring](PERFORMANCE_ANALYSIS.md#monitoring)
3. 📈 **Scaling Strategies** → [Integration Guide: Scaling Patterns](INTEGRATION_GUIDE.md#scaling-patterns)

---

This implementation guide provides the detailed technical foundation missing from the high-level BLT_Hyena documentation, bridging the gap between architectural concepts and practical implementation details.

## 📖 Complete Cross-Reference Index

### Core Documentation
- **[Technical Guide](TECHNICAL_GUIDE.md)** - High-level BLT_Hyena overview and conceptual framework
- **[Architecture Guide](ARCHITECTURE.md)** - System architecture and component relationships  
- **[BLT Position Embeddings](BLT_POSITION_EMBEDDINGS.md)** - Core position embedding system details

### Implementation Resources
- **[API Reference](API_REFERENCE.md)** - Complete function signatures and class documentation
- **[Integration Guide](INTEGRATION_GUIDE.md)** - Step-by-step integration patterns and examples
- **[Performance Analysis](PERFORMANCE_ANALYSIS.md)** - Benchmarking, optimization, and scaling guidance

### Quick Access
- **[Main README](../README.md)** - Project overview and getting started
- **[Documentation Index](README.md)** - Complete documentation navigation
- **[Examples](EXAMPLES.md)** - Quick implementation examples

### External References
- **External Patcher Implementation** - `/Users/sanjeevadodlapati/Downloads/Repos/blt_tutorial/bytelatent/data/patcher.py`
- **BLT Tutorial Repository** - `/Users/sanjeevadodlapati/Downloads/Repos/blt_tutorial/` for hands-on examples
- **Original BLT Paper** - [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2309.00268)

> **💡 Tutorial Navigation Tip**: Use your browser's search (Ctrl/Cmd+F) to quickly find specific concepts across all documentation files. All cross-references are designed to work together as a comprehensive learning system.
