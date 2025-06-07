# Hyena-GLT API Reference

**Complete API Documentation for Hyena-GLT Framework**

[![API Version](https://img.shields.io/badge/API-v1.0.0-blue.svg)](https://hyena-glt.readthedocs.io/)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

---

## Table of Contents

1. [Core Model Classes](#core-model-classes)
2. [Position Embedding System](#position-embedding-system)
3. [Patcher Integration](#patcher-integration)
4. [Training Components](#training-components)
5. [Configuration Classes](#configuration-classes)
6. [Data Processing](#data-processing)
7. [Utilities](#utilities)

---

## Core Model Classes

### HyenaGLT

Main model class integrating all Hyena-GLT components.

```python
class HyenaGLT(nn.Module):
    """
    Main Hyena-GLT model combining BLT tokenization with Hyena convolutions
    for efficient genomic sequence modeling.
    
    Args:
        config (HyenaGLTConfig): Model configuration
        external_patcher (Optional[Patcher]): External patcher for advanced tokenization
        
    Attributes:
        config: Model configuration
        embeddings: Input embedding layer
        position_manager: BLT position management system
        initial_merger: Adaptive token merger
        layers: Stack of HyenaGLT blocks
        final_norm: Final layer normalization
    """
    
    def __init__(self, config: HyenaGLTConfig, external_patcher: Optional[Patcher] = None):
        super().__init__()
        # Implementation details in source code
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (Optional[torch.Tensor]): Attention mask [batch_size, seq_len]
            position_ids (Optional[torch.Tensor]): Position IDs [batch_size, seq_len]
            output_hidden_states (bool): Whether to return hidden states
            output_attentions (bool): Whether to return attention weights
            return_dict (bool): Whether to return BaseModelOutput
            
        Returns:
            Union[Tuple, BaseModelOutput]: Model outputs
        """
        
    def forward_with_patching(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        return_patches: bool = False,
    ) -> PatchedModelOutput:
        """
        Forward pass with external patcher integration.
        
        Args:
            sequence (str): Raw input sequence
            max_length (Optional[int]): Maximum sequence length
            return_patches (bool): Whether to return patch information
            
        Returns:
            PatchedModelOutput: Output with patch information
        """
```

### HyenaGLTBlock

Individual processing block within the model.

```python
class HyenaGLTBlock(nn.Module):
    """
    Complete Hyena-GLT processing block with cross-attention bridges
    and position-aware processing.
    
    Args:
        config (HyenaGLTConfig): Model configuration
        
    Attributes:
        hyena_operator: Core Hyena convolution operator
        cross_attention_bridge: Position information bridge
        feed_forward: Feed-forward network
        layer_norm: Layer normalization
    """
    
    def __init__(self, config: HyenaGLTConfig):
        super().__init__()
        # Implementation details
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the block.
        
        Args:
            hidden_states (torch.Tensor): Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask (Optional[torch.Tensor]): Attention mask
            position_info (Optional[Dict]): Position information from BLT system
            
        Returns:
            torch.Tensor: Processed hidden states
        """
```

### HyenaOperator

Core Hyena convolution operator optimized for genomic sequences.

```python
class HyenaOperator(nn.Module):
    """
    Hyena convolution operator optimized for genomic sequences.
    Features segment-aware processing and multi-channel filtering.
    
    Args:
        config (HyenaGLTConfig): Model configuration
        
    Attributes:
        short_conv: Short-range convolution
        filter_fn: Long-range filter function
        output_projection: Output projection layer
    """
    
    def __init__(self, config: HyenaGLTConfig):
        super().__init__()
        # Implementation details
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        segment_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Apply Hyena convolution with genomic awareness.
        
        Args:
            hidden_states (torch.Tensor): Input representations
            segment_info (Optional[Dict]): Segment boundary information
            
        Returns:
            torch.Tensor: Convolved representations
        """
```

---

## Position Embedding System

### BLTPositionManager

Manages the complete BLT position embedding system.

```python
class BLTPositionManager(nn.Module):
    """
    Manages the complete BLT position embedding system including
    segment-aware encoding and cross-attention bridges.
    
    Args:
        config (HyenaGLTConfig): Model configuration
        
    Attributes:
        segment_encoder: Segment-aware positional encoding
        cross_attention_bridge: Information flow bridge
        position_tracker: Tracks position through transformations
    """
    
    def __init__(self, config: HyenaGLTConfig):
        super().__init__()
        # Implementation details
        
    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_boundaries: torch.Tensor,
        return_position_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Apply BLT position encoding with patch awareness.
        
        Args:
            input_embeddings (torch.Tensor): Input embeddings [batch_size, seq_len, hidden_size]
            patch_boundaries (torch.Tensor): Patch boundary positions
            return_position_info (bool): Whether to return position tracking info
            
        Returns:
            Tuple[torch.Tensor, Optional[Dict]]: Position-encoded embeddings and info
        """
```

### SegmentAwarePositionalEncoding

Handles position encoding for variable-length patches.

```python
class SegmentAwarePositionalEncoding(nn.Module):
    """
    Handles position encoding for variable-length patches after token merging.
    
    Tracks three critical pieces of information:
    1. Global Position: Original absolute position before merging
    2. Patch Length: Number of tokens merged into current patch  
    3. Position in Patch: Relative position within merged patch (0.0 to 1.0)
    
    Args:
        config (HyenaGLTConfig): Model configuration
        
    Attributes:
        global_pos_encoding: Global position encoding
        patch_length_encoding: Patch length encoding
        relative_pos_encoding: Within-patch position encoding
    """
    
    def __init__(self, config: HyenaGLTConfig):
        super().__init__()
        # Implementation details
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        global_positions: torch.Tensor,
        patch_lengths: torch.Tensor,
        relative_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply segment-aware position encoding.
        
        Args:
            hidden_states (torch.Tensor): Input hidden states
            global_positions (torch.Tensor): Global position indices
            patch_lengths (torch.Tensor): Length of each patch
            relative_positions (torch.Tensor): Position within each patch
            
        Returns:
            torch.Tensor: Position-encoded hidden states
        """
```

### CrossAttentionPositionBridge

Implements U-shape information flow between byte and patch levels.

```python
class CrossAttentionPositionBridge(nn.Module):
    """
    Implements U-shape information flow: Byte ↔ Patch ↔ Byte
    
    Functions:
    - encode_byte_to_patch(): Aggregate byte-level info into patch representations
    - decode_patch_to_byte(): Reconstruct byte-level info from patch representations
    
    Args:
        config (HyenaGLTConfig): Model configuration
        
    Attributes:
        byte_to_patch_attention: Byte→Patch cross-attention
        patch_to_byte_attention: Patch→Byte cross-attention
        position_projection: Position information projection
    """
    
    def __init__(self, config: HyenaGLTConfig):
        super().__init__()
        # Implementation details
        
    def encode_byte_to_patch(
        self,
        byte_representations: torch.Tensor,
        patch_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate byte-level representations into patch-level.
        
        Args:
            byte_representations (torch.Tensor): Byte-level representations
            patch_boundaries (torch.Tensor): Patch boundary information
            
        Returns:
            torch.Tensor: Patch-level representations
        """
        
    def decode_patch_to_byte(
        self,
        patch_representations: torch.Tensor,
        target_byte_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct byte-level representations from patch-level.
        
        Args:
            patch_representations (torch.Tensor): Patch-level representations
            target_byte_positions (torch.Tensor): Target byte positions
            
        Returns:
            torch.Tensor: Reconstructed byte-level representations
        """
```

---

## Patcher Integration

### AdvancedPatcherConfig

Configuration for external patcher integration.

```python
@dataclass
class AdvancedPatcherConfig:
    """
    Configuration for advanced patcher integration with Hyena-GLT.
    
    Attributes:
        patching_mode (str): Patching strategy ('entropy', 'bpe', 'space', 'static', 'byte')
        threshold (float): Entropy threshold for patching (default: 1.335442066192627)
        threshold_add (float): Additional threshold for dual-threshold mode
        monotonicity (bool): Enforce monotonic patching constraints
        min_patch_length (int): Minimum patch size
        max_patch_length (int): Maximum patch size
        batch_size (int): Batch processing size
        no_cache (bool): Disable caching for memory optimization
        device (str): Processing device ('cpu', 'cuda')
    """
    
    patching_mode: str = 'entropy'
    threshold: float = 1.335442066192627
    threshold_add: float = 0.0
    monotonicity: bool = True
    min_patch_length: int = 1
    max_patch_length: int = 512
    batch_size: int = 32
    no_cache: bool = False
    device: str = 'auto'
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_modes = {'entropy', 'bpe', 'bpe_patcher', 'space', 'static', 'byte'}
        if self.patching_mode not in valid_modes:
            raise ValueError(f"Invalid patching_mode: {self.patching_mode}. "
                           f"Must be one of {valid_modes}")
        
        if self.min_patch_length < 1:
            raise ValueError("min_patch_length must be >= 1")
        
        if self.max_patch_length < self.min_patch_length:
            raise ValueError("max_patch_length must be >= min_patch_length")
```

### PatcherIntegrator

Integrates external patcher with Hyena-GLT model.

```python
class PatcherIntegrator:
    """
    Integrates external patcher functionality with Hyena-GLT model.
    
    Args:
        model (HyenaGLT): Hyena-GLT model instance
        patcher (Patcher): External patcher instance
        config (AdvancedPatcherConfig): Integration configuration
        
    Attributes:
        model: Hyena-GLT model
        patcher: External patcher
        config: Integration configuration
    """
    
    def __init__(
        self,
        model: HyenaGLT,
        patcher: 'Patcher',
        config: AdvancedPatcherConfig,
    ):
        self.model = model
        self.patcher = patcher
        self.config = config
        
    def process_with_patching(
        self,
        sequences: Union[str, List[str]],
        return_patch_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Process sequences with external patcher integration.
        
        Args:
            sequences (Union[str, List[str]]): Input sequences
            return_patch_info (bool): Whether to return patch information
            
        Returns:
            Union[torch.Tensor, Tuple]: Model outputs with optional patch info
        """
        
    def get_patch_statistics(self, sequences: List[str]) -> Dict[str, float]:
        """
        Analyze patching statistics for input sequences.
        
        Args:
            sequences (List[str]): Input sequences
            
        Returns:
            Dict[str, float]: Patching statistics
        """
```

---

## Training Components

### EnhancedTrainingPipeline

Complete training pipeline with genomic-specific optimizations.

```python
class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline with genomic-specific optimizations
    and advanced monitoring capabilities.
    
    Args:
        config (HyenaGLTConfig): Model configuration
        training_args (TrainingArguments): Training configuration
        
    Attributes:
        model: HyenaGLT model instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        metrics_tracker: Training metrics tracker
    """
    
    def __init__(
        self,
        config: HyenaGLTConfig,
        training_args: TrainingArguments,
    ):
        # Implementation details
        
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> TrainingResults:
        """
        Execute training loop with comprehensive monitoring.
        
        Args:
            train_dataloader (DataLoader): Training data loader
            val_dataloader (Optional[DataLoader]): Validation data loader
            callbacks (Optional[List[Callback]]): Training callbacks
            
        Returns:
            TrainingResults: Comprehensive training results
        """
        
    def evaluate(
        self,
        eval_dataloader: DataLoader,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            eval_dataloader (DataLoader): Evaluation data loader
            metrics (Optional[List[str]]): Metrics to compute
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
```

### TrainingArguments

Configuration for training process.

```python
@dataclass
class TrainingArguments:
    """
    Training configuration arguments.
    
    Attributes:
        output_dir (str): Output directory for checkpoints
        num_train_epochs (int): Number of training epochs
        per_device_train_batch_size (int): Training batch size per device
        per_device_eval_batch_size (int): Evaluation batch size per device
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        warmup_steps (int): Warmup steps
        logging_steps (int): Logging frequency
        save_steps (int): Checkpoint saving frequency
        eval_steps (int): Evaluation frequency
        mixed_precision (str): Mixed precision mode ('fp16', 'bf16', 'fp8')
        gradient_checkpointing (bool): Enable gradient checkpointing
        dataloader_num_workers (int): Number of data loader workers
        remove_unused_columns (bool): Remove unused columns from dataset
        report_to (List[str]): Experiment tracking services
    """
    
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = True
    report_to: List[str] = None
    
    def __post_init__(self):
        """Validate training arguments."""
        if self.report_to is None:
            self.report_to = []
```

---

## Configuration Classes

### HyenaGLTConfig

Main model configuration class.

```python
@dataclass
class HyenaGLTConfig:
    """
    Configuration class for HyenaGLT model.
    
    Attributes:
        # Model architecture
        genomic_vocab_size (int): Size of genomic vocabulary
        hidden_size (int): Hidden dimension size
        num_layers (int): Number of Hyena-GLT blocks
        num_attention_heads (int): Number of attention heads
        intermediate_size (int): Feed-forward intermediate size
        
        # Sequence processing
        max_position_embeddings (int): Maximum sequence length
        max_segment_length (int): Maximum patch length after merging
        
        # Hyena operator settings
        filter_size (int): Hyena filter dimension
        short_filter_order (int): Short convolution kernel size
        
        # Token merging
        merge_threshold (float): Entropy threshold for merging
        min_patch_size (int): Minimum tokens per patch
        max_patch_size (int): Maximum tokens per patch
        
        # External patcher integration
        use_external_patcher (bool): Whether to use external patcher
        patcher_threshold (float): External patcher threshold
        patcher_monotonicity (bool): External patcher monotonicity
        
        # Training settings
        dropout (float): Dropout probability  
        layer_norm_eps (float): Layer normalization epsilon
        gradient_checkpointing (bool): Memory optimization
        
        # Hardware optimization
        mixed_precision (str): Mixed precision mode
        compile_model (bool): Use torch.compile optimization
    """
    
    # Model architecture
    genomic_vocab_size: int = 1000
    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    
    # Sequence processing
    max_position_embeddings: int = 2048
    max_segment_length: int = 64
    
    # Hyena operator settings
    filter_size: int = 256
    short_filter_order: int = 3
    
    # Token merging
    merge_threshold: float = 0.1
    min_patch_size: int = 1
    max_patch_size: int = 16
    
    # External patcher integration
    use_external_patcher: bool = False
    patcher_threshold: float = 1.335442066192627
    patcher_monotonicity: bool = True
    
    # Training settings
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    gradient_checkpointing: bool = False
    
    # Hardware optimization
    mixed_precision: str = "fp16"
    compile_model: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        
        if self.max_patch_size < self.min_patch_size:
            raise ValueError("max_patch_size must be >= min_patch_size")
```

---

## Data Processing

### GenomicDataset

Dataset class for genomic sequences.

```python
class GenomicDataset(Dataset):
    """
    Dataset class for genomic sequences with support for various formats.
    
    Args:
        sequences (List[Dict]): List of sequence dictionaries
        tokenizer (GenomicTokenizer): Tokenizer instance
        max_length (int): Maximum sequence length
        return_tensors (str): Return tensor format
        
    Attributes:
        sequences: Input sequences
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        return_tensors: Return tensor format
    """
    
    def __init__(
        self,
        sequences: List[Dict[str, Any]],
        tokenizer: 'GenomicTokenizer',
        max_length: int = 512,
        return_tensors: str = 'pt',
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Item index
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized sequence data
        """
```

### GenomicTokenizer

Tokenizer for genomic sequences.

```python
class GenomicTokenizer:
    """
    Tokenizer for genomic sequences with k-mer support.
    
    Args:
        vocab_size (int): Vocabulary size
        kmer_size (int): K-mer size for tokenization
        special_tokens (Optional[Dict]): Special tokens dictionary
        
    Attributes:
        vocab_size: Vocabulary size
        kmer_size: K-mer size
        vocab: Vocabulary mapping
        special_tokens: Special tokens
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        kmer_size: int = 3,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        self.vocab_size = vocab_size
        self.kmer_size = kmer_size
        self.special_tokens = special_tokens or {}
        self._build_vocab()
        
    def encode(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict[str, torch.Tensor]]:
        """
        Encode genomic sequence to token IDs.
        
        Args:
            sequence (str): Input genomic sequence
            max_length (Optional[int]): Maximum sequence length
            padding (bool): Whether to pad sequences
            truncation (bool): Whether to truncate sequences
            return_tensors (Optional[str]): Return tensor format
            
        Returns:
            Union[List[int], Dict]: Encoded sequence
        """
        
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to genomic sequence.
        
        Args:
            token_ids (Union[List[int], torch.Tensor]): Token IDs to decode
            skip_special_tokens (bool): Whether to skip special tokens
            
        Returns:
            str: Decoded genomic sequence
        """
```

---

## Utilities

### create_genomic_dataloaders

Utility function to create genomic data loaders.

```python
def create_genomic_dataloaders(
    dataset: GenomicDataset,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train/validation/test data loaders from genomic dataset.
    
    Args:
        dataset (GenomicDataset): Input genomic dataset
        batch_size (int): Batch size for data loaders
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        shuffle (bool): Whether to shuffle training data
        num_workers (int): Number of data loader workers
        pin_memory (bool): Whether to use pinned memory
        
    Returns:
        Dict[str, DataLoader]: Dictionary containing train/val/test loaders
    """
```

### get_optimal_precision_config

Get optimal precision configuration for specific tasks.

```python
def get_optimal_precision_config(
    task_type: str,
    hardware: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get optimal precision configuration for genomic tasks.
    
    Args:
        task_type (str): Type of genomic task
        hardware (Optional[str]): Target hardware platform
        
    Returns:
        Dict[str, Any]: Optimal precision configuration
        
    Available task types:
        - 'genome_annotation': Genomic sequence annotation
        - 'variant_effect': Variant effect prediction  
        - 'protein_function': Protein function prediction
        - 'sequence_generation': Genomic sequence generation
        - 'motif_discovery': Regulatory motif discovery
    """
```

### model_summary

Generate detailed model summary.

```python
def model_summary(
    model: HyenaGLT,
    input_shape: Tuple[int, ...] = (1, 512),
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate comprehensive model summary with parameter counts and memory usage.
    
    Args:
        model (HyenaGLT): Model instance
        input_shape (Tuple[int, ...]): Input tensor shape
        device (str): Device for computation
        verbose (bool): Whether to print detailed summary
        
    Returns:
        Dict[str, Any]: Model summary statistics
    """
```

---

## Output Classes

### PatchedModelOutput

Output class for models with patching information.

```python
@dataclass
class PatchedModelOutput:
    """
    Output from models with external patcher integration.
    
    Attributes:
        last_hidden_state (torch.Tensor): Final hidden states
        hidden_states (Optional[Tuple[torch.Tensor]]): All hidden states
        attentions (Optional[Tuple[torch.Tensor]]): Attention weights
        patch_boundaries (torch.Tensor): Patch boundary positions
        patch_lengths (torch.Tensor): Length of each patch
        compression_ratio (float): Sequence compression ratio
        patch_info (Dict[str, Any]): Additional patch information
    """
    
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    patch_boundaries: torch.Tensor = None
    patch_lengths: torch.Tensor = None
    compression_ratio: float = 1.0
    patch_info: Dict[str, Any] = None
```

### TrainingResults

Results from training process.

```python
@dataclass  
class TrainingResults:
    """
    Comprehensive training results.
    
    Attributes:
        train_loss (List[float]): Training loss history
        eval_loss (List[float]): Evaluation loss history
        metrics (Dict[str, List[float]]): Training metrics history
        best_model_path (str): Path to best model checkpoint
        training_time (float): Total training time in seconds
        convergence_epoch (int): Epoch where convergence occurred
    """
    
    train_loss: List[float]
    eval_loss: List[float] 
    metrics: Dict[str, List[float]]
    best_model_path: str
    training_time: float
    convergence_epoch: int
```

---

## Error Classes

### HyenaGLTError

Base exception class for Hyena-GLT specific errors.

```python
class HyenaGLTError(Exception):
    """Base exception class for Hyena-GLT specific errors."""
    pass

class ConfigurationError(HyenaGLTError):
    """Raised when model configuration is invalid."""
    pass

class PatchingError(HyenaGLTError):
    """Raised when patcher integration fails."""
    pass

class TrainingError(HyenaGLTError):
    """Raised when training process encounters errors."""
    pass
```

---

## Version Information

```python
__version__ = "1.0.0"
__author__ = "Hyena-GLT Team"
__license__ = "MIT"
__email__ = "support@hyena-glt.com"
```

---

## Quick Reference

### Most Common Usage Patterns

```python
# 1. Basic model setup
from hyena_glt import HyenaGLT, HyenaGLTConfig

config = HyenaGLTConfig(genomic_vocab_size=256, hidden_size=512)  
model = HyenaGLT(config)

# 2. With external patcher
from bytelatent.data.patcher import Patcher

patcher = Patcher(patching_mode='entropy', threshold=1.335442066192627)
model = HyenaGLT(config, external_patcher=patcher)

# 3. Training setup
from hyena_glt.training import EnhancedTrainingPipeline, TrainingArguments

training_args = TrainingArguments(output_dir="./checkpoints")
trainer = EnhancedTrainingPipeline(config, training_args)

# 4. Data processing
from hyena_glt.data import GenomicDataset, GenomicTokenizer

tokenizer = GenomicTokenizer(vocab_size=1000, kmer_size=3)
dataset = GenomicDataset(sequences, tokenizer, max_length=512)
```

---

*This API reference is automatically generated from the source code. For the most up-to-date information, refer to the source code documentation and type hints.*
