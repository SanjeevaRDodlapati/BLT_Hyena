# 📊 Data Pipeline: Processing Real Genomic Data

**Goal**: Learn to build robust data pipelines for genomic sequences at scale.

**Time**: ~50 minutes  
**Prerequisites**: [Fundamentals](01_FUNDAMENTALS.md) completed

---

## 🎯 What You'll Build

By the end of this tutorial, you'll have a complete data pipeline that can:
- ✅ Load FASTA files and genomic databases
- ✅ Preprocess sequences (quality control, filtering, augmentation)
- ✅ Create efficient datasets and dataloaders
- ✅ Handle different sequence types (DNA, RNA, protein)
- ✅ Scale to large datasets with smart caching

---

## 📁 Working with Real Data

### 1. FASTA File Processing

Let's start with the most common genomic data format:

```python
from hyena_glt.data import GenomicDataset, GenomicTokenizer
from pathlib import Path
import torch

# First, let's create some sample FASTA data
sample_fasta = """
>sequence_1 Human chromosome 1 fragment
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
>sequence_2 Human chromosome 2 fragment  
TTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAACCGG
AATTCCGGAATTCCGGAATTCCGGAATTCCGGAATTCCGGAATTCCGGAATTCCGG
>sequence_3 Mitochondrial DNA segment
CGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA
TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
"""

# Save sample data
sample_file = Path("sample_genomic_data.fasta")
with open(sample_file, "w") as f:
    f.write(sample_fasta)

print("📄 Sample FASTA file created")
```

### 2. Loading and Parsing

```python
def parse_fasta_file(file_path):
    """Parse a FASTA file and return sequences with metadata."""
    sequences = []
    labels = []
    metadata = []
    
    current_seq = ""
    current_header = ""
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_seq:
                    sequences.append(current_seq)
                    metadata.append(current_header)
                    
                    # Extract label from header (example: chromosome number)
                    if "chromosome 1" in current_header.lower():
                        labels.append(0)
                    elif "chromosome 2" in current_header.lower():
                        labels.append(1)
                    else:
                        labels.append(2)  # Other (mitochondrial, etc.)
                
                # Start new sequence
                current_header = line[1:]  # Remove '>'
                current_seq = ""
            else:
                current_seq += line
        
        # Don't forget the last sequence
        if current_seq:
            sequences.append(current_seq)
            metadata.append(current_header)
            if "chromosome 1" in current_header.lower():
                labels.append(0)
            elif "chromosome 2" in current_header.lower():
                labels.append(1)
            else:
                labels.append(2)
    
    return sequences, labels, metadata

# Parse our sample file
sequences, labels, metadata = parse_fasta_file(sample_file)

print(f"📊 Loaded {len(sequences)} sequences")
print(f"📏 Sequence lengths: {[len(seq) for seq in sequences]}")
print(f"🏷️ Labels: {labels}")
```

---

## 🔧 Data Preprocessing Pipeline

### 1. Quality Control and Filtering

```python
def preprocess_sequences(sequences, labels, metadata, min_length=50, max_length=5000):
    """Apply quality control and filtering to genomic sequences."""
    
    processed_sequences = []
    processed_labels = []
    processed_metadata = []
    
    for seq, label, meta in zip(sequences, labels, metadata):
        # Remove any non-standard nucleotides
        clean_seq = ''.join(c for c in seq.upper() if c in 'ATCG')
        
        # Length filtering
        if len(clean_seq) < min_length or len(clean_seq) > max_length:
            continue
            
        # Quality checks
        gc_content = (clean_seq.count('G') + clean_seq.count('C')) / len(clean_seq)
        if gc_content < 0.1 or gc_content > 0.9:  # Extreme GC content
            continue
            
        processed_sequences.append(clean_seq)
        processed_labels.append(label)
        processed_metadata.append(meta)
    
    return processed_sequences, processed_labels, processed_metadata

# Apply preprocessing
clean_sequences, clean_labels, clean_metadata = preprocess_sequences(
    sequences, labels, metadata,
    min_length=50,
    max_length=1000
)

print(f"✅ After preprocessing: {len(clean_sequences)} sequences")
for i, seq in enumerate(clean_sequences):
    gc_content = (seq.count('G') + seq.count('C')) / len(seq)
    print(f"Sequence {i+1}: Length={len(seq)}, GC={gc_content:.2f}, Label={clean_labels[i]}")
```

### 2. Data Augmentation

```python
import random

def augment_dna_sequence(sequence, augmentation_prob=0.3):
    """Apply data augmentation to DNA sequences."""
    
    augmented_sequences = [sequence]  # Always include original
    
    # Reverse complement
    if random.random() < augmentation_prob:
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        reverse_complement = ''.join(complement_map.get(base, base) for base in sequence[::-1])
        augmented_sequences.append(reverse_complement)
    
    # Random mutations (very small probability)
    if random.random() < augmentation_prob * 0.1:  # 3% chance
        mutated = list(sequence)
        mutation_positions = random.sample(range(len(sequence)), min(2, len(sequence)//100))
        bases = ['A', 'T', 'G', 'C']
        
        for pos in mutation_positions:
            original_base = mutated[pos]
            new_base = random.choice([b for b in bases if b != original_base])
            mutated[pos] = new_base
        
        augmented_sequences.append(''.join(mutated))
    
    # Subsequence extraction
    if random.random() < augmentation_prob and len(sequence) > 100:
        start = random.randint(0, len(sequence) // 4)
        end = random.randint(3 * len(sequence) // 4, len(sequence))
        subsequence = sequence[start:end]
        augmented_sequences.append(subsequence)
    
    return augmented_sequences

# Apply augmentation
augmented_data = []
augmented_labels = []

for seq, label in zip(clean_sequences, clean_labels):
    aug_seqs = augment_dna_sequence(seq)
    augmented_data.extend(aug_seqs)
    augmented_labels.extend([label] * len(aug_seqs))

print(f"🔄 After augmentation: {len(augmented_data)} sequences")
```

---

## 📦 Creating Datasets

### 1. Basic Dataset Creation

```python
from hyena_glt.data import GenomicDataset
from hyena_glt import GenomicTokenizer

# Initialize tokenizer
tokenizer = GenomicTokenizer(
    sequence_type="dna",
    vocab_size=4096,
    kmer_size=3,
    max_length=1024
)

# Create dataset
dataset = GenomicDataset(
    sequences=augmented_data,
    labels=augmented_labels,
    tokenizer=tokenizer,
    max_length=512,
    return_attention_mask=True,
    return_special_tokens_mask=False
)

print(f"📦 Dataset created with {len(dataset)} samples")

# Test dataset
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Input shape: {sample['input_ids'].shape}")
print(f"Label: {sample['labels']}")
```

### 2. Advanced Dataset with Caching

```python
import os
import pickle
from torch.utils.data import Dataset

class CachedGenomicDataset(Dataset):
    """Genomic dataset with intelligent caching for large-scale data."""
    
    def __init__(self, data_dir, tokenizer, max_length=1024, cache_dir="./cache"):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load or create file index
        self.file_index = self._build_file_index()
        
    def _build_file_index(self):
        """Build index of all FASTA files and their sequences."""
        index_file = self.cache_dir / "file_index.pkl"
        
        if index_file.exists():
            with open(index_file, 'rb') as f:
                return pickle.load(f)
        
        index = []
        fasta_files = list(self.data_dir.glob("*.fasta")) + list(self.data_dir.glob("*.fa"))
        
        for file_path in fasta_files:
            sequences, labels, metadata = parse_fasta_file(file_path)
            for i, (seq, label, meta) in enumerate(zip(sequences, labels, metadata)):
                index.append({
                    'file': str(file_path),
                    'sequence_id': i,
                    'length': len(seq),
                    'label': label,
                    'metadata': meta
                })
        
        # Save index
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)
        
        return index
    
    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):
        item = self.file_index[idx]
        
        # Check cache first
        cache_file = self.cache_dir / f"item_{idx}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load and process sequence
        sequences, labels, _ = parse_fasta_file(item['file'])
        sequence = sequences[item['sequence_id']]
        label = labels[item['sequence_id']]
        
        # Tokenize
        tokenized = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        result = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        # Cache result
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result

print("💾 Cached dataset class defined")
```

---

## 🚀 DataLoaders and Batching

### 1. Basic DataLoader

```python
from torch.utils.data import DataLoader

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

# Test the dataloader
print("🔄 Testing dataloader...")
for i, batch in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    
    if i >= 2:  # Just test first 3 batches
        break

print("✅ DataLoader working correctly!")
```

### 2. Custom Collate Function

```python
def genomic_collate_fn(batch):
    """Custom collate function for variable-length genomic sequences."""
    
    # Separate components
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Find max length in batch
    max_length = max(ids.size(0) for ids in input_ids)
    
    # Pad sequences
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        # Pad to max length
        padding_length = max_length - ids.size(0)
        
        padded_ids = torch.cat([ids, torch.zeros(padding_length, dtype=ids.dtype)])
        padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)])
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_masks),
        'labels': torch.stack(labels)
    }

# Create dataloader with custom collate function
custom_dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=genomic_collate_fn,
    num_workers=1
)

print("🎯 Custom dataloader created")
```

---

## 📊 Multi-Dataset Handling

### 1. Combining Multiple Data Sources

```python
from torch.utils.data import ConcatDataset, WeightedRandomSampler

# Create datasets for different sequence types
dna_sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "TTAACCGGTTAA"]
rna_sequences = ["AUCGAUCGAUCG", "GCUAGCUAGCUA", "UUAACCGGUUAA"]  # RNA has U instead of T
protein_sequences = ["MKTLLLTLLVVATIAIAVLQANLLRPGDRSSF", "ARTKQTARKSTGGKAPRKQL"]

# Create separate datasets
dna_tokenizer = GenomicTokenizer(sequence_type="dna", vocab_size=4096)
rna_tokenizer = GenomicTokenizer(sequence_type="rna", vocab_size=4096)
protein_tokenizer = GenomicTokenizer(sequence_type="protein", vocab_size=8192)

dna_dataset = GenomicDataset(
    sequences=dna_sequences,
    labels=[0] * len(dna_sequences),  # DNA = 0
    tokenizer=dna_tokenizer
)

rna_dataset = GenomicDataset(
    sequences=rna_sequences,
    labels=[1] * len(rna_sequences),  # RNA = 1
    tokenizer=rna_tokenizer
)

protein_dataset = GenomicDataset(
    sequences=protein_sequences,
    labels=[2] * len(protein_sequences),  # Protein = 2
    tokenizer=protein_tokenizer
)

# Combine datasets
combined_dataset = ConcatDataset([dna_dataset, rna_dataset, protein_dataset])

print(f"📊 Combined dataset size: {len(combined_dataset)}")
```

### 2. Weighted Sampling

```python
# Create weights for balanced sampling
dataset_sizes = [len(dna_dataset), len(rna_dataset), len(protein_dataset)]
weights = []

for size in dataset_sizes:
    # Inverse weight to balance datasets
    weight = 1.0 / size
    weights.extend([weight] * size)

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(combined_dataset),
    replacement=True
)

balanced_dataloader = DataLoader(
    combined_dataset,
    batch_size=6,
    sampler=sampler,
    num_workers=1
)

print("⚖️ Balanced dataloader created")

# Test balanced sampling
label_counts = {0: 0, 1: 0, 2: 0}
for i, batch in enumerate(balanced_dataloader):
    for label in batch['labels']:
        label_counts[label.item()] += 1
    
    if i >= 10:  # Test 10 batches
        break

print(f"📊 Label distribution in 10 batches: {label_counts}")
```

---

## 🎛️ Data Pipeline Configuration

### 1. Configuration Management

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataPipelineConfig:
    """Configuration for genomic data pipeline."""
    
    # Data sources
    data_dir: str = "./data"
    file_patterns: List[str] = None
    
    # Preprocessing
    min_sequence_length: int = 50
    max_sequence_length: int = 2048
    min_gc_content: float = 0.1
    max_gc_content: float = 0.9
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.3
    include_reverse_complement: bool = True
    
    # Tokenization
    sequence_type: str = "dna"
    vocab_size: int = 4096
    kmer_size: int = 3
    max_tokenized_length: int = 1024
    
    # DataLoader
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    
    # Caching
    use_cache: bool = True
    cache_dir: str = "./cache"
    
    def __post_init__(self):
        if self.file_patterns is None:
            self.file_patterns = ["*.fasta", "*.fa", "*.fas"]

# Create configuration
config = DataPipelineConfig(
    data_dir="./genomic_data",
    max_sequence_length=1000,
    batch_size=16,
    use_augmentation=True
)

print("⚙️ Data pipeline configuration created")
print(f"Batch size: {config.batch_size}")
print(f"Max sequence length: {config.max_sequence_length}")
```

### 2. Complete Pipeline Class

```python
class GenomicDataPipeline:
    """Complete genomic data processing pipeline."""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
    
    def setup(self):
        """Setup the complete pipeline."""
        print("🔧 Setting up genomic data pipeline...")
        
        # Initialize tokenizer
        self.tokenizer = GenomicTokenizer(
            sequence_type=self.config.sequence_type,
            vocab_size=self.config.vocab_size,
            kmer_size=self.config.kmer_size,
            max_length=self.config.max_tokenized_length
        )
        
        # Load and process data
        sequences, labels = self._load_data()
        
        # Create dataset
        if self.config.use_cache:
            self.dataset = CachedGenomicDataset(
                data_dir=self.config.data_dir,
                tokenizer=self.tokenizer,
                max_length=self.config.max_tokenized_length,
                cache_dir=self.config.cache_dir
            )
        else:
            self.dataset = GenomicDataset(
                sequences=sequences,
                labels=labels,
                tokenizer=self.tokenizer,
                max_length=self.config.max_tokenized_length
            )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        print(f"✅ Pipeline ready with {len(self.dataset)} samples")
    
    def _load_data(self):
        """Load data from configured sources."""
        # For demo, use our existing data
        return augmented_data, augmented_labels
    
    def get_dataloader(self):
        """Get the configured dataloader."""
        if self.dataloader is None:
            self.setup()
        return self.dataloader
    
    def get_sample_batch(self):
        """Get a sample batch for testing."""
        dataloader = self.get_dataloader()
        return next(iter(dataloader))

# Test the complete pipeline
pipeline = GenomicDataPipeline(config)
pipeline.setup()

sample_batch = pipeline.get_sample_batch()
print(f"📦 Sample batch shape: {sample_batch['input_ids'].shape}")
```

---

## 🧪 Testing Your Pipeline

```python
def test_data_pipeline(pipeline):
    """Comprehensive testing of the data pipeline."""
    
    print("🧪 Testing data pipeline...")
    
    # Test 1: Basic functionality
    dataloader = pipeline.get_dataloader()
    batch = next(iter(dataloader))
    
    assert 'input_ids' in batch, "Missing input_ids"
    assert 'attention_mask' in batch, "Missing attention_mask"
    assert 'labels' in batch, "Missing labels"
    
    print("✅ Test 1 passed: Basic functionality")
    
    # Test 2: Batch consistency
    batch_size = batch['input_ids'].shape[0]
    assert batch['attention_mask'].shape[0] == batch_size, "Inconsistent batch size"
    assert batch['labels'].shape[0] == batch_size, "Inconsistent batch size"
    
    print("✅ Test 2 passed: Batch consistency")
    
    # Test 3: Data types
    assert batch['input_ids'].dtype == torch.long, "Wrong input_ids dtype"
    assert batch['attention_mask'].dtype in [torch.long, torch.bool], "Wrong attention_mask dtype"
    assert batch['labels'].dtype == torch.long, "Wrong labels dtype"
    
    print("✅ Test 3 passed: Data types")
    
    # Test 4: Value ranges
    assert batch['input_ids'].min() >= 0, "Negative token IDs"
    assert batch['input_ids'].max() < pipeline.tokenizer.vocab_size, "Token ID out of range"
    
    print("✅ Test 4 passed: Value ranges")
    
    # Test 5: Multiple batches
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        if batch_count >= 5:  # Test 5 batches
            break
    
    print(f"✅ Test 5 passed: Successfully processed {batch_count} batches")
    
    print("🎉 All tests passed! Pipeline is ready for training.")

# Run tests
test_data_pipeline(pipeline)
```

---

## 🎯 Key Takeaways

You now know how to:

✅ **Parse FASTA Files**: Load and process real genomic data formats  
✅ **Quality Control**: Filter and clean sequences for training  
✅ **Data Augmentation**: Increase dataset diversity with biological augmentations  
✅ **Efficient Datasets**: Create scalable datasets with caching  
✅ **Custom DataLoaders**: Handle variable-length sequences and batching  
✅ **Multi-Source Data**: Combine different data types and sources  
✅ **Pipeline Testing**: Validate your data pipeline thoroughly  

---

## 🚀 What's Next?

Your data pipeline is ready! Now choose your path:

### 🏋️ **Start Training**
→ [04_TRAINING.md](04_TRAINING.md) - Use your pipeline to train BLT_Hyena models with proper optimization

### 📊 **Model Evaluation**  
→ [05_EVALUATION.md](05_EVALUATION.md) - Evaluate model performance on genomic tasks

### 🔬 **Advanced Preprocessing**
→ [07_ADVANCED.md](07_ADVANCED.md) - Advanced data preprocessing, multi-modal data, and custom tokenizers

### 🔧 **Hyena Deep Dive**
→ [02_HYENA_INTEGRATION.md](02_HYENA_INTEGRATION.md) - Understand the Hyena operators powering your model

---

## 📝 Production Tips

**For Large Datasets**:
- Use `CachedGenomicDataset` with SSD storage
- Set `num_workers=4-8` for faster loading  
- Enable `pin_memory=True` for GPU training

**Memory Optimization**:
- Use smaller batch sizes and gradient accumulation
- Implement sequence length bucketing
- Consider streaming datasets for massive files

**Quality Control**:
- Always validate GC content and sequence quality
- Remove sequences with excessive N's or unusual patterns
- Balance dataset composition across different sources

---

*Next up: [Training Your Models →](04_TRAINING.md)*
