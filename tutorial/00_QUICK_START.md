# 🚀 Quick Start: Your First BLT_Hyena Model in 5 Minutes

**Goal**: Get a working BLT_Hyena model running immediately to see the framework in action.

**Time**: ~5 minutes  
**Prerequisites**: Python 3.8+, PyTorch installed

---

## ⚡ Instant Setup

### 1. Import the Essentials

```python
import torch
from hyena_glt import HyenaGLT, HyenaGLTConfig, GenomicTokenizer, GenomicDataset

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 2. Create Your First Model

```python
# Create a small, fast config for testing
config = HyenaGLTConfig(
    # Small model for quick testing
    hidden_size=256,
    num_layers=4,
    num_attention_heads=8,
    
    # Genomic-specific settings
    sequence_type="dna",
    genomic_vocab_size=4096,
    max_position_embeddings=1024,
    
    # Fast training settings
    dropout=0.1,
    local_encoder_layers=1,
    local_decoder_layers=1,
    dynamic_patching=True,
    patch_size=4
)

# Initialize the model
model = HyenaGLT(config).to(device)

print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 3. Prepare Some Test Data

```python
# Create sample genomic sequences
dna_sequences = [
    "ATCGATCGATCGATCGATCG",
    "GCTAGCTAGCTAGCTAGCTA", 
    "TTAACCGGTTAACCGGTTAA",
    "CGTACGTACGTACGTACGTA",
    "AAAAGGGGCCCCTTTTAAAA",
    "GCTAAGCTGCTAAGCTGCTA"
]

# Create labels (0=normal, 1=variant)
labels = [0, 1, 0, 1, 1, 0]

print(f"📊 Test data: {len(dna_sequences)} sequences")
```

### 4. Tokenize the Data

```python
# Initialize the genomic tokenizer
tokenizer = GenomicTokenizer(
    sequence_type="dna",
    vocab_size=config.genomic_vocab_size,
    max_length=config.max_position_embeddings
)

# Tokenize sequences
tokenized = tokenizer(
    dna_sequences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(f"📝 Tokenized shape: {tokenized['input_ids'].shape}")
```

### 5. Run Your First Forward Pass

```python
# Move data to device
input_ids = tokenized['input_ids'].to(device)
attention_mask = tokenized['attention_mask'].to(device)

# Run the model
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )

print(f"🎯 Output shape: {outputs['last_hidden_state'].shape}")
print(f"✅ Forward pass successful!")
```

---

## 🧪 Quick Training Example

Let's train the model for just a few steps to see the training loop:

```python
from torch.utils.data import DataLoader
from hyena_glt.training import TrainingConfig

# Create dataset
dataset = GenomicDataset(
    sequences=dna_sequences,
    labels=labels,
    tokenizer=tokenizer,
    max_length=512
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Simple training setup
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

print("🚀 Training for 3 steps...")

for step, batch in enumerate(dataloader):
    if step >= 3:  # Just 3 quick steps
        break
        
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels_tensor = batch['labels'].to(device)
    
    # Forward pass
    outputs = model(input_ids, attention_mask)
    
    # Simple classification head (normally you'd use HyenaGLTForSequenceClassification)
    logits = outputs['last_hidden_state'].mean(dim=1)  # Simple pooling
    
    # Compute loss (simplified)
    if logits.size(-1) != len(set(labels)):
        # Add a projection layer for proper classification
        proj = torch.nn.Linear(logits.size(-1), len(set(labels))).to(device)
        logits = proj(logits)
    
    loss = criterion(logits, labels_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Step {step + 1}: Loss = {loss.item():.4f}")

print("✅ Quick training complete!")
```

---

## 🎉 Congratulations!

You've successfully:
- ✅ Created a BLT_Hyena model
- ✅ Tokenized genomic sequences  
- ✅ Run forward inference
- ✅ Executed a basic training loop

**Your model is working!** 🎊

---

## 🚀 What's Next?

Now that you've seen BLT_Hyena in action, choose your learning path:

### 📚 **Learn the Fundamentals** 
→ [01_FUNDAMENTALS.md](01_FUNDAMENTALS.md) - Understand BLT architecture, Hyena operators, and core concepts

### 🔧 **Build Real Applications**
→ [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md) - Process real genomic data and scale up your experiments

### 🎯 **Train Production Models**
→ [04_TRAINING.md](04_TRAINING.md) - Complete training workflows with evaluation, checkpointing, and optimization

### ⚡ **Quick Reference**
→ [troubleshooting/quick_reference.md](troubleshooting/quick_reference.md) - Common commands and solutions

---

## 💡 Quick Tips

**Memory Issues?** Reduce model size:
```python
config = HyenaGLTConfig(
    hidden_size=128,    # Smaller
    num_layers=2,       # Fewer layers
    batch_size=1        # Smaller batches
)
```

**Want Real Performance?** Use the pre-built configs:
```python
config = HyenaGLTConfig.from_pretrained("hyena-glt-base")
```

**Need Help?** Check the [troubleshooting guide](troubleshooting/) or see [examples/](examples/) for more code samples.

---

*Next up: [Understanding BLT_Hyena Fundamentals →](01_FUNDAMENTALS.md)*
