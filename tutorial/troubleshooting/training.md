# Training Issues Troubleshooting

## Memory Errors

### Out-of-Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Solutions:**

1. **Reduce batch size:**
```python
# Instead of:
batch_size = 64

# Try:
batch_size = 32  # or 16, 8
```

2. **Enable gradient accumulation:**
```python
trainer = HyenaGLTTrainer(
    model=model,
    args=training_args,
    gradient_accumulation_steps=4  # Effective batch size = batch_size * 4
)
```

3. **Use mixed precision training:**
```python
training_args = TrainingArguments(
    fp16=True,  # Enable half-precision
    dataloader_pin_memory=False  # Reduce memory usage
)
```

4. **Optimize model configuration:**
```python
config = HyenaGLTConfig(
    hidden_size=512,  # Reduce from 768
    num_hidden_layers=8,  # Reduce from 12
    intermediate_size=1024  # Reduce from 2048
)
```

## Slow Training Performance

### Problem: Training is extremely slow

**Diagnostic steps:**

1. **Check GPU utilization:**
```bash
nvidia-smi -l 1  # Monitor GPU usage
```

2. **Profile training:**
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
               torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Training step
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Solutions:**

1. **Optimize data loading:**
```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Increase for faster data loading
    pin_memory=True,  # If using GPU
    persistent_workers=True  # Reduce worker startup time
)
```

2. **Use compiled models (PyTorch 2.0+):**
```python
model = torch.compile(model)
```

3. **Enable efficient attention:**
```python
# Use flash attention if available
config = HyenaGLTConfig(
    use_flash_attention=True
)
```

## Convergence Problems

### Model not learning / Loss not decreasing

**Diagnostic checklist:**

1. **Check learning rate:**
```python
# Learning rate too high or too low
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Try different values

# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

2. **Verify data preprocessing:**
```python
# Check tokenization
print("Sample tokens:", tokenizer.encode("ATCGATCG"))
print("Decoded:", tokenizer.decode([1, 2, 3, 4]))

# Verify labels
print("Label distribution:", torch.bincount(labels))
```

3. **Check gradient flow:**
```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm = {param.grad.norm():.6f}")
        else:
            print(f"{name}: No gradient")

# After loss.backward()
check_gradients(model)
```

**Solutions:**

1. **Adjust learning rate schedule:**
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=1e-3,
    steps_per_epoch=len(dataloader),
    epochs=num_epochs
)
```

2. **Use gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. **Try different optimizers:**
```python
# AdamW with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Or Lion optimizer
from lion_pytorch import Lion
optimizer = Lion(model.parameters(), lr=1e-4)
```

### Loss exploding / NaN values

**Solutions:**

1. **Enable gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

2. **Lower learning rate:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Much lower
```

3. **Check for unstable operations:**
```python
# Add numerical stability
def stable_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    return torch.softmax(x - x_max, dim=dim)
```

## Hyperparameter Tuning

### Systematic hyperparameter search

```python
import optuna

def objective(trial):
    # Define hyperparameter space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    hidden_size = trial.suggest_categorical('hidden_size', [512, 768, 1024])
    
    # Create model and train
    config = HyenaGLTConfig(hidden_size=hidden_size)
    model = HyenaGLT(config)
    
    # Quick training
    trainer = HyenaGLTTrainer(
        model=model,
        args=TrainingArguments(
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=3,
            evaluation_strategy="steps",
            eval_steps=100
        )
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    return eval_results['eval_loss']

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best hyperparameters:", study.best_params)
```

## Distributed Training Issues

### Multi-GPU training problems

**Setup verification:**
```python
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
```

**Solutions:**

1. **Proper distributed setup:**
```python
# Use torchrun for launching
# torchrun --nproc_per_node=2 train_script.py

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

# Wrap model
model = DDP(model, device_ids=[local_rank])
```

2. **Use Accelerate library:**
```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Training loop
for batch in dataloader:
    outputs = model(batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

## Debug Mode

### Enable detailed debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Add debug prints
class DebugHyenaGLT(HyenaGLT):
    def forward(self, *args, **kwargs):
        print(f"Input shape: {args[0].shape}")
        outputs = super().forward(*args, **kwargs)
        print(f"Output shape: {outputs.last_hidden_state.shape}")
        return outputs

model = DebugHyenaGLT(config)
```

## Common Error Messages

### "Expected tensor to have dtype torch.float32"

**Solution:**
```python
# Ensure correct dtypes
inputs = inputs.float()
labels = labels.long()  # For classification
```

### "Size mismatch" errors

**Solution:**
```python
# Check tensor dimensions
print(f"Model expects: {model.config.vocab_size}")
print(f"Input max token: {inputs.max()}")

# Reshape if necessary
if len(inputs.shape) == 1:
    inputs = inputs.unsqueeze(0)  # Add batch dimension
```

### "Cannot access storage" during loading

**Solution:**
```python
# Load with map_location
checkpoint = torch.load('model.pt', map_location='cpu')

# Or specify device
checkpoint = torch.load('model.pt', map_location=f'cuda:{local_rank}')
```

## Performance Monitoring

### Track training metrics

```python
import wandb

wandb.init(project="blt-hyena")

# Log metrics
wandb.log({
    "train_loss": loss.item(),
    "learning_rate": scheduler.get_last_lr()[0],
    "gpu_memory": torch.cuda.memory_allocated() / 1e9
})
```

For more specific issues, check the [GitHub Issues](https://github.com/sdodlapati3/BLT_Hyena/issues) or create a new issue with:
- Complete error message
- System information (GPU, CUDA version, PyTorch version)
- Minimal reproducible example
- Configuration settings used
