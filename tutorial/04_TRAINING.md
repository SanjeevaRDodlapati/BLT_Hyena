# 04 - Training Workflows

**Estimated Time:** 60 minutes  
**Prerequisites:** [02 - Hyena Integration](02_HYENA_INTEGRATION.md), [03 - Data Pipeline](03_DATA_PIPELINE.md)  
**Next:** [05 - Evaluation](05_EVALUATION.md)

## Overview

This tutorial covers complete training workflows for BLT_Hyena models, from basic training loops to advanced distributed training and hyperparameter optimization. You'll learn to train models for various genomic tasks including classification, generation, and variant calling.

## What You'll Learn

- Setting up training configurations and environments
- Implementing custom training loops with HyenaGLTTrainer
- Distributed training across multiple GPUs
- Hyperparameter optimization and model selection
- Training monitoring and debugging techniques
- Task-specific training strategies

## Training Environment Setup

### Basic Training Configuration

```python
from hyena_glt import HyenaGLTConfig, HyenaGLT, HyenaGLTTrainer
from hyena_glt.training import TrainingConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Model configuration
model_config = HyenaGLTConfig(
    vocab_size=4,  # A, T, G, C
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    
    # Hyena settings
    use_hyena=True,
    hyena_order=2,
    hyena_filter_size=128,
    max_seq_len=4096,
    
    # Training optimizations
    gradient_checkpointing=True,
    use_fft_conv=True,
    dropout=0.1,
    layer_norm_eps=1e-5
)

# Training configuration
training_config = TrainingConfig(
    # Optimization
    learning_rate=5e-4,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    
    # Learning rate schedule
    lr_scheduler_type="cosine",
    warmup_steps=1000,
    max_steps=50000,
    
    # Training dynamics
    batch_size=8,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    
    # Logging and saving
    logging_steps=100,
    eval_steps=1000,
    save_steps=2000,
    save_total_limit=3,
    
    # Output
    output_dir="./training_output",
    run_name="hyena_glt_genomic_v1"
)

print("Training configuration initialized")
```

### Data Loading for Training

```python
from hyena_glt.data import GenomicDataset, GenomicTokenizer
from torch.utils.data import random_split

def setup_training_data():
    """Set up training and validation datasets"""
    
    # Initialize tokenizer
    tokenizer = GenomicTokenizer()
    
    # Load dataset (using the data pipeline from tutorial 03)
    dataset = GenomicDataset(
        fasta_path="data/genomic_sequences.fa",
        tokenizer=tokenizer,
        max_length=4096,
        task_type="classification",  # or "generation", "variant_calling"
        include_labels=True
    )
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset, tokenizer

# Create data loaders
def create_data_loaders(train_dataset, val_dataset, training_config):
    """Create training and validation data loaders"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

train_dataset, val_dataset, tokenizer = setup_training_data()
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, training_config)
```

## Custom Training Loop

### Basic Training Implementation

```python
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class GenomicTrainer:
    """Custom trainer for genomic tasks"""
    
    def __init__(self, model, train_loader, val_loader, config, tokenizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        
        # Setup optimizer and scheduler
        self.setup_optimization()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def setup_optimization(self):
        """Initialize optimizer and learning rate scheduler"""
        
        # Separate parameters for different components
        hyena_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'hyena' in name.lower():
                hyena_params.append(param)
            else:
                other_params.append(param)
        
        # Different learning rates for Hyena vs other components
        param_groups = [
            {'params': hyena_params, 'lr': self.config.learning_rate * 0.5},
            {'params': other_params, 'lr': self.config.learning_rate}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
        
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
    def setup_logging(self):
        """Initialize logging with wandb"""
        wandb.init(
            project="blt-hyena-genomic",
            name=self.config.run_name,
            config=vars(self.config)
        )
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_steps = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.model.device)
            labels = batch['labels'].to(self.model.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    wandb.log({
                        'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                        'train/learning_rate': current_lr,
                        'train/global_step': self.global_step
                    })
                
                # Validation
                if self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.validate()
                    wandb.log(val_metrics)
                    
                    # Save best model
                    if val_metrics['val/loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val/loss']
                        self.save_model("best_model")
                
                # Regular checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_model(f"checkpoint_step_{self.global_step}")
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_steps += 1
            
            progress_bar.set_postfix({
                'loss': total_loss / total_steps,
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Early stopping check
            if self.global_step >= self.config.max_steps:
                break
        
        return total_loss / total_steps
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Collect predictions for metrics
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        self.model.train()  # Switch back to training mode
        
        return {
            'val/loss': avg_loss,
            'val/accuracy': accuracy,
            'val/f1_score': f1,
            'val/global_step': self.global_step
        }
    
    def save_model(self, checkpoint_name):
        """Save model checkpoint"""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(self.config.output_dir, f"{checkpoint_name}.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_config': self.model.config
        }, checkpoint_path)
        
        print(f"Model saved to {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.max_steps} steps")
        
        while self.global_step < self.config.max_steps:
            epoch_loss = self.train_epoch()
            self.epoch += 1
            
            print(f"Epoch {self.epoch} completed. Average loss: {epoch_loss:.4f}")
            
            if self.global_step >= self.config.max_steps:
                break
        
        # Final validation and save
        final_metrics = self.validate()
        wandb.log(final_metrics)
        self.save_model("final_model")
        
        print("Training completed!")
        wandb.finish()

# Run training
model = HyenaGLT(model_config)
trainer = GenomicTrainer(model, train_loader, val_loader, training_config, tokenizer)
trainer.train()
```

## Task-Specific Training Strategies

### Classification Training

```python
class ClassificationTrainer(GenomicTrainer):
    """Specialized trainer for genomic classification tasks"""
    
    def __init__(self, model, train_loader, val_loader, config, tokenizer, num_classes):
        super().__init__(model, train_loader, val_loader, config, tokenizer)
        self.num_classes = num_classes
        
        # Classification-specific loss function
        self.criterion = nn.CrossEntropyLoss(
            weight=self.calculate_class_weights()
        )
    
    def calculate_class_weights(self):
        """Calculate class weights for imbalanced datasets"""
        # Count class frequencies in training data
        class_counts = torch.zeros(self.num_classes)
        
        for batch in self.train_loader:
            labels = batch['labels']
            for class_idx in range(self.num_classes):
                class_counts[class_idx] += (labels == class_idx).sum()
        
        # Calculate inverse frequency weights
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.num_classes * class_counts)
        
        return class_weights
    
    def compute_classification_metrics(self, predictions, labels):
        """Compute detailed classification metrics"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Detailed metrics
        report = classification_report(labels_np, pred_np, output_dict=True)
        cm = confusion_matrix(labels_np, pred_np)
        
        return {
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'confusion_matrix': cm.tolist()
        }

# Configure for variant calling classification
classification_config = HyenaGLTConfig(
    vocab_size=4,
    hidden_size=768,
    num_layers=12,
    use_hyena=True,
    
    # Classification head
    num_classes=3,  # Normal, SNV, Indel
    task_type="classification",
    
    # Optimizations for classification
    use_pooling=True,
    pooling_type="attention",  # attention, max, mean
    classifier_dropout=0.1
)
```

### Generation Training

```python
class GenerationTrainer(GenomicTrainer):
    """Specialized trainer for genomic sequence generation"""
    
    def __init__(self, model, train_loader, val_loader, config, tokenizer):
        super().__init__(model, train_loader, val_loader, config, tokenizer)
        
        # Generation-specific settings
        self.causal_mask = True
        
    def train_step_generation(self, batch):
        """Training step for autoregressive generation"""
        input_ids = batch['input_ids']
        
        # Create causal targets (shift by 1)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        
        # Forward pass
        outputs = self.model(inputs, use_cache=False)
        logits = outputs.logits
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        return loss
    
    def generate_sample(self, prompt_sequence, max_length=100):
        """Generate a sample sequence during validation"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = torch.tensor([prompt_sequence]).to(self.model.device)
            
            # Generate sequence
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=1,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode back to sequence
            generated_sequence = self.tokenizer.decode(generated[0])
            
        return generated_sequence
    
    def validate_generation(self):
        """Validation with generation quality metrics"""
        metrics = super().validate()
        
        # Generate sample sequences
        sample_prompts = ["ATG", "GCG", "TAC"]  # Start codons
        samples = []
        
        for prompt in sample_prompts:
            prompt_tokens = self.tokenizer.encode(prompt)
            generated = self.generate_sample(prompt_tokens, max_length=50)
            samples.append({
                'prompt': prompt,
                'generated': generated
            })
        
        # Log samples to wandb
        wandb.log({
            'generated_samples': wandb.Table(
                columns=['prompt', 'generated'],
                data=[[s['prompt'], s['generated']] for s in samples]
            )
        })
        
        return metrics

# Configure for sequence generation
generation_config = HyenaGLTConfig(
    vocab_size=4,
    hidden_size=768,
    num_layers=12,
    use_hyena=True,
    
    # Generation settings
    task_type="generation",
    use_causal_mask=True,
    max_position_embeddings=8192,
    
    # Generation-specific optimizations
    use_cache=True,
    tie_word_embeddings=True
)
```

## Distributed Training

### Multi-GPU Training Setup

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed_training():
    """Initialize distributed training"""
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=int(os.environ.get('WORLD_SIZE', 1)),
        rank=int(os.environ.get('RANK', 0))
    )
    
    # Set device
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return device, local_rank

def create_distributed_model(model_config, device):
    """Create model for distributed training"""
    
    # Create model
    model = HyenaGLT(model_config)
    model.to(device)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
        gradient_as_bucket_view=True
    )
    
    return model

def create_distributed_dataloader(dataset, config, is_train=True):
    """Create dataloader for distributed training"""
    
    sampler = DistributedSampler(
        dataset,
        shuffle=is_train,
        drop_last=is_train
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, sampler

# Distributed training script
class DistributedTrainer(GenomicTrainer):
    """Trainer with distributed training support"""
    
    def __init__(self, model, train_loader, val_loader, config, tokenizer, device, local_rank):
        self.device = device
        self.local_rank = local_rank
        self.is_main_process = local_rank == 0
        
        super().__init__(model, train_loader, val_loader, config, tokenizer)
        
    def setup_logging(self):
        """Only log from main process"""
        if self.is_main_process:
            super().setup_logging()
    
    def save_model(self, checkpoint_name):
        """Only save from main process"""
        if self.is_main_process:
            # Save the underlying model (not DDP wrapper)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            
            checkpoint_path = os.path.join(self.config.output_dir, f"{checkpoint_name}.pt")
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'global_step': self.global_step,
                'epoch': self.epoch,
                'config': self.config
            }, checkpoint_path)
            
            print(f"Model saved to {checkpoint_path}")
    
    def train_epoch(self):
        """Distributed training epoch"""
        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        return super().train_epoch()

# Launch distributed training
if __name__ == "__main__":
    device, local_rank = setup_distributed_training()
    
    # Create distributed model and data
    model = create_distributed_model(model_config, device)
    dist_train_loader, train_sampler = create_distributed_dataloader(train_dataset, training_config, True)
    dist_val_loader, val_sampler = create_distributed_dataloader(val_dataset, training_config, False)
    
    # Train with distributed trainer
    trainer = DistributedTrainer(
        model, dist_train_loader, dist_val_loader, 
        training_config, tokenizer, device, local_rank
    )
    trainer.train()
    
    # Cleanup
    dist.destroy_process_group()
```

## Hyperparameter Optimization

### Optuna Integration

```python
import optuna
from optuna.integration import WeightsAndBiasesCallback

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    config = HyenaGLTConfig(
        vocab_size=4,
        hidden_size=trial.suggest_categorical('hidden_size', [512, 768, 1024]),
        num_layers=trial.suggest_int('num_layers', 6, 16),
        
        # Hyena-specific
        hyena_order=trial.suggest_int('hyena_order', 1, 3),
        hyena_filter_size=trial.suggest_categorical('hyena_filter_size', [64, 128, 256]),
        
        # Training parameters
        dropout=trial.suggest_float('dropout', 0.05, 0.3),
        use_hyena=True,
        max_seq_len=4096
    )
    
    training_config = TrainingConfig(
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float('weight_decay', 0.001, 0.1, log=True),
        batch_size=trial.suggest_categorical('batch_size', [4, 8, 16]),
        warmup_steps=trial.suggest_int('warmup_steps', 500, 2000),
        max_steps=10000,  # Shorter for hyperparameter search
        
        output_dir=f"./optuna_trial_{trial.number}",
        run_name=f"trial_{trial.number}"
    )
    
    # Create and train model
    model = HyenaGLT(config)
    trainer = GenomicTrainer(model, train_loader, val_loader, training_config, tokenizer)
    
    # Train and get validation metrics
    trainer.train()
    final_metrics = trainer.validate()
    
    # Return metric to optimize (minimize validation loss)
    return final_metrics['val/loss']

# Run hyperparameter optimization
def run_hyperparameter_search():
    """Run Optuna hyperparameter search"""
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name='blt_hyena_optimization',
        storage='sqlite:///hyena_optimization.db',
        load_if_exists=True
    )
    
    # Add W&B callback
    wandb_callback = WeightsAndBiasesCallback(
        metric_name='val_loss',
        wandb_kwargs={'project': 'blt-hyena-optimization'}
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=50,
        callbacks=[wandb_callback],
        show_progress_bar=True
    )
    
    # Print best parameters
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    return study.best_params

# best_params = run_hyperparameter_search()
```

## Advanced Training Techniques

### Curriculum Learning

```python
class CurriculumTrainer(GenomicTrainer):
    """Trainer with curriculum learning for progressive sequence lengths"""
    
    def __init__(self, model, datasets_by_length, config, tokenizer):
        # datasets_by_length: {length: dataset} dictionary
        self.datasets_by_length = datasets_by_length
        self.current_length = min(datasets_by_length.keys())
        
        # Initialize with shortest sequences
        current_dataset = datasets_by_length[self.current_length]
        train_loader = DataLoader(current_dataset, batch_size=config.batch_size, shuffle=True)
        
        super().__init__(model, train_loader, None, config, tokenizer)
        
        # Curriculum schedule
        self.length_schedule = sorted(datasets_by_length.keys())
        self.steps_per_length = 5000
        
    def should_increase_difficulty(self):
        """Check if we should move to longer sequences"""
        return (self.global_step % self.steps_per_length == 0 and 
                self.current_length < max(self.length_schedule))
    
    def update_curriculum(self):
        """Update to next difficulty level"""
        current_idx = self.length_schedule.index(self.current_length)
        if current_idx < len(self.length_schedule) - 1:
            self.current_length = self.length_schedule[current_idx + 1]
            
            # Create new dataloader
            new_dataset = self.datasets_by_length[self.current_length]
            self.train_loader = DataLoader(
                new_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
            
            print(f"Curriculum updated to sequence length: {self.current_length}")
    
    def train_epoch(self):
        """Training with curriculum updates"""
        if self.should_increase_difficulty():
            self.update_curriculum()
        
        return super().train_epoch()

# Create curriculum datasets
def create_curriculum_datasets():
    """Create datasets with different sequence lengths"""
    lengths = [512, 1024, 2048, 4096]
    datasets = {}
    
    for length in lengths:
        datasets[length] = GenomicDataset(
            fasta_path="data/genomic_sequences.fa",
            tokenizer=tokenizer,
            max_length=length,
            task_type="generation"
        )
    
    return datasets

# Use curriculum learning
curriculum_datasets = create_curriculum_datasets()
curriculum_trainer = CurriculumTrainer(
    model, curriculum_datasets, training_config, tokenizer
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer(GenomicTrainer):
    """Trainer with automatic mixed precision"""
    
    def __init__(self, model, train_loader, val_loader, config, tokenizer):
        super().__init__(model, train_loader, val_loader, config, tokenizer)
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler()
        
    def train_step(self, batch):
        """Training step with automatic mixed precision"""
        input_ids = batch['input_ids'].to(self.model.device)
        labels = batch['labels'].to(self.model.device)
        
        # Forward pass with autocast
        with autocast():
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with scaled gradients
        self.scaler.scale(loss).backward()
        
        return loss
    
    def optimizer_step(self):
        """Optimizer step with gradient scaling"""
        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
```

## Training Monitoring and Debugging

### Advanced Logging

```python
class AdvancedLogger:
    """Comprehensive training logger"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.gradient_norms = []
        self.weight_norms = []
        
    def log_model_statistics(self):
        """Log detailed model statistics"""
        stats = {}
        
        # Parameter statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        stats['model/total_parameters'] = total_params
        stats['model/trainable_parameters'] = trainable_params
        
        # Layer-wise statistics
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                stats[f'weights/{name}/mean'] = weight.mean().item()
                stats[f'weights/{name}/std'] = weight.std().item()
                stats[f'weights/{name}/norm'] = weight.norm().item()
        
        return stats
    
    def log_gradient_statistics(self):
        """Log gradient statistics"""
        stats = {}
        total_norm = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                stats[f'gradients/{name}/norm'] = grad_norm.item()
                total_norm += grad_norm.item() ** 2
        
        stats['gradients/total_norm'] = total_norm ** 0.5
        self.gradient_norms.append(stats['gradients/total_norm'])
        
        return stats
    
    def log_hyena_specific_metrics(self):
        """Log Hyena-specific metrics"""
        stats = {}
        
        for name, module in self.model.named_modules():
            if 'hyena' in name.lower():
                # Log Hyena filter statistics
                if hasattr(module, 'filters'):
                    filters = module.filters
                    stats[f'hyena/{name}/filter_mean'] = filters.mean().item()
                    stats[f'hyena/{name}/filter_std'] = filters.std().item()
                
                # Log Hyena gate statistics if present
                if hasattr(module, 'gate'):
                    gate = module.gate
                    stats[f'hyena/{name}/gate_mean'] = gate.mean().item()
        
        return stats

# Usage in training loop
logger = AdvancedLogger(model, training_config)

# In training step
model_stats = logger.log_model_statistics()
gradient_stats = logger.log_gradient_statistics()
hyena_stats = logger.log_hyena_specific_metrics()

wandb.log({**model_stats, **gradient_stats, **hyena_stats})
```

## Production Training Pipeline

### Complete Training Script

```python
#!/usr/bin/env python3
"""
Production training script for BLT_Hyena models
"""

import argparse
import os
import yaml
from pathlib import Path
import torch
import wandb
from hyena_glt import HyenaGLTConfig, HyenaGLT
from hyena_glt.training import TrainingConfig

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train BLT_Hyena model')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Split into model and training configs
    model_config = HyenaGLTConfig(**config_dict['model'])
    training_config = TrainingConfig(**config_dict['training'])
    
    return model_config, training_config

def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    model_config, training_config = load_config(args.config)
    training_config.output_dir = args.output_dir
    
    # Setup distributed training if needed
    if args.local_rank != -1:
        device, local_rank = setup_distributed_training()
        model = create_distributed_model(model_config, device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HyenaGLT(model_config).to(device)
    
    # Load data
    train_dataset, val_dataset, tokenizer = setup_training_data_from_path(args.data_path)
    
    if args.local_rank != -1:
        train_loader, _ = create_distributed_dataloader(train_dataset, training_config, True)
        val_loader, _ = create_distributed_dataloader(val_dataset, training_config, False)
    else:
        train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, training_config)
    
    # Create trainer
    if args.local_rank != -1:
        trainer = DistributedTrainer(
            model, train_loader, val_loader, training_config, tokenizer, device, args.local_rank
        )
    else:
        trainer = GenomicTrainer(model, train_loader, val_loader, training_config, tokenizer)
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
```

### Configuration Template

```yaml
# config/genomic_classification.yaml
model:
  vocab_size: 4
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  
  # Hyena settings
  use_hyena: true
  hyena_order: 2
  hyena_filter_size: 128
  max_seq_len: 4096
  
  # Task settings
  num_classes: 3
  task_type: "classification"
  
  # Optimizations
  gradient_checkpointing: true
  use_fft_conv: true
  dropout: 0.1

training:
  # Optimization
  learning_rate: 5e-4
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  
  # Schedule
  lr_scheduler_type: "cosine"
  warmup_steps: 1000
  max_steps: 50000
  
  # Training
  batch_size: 8
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  
  # Logging
  logging_steps: 100
  eval_steps: 1000
  save_steps: 2000
  save_total_limit: 3
  
  # Experiment
  run_name: "hyena_glt_genomic_classification_v1"
```

## Key Takeaways

1. **Configuration**: Use structured configs for reproducible training
2. **Monitoring**: Comprehensive logging helps debug training issues
3. **Optimization**: Mixed precision and distributed training for efficiency
4. **Validation**: Regular evaluation prevents overfitting
5. **Checkpointing**: Save models regularly for recovery and analysis
6. **Hyperparameters**: Use systematic search for optimal parameters

## Troubleshooting

### Common Training Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use gradient accumulation
   - Try mixed precision training

2. **Slow Convergence**
   - Adjust learning rate
   - Use learning rate scheduling
   - Check data quality and preprocessing
   - Monitor gradient norms

3. **Training Instability**
   - Enable gradient clipping
   - Reduce learning rate
   - Check for data issues
   - Use more conservative hyperparameters

## Next Steps

Continue to [05 - Evaluation](05_EVALUATION.md) to learn about comprehensive model evaluation, or explore [06 - Production](06_PRODUCTION.md) for deployment strategies.

## Additional Resources

- [Training Best Practices](../docs/TRAINING_BEST_PRACTICES.md)
- [Distributed Training Guide](../docs/DISTRIBUTED_TRAINING.md)
- [Hyperparameter Optimization](../docs/HYPERPARAMETER_OPTIMIZATION.md)
