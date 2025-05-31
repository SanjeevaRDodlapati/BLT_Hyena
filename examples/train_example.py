"""Example training script for Hyena-GLT model."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import Hyena-GLT components
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data import DNATokenizer
from hyena_glt.model import HyenaGLTForSequenceClassification
from hyena_glt.training import HyenaGLTTrainer, TrainingConfig


def create_dummy_genomic_data(num_samples: int = 1000, seq_length: int = 512):
    """Create dummy genomic sequence data for testing."""
    # Generate random DNA sequences
    nucleotides = ['A', 'T', 'G', 'C']
    sequences = []
    labels = []

    for _ in range(num_samples):
        # Random sequence
        sequence = ''.join(np.random.choice(nucleotides, size=seq_length))
        sequences.append(sequence)

        # Dummy binary classification: high GC content vs low GC content
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        label = 1 if gc_content > 0.5 else 0
        labels.append(label)

    return sequences, labels


def main():
    """Main training script."""
    # Create output directory
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Configure model
    model_config = HyenaGLTConfig(
        # Model architecture
        hidden_size=256,
        num_hyena_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,

        # Genomic-specific
        genomic_vocab_size=4096,  # Large vocab for k-mers
        max_position_embeddings=1024,

        # Task-specific
        num_labels=2,  # Binary classification
        task_type="sequence_classification",

        # BLT-specific
        local_encoder_layers=2,
        local_decoder_layers=2,
        patch_size=4,

        # Hyena-specific
        hyena_order=3,
        hyena_filter_size=64,
        use_conv_bias=True,

        # Training efficiency
        gradient_checkpointing=True
    )

    # Configure training
    training_config = TrainingConfig(
        # Basic parameters
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,

        # Optimization
        learning_rate=1e-4,
        weight_decay=0.01,
        optimizer_type="adamw",
        scheduler_type="cosine",
        warmup_steps=100,

        # Logging and evaluation
        eval_steps=50,
        save_steps=100,
        logging_steps=25,
        log_level="INFO",

        # Checkpointing
        output_dir=output_dir,
        save_total_limit=3,

        # Mixed precision
        fp16=torch.cuda.is_available(),

        # Experiment tracking
        use_wandb=False,  # Set to True if you have wandb configured

        # Early stopping
        early_stopping=True,
        early_stopping_patience=3,
        early_stopping_metric="eval_loss"
    )

    print("Generating dummy genomic data...")
    # Create dummy data
    train_sequences, train_labels = create_dummy_genomic_data(800, 256)
    eval_sequences, eval_labels = create_dummy_genomic_data(200, 256)

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = DNATokenizer(k=3)  # Use 3-mers

    # Tokenize sequences
    def tokenize_data(sequences, labels):
        input_ids = []
        attention_masks = []

        for sequence in sequences:
            tokens = tokenizer.encode(sequence, max_length=512, padding=True, truncation=True)
            input_ids.append(tokens['input_ids'])
            attention_masks.append(tokens['attention_mask'])

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_masks),
            'labels': torch.tensor(labels)
        }

    print("Tokenizing training data...")
    train_data = tokenize_data(train_sequences, train_labels)
    print("Tokenizing evaluation data...")
    eval_data = tokenize_data(eval_sequences, eval_labels)

    # Create datasets
    train_dataset = TensorDataset(
        train_data['input_ids'],
        train_data['attention_mask'],
        train_data['labels']
    )

    eval_dataset = TensorDataset(
        eval_data['input_ids'],
        eval_data['attention_mask'],
        eval_data['labels']
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0
    )

    print("Initializing model...")
    # Create model
    model = HyenaGLTForSequenceClassification(model_config)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    print("Creating trainer...")
    trainer = HyenaGLTTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer
    )

    print("Starting training...")
    # Train model
    training_results = trainer.train()

    print("Training completed!")
    print(f"Final results: {training_results}")

    # Evaluate final model
    print("Running final evaluation...")
    final_metrics = trainer.evaluate()
    print(f"Final evaluation metrics: {final_metrics}")

    # Save trained model
    model_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Test prediction
    print("Testing prediction...")
    predictions = trainer.predict(eval_dataloader)
    print(f"Generated predictions for {len(predictions['predictions'])} samples")

    # Calculate accuracy
    if 'targets' in predictions:
        pred_classes = torch.argmax(predictions['predictions'], dim=-1)
        accuracy = (pred_classes == predictions['targets']).float().mean().item()
        print(f"Final test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
