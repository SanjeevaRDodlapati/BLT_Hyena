#!/usr/bin/env python3
"""
Streamlined Training Example for Hyena-GLT Models

This example demonstrates the key training capabilities with a focus on:
- Easy-to-use training workflows
- Best practices for genomic sequence modeling
- Production-ready configurations
- Clear examples for common use cases

Author: Hyena-GLT Development Team
Version: 1.1.0
"""

import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import Hyena-GLT components
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data import DNATokenizer
from hyena_glt.model import HyenaGLTForSequenceClassification
from hyena_glt.training import HyenaGLTTrainer, TrainingConfig

# Note: Using built-in training utilities instead of custom utils
# from examples.utils.model_utils import quick_train_model


def quick_train_model(sequences, labels, sequence_type="dna", task_type="classification",
                     epochs=5, batch_size=16, learning_rate=1e-4, verbose=True, save_path=None):
    """
    Quick training utility function for genomic models.

    Args:
        sequences: List of genomic sequences
        labels: List of labels
        sequence_type: Type of sequences ('dna', 'rna', 'protein')
        task_type: Type of task ('classification', 'regression')
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        verbose: Whether to print progress
        save_path: Path to save the model

    Returns:
        Tuple of (model, trainer, metrics)
    """
    from torch.utils.data import DataLoader

    from hyena_glt.data import DNATokenizer, GenomicDataset

    # Setup tokenizer
    tokenizer = DNATokenizer()

    # Create dataset
    data = [{"sequence": seq, "labels": label} for seq, label in zip(sequences, labels, strict=False)]
    dataset = GenomicDataset(data, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model config
    config = HyenaGLTConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_layers=4,  # Changed from num_hidden_layers
        num_attention_heads=8,
        max_position_embeddings=512
        # Note: num_labels might be handled differently in this architecture
    )

    # Create model
    model = HyenaGLTForSequenceClassification(config)

    # Create trainer config
    training_config = TrainingConfig(
        output_dir=save_path or "./outputs/quick_train",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=500,
        logging_steps=100,
        evaluation_strategy="no"
    )

    # Create trainer
    trainer = HyenaGLTTrainer(
        model=model,
        config=training_config,
        train_dataloader=dataloader,
        tokenizer=tokenizer
    )

    # Train model
    if verbose:
        print(f"Starting training for {epochs} epochs...")

    metrics = trainer.train()

    if save_path:
        trainer.save_model(save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    return model, trainer, metrics


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_genomic_data(num_samples: int = 1000, sequence_type: str = "dna") -> tuple[list[str], list[int]]:
    """
    Create synthetic genomic sequence data.

    Args:
        num_samples: Number of samples to generate
        sequence_type: Type of sequence ('dna', 'rna', 'protein')

    Returns:
        Tuple of (sequences, labels)
    """
    if sequence_type == "dna":
        alphabet = ['A', 'T', 'G', 'C']
        seq_length_range = (100, 1000)
    elif sequence_type == "rna":
        alphabet = ['A', 'U', 'G', 'C']
        seq_length_range = (100, 800)
    elif sequence_type == "protein":
        alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        seq_length_range = (50, 500)
    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}")

    sequences = []
    labels = []

    for _ in range(num_samples):
        # Generate random sequence
        seq_length = np.random.randint(*seq_length_range)
        sequence = ''.join(np.random.choice(alphabet, size=seq_length))
        sequences.append(sequence)

        # Generate label based on sequence characteristics
        if sequence_type == "dna":
            # Label based on GC content and presence of start codon
            gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
            has_start_codon = 'ATG' in sequence
            label = 1 if (gc_content > 0.5 and has_start_codon) else 0
        elif sequence_type == "rna":
            # Label based on AU content
            au_content = (sequence.count('A') + sequence.count('U')) / len(sequence)
            label = 1 if au_content > 0.5 else 0
        else:  # protein
            # Label based on hydrophobic amino acid content
            hydrophobic_aas = {'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V'}
            hydrophobic_content = sum(1 for aa in sequence if aa in hydrophobic_aas) / len(sequence)
            label = 1 if hydrophobic_content > 0.4 else 0

        labels.append(label)

    return sequences, labels


def example_1_basic_training():
    """Example 1: Basic DNA sequence classification."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Basic DNA Sequence Classification")
    logger.info("=" * 60)

    # Create data
    logger.info("Generating DNA sequence data...")
    sequences, labels = create_genomic_data(1000, "dna")

    # Quick training using utility function
    logger.info("Starting quick training...")
    model, trainer, metrics = quick_train_model(
        sequences=sequences,
        labels=labels,
        sequence_type="dna",
        task_type="classification",
        epochs=5,
        batch_size=16,
        learning_rate=1e-4,
        verbose=True,
        save_path="./outputs/basic_dna_model"
    )

    logger.info(f"Training completed! Final metrics: {metrics}")
    return model, trainer, metrics


# Alias for backward compatibility
def run_basic_dna_classification(sequences=None, labels=None, num_epochs=5):
    """Alias for example_1_basic_training with custom data support."""
    if sequences is None or labels is None:
        return example_1_basic_training()
    else:
        setup_logging()
        return quick_train_model(
            sequences=sequences,
            labels=labels,
            sequence_type="dna",
            task_type="classification",
            epochs=num_epochs,
            batch_size=16,
            learning_rate=1e-4,
            verbose=True
        )


def example_2_advanced_configuration():
    """Example 2: Advanced training with custom configuration."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Advanced Training Configuration")
    logger.info("=" * 60)

    # Create output directory
    output_dir = "./outputs/advanced_training"
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    logger.info("Generating genomic data...")
    train_sequences, train_labels = create_genomic_data(800, "dna")
    eval_sequences, eval_labels = create_genomic_data(200, "dna")

    # Configure model for long sequences
    model_config = HyenaGLTConfig(
        # Model architecture
        hidden_size=512,
        num_hyena_layers=8,
        num_attention_heads=16,
        intermediate_size=2048,

        # Genomic-specific
        genomic_vocab_size=4096,
        max_position_embeddings=2048,  # Support longer sequences

        # Task-specific
        num_labels=2,
        task_type="sequence_classification",

        # BLT-specific optimizations
        local_encoder_layers=2,
        local_decoder_layers=2,
        patch_size=8,  # Larger patches for efficiency

        # Hyena-specific
        hyena_order=3,
        hyena_filter_size=128,  # Larger filters for complex patterns
        use_conv_bias=True,

        # Training efficiency
        gradient_checkpointing=True
    )

    # Advanced training configuration
    training_config = TrainingConfig(
        # Basic parameters
        num_epochs=10,
        batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch size: 32
        max_grad_norm=1.0,

        # Optimization
        learning_rate=1e-4,
        weight_decay=0.01,
        optimizer_type="adamw",
        scheduler_type="cosine",
        warmup_steps=200,
        layer_wise_decay=0.9,  # Layer-wise learning rate decay

        # Logging and evaluation
        eval_steps=50,
        save_steps=100,
        logging_steps=25,
        log_level="INFO",

        # Checkpointing
        output_dir=output_dir,
        save_total_limit=3,
        save_best_only=True,

        # Mixed precision training
        fp16=torch.cuda.is_available(),

        # Multi-task learning (if applicable)
        multi_task=False,

        # Curriculum learning
        curriculum_learning=True,
        curriculum_strategy="linear",
        curriculum_steps=300,

        # Early stopping
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_metric="eval_accuracy"
    )

    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = DNATokenizer(k=3)  # Use 3-mers

    # Tokenize data
    def tokenize_sequences(sequences, labels, max_length=1024):
        input_ids = []
        attention_masks = []

        for sequence in sequences:
            tokens = tokenizer.encode(
                sequence,
                max_length=max_length,
                padding=True,
                truncation=True
            )
            input_ids.append(tokens['input_ids'])
            attention_masks.append(tokens['attention_mask'])

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_masks),
            'labels': torch.tensor(labels)
        }

    logger.info("Tokenizing sequences...")
    train_data = tokenize_sequences(train_sequences, train_labels)
    eval_data = tokenize_sequences(eval_sequences, eval_labels)

    # Create datasets and dataloaders
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

    train_dataloader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=training_config.batch_size, shuffle=False)

    # Initialize model
    logger.info("Initializing model...")
    model = HyenaGLTForSequenceClassification(model_config)

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = HyenaGLTTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer
    )

    # Train model
    logger.info("Starting training...")
    training_results = trainer.train()

    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()

    logger.info("Training completed!")
    logger.info(f"Final training results: {training_results}")
    logger.info(f"Final evaluation results: {eval_results}")

    return model, trainer, training_results, eval_results


def example_3_protein_function_prediction():
    """Example 3: Protein function prediction with specialized configuration."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Protein Function Prediction")
    logger.info("=" * 60)

    # Generate protein data
    logger.info("Generating protein sequence data...")
    sequences, labels = create_genomic_data(500, "protein")

    # Use specialized protein configuration
    model, trainer, metrics = quick_train_model(
        sequences=sequences,
        labels=labels,
        sequence_type="protein",
        task_type="classification",
        epochs=8,
        batch_size=12,
        learning_rate=2e-4,
        validation_split=0.2,
        verbose=True,
        save_path="./outputs/protein_function_model"
    )

    logger.info("Protein function prediction training completed!")
    logger.info(f"Final metrics: {metrics}")
    return model, trainer, metrics


def example_4_multi_task_training():
    """Example 4: Multi-task learning demonstration."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Multi-Task Learning (Conceptual)")
    logger.info("=" * 60)

    # Note: This is a conceptual example showing multi-task setup
    # Full implementation would require additional components

    logger.info("Setting up multi-task learning configuration...")

    # Multi-task configuration
    config = HyenaGLTConfig(
        hidden_size=512,
        num_hyena_layers=12,
        genomic_vocab_size=4096,
        multi_task=True,  # Enable multi-task mode
        task_type="multi_task"
    )

    # Training configuration for multi-task
    training_config = TrainingConfig(
        num_epochs=15,
        batch_size=8,
        learning_rate=1e-4,
        multi_task=True,
        task_weights={
            "promoter_prediction": 0.4,
            "gene_annotation": 0.3,
            "variant_effect": 0.3
        },
        task_weighting_strategy="adaptive"
    )

    logger.info("Multi-task configuration created (implementation would require task-specific datasets)")
    logger.info("Key features:")
    logger.info("- Shared backbone with task-specific heads")
    logger.info("- Adaptive task weighting")
    logger.info("- Balanced sampling across tasks")

    return config, training_config


def run_all_examples():
    """Run all training examples."""
    logger = setup_logging()

    logger.info("Running all Hyena-GLT training examples...")

    results = {}

    try:
        # Example 1: Basic training
        logger.info("\n" + "="*80)
        model1, trainer1, metrics1 = example_1_basic_training()
        results['basic_dna'] = metrics1

        # Example 2: Advanced configuration
        logger.info("\n" + "="*80)
        model2, trainer2, train_results2, eval_results2 = example_2_advanced_configuration()
        results['advanced_dna'] = {**train_results2, **eval_results2}

        # Example 3: Protein function prediction
        logger.info("\n" + "="*80)
        model3, trainer3, metrics3 = example_3_protein_function_prediction()
        results['protein_function'] = metrics3

        # Example 4: Multi-task setup
        logger.info("\n" + "="*80)
        config4, training_config4 = example_4_multi_task_training()
        results['multi_task'] = "Configuration created"

        logger.info("\n" + "="*80)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("="*80)

        # Summary
        logger.info("\nSummary of Results:")
        for example_name, result in results.items():
            logger.info(f"- {example_name}: {result}")

        return results

    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        raise


def main():
    """Main function with command line options."""
    import argparse

    parser = argparse.ArgumentParser(description="Streamlined Hyena-GLT Training Examples")
    parser.add_argument("--example", choices=['1', '2', '3', '4', 'all'], default='all',
                        help="Which example to run")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run selected example
    if args.example == '1':
        example_1_basic_training()
    elif args.example == '2':
        example_2_advanced_configuration()
    elif args.example == '3':
        example_3_protein_function_prediction()
    elif args.example == '4':
        example_4_multi_task_training()
    else:
        run_all_examples()


if __name__ == "__main__":
    main()
