"""
Integration tests for complete workflows.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model.hyena_glt import (
    HyenaGLT, HyenaGLTForSequenceClassification,
    HyenaGLTForTokenClassification
)
from hyena_glt.data.tokenizer import DNATokenizer, ProteinTokenizer
from hyena_glt.data.dataset import GenomicDataset
from hyena_glt.training.trainer import HyenaGLTTrainer
from hyena_glt.training.config import TrainingConfig
from hyena_glt.evaluation.evaluator import HyenaGLTEvaluator
from hyena_glt.evaluation.metrics import compute_classification_metrics
from tests.utils import TestConfig, DataGenerator


@pytest.mark.integration
class TestTrainingWorkflow:
    """Test complete training workflows."""
    
    def test_dna_classification_workflow(self):
        """Test end-to-end DNA classification workflow."""
        # Configuration
        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model_config.num_labels = 3
        
        training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=2,
            eval_steps=5,
            save_steps=10,
            logging_steps=2,
            warmup_steps=2
        )
        
        # Create model and tokenizer
        model = HyenaGLTForSequenceClassification(model_config)
        tokenizer = DNATokenizer()
        
        # Generate sample data
        sequences = []
        labels = []
        for i in range(20):
            seq_length = torch.randint(50, 100, (1,)).item()
            seq = DataGenerator.generate_dna_sequence(seq_length)
            seq_str = ''.join(['ATCG'[x] for x in seq])
            sequences.append(seq_str)
            labels.append(i % 3)  # 3 classes
        
        # Create datasets
        train_dataset = GenomicDataset(
            sequences=sequences[:16],
            labels=labels[:16],
            tokenizer=tokenizer,
            max_length=model_config.sequence_length
        )
        
        val_dataset = GenomicDataset(
            sequences=sequences[16:],
            labels=labels[16:],
            tokenizer=tokenizer,
            max_length=model_config.sequence_length
        )
        
        # Train model
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.output_dir = temp_dir
            
            trainer = HyenaGLTTrainer(
                model=model,
                config=training_config,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer
            )
            
            # Training should complete without errors
            trainer.train()
            
            # Check that checkpoints were saved
            assert os.path.exists(os.path.join(temp_dir, "pytorch_model.bin"))
            
            # Test evaluation
            evaluator = HyenaGLTEvaluator(model, tokenizer)
            results = evaluator.evaluate(val_dataset)
            
            assert 'accuracy' in results
            assert 'loss' in results
            assert 0 <= results['accuracy'] <= 1
    
    def test_protein_sequence_generation_workflow(self):
        """Test protein sequence generation workflow."""
        # Configuration for generation
        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model_config.vocab_size = 25  # 20 amino acids + special tokens
        
        training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=1,
            eval_steps=5,
            save_steps=10
        )
        
        # Create model and tokenizer
        model = HyenaGLT(model_config)
        tokenizer = ProteinTokenizer()
        
        # Generate protein sequences
        sequences = []
        for i in range(16):
            seq_length = torch.randint(30, 80, (1,)).item()
            seq = DataGenerator.generate_protein_sequence(seq_length, vocab_size=20)
            # Convert to amino acid string
            aa_chars = "ACDEFGHIKLMNPQRSTVWY"
            seq_str = ''.join([aa_chars[x] for x in seq])
            sequences.append(seq_str)
        
        # Create dataset for language modeling
        train_dataset = GenomicDataset(
            sequences=sequences[:12],
            tokenizer=tokenizer,
            max_length=model_config.sequence_length,
            task_type='generation'
        )
        
        val_dataset = GenomicDataset(
            sequences=sequences[12:],
            tokenizer=tokenizer,
            max_length=model_config.sequence_length,
            task_type='generation'
        )
        
        # Train for sequence generation
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.output_dir = temp_dir
            
            trainer = HyenaGLTTrainer(
                model=model,
                config=training_config,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer
            )
            
            # Training should complete
            trainer.train()
            
            # Test generation
            model.eval()
            with torch.no_grad():
                input_ids = tokenizer.encode("MKT")[:10]  # Short prompt
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                
                # Generate sequence
                generated = model.generate(
                    input_ids,
                    max_length=50,
                    temperature=1.0,
                    do_sample=True
                )
                
                assert generated.shape[0] == 1
                assert generated.shape[1] <= 50
    
    def test_token_classification_workflow(self):
        """Test token-level classification workflow."""
        # Configuration
        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model_config.num_labels = 5  # 5 annotation classes
        
        training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=1,
            eval_steps=5
        )
        
        # Create model and tokenizer
        model = HyenaGLTForTokenClassification(model_config)
        tokenizer = DNATokenizer()
        
        # Generate data with token-level labels
        sequences = []
        token_labels = []
        
        for i in range(16):
            seq_length = 60
            seq = DataGenerator.generate_dna_sequence(seq_length)
            seq_str = ''.join(['ATCG'[x] for x in seq])
            
            # Create random token labels
            labels = torch.randint(0, 5, (seq_length,)).tolist()
            
            sequences.append(seq_str)
            token_labels.append(labels)
        
        # Create dataset
        train_dataset = GenomicDataset(
            sequences=sequences[:12],
            token_labels=token_labels[:12],
            tokenizer=tokenizer,
            max_length=model_config.sequence_length,
            task_type='token_classification'
        )
        
        val_dataset = GenomicDataset(
            sequences=sequences[12:],
            token_labels=token_labels[12:],
            tokenizer=tokenizer,
            max_length=model_config.sequence_length,
            task_type='token_classification'
        )
        
        # Train model
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.output_dir = temp_dir
            
            trainer = HyenaGLTTrainer(
                model=model,
                config=training_config,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer
            )
            
            trainer.train()
            
            # Evaluate
            evaluator = HyenaGLTEvaluator(model, tokenizer)
            results = evaluator.evaluate(val_dataset)
            
            assert 'token_accuracy' in results
            assert 'token_f1' in results


@pytest.mark.integration
class TestDataProcessingWorkflow:
    """Test data processing workflows."""
    
    def test_large_dataset_processing(self):
        """Test processing larger datasets."""
        # Create larger dataset
        num_sequences = 100
        sequences = []
        labels = []
        
        for i in range(num_sequences):
            seq_length = torch.randint(100, 300, (1,)).item()
            seq = DataGenerator.generate_dna_sequence(seq_length)
            seq_str = ''.join(['ATCG'[x] for x in seq])
            sequences.append(seq_str)
            labels.append(i % 4)  # 4 classes
        
        tokenizer = DNATokenizer()
        
        # Create dataset
        dataset = GenomicDataset(
            sequences=sequences,
            labels=labels,
            tokenizer=tokenizer,
            max_length=512
        )
        
        # Test data loading
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing in tests
        )
        
        # Should be able to iterate through all batches
        batch_count = 0
        for batch in dataloader:
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch
            
            batch_size = batch['input_ids'].shape[0]
            assert batch_size <= 8
            
            batch_count += 1
        
        expected_batches = (num_sequences + 7) // 8
        assert batch_count == expected_batches
    
    def test_mixed_length_sequences(self):
        """Test handling sequences of very different lengths."""
        sequences = [
            "ATCG",  # Very short
            "ATCG" * 50,  # Medium
            "ATCG" * 200,  # Long
            "AT",  # Very short
            "ATCG" * 100  # Medium-long
        ]
        
        labels = [0, 1, 2, 0, 1]
        
        tokenizer = DNATokenizer()
        
        dataset = GenomicDataset(
            sequences=sequences,
            labels=labels,
            tokenizer=tokenizer,
            max_length=512
        )
        
        # All sequences should be processed successfully
        for i in range(len(sequences)):
            item = dataset[i]
            assert 'input_ids' in item
            assert item['input_ids'].shape[0] <= 512


@pytest.mark.integration
@pytest.mark.slow
class TestModelScaling:
    """Test model scaling and performance."""
    
    def test_medium_model_training(self):
        """Test training medium-sized model."""
        # Use medium configuration
        model_config = HyenaGLTConfig(**TestConfig.MEDIUM_CONFIG)
        model_config.num_labels = 3
        
        training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=4,
            num_epochs=1,
            eval_steps=10,
            gradient_accumulation_steps=2
        )
        
        model = HyenaGLTForSequenceClassification(model_config)
        tokenizer = DNATokenizer()
        
        # Generate more data
        sequences = []
        labels = []
        for i in range(50):
            seq_length = torch.randint(200, 400, (1,)).item()
            seq = DataGenerator.generate_dna_sequence(seq_length)
            seq_str = ''.join(['ATCG'[x] for x in seq])
            sequences.append(seq_str)
            labels.append(i % 3)
        
        train_dataset = GenomicDataset(
            sequences=sequences[:40],
            labels=labels[:40],
            tokenizer=tokenizer,
            max_length=model_config.sequence_length
        )
        
        val_dataset = GenomicDataset(
            sequences=sequences[40:],
            labels=labels[40:],
            tokenizer=tokenizer,
            max_length=model_config.sequence_length
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.output_dir = temp_dir
            
            trainer = HyenaGLTTrainer(
                model=model,
                config=training_config,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer
            )
            
            # Should complete training
            trainer.train()
            
            # Test memory usage is reasonable
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                assert memory_used < 8.0  # Should use less than 8GB
    
    @pytest.mark.gpu
    def test_gpu_training(self):
        """Test GPU training if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model_config.num_labels = 2
        
        model = HyenaGLTForSequenceClassification(model_config)
        model = model.cuda()
        
        tokenizer = DNATokenizer()
        
        # Small dataset for GPU test
        sequences = ["ATCG" * 50 for _ in range(8)]
        labels = [i % 2 for i in range(8)]
        
        dataset = GenomicDataset(
            sequences=sequences,
            labels=labels,
            tokenizer=tokenizer,
            max_length=256
        )
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Test forward pass on GPU
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                
                assert outputs.logits.device.type == 'cuda'
                assert outputs.loss.device.type == 'cuda'
                break


@pytest.mark.integration
class TestModelSerialization:
    """Test model saving and loading."""
    
    def test_complete_model_save_load(self):
        """Test saving and loading complete model."""
        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model_config.num_labels = 3
        
        original_model = HyenaGLTForSequenceClassification(model_config)
        tokenizer = DNATokenizer()
        
        # Create test input
        test_sequence = "ATCGATCGATCG"
        encoded = tokenizer.encode(test_sequence)
        input_tensor = torch.tensor(encoded).unsqueeze(0)
        
        # Get original output
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(input_tensor)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model")
            
            # Save model
            original_model.save_pretrained(model_path)
            
            # Load model
            loaded_model = HyenaGLTForSequenceClassification.from_pretrained(
                model_path,
                config=model_config
            )
            
            # Test loaded model produces same output
            loaded_model.eval()
            with torch.no_grad():
                loaded_output = loaded_model(input_tensor)
            
            torch.testing.assert_close(
                original_output.logits,
                loaded_output.logits,
                rtol=1e-5,
                atol=1e-6
            )
    
    def test_checkpoint_resuming(self):
        """Test resuming training from checkpoint."""
        model_config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=3,
            save_steps=5
        )
        
        model = HyenaGLT(model_config)
        tokenizer = DNATokenizer()
        
        # Small dataset
        sequences = ["ATCG" * 30 for _ in range(12)]
        dataset = GenomicDataset(
            sequences=sequences,
            tokenizer=tokenizer,
            max_length=128,
            task_type='generation'
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config.output_dir = temp_dir
            
            # Start training
            trainer = HyenaGLTTrainer(
                model=model,
                config=training_config,
                train_dataset=dataset,
                tokenizer=tokenizer
            )
            
            # Train for a few steps
            trainer.train(max_steps=10)
            
            # Check checkpoint exists
            checkpoint_dirs = [d for d in os.listdir(temp_dir) if d.startswith('checkpoint')]
            assert len(checkpoint_dirs) > 0
            
            # Resume from checkpoint
            latest_checkpoint = os.path.join(temp_dir, checkpoint_dirs[-1])
            
            new_trainer = HyenaGLTTrainer(
                model=model,
                config=training_config,
                train_dataset=dataset,
                tokenizer=tokenizer,
                resume_from_checkpoint=latest_checkpoint
            )
            
            # Should resume successfully
            new_trainer.train(max_steps=20)
