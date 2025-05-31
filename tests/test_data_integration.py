"""
Integration tests for Hyena-GLT data infrastructure components.
Tests the interaction between tokenizers, datasets, collators, and loaders.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data import (
    DNATokenizer,
    GenomicDataLoader,
    GenomicDataset,
    GenomicPreprocessor,
    LengthGroupedSampler,
    MultiModalCollator,
    SequenceCollator,
    StreamingDataLoader,
    create_genomic_dataloaders,
)


class TestDataIntegration:
    """Test integration between data components."""

    def get_sample_dna_sequences(self):
        """Generate sample DNA sequences for testing."""
        sequences = [
            "ATCGATCGATCG",
            "GCTAGCTAGCTA",
            "ATGCATGCATGC",
            "CGATCGATCGAT",
            "TAGCTAGCTAGC",
        ]
        labels = [0, 1, 0, 1, 0]
        return sequences, labels

    def get_config(self):
        """Create test configuration."""
        return HyenaGLTConfig(
            vocab_size=6,  # A, T, C, G, UNK, PAD
            hidden_size=32,
            max_position_embeddings=64,
            num_layers=2,
        )

    def get_temp_data_dir(self):
        """Create temporary directory for test data."""
        return tempfile.mkdtemp()

    def test_tokenizer_dataset_integration(self):
        """Test tokenizer integration with dataset."""
        sequences, labels = self.get_sample_dna_sequences()
        config = self.get_config()

        # Initialize tokenizer
        tokenizer = DNATokenizer(vocab_size=6, sequence_length=32)

        # Create dataset
        dataset = GenomicDataset(
            sequences=sequences, labels=labels, tokenizer=tokenizer, max_length=32
        )

        # Test dataset items
        assert len(dataset) == len(sequences)

        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape[0] <= 32
        assert sample["labels"] == 0

        # Verify tokenization worked
        assert torch.is_tensor(sample["input_ids"])
        assert sample["input_ids"].dtype == torch.long

    def test_preprocessing_integration(self, sample_dna_sequences, temp_data_dir):
        """Test preprocessing integration with other components."""
        sequences, labels = sample_dna_sequences

        # Initialize preprocessor
        preprocessor = GenomicPreprocessor(sequence_type="dna", quality_threshold=0.8)

        # Create test FASTA file
        fasta_path = Path(temp_data_dir) / "test.fasta"
        with open(fasta_path, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")

        # Process file
        processed_data = preprocessor.process_file(str(fasta_path))

        assert len(processed_data) == len(sequences)
        for item in processed_data:
            assert "sequence" in item
            assert "sequence_id" in item
            assert len(item["sequence"]) > 0

    def test_collator_integration(self):
        """Test collator integration with dataset."""
        sequences, labels = self.get_sample_dna_sequences()
        config = self.get_config()

        # Setup components
        tokenizer = DNATokenizer(vocab_size=6, sequence_length=32)
        dataset = GenomicDataset(
            sequences=sequences, labels=labels, tokenizer=tokenizer, max_length=32
        )
        collator = SequenceCollator(
            tokenizer=tokenizer, max_length=32, padding="max_length"
        )

        # Test collation
        batch_samples = [dataset[i] for i in range(3)]
        batch = collator(batch_samples)

        assert isinstance(batch.input_ids, torch.Tensor)
        assert isinstance(batch.attention_mask, torch.Tensor)
        assert isinstance(batch.labels, torch.Tensor)

        assert batch.input_ids.shape[0] == 3  # batch size
        assert batch.attention_mask.shape[0] == 3
        assert batch.labels.shape[0] == 3

    def test_multimodal_collator(self, sample_dna_sequences, config):
        """Test multi-modal collator."""
        sequences, labels = sample_dna_sequences

        # Create multi-modal data (DNA + RNA)
        dna_tokenizer = DNATokenizer(vocab_size=6, sequence_length=32)

        # Simulate multi-modal samples
        samples = []
        for i, (seq, label) in enumerate(zip(sequences[:2], labels[:2], strict=False)):
            # Tokenize DNA sequence
            dna_tokens = dna_tokenizer.encode(seq, max_length=32)

            # Create sample with both modalities
            sample = {
                "dna_input_ids": torch.tensor(dna_tokens, dtype=torch.long),
                "rna_input_ids": torch.tensor(
                    dna_tokens, dtype=torch.long
                ),  # For testing
                "labels": torch.tensor(label, dtype=torch.long),
            }
            samples.append(sample)

        collator = MultiModalCollator(
            modalities=["dna", "rna"],
            tokenizers={"dna": dna_tokenizer, "rna": dna_tokenizer},
            max_length=32,
        )

        batch = collator(samples)

        assert hasattr(batch, "dna_input_ids")
        assert hasattr(batch, "rna_input_ids")
        assert hasattr(batch, "labels")
        assert batch.dna_input_ids.shape[0] == 2

    def test_dataloader_integration(self):
        """Test data loader integration."""
        sequences, labels = self.get_sample_dna_sequences()
        config = self.get_config()

        # Setup components
        tokenizer = DNATokenizer(vocab_size=6, sequence_length=32)
        dataset = GenomicDataset(
            sequences=sequences, labels=labels, tokenizer=tokenizer, max_length=32
        )

        # Test GenomicDataLoader
        dataloader = GenomicDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            tokenizer=tokenizer,
            max_length=32,
        )

        # Test iteration
        batches = list(dataloader)
        assert len(batches) >= 2  # Should have at least 2 batches

        first_batch = batches[0]
        assert hasattr(first_batch, "input_ids")
        assert hasattr(first_batch, "attention_mask")
        assert hasattr(first_batch, "labels")
        assert first_batch.input_ids.shape[0] == 2  # batch size

    def test_length_grouped_sampler(self, sample_dna_sequences, config):
        """Test length-grouped sampler for efficient batching."""
        sequences, labels = sample_dna_sequences

        # Create sequences with different lengths
        varied_sequences = [
            "ATCG",  # Length 4
            "ATCGATCG",  # Length 8
            "ATCGATCGATCG",  # Length 12
            "ATCG" * 4,  # Length 16
            "ATCG" * 5,  # Length 20
        ]

        tokenizer = DNATokenizer(vocab_size=6, sequence_length=64)
        dataset = GenomicDataset(
            sequences=varied_sequences,
            labels=labels,
            tokenizer=tokenizer,
            max_length=64,
        )

        # Test length-grouped sampler
        sampler = LengthGroupedSampler(
            dataset=dataset,
            batch_size=2,
            length_fn=lambda x: len(dataset[x]["input_ids"]),
        )

        # Check that sampler groups similar lengths
        batch_indices = list(sampler)
        assert len(batch_indices) > 0

        # Each batch should be a list of indices
        for batch in batch_indices:
            assert isinstance(batch, list)
            assert len(batch) <= 2  # batch size

    def test_create_genomic_dataloaders(self, sample_dna_sequences, config):
        """Test convenience function for creating data loaders."""
        sequences, labels = sample_dna_sequences

        # Split data
        train_sequences = sequences[:3]
        train_labels = labels[:3]
        val_sequences = sequences[3:]
        val_labels = labels[3:]

        tokenizer = DNATokenizer(vocab_size=6, sequence_length=32)

        # Create datasets
        train_dataset = GenomicDataset(
            sequences=train_sequences,
            labels=train_labels,
            tokenizer=tokenizer,
            max_length=32,
        )
        val_dataset = GenomicDataset(
            sequences=val_sequences,
            labels=val_labels,
            tokenizer=tokenizer,
            max_length=32,
        )

        # Test convenience function
        dataloaders = create_genomic_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=32,
        )

        assert "train" in dataloaders
        assert "val" in dataloaders

        # Test train loader
        train_batch = next(iter(dataloaders["train"]))
        assert hasattr(train_batch, "input_ids")
        assert train_batch.input_ids.shape[0] <= 2

        # Test val loader
        val_batch = next(iter(dataloaders["val"]))
        assert hasattr(val_batch, "input_ids")
        assert val_batch.input_ids.shape[0] <= 2

    def test_streaming_dataloader(self, sample_dna_sequences, temp_data_dir):
        """Test streaming data loader for large datasets."""
        sequences, labels = sample_dna_sequences

        # Create multiple FASTA files to simulate streaming
        for i in range(3):
            fasta_path = Path(temp_data_dir) / f"chunk_{i}.fasta"
            with open(fasta_path, "w") as f:
                for j, seq in enumerate(sequences):
                    f.write(f">seq_{i}_{j}\n{seq}\n")

        file_paths = list(Path(temp_data_dir).glob("*.fasta"))
        tokenizer = DNATokenizer(vocab_size=6, sequence_length=32)

        # Test StreamingDataLoader
        streaming_loader = StreamingDataLoader(
            file_paths=file_paths,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=32,
            buffer_size=10,
        )

        # Test iteration
        batch_count = 0
        for batch in streaming_loader:
            assert hasattr(batch, "input_ids")
            assert hasattr(batch, "attention_mask")
            assert batch.input_ids.shape[0] <= 2
            batch_count += 1
            if batch_count >= 3:  # Test a few batches
                break

        assert batch_count > 0

    def test_end_to_end_pipeline(self, sample_dna_sequences, config):
        """Test complete end-to-end data pipeline."""
        sequences, labels = sample_dna_sequences

        # 1. Preprocessing
        preprocessor = GenomicPreprocessor(sequence_type="dna", quality_threshold=0.8)

        # 2. Tokenization
        tokenizer = DNATokenizer(vocab_size=6, sequence_length=32)

        # 3. Dataset creation
        dataset = GenomicDataset(
            sequences=sequences, labels=labels, tokenizer=tokenizer, max_length=32
        )

        # 4. Data loading with custom collation
        dataloader = GenomicDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
            tokenizer=tokenizer,
            max_length=32,
            optimization_level="balanced",
        )

        # 5. Test complete pipeline
        batch_count = 0
        total_samples = 0

        for batch in dataloader:
            # Verify batch structure
            assert hasattr(batch, "input_ids")
            assert hasattr(batch, "attention_mask")
            assert hasattr(batch, "labels")

            # Verify tensor properties
            assert batch.input_ids.dtype == torch.long
            assert batch.attention_mask.dtype == torch.long
            assert batch.labels.dtype == torch.long

            # Verify shapes
            batch_size = batch.input_ids.shape[0]
            assert batch.attention_mask.shape[0] == batch_size
            assert batch.labels.shape[0] == batch_size

            total_samples += batch_size
            batch_count += 1

        assert batch_count > 0
        assert total_samples == len(sequences)


if __name__ == "__main__":
    # Run basic smoke test
    test_instance = TestDataIntegration()

    # Sample data
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "ATGCATGCATGC"]
    labels = [0, 1, 0]

    # Basic config
    config = HyenaGLTConfig(
        vocab_size=6, hidden_size=32, max_position_embeddings=64, num_layers=2
    )

    print("🧪 Running basic integration tests...")

    try:
        test_instance.test_tokenizer_dataset_integration((sequences, labels), config)
        print("✅ Tokenizer-Dataset integration test passed")

        test_instance.test_collator_integration((sequences, labels), config)
        print("✅ Collator integration test passed")

        test_instance.test_dataloader_integration((sequences, labels), config)
        print("✅ DataLoader integration test passed")

        test_instance.test_end_to_end_pipeline((sequences, labels), config)
        print("✅ End-to-end pipeline test passed")

        print("\n🎉 All integration tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
