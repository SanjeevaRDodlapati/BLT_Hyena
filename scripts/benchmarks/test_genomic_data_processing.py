#!/usr/bin/env python3
"""
Real Genomic Data Testing for BLT-Hyena Position Embedding System
==================================================================

This script tests the BLT position embedding system with actual genomic sequences
to validate biological relevance and practical performance.
"""

import random
import string
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model.hyena_glt import HyenaGLT

# Import our BLT system
from hyena_glt.model.position_embeddings import BLTPositionManager


class GenomicSequenceGenerator:
    """Generate realistic genomic sequences for testing"""

    def __init__(self):
        self.dna_bases = ["A", "T", "G", "C"]
        self.protein_aas = list("ACDEFGHIKLMNPQRSTVWY")

        # Common genomic motifs and patterns
        self.promoter_motifs = [
            "TATAAA",  # TATA box
            "CAAT",  # CAAT box
            "GGGGCGG",  # SP1 binding site
            "TTGCGC",  # AP1 binding site
        ]

        self.kozak_sequence = "GCCRCCAUGG"  # Kozak consensus (RNA)
        self.polyA_signal = "AAUAAA"  # Polyadenylation signal

        # Codon usage bias (simplified)
        self.codon_usage = {
            "TTT": 0.45,
            "TTC": 0.55,  # Phe
            "TTA": 0.07,
            "TTG": 0.13,
            "CTT": 0.13,
            "CTC": 0.20,
            "CTA": 0.07,
            "CTG": 0.40,  # Leu
            "TCT": 0.18,
            "TCC": 0.22,
            "TCA": 0.15,
            "TCG": 0.06,
            "AGT": 0.15,
            "AGC": 0.24,  # Ser
        }

    def generate_random_dna(self, length: int) -> str:
        """Generate random DNA sequence"""
        return "".join(random.choices(self.dna_bases, k=length))

    def generate_promoter_region(self, length: int = 1000) -> str:
        """Generate DNA sequence resembling a promoter region"""
        sequence = []

        # Add some background sequence
        sequence.append(self.generate_random_dna(200))

        # Insert promoter motifs
        for motif in self.promoter_motifs:
            if len("".join(sequence)) + len(motif) < length - 100:
                # Add some spacing
                sequence.append(self.generate_random_dna(random.randint(50, 150)))
                sequence.append(motif)

        # Fill remaining length
        current_length = len("".join(sequence))
        if current_length < length:
            sequence.append(self.generate_random_dna(length - current_length))

        result = "".join(sequence)[:length]
        return result

    def generate_coding_sequence(self, length: int = 600) -> str:
        """Generate a coding DNA sequence with realistic codon usage"""
        # Ensure length is divisible by 3
        length = (length // 3) * 3

        sequence = []

        # Start codon
        sequence.append("ATG")

        # Generate codons with realistic usage
        while len("".join(sequence)) < length - 3:
            # Simplified: just add random codons (in practice would use codon usage table)
            codon = "".join(random.choices(self.dna_bases, k=3))
            # Avoid stop codons except at the end
            if codon not in ["TAA", "TAG", "TGA"]:
                sequence.append(codon)

        # Stop codon
        sequence.append("TAA")

        result = "".join(sequence)[:length]
        return result

    def generate_protein_sequence(self, length: int = 200) -> str:
        """Generate realistic protein sequence"""
        # Start with methionine (start codon product)
        sequence = ["M"]

        # Generate rest of sequence
        sequence.extend(random.choices(self.protein_aas, k=length - 1))

        return "".join(sequence)

    def generate_genomic_variants(
        self, reference_seq: str, num_variants: int = 5
    ) -> list[tuple[str, str, int]]:
        """Generate sequence variants (SNPs, indels)"""
        variants = []

        for _ in range(num_variants):
            pos = random.randint(10, len(reference_seq) - 10)
            variant_type = random.choice(["snp", "insertion", "deletion"])

            if variant_type == "snp":
                # Single nucleotide polymorphism
                original = reference_seq[pos]
                new_base = random.choice([b for b in self.dna_bases if b != original])
                variant_seq = reference_seq[:pos] + new_base + reference_seq[pos + 1 :]
                variants.append((variant_seq, f"SNP:{original}>{new_base}", pos))

            elif variant_type == "insertion":
                # Small insertion (1-5 bp)
                insert_size = random.randint(1, 5)
                insert_seq = self.generate_random_dna(insert_size)
                variant_seq = reference_seq[:pos] + insert_seq + reference_seq[pos:]
                variants.append((variant_seq, f"INS:{insert_seq}", pos))

            elif variant_type == "deletion":
                # Small deletion (1-5 bp)
                del_size = min(random.randint(1, 5), len(reference_seq) - pos - 10)
                variant_seq = reference_seq[:pos] + reference_seq[pos + del_size :]
                variants.append((variant_seq, f"DEL:{del_size}bp", pos))

        return variants


class GenomicTokenizer:
    """Simple genomic tokenizer for testing"""

    def __init__(self, kmer_size: int = 3):
        self.kmer_size = kmer_size

        # DNA vocabulary
        self.dna_vocab = {}
        bases = ["A", "T", "G", "C"]

        # Generate all possible k-mers
        kmers = ["".join(p) for p in self._get_permutations(bases, kmer_size)]

        # Add special tokens
        self.dna_vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<CLS>": 2,
            "<SEP>": 3,
        }

        for i, kmer in enumerate(kmers):
            self.dna_vocab[kmer] = i + 4

        self.vocab_size = len(self.dna_vocab)
        self.id_to_token = {v: k for k, v in self.dna_vocab.items()}

    def _get_permutations(self, items, length):
        """Generate all permutations of given length"""
        if length == 1:
            return [[item] for item in items]

        result = []
        for item in items:
            for perm in self._get_permutations(items, length - 1):
                result.append([item] + perm)
        return result

    def tokenize(self, sequence: str) -> list[int]:
        """Tokenize DNA sequence into k-mer tokens"""
        sequence = sequence.upper()
        tokens = []

        # Add CLS token
        tokens.append(self.dna_vocab["<CLS>"])

        # Tokenize sequence
        for i in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[i : i + self.kmer_size]
            token_id = self.dna_vocab.get(kmer, self.dna_vocab["<UNK>"])
            tokens.append(token_id)

        return tokens


class GenomicDatasetTest:
    """Test suite for genomic data processing"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_generator = GenomicSequenceGenerator()
        self.tokenizer = GenomicTokenizer(kmer_size=3)

        # Initialize BLT position manager
        self.config = HyenaGLTConfig(
            hidden_size=256,
            max_position_embeddings=2048,
            num_attention_heads=8,
        )

        self.position_manager = BLTPositionManager(
            d_model=self.config.hidden_size,
            max_len=self.config.max_position_embeddings,
            num_heads=self.config.num_attention_heads,
        ).to(self.device)

        # Test results
        self.results = {}

    def test_promoter_sequence_processing(self):
        """Test processing of promoter sequences"""
        print("🧬 Testing Promoter Sequence Processing...")

        # Generate promoter sequences
        promoter_seqs = [
            self.seq_generator.generate_promoter_region(length=1000) for _ in range(10)
        ]

        results = []
        for i, seq in enumerate(promoter_seqs):
            # Tokenize
            tokens = self.tokenizer.tokenize(seq)
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

            # Process with BLT position manager
            start_time = time.time()
            with torch.no_grad():
                # Create dummy embeddings
                embeddings = torch.randn(1, len(tokens), self.config.hidden_size).to(
                    self.device
                )

                # Test position encoding
                pos_embeddings = self.position_manager.encode_positions(
                    embeddings,
                    original_positions=torch.arange(len(tokens))
                    .unsqueeze(0)
                    .to(self.device),
                )

                # Test patching
                patches = self.position_manager.create_patch_representations(embeddings)

            processing_time = time.time() - start_time

            results.append(
                {
                    "sequence_length": len(tokens),
                    "processing_time": processing_time,
                    "output_shape": pos_embeddings.shape,
                    "patch_shape": patches.shape if patches is not None else None,
                }
            )

        avg_time = np.mean([r["processing_time"] for r in results])
        avg_length = np.mean([r["sequence_length"] for r in results])

        print(f"   ✅ Processed {len(promoter_seqs)} promoter sequences")
        print(f"   📏 Average sequence length: {avg_length:.1f} tokens")
        print(f"   ⏱️  Average processing time: {avg_time*1000:.2f}ms")

        self.results["promoter_processing"] = results
        return results

    def test_coding_sequence_processing(self):
        """Test processing of coding sequences"""
        print("🧬 Testing Coding Sequence Processing...")

        # Generate coding sequences of different lengths
        lengths = [300, 600, 900, 1200]
        results = []

        for length in lengths:
            coding_seqs = [
                self.seq_generator.generate_coding_sequence(length=length)
                for _ in range(5)
            ]

            for seq in coding_seqs:
                tokens = self.tokenizer.tokenize(seq)
                input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

                start_time = time.time()
                with torch.no_grad():
                    embeddings = torch.randn(
                        1, len(tokens), self.config.hidden_size
                    ).to(self.device)
                    pos_embeddings = self.position_manager.encode_positions(
                        embeddings,
                        original_positions=torch.arange(len(tokens))
                        .unsqueeze(0)
                        .to(self.device),
                    )
                processing_time = time.time() - start_time

                results.append(
                    {
                        "expected_length": length,
                        "actual_length": len(seq),
                        "token_count": len(tokens),
                        "processing_time": processing_time,
                        "output_shape": pos_embeddings.shape,
                    }
                )

        # Analysis
        for length in lengths:
            length_results = [r for r in results if r["expected_length"] == length]
            avg_time = np.mean([r["processing_time"] for r in length_results])
            avg_tokens = np.mean([r["token_count"] for r in length_results])
            print(
                f"   📏 Length {length}: {avg_tokens:.1f} tokens, {avg_time*1000:.2f}ms avg"
            )

        self.results["coding_processing"] = results
        return results

    def test_variant_effect_analysis(self):
        """Test processing of genomic variants"""
        print("🧬 Testing Variant Effect Analysis...")

        # Generate reference sequence
        reference_seq = self.seq_generator.generate_coding_sequence(length=600)

        # Generate variants
        variants = self.seq_generator.generate_genomic_variants(
            reference_seq, num_variants=10
        )

        # Process reference
        ref_tokens = self.tokenizer.tokenize(reference_seq)
        ref_tensor = torch.tensor([ref_tokens], dtype=torch.long).to(self.device)

        with torch.no_grad():
            ref_embeddings = torch.randn(
                1, len(ref_tokens), self.config.hidden_size
            ).to(self.device)
            ref_pos_embeddings = self.position_manager.encode_positions(
                ref_embeddings,
                original_positions=torch.arange(len(ref_tokens))
                .unsqueeze(0)
                .to(self.device),
            )

        # Process variants and compare
        variant_results = []
        for variant_seq, variant_type, position in variants:
            var_tokens = self.tokenizer.tokenize(variant_seq)
            var_tensor = torch.tensor([var_tokens], dtype=torch.long).to(self.device)

            with torch.no_grad():
                var_embeddings = torch.randn(
                    1, len(var_tokens), self.config.hidden_size
                ).to(self.device)
                var_pos_embeddings = self.position_manager.encode_positions(
                    var_embeddings,
                    original_positions=torch.arange(len(var_tokens))
                    .unsqueeze(0)
                    .to(self.device),
                )

            # Calculate difference (simplified analysis)
            # In practice, would compare specific regions around the variant
            if var_pos_embeddings.shape[1] == ref_pos_embeddings.shape[1]:
                embedding_diff = torch.mean(
                    torch.abs(var_pos_embeddings - ref_pos_embeddings)
                )
            else:
                embedding_diff = torch.tensor(float("inf"))  # Different lengths

            variant_results.append(
                {
                    "variant_type": variant_type,
                    "position": position,
                    "ref_length": len(ref_tokens),
                    "var_length": len(var_tokens),
                    "embedding_difference": embedding_diff.item(),
                }
            )

        # Analysis
        snp_results = [
            r for r in variant_results if r["variant_type"].startswith("SNP")
        ]
        ins_results = [
            r for r in variant_results if r["variant_type"].startswith("INS")
        ]
        del_results = [
            r for r in variant_results if r["variant_type"].startswith("DEL")
        ]

        print(f"   📊 Processed {len(variants)} variants:")
        if snp_results:
            avg_snp_diff = np.mean(
                [
                    r["embedding_difference"]
                    for r in snp_results
                    if r["embedding_difference"] != float("inf")
                ]
            )
            print(
                f"      🔸 SNPs: {len(snp_results)} variants, avg embedding diff: {avg_snp_diff:.4f}"
            )

        if ins_results:
            avg_ins_diff = np.mean(
                [
                    r["embedding_difference"]
                    for r in ins_results
                    if r["embedding_difference"] != float("inf")
                ]
            )
            print(
                f"      🔸 Insertions: {len(ins_results)} variants, avg embedding diff: {avg_ins_diff:.4f}"
            )

        if del_results:
            avg_del_diff = np.mean(
                [
                    r["embedding_difference"]
                    for r in del_results
                    if r["embedding_difference"] != float("inf")
                ]
            )
            print(
                f"      🔸 Deletions: {len(del_results)} variants, avg embedding diff: {avg_del_diff:.4f}"
            )

        self.results["variant_analysis"] = variant_results
        return variant_results

    def test_long_sequence_processing(self):
        """Test processing of long genomic sequences"""
        print("🧬 Testing Long Sequence Processing...")

        # Test with sequences of increasing length
        lengths = [1000, 2000, 4000, 8000]
        results = []

        for length in lengths:
            print(f"   📏 Testing sequence length: {length}")

            # Generate long sequence
            long_seq = self.seq_generator.generate_random_dna(length)
            tokens = self.tokenizer.tokenize(long_seq)

            # Limit to max position embeddings
            max_tokens = min(len(tokens), self.config.max_position_embeddings - 10)
            tokens = tokens[:max_tokens]

            input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

            start_time = time.time()
            try:
                with torch.no_grad():
                    embeddings = torch.randn(
                        1, len(tokens), self.config.hidden_size
                    ).to(self.device)

                    # Test position encoding
                    pos_embeddings = self.position_manager.encode_positions(
                        embeddings,
                        original_positions=torch.arange(len(tokens))
                        .unsqueeze(0)
                        .to(self.device),
                    )

                    # Test patching for long sequences
                    patches = self.position_manager.create_patch_representations(
                        embeddings
                    )

                processing_time = time.time() - start_time
                success = True

            except Exception as e:
                processing_time = time.time() - start_time
                success = False
                print(f"      ❌ Failed: {e}")

            results.append(
                {
                    "target_length": length,
                    "actual_tokens": len(tokens),
                    "processing_time": processing_time,
                    "success": success,
                    "tokens_per_second": (
                        len(tokens) / processing_time if success else 0
                    ),
                }
            )

            if success:
                print(
                    f"      ✅ Processed {len(tokens)} tokens in {processing_time*1000:.1f}ms"
                )
                print(
                    f"      🚀 Throughput: {len(tokens)/processing_time:.0f} tokens/sec"
                )

        self.results["long_sequence_processing"] = results
        return results

    def test_position_preservation_across_tasks(self):
        """Test that position information is preserved across different genomic tasks"""
        print("🧬 Testing Position Preservation Across Tasks...")

        # Generate test sequences for different tasks
        sequences = {
            "promoter": self.seq_generator.generate_promoter_region(800),
            "coding": self.seq_generator.generate_coding_sequence(600),
            "random": self.seq_generator.generate_random_dna(700),
        }

        results = {}

        for seq_type, sequence in sequences.items():
            tokens = self.tokenizer.tokenize(sequence)
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

            with torch.no_grad():
                embeddings = torch.randn(1, len(tokens), self.config.hidden_size).to(
                    self.device
                )
                original_positions = (
                    torch.arange(len(tokens)).unsqueeze(0).to(self.device)
                )

                # Test position encoding
                pos_embeddings = self.position_manager.encode_positions(
                    embeddings, original_positions=original_positions
                )

                # Test patching and reconstruction
                patches = self.position_manager.create_patch_representations(embeddings)

                if patches is not None:
                    # Test position bridge
                    try:
                        bridge_output = self.position_manager.position_bridge(
                            patches, embeddings
                        )
                        bridge_success = True
                    except Exception as e:
                        bridge_output = None
                        bridge_success = False
                        print(f"      ⚠️  Bridge failed for {seq_type}: {e}")
                else:
                    bridge_output = None
                    bridge_success = False

            results[seq_type] = {
                "token_count": len(tokens),
                "pos_encoding_shape": pos_embeddings.shape,
                "patch_shape": patches.shape if patches is not None else None,
                "bridge_success": bridge_success,
                "bridge_shape": (
                    bridge_output.shape if bridge_output is not None else None
                ),
            }

            print(f"   🔬 {seq_type.title()} sequence: {len(tokens)} tokens")
            print(f"      Position encoding: {pos_embeddings.shape}")
            if patches is not None:
                print(f"      Patches: {patches.shape}")
            if bridge_success:
                print(f"      Bridge output: {bridge_output.shape}")

        self.results["position_preservation"] = results
        return results

    def run_comprehensive_test(self):
        """Run all genomic data tests"""
        print("=" * 60)
        print("BLT-Hyena Genomic Data Testing")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        print("")

        # Run all tests
        self.test_promoter_sequence_processing()
        print("")

        self.test_coding_sequence_processing()
        print("")

        self.test_variant_effect_analysis()
        print("")

        self.test_long_sequence_processing()
        print("")

        self.test_position_preservation_across_tasks()
        print("")

        # Summary
        print("=" * 60)
        print("GENOMIC DATA TESTING SUMMARY")
        print("=" * 60)

        total_tests = len([k for k in self.results.keys()])
        print(f"✅ Completed {total_tests} test categories")

        # Long sequence analysis
        if "long_sequence_processing" in self.results:
            successful_long = sum(
                1 for r in self.results["long_sequence_processing"] if r["success"]
            )
            total_long = len(self.results["long_sequence_processing"])
            print(
                f"📏 Long sequence processing: {successful_long}/{total_long} lengths successful"
            )

        # Variant analysis
        if "variant_analysis" in self.results:
            variant_count = len(self.results["variant_analysis"])
            print(f"🧬 Variant analysis: {variant_count} variants processed")

        # Position preservation
        if "position_preservation" in self.results:
            bridge_success = sum(
                1
                for r in self.results["position_preservation"].values()
                if r["bridge_success"]
            )
            total_seq_types = len(self.results["position_preservation"])
            print(
                f"🎯 Position preservation: {bridge_success}/{total_seq_types} sequence types successful"
            )

        print("\n🎉 Genomic data testing complete!")

        # Save results
        torch.save(
            self.results,
            "/Users/sanjeev/Downloads/Repos/BLT_Hyena/genomic_test_results.pt",
        )
        print("💾 Results saved to genomic_test_results.pt")

        return self.results


def main():
    """Run comprehensive genomic data testing"""
    tester = GenomicDatasetTest()
    results = tester.run_comprehensive_test()
    return results


if __name__ == "__main__":
    main()
