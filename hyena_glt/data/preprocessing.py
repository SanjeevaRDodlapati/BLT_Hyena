"""Advanced genomic data preprocessing and augmentation utilities."""

import random
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .utils import calculate_gc_content, reverse_complement, translate_dna


class GenomicPreprocessor:
    """Advanced preprocessor for genomic sequences with quality control and standardization."""
    
    def __init__(
        self,
        sequence_type: str = "dna",
        min_length: int = 50,
        max_length: int = 10000,
        quality_threshold: float = 20.0,
        max_n_ratio: float = 0.1,
        remove_duplicates: bool = True,
        normalize_case: bool = True,
        filter_non_standard: bool = True
    ):
        self.sequence_type = sequence_type.lower()
        self.min_length = min_length
        self.max_length = max_length
        self.quality_threshold = quality_threshold
        self.max_n_ratio = max_n_ratio
        self.remove_duplicates = remove_duplicates
        self.normalize_case = normalize_case
        self.filter_non_standard = filter_non_standard
        
        # Define valid characters for each sequence type
        self.valid_chars = {
            "dna": set("ATCGN"),
            "rna": set("AUCGN"),
            "protein": set("ACDEFGHIKLMNPQRSTVWYX")
        }
        
        self.stats = {
            "total_sequences": 0,
            "filtered_length": 0,
            "filtered_quality": 0,
            "filtered_n_content": 0,
            "filtered_invalid_chars": 0,
            "duplicates_removed": 0,
            "final_sequences": 0
        }
    
    def preprocess_sequences(
        self,
        sequences: List[Union[str, SeqRecord]],
        qualities: Optional[List[List[float]]] = None,
        headers: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Preprocess a batch of genomic sequences with comprehensive quality control.
        
        Args:
            sequences: List of sequences (strings or SeqRecord objects)
            qualities: Optional quality scores for each sequence
            headers: Optional headers/identifiers for sequences
            
        Returns:
            Tuple of (processed_sequences, processed_headers, statistics)
        """
        self.stats = {key: 0 for key in self.stats.keys()}
        
        # Convert inputs to standardized format
        seq_strings = []
        seq_headers = []
        seq_qualities = qualities or [None] * len(sequences)
        
        for i, seq in enumerate(sequences):
            if isinstance(seq, SeqRecord):
                seq_strings.append(str(seq.seq))
                seq_headers.append(seq.id if seq.id else f"seq_{i}")
            else:
                seq_strings.append(str(seq))
                seq_headers.append(headers[i] if headers and i < len(headers) else f"seq_{i}")
        
        self.stats["total_sequences"] = len(seq_strings)
        
        processed_sequences = []
        processed_headers = []
        seen_sequences = set() if self.remove_duplicates else None
        
        for seq, header, quality in zip(seq_strings, seq_headers, seq_qualities):
            # Apply preprocessing pipeline
            result = self._preprocess_single_sequence(seq, quality)
            
            if result is None:
                continue
                
            processed_seq = result
            
            # Check for duplicates
            if self.remove_duplicates:
                if processed_seq in seen_sequences:
                    self.stats["duplicates_removed"] += 1
                    continue
                seen_sequences.add(processed_seq)
            
            processed_sequences.append(processed_seq)
            processed_headers.append(header)
        
        self.stats["final_sequences"] = len(processed_sequences)
        
        return processed_sequences, processed_headers, self.stats.copy()
    
    def _preprocess_single_sequence(
        self,
        sequence: str,
        quality: Optional[List[float]] = None
    ) -> Optional[str]:
        """Preprocess a single sequence through the quality control pipeline."""
        
        # Normalize case
        if self.normalize_case:
            sequence = sequence.upper()
        
        # Remove whitespace and newlines
        sequence = re.sub(r'\s+', '', sequence)
        
        # Length filtering
        if len(sequence) < self.min_length or len(sequence) > self.max_length:
            self.stats["filtered_length"] += 1
            return None
        
        # Quality filtering (if quality scores provided)
        if quality is not None:
            avg_quality = np.mean(quality)
            if avg_quality < self.quality_threshold:
                self.stats["filtered_quality"] += 1
                return None
            
            # Filter low-quality regions
            sequence = self._filter_by_quality(sequence, quality)
            if len(sequence) < self.min_length:
                self.stats["filtered_length"] += 1
                return None
        
        # Handle sequence-specific preprocessing
        if self.sequence_type == "dna":
            sequence = self._preprocess_dna(sequence)
        elif self.sequence_type == "rna":
            sequence = self._preprocess_rna(sequence)
        elif self.sequence_type == "protein":
            sequence = self._preprocess_protein(sequence)
        
        if sequence is None:
            return None
        
        # N/X content filtering
        if self.sequence_type in ["dna", "rna"]:
            n_ratio = sequence.count('N') / len(sequence)
        else:
            n_ratio = sequence.count('X') / len(sequence)
        
        if n_ratio > self.max_n_ratio:
            self.stats["filtered_n_content"] += 1
            return None
        
        # Invalid character filtering
        if self.filter_non_standard:
            valid_chars = self.valid_chars[self.sequence_type]
            if not set(sequence).issubset(valid_chars):
                self.stats["filtered_invalid_chars"] += 1
                return None
        
        return sequence
    
    def _preprocess_dna(self, sequence: str) -> Optional[str]:
        """DNA-specific preprocessing."""
        # Replace ambiguous nucleotides with N
        sequence = re.sub(r'[^ATCGN]', 'N', sequence)
        return sequence
    
    def _preprocess_rna(self, sequence: str) -> Optional[str]:
        """RNA-specific preprocessing."""
        # Convert T to U
        sequence = sequence.replace('T', 'U')
        # Replace ambiguous nucleotides with N
        sequence = re.sub(r'[^AUCGN]', 'N', sequence)
        return sequence
    
    def _preprocess_protein(self, sequence: str) -> Optional[str]:
        """Protein-specific preprocessing."""
        # Replace ambiguous amino acids with X
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', sequence)
        return sequence
    
    def _filter_by_quality(
        self,
        sequence: str,
        quality: List[float]
    ) -> str:
        """Filter sequence based on quality scores."""
        if len(quality) != len(sequence):
            warnings.warn("Quality and sequence length mismatch, skipping quality filtering")
            return sequence
        
        # Keep only high-quality positions
        filtered_chars = []
        for char, qual in zip(sequence, quality):
            if qual >= self.quality_threshold:
                filtered_chars.append(char)
        
        return ''.join(filtered_chars)
    
    def preprocess_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        file_format: str = "auto"
    ) -> Dict[str, Any]:
        """
        Preprocess sequences from a file.
        
        Args:
            input_path: Path to input file (FASTA, FASTQ, etc.)
            output_path: Optional output path
            file_format: File format ('fasta', 'fastq', or 'auto')
            
        Returns:
            Dictionary with preprocessing statistics and results
        """
        input_path = Path(input_path)
        
        # Auto-detect format
        if file_format == "auto":
            file_format = self._detect_file_format(input_path)
        
        # Load sequences
        sequences, headers, qualities = self._load_sequences(input_path, file_format)
        
        # Preprocess
        processed_seqs, processed_headers, stats = self.preprocess_sequences(
            sequences, qualities, headers
        )
        
        # Save if output path provided
        if output_path:
            self._save_sequences(processed_seqs, processed_headers, output_path, file_format)
        
        return {
            "sequences": processed_seqs,
            "headers": processed_headers,
            "statistics": stats,
            "input_file": str(input_path),
            "output_file": str(output_path) if output_path else None
        }
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Auto-detect file format based on extension and content."""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.fa', '.fasta', '.fas']:
            return 'fasta'
        elif suffix in ['.fq', '.fastq']:
            return 'fastq'
        else:
            # Try to detect from content
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('>'):
                        return 'fasta'
                    elif first_line.startswith('@'):
                        return 'fastq'
            except Exception:
                pass
        
        return 'fasta'  # Default fallback
    
    def _load_sequences(
        self,
        file_path: Path,
        file_format: str
    ) -> Tuple[List[str], List[str], List[Optional[List[float]]]]:
        """Load sequences from file."""
        sequences = []
        headers = []
        qualities = []
        
        try:
            if file_format == 'fasta':
                for record in SeqIO.parse(file_path, 'fasta'):
                    sequences.append(str(record.seq))
                    headers.append(record.id)
                    qualities.append(None)
            elif file_format == 'fastq':
                for record in SeqIO.parse(file_path, 'fastq'):
                    sequences.append(str(record.seq))
                    headers.append(record.id)
                    # Convert Phred scores to numeric
                    qual_scores = [ord(q) - 33 for q in record.letter_annotations['phred_quality']]
                    qualities.append(qual_scores)
        except Exception as e:
            warnings.warn(f"Error loading file {file_path}: {e}")
            # Fallback to simple text parsing
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if content.startswith('>'):
                    # Simple FASTA parsing
                    entries = content.split('>')[1:]
                    for i, entry in enumerate(entries):
                        lines = entry.strip().split('\n')
                        header = lines[0]
                        sequence = ''.join(lines[1:])
                        sequences.append(sequence)
                        headers.append(header)
                        qualities.append(None)
        
        return sequences, headers, qualities
    
    def _save_sequences(
        self,
        sequences: List[str],
        headers: List[str],
        output_path: Union[str, Path],
        file_format: str
    ):
        """Save processed sequences to file."""
        output_path = Path(output_path)
        
        records = []
        for seq, header in zip(sequences, headers):
            record = SeqRecord(Seq(seq), id=header, description="")
            records.append(record)
        
        try:
            SeqIO.write(records, output_path, file_format)
        except Exception as e:
            warnings.warn(f"Error saving with BioPython: {e}. Falling back to simple format.")
            # Fallback to simple text format
            with open(output_path, 'w') as f:
                for seq, header in zip(sequences, headers):
                    f.write(f">{header}\n{seq}\n")


class SequenceAugmenter:
    """Advanced sequence augmentation for genomic data."""
    
    def __init__(
        self,
        sequence_type: str = "dna",
        augmentation_ratio: float = 0.1,
        seed: Optional[int] = None
    ):
        self.sequence_type = sequence_type.lower()
        self.augmentation_ratio = augmentation_ratio
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Define augmentation strategies for each sequence type
        self.strategies = {
            "dna": [
                "reverse_complement",
                "random_mutation",
                "random_insertion",
                "random_deletion",
                "gc_bias_shift",
                "motif_masking"
            ],
            "rna": [
                "random_mutation",
                "random_insertion", 
                "random_deletion",
                "gc_bias_shift",
                "secondary_structure_perturbation"
            ],
            "protein": [
                "conservative_mutation",
                "random_insertion",
                "random_deletion",
                "domain_shuffling",
                "hydrophobicity_shift"
            ]
        }
    
    def augment_sequences(
        self,
        sequences: List[str],
        labels: Optional[List[Any]] = None,
        num_augmentations: int = 1
    ) -> Tuple[List[str], List[Any]]:
        """
        Apply augmentation to a list of sequences.
        
        Args:
            sequences: List of input sequences
            labels: Optional labels for sequences
            num_augmentations: Number of augmented versions per sequence
            
        Returns:
            Tuple of (augmented_sequences, augmented_labels)
        """
        augmented_sequences = []
        augmented_labels = []
        
        for i, seq in enumerate(sequences):
            # Add original sequence
            augmented_sequences.append(seq)
            if labels:
                augmented_labels.append(labels[i])
            
            # Add augmented versions
            for _ in range(num_augmentations):
                aug_seq = self._augment_single_sequence(seq)
                augmented_sequences.append(aug_seq)
                
                if labels:
                    # For most tasks, augmented sequences keep the same label
                    augmented_labels.append(labels[i])
        
        return augmented_sequences, augmented_labels
    
    def _augment_single_sequence(self, sequence: str) -> str:
        """Apply a random augmentation to a single sequence."""
        available_strategies = self.strategies[self.sequence_type]
        strategy = random.choice(available_strategies)
        
        if strategy == "reverse_complement" and self.sequence_type == "dna":
            return self._reverse_complement_augment(sequence)
        elif strategy == "random_mutation":
            return self._random_mutation(sequence)
        elif strategy == "random_insertion":
            return self._random_insertion(sequence)
        elif strategy == "random_deletion":
            return self._random_deletion(sequence)
        elif strategy == "gc_bias_shift":
            return self._gc_bias_shift(sequence)
        elif strategy == "motif_masking":
            return self._motif_masking(sequence)
        elif strategy == "conservative_mutation":
            return self._conservative_mutation(sequence)
        elif strategy == "domain_shuffling":
            return self._domain_shuffling(sequence)
        else:
            # Fallback to random mutation
            return self._random_mutation(sequence)
    
    def _reverse_complement_augment(self, sequence: str) -> str:
        """Generate reverse complement (DNA only)."""
        return reverse_complement(sequence)
    
    def _random_mutation(self, sequence: str) -> str:
        """Apply random point mutations."""
        if len(sequence) == 0:
            return sequence
        
        sequence_list = list(sequence)
        num_mutations = max(1, int(len(sequence) * self.augmentation_ratio))
        
        # Define possible substitutions
        if self.sequence_type == "dna":
            alphabet = "ATCG"
        elif self.sequence_type == "rna":
            alphabet = "AUCG"
        else:  # protein
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
        
        for _ in range(num_mutations):
            pos = random.randint(0, len(sequence_list) - 1)
            original_char = sequence_list[pos]
            
            # Choose a different character
            new_chars = [c for c in alphabet if c != original_char]
            if new_chars:
                sequence_list[pos] = random.choice(new_chars)
        
        return ''.join(sequence_list)
    
    def _random_insertion(self, sequence: str) -> str:
        """Apply random insertions."""
        if len(sequence) == 0:
            return sequence
        
        sequence_list = list(sequence)
        num_insertions = max(1, int(len(sequence) * self.augmentation_ratio))
        
        if self.sequence_type == "dna":
            alphabet = "ATCG"
        elif self.sequence_type == "rna":
            alphabet = "AUCG"
        else:
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
        
        for _ in range(num_insertions):
            pos = random.randint(0, len(sequence_list))
            new_char = random.choice(alphabet)
            sequence_list.insert(pos, new_char)
        
        return ''.join(sequence_list)
    
    def _random_deletion(self, sequence: str) -> str:
        """Apply random deletions."""
        if len(sequence) <= 1:
            return sequence
        
        sequence_list = list(sequence)
        num_deletions = max(1, min(len(sequence) // 2, int(len(sequence) * self.augmentation_ratio)))
        
        for _ in range(num_deletions):
            if len(sequence_list) > 1:
                pos = random.randint(0, len(sequence_list) - 1)
                sequence_list.pop(pos)
        
        return ''.join(sequence_list)
    
    def _gc_bias_shift(self, sequence: str) -> str:
        """Shift GC content while preserving some sequence properties."""
        if self.sequence_type not in ["dna", "rna"]:
            return self._random_mutation(sequence)
        
        sequence_list = list(sequence)
        gc_chars = ["G", "C"] if self.sequence_type == "dna" else ["G", "C"]
        at_chars = ["A", "T"] if self.sequence_type == "dna" else ["A", "U"]
        
        # Randomly convert some AT to GC or vice versa
        num_changes = max(1, int(len(sequence) * self.augmentation_ratio))
        
        for _ in range(num_changes):
            pos = random.randint(0, len(sequence_list) - 1)
            char = sequence_list[pos]
            
            if char in at_chars:
                sequence_list[pos] = random.choice(gc_chars)
            elif char in gc_chars:
                sequence_list[pos] = random.choice(at_chars)
        
        return ''.join(sequence_list)
    
    def _motif_masking(self, sequence: str) -> str:
        """Mask common motifs with N/X characters."""
        if self.sequence_type not in ["dna", "rna"]:
            return self._random_mutation(sequence)
        
        # Common motifs to mask
        motifs = ["TATA", "CAAT", "GGCC", "ATG", "TAG", "TAA", "TGA"]
        if self.sequence_type == "rna":
            motifs = [motif.replace("T", "U") for motif in motifs]
        
        sequence_list = list(sequence)
        mask_char = "N" if self.sequence_type in ["dna", "rna"] else "X"
        
        for motif in motifs:
            if motif in sequence:
                # Randomly decide whether to mask this motif
                if random.random() < self.augmentation_ratio:
                    start = sequence.find(motif)
                    if start != -1:
                        for i in range(start, start + len(motif)):
                            if i < len(sequence_list):
                                sequence_list[i] = mask_char
        
        return ''.join(sequence_list)
    
    def _conservative_mutation(self, sequence: str) -> str:
        """Apply conservative mutations for protein sequences."""
        if self.sequence_type != "protein":
            return self._random_mutation(sequence)
        
        # Conservative substitution groups
        conservative_groups = {
            "A": ["A", "G", "S", "T"],
            "R": ["R", "K", "H"],
            "N": ["N", "D", "Q", "E"],
            "D": ["D", "E", "N", "Q"],
            "C": ["C"],
            "Q": ["Q", "E", "N", "D"],
            "E": ["E", "D", "Q", "N"],
            "G": ["G", "A", "S", "T"],
            "H": ["H", "R", "K"],
            "I": ["I", "L", "V"],
            "L": ["L", "I", "V", "M"],
            "K": ["K", "R", "H"],
            "M": ["M", "L", "I", "V"],
            "F": ["F", "Y", "W"],
            "P": ["P"],
            "S": ["S", "T", "A", "G"],
            "T": ["T", "S", "A", "G"],
            "W": ["W", "F", "Y"],
            "Y": ["Y", "F", "W"],
            "V": ["V", "I", "L"]
        }
        
        sequence_list = list(sequence)
        num_mutations = max(1, int(len(sequence) * self.augmentation_ratio))
        
        for _ in range(num_mutations):
            pos = random.randint(0, len(sequence_list) - 1)
            original_char = sequence_list[pos]
            
            if original_char in conservative_groups:
                conservative_options = [c for c in conservative_groups[original_char] if c != original_char]
                if conservative_options:
                    sequence_list[pos] = random.choice(conservative_options)
        
        return ''.join(sequence_list)
    
    def _domain_shuffling(self, sequence: str) -> str:
        """Shuffle sequence domains (crude approximation)."""
        if len(sequence) < 20:
            return sequence
        
        # Split into rough "domains" and shuffle
        domain_size = len(sequence) // 4
        domains = []
        
        for i in range(0, len(sequence), domain_size):
            domains.append(sequence[i:i + domain_size])
        
        random.shuffle(domains)
        return ''.join(domains)


class MotifExtractor:
    """Extract and analyze biological motifs from sequences."""
    
    def __init__(self, sequence_type: str = "dna"):
        self.sequence_type = sequence_type.lower()
        
        # Common motifs for each sequence type
        self.known_motifs = {
            "dna": {
                "TATA_box": "TATAAA",
                "CAAT_box": "CAAT",
                "GC_box": "GGCC",
                "start_codon": "ATG",
                "stop_codons": ["TAG", "TAA", "TGA"],
                "kozak": "GCCRCCATGG",
                "poly_A": "AATAAA"
            },
            "rna": {
                "kozak": "GCCGCCAUGG",
                "poly_A": "AAUAAA",
                "ribosome_binding": "AGGAGG",
                "hairpin": "GGGGAAACCCCC"
            },
            "protein": {
                "signal_peptide": "MKLLILTCLVAVAL",
                "nuclear_localization": "PKKKRKV",
                "transmembrane": "LVIWGAAFVGFIMIY"
            }
        }
    
    def extract_motifs(
        self,
        sequences: List[str],
        motif_length: int = 6,
        min_occurrences: int = 2
    ) -> Dict[str, int]:
        """
        Extract recurring motifs from sequences.
        
        Args:
            sequences: List of sequences to analyze
            motif_length: Length of motifs to extract
            min_occurrences: Minimum occurrences to consider a motif
            
        Returns:
            Dictionary of motifs and their occurrence counts
        """
        motif_counts = Counter()
        
        for sequence in sequences:
            # Extract all k-mers of specified length
            for i in range(len(sequence) - motif_length + 1):
                motif = sequence[i:i + motif_length]
                motif_counts[motif] += 1
        
        # Filter by minimum occurrences
        filtered_motifs = {
            motif: count for motif, count in motif_counts.items()
            if count >= min_occurrences
        }
        
        return filtered_motifs
    
    def find_known_motifs(self, sequence: str) -> Dict[str, List[Tuple[int, str]]]:
        """
        Find known biological motifs in a sequence.
        
        Args:
            sequence: Sequence to search
            
        Returns:
            Dictionary mapping motif names to list of (position, match) tuples
        """
        found_motifs = {}
        
        known = self.known_motifs.get(self.sequence_type, {})
        
        for motif_name, motif_pattern in known.items():
            if isinstance(motif_pattern, list):
                # Multiple patterns for this motif
                matches = []
                for pattern in motif_pattern:
                    matches.extend(self._find_pattern_matches(sequence, pattern))
                found_motifs[motif_name] = matches
            else:
                # Single pattern
                found_motifs[motif_name] = self._find_pattern_matches(sequence, motif_pattern)
        
        return found_motifs
    
    def _find_pattern_matches(self, sequence: str, pattern: str) -> List[Tuple[int, str]]:
        """Find all matches of a pattern in a sequence."""
        matches = []
        
        # Handle IUPAC ambiguity codes (simplified)
        if 'R' in pattern:  # A or G
            pattern = pattern.replace('R', '[AG]')
        if 'Y' in pattern:  # C or T
            pattern = pattern.replace('Y', '[CT]')
        if 'N' in pattern:  # Any nucleotide
            pattern = pattern.replace('N', '[ATCG]')
        
        try:
            import re
            for match in re.finditer(pattern, sequence):
                matches.append((match.start(), match.group()))
        except re.error:
            # Fallback to exact string matching
            start = 0
            while True:
                pos = sequence.find(pattern, start)
                if pos == -1:
                    break
                matches.append((pos, pattern))
                start = pos + 1
        
        return matches


class QualityController:
    """Quality control utilities for genomic sequences."""
    
    @staticmethod
    def analyze_sequence_quality(
        sequences: List[str],
        sequence_type: str = "dna"
    ) -> Dict[str, Any]:
        """
        Comprehensive quality analysis of sequence datasets.
        
        Args:
            sequences: List of sequences to analyze
            sequence_type: Type of sequences ('dna', 'rna', 'protein')
            
        Returns:
            Dictionary with quality metrics
        """
        if not sequences:
            return {"error": "No sequences provided"}
        
        lengths = [len(seq) for seq in sequences]
        
        analysis = {
            "total_sequences": len(sequences),
            "length_stats": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": np.mean(lengths),
                "median": np.median(lengths),
                "std": np.std(lengths)
            },
            "composition": {},
            "quality_flags": {
                "too_short": 0,
                "too_long": 0,
                "high_n_content": 0,
                "invalid_characters": 0,
                "duplicates": 0
            }
        }
        
        # Composition analysis
        if sequence_type in ["dna", "rna"]:
            all_chars = ''.join(sequences)
            total_chars = len(all_chars)
            
            analysis["composition"] = {
                "A": all_chars.count('A') / total_chars if total_chars > 0 else 0,
                "T" if sequence_type == "dna" else "U": 
                    all_chars.count('T' if sequence_type == "dna" else 'U') / total_chars if total_chars > 0 else 0,
                "G": all_chars.count('G') / total_chars if total_chars > 0 else 0,
                "C": all_chars.count('C') / total_chars if total_chars > 0 else 0,
                "N": all_chars.count('N') / total_chars if total_chars > 0 else 0,
                "GC_content": (all_chars.count('G') + all_chars.count('C')) / total_chars if total_chars > 0 else 0
            }
        
        # Quality flags
        seen_sequences = set()
        valid_chars = {"dna": "ATCGN", "rna": "AUCGN", "protein": "ACDEFGHIKLMNPQRSTVWYX"}
        valid_set = set(valid_chars.get(sequence_type, ""))
        
        for seq in sequences:
            # Length checks
            if len(seq) < 50:
                analysis["quality_flags"]["too_short"] += 1
            if len(seq) > 10000:
                analysis["quality_flags"]["too_long"] += 1
            
            # N/X content
            if sequence_type in ["dna", "rna"]:
                n_ratio = seq.count('N') / len(seq) if len(seq) > 0 else 0
            else:
                n_ratio = seq.count('X') / len(seq) if len(seq) > 0 else 0
            
            if n_ratio > 0.1:
                analysis["quality_flags"]["high_n_content"] += 1
            
            # Invalid characters
            if not set(seq.upper()).issubset(valid_set):
                analysis["quality_flags"]["invalid_characters"] += 1
            
            # Duplicates
            if seq in seen_sequences:
                analysis["quality_flags"]["duplicates"] += 1
            else:
                seen_sequences.add(seq)
        
        return analysis
    
    @staticmethod
    def filter_sequences_by_quality(
        sequences: List[str],
        min_length: int = 50,
        max_length: int = 10000,
        max_n_ratio: float = 0.1,
        remove_duplicates: bool = True,
        sequence_type: str = "dna"
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Filter sequences based on quality criteria.
        
        Returns:
            Tuple of (filtered_sequences, filter_statistics)
        """
        stats = {
            "original_count": len(sequences),
            "filtered_length": 0,
            "filtered_n_content": 0,
            "filtered_duplicates": 0,
            "final_count": 0
        }
        
        filtered = []
        seen = set() if remove_duplicates else None
        
        valid_chars = {"dna": "ATCGN", "rna": "AUCGN", "protein": "ACDEFGHIKLMNPQRSTVWYX"}
        valid_set = set(valid_chars.get(sequence_type, ""))
        
        for seq in sequences:
            # Length filter
            if len(seq) < min_length or len(seq) > max_length:
                stats["filtered_length"] += 1
                continue
            
            # N/X content filter
            if sequence_type in ["dna", "rna"]:
                n_ratio = seq.count('N') / len(seq) if len(seq) > 0 else 0
            else:
                n_ratio = seq.count('X') / len(seq) if len(seq) > 0 else 0
            
            if n_ratio > max_n_ratio:
                stats["filtered_n_content"] += 1
                continue
            
            # Duplicate filter
            if remove_duplicates:
                if seq in seen:
                    stats["filtered_duplicates"] += 1
                    continue
                seen.add(seq)
            
            filtered.append(seq)
        
        stats["final_count"] = len(filtered)
        return filtered, stats
