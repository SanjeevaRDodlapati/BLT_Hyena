"""
Data utilities for generating and processing genomic datasets.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import random
from collections import Counter
import warnings

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    warnings.warn("BioPython not available. Some functionality will be limited.")


def generate_synthetic_genomic_data(
    n_samples: int = 1000,
    seq_length: int = 256,
    sequence_type: str = "dna",
    task_type: str = "classification",
    num_classes: int = 5,
    seed: int = 42,
    add_noise: bool = True,
    include_motifs: bool = True
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """
    Generate synthetic genomic data for training and testing.
    
    Args:
        n_samples: Number of sequences to generate
        seq_length: Length of each sequence
        sequence_type: Type of sequence ('dna', 'rna', 'protein')
        task_type: Type of task ('classification', 'regression')
        num_classes: Number of classes for classification
        seed: Random seed for reproducibility
        add_noise: Whether to add random noise to sequences
        include_motifs: Whether to include known genomic motifs
        
    Returns:
        Tuple of (sequences, labels, metadata)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Define alphabets
    alphabets = {
        'dna': ['A', 'T', 'C', 'G'],
        'rna': ['A', 'U', 'C', 'G'],
        'protein': list("ACDEFGHIKLMNPQRSTVWY")
    }
    
    if sequence_type not in alphabets:
        raise ValueError(f"Unsupported sequence type: {sequence_type}")
    
    alphabet = alphabets[sequence_type]
    sequences = []
    labels = []
    
    # Known motifs for different sequence types
    motifs = {
        'dna': {
            'tata_box': 'TATAAA',
            'kozak': 'GCCRCCATGG',
            'polya': 'AATAAA',
            'cpg': 'CG' * 3,
            'gc_rich': 'GCGCGC'
        },
        'rna': {
            'kozak': 'GCCGCCAUGG',
            'polya': 'AAUAAA',
            'hairpin': 'GGGGAAACCCCC',
            'ribosome': 'AGGAGG'
        },
        'protein': {
            'signal_peptide': 'MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRF',
            'nuclear_localization': 'PKKKRKV',
            'transmembrane': 'LVIWGAAFVGFIMIY'
        }
    }
    
    for i in range(n_samples):
        # Generate base sequence
        seq = ''.join(np.random.choice(alphabet, seq_length))
        
        if sequence_type == 'dna':
            label = _generate_dna_label(seq, num_classes, include_motifs, motifs['dna'])
        elif sequence_type == 'rna':
            label = _generate_rna_label(seq, num_classes, include_motifs, motifs['rna'])
        else:  # protein
            label = _generate_protein_label(seq, num_classes, include_motifs, motifs['protein'])
        
        # Add noise if requested
        if add_noise and np.random.random() < 0.1:
            seq = _add_sequence_noise(seq, alphabet, noise_rate=0.05)
        
        sequences.append(seq)
        labels.append(label)
    
    # Generate metadata
    metadata = {
        'sequence_type': sequence_type,
        'task_type': task_type,
        'num_samples': n_samples,
        'sequence_length': seq_length,
        'num_classes': num_classes,
        'class_distribution': dict(Counter(labels)),
        'alphabet_size': len(alphabet),
        'has_motifs': include_motifs,
        'noise_added': add_noise,
        'seed': seed
    }
    
    return sequences, labels, metadata


def _generate_dna_label(seq: str, num_classes: int, include_motifs: bool, motifs: Dict[str, str]) -> int:
    """Generate label for DNA sequence based on content and motifs."""
    gc_content = (seq.count('G') + seq.count('C')) / len(seq)
    
    # Check for specific motifs first
    if include_motifs:
        for motif_name, motif_seq in motifs.items():
            if motif_seq in seq:
                if motif_name == 'tata_box':
                    return min(4, num_classes - 1)  # Promoter class
                elif motif_name == 'cpg':
                    return min(3, num_classes - 1)  # CpG island
    
    # Base classification on GC content
    if gc_content < 0.3:
        return 0  # AT-rich
    elif gc_content < 0.5:
        return 1  # Moderate GC
    elif gc_content < 0.7:
        return 2  # GC-rich
    else:
        return min(3, num_classes - 1)  # Very GC-rich


def _generate_rna_label(seq: str, num_classes: int, include_motifs: bool, motifs: Dict[str, str]) -> int:
    """Generate label for RNA sequence based on structure and motifs."""
    au_content = (seq.count('A') + seq.count('U')) / len(seq)
    
    if include_motifs:
        for motif_name, motif_seq in motifs.items():
            if motif_seq in seq:
                if motif_name == 'ribosome':
                    return min(4, num_classes - 1)  # Ribosomal RNA
                elif motif_name == 'polya':
                    return min(3, num_classes - 1)  # mRNA
    
    # Classification based on AU content (inverse of GC)
    if au_content > 0.7:
        return 0  # AU-rich (regulatory RNA)
    elif au_content > 0.5:
        return 1  # Moderate AU
    else:
        return 2  # GC-rich (structural RNA)


def _generate_protein_label(seq: str, num_classes: int, include_motifs: bool, motifs: Dict[str, str]) -> int:
    """Generate label for protein sequence based on composition and motifs."""
    hydrophobic_aa = set('AILMFPWV')
    hydrophobic_content = sum(1 for aa in seq if aa in hydrophobic_aa) / len(seq)
    
    if include_motifs:
        for motif_name, motif_seq in motifs.items():
            if motif_seq in seq:
                if motif_name == 'signal_peptide':
                    return min(4, num_classes - 1)  # Secreted protein
                elif motif_name == 'transmembrane':
                    return min(3, num_classes - 1)  # Membrane protein
    
    # Classification based on hydrophobicity
    if hydrophobic_content > 0.6:
        return 0  # Membrane protein
    elif hydrophobic_content > 0.4:
        return 1  # Globular protein
    else:
        return 2  # Hydrophilic protein


def _add_sequence_noise(seq: str, alphabet: List[str], noise_rate: float = 0.05) -> str:
    """Add random noise to sequence."""
    seq_list = list(seq)
    num_changes = int(len(seq) * noise_rate)
    
    for _ in range(num_changes):
        pos = np.random.randint(0, len(seq_list))
        seq_list[pos] = np.random.choice(alphabet)
    
    return ''.join(seq_list)


def load_genomic_dataset(
    file_path: Union[str, Path],
    file_format: str = "auto",
    max_sequences: Optional[int] = None,
    min_length: int = 50,
    max_length: int = 10000
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Load genomic sequences from various file formats.
    
    Args:
        file_path: Path to the genomic file
        file_format: Format of the file ('fasta', 'fastq', 'auto')
        max_sequences: Maximum number of sequences to load
        min_length: Minimum sequence length to include
        max_length: Maximum sequence length to include
        
    Returns:
        Tuple of (sequences, descriptions, metadata)
    """
    if not HAS_BIOPYTHON:
        raise ImportError("BioPython is required for loading genomic files")
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect format
    if file_format == "auto":
        suffix = file_path.suffix.lower()
        if suffix in ['.fa', '.fasta']:
            file_format = 'fasta'
        elif suffix in ['.fq', '.fastq']:
            file_format = 'fastq'
        else:
            raise ValueError(f"Cannot auto-detect format for file: {file_path}")
    
    sequences = []
    descriptions = []
    count = 0
    
    try:
        for record in SeqIO.parse(file_path, file_format):
            seq_str = str(record.seq).upper()
            
            # Filter by length
            if len(seq_str) < min_length or len(seq_str) > max_length:
                continue
            
            sequences.append(seq_str)
            descriptions.append(record.description)
            count += 1
            
            if max_sequences and count >= max_sequences:
                break
                
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {e}")
    
    # Generate metadata
    metadata = {
        'file_path': str(file_path),
        'file_format': file_format,
        'num_sequences': len(sequences),
        'sequence_lengths': {
            'min': min(len(seq) for seq in sequences) if sequences else 0,
            'max': max(len(seq) for seq in sequences) if sequences else 0,
            'mean': np.mean([len(seq) for seq in sequences]) if sequences else 0
        },
        'sequence_composition': _analyze_sequence_composition(sequences)
    }
    
    return sequences, descriptions, metadata


def create_balanced_dataset(
    sequences: List[str],
    labels: List[int],
    target_samples_per_class: Optional[int] = None,
    method: str = "oversample",
    seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Create a balanced dataset from imbalanced data.
    
    Args:
        sequences: List of sequences
        labels: List of corresponding labels
        target_samples_per_class: Target number of samples per class
        method: Balancing method ('oversample', 'undersample', 'mixed')
        seed: Random seed
        
    Returns:
        Tuple of (balanced_sequences, balanced_labels)
    """
    np.random.seed(seed)
    
    # Group sequences by class
    class_sequences = {}
    for seq, label in zip(sequences, labels):
        if label not in class_sequences:
            class_sequences[label] = []
        class_sequences[label].append(seq)
    
    # Determine target size
    class_sizes = [len(seqs) for seqs in class_sequences.values()]
    if target_samples_per_class is None:
        if method == "oversample":
            target_samples_per_class = max(class_sizes)
        elif method == "undersample":
            target_samples_per_class = min(class_sizes)
        else:  # mixed
            target_samples_per_class = int(np.median(class_sizes))
    
    balanced_sequences = []
    balanced_labels = []
    
    for label, seqs in class_sequences.items():
        current_size = len(seqs)
        
        if current_size >= target_samples_per_class:
            # Undersample
            selected_seqs = np.random.choice(seqs, target_samples_per_class, replace=False)
        else:
            # Oversample
            selected_seqs = np.random.choice(seqs, target_samples_per_class, replace=True)
        
        balanced_sequences.extend(selected_seqs)
        balanced_labels.extend([label] * target_samples_per_class)
    
    # Shuffle the balanced dataset
    combined = list(zip(balanced_sequences, balanced_labels))
    np.random.shuffle(combined)
    balanced_sequences, balanced_labels = zip(*combined)
    
    return list(balanced_sequences), list(balanced_labels)


def validate_genomic_sequences(
    sequences: List[str],
    sequence_type: str = "dna",
    strict: bool = True
) -> Tuple[List[bool], Dict[str, Any]]:
    """
    Validate genomic sequences for correctness.
    
    Args:
        sequences: List of sequences to validate
        sequence_type: Type of sequence ('dna', 'rna', 'protein')
        strict: Whether to use strict validation
        
    Returns:
        Tuple of (validation_results, validation_summary)
    """
    valid_chars = {
        'dna': set('ATCGN'),
        'rna': set('AUCGN'),
        'protein': set('ACDEFGHIKLMNPQRSTVWY*X')
    }
    
    if sequence_type not in valid_chars:
        raise ValueError(f"Unsupported sequence type: {sequence_type}")
    
    allowed_chars = valid_chars[sequence_type]
    results = []
    issues = {
        'invalid_characters': 0,
        'empty_sequences': 0,
        'too_short': 0,
        'high_n_content': 0
    }
    
    for seq in sequences:
        is_valid = True
        
        # Check if empty
        if not seq or len(seq.strip()) == 0:
            is_valid = False
            issues['empty_sequences'] += 1
        
        # Check characters
        elif not set(seq.upper()).issubset(allowed_chars):
            if strict:
                is_valid = False
            issues['invalid_characters'] += 1
        
        # Check length
        elif len(seq) < 10:  # Minimum reasonable length
            if strict:
                is_valid = False
            issues['too_short'] += 1
        
        # Check N content for DNA/RNA
        elif sequence_type in ['dna', 'rna'] and seq.upper().count('N') / len(seq) > 0.1:
            if strict:
                is_valid = False
            issues['high_n_content'] += 1
        
        results.append(is_valid)
    
    summary = {
        'total_sequences': len(sequences),
        'valid_sequences': sum(results),
        'invalid_sequences': len(results) - sum(results),
        'validation_rate': sum(results) / len(results) if results else 0,
        'issues_found': issues
    }
    
    return results, summary


def _analyze_sequence_composition(sequences: List[str]) -> Dict[str, Any]:
    """Analyze the composition of a list of sequences."""
    if not sequences:
        return {}
    
    # Combine all sequences
    all_seq = ''.join(sequences)
    char_counts = Counter(all_seq.upper())
    total_chars = len(all_seq)
    
    composition = {char: count / total_chars for char, count in char_counts.items()}
    
    # Additional statistics
    gc_contents = []
    for seq in sequences:
        if seq:
            gc = (seq.upper().count('G') + seq.upper().count('C')) / len(seq)
            gc_contents.append(gc)
    
    result = {
        'character_frequencies': composition,
        'most_common_chars': char_counts.most_common(5),
        'gc_content_stats': {
            'mean': np.mean(gc_contents) if gc_contents else 0,
            'std': np.std(gc_contents) if gc_contents else 0,
            'min': min(gc_contents) if gc_contents else 0,
            'max': max(gc_contents) if gc_contents else 0
        } if any(c in 'GC' for c in char_counts) else None
    }
    
    return result
