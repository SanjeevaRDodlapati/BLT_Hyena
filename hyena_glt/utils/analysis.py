"""
Analysis utilities for Hyena-GLT framework.

This module provides analysis functions for tokenization, sequences, and model outputs.
"""

from typing import Dict, List, Union
import statistics


def analyze_tokenization(tokenizer, sequences: List[str]) -> Dict[str, Union[float, int]]:
    """
    Analyze tokenization statistics for a list of sequences.
    
    Args:
        tokenizer: The tokenizer instance (GenomicTokenizer, DNATokenizer, etc.)
        sequences: List of sequences to analyze
        
    Returns:
        Dictionary containing tokenization statistics:
        - avg_tokens: Average number of tokens per sequence
        - total_tokens: Total number of tokens
        - avg_sequence_length: Average character length of sequences
        - compression_ratio: Ratio of characters to tokens (compression factor)
        - min_tokens: Minimum tokens in any sequence
        - max_tokens: Maximum tokens in any sequence
    """
    if not sequences:
        return {
            'avg_tokens': 0.0,
            'total_tokens': 0,
            'avg_sequence_length': 0.0,
            'compression_ratio': 0.0,
            'min_tokens': 0,
            'max_tokens': 0
        }
    
    token_counts = []
    char_counts = []
    
    for sequence in sequences:
        # Tokenize the sequence
        tokens = tokenizer.encode(sequence)
        token_counts.append(len(tokens))
        char_counts.append(len(sequence))
    
    total_tokens = sum(token_counts)
    total_chars = sum(char_counts)
    
    stats = {
        'avg_tokens': statistics.mean(token_counts),
        'total_tokens': total_tokens,
        'avg_sequence_length': statistics.mean(char_counts),
        'compression_ratio': total_chars / total_tokens if total_tokens > 0 else 0.0,
        'min_tokens': min(token_counts) if token_counts else 0,
        'max_tokens': max(token_counts) if token_counts else 0
    }
    
    return stats


def analyze_sequence_composition(sequences: List[str]) -> Dict[str, Union[float, Dict[str, int]]]:
    """
    Analyze the composition of genomic sequences.
    
    Args:
        sequences: List of genomic sequences
        
    Returns:
        Dictionary containing composition statistics
    """
    if not sequences:
        return {
            'total_length': 0,
            'avg_length': 0.0,
            'nucleotide_counts': {},
            'gc_content': 0.0
        }
    
    total_length = sum(len(seq) for seq in sequences)
    nucleotide_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
    
    for sequence in sequences:
        for nucleotide in sequence.upper():
            if nucleotide in nucleotide_counts:
                nucleotide_counts[nucleotide] += 1
            else:
                nucleotide_counts['N'] += 1  # Unknown nucleotides
    
    gc_count = nucleotide_counts['G'] + nucleotide_counts['C']
    at_count = nucleotide_counts['A'] + nucleotide_counts['T']
    gc_content = gc_count / (gc_count + at_count) if (gc_count + at_count) > 0 else 0.0
    
    return {
        'total_length': total_length,
        'avg_length': total_length / len(sequences),
        'nucleotide_counts': nucleotide_counts,
        'gc_content': gc_content
    }
