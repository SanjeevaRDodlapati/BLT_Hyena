"""Utility functions for genomic data processing."""

from itertools import product
from typing import Any


def reverse_complement(sequence: str) -> str:
    """Generate reverse complement of a DNA sequence."""
    complement_map = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "a": "t",
        "t": "a",
        "c": "g",
        "g": "c",
        "N": "N",
        "n": "n",
    }

    complement = "".join(complement_map.get(base, base) for base in sequence)
    return complement[::-1]


def translate_dna(sequence: str, reading_frame: int = 0) -> str:
    """Translate DNA sequence to protein sequence."""
    genetic_code = {
        "TTT": "F",
        "TTC": "F",
        "TTA": "L",
        "TTG": "L",
        "TCT": "S",
        "TCC": "S",
        "TCA": "S",
        "TCG": "S",
        "TAT": "Y",
        "TAC": "Y",
        "TAA": "*",
        "TAG": "*",
        "TGT": "C",
        "TGC": "C",
        "TGA": "*",
        "TGG": "W",
        "CTT": "L",
        "CTC": "L",
        "CTA": "L",
        "CTG": "L",
        "CCT": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",
        "CAT": "H",
        "CAC": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGT": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "ATT": "I",
        "ATC": "I",
        "ATA": "I",
        "ATG": "M",
        "ACT": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",
        "AAT": "N",
        "AAC": "N",
        "AAA": "K",
        "AAG": "K",
        "AGT": "S",
        "AGC": "S",
        "AGA": "R",
        "AGG": "R",
        "GTT": "V",
        "GTC": "V",
        "GTA": "V",
        "GTG": "V",
        "GCT": "A",
        "GCC": "A",
        "GCA": "A",
        "GCG": "A",
        "GAT": "D",
        "GAC": "D",
        "GAA": "E",
        "GAG": "E",
        "GGT": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",
    }

    # Start from specified reading frame
    sequence = sequence[reading_frame:].upper()

    protein = []
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i : i + 3]
        if len(codon) == 3:
            amino_acid = genetic_code.get(codon, "X")
            if amino_acid == "*":  # Stop codon
                break
            protein.append(amino_acid)

    return "".join(protein)


def generate_kmers(alphabet: str, k: int) -> list[str]:
    """Generate all possible k-mers from given alphabet."""
    return ["".join(kmer) for kmer in product(alphabet, repeat=k)]


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of a DNA/RNA sequence."""
    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    total_count = len([base for base in sequence if base in "ATCGU"])

    if total_count == 0:
        return 0.0

    return gc_count / total_count


def find_orfs(sequence: str, min_length: int = 100) -> list[dict[str, Any]]:
    """Find open reading frames in a DNA sequence."""
    sequence = sequence.upper()
    orfs = []

    # Check all six reading frames (3 forward, 3 reverse)
    for strand in [1, -1]:
        seq = sequence if strand == 1 else reverse_complement(sequence)

        for frame in range(3):
            start_pos = frame

            while start_pos < len(seq) - 2:
                # Look for start codon
                if seq[start_pos : start_pos + 3] == "ATG":
                    # Look for stop codon
                    for end_pos in range(start_pos + 3, len(seq) - 2, 3):
                        codon = seq[end_pos : end_pos + 3]
                        if codon in ["TAA", "TAG", "TGA"]:
                            orf_length = end_pos - start_pos + 3
                            if orf_length >= min_length:
                                orfs.append(
                                    {
                                        "start": start_pos,
                                        "end": end_pos + 3,
                                        "length": orf_length,
                                        "strand": strand,
                                        "frame": frame,
                                        "sequence": seq[start_pos : end_pos + 3],
                                        "protein": translate_dna(
                                            seq[start_pos : end_pos + 3]
                                        ),
                                    }
                                )
                            break
                    start_pos = start_pos + 3
                else:
                    start_pos += 1

    return orfs


def sliding_window(sequence: str, window_size: int, step_size: int = 1) -> list[str]:
    """Generate sliding windows over a sequence."""
    windows = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        windows.append(sequence[i : i + window_size])
    return windows


def validate_sequence(
    sequence: str, sequence_type: str = "dna"
) -> tuple[bool, list[str]]:
    """Validate genomic sequence and return errors if any."""
    errors = []
    sequence = sequence.upper().strip()

    if not sequence:
        errors.append("Empty sequence")
        return False, errors

    if sequence_type == "dna":
        valid_chars = set("ATCGN")
        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            errors.append(f"Invalid DNA characters: {', '.join(invalid_chars)}")

    elif sequence_type == "rna":
        valid_chars = set("AUCGN")
        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            errors.append(f"Invalid RNA characters: {', '.join(invalid_chars)}")

    elif sequence_type == "protein":
        valid_chars = set("ACDEFGHIKLMNPQRSTVWYX")
        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            errors.append(f"Invalid protein characters: {', '.join(invalid_chars)}")

    return len(errors) == 0, errors


def sequence_stats(sequence: str, sequence_type: str = "dna") -> dict[str, Any]:
    """Calculate basic statistics for a genomic sequence."""
    sequence = sequence.upper().strip()
    stats: dict[str, Any] = {
        "length": len(sequence),
        "composition": {},
    }

    if sequence_type in ["dna", "rna"]:
        for base in "ATCGUN" if sequence_type == "rna" else "ATCGN":
            count = sequence.count(base)
            stats["composition"][base] = {
                "count": count,
                "frequency": count / len(sequence) if len(sequence) > 0 else 0,
            }

        if sequence_type == "dna":
            stats["gc_content"] = calculate_gc_content(sequence)
            stats["orfs"] = len(find_orfs(sequence))

    elif sequence_type == "protein":
        amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
        for aa in amino_acids:
            count = sequence.count(aa)
            stats["composition"][aa] = {
                "count": count,
                "frequency": count / len(sequence) if len(sequence) > 0 else 0,
            }

    return stats


def mask_sequence(
    sequence: str, mask_probability: float = 0.15, mask_token: str = "[MASK]"
) -> tuple[str, list[int]]:
    """Randomly mask positions in a sequence for MLM training."""
    import random

    sequence_list = list(sequence)
    masked_positions = []

    for i, _char in enumerate(sequence_list):
        if random.random() < mask_probability:
            sequence_list[i] = mask_token
            masked_positions.append(i)

    return "".join(sequence_list), masked_positions
