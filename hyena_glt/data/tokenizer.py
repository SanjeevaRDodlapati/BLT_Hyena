"""Genomic tokenizers for DNA, RNA, and protein sequences."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizer

from .utils import generate_kmers, reverse_complement


class GenomicTokenizer(PreTrainedTokenizer):
    """Base class for genomic sequence tokenizers."""
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        sequence_type: str = "dna",
        kmer_size: int = 3,
        overlap_size: int = 1,
        enable_reverse_complement: bool = True,
        max_length: int = 32768,
        padding_side: str = "right",
        **kwargs
    ):
        self.sequence_type = sequence_type
        self.kmer_size = kmer_size
        self.overlap_size = overlap_size
        self.enable_reverse_complement = enable_reverse_complement
        
        # Initialize special tokens
        self.special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]", 
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
        }
        
        # Add sequence-specific special tokens
        if sequence_type == "dna":
            self.special_tokens.update({
                "n_token": "[N]",  # Unknown nucleotide
                "rc_token": "[RC]",  # Reverse complement marker
            })
        elif sequence_type == "protein":
            self.special_tokens.update({
                "x_token": "[X]",  # Unknown amino acid
                "stop_token": "[STOP]",  # Stop codon
            })
        
        # Build vocabulary BEFORE calling parent __init__
        if vocab_file and Path(vocab_file).exists():
            self.vocab = self._load_vocab(vocab_file)
        else:
            self.vocab = self._build_vocab()
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        super().__init__(
            pad_token=self.special_tokens["pad_token"],
            unk_token=self.special_tokens["unk_token"],
            cls_token=self.special_tokens["cls_token"],
            sep_token=self.special_tokens["sep_token"],
            mask_token=self.special_tokens["mask_token"],
            bos_token=self.special_tokens["bos_token"],
            eos_token=self.special_tokens["eos_token"],
            padding_side=padding_side,
            model_max_length=max_length,
            **kwargs
        )
        
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary based on sequence type and k-mer size."""
        vocab = {}
        
        # Add special tokens first
        for token in self.special_tokens.values():
            vocab[token] = len(vocab)
        
        # Generate k-mers based on sequence type
        if self.sequence_type == "dna":
            alphabet = "ATCG"
        elif self.sequence_type == "rna":
            alphabet = "AUCG"
        elif self.sequence_type == "protein":
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
        else:
            raise ValueError(f"Unsupported sequence type: {self.sequence_type}")
        
        # Generate all possible k-mers
        kmers = generate_kmers(alphabet, self.kmer_size)
        for kmer in kmers:
            vocab[kmer] = len(vocab)
        
        # Add single nucleotides/amino acids
        for char in alphabet:
            if char not in vocab:
                vocab[char] = len(vocab)
        
        return vocab
    
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary from file."""
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        return vocab
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save vocabulary to file."""
        vocab_file = Path(save_directory) / f"{filename_prefix or 'vocab'}.json"
        vocab_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f, indent=2)
        
        return (str(vocab_file),)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary."""
        return self.vocab.copy()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize sequence into k-mers and individual characters."""
        # Clean sequence
        sequence = self._preprocess_sequence(text)
        
        tokens = []
        
        # Extract k-mers with overlap
        for i in range(0, len(sequence) - self.kmer_size + 1, self.kmer_size - self.overlap_size):
            kmer = sequence[i:i + self.kmer_size]
            if kmer in self.vocab:
                tokens.append(kmer)
            else:
                # Fall back to individual characters
                for char in kmer:
                    tokens.append(char if char in self.vocab else self.unk_token)
        
        # Handle remaining characters
        remaining_start = len(sequence) - (len(sequence) - self.kmer_size) % (self.kmer_size - self.overlap_size)
        for i in range(remaining_start, len(sequence)):
            char = sequence[i]
            if char not in [token for token in tokens if len(token) == 1]:
                tokens.append(char if char in self.vocab else self.unk_token)
        
        return tokens
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess genomic sequence."""
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Remove whitespace and newlines
        sequence = re.sub(r'\s+', '', sequence)
        
        # Handle sequence-specific preprocessing
        if self.sequence_type == "dna":
            # Replace ambiguous nucleotides
            sequence = re.sub(r'[^ATCG]', 'N', sequence)
        elif self.sequence_type == "rna":
            # Replace T with U, handle ambiguous nucleotides
            sequence = sequence.replace('T', 'U')
            sequence = re.sub(r'[^AUCG]', 'N', sequence)
        elif self.sequence_type == "protein":
            # Handle ambiguous amino acids
            sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', sequence)
        
        return sequence
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.vocab.get(token, self.vocab[self.unk_token])
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def encode_plus(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
        """Enhanced encoding with genomic-specific features."""
        
        if isinstance(text, str):
            sequences = [text]
        else:
            sequences = text
        
        all_input_ids = []
        all_attention_masks = []
        
        for sequence in sequences:
            # Tokenize sequence
            tokens = self._tokenize(sequence)
            
            # Add special tokens
            if add_special_tokens:
                tokens = [self.cls_token] + tokens + [self.sep_token]
            
            # Convert to IDs
            input_ids = [self._convert_token_to_id(token) for token in tokens]
            
            # Handle max length
            if max_length and len(input_ids) > max_length:
                if truncation:
                    input_ids = input_ids[:max_length]
                    if add_special_tokens:
                        input_ids[-1] = self._convert_token_to_id(self.sep_token)
            
            # Create attention mask
            attention_mask = [1] * len(input_ids)
            
            # Padding
            if padding and max_length:
                pad_length = max_length - len(input_ids)
                if pad_length > 0:
                    pad_id = self._convert_token_to_id(self.pad_token)
                    input_ids.extend([pad_id] * pad_length)
                    attention_mask.extend([0] * pad_length)
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
        
        # Prepare output
        output = {
            "input_ids": all_input_ids[0] if len(all_input_ids) == 1 else all_input_ids,
            "attention_mask": all_attention_masks[0] if len(all_attention_masks) == 1 else all_attention_masks,
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            for key, value in output.items():
                if isinstance(value[0], list):
                    output[key] = torch.tensor(value)
                else:
                    output[key] = torch.tensor([value])
        
        return output
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to sequence."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens.values()]
        
        # Reconstruct sequence from k-mers
        if not tokens:
            return ""
        
        sequence = tokens[0]
        for token in tokens[1:]:
            if len(token) == 1:
                sequence += token
            else:
                # Handle k-mer overlap
                overlap = min(len(sequence), self.overlap_size)
                if sequence[-overlap:] == token[:overlap]:
                    sequence += token[overlap:]
                else:
                    sequence += token
        
        return sequence


class DNATokenizer(GenomicTokenizer):
    """Specialized tokenizer for DNA sequences."""
    
    def __init__(self, **kwargs):
        kwargs["sequence_type"] = "dna"
        super().__init__(**kwargs)
    
    def encode_with_reverse_complement(
        self, 
        sequence: str, 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Encode sequence and its reverse complement."""
        # Encode forward sequence
        forward_encoding = self.encode_plus(sequence, **kwargs)
        
        # Encode reverse complement if enabled
        if self.enable_reverse_complement:
            rc_sequence = reverse_complement(sequence)
            rc_encoding = self.encode_plus(rc_sequence, **kwargs)
            
            return {
                "forward": forward_encoding,
                "reverse_complement": rc_encoding,
            }
        
        return {"forward": forward_encoding}


class RNATokenizer(GenomicTokenizer):
    """Specialized tokenizer for RNA sequences."""
    
    def __init__(self, **kwargs):
        kwargs["sequence_type"] = "rna"
        kwargs["enable_reverse_complement"] = kwargs.get("enable_reverse_complement", False)
        super().__init__(**kwargs)


class ProteinTokenizer(GenomicTokenizer):
    """Specialized tokenizer for protein sequences."""
    
    def __init__(self, **kwargs):
        kwargs["sequence_type"] = "protein"
        kwargs["enable_reverse_complement"] = False
        kwargs["kmer_size"] = kwargs.get("kmer_size", 2)  # Smaller k-mers for proteins
        super().__init__(**kwargs)
    
    def encode_with_properties(
        self,
        sequence: str,
        include_properties: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Encode protein sequence with amino acid properties."""
        encoding = self.encode_plus(sequence, **kwargs)
        
        if include_properties:
            # Add amino acid property features (hydrophobicity, charge, etc.)
            # This would be implemented based on specific property encodings
            pass
        
        return encoding
