"""Main Hyena-GLT model combining BLT's token merging with Hyena operators."""

from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from ..config import HyenaGLTConfig
from .layers import HyenaGLTBlock, AdaptiveTokenMerger
from .heads import SequenceClassificationHead, TokenClassificationHead, SequenceGenerationHead, MultiTaskHead
from .operators import GenomicPositionalEncoding


class HyenaGLTPretrainedModel(PreTrainedModel):
    """Base class for Hyena-GLT models."""
    
    config_class = HyenaGLTConfig
    base_model_prefix = "hyena_glt"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class HyenaGLT(HyenaGLTPretrainedModel):
    """
    Hyena-GLT: Genome Language Transformer
    
    A hybrid model combining BLT's byte latent tokenization with Savanna's Striped Hyena blocks
    for efficient genomic sequence modeling.
    """
    
    def __init__(self, config: HyenaGLTConfig):
        super().__init__(config)
        self.config = config
        
        # Embedding layers
        self.token_embeddings = nn.Embedding(
            config.genomic_vocab_size, 
            config.hidden_size,
            padding_idx=0
        )
        
        # Positional encoding
        self.pos_encoding = GenomicPositionalEncoding(
            d_model=config.hidden_size,
            max_len=config.max_position_embeddings,
        )
        
        # Local encoder (BLT-inspired)
        if config.local_encoder_layers > 0:
            self.local_encoder = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    activation='gelu',
                    batch_first=True,
                )
                for _ in range(config.local_encoder_layers)
            ])
        else:
            self.local_encoder = None
        
        # Initial token merger
        if config.dynamic_patching:
            self.initial_merger = AdaptiveTokenMerger(
                config=config,
                d_model=config.hidden_size,
                min_patch_size=config.min_patch_size,
                max_patch_size=config.max_patch_size,
            )
        else:
            self.initial_merger = None
        
        # Main Hyena-GLT blocks
        self.layers = nn.ModuleList([
            HyenaGLTBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Local decoder (BLT-inspired)
        if config.local_decoder_layers > 0:
            self.local_decoder = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    activation='gelu',
                    batch_first=True,
                )
                for _ in range(config.local_decoder_layers)
            ])
        else:
            self.local_decoder = None
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self):
        """Return input embeddings."""
        return self.token_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        """Set input embeddings."""
        self.token_embeddings = new_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_merge_info: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            return_dict: Whether to return ModelOutput
            output_hidden_states: Whether to return all hidden states
            output_merge_info: Whether to return token merging information
            
        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)  # (batch, seq_len, hidden_size)
        
        # Add positional encoding
        pos_encoding = self.pos_encoding(seq_len)
        hidden_states = hidden_states + pos_encoding.unsqueeze(0)
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Store original sequence for cross-attention
        original_sequence = hidden_states.clone()
        
        # Local encoder processing
        if self.local_encoder is not None:
            for layer in self.local_encoder:
                if self.gradient_checkpointing and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, attention_mask
                    )
                else:
                    hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask == 0)
        
        # Initial token merging
        merge_info_list = []
        segment_boundaries = None
        
        if self.initial_merger is not None:
            merged_states, segment_boundaries, merge_info = self.initial_merger(
                hidden_states, attention_mask
            )
            hidden_states = merged_states
            
            # Update attention mask for merged sequence
            new_seq_len = hidden_states.size(1)
            attention_mask = torch.ones(
                (batch_size, new_seq_len),
                device=input_ids.device,
                dtype=attention_mask.dtype
            )
            
            if output_merge_info:
                merge_info_list.append(merge_info)
        
        # Main Hyena-GLT layers
        all_hidden_states = [] if output_hidden_states else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Determine if we should use cross-attention
            use_cross_attention = (
                layer.cross_attention is not None and 
                original_sequence is not None
            )
            
            if self.gradient_checkpointing and self.training:
                hidden_states, layer_merge_info = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    original_sequence if use_cross_attention else None,
                    attention_mask,
                    segment_boundaries,
                    output_merge_info,
                )
            else:
                hidden_states, layer_merge_info = layer(
                    hidden_states=hidden_states,
                    original_sequence=original_sequence if use_cross_attention else None,
                    attention_mask=attention_mask,
                    segment_boundaries=segment_boundaries,
                    return_merge_info=output_merge_info,
                )
            
            if layer_merge_info is not None and output_merge_info:
                merge_info_list.append(layer_merge_info)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Local decoder processing
        if self.local_decoder is not None:
            # Use original sequence as memory for decoder
            memory = original_sequence
            
            for layer in self.local_decoder:
                if self.gradient_checkpointing and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, memory
                    )
                else:
                    hidden_states = layer(hidden_states, memory)
        
        # Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Prepare outputs
        outputs = {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states if output_hidden_states else None,
            "attention_mask": attention_mask,
        }
        
        if output_merge_info:
            outputs["merge_info"] = merge_info_list
        
        if not return_dict:
            return tuple(v for v in outputs.values() if v is not None)
        
        return outputs


class HyenaGLTForSequenceClassification(HyenaGLTPretrainedModel):
    """Hyena-GLT model for sequence classification tasks."""
    
    def __init__(self, config: HyenaGLTConfig, num_classes: int = 2):
        super().__init__(config)
        self.num_classes = num_classes
        
        # Base model
        self.hyena_glt = HyenaGLT(config)
        
        # Classification head
        self.classifier = SequenceClassificationHead(
            config=config,
            num_classes=num_classes,
            pooling_strategy="cls",
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for sequence classification."""
        
        # Get base model outputs
        outputs = self.hyena_glt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Apply classification head
        logits = self.classifier(
            hidden_states=outputs["last_hidden_state"],
            attention_mask=attention_mask,
        )
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss
        
        return result


class HyenaGLTForTokenClassification(HyenaGLTPretrainedModel):
    """Hyena-GLT model for token classification tasks."""
    
    def __init__(self, config: HyenaGLTConfig, num_classes: int = 2):
        super().__init__(config)
        self.num_classes = num_classes
        
        # Base model
        self.hyena_glt = HyenaGLT(config)
        
        # Token classification head
        self.classifier = TokenClassificationHead(
            config=config,
            num_classes=num_classes,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for token classification."""
        
        # Get base model outputs
        outputs = self.hyena_glt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Apply token classification head
        result = self.classifier(
            hidden_states=outputs["last_hidden_state"],
            labels=labels,
            attention_mask=attention_mask,
        )
        
        return result


class HyenaGLTForSequenceGeneration(HyenaGLTPretrainedModel):
    """Hyena-GLT model for sequence generation tasks."""
    
    def __init__(self, config: HyenaGLTConfig, vocab_size: Optional[int] = None):
        super().__init__(config)
        self.vocab_size = vocab_size or config.genomic_vocab_size
        
        # Base model
        self.hyena_glt = HyenaGLT(config)
        
        # Generation head
        self.generator = SequenceGenerationHead(
            config=config,
            vocab_size=self.vocab_size,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for sequence generation."""
        
        # Get base model outputs
        outputs = self.hyena_glt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Apply generation head
        result = self.generator(
            hidden_states=outputs["last_hidden_state"],
            labels=labels,
            attention_mask=attention_mask,
        )
        
        return result


class HyenaGLTForMultiTask(HyenaGLTPretrainedModel):
    """Hyena-GLT model for multi-task learning."""
    
    def __init__(
        self, 
        config: HyenaGLTConfig, 
        task_configs: Dict[str, Dict[str, Any]]
    ):
        super().__init__(config)
        self.task_configs = task_configs
        
        # Base model
        self.hyena_glt = HyenaGLT(config)
        
        # Multi-task head
        self.multi_task_head = MultiTaskHead(
            config=config,
            task_configs=task_configs,
            task_weights=config.task_weights,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task learning."""
        
        if task is None:
            raise ValueError("Task must be specified for multi-task model")
        
        # Get base model outputs
        outputs = self.hyena_glt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Apply task-specific head
        result = self.multi_task_head(
            hidden_states=outputs["last_hidden_state"],
            task=task,
            labels=labels,
            attention_mask=attention_mask,
        )
        
        return result
