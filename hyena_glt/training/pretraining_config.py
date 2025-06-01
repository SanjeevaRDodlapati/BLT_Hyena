"""
Pretraining configuration system for Hyena-GLT models.

This module provides configuration management for genomic pretraining
based on the OpenGenome dataset structure found in savanna.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from ..config import HyenaGLTConfig


@dataclass
class GenomicDataConfig:
    """Configuration for genomic datasets."""
    
    # Data paths
    train_data_paths: List[str] = field(default_factory=list)
    valid_data_paths: List[str] = field(default_factory=list)
    test_data_paths: List[str] = field(default_factory=list)
    
    # Data weights for mixing
    train_weights: Optional[List[float]] = None
    valid_weights: Optional[List[float]] = None
    test_weights: Optional[List[float]] = None
    
    # Sequence properties
    sequence_type: str = "dna"  # "dna", "rna", "protein"
    max_sequence_length: int = 1024
    enforce_sample_length: bool = True
    
    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # Data preprocessing
    remove_invalid_chars: bool = True
    uppercase_sequences: bool = True
    filter_min_length: int = 10
    
    # OpenGenome dataset paths (examples from savanna config)
    use_opengenome: bool = False
    opengenome_base_path: str = "/nfs/ehr/sdodl001/data/opengenome2"
    
    # Specific dataset configurations
    datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.sequence_type not in ["dna", "rna", "protein"]:
            raise ValueError(f"sequence_type must be one of ['dna', 'rna', 'protein'], got {self.sequence_type}")
        
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        
        if self.filter_min_length <= 0:
            raise ValueError("filter_min_length must be positive")


@dataclass 
class PretrainingStrategyConfig:
    """Configuration for pretraining strategies."""
    
    # Strategy type
    strategy: str = "MLM"  # "AR", "MLM", "OADM", "SPAN"
    
    # MLM parameters
    mask_probability: float = 0.15
    mask_token_ratio: float = 0.8  # 80% mask, 10% random, 10% original
    random_token_ratio: float = 0.1
    
    # Span masking parameters
    span_mask_probability: float = 0.2
    max_span_length: int = 10
    geometric_span_length: bool = True
    span_length_distribution: str = "geometric"  # "geometric", "uniform"
    
    # OADM parameters
    oadm_order_probability: float = 0.5
    oadm_max_positions: Optional[int] = None
    
    # Autoregressive parameters
    ar_direction: str = "left_to_right"  # "left_to_right", "right_to_left", "bidirectional"
    
    def __post_init__(self):
        """Validate strategy configuration."""
        valid_strategies = ["AR", "MLM", "OADM", "SPAN"]
        if self.strategy.upper() not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        
        self.strategy = self.strategy.upper()


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    
    # Loss function type
    loss_function: str = "cross_entropy"  # "cross_entropy", "focal", "label_smoothing"
    
    # Label smoothing
    label_smoothing: float = 0.0
    
    # Focal loss parameters
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # Loss weighting
    class_weights: Optional[List[float]] = None
    
    # Auxiliary losses
    use_auxiliary_losses: bool = False
    auxiliary_loss_weight: float = 0.1
    
    def __post_init__(self):
        """Validate loss configuration."""
        valid_losses = ["cross_entropy", "focal", "label_smoothing"]
        if self.loss_function not in valid_losses:
            raise ValueError(f"loss_function must be one of {valid_losses}")


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    
    # Optimizer
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Adam parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Learning rate schedule
    scheduler_type: str = "cosine"  # "cosine", "linear", "polynomial", "constant"
    warmup_steps: int = 1000
    warmup_ratio: Optional[float] = None  # Alternative to warmup_steps
    
    # Cosine scheduler parameters
    num_cycles: float = 0.5
    final_lr_ratio: float = 0.0
    
    # Polynomial scheduler parameters
    lr_power: float = 1.0
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    gradient_clipping_type: str = "norm"  # "norm", "value"
    
    # Advanced optimization
    use_layer_wise_lr_decay: bool = False
    layer_wise_lr_decay: float = 0.95
    use_bias_correction: bool = True


@dataclass
class HardwareConfig:
    """Configuration for hardware and performance."""
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    fp8: bool = False
    
    # Memory optimization
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    
    # Distributed training
    use_distributed: bool = False
    local_rank: int = -1
    
    # Performance
    compile_model: bool = False
    use_flash_attention: bool = True
    
    # Memory management
    empty_cache_steps: int = 100
    max_memory_usage: Optional[float] = None  # In GB


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    # Basic logging
    log_level: str = "INFO"
    log_steps: int = 100
    
    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps", "epoch"
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5
    save_best_only: bool = False
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "hyena-glt-pretraining"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Tensorboard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "./logs"
    
    # Output
    output_dir: str = "./pretraining_outputs"
    overwrite_output_dir: bool = False
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: str = "eval_loss"
    early_stopping_threshold: float = 0.0


@dataclass
class HyenaGLTPretrainingConfig:
    """Complete configuration for Hyena-GLT pretraining."""
    
    # Model configuration
    model: HyenaGLTConfig = field(default_factory=HyenaGLTConfig)
    
    # Training configuration
    num_epochs: int = 10
    batch_size: int = 32
    
    # Sub-configurations
    data: GenomicDataConfig = field(default_factory=GenomicDataConfig)
    strategy: PretrainingStrategyConfig = field(default_factory=PretrainingStrategyConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Random seed
    seed: int = 42
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    def save_to_yaml(self, file_path: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'HyenaGLTPretrainingConfig':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    # Aliases for compatibility
    def save_to_file(self, file_path: str):
        """Alias for save_to_yaml."""
        return self.save_to_yaml(file_path)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'HyenaGLTPretrainingConfig':
        """Alias for load_from_yaml."""
        return cls.load_from_yaml(file_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'data': self.data.__dict__,
            'strategy': self.strategy.__dict__,
            'loss': self.loss.__dict__,
            'optimization': self.optimization.__dict__,
            'hardware': self.hardware.__dict__,
            'logging': self.logging.__dict__,
            'seed': self.seed,
            'resume_from_checkpoint': self.resume_from_checkpoint
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HyenaGLTPretrainingConfig':
        """Create configuration from dictionary."""
        # Extract model config
        model_config = HyenaGLTConfig(**config_dict.get('model', {}))
        
        # Extract sub-configurations
        data_config = GenomicDataConfig(**config_dict.get('data', {}))
        strategy_config = PretrainingStrategyConfig(**config_dict.get('strategy', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        optimization_config = OptimizationConfig(**config_dict.get('optimization', {}))
        hardware_config = HardwareConfig(**config_dict.get('hardware', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        return cls(
            model=model_config,
            num_epochs=config_dict.get('num_epochs', 10),
            batch_size=config_dict.get('batch_size', 32),
            data=data_config,
            strategy=strategy_config,
            loss=loss_config,
            optimization=optimization_config,
            hardware=hardware_config,
            logging=logging_config,
            seed=config_dict.get('seed', 42),
            resume_from_checkpoint=config_dict.get('resume_from_checkpoint')
        )


class OpenGenomeConfigBuilder:
    """Builder for OpenGenome dataset configurations based on savanna's approach."""
    
    OPENGENOME_DATASETS = {
        "metagenomics": {
            "path": "metagenomics/metagenomics_bpe_4096.jsonl.zst",
            "description": "Metagenomic sequences",
            "weight": 1.0
        },
        "hg38": {
            "path": "hg38/hg38_bpe_4096.jsonl.zst", 
            "description": "Human genome (hg38)",
            "weight": 1.0
        },
        "ncrna": {
            "path": "ncrna/ncrna_bpe_4096.jsonl.zst",
            "description": "Non-coding RNA sequences",
            "weight": 0.5
        },
        "mrna_splice": {
            "path": "mrna_splice/mrna_splice_bpe_4096.jsonl.zst",
            "description": "mRNA splice data",
            "weight": 0.5
        },
        "promoter": {
            "path": "promoter/promoter_bpe_4096.jsonl.zst",
            "description": "Promoter sequences",
            "weight": 0.3
        },
        "eukaryotes": {
            "animalia": {
                "path": "eukaryotes/animalia/animalia_bpe_4096.jsonl.zst",
                "description": "Animal genomic sequences",
                "weight": 1.0
            },
            "fungi": {
                "path": "eukaryotes/fungi/fungi_bpe_4096.jsonl.zst", 
                "description": "Fungal genomic sequences",
                "weight": 0.5
            },
            "plantae": {
                "path": "eukaryotes/plantae/plantae_bpe_4096.jsonl.zst",
                "description": "Plant genomic sequences", 
                "weight": 1.0
            },
            "chromista": {
                "path": "eukaryotes/chromista/chromista_bpe_4096.jsonl.zst",
                "description": "Chromista genomic sequences",
                "weight": 0.3
            },
            "protista": {
                "path": "eukaryotes/protista/protista_bpe_4096.jsonl.zst",
                "description": "Protist genomic sequences",
                "weight": 0.3
            }
        }
    }
    
    @classmethod
    def create_opengenome_config(
        cls,
        base_path: str = "/nfs/ehr/sdodl001/data/opengenome2",
        included_datasets: Optional[List[str]] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> GenomicDataConfig:
        """
        Create genomic data configuration for OpenGenome datasets.
        
        Args:
            base_path: Base path to OpenGenome data
            included_datasets: List of datasets to include (default: all)
            custom_weights: Custom weights for datasets
            
        Returns:
            GenomicDataConfig with OpenGenome paths
        """
        if included_datasets is None:
            included_datasets = list(cls.OPENGENOME_DATASETS.keys())
        
        train_paths = []
        train_weights = []
        
        for dataset_name in included_datasets:
            if dataset_name == "eukaryotes":
                # Handle eukaryote subdirectories
                for kingdom, info in cls.OPENGENOME_DATASETS["eukaryotes"].items():
                    full_path = os.path.join(base_path, info["path"])
                    if os.path.exists(full_path):
                        train_paths.append(full_path)
                        weight = custom_weights.get(f"eukaryotes_{kingdom}", info["weight"]) if custom_weights else info["weight"]
                        train_weights.append(weight)
            else:
                if dataset_name in cls.OPENGENOME_DATASETS:
                    info = cls.OPENGENOME_DATASETS[dataset_name]
                    full_path = os.path.join(base_path, info["path"])
                    if os.path.exists(full_path):
                        train_paths.append(full_path)
                        weight = custom_weights.get(dataset_name, info["weight"]) if custom_weights else info["weight"]
                        train_weights.append(weight)
        
        return GenomicDataConfig(
            train_data_paths=train_paths,
            train_weights=train_weights,
            sequence_type="dna",
            max_sequence_length=4096,  # Match OpenGenome tokenization
            use_opengenome=True,
            opengenome_base_path=base_path
        )


def create_pretraining_configs():
    """Create example pretraining configurations."""
    
    configs = {}
    
    # 1. Small model for testing/development (also add mlm_small alias)
    small_config = HyenaGLTPretrainingConfig(
        model=HyenaGLTConfig(
            vocab_size=8,  # DNA + special tokens
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=512,
        ),
        num_epochs=5,
        batch_size=16,
        data=GenomicDataConfig(
            sequence_type="dna",
            max_sequence_length=512,
            num_workers=2
        ),
        strategy=PretrainingStrategyConfig(
            strategy="MLM",
            mask_probability=0.15
        ),
        optimization=OptimizationConfig(
            learning_rate=5e-4,
            warmup_steps=100
        ),
        logging=LoggingConfig(
            log_steps=10,
            eval_steps=50,
            save_steps=100,
            output_dir="./outputs/small_test"
        )
    )
    
    # Add aliases for compatibility
    configs["small_test"] = small_config
    configs["mlm_small"] = small_config  # Test compatibility
    configs["ar_base"] = small_config  # Test compatibility
    
    # 2. Medium model for standard pretraining
    configs["medium_dna"] = HyenaGLTPretrainingConfig(
        model=HyenaGLTConfig(
            vocab_size=8,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=1024,
        ),
        num_epochs=20,
        batch_size=32,
        data=GenomicDataConfig(
            sequence_type="dna",
            max_sequence_length=1024,
            num_workers=8
        ),
        strategy=PretrainingStrategyConfig(
            strategy="MLM",
            mask_probability=0.15
        ),
        optimization=OptimizationConfig(
            learning_rate=1e-4,
            warmup_steps=1000,
            scheduler_type="cosine"
        ),
        hardware=HardwareConfig(
            fp16=True,
            gradient_accumulation_steps=2
        ),
        logging=LoggingConfig(
            use_wandb=True,
            output_dir="./outputs/medium_dna"
        )
    )
    
    # 3. Large model with OpenGenome data
    configs["large_opengenome"] = HyenaGLTPretrainingConfig(
        model=HyenaGLTConfig(
            vocab_size=8,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            max_position_embeddings=4096,
        ),
        num_epochs=50,
        batch_size=16,
        data=OpenGenomeConfigBuilder.create_opengenome_config(
            included_datasets=["hg38", "metagenomics", "eukaryotes"],
            custom_weights={"hg38": 2.0, "metagenomics": 1.0}
        ),
        strategy=PretrainingStrategyConfig(
            strategy="MLM",
            mask_probability=0.15
        ),
        optimization=OptimizationConfig(
            learning_rate=5e-5,
            warmup_steps=5000,
            scheduler_type="cosine",
            use_layer_wise_lr_decay=True
        ),
        hardware=HardwareConfig(
            bf16=True,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True
        ),
        logging=LoggingConfig(
            use_wandb=True,
            early_stopping=True,
            early_stopping_patience=3,
            output_dir="./outputs/large_opengenome"
        )
    )
    
    # 4. Protein pretraining configuration
    configs["protein_mlm"] = HyenaGLTPretrainingConfig(
        model=HyenaGLTConfig(
            vocab_size=25,  # 20 amino acids + special tokens
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=2048,
        ),
        batch_size=24,
        data=GenomicDataConfig(
            sequence_type="protein",
            max_sequence_length=2048,
            num_workers=8
        ),
        strategy=PretrainingStrategyConfig(
            strategy="MLM",
            mask_probability=0.15
        ),
        optimization=OptimizationConfig(
            learning_rate=1e-4,
            warmup_steps=2000
        ),
        logging=LoggingConfig(
            output_dir="./outputs/protein_mlm"
        )
    )
    
    # 5. Multi-strategy pretraining (OADM)
    configs["dna_oadm"] = HyenaGLTPretrainingConfig(
        model=HyenaGLTConfig(
            vocab_size=8,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=1024,
        ),
        batch_size=32,
        data=GenomicDataConfig(
            sequence_type="dna",
            max_sequence_length=1024
        ),
        strategy=PretrainingStrategyConfig(
            strategy="OADM",
            oadm_order_probability=0.5
        ),
        optimization=OptimizationConfig(
            learning_rate=1e-4,
            warmup_steps=1000
        ),
        logging=LoggingConfig(
            output_dir="./outputs/dna_oadm"
        )
    )
    
    return configs


def main():
    """Example usage of configuration system."""
    
    # Create example configurations
    configs = create_pretraining_configs()
    
    # Save configurations to files
    output_dir = "./example_configs"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, config in configs.items():
        config_path = os.path.join(output_dir, f"{name}.yaml")
        config.save_to_yaml(config_path)
        print(f"Saved {name} configuration to {config_path}")
    
    # Demonstrate loading configuration
    test_config_path = os.path.join(output_dir, "medium_dna.yaml")
    loaded_config = HyenaGLTPretrainingConfig.load_from_yaml(test_config_path)
    print(f"\nLoaded configuration from {test_config_path}")
    print(f"Model hidden size: {loaded_config.model.hidden_size}")
    print(f"Training epochs: {loaded_config.num_epochs}")
    print(f"Pretraining strategy: {loaded_config.strategy.strategy}")
    
    # Create OpenGenome configuration
    opengenome_config = OpenGenomeConfigBuilder.create_opengenome_config(
        included_datasets=["hg38", "metagenomics"],
        custom_weights={"hg38": 2.0, "metagenomics": 1.0}
    )
    print(f"\nOpenGenome configuration:")
    print(f"Number of training files: {len(opengenome_config.train_data_paths)}")
    print(f"Training weights: {opengenome_config.train_weights}")


if __name__ == "__main__":
    main()
