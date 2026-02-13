"""
Configuration file for the Transformer Hackathon.

This file contains all hyperparameters and settings for the transformer model,
training pipeline, and evaluation. Participants can modify these values to
experiment with different configurations.

TODO: Experiment with these hyperparameters to improve your model!
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """
    Model architecture configuration.
    
    Attributes:
        vocab_size: Size of the vocabulary (set by tokenizer)
        d_model: Embedding dimension (default: 512)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 6)
        d_ff: Feed-forward network hidden dimension (default: 2048)
        max_seq_len: Maximum sequence length (default: 128)
        dropout: Dropout probability (default: 0.1)
        
    TODO: Try increasing n_layers or d_model for better performance!
    """
    vocab_size: int = 5000
    d_model: int = 100
    n_heads: int = 5
    n_layers: int = 2
    d_ff: int = 2048
    max_seq_len: int = 128
    dropout: float = 0.1
    
    # Activation function for FFN
    activation: str = "gelu"  # Options: "gelu", "relu"
    
    # Layer normalization epsilon
    layer_norm_eps: float = 1e-6
    
    # Whether to use pre-layer normalization (more stable training)
    pre_norm: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


@dataclass
class TrainingConfig:
    """
    Training configuration.
    
    Attributes:
        max_time_minutes: Maximum training time in minutes (default: 45)
        batch_size: Training batch size (default: 16)
        learning_rate: Initial learning rate (default: 3e-4)
        weight_decay: Weight decay for AdamW (default: 0.1)
        max_grad_norm: Gradient clipping threshold (default: 1.0)
        
    TODO: Experiment with batch size and learning rate!
    """
    # Time constraint
    max_time_minutes: float = 60.0  # Updated from 45 for 1-hour training
    
    # Batch settings
    batch_size: int = 16
    eval_batch_size: int = 32
    
    # TODO: Try gradient accumulation for effectively larger batch sizes!
    gradient_accumulation_steps: int = 1
    
    # Optimizer settings
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    
    # Learning rate schedule
    # TODO: Experiment with different schedules!
    lr_scheduler: str = "cosine"  # Options: "cosine", "linear", "constant"
    warmup_ratio: float = 0.1  # Fraction of training for warmup
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of initial LR
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # TODO: Enable mixed precision for faster training on modern GPUs!
    use_mixed_precision: bool = False
    
    # Checkpointing
    checkpoint_interval_minutes: float = 5.0
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_interval_steps: int = 10
    eval_interval_steps: int = 500
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Number of workers for data loading
    num_workers: int = 0  # Set to 0 for Colab compatibility


@dataclass
class DataConfig:
    """
    Data configuration.
    
    Attributes:
        dataset_name: Name of the dataset to use
        data_dir: Directory to store downloaded data
        train_split: Fraction of data for training
        max_stories: Number of TinyStories to use (40K for 1-hour T4 training)
        
    TODO: Try different datasets or data augmentation!
    """
    dataset_name: str = "tinystories"  # Changed from "tiny_shakespeare"
    max_stories: int = 40000  # Optimized for 1-hour T4 GPU training
    data_dir: str = "data/raw"
    
    # Data splits
    train_split: float = 0.9
    val_split: float = 0.1
    
    # Tokenizer settings
    tokenizer_type: str = "char"  # Options: "char", "bpe"
    min_freq: int = 1  # Minimum frequency for vocabulary
    
    # Preprocessing
    lowercase: bool = False
    
    # Cache
    cache_dir: str = "data/cache"


@dataclass
class GenerationConfig:
    """
    Text generation configuration.
    
    Attributes:
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        
    TODO: Experiment with different sampling strategies!
    """
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Seed for reproducible generation
    seed: Optional[int] = None


@dataclass
class HuggingFaceConfig:
    """
    Hugging Face integration configuration.
    
    Attributes:
        leaderboard_repo: HF dataset repo for the leaderboard
        token_env_var: Environment variable for HF token
    """
    leaderboard_repo: str = "transformer-hackathon/leaderboard"
    token_env_var: str = "HF_TOKEN"
    
    # Upload settings
    upload_model: bool = False  # Whether to upload model checkpoint
    model_repo_prefix: str = "transformer-hackathon/model-"


@dataclass
class Config:
    """
    Master configuration combining all sub-configs.
    
    Usage:
        config = Config()
        config.model.d_model = 768  # Modify model size
        config.training.batch_size = 32  # Increase batch size
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    @property
    def device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["=" * 50, "Configuration", "=" * 50]
        for section_name in ["model", "training", "data", "generation"]:
            section = getattr(self, section_name)
            lines.append(f"\n[{section_name.upper()}]")
            for key, value in vars(section).items():
                if not key.startswith("_"):
                    lines.append(f"  {key}: {value}")
        lines.append(f"\n[DEVICE]")
        lines.append(f"  device: {self.device}")
        lines.append("=" * 50)
        return "\n".join(lines)


# Default configuration instance
default_config = Config()


# =============================================================================
# OPTIMIZATION IDEAS FOR PARTICIPANTS
# =============================================================================
"""
Here are some ideas to improve your model's performance:

1. MIXED PRECISION TRAINING (Easy Win!)
   - Set config.training.use_mixed_precision = True
   - Can give 2x speedup on modern GPUs
   
2. GRADIENT ACCUMULATION
   - Increase config.training.gradient_accumulation_steps
   - Effectively larger batch size without more memory
   
3. LEARNING RATE TUNING
   - Try different config.training.learning_rate values
   - Common range: 1e-4 to 1e-3
   
4. MODEL SIZE
   - Increase config.model.d_model (256, 512, 768, 1024)
   - More layers: config.model.n_layers (4, 6, 8, 12)
   - Trade-off: larger = better quality, slower training
   
5. SEQUENCE LENGTH
   - Increase config.model.max_seq_len for more context
   - Watch out for memory usage (quadratic in attention)
   
6. BETTER ATTENTION (Advanced)
   - Implement Flash Attention in model/attention.py
   - Use torch.nn.functional.scaled_dot_product_attention
   
7. DATA AUGMENTATION
   - Random masking
   - Synonym replacement
   - Back-translation

8. CUSTOM OPTIMIZER
   - Try Lion optimizer
   - Implement AdaFactor for lower memory
   
Good luck! 🚀
"""
