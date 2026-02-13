"""
Dataset loading and preprocessing for the Transformer Hackathon.

This module provides:
    - TextDataset: PyTorch Dataset for language modeling
    - download_tinystories: Download the TinyStories dataset (default)
    - download_tiny_shakespeare: Download the TinyShakespeare dataset
    - create_dataloaders: Create train/val dataloaders

The default dataset is TinyStories (~500MB), which contains simple short stories
ideal for training small language models.
"""

import os
import urllib.request
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from data.tokenizer import CharTokenizer, SimpleTokenizer


# Dataset URLs
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tinystories(data_dir: str = "data/raw", max_stories: int = 40000) -> str:
    """
    Download the TinyStories dataset from Hugging Face.
    
    TinyStories is a dataset of short stories written in simple English,
    designed for training small language models. It's much larger than
    TinyShakespeare and produces better results.
    
    Dataset size guide for T4 GPU (1 hour training):
        - 10K stories (~25 MB): Fast iteration, perplexity ~20-25
        - 20K stories (~50 MB): Good balance, perplexity ~15-20
        - 40K stories (~100 MB): Best quality, perplexity ~10-18 ✅ RECOMMENDED
        - 50K+ stories: May exceed 1 hour
    
    Args:
        data_dir: Directory to save the file
        max_stories: Maximum number of stories to use (default: 40000)
                    Optimized for 1-hour training on T4 GPU.
                    Use -1 for all stories (~2.1M)
        
    Returns:
        Path to the downloaded file
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "tinystories.txt")
    
    if not os.path.exists(file_path):
        print(f"Downloading TinyStories dataset...")
        print("This may take a few minutes on first run...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets library...")
            import subprocess
            subprocess.check_call(["pip", "install", "datasets", "-q"])
            from datasets import load_dataset
        
        try:
            # Load TinyStories dataset from Hugging Face
            dataset = load_dataset("roneneldan/TinyStories", split="train")
            
            # Limit number of stories if specified
            if max_stories > 0 and len(dataset) > max_stories:
                dataset = dataset.select(range(max_stories))
            
            print(f"Processing {len(dataset):,} stories...")
            
            # Combine stories into single text file
            texts = []
            for item in dataset:
                text = item.get("text", "")
                if text:
                    texts.append(text.strip())
            
            full_text = "\n\n".join(texts)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print(f"Saved {len(full_text):,} characters to {file_path}")
            
        except Exception as e:
            print(f"Error downloading TinyStories: {e}")
            print("Falling back to TinyShakespeare...")
            return download_tiny_shakespeare(data_dir)
    else:
        print(f"Dataset already exists at {file_path}")
    
    return file_path


def download_tiny_shakespeare(data_dir: str = "data/raw") -> str:
    """
    Download the TinyShakespeare dataset (fallback/alternative).
    
    Args:
        data_dir: Directory to save the file
        
    Returns:
        Path to the downloaded file
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "tiny_shakespeare.txt")
    
    if not os.path.exists(file_path):
        print(f"Downloading TinyShakespeare dataset...")
        try:
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, file_path)
            print(f"Downloaded to {file_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download manually from:")
            print(TINY_SHAKESPEARE_URL)
            raise
    else:
        print(f"Dataset already exists at {file_path}")
    
    return file_path


class TextDataset(Dataset):
    """
    PyTorch Dataset for language modeling.
    
    Creates sequences of fixed length from text for training.
    Each sample is (input_ids, target_ids) where target is input shifted by 1.
    
    Args:
        text: The text to create dataset from
        tokenizer: Tokenizer to use for encoding
        seq_len: Sequence length for each sample
        
    Example:
        >>> tokenizer = CharTokenizer("hello world")
        >>> dataset = TextDataset("hello world", tokenizer, seq_len=5)
        >>> x, y = dataset[0]
        >>> print(x.shape, y.shape)  # torch.Size([5]), torch.Size([5])
    """
    
    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        seq_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Encode the entire text
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        # Calculate number of complete sequences
        self.n_samples = max(0, len(self.tokens) - seq_len)
        
        print(f"Dataset created with {len(self.tokens):,} tokens, {self.n_samples:,} samples")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_ids, target_ids) both of shape (seq_len,)
        """
        # Input: tokens from idx to idx + seq_len
        # Target: tokens from idx + 1 to idx + seq_len + 1 (shifted by 1)
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        
        return x, y


def create_dataloaders(
    data_path: Optional[str] = None,
    dataset_name: str = "tinystories",
    seq_len: int = 128,
    batch_size: int = 16,
    train_split: float = 0.9,
    num_workers: int = 0,
    seed: int = 42,
    max_stories: int = 40000
) -> Tuple[DataLoader, DataLoader, SimpleTokenizer]:
    """
    Create train and validation dataloaders.
    
    Dataset size guide for T4 GPU (1 hour training):
        - 10K stories (~25 MB): Fast iteration, perplexity ~20-25
        - 20K stories (~50 MB): Good balance, perplexity ~15-20
        - 40K stories (~100 MB): Best quality, perplexity ~10-18 ✅ DEFAULT
        - 50K+ stories: May exceed 1 hour
    
    Args:
        data_path: Path to text file (downloads dataset if None)
        dataset_name: Dataset to use ("tinystories" or "shakespeare")
        seq_len: Sequence length for samples
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        max_stories: Max stories for TinyStories (default: 40000)
                    Optimized for 1-hour training on T4 GPU
        
    Returns:
        Tuple of (train_loader, val_loader, tokenizer)
        
    Example:
        >>> train_loader, val_loader, tokenizer = create_dataloaders(batch_size=32)
        >>> for batch_idx, (x, y) in enumerate(train_loader):
        ...     print(x.shape, y.shape)  # (32, 128), (32, 128)
        ...     break
    """
    # Download dataset if needed
    if data_path is None:
        if dataset_name.lower() == "tinystories":
            data_path = download_tinystories(max_stories=max_stories)
        else:
            data_path = download_tiny_shakespeare()
    
    # Load text
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded {len(text):,} characters")
    
    # Create tokenizer (switch to word-level for GloVe compatibility)
    tokenizer = SimpleTokenizer(min_freq=1, max_vocab_size=None)
    tokenizer.build_vocab(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    dataset = TextDataset(text, tokenizer, seq_len)
    
    # Split into train/val
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator
    )
    
    print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

    use_pin_memory = torch.cuda.is_available()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True  # Drop incomplete batches for consistent batch size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, tokenizer


def load_wikitext(data_dir: str = "data/raw", version: str = "wikitext-2") -> str:
    """
    Load WikiText dataset (alternative dataset option).
    
    Note: This requires the 'datasets' library from Hugging Face.
    
    Args:
        data_dir: Directory to cache data
        version: WikiText version ("wikitext-2" or "wikitext-103")
        
    Returns:
        Path to the downloaded file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the 'datasets' library: pip install datasets")
        raise
    
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{version}.txt")
    
    if not os.path.exists(file_path):
        print(f"Downloading {version} dataset...")
        dataset = load_dataset("wikitext", f"{version}-raw-v1", split="train")
        
        # Combine all text
        text = "\n".join(dataset["text"])
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Saved to {file_path}")
    
    return file_path


if __name__ == "__main__":
    # Quick test
    print("Testing dataset loading with TinyStories...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        batch_size=4, seq_len=32, max_stories=1000  # Small for testing
    )
    
    # Print a sample
    x, y = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nSample text: {tokenizer.decode(x[0].tolist())[:100]}...")
