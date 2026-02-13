#!/usr/bin/env python3
"""
Evaluation script for the Transformer Hackathon.

This script evaluates a trained GPT model with:
    - Perplexity on validation set
    - Loss calculation
    - Generation quality metrics (distinct-n)
    - Sample text generation
    - Training efficiency summary

Usage:
    python evaluate.py                           # Evaluate latest checkpoint
    python evaluate.py --checkpoint best.pt      # Evaluate specific checkpoint
    python evaluate.py --generate                # Include text generation samples
"""

import argparse
import os
import sys
import json
from typing import Optional, Dict, List

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import GPTModel
from data import create_dataloaders, CharTokenizer, SimpleTokenizer
from utils.metrics import (
    compute_perplexity, 
    compute_accuracy,
    compute_distinct_n,
    compute_repetition_rate
)
from utils.checkpoint import load_checkpoint, get_latest_checkpoint, get_checkpoint_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint (uses latest if not specified)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory containing checkpoints")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to evaluation data")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Evaluation batch size")
    parser.add_argument("--generate", action="store_true",
                       help="Generate sample text")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of text samples to generate")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate per sample")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (JSON)")
    
    return parser.parse_args()


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: torch.device
) -> tuple:
    """
    Load model and tokenizer from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer, checkpoint_info)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load tokenizer (auto-detect char vs word)
    tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.json")
    if os.path.exists(tokenizer_path):
        try:
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tok_meta = json.load(f)
            tok_type = tok_meta.get('type', 'char')
            if tok_type == 'word':
                tokenizer = SimpleTokenizer.load(tokenizer_path)
            else:
                tokenizer = CharTokenizer.load(tokenizer_path)
        except Exception:
            # Fallback to char
            tokenizer = CharTokenizer.load(tokenizer_path)
    else:
        # Try to get from checkpoint
        tok_data = checkpoint.get('tokenizer', {})
        if tok_data and tok_data.get('char_to_id'):
            tokenizer = CharTokenizer()
            tokenizer.char_to_id = tok_data['char_to_id']
            tokenizer.id_to_char = {v: k for k, v in tok_data['char_to_id'].items()}
        else:
            raise ValueError("Tokenizer not found in checkpoint or checkpoint directory")
    
    # Create model with same configuration
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.ModelConfig.d_model,
        n_heads=config.ModelConfig.n_heads,
        n_layers=config.ModelConfig.n_layers,
        d_ff=config.ModelConfig.d_model * 4,
        max_seq_len=config.ModelConfig.max_seq_len,
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Checkpoint info
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'step': checkpoint.get('step', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
        'metrics': checkpoint.get('metrics', {}),
    }
    
    return model, tokenizer, info


def evaluate_perplexity(
    model: GPTModel,
    dataloader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model perplexity on a dataset.
    
    Args:
        max_batches: Limit number of batches (None for all)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
                
            x, y = x.to(device), y.to(device)
            
            output = model(x, labels=y)
            loss = output.loss
            
            # Calculate accuracy
            predictions = output.logits.argmax(dim=-1)
            correct = (predictions == y).sum().item()
            
            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            total_correct += correct
    
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = compute_perplexity(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'total_tokens': total_tokens
    }


def evaluate_generation_quality(
    model: GPTModel,
    tokenizer: CharTokenizer,
    prompts: List[str],
    max_tokens: int = 100,
    device: torch.device = None
) -> Dict[str, any]:
    """
    Evaluate generation quality metrics.
    
    Returns:
        Dictionary with generation metrics and samples
    """
    model.eval()
    
    generated_texts = []
    all_tokens = []
    
    for prompt in prompts:
        # Encode prompt
        prompt_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], device=device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
        
        # Decode
        generated_tokens = output_ids[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)
        
        generated_texts.append({
            'prompt': prompt,
            'generated': generated_text,
            'new_tokens': len(generated_tokens) - len(prompt_ids)
        })
        
        # Track tokens for diversity metrics
        all_tokens.extend(generated_tokens[len(prompt_ids):])
    
    # Calculate diversity metrics
    distinct_1 = compute_distinct_n(all_tokens, n=1)
    distinct_2 = compute_distinct_n(all_tokens, n=2)
    distinct_3 = compute_distinct_n(all_tokens, n=3)
    repetition_rate = compute_repetition_rate(all_tokens)
    
    return {
        'samples': generated_texts,
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'distinct_3': distinct_3,
        'repetition_rate': repetition_rate,
        'total_generated_tokens': len(all_tokens)
    }


def print_results(
    eval_metrics: Dict,
    gen_metrics: Optional[Dict] = None,
    checkpoint_info: Optional[Dict] = None
):
    """Print evaluation results in a nice format."""
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    if checkpoint_info:
        print(f"\nCheckpoint Info:")
        print(f"  Epoch: {checkpoint_info.get('epoch', 'N/A')}")
        print(f"  Step: {checkpoint_info.get('step', 'N/A')}")
        print(f"  Training Loss: {checkpoint_info.get('loss', 'N/A')}")
    
    print(f"\n📊 Language Modeling Metrics:")
    print(f"  Validation Loss: {eval_metrics['loss']:.4f}")
    print(f"  Perplexity: {eval_metrics['perplexity']:.2f}")
    print(f"  Token Accuracy: {eval_metrics['accuracy']*100:.2f}%")
    print(f"  Tokens Evaluated: {eval_metrics['total_tokens']:,}")
    
    if gen_metrics:
        print(f"\n📝 Generation Quality Metrics:")
        print(f"  Distinct-1: {gen_metrics['distinct_1']:.4f}")
        print(f"  Distinct-2: {gen_metrics['distinct_2']:.4f}")
        print(f"  Distinct-3: {gen_metrics['distinct_3']:.4f}")
        print(f"  Repetition Rate: {gen_metrics['repetition_rate']:.4f}")
        
        print(f"\n📖 Sample Generations:")
        for i, sample in enumerate(gen_metrics['samples'], 1):
            print(f"\n  Sample {i}:")
            print(f"  Prompt: {sample['prompt'][:50]}...")
            print(f"  Generated ({sample['new_tokens']} tokens):")
            # Show first 200 chars of generation
            gen_text = sample['generated'][len(sample['prompt']):]
            print(f"  {gen_text[:200]}...")
    
    print("\n" + "=" * 70)


def main():
    args = parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        if not checkpoint_path:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            print("Train a model first with: python train.py")
            sys.exit(1)
    
    # Load model
    model, tokenizer, checkpoint_info = load_model_and_tokenizer(
        checkpoint_path, device
    )
    
    # Create validation dataloader
    print("\nLoading validation data...")
    _, val_loader, _ = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=128
    )
    
    # Evaluate perplexity
    print("\nEvaluating perplexity...")
    eval_metrics = evaluate_perplexity(model, val_loader, device)
    
    # Evaluate generation quality
    gen_metrics = None
    if args.generate:
        print("\nEvaluating generation quality...")
        prompts = [
            "To be or not to be",
            "The king said",
            "In the forest",
            "Once upon a time",
            "The moon was"
        ][:args.n_samples]
        
        gen_metrics = evaluate_generation_quality(
            model, tokenizer, prompts,
            max_tokens=args.max_tokens,
            device=device
        )
    
    # Print results
    print_results(eval_metrics, gen_metrics, checkpoint_info)
    
    # Save results if requested
    if args.output:
        results = {
            'checkpoint': checkpoint_path,
            'checkpoint_info': checkpoint_info,
            'eval_metrics': eval_metrics,
            'gen_metrics': gen_metrics
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {args.output}")
    
    return eval_metrics, gen_metrics


if __name__ == "__main__":
    main()
