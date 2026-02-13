#!/usr/bin/env python3
"""
Text generation script for the Transformer Hackathon.

This script generates text using a trained GPT model with:
    - Interactive prompt mode
    - Configurable sampling (temperature, top-k, top-p)
    - Batch generation
    - Output to file option

Usage:
    python generate.py                           # Interactive mode
    python generate.py --prompt "Once upon"      # Generate from prompt
    python generate.py --prompt "The king" --n 5 # Generate 5 samples
    python generate.py --temperature 1.2         # More random
"""

import argparse
import os
import sys
from typing import Optional, List

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPTModel
from data import CharTokenizer, SimpleTokenizer
from utils.checkpoint import get_latest_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with a trained transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint (uses latest if not specified)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory containing checkpoints")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for generation (interactive if not provided)")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum tokens to generate")
    parser.add_argument("--n", type=int, default=1,
                       help="Number of samples to generate")
    
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                       help="Penalty for repeating tokens")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output file (prints to console if not specified)")
    parser.add_argument("--no-prompt", action="store_true",
                       help="Don't include prompt in output")
    
    return parser.parse_args()


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: torch.device
) -> tuple:
    """Load model and tokenizer from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    config = checkpoint.get('config', {})
    
    # Load tokenizer (auto-detect char vs word)
    tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.json")
    if os.path.exists(tokenizer_path):
        try:
            import json as _json
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tok_meta = _json.load(f)
            tok_type = tok_meta.get('type', 'char')
            if tok_type == 'word':
                tokenizer = SimpleTokenizer.load(tokenizer_path)
            else:
                tokenizer = CharTokenizer.load(tokenizer_path)
        except Exception:
            tokenizer = CharTokenizer.load(tokenizer_path)
    else:
        tok_data = checkpoint.get('tokenizer', {})
        if tok_data and tok_data.get('char_to_id'):
            tokenizer = CharTokenizer()
            tokenizer.char_to_id = tok_data['char_to_id']
            tokenizer.id_to_char = {v: k for k, v in tok_data['char_to_id'].items()}
        else:
            raise ValueError("Tokenizer not found")
    
    # Infer robust configuration from checkpoint when needed
    state_dict = checkpoint.get('model_state_dict', {})

    def infer_n_layers(sd):
        try:
            prefixes = [k for k in sd.keys() if k.startswith('decoder.layers.')]
            layer_indices = []
            for k in prefixes:
                parts = k.split('.')
                if len(parts) > 2 and parts[0] == 'decoder' and parts[1] == 'layers':
                    try:
                        layer_indices.append(int(parts[2]))
                    except ValueError:
                        pass
            return (max(layer_indices) + 1) if layer_indices else None
        except Exception:
            return None

    cfg_d_model = config.get('d_model', 512)
    cfg_n_heads = config.get('n_heads', 8)
    cfg_n_layers = config.get('n_layers', None)
    cfg_seq_len = config.get('seq_len', 128)

    inferred_layers = infer_n_layers(state_dict)
    if cfg_n_layers is None and inferred_layers is not None:
        cfg_n_layers = inferred_layers
    elif cfg_n_layers is not None and inferred_layers is not None and cfg_n_layers != inferred_layers:
        print(f"[WARNING] Checkpoint layer count ({inferred_layers}) differs from saved config ({cfg_n_layers}). Using {inferred_layers} to match weights.")
        cfg_n_layers = inferred_layers

    if cfg_n_layers is None:
        cfg_n_layers = 6

    # Create model
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg_d_model,
        n_heads=cfg_n_heads,
        n_layers=cfg_n_layers,
        d_ff=cfg_d_model * 4,
        max_seq_len=cfg_seq_len,
    )
    
    # Load weights (strict by default; if mismatch, retry non-strict)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[WARNING] Strict load failed with error: {e}\nRetrying with strict=False (some layers may be randomly initialized).")
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def generate_text(
    model: GPTModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: torch.device = None
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt to start generation
        max_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        repetition_penalty: Penalty for repeating tokens
        device: Device to run generation on
        
    Returns:
        Generated text (including prompt)
    """
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], device=device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text


def interactive_mode(
    model: GPTModel,
    tokenizer: CharTokenizer,
    args,
    device: torch.device
):
    """Run interactive generation mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TEXT GENERATION")
    print("=" * 60)
    print("Enter a prompt to generate text. Type 'quit' to exit.")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not prompt:
            continue
        
        if prompt.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        print("\nGenerating...\n")
        
        generated = generate_text(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device
        )
        
        print("-" * 60)
        print(generated)
        print("-" * 60 + "\n")


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
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Interactive mode if no prompt provided
    if args.prompt is None:
        interactive_mode(model, tokenizer, args, device)
        return
    
    # Generate from provided prompt
    print(f"\nGenerating {args.n} sample(s) from prompt: \"{args.prompt}\"")
    print("=" * 60)
    
    outputs = []
    for i in range(args.n):
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device
        )
        
        if args.no_prompt:
            generated = generated[len(args.prompt):]
        
        outputs.append(generated)
        
        if args.n > 1:
            print(f"\n--- Sample {i+1} ---")
        print(generated)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, output in enumerate(outputs):
                if args.n > 1:
                    f.write(f"--- Sample {i+1} ---\n")
                f.write(output)
                f.write("\n\n")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
