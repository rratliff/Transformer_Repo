#!/usr/bin/env python3
"""
Interactive Chat Interface for Transformer Model

Chat with your trained model in real-time! This script provides:
- ChatGPT-like interactive interface
- Helpful prompt suggestions for TinyStories
- Auto-loads the best checkpoint
- Real-time generation with streaming output

Usage:
    python chat.py                    # Use best checkpoint
    python chat.py --checkpoint final.pt  # Use specific checkpoint
"""

import argparse
import os
import sys
from typing import Optional

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPTModel
from data import CharTokenizer
from utils.checkpoint import get_latest_checkpoint


# Prompt suggestions tailored for TinyStories
PROMPT_SUGGESTIONS = [
    "Once upon a time",
    "One day, a little",
    "There was a",
    "A boy named",
    "A girl named",
    "In a small town",
    "The sun was shining",
    "It was a beautiful day",
    "A cat and a dog",
    "The little bird",
]


def print_banner():
    """Print the chat interface banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║                    🤖 TRANSFORMER CHAT INTERFACE                      ║
║                                                                       ║
║   Chat with your trained model in real-time!                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_prompt_suggestions():
    """Print helpful prompt suggestions."""
    print("\n💡 Suggested Prompts (TinyStories work best!):")
    print("-" * 70)
    for i, prompt in enumerate(PROMPT_SUGGESTIONS, 1):
        print(f"  {i}. {prompt}")
    print("-" * 70)
    print("\nℹ️  Tips:")
    print("  • Use simple, story-like prompts")
    print("  • Start with common phrases like 'Once upon a time'")
    print("  • Keep prompts short (3-5 words)")
    print("  • Type 'help' for more suggestions")
    print("  • Type 'quit' or 'exit' to leave")
    print()


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    print(f"\n📂 Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocab_size from checkpoint (critical for TinyStories!)
    vocab_size = checkpoint.get('vocab_size', None)
    
    # If not in checkpoint, try to get from tokenizer
    if vocab_size is None:
        if 'vocab' in checkpoint:
            vocab_size = len(checkpoint['vocab'])
        elif 'tokenizer' in checkpoint and 'char_to_id' in checkpoint['tokenizer']:
            vocab_size = len(checkpoint['tokenizer']['char_to_id'])
        else:
            vocab_size = 65  # Fallback default
    
    # Get model config
    model_config = checkpoint.get('model_config', {})
    
    # Create model with correct vocab_size
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=model_config.get('d_model', 512),
        n_heads=model_config.get('n_heads', 8),
        n_layers=model_config.get('n_layers', 6),
        d_ff=model_config.get('d_ff', 2048),
        max_seq_len=model_config.get('max_seq_len', 128),
        dropout=0.0,  # No dropout for inference
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load tokenizer with proper vocabulary
    tokenizer = CharTokenizer()
    
    # Try multiple ways to load the vocabulary
    if 'tokenizer' in checkpoint and 'char_to_id' in checkpoint['tokenizer']:
        # Load from tokenizer dict
        tokenizer.char_to_id = checkpoint['tokenizer']['char_to_id']
        tokenizer.id_to_char = {v: k for k, v in tokenizer.char_to_id.items()}
    elif 'vocab' in checkpoint:
        # Load from vocab list
        tokenizer.vocab = checkpoint['vocab']
        tokenizer.char_to_id = {c: i for i, c in enumerate(tokenizer.vocab)}
        tokenizer.id_to_char = {i: c for i, c in enumerate(tokenizer.vocab)}
    else:
        print("⚠️  Warning: Could not load vocabulary from checkpoint")
        print("   Using default vocabulary (may produce incorrect results)")
    
    # Print model info
    print(f"✅ Model loaded successfully!")
    print(f"   • Parameters: {model.get_num_params():,}")
    print(f"   • Vocabulary size: {vocab_size}")
    print(f"   • Training step: {checkpoint.get('step', 'unknown')}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if 'perplexity' in metrics:
            print(f"   • Perplexity: {metrics['perplexity']:.2f}")
    
    return model, tokenizer


def generate_response(
    model: GPTModel,
    tokenizer: CharTokenizer,
    prompt: str,
    device: torch.device,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
) -> str:
    """Generate a response from the model."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    # Return only the new text (remove prompt)
    return generated_text[len(prompt):]


def chat_loop(model: GPTModel, tokenizer: CharTokenizer, device: torch.device):
    """Main chat loop."""
    print("\n" + "=" * 70)
    print("💬 CHAT SESSION STARTED")
    print("=" * 70)
    print("Type your prompt and press Enter to generate a response.")
    print("Commands: 'help', 'suggestions', 'settings', 'quit', 'exit'")
    print("=" * 70 + "\n")
    
    # Default generation settings
    settings = {
        'max_tokens': 200,
        'temperature': 0.8,
        'top_k': 40,
        'top_p': 0.9,
    }
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            
            elif user_input.lower() == 'help':
                print_prompt_suggestions()
                continue
            
            elif user_input.lower() == 'suggestions':
                print("\n💡 Quick suggestions:")
                for i, prompt in enumerate(PROMPT_SUGGESTIONS[:5], 1):
                    print(f"  {i}. {prompt}")
                print()
                continue
            
            elif user_input.lower() == 'settings':
                print("\n⚙️  Current Settings:")
                print(f"  • Max tokens: {settings['max_tokens']}")
                print(f"  • Temperature: {settings['temperature']}")
                print(f"  • Top-k: {settings['top_k']}")
                print(f"  • Top-p: {settings['top_p']}")
                print("\nTo change settings, edit the script or use command-line args.")
                print()
                continue
            
            # Generate response
            print("\n🤖 Model: ", end="", flush=True)
            
            try:
                response = generate_response(
                    model, tokenizer, user_input, device,
                    **settings
                )
                print(response)
                print()
                
            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print("Try a different prompt or check your model.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            continue


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Chat with your trained transformer model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: best.pt or latest)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9)"
    )
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU (slower)")
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Try best.pt first, then final.pt, then latest
        best_path = os.path.join(args.checkpoint_dir, "best.pt")
        final_path = os.path.join(args.checkpoint_dir, "final.pt")
        
        if os.path.exists(best_path):
            checkpoint_path = best_path
            print("📊 Using best checkpoint (lowest perplexity)")
        elif os.path.exists(final_path):
            checkpoint_path = final_path
            print("📊 Using final checkpoint")
        else:
            checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
            if checkpoint_path:
                print("📊 Using latest checkpoint")
            else:
                print("❌ No checkpoints found!")
                print(f"   Please train a model first: python train.py")
                sys.exit(1)
    
    # Load model
    model, tokenizer = load_model(checkpoint_path, device)
    
    # Print prompt suggestions
    print_prompt_suggestions()
    
    # Update settings from args
    settings = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
    }
    
    # Start chat loop
    chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()
