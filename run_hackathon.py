#!/usr/bin/env python3
"""
Automated Hackathon Pipeline for the Transformer Hackathon.

This is the MAIN script participants should run. It:
    1. Prompts for participant name and team
    2. Confirms/configures Hugging Face token
    3. Trains for exactly 45 minutes
    4. Runs evaluation metrics
    5. Generates sample text
    6. Uploads results to leaderboard
    7. Displays final metrics and ranking

Usage:
    python run_hackathon.py
    
That's it! Follow the prompts.
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import Optional, Dict

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print the hackathon banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ████████╗██████╗  █████╗ ███╗   ██╗███████╗███████╗ ██████╗        ║
║   ╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔═══██╗       ║
║      ██║   ██████╔╝███████║██╔██╗ ██║███████╗█████╗  ██║   ██║       ║
║      ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██╔══╝  ██║   ██║       ║
║      ██║   ██║  ██║██║  ██║██║ ╚████║███████║██║     ╚██████╔╝       ║
║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝      ╚═════╝        ║
║                                                                       ║
║                    HACKATHON RUNNER v1.0                              ║
║                                                                       ║
║   Build Your Own Transformer • Train • Compete • Win!                ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def get_participant_info() -> Dict[str, str]:
    """Get participant name and team from user input."""
    print("\n📝 PARTICIPANT REGISTRATION")
    print("-" * 50)
    
    # Get name
    while True:
        name = input("Enter your name: ").strip()
        if name:
            break
        print("Name cannot be empty. Please try again.")
    
    # Get team (optional)
    team = input("Enter your team name (press Enter to skip): ").strip()
    if not team:
        team = "Solo"
    
    # Confirm
    print(f"\n✓ Name: {name}")
    print(f"✓ Team: {team}")
    
    confirm = input("\nIs this correct? (Y/n): ").strip().lower()
    if confirm in ('n', 'no'):
        return get_participant_info()
    
    return {'name': name, 'team': team}


def check_huggingface_token() -> Optional[str]:
    """Check Hugging Face token from environment (Colab secrets or env var)."""
    print("\n🔑 LEADERBOARD CONFIGURATION")
    print("-" * 50)
    
    # Check environment variable (set by Colab secrets or manually)
    token = os.environ.get("HF_TOKEN")
    if token:
        masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "****"
        print(f"✅ Found HF token: {masked}")
        print("   Leaderboard upload enabled!")
        return token
    
    print("ℹ️  No HF_TOKEN found in environment")
    print("✅ Using embedded hackathon token for leaderboard!")
    print("   (Code will auto-upload to the shared leaderboard)")
    return None


def run_training(training_time: float = 45.0) -> Dict:
    """Run the training process."""
    print("\n🏋️ TRAINING PHASE")
    print("-" * 50)
    print(f"Training will run for {training_time} minutes")
    print("Feel free to watch the progress or grab a coffee! ☕")
    print()
    
    # Import training module
    from train import train
    
    # Create args namespace
    class Args:
        max_time = 60  # Updated from 45 for 1-hour training
        batch_size = 16
        seq_len = 128
        lr = 3e-4
        d_model = 512
        n_layers = 6
        n_heads = 8
        use_amp = torch.cuda.is_available()  # Auto-enable on GPU
        grad_accum_steps = 1
        max_grad_norm = 1.0
        checkpoint_dir = "checkpoints"
        checkpoint_interval = 5.0
        resume = False
        data_path = None
        log_interval = 100  # Log every 100 steps (reduced from 10 to avoid spam)
        eval_interval = 500
        seed = 42
        quiet = False
    
    args = Args()
    
    # Run training
    model, tokenizer, metrics = train(args)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'metrics': metrics.get_summary(),
        'checkpoint_dir': args.checkpoint_dir
    }


def run_evaluation(checkpoint_dir: str) -> Dict:
    """Run evaluation on the trained model."""
    print("\n📊 EVALUATION PHASE")
    print("-" * 50)
    
    from evaluate import (
        load_model_and_tokenizer,
        evaluate_perplexity,
        evaluate_generation_quality
    )
    from data import create_dataloaders
    from utils.checkpoint import get_latest_checkpoint
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint_path = os.path.join(checkpoint_dir, "final.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    
    model, tokenizer, _ = load_model_and_tokenizer(checkpoint_path, device)
    
    # Create validation dataloader
    _, val_loader, _ = create_dataloaders(batch_size=32, seq_len=128)
    
    # Evaluate perplexity
    print("Computing perplexity...")
    # Limit to 100 batches for faster evaluation (~3,200 tokens)
    # This is sufficient for accurate perplexity estimation
    eval_metrics = evaluate_perplexity(model, val_loader, device, max_batches=100)
    
    # Evaluate generation quality
    print("Evaluating generation quality...")
    prompts = [
        "To be or not to be",
        "The king proclaimed",
        "In the depths of",
        "Once upon a time",
        "The stars above"
    ]
    gen_metrics = evaluate_generation_quality(
        model, tokenizer, prompts,
        max_tokens=100, device=device
    )
    
    return {
        'eval_metrics': eval_metrics,
        'gen_metrics': gen_metrics
    }


def upload_results(
    participant: Dict,
    metrics: Dict,
    eval_results: Dict,
    hf_token: Optional[str]
) -> bool:
    """Upload results to leaderboard."""
    print("\n📤 UPLOADING RESULTS")
    print("-" * 50)
    
    from utils.huggingface_upload import upload_to_leaderboard
    
    return upload_to_leaderboard(
        name=participant['name'],
        team=participant['team'],
        perplexity=eval_results['eval_metrics']['perplexity'],
        loss=eval_results['eval_metrics']['loss'],
        tokens_per_sec=metrics.get('avg_tokens_per_sec', 0),
        token=hf_token,
        model_config={
            'total_steps': metrics.get('total_steps', 0),
            'total_tokens': metrics.get('total_tokens', 0),
        }
    )


def print_final_results(
    participant: Dict,
    metrics: Dict,
    eval_results: Dict,
    training_time: float
):
    """Print final results summary."""
    eval_m = eval_results['eval_metrics']
    gen_m = eval_results['gen_metrics']
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "🏆 FINAL RESULTS 🏆" + " " * 21 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Participant: {participant['name']:<53}║")
    print(f"║  Team: {participant['team']:<60}║")
    print("╠" + "═" * 68 + "╣")
    print("║  PERFORMANCE METRICS" + " " * 47 + "║")
    print(f"║  • Perplexity: {eval_m['perplexity']:<52.2f}║")
    print(f"║  • Validation Loss: {eval_m['loss']:<47.4f}║")
    print(f"║  • Token Accuracy: {eval_m['accuracy']*100:<47.2f}%║")
    print("╠" + "═" * 68 + "╣")
    print("║  EFFICIENCY METRICS" + " " * 48 + "║")
    print(f"║  • Tokens/Second: {metrics.get('avg_tokens_per_sec', 0):<49.0f}║")
    print(f"║  • Training Time: {training_time:<49.1f}min║")
    print(f"║  • Total Tokens: {metrics.get('total_tokens', 0):<50,}║")
    print("╠" + "═" * 68 + "╣")
    print("║  GENERATION QUALITY" + " " * 48 + "║")
    print(f"║  • Distinct-1: {gen_m['distinct_1']:<52.4f}║")
    print(f"║  • Distinct-2: {gen_m['distinct_2']:<52.4f}║")
    print(f"║  • Repetition Rate: {gen_m['repetition_rate']:<47.4f}║")
    print("╚" + "═" * 68 + "╝")
    
    # Sample generation
    print("\n📖 SAMPLE GENERATION")
    print("-" * 70)
    sample = gen_m['samples'][0]
    print(f"Prompt: {sample['prompt']}")
    print(f"Generated:\n{sample['generated'][:300]}...")
    print("-" * 70)


def save_results_locally(
    participant: Dict,
    metrics: Dict,
    eval_results: Dict,
    training_time: float
):
    """Save results to local file."""
    os.makedirs("results", exist_ok=True)
    
    results = {
        'participant': participant,
        'timestamp': datetime.utcnow().isoformat(),
        'training_time_minutes': training_time,
        'training_metrics': metrics,
        'evaluation': {
            'perplexity': eval_results['eval_metrics']['perplexity'],
            'loss': eval_results['eval_metrics']['loss'],
            'accuracy': eval_results['eval_metrics']['accuracy'],
            'distinct_1': eval_results['gen_metrics']['distinct_1'],
            'distinct_2': eval_results['gen_metrics']['distinct_2'],
            'repetition_rate': eval_results['gen_metrics']['repetition_rate'],
        },
        'samples': eval_results['gen_metrics']['samples']
    }
    
    filename = f"results/hackathon_results_{participant['name'].replace(' ', '_')}_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")


def main():
    """Main hackathon runner."""
    parser = argparse.ArgumentParser(description="Run the transformer hackathon pipeline")
    parser.add_argument("--time", type=float, default=45.0,
                       help="Training time in minutes (default: 45)")
    parser.add_argument("--skip-train", action="store_true",
                       help="Skip training (use existing checkpoint)")
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected. Training will be slower on CPU.")
        proceed = input("Continue anyway? (y/N): ").strip().lower()
        if proceed not in ('y', 'yes'):
            print("Exiting. Try running on a machine with a GPU!")
            sys.exit(0)
    
    # Get participant info
    participant = get_participant_info()
    
    # Check HuggingFace token
    hf_token = check_huggingface_token()
    
    # Confirm start
    print("\n" + "=" * 50)
    print("READY TO START")
    print("=" * 50)
    print(f"• Training time: {args.time} minutes")
    print(f"• Participant: {participant['name']}")
    print(f"• Team: {participant['team']}")
    # Show correct leaderboard status (embedded token is always available)
    if hf_token:
        print(f"• Leaderboard: Enabled (your HF token)")
    else:
        print(f"• Leaderboard: Enabled (embedded token)")
    print("=" * 50)
    
    input("\nPress Enter to start training... 🚀")
    
    # Track start time
    start_time = time.time()
    
    # Run training
    if not args.skip_train:
        training_result = run_training(args.time)
        metrics = training_result['metrics']
        checkpoint_dir = training_result['checkpoint_dir']
    else:
        metrics = {}
        checkpoint_dir = "checkpoints"
    
    # Run evaluation
    eval_results = run_evaluation(checkpoint_dir)
    
    # Calculate actual training time
    total_time = (time.time() - start_time) / 60
    
    # Upload results
    if hf_token or True:  # Always try, will save locally if no token
        upload_results(participant, metrics, eval_results, hf_token)
    
    # Print final results
    print_final_results(participant, metrics, eval_results, args.time)
    
    # Save locally
    save_results_locally(participant, metrics, eval_results, args.time)
    
    # Final message
    print("\n" + "=" * 70)
    print("🎉 HACKATHON RUN COMPLETE!")
    print("=" * 70)
    print("Your results have been recorded. Great job!")
    print("\nNext steps:")
    print("  1. Check your ranking on the leaderboard")
    print("  2. Try optimizing your model (see README.md for ideas)")
    print("  3. Run again with: python run_hackathon.py")
    print("\nGood luck! 🚀")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Goodbye!")
        sys.exit(0)
