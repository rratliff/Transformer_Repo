#!/usr/bin/env python3
"""
Update the HuggingFace leaderboard README with a formatted table.

This script fetches the leaderboard data and creates a nicely formatted
README.md with rankings sorted by perplexity (lower is better).
"""

import json
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

LEADERBOARD_REPO = "abhisu30/transformer-hackathon-leaderboard"


def fetch_leaderboard():
    """Fetch current leaderboard data."""
    try:
        file = hf_hub_download(
            repo_id=LEADERBOARD_REPO,
            filename="leaderboard.json",
            repo_type="dataset"
        )
        with open(file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error fetching leaderboard: {e}")
        return {"entries": []}


def create_readme(leaderboard_data):
    """Create a formatted README with leaderboard table."""
    entries = leaderboard_data.get("entries", [])
    
    # Sort by perplexity (lower is better)
    sorted_entries = sorted(entries, key=lambda x: x.get("perplexity", float('inf')))
    
    readme = """# 🏆 Transformer Hackathon Leaderboard

Build Your Own GPT • Train • Compete • Win!

## 📊 Current Standings

**Lower perplexity = better model performance**

| Rank | Name | Team | Perplexity ⬇️ | Loss | Tokens/sec | Timestamp |
|------|------|------|--------------|------|------------|-----------|
"""
    
    for rank, entry in enumerate(sorted_entries, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
        name = entry.get("name", "Unknown")
        team = entry.get("team", "-")
        perplexity = entry.get("perplexity", 0)
        loss = entry.get("loss", 0)
        tokens_per_sec = entry.get("tokens_per_sec", 0)
        timestamp = entry.get("timestamp", "")
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            time_str = timestamp[:16] if timestamp else "-"
        
        readme += f"| {medal} | {name} | {team} | **{perplexity:.4f}** | {loss:.4f} | {tokens_per_sec:.0f} | {time_str} |\n"
    
    if not sorted_entries:
        readme += "| - | No submissions yet | - | - | - | - | - |\n"
    
    readme += """
---

## 🎯 Competition Categories

- **Best Perplexity**: Lowest perplexity score wins! 🏆
- **Best Efficiency**: Highest tokens/second during training ⚡
- **Most Creative**: Best generated text samples 🎨

## 🚀 How to Participate

```bash
# Clone the repository
git clone https://github.com/abhishekadile/Transformer_Repo-
cd Transformer_Repo-

# Install dependencies
pip install -r requirements.txt

# Run the hackathon pipeline
python run_hackathon.py
```

The script will:
1. Download the TinyStories dataset
2. Train your model for 45 minutes
3. Evaluate performance
4. Submit to this leaderboard!

## 📈 Leaderboard Schema

- `name`: Participant name
- `team`: Team name (optional)
- `perplexity`: Model perplexity - **lower is better**
- `loss`: Final validation cross-entropy loss
- `tokens_per_sec`: Training throughput
- `timestamp`: Submission time

## 🔧 Optimization Tips

- 🚀 Enable mixed precision: `--use-amp`
- 📦 Increase batch size if GPU allows
- 📈 Try different learning rates
- 🧠 Experiment with model size in `config.py`

## 📚 Resources

- [GitHub Repository](https://github.com/abhishekadile/Transformer_Repo-)
- [Google Colab Guide](https://github.com/abhishekadile/Transformer_Repo-/blob/main/COLAB_GUIDE.md)
- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)

Good luck and have fun! 🎉
"""
    
    return readme


def upload_readme(readme_content, token):
    """Upload the README to HuggingFace."""
    try:
        api = HfApi(token=token)
        
        # Save README locally
        with open("temp_README.md", "w") as f:
            f.write(readme_content)
        
        # Upload
        api.upload_file(
            path_or_fileobj="temp_README.md",
            path_in_repo="README.md",
            repo_id=LEADERBOARD_REPO,
            repo_type="dataset",
            commit_message="Update leaderboard rankings"
        )
        
        # Cleanup
        import os
        os.remove("temp_README.md")
        
        print(f"✅ README updated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading README: {e}")
        return False


def main():
    """Main function."""
    # Fetch leaderboard
    print("Fetching leaderboard data...")
    leaderboard_data = fetch_leaderboard()
    
    # Create README
    print("Creating formatted README...")
    readme = create_readme(leaderboard_data)
    
    # Print preview
    print("\n" + "="*70)
    print("README PREVIEW:")
    print("="*70)
    print(readme[:500] + "...")
    print("="*70)
    
    # Get token
    from utils.huggingface_upload import HACKATHON_TOKEN
    
    # Upload
    print("\nUploading to HuggingFace...")
    upload_readme(readme, HACKATHON_TOKEN)
    
    print(f"\n🎉 View the leaderboard: https://huggingface.co/datasets/{LEADERBOARD_REPO}")


if __name__ == "__main__":
    main()
