# Transformer Hackathon 🚀

Build your own GPT-style transformer model from scratch and compete on the leaderboard!

**🏆 Leaderboard:** [https://huggingface.co/datasets/abhisu30/transformer-hackathon-leaderboard](https://huggingface.co/datasets/abhisu30/transformer-hackathon-leaderboard)

## Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the hackathon pipeline (60 minutes on T4 GPU)
python run_hackathon.py --time 60

# 3. Follow the prompts!
```

That's it! The script will automatically:
- Download **TinyStories dataset** (~100 MB, 40K stories)
- Train your model for **60 minutes**
- Evaluate performance (perplexity, tokens/sec)
- Upload to the leaderboard

**Expected Results (T4 GPU):**
- Final Perplexity: **10-18**
- Training Speed: **1,500-2,000 tokens/sec**
- Total Tokens Processed: **~6-8 million**

---

## 🤖 Chat with Your Model (ChatGPT-Style!)

After training, interact with your model in real-time:

```bash
# Start interactive chat
python chat.py
```

**What you'll see:**
```
You: Once upon a time
🤖 Model: there was a little girl named Lily. She loved to play with her toys...

You: A boy named
🤖 Model: Tom went to the park. He saw a big tree and wanted to climb it...
```

**Built-in features:**
- 💡 10 prompt suggestions for TinyStories
- 🎛️ Adjustable creativity (temperature)
- 🏆 Auto-loads your best checkpoint
- ⌨️ Type `help` for suggestions, `quit` to exit

**Customize generation:**
```bash
python chat.py --temperature 1.0    # More creative
python chat.py --max-tokens 300     # Longer responses
```

📖 **Full guide:** [CHAT_GUIDE.md](CHAT_GUIDE.md)

---

## 📁 Repository Structure

```
Transformer_Repo-/
├── README.md                 # This file
├── COLAB_GUIDE.md           # Google Colab instructions
├── requirements.txt          # Python dependencies
├── config.py                 # Hyperparameters (MODIFY THIS!)
│
├── model/                    # 🧠 Transformer components
│   ├── Notebooks (for learning):
│   │   ├── Attention.ipynb        # ✨ Refactored with practice cells
│   │   ├── Embeddings.ipynb       # ✨ Refactored with practice cells
│   │   ├── FeedForward.ipynb      # ✨ Refactored with practice cells
│   │   ├── Decoder.ipynb          # Decoder blocks
│   │   └── Transformer.ipynb      # Complete GPT model
│   │
│   └── Python files (used by training):
│       ├── embeddings.py          # Token + positional embeddings
│       ├── attention.py           # Multi-head self-attention
│       ├── feedforward.py         # Feed-forward network
│       ├── decoder_block.py       # Decoder layer (GPT-style)
│       ├── decoder.py             # Full decoder stack
│       └── transformer.py         # Complete GPT model
│
├── data/                     # 📚 Data handling
│   ├── tokenizer.py          # Character-level tokenizer
│   ├── dataset.py            # Dataset loading (TinyStories 40K)
│   └── Data_Processing.ipynb # Data exploration notebook
│
├── utils/                    # 🔧 Utilities
│   ├── metrics.py            # Evaluation metrics
│   ├── checkpoint.py         # Save/load models
│   └── huggingface_upload.py # Leaderboard integration
│
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── generate.py               # Text generation
├── chat.py                   # 🤖 Interactive chat interface (NEW!)
└── run_hackathon.py          # 🏆 Main hackathon script
```

---

## 🏗️ Architecture Overview

**GPT-Style Decoder-Only Transformer** (No Encoder!)

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPT-Style Transformer                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: "The cat sat on the"                                    │
│              ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  TOKEN EMBEDDING + POSITIONAL ENCODING                   │   │
│  │  • Convert tokens to vectors                             │   │
│  │  • Add position information (sinusoidal)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│              ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              DECODER BLOCK (×6)                          │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  MASKED MULTI-HEAD SELF-ATTENTION                  │  │   │
│  │  │  • Q, K, V projections                             │  │   │
│  │  │  • 8 attention heads                               │  │   │
│  │  │  • Causal masking (can only see past)              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │              ↓ + Residual + LayerNorm                    │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  FEED-FORWARD NETWORK                              │  │   │
│  │  │  • Linear(512 → 2048)                              │  │   │
│  │  │  • GELU activation                                 │  │   │
│  │  │  • Linear(2048 → 512)                              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │              ↓ + Residual + LayerNorm                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│              ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  OUTPUT PROJECTION                                        │   │
│  │  • Linear(512 → vocab_size)                              │   │
│  │  • Softmax → probability distribution                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│              ↓                                                   │
│  Output: "mat" (predicted next token)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Default Configuration:
• d_model = 512 (embedding dimension)
• n_heads = 8 (attention heads)
• n_layers = 6 (decoder blocks)
• d_ff = 2048 (feed-forward dimension)
• max_seq_len = 128 (context window)
• vocab_size = ~65 (character-level)
• Parameters = ~10M
• Dataset = TinyStories (40K stories, ~100 MB)
• Training Time = 60 minutes (T4 GPU)
```

---

## 📓 Learning with Notebooks

The repository includes **refactored Jupyter notebooks** for hands-on learning:

### ✨ Refactored Notebooks (with Practice Cells)

1. **Attention.ipynb** - Multi-head self-attention
   - Individual cells for each function
   - Practice cells after each implementation
   - Learn by doing!

2. **Embeddings.ipynb** - Token and positional embeddings
   - TokenEmbedding, PositionalEncoding, TransformerEmbedding
   - Step-by-step breakdown

3. **FeedForward.ipynb** - Position-wise feed-forward networks
   - Simple but crucial component
   - Practice implementing from scratch

### 📚 Complete Notebooks

4. **Decoder.ipynb** - Decoder blocks and stack
5. **Transformer.ipynb** - Complete GPT model

**How to Use:**
- Open notebooks in Jupyter or Google Colab
- Run cells sequentially
- Try implementing functions in practice cells
- Compare with reference implementations

## 🚀 One-Click Colab Setup

Run this cell to set everything up instantly!

```python
# 🚀 ONE-CLICK SETUP
# Run this cell to set everything up!

# Check GPU
import torch
assert torch.cuda.is_available(), "⚠️ Enable GPU: Runtime > Change runtime type > GPU"

# Clone repository
!git clone https://github.com/abhishekadile/Transformer_Repo-.git
%cd Transformer_Repo-

# Install dependencies
!pip install -r requirements.txt

# Start Hackathon! (5 minutes on MacbookPro)
!python run_hackathon.py --time 5 --name Glove 
```

---

## 🤖 Interactive Chat with Your Model

Chat with your trained model in real-time using our ChatGPT-like interface.

```python
# After training, chat with your model!
!python chat.py
```

### Features

- 💬 **ChatGPT-like interface** - Interactive terminal chat
- 💡 **Smart prompt suggestions** - Tailored for TinyStories
- 🏆 **Auto-loads best checkpoint** - Uses `best.pt` automatically
- ⚡ **Real-time generation** - See your model's creativity!

### Example Session

```
You: Once upon a time
🤖 Model: there was a little girl named Lily. She loved to play with her toys...

You: A boy named
🤖 Model: Tom went to the park. He saw a big tree and wanted to climb it...
```

### Prompt Suggestions

The chat interface includes 10 built-in prompts perfect for TinyStories:
- "Once upon a time"
- "One day, a little"
- "There was a"
- "A boy named" / "A girl named"
- And more!

### Custom Settings

```python
# Longer responses
!python chat.py --max-tokens 300

# More creative (higher temperature)
!python chat.py --temperature 1.0

# More focused (lower temperature)
!python chat.py --temperature 0.5
```

📖 **Full guide:** See [CHAT_GUIDE.md](CHAT_GUIDE.md) for detailed usage and tips!

---


## 🔧 Optimization Ideas

Here are proven techniques to improve your model:

### Easy Wins 🟢

```python
# config.py
# 1. Enable mixed precision (2x speedup on modern GPUs!)
config.training.use_amp = True

# 2. Increase batch size if memory allows
config.training.batch_size = 32

# 3. Try different learning rates
config.training.learning_rate = 1e-3  # or 5e-4
```

### Medium Difficulty 🟡

```python
# 1. Gradient accumulation for larger effective batch size
config.training.gradient_accumulation_steps = 4

# 2. Bigger model (if GPU memory allows)
config.model.d_model = 768
config.model.n_layers = 8

# 3. Adjust dataset size
config.data.max_stories = 50000  # More data = better model
```

### Advanced 🔴

```python
# 1. Implement KV caching for faster generation
# See model/transformer.py - GPTModel.generate()

# 2. Try different attention mechanisms
# Flash Attention, Linear Attention, etc.

# 3. Experiment with different architectures
# Add cross-attention, use different normalization, etc.
```

---

## 📊 Dataset Information

**TinyStories Dataset**
- **Source:** Hugging Face `roneneldan/TinyStories`
- **Default Size:** 40,000 stories (~100 MB)
- **Format:** Simple children's stories
- **Tokenization:** Character-level (vocab size ~65)

**Why TinyStories?**
- Fast to download and process
- Simple language = easier to learn
- Good for 60-minute training sessions
- Produces coherent output

**Adjusting Dataset Size:**
```python
# config.py
config.data.max_stories = 50000  # Increase for better quality
# Training time scales linearly with dataset size
```

---

## 🎯 Training Tips

### For Google Colab (T4 GPU)

```bash
# Recommended settings for 60-minute training
python run_hackathon.py --time 60 --batch-size 16 --use-amp
```

**Expected Performance:**
- Perplexity: 10-18
- Speed: 1,500-2,000 tokens/sec
- Total tokens: ~6-8 million

### For Local Training (CPU)

```bash
# Slower but works!
python run_hackathon.py --time 10 --batch-size 8
```

**Expected Performance:**
- Perplexity: 15-25 (less training time)
- Speed: 500-1,000 tokens/sec
- Use smaller dataset for faster iteration

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory**
```python
# Reduce batch size
config.training.batch_size = 8

# Or reduce model size
config.model.d_model = 256
config.model.n_layers = 4
```

**2. Slow Training**
```python
# Enable mixed precision
config.training.use_amp = True

# Reduce dataset size
config.data.max_stories = 20000
```

**3. Poor Generation Quality**
```python
# Train longer
python run_hackathon.py --time 120

# Use more data
config.data.max_stories = 60000

# Bigger model
config.model.d_model = 768
```

---

## 📚 Additional Resources

- **Original Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **GPT Paper:** [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Illustrated Transformer:** [Jay Alammar's Blog](http://jalammar.github.io/illustrated-transformer/)
- **TinyStories Paper:** [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)

---

## 🤝 Contributing

Found a bug? Have an optimization idea? PRs welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-optimization`)
3. Commit your changes (`git commit -m 'Add amazing optimization'`)
4. Push to the branch (`git push origin feature/amazing-optimization`)
5. Open a Pull Request

---

## 📝 License

MIT License - feel free to use this for learning and teaching!

---

## 🙏 Acknowledgments

- TinyStories dataset by Microsoft Research
- Inspired by Andrej Karpathy's nanoGPT
- Built for educational purposes

---

**Happy Hacking! 🚀**

Questions? Open an issue or reach out on the leaderboard discussion!
