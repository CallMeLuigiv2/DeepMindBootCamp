# Transformer from Scratch

## Overview

Build a complete Transformer architecture -- every component from positional encoding to the final softmax -- using only basic PyTorch building blocks. No `nn.TransformerEncoder`, no `nn.MultiheadAttention`. Then train it on a number sorting task and a character-level language model.

## Learning Objectives

- Implement sinusoidal positional encoding
- Build multi-head self-attention from scratch: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- Construct encoder and decoder blocks with residual connections and layer normalization
- Assemble a full encoder-decoder Transformer for sequence-to-sequence tasks
- Build a GPT-style decoder-only Transformer for autoregressive language modeling
- Understand causal masking, embedding scaling, and learning rate warmup

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train the encoder-decoder Transformer on number sorting
python train.py --config config.yaml --task sorting

# Train the GPT decoder-only model on character-level LM
python train.py --config config.yaml --task language_model --corpus data/shakespeare.txt

# Evaluate and generate text
python evaluate.py --checkpoint checkpoints/best_model.pt --task language_model --generate
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | PositionalEncoding, MultiHeadAttention, TransformerBlock, full Transformer, GPT |
| `data.py` | Sorting dataset generation, character-level text dataset, DataLoader creation |
| `train.py` | Training loop with warmup schedule, validation, checkpointing |
| `evaluate.py` | Sorting accuracy evaluation, text generation with temperature sampling |
| `utils.py` | Logging, LR scheduling, metric tracking, visualization helpers |
| `config.yaml` | Default hyperparameters for both tasks |
| `notebooks/analysis.ipynb` | Attention pattern visualization and ablation analysis |
