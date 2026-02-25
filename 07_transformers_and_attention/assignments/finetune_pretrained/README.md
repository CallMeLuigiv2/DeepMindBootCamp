# Fine-Tune Pretrained Transformer

## Overview

Fine-tune BERT for sentiment classification, analyze attention patterns from pretrained models, compare fine-tuning vs training from scratch, and implement GPT-2 text generation with custom sampling strategies.

## Learning Objectives

- Fine-tune `bert-base-uncased` with discriminative learning rates and warmup
- Visualize and interpret multi-layer, multi-head attention patterns
- Quantify the data efficiency of transfer learning vs training from scratch
- Implement greedy, temperature, top-k, and top-p sampling for GPT-2

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Fine-tune BERT on sentiment analysis
python train.py --config config.yaml --task finetune_bert

# Train a small Transformer from scratch for comparison
python train.py --config config.yaml --task from_scratch

# Evaluate and analyze attention patterns
python evaluate.py --checkpoint checkpoints/bert_best.pt --visualize-attention

# Generate text with GPT-2 using different sampling strategies
python evaluate.py --task generate --temperatures 0.5 0.7 1.0 1.5
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | HuggingFace model wrapper with classification head, from-scratch classifier |
| `data.py` | Dataset loading with HF tokenizer, DataLoader creation |
| `train.py` | Fine-tuning with discriminative LR, warmup, gradient clipping |
| `evaluate.py` | Metrics computation, attention visualization, GPT-2 generation |
| `utils.py` | Logging, metrics, attention rollout, visualization helpers |
| `config.yaml` | Hyperparameters for fine-tuning and from-scratch training |
| `notebooks/analysis.ipynb` | Interactive attention analysis and comparison plots |
