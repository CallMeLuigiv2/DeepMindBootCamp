# Seq2Seq with Attention

## Overview

Build a complete sequence-to-sequence model with Bahdanau attention for date format conversion. You will first implement a vanilla encoder-decoder, then add attention and observe how it resolves the information bottleneck. Finally, you will visualize attention weights and implement beam search decoding.

## Learning Objectives

- Implement an encoder-decoder architecture from scratch
- Understand the information bottleneck in fixed-context-vector seq2seq models
- Implement Bahdanau (additive) attention: $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$, context vector $c_i = \sum_j \alpha_{ij} h_j$
- Visualize attention heatmaps and interpret learned alignment patterns
- Implement beam search with length normalization

## The Task: Date Format Conversion

Convert human-readable dates to machine-readable format:

```
"January 5, 2023"      -->  "2023-01-05"
"5th of March, 1998"   -->  "1998-03-05"
"Mar 21, 2001"         -->  "2001-03-21"
```

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train the model (vanilla encoder-decoder first, then with attention)
python train.py --config config.yaml

# Train with attention
python train.py --config config.yaml --use-attention

# Evaluate and generate attention visualizations
python evaluate.py --checkpoint checkpoints/best_model.pt --visualize

# Run beam search comparison
python evaluate.py --checkpoint checkpoints/best_model.pt --beam-search --beam-widths 1 3 5 10
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | Encoder, Decoder, BahdanauAttention, and Seq2Seq model classes |
| `data.py` | Synthetic date dataset generation, character tokenization, DataLoader creation |
| `train.py` | Training loop with teacher forcing, validation, checkpointing |
| `evaluate.py` | Evaluation metrics, attention visualization, beam search decoding |
| `utils.py` | Logging, metric tracking, visualization helpers |
| `config.yaml` | Default hyperparameters |
| `notebooks/analysis.ipynb` | Interactive exploration and attention analysis |
