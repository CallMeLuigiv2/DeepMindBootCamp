"""
Transformer from Scratch - Evaluation Script

Evaluates the sorting Transformer or GPT language model.

Usage:
    python evaluate.py --checkpoint checkpoints/sorting_final.pt --task sorting
    python evaluate.py --checkpoint checkpoints/gpt_final.pt --task language_model --generate
"""

import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data import create_char_dataloaders, CharDataset
from model import Transformer, GPT
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Transformer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str, choices=["sorting", "language_model"], required=True)
    parser.add_argument("--generate", action="store_true", help="Generate text (LM task)")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def evaluate_sorting_model(model, device, config):
    """Evaluate sorting accuracy across different sequence lengths.

    Returns results broken down by sequence length.
    """
    # YOUR CODE HERE
    # 1. For each sequence length in [5, 6, 7, 8, 9, 10]:
    #    a. Generate 200 random sequences
    #    b. Run greedy decoding
    #    c. Compute exact-match accuracy
    # 2. Print results table
    # 3. Show 10 example predictions (input, expected, predicted)
    raise NotImplementedError("Implement evaluate_sorting_model")


def extract_attention_weights(model, src, tgt, device):
    """Extract attention weights from all layers and heads.

    Returns a dict mapping (layer_idx, head_idx) to attention weight matrices.
    """
    # YOUR CODE HERE
    # Run forward pass and collect attention weights from each layer
    raise NotImplementedError("Implement extract_attention_weights")


def visualize_sorting_attention(
    model, device, config, output_dir="results"
):
    """Visualize attention patterns from the sorting Transformer.

    Shows how different heads attend to different parts of the input.
    """
    # YOUR CODE HERE
    # 1. Generate a sample sorting input
    # 2. Extract attention weights from all layers/heads
    # 3. Create a grid of heatmaps
    raise NotImplementedError("Implement visualize_sorting_attention")


def generate_text_samples(model, dataset, device, temperatures, num_tokens=200):
    """Generate text at different temperatures and display results."""
    set_seed(42)
    prompt_text = "ROMEO:\n"
    prompt_ids = [dataset.char2idx.get(ch, 0) for ch in prompt_text]
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"Prompt: {repr(prompt_text)}\n")
    print("=" * 60)

    for temp in temperatures:
        generated = model.generate(prompt, max_new_tokens=num_tokens, temperature=temp)
        text = dataset.decode(generated[0].tolist())
        print(f"\nTemperature = {temp}:")
        print("-" * 40)
        print(text)
        print()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.task == "sorting":
        # Reconstruct and load model
        # model = Transformer(...)
        # model.load_state_dict(checkpoint["model_state_dict"])
        # evaluate_sorting_model(model, device, config)
        # visualize_sorting_attention(model, device, config, args.output_dir)
        print("TODO: Load sorting model and evaluate")

    elif args.task == "language_model":
        if args.generate:
            # Reconstruct and load model
            # model = GPT(...)
            # model.load_state_dict(checkpoint["model_state_dict"])
            # generate_text_samples(model, dataset, device, args.temperatures)
            print("TODO: Load GPT model and generate text")


if __name__ == "__main__":
    main()
