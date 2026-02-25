"""
Seq2Seq with Attention - Evaluation Script

Computes exact-match accuracy, generates attention visualizations,
and runs beam search comparison.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --visualize
    python evaluate.py --checkpoint checkpoints/best_model.pt --beam-search
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data import create_dataloaders, EOS_IDX, PAD_IDX, FORMATS
from model import Seq2Seq
from utils import visualize_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Seq2Seq model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--visualize", action="store_true", help="Generate attention heatmaps")
    parser.add_argument("--beam-search", action="store_true", help="Run beam search comparison")
    parser.add_argument("--beam-widths", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--num-viz", type=int, default=10, help="Number of attention visualizations")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    return parser.parse_args()


def decode_tokens(token_ids: torch.Tensor, idx2char: Dict[int, str]) -> str:
    """Convert token IDs back to a string, stopping at EOS."""
    chars = []
    for idx in token_ids.tolist():
        if idx == EOS_IDX:
            break
        if idx > 2:  # Skip PAD, BOS, EOS
            chars.append(idx2char.get(idx, "?"))
    return "".join(chars)


@torch.no_grad()
def compute_accuracy(
    model: Seq2Seq,
    data_loader,
    trg_idx2char: Dict[int, str],
    device: torch.device,
) -> Tuple[float, List[Tuple[str, str, str]]]:
    """Compute exact-match accuracy on a dataset.

    Returns:
        accuracy: Fraction of perfectly predicted sequences
        examples: List of (source, target, prediction) strings
    """
    model.eval()
    correct = 0
    total = 0
    examples = []

    for src, trg, src_mask in data_loader:
        src = src.to(device)
        src_mask = src_mask.to(device)

        predictions, _ = model.greedy_decode(src, src_mask=src_mask)

        for i in range(src.size(0)):
            pred_str = decode_tokens(predictions[i], trg_idx2char)
            trg_str = decode_tokens(trg[i], trg_idx2char)

            if pred_str == trg_str:
                correct += 1
            total += 1

            if len(examples) < 100:
                examples.append((None, trg_str, pred_str))

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, examples


def beam_search_decode(
    model: Seq2Seq,
    src: torch.Tensor,
    beam_width: int = 5,
    max_len: int = 12,
    length_norm_alpha: float = 0.6,
    src_mask: Optional[torch.Tensor] = None,
) -> List[Tuple[float, List[int]]]:
    """Beam search decoding for a single source sequence.

    Args:
        model: Trained Seq2Seq model
        src: (1, src_len) - Single source sequence
        beam_width: Number of beams to maintain
        max_len: Maximum output length
        length_norm_alpha: Exponent for length normalization
        src_mask: (1, src_len) - Source padding mask

    Returns:
        List of (score, token_ids) tuples sorted by score (highest first)
    """
    # YOUR CODE HERE
    # 1. Encode the source sequence
    # 2. Initialize beams: [(log_prob, [BOS], hidden_state, cell_state)]
    # 3. At each step:
    #    a. Expand each beam with top-K next tokens
    #    b. Keep the overall top-K beams
    #    c. Move completed beams (generated EOS) to finished list
    # 4. Apply length normalization: score = log_prob / (length ^ alpha)
    # 5. Return finished beams sorted by normalized score
    raise NotImplementedError("Implement beam_search_decode")


def evaluate_beam_search(
    model: Seq2Seq,
    data_loader,
    trg_idx2char: Dict[int, str],
    beam_widths: List[int],
    device: torch.device,
    length_norm_alpha: float = 0.6,
) -> Dict[int, float]:
    """Evaluate model with different beam widths.

    Returns:
        Dictionary mapping beam_width to accuracy.
    """
    # YOUR CODE HERE
    # For each beam width, run beam search on the test set and compute accuracy
    raise NotImplementedError("Implement evaluate_beam_search")


def generate_attention_visualizations(
    model: Seq2Seq,
    data_loader,
    src_idx2char: Dict[int, str],
    trg_idx2char: Dict[int, str],
    device: torch.device,
    num_examples: int = 10,
    output_dir: str = "results",
):
    """Generate and save attention heatmap visualizations.

    Produces one heatmap per example showing which input characters
    the decoder attended to when generating each output character.
    """
    # YOUR CODE HERE
    # 1. Run greedy decoding to collect attention weights
    # 2. For each example, create a heatmap with:
    #    - x-axis: input characters
    #    - y-axis: output characters
    #    - Intensity: attention weight
    # 3. Save each heatmap to output_dir
    raise NotImplementedError("Implement generate_attention_visualizations")


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_cfg = config["data"]
    _, _, test_loader, src_c2i, trg_c2i, src_i2c, trg_i2c = create_dataloaders(
        num_samples=data_cfg["num_samples"],
        train_size=data_cfg["train_size"],
        val_size=data_cfg["val_size"],
        test_size=data_cfg["test_size"],
        batch_size=config["training"]["batch_size"],
        seed=data_cfg["random_seed"],
    )

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Note: You need to reconstruct the model architecture before loading state_dict
    # model = build_model(...)
    # model.load_state_dict(checkpoint["model_state_dict"])

    # Compute accuracy
    accuracy, examples = compute_accuracy(model, test_loader, trg_i2c, device)
    print(f"\nExact-match accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("\nSample predictions:")
    for _, trg, pred in examples[:10]:
        status = "OK" if trg == pred else "WRONG"
        print(f"  Target: {trg}  Predicted: {pred}  [{status}]")

    # Attention visualization
    if args.visualize:
        generate_attention_visualizations(
            model, test_loader, src_i2c, trg_i2c, device,
            num_examples=args.num_viz, output_dir=args.output_dir,
        )
        print(f"\nAttention visualizations saved to {args.output_dir}/")

    # Beam search comparison
    if args.beam_search:
        beam_cfg = config["beam_search"]
        results = evaluate_beam_search(
            model, test_loader, trg_i2c, args.beam_widths, device,
            length_norm_alpha=beam_cfg["length_norm_alpha"],
        )
        print("\nBeam Search Results:")
        for width, acc in sorted(results.items()):
            print(f"  Beam width {width}: accuracy = {acc:.4f} ({acc*100:.1f}%)")


if __name__ == "__main__":
    main()
