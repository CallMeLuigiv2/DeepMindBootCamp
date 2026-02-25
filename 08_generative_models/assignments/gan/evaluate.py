"""
GAN - Evaluation Script

Generates samples, computes FID scores, and creates training progression visualizations.

Usage:
    python evaluate.py --checkpoint checkpoints/dcgan_best.pt --compute-fid
    python evaluate.py --checkpoint checkpoints/dcgan_best.pt --generate --num-samples 64
"""

import argparse
import os

import torch
import yaml

from model import DCGANGenerator
from utils import set_seed, save_sample_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def compute_fid_score(generator, real_dataloader, device, num_samples=10000):
    """Compute FID score between real and generated images.

    Uses torchmetrics FrechetInceptionDistance.
    """
    # YOUR CODE HERE
    # 1. Generate num_samples fake images
    # 2. Load num_samples real images
    # 3. Compute FID using torchmetrics or pytorch-fid
    raise NotImplementedError("Implement compute_fid_score")


def create_training_progression(sample_dir, output_path):
    """Compile sample grids from different epochs into a training progression figure."""
    # YOUR CODE HERE
    # Load sample grids from epochs 1, 5, 10, 25, 50
    # Arrange them in a single figure showing improvement over time
    raise NotImplementedError("Implement create_training_progression")


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # TODO: Load generator from checkpoint
    # TODO: Generate samples and/or compute FID

    print("TODO: Load model and run evaluation")


if __name__ == "__main__":
    main()
