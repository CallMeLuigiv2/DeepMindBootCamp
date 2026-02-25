"""Training script matching the paper's methodology.

Supports multiple random seeds for reproducibility verification.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --seed 1
    python train.py --config config.yaml --all-seeds
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from shared_utils.common import set_seed, get_device, save_checkpoint, TrainingLogger

from model import ResNetCIFAR, PlainNetCIFAR
from data import load_paper_dataset
from utils import ExperimentLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_one_run(
    config: dict,
    device: torch.device,
    seed: int,
    experiment_logger: ExperimentLogger,
) -> float:
    """Train a single run with a given seed.

    Args:
        config: Configuration dictionary.
        device: Device to train on.
        seed: Random seed.
        experiment_logger: Logger for tracking runs.

    Returns:
        Test accuracy.
    """
    set_seed(seed)
    train_cfg = config["training"]
    model_cfg = config["model"]

    # YOUR CODE HERE
    # 1. Create model based on config (ResNetCIFAR or PlainNetCIFAR)
    # 2. Load dataset matching the paper's setup
    # 3. Create optimizer matching the paper (SGD with momentum for ResNet)
    # 4. Create LR scheduler matching the paper
    # 5. Training loop:
    #    - Train for num_epochs
    #    - Log train_loss, val_loss, val_acc each epoch
    #    - Save best model checkpoint
    # 6. Evaluate on test set
    # 7. Return test accuracy
    raise NotImplementedError("Implement training run matching paper methodology")


def main():
    parser = argparse.ArgumentParser(description="Paper Reproduction Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-seeds", action="store_true", help="Run with all configured seeds")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    logger.info(f"Using device: {device}")

    experiment_logger = ExperimentLogger(f"{config['paper']}_reproduction")

    if args.all_seeds:
        seeds = config["experiment"]["seeds"]
    else:
        seeds = [args.seed]

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Run with seed={seed}")
        logger.info(f"{'='*60}")
        test_acc = train_one_run(config, device, seed, experiment_logger)
        logger.info(f"Seed {seed}: Test accuracy = {test_acc:.4f}")

    experiment_logger.summary()

    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    experiment_logger.save(str(save_dir / "experiment_results.json"))


if __name__ == "__main__":
    main()
