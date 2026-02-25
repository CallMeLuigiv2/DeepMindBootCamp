"""A/B experiment framework: train baseline vs improvement.

Usage:
    python train.py --config config.yaml --variant baseline
    python train.py --config config.yaml --variant improved
    python train.py --config config.yaml --ab-comparison
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from shared_utils.common import set_seed, get_device, save_checkpoint, TrainingLogger

from model import PaperModel
from data import load_experiment_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_variant(
    config: dict,
    device: torch.device,
    seed: int,
    use_improvement: bool,
) -> dict:
    """Train a single variant (baseline or improved) with a given seed.

    Args:
        config: Configuration dictionary.
        device: Device to train on.
        seed: Random seed.
        use_improvement: Whether to use the improved variant.

    Returns:
        Dictionary with training results.
    """
    set_seed(seed)
    model_cfg = config["model"]
    train_cfg = config["training"]
    variant_name = "improved" if use_improvement else "baseline"

    model = PaperModel(
        num_layers=model_cfg["num_layers"],
        num_classes=model_cfg["num_classes"],
        use_improvement=use_improvement,
        improvement_config=config.get("improvement", {}),
    ).to(device)

    logger.info(f"[{variant_name}] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, val_loader, test_loader = load_experiment_data(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        augment=config["data"]["augment"],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=train_cfg["lr_milestones"], gamma=train_cfg["lr_gamma"],
    )
    criterion = nn.CrossEntropyLoss()
    training_logger = TrainingLogger()

    best_val_acc = 0.0

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        # --- Training ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # YOUR CODE HERE
            # Standard training step:
            # 1. optimizer.zero_grad()
            # 2. output = model(data)
            # 3. loss = criterion(output, target)
            # 4. loss.backward()
            # 5. optimizer.step()
            raise NotImplementedError("Implement training step")

            total_loss += loss.item() * data.size(0)
            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                val_correct += model(data).argmax(1).eq(target).sum().item()
                val_total += data.size(0)

        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        training_logger.log(
            train_loss=total_loss / total, train_acc=correct / total, val_acc=val_acc,
        )

        if epoch % 50 == 0 or epoch == train_cfg["num_epochs"]:
            logger.info(
                f"  [{variant_name} seed={seed}] Epoch {epoch}: "
                f"loss={total_loss/total:.4f}, acc={correct/total:.4f}, val_acc={val_acc:.4f}"
            )

    # --- Test ---
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            test_correct += model(data).argmax(1).eq(target).sum().item()
            test_total += data.size(0)

    test_acc = test_correct / test_total
    logger.info(f"  [{variant_name} seed={seed}] Test accuracy: {test_acc:.4f}")

    return {
        "variant": variant_name,
        "seed": seed,
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "history": training_logger.history,
    }


def run_ab_comparison(config: dict, device: torch.device) -> None:
    """Run full A/B comparison: baseline vs improved across multiple seeds."""
    seeds = config["experiment"]["seeds"]

    baseline_results = []
    improved_results = []

    for seed in seeds:
        logger.info(f"\n--- Seed {seed}: Baseline ---")
        bl = train_variant(config, device, seed, use_improvement=False)
        baseline_results.append(bl)

        logger.info(f"\n--- Seed {seed}: Improved ---")
        imp = train_variant(config, device, seed, use_improvement=True)
        improved_results.append(imp)

    # Summary
    bl_accs = [r["test_acc"] for r in baseline_results]
    imp_accs = [r["test_acc"] for r in improved_results]

    import numpy as np
    logger.info("\n" + "=" * 60)
    logger.info("A/B COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"Baseline: {np.mean(bl_accs):.4f} +/- {np.std(bl_accs):.4f}")
    logger.info(f"Improved: {np.mean(imp_accs):.4f} +/- {np.std(imp_accs):.4f}")
    logger.info(f"Delta:    {np.mean(imp_accs) - np.mean(bl_accs):+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Paper Improvement Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "improved"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ab-comparison", action="store_true", help="Run full A/B comparison")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    logger.info(f"Using device: {device}")

    if args.ab_comparison:
        run_ab_comparison(config, device)
    else:
        use_improvement = args.variant == "improved"
        train_variant(config, device, args.seed, use_improvement)


if __name__ == "__main__":
    main()
