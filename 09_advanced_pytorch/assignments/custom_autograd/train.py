"""Training script for custom autograd experiments.

Trains:
1. Binary activation network (STE) on MNIST - compares HardThreshold vs Clamped STE
2. Regression model with asymmetric loss vs standard MSE on synthetic data

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --experiment ste
    python train.py --config config.yaml --experiment asymmetric
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from shared_utils.common import set_seed, get_device, save_checkpoint, TrainingLogger

from data import load_mnist_flat, generate_regression_data
from model import BinaryActivationNetwork, SimpleRegressor, AsymmetricMSELoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_ste_network(config: dict, device: torch.device) -> dict:
    """Train binary activation network on MNIST with STE.

    Trains with both HardThresholdSTE and ClampedSTE for comparison.

    Args:
        config: Configuration dictionary.
        device: Device to train on.

    Returns:
        Dictionary with training results for each STE variant.
    """
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    train_loader, val_loader, test_loader = load_mnist_flat(
        batch_size=data_cfg["batch_size"],
        val_split=data_cfg["val_split"],
        num_workers=data_cfg["num_workers"],
    )

    results = {}
    for variant in ["hard_threshold", "clamped"]:
        logger.info(f"Training with {variant} STE...")

        set_seed(42)
        model = BinaryActivationNetwork(
            input_size=model_cfg["input_size"],
            hidden_sizes=model_cfg["hidden_sizes"],
            num_classes=model_cfg["num_classes"],
            ste_variant=variant,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()
        training_logger = TrainingLogger()

        for epoch in range(1, train_cfg["num_epochs"] + 1):
            # --- Training ---
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                # YOUR CODE HERE
                # 1. Forward pass: output = model(data)
                # 2. Compute loss: loss = criterion(output, target)
                # 3. Backward pass: loss.backward()
                # 4. Optimizer step: optimizer.step()
                raise NotImplementedError("Implement STE training step")

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

            train_loss = total_loss / total
            train_acc = correct / total

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item() * data.size(0)
                    val_correct += output.argmax(dim=1).eq(target).sum().item()
                    val_total += data.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            training_logger.log(
                train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss, val_acc=val_acc,
            )
            logger.info(
                f"  [{variant}] Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            )

        # --- Test ---
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                test_correct += model(data).argmax(dim=1).eq(target).sum().item()
                test_total += data.size(0)

        test_acc = test_correct / test_total
        logger.info(f"  [{variant}] Test Accuracy: {test_acc:.4f}")

        results[variant] = {
            "test_acc": test_acc,
            "history": training_logger.history,
        }

        # Save checkpoint
        save_dir = Path(config["logging"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model, optimizer, train_cfg["num_epochs"], train_loss,
            str(save_dir / f"ste_{variant}_final.pt"),
            test_acc=test_acc,
        )

    return results


def train_asymmetric_loss(config: dict, device: torch.device) -> dict:
    """Train regression models comparing standard MSE vs asymmetric MSE.

    Args:
        config: Configuration dictionary.
        device: Device to train on.

    Returns:
        Dictionary with training results for each loss variant.
    """
    loss_cfg = config["asymmetric_loss"]
    train_cfg = config["training"]

    train_loader, test_loader = generate_regression_data()

    results = {}
    for loss_name in ["mse", "asymmetric"]:
        logger.info(f"Training regressor with {loss_name} loss...")

        set_seed(42)
        model = SimpleRegressor(input_dim=1, hidden_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

        if loss_name == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = AsymmetricMSELoss(
                alpha=loss_cfg["alpha"],
                beta=loss_cfg["beta"],
            )

        training_logger = TrainingLogger()

        for epoch in range(1, 50 + 1):  # More epochs for regression
            model.train()
            total_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()

                # YOUR CODE HERE
                # 1. Forward pass: y_pred = model(x_batch)
                # 2. Compute loss: loss = criterion(y_pred, y_batch)
                # 3. Backward pass: loss.backward()
                # 4. Optimizer step: optimizer.step()
                raise NotImplementedError("Implement asymmetric loss training step")

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            training_logger.log(train_loss=avg_loss)

            if epoch % 10 == 0:
                logger.info(f"  [{loss_name}] Epoch {epoch}: Loss={avg_loss:.4f}")

        # Evaluate
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                preds = model(x_batch).cpu()
                all_preds.append(preds)
                all_true.append(y_batch)

        preds = torch.cat(all_preds)
        true = torch.cat(all_true)
        bias = (preds - true).mean().item()
        logger.info(f"  [{loss_name}] Mean prediction bias: {bias:+.4f}")

        results[loss_name] = {
            "predictions": preds,
            "true_values": true,
            "bias": bias,
            "history": training_logger.history,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Custom Autograd Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "ste", "asymmetric"],
        help="Which experiment to run",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    logger.info(f"Using device: {device}")

    if args.experiment in ("all", "ste"):
        logger.info("=" * 60)
        logger.info("Experiment: Binary Activation Network with STE")
        logger.info("=" * 60)
        ste_results = train_ste_network(config, device)

    if args.experiment in ("all", "asymmetric"):
        logger.info("=" * 60)
        logger.info("Experiment: Asymmetric MSE Loss vs Standard MSE")
        logger.info("=" * 60)
        asym_results = train_asymmetric_loss(config, device)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
