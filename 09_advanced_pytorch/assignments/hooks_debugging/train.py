"""Training script for hooks and debugging experiments.

Runs:
1. Gradient flow experiment: deep sigmoid vs deep relu vs batchnorm vs residual
2. DANN domain adaptation: baseline classifier vs DANN
3. Pruning: train, prune at various ratios, fine-tune

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --experiment gradient_flow
    python train.py --config config.yaml --experiment dann
    python train.py --config config.yaml --experiment pruning
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from shared_utils.common import set_seed, get_device, save_checkpoint, TrainingLogger

from data import load_mnist_flat, load_dann_data
from model import DeepNetwork, DANN, SimpleMNISTNet
from utils import GradientFlowVisualizer, ActivationMonitor, MagnitudePruner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_gradient_flow(config: dict, device: torch.device) -> dict:
    """Train deep networks with different configurations and visualize gradient flow.

    Compares: (a) deep sigmoid, (b) deep sigmoid + BN, (c) deep ReLU, (d) deep ReLU + residual.

    Args:
        config: Configuration dictionary.
        device: Device to train on.

    Returns:
        Dictionary with gradient flow stats for each configuration.
    """
    gf_cfg = config["gradient_flow"]
    train_loader, val_loader, _ = load_mnist_flat(batch_size=config["data"]["batch_size"])

    configs = [
        {"activation": "sigmoid", "use_batchnorm": False, "use_residual": False, "name": "sigmoid"},
        {"activation": "sigmoid", "use_batchnorm": True,  "use_residual": False, "name": "sigmoid+BN"},
        {"activation": "relu",    "use_batchnorm": False, "use_residual": False, "name": "relu"},
        {"activation": "relu",    "use_batchnorm": False, "use_residual": True,  "name": "relu+residual"},
    ]

    results = {}
    for cfg in configs:
        logger.info(f"Training: {cfg['name']}")
        set_seed(42)

        model = DeepNetwork(
            input_size=784,
            hidden_size=gf_cfg["hidden_size"],
            num_layers=gf_cfg["num_layers"],
            num_classes=10,
            activation=cfg["activation"],
            use_batchnorm=cfg["use_batchnorm"],
            use_residual=cfg["use_residual"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        viz = GradientFlowVisualizer(model)

        # Train a few steps to get gradient flow stats
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 5:
                break
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        results[cfg["name"]] = dict(viz.grad_stats)
        save_dir = Path(config["logging"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        viz.plot(title=f"Gradient Flow: {cfg['name']}", save_path=str(save_dir / f"grad_flow_{cfg['name']}.png"))
        viz.close()

    return results


def train_dann(config: dict, device: torch.device) -> dict:
    """Train DANN for domain adaptation: noisy MNIST -> clean MNIST.

    Args:
        config: Configuration dictionary.
        device: Device to train on.

    Returns:
        Dictionary with accuracy results for baseline and DANN.
    """
    dann_cfg = config["dann"]
    train_cfg = config["training"]

    source_loader, target_train_loader, target_test_loader = load_dann_data(
        batch_size=config["data"]["batch_size"],
        noise_std=config["data"]["noise_std"],
    )

    # --- DANN training ---
    logger.info("Training DANN...")
    set_seed(42)

    model = DANN(
        input_size=784,
        feature_dim=dann_cfg["feature_dim"],
        hidden_dim=dann_cfg["hidden_dim"],
        num_classes=dann_cfg["num_classes"],
        lambda_val=dann_cfg["lambda_val"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    task_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        model.train()
        total_task_loss = 0.0
        total_domain_loss = 0.0
        n_batches = 0

        target_iter = iter(target_train_loader)

        for source_batch in tqdm(source_loader, desc=f"DANN Epoch {epoch}", leave=False):
            s_data, s_labels, s_domain = [t.to(device) for t in source_batch]

            try:
                t_data, _, t_domain = next(target_iter)
            except StopIteration:
                target_iter = iter(target_train_loader)
                t_data, _, t_domain = next(target_iter)
            t_data, t_domain = t_data.to(device), t_domain.to(device)

            optimizer.zero_grad()

            # YOUR CODE HERE
            # 1. Forward source through DANN: s_task_logits, s_domain_logits = model(s_data)
            # 2. Forward target through DANN: _, t_domain_logits = model(t_data)
            # 3. Task loss on source: task_loss = task_criterion(s_task_logits, s_labels)
            # 4. Domain loss: domain_loss = domain_criterion(s_domain_logits, s_domain) + domain_criterion(t_domain_logits, t_domain)
            # 5. Total loss = task_loss + domain_loss
            # 6. Backward and step
            raise NotImplementedError("Implement DANN training step")

            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss.item()
            n_batches += 1

        logger.info(
            f"  Epoch {epoch}: Task Loss={total_task_loss/n_batches:.4f}, "
            f"Domain Loss={total_domain_loss/n_batches:.4f}"
        )

    # Evaluate on target test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for t_data, t_labels, _ in target_test_loader:
            t_data, t_labels = t_data.to(device), t_labels.to(device)
            task_logits, _ = model(t_data)
            correct += task_logits.argmax(1).eq(t_labels).sum().item()
            total += t_labels.size(0)

    dann_acc = correct / total
    logger.info(f"DANN target accuracy: {dann_acc:.4f}")

    return {"dann_target_acc": dann_acc}


def train_pruning(config: dict, device: torch.device) -> dict:
    """Train a model, then prune at various ratios and measure accuracy.

    Args:
        config: Configuration dictionary.
        device: Device to train on.

    Returns:
        Dictionary with pruning ratio -> accuracy mapping.
    """
    train_cfg = config["training"]
    prune_cfg = config["pruning"]

    train_loader, val_loader, test_loader = load_mnist_flat(batch_size=config["data"]["batch_size"])

    # Train the base model
    logger.info("Training base model for pruning experiments...")
    set_seed(42)
    model = SimpleMNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        model.train()
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

    # Test unpruned accuracy
    def evaluate_model(model_to_eval):
        model_to_eval.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                correct += model_to_eval(data).argmax(1).eq(target).sum().item()
                total += target.size(0)
        return correct / total

    base_acc = evaluate_model(model)
    logger.info(f"Unpruned test accuracy: {base_acc:.4f}")

    # Prune at various ratios
    results = {"unpruned": base_acc}
    for ratio in prune_cfg["prune_ratios"]:
        import copy
        pruned_model = copy.deepcopy(model)
        pruner = MagnitudePruner(pruned_model, prune_ratio=ratio)
        acc = evaluate_model(pruned_model)
        logger.info(f"  Prune {ratio:.0%}: accuracy = {acc:.4f}")
        results[f"prune_{ratio}"] = acc
        pruner.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Hooks and Debugging Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "gradient_flow", "dann", "pruning"],
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    logger.info(f"Using device: {device}")

    if args.experiment in ("all", "gradient_flow"):
        logger.info("=" * 60)
        logger.info("Experiment: Gradient Flow Visualization")
        logger.info("=" * 60)
        train_gradient_flow(config, device)

    if args.experiment in ("all", "dann"):
        logger.info("=" * 60)
        logger.info("Experiment: DANN Domain Adaptation")
        logger.info("=" * 60)
        train_dann(config, device)

    if args.experiment in ("all", "pruning"):
        logger.info("=" * 60)
        logger.info("Experiment: Magnitude Pruning")
        logger.info("=" * 60)
        train_pruning(config, device)


if __name__ == "__main__":
    main()
