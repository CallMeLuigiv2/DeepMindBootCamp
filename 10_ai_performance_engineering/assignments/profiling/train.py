"""Training script with profiling integration.

Modes:
- baseline: Deliberately slow pipeline with all anti-patterns
- profile: Run torch.profiler on the baseline and generate Chrome traces
- optimized: Fully optimized pipeline

Usage:
    python train.py --config config.yaml --mode baseline
    python train.py --config config.yaml --mode profile
    python train.py --config config.yaml --mode optimized
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import yaml
from tqdm import tqdm

from shared_utils.common import set_seed, get_device, TrainingLogger

from model import ResNet18Inefficient, ResNet18Standard
from data import load_cifar10_profiling
from utils import PhaseTimer, setup_profiler, measure_throughput, get_gpu_memory_stats, reset_memory_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_baseline(config: dict, device: torch.device) -> None:
    """Train with all anti-patterns active (deliberately slow)."""
    set_seed(42)
    bl_cfg = config["baseline"]

    model = ResNet18Inefficient(num_classes=config["model"]["num_classes"]).to(device)
    train_loader, _ = load_cifar10_profiling(
        batch_size=bl_cfg["batch_size"],
        num_workers=bl_cfg["num_workers"],
        pin_memory=bl_cfg["pin_memory"],
        heavy_augmentation=bl_cfg["heavy_augmentation"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    phase_timer = PhaseTimer(use_cuda=device.type == "cuda")

    num_steps = config["profiling"]["num_steps"]
    data_iter = iter(train_loader)

    model.train()
    reset_memory_stats()

    for step in range(num_steps):
        # Anti-pattern: data loading in main process
        phase_timer.start("data_loading")
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data, target = next(data_iter)
        phase_timer.stop("data_loading")

        # Anti-pattern: synchronous transfer, no pin_memory
        phase_timer.start("data_transfer")
        data, target = data.to(device), target.to(device)
        phase_timer.stop("data_transfer")

        # Anti-pattern: unnecessary clone
        if bl_cfg["clone_inputs"]:
            data = data.clone()

        phase_timer.start("forward")
        output = model(data)
        phase_timer.stop("forward")

        phase_timer.start("loss_sync")
        loss = criterion(output, target)
        # Anti-pattern: sync every step with loss.item()
        if bl_cfg["sync_every_step"]:
            loss_val = loss.item()
            if step % 1 == 0:
                pass  # print forces sync
        phase_timer.stop("loss_sync")

        phase_timer.start("backward")
        optimizer.zero_grad()
        loss.backward()
        phase_timer.stop("backward")

        phase_timer.start("optimizer_step")
        optimizer.step()
        phase_timer.stop("optimizer_step")

        # Anti-pattern: convert to numpy for "logging"
        if bl_cfg["log_tensors"] and step % 10 == 0:
            _ = output.detach().cpu().numpy()

    logger.info("Baseline Phase Timing:")
    phase_timer.report(skip_first_n=config["profiling"]["warmup_steps"])
    mem = get_gpu_memory_stats()
    logger.info(f"Peak GPU memory: {mem['peak_mb']:.1f} MB")


def train_profiled(config: dict, device: torch.device) -> None:
    """Run torch.profiler on the baseline pipeline."""
    set_seed(42)
    bl_cfg = config["baseline"]

    model = ResNet18Inefficient(num_classes=config["model"]["num_classes"]).to(device)
    train_loader, _ = load_cifar10_profiling(
        batch_size=bl_cfg["batch_size"],
        num_workers=bl_cfg["num_workers"],
        pin_memory=bl_cfg["pin_memory"],
        heavy_augmentation=bl_cfg["heavy_augmentation"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    profiler = setup_profiler(trace_dir=config["profiling"]["trace_dir"])
    data_iter = iter(train_loader)

    model.train()
    with profiler:
        for step in range(config["profiling"]["profile_steps"] + 5):
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                data, target = next(data_iter)

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            profiler.step()

    logger.info(f"Chrome trace saved to {config['profiling']['trace_dir']}/")
    logger.info("Open in chrome://tracing to visualize.")


def train_optimized(config: dict, device: torch.device) -> None:
    """Train with all optimizations applied."""
    set_seed(42)
    opt_cfg = config["optimized"]

    # YOUR CODE HERE
    # Build the optimized pipeline:
    # 1. Use ResNet18Standard (efficient model)
    # 2. Light augmentation, optimal num_workers, pin_memory, persistent_workers
    # 3. Non-blocking data transfer
    # 4. No unnecessary syncs (only log loss every N steps using detach())
    # 5. Larger batch size
    # 6. Optional: mixed precision with autocast + GradScaler
    # 7. Optional: torch.compile
    # 8. Measure throughput and compare to baseline
    raise NotImplementedError("Implement optimized training pipeline")


def main():
    parser = argparse.ArgumentParser(description="Profiling Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "profile", "optimized"])
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Mode: {args.mode}")

    if args.mode == "baseline":
        train_baseline(config, device)
    elif args.mode == "profile":
        train_profiled(config, device)
    elif args.mode == "optimized":
        train_optimized(config, device)


if __name__ == "__main__":
    main()
