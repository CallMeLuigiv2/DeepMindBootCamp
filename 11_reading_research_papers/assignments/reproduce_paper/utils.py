"""Experiment logging, result comparison, and decision log helpers.

All functions in this file are fully implemented (no stubs).
"""

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class ExperimentLogger:
    """Track experiments across multiple runs with different seeds.

    Usage:
        logger = ExperimentLogger("resnet_reproduction")
        for seed in [42, 1, 2]:
            logger.start_run(seed=seed)
            for epoch in range(num_epochs):
                logger.log_epoch(train_loss=..., val_loss=..., val_acc=...)
            logger.end_run(test_acc=...)
        logger.summary()
        logger.save("results/experiment.json")
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.runs: list[dict] = []
        self._current_run: Optional[dict] = None

    def start_run(self, seed: int, **metadata):
        self._current_run = {
            "seed": seed,
            "metadata": metadata,
            "start_time": time.time(),
            "epochs": [],
            "final_metrics": {},
        }

    def log_epoch(self, **metrics):
        if self._current_run is not None:
            self._current_run["epochs"].append(metrics)

    def end_run(self, **final_metrics):
        if self._current_run is not None:
            self._current_run["end_time"] = time.time()
            self._current_run["duration_s"] = (
                self._current_run["end_time"] - self._current_run["start_time"]
            )
            self._current_run["final_metrics"] = final_metrics
            self.runs.append(self._current_run)
            self._current_run = None

    def summary(self):
        """Print a summary of all runs with mean and std across seeds."""
        if not self.runs:
            print("No completed runs.")
            return

        print(f"\nExperiment: {self.experiment_name}")
        print(f"Number of runs: {len(self.runs)}")
        print(f"Seeds: {[r['seed'] for r in self.runs]}")

        # Aggregate final metrics
        all_metrics = {}
        for run in self.runs:
            for key, val in run["final_metrics"].items():
                all_metrics.setdefault(key, []).append(val)

        print("\nFinal Metrics (mean +/- std):")
        for key, values in all_metrics.items():
            arr = np.array(values)
            print(f"  {key}: {arr.mean():.4f} +/- {arr.std():.4f}")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "runs": self.runs,
            }, f, indent=2, default=str)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.experiment_name = data["experiment_name"]
        self.runs = data["runs"]


class DecisionLog:
    """Track implementation decisions where the paper was ambiguous.

    Usage:
        log = DecisionLog()
        log.add(
            title="Shortcut connection type",
            paper_says="'We consider three options: (A) zero-padding, (B) projection, (C) all projections'",
            ambiguity="Paper uses option B for downsampling and A for others, but does not specify this clearly for CIFAR-10",
            decision="Use option A (identity) for same-dimension shortcuts and option B (1x1 conv) for dimension changes",
            justification="This matches the CIFAR-10 experiments described in Section 4.2",
            impact="Moderate -- projection shortcuts add parameters but may improve accuracy slightly",
        )
        log.save("decision_log.md")
    """

    def __init__(self):
        self.decisions: list[dict] = []

    def add(
        self,
        title: str,
        paper_says: str,
        ambiguity: str,
        decision: str,
        justification: str,
        impact: str = "Unknown",
    ):
        self.decisions.append({
            "title": title,
            "paper_says": paper_says,
            "ambiguity": ambiguity,
            "decision": decision,
            "justification": justification,
            "impact": impact,
        })

    def save(self, path: str):
        """Save decision log as markdown."""
        lines = ["# Implementation Decision Log\n"]
        for i, d in enumerate(self.decisions, 1):
            lines.append(f"## Decision {i}: {d['title']}\n")
            lines.append(f"- **What the paper says:** {d['paper_says']}")
            lines.append(f"- **What is ambiguous:** {d['ambiguity']}")
            lines.append(f"- **What I decided:** {d['decision']}")
            lines.append(f"- **Why:** {d['justification']}")
            lines.append(f"- **Impact:** {d['impact']}")
            lines.append("")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def __len__(self) -> int:
        return len(self.decisions)


def compare_results(
    our_results: dict[str, float],
    paper_results: dict[str, float],
    metric_name: str = "Metric",
) -> str:
    """Create a comparison table between our results and the paper's.

    Args:
        our_results: Dictionary mapping experiment names to our metric values.
        paper_results: Dictionary mapping experiment names to the paper's values.
        metric_name: Name of the metric being compared.

    Returns:
        Formatted comparison table as string.
    """
    all_experiments = sorted(set(our_results.keys()) | set(paper_results.keys()))

    header = f"{'Experiment':<30} {'Paper':>12} {'Ours':>12} {'Gap':>12}"
    separator = "-" * len(header)
    lines = [header, separator]

    for exp in all_experiments:
        paper_val = paper_results.get(exp, None)
        our_val = our_results.get(exp, None)

        paper_str = f"{paper_val:.4f}" if paper_val is not None else "N/A"
        our_str = f"{our_val:.4f}" if our_val is not None else "N/A"

        if paper_val is not None and our_val is not None:
            gap = our_val - paper_val
            gap_str = f"{gap:+.4f}"
        else:
            gap_str = "N/A"

        lines.append(f"{exp:<30} {paper_str:>12} {our_str:>12} {gap_str:>12}")

    return "\n".join(lines)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count parameters per module for comparison with paper.

    Returns:
        Dictionary mapping module names to parameter counts.
    """
    result = {}
    total = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        result[name] = params
        total += params
    result["TOTAL"] = total
    return result
