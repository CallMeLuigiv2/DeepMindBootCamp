"""Ablation study helpers, hypothesis tracking, and statistical comparison.

All functions in this file are fully implemented (no stubs).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


class HypothesisTracker:
    """Track and document your research hypothesis.

    Usage:
        hyp = HypothesisTracker(
            observation="ResNet-20 shows higher training loss variance across seeds ...",
            proposed_change="Add stochastic depth to regularize training ...",
            expected_effect="Reduced variance in training loss, slightly better test accuracy ...",
            test_method="Train baseline vs stochastic depth, 3 seeds each, compare mean+std ...",
            success_criterion="Stochastic depth achieves higher mean accuracy OR lower std ...",
        )
        hyp.save("hypothesis.md")
    """

    def __init__(
        self,
        observation: str,
        proposed_change: str,
        expected_effect: str,
        test_method: str,
        success_criterion: str,
    ):
        self.observation = observation
        self.proposed_change = proposed_change
        self.expected_effect = expected_effect
        self.test_method = test_method
        self.success_criterion = success_criterion
        self.result: Optional[str] = None
        self.analysis: Optional[str] = None

    def record_result(self, result: str, analysis: str):
        """Record the experimental result and analysis."""
        self.result = result
        self.analysis = analysis

    def save(self, path: str):
        """Save hypothesis as markdown."""
        lines = [
            "# HYPOTHESIS",
            "",
            f"**Observation:** {self.observation}",
            "",
            f"**Proposed change:** {self.proposed_change}",
            "",
            f"**Expected effect:** {self.expected_effect}",
            "",
            f"**How to test:** {self.test_method}",
            "",
            f"**Success criterion:** {self.success_criterion}",
        ]
        if self.result:
            lines.extend([
                "",
                "---",
                "",
                "# RESULT",
                "",
                f"**Result:** {self.result}",
                "",
                f"**Analysis:** {self.analysis}",
            ])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))


def statistical_comparison(
    baseline_scores: list[float],
    improved_scores: list[float],
    metric_name: str = "accuracy",
    alpha: float = 0.05,
) -> dict:
    """Run statistical tests comparing baseline and improved results.

    Performs:
    - Welch's t-test (does not assume equal variances)
    - Effect size (Cohen's d)

    Args:
        baseline_scores: Metric values from baseline runs (one per seed).
        improved_scores: Metric values from improved runs (one per seed).
        metric_name: Name of the metric for reporting.
        alpha: Significance level.

    Returns:
        Dictionary with test results.
    """
    baseline = np.array(baseline_scores)
    improved = np.array(improved_scores)

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(improved, baseline, equal_var=False)

    # Cohen's d (effect size)
    pooled_std = np.sqrt((baseline.std() ** 2 + improved.std() ** 2) / 2)
    cohens_d = (improved.mean() - baseline.mean()) / pooled_std if pooled_std > 0 else 0

    significant = p_value < alpha
    direction = "improved" if improved.mean() > baseline.mean() else "baseline"

    result = {
        "metric": metric_name,
        "baseline_mean": float(baseline.mean()),
        "baseline_std": float(baseline.std()),
        "improved_mean": float(improved.mean()),
        "improved_std": float(improved.std()),
        "difference": float(improved.mean() - baseline.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": significant,
        "better": direction,
        "n_baseline": len(baseline_scores),
        "n_improved": len(improved_scores),
    }

    return result


def print_comparison_report(result: dict):
    """Print a formatted comparison report."""
    print(f"\n{'='*60}")
    print(f"Statistical Comparison: {result['metric']}")
    print(f"{'='*60}")
    print(f"  Baseline:  {result['baseline_mean']:.4f} +/- {result['baseline_std']:.4f} (n={result['n_baseline']})")
    print(f"  Improved:  {result['improved_mean']:.4f} +/- {result['improved_std']:.4f} (n={result['n_improved']})")
    print(f"  Difference: {result['difference']:+.4f}")
    print(f"  t-statistic: {result['t_statistic']:.3f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Cohen's d: {result['cohens_d']:.3f}")

    if abs(result['cohens_d']) < 0.2:
        effect_label = "negligible"
    elif abs(result['cohens_d']) < 0.5:
        effect_label = "small"
    elif abs(result['cohens_d']) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"

    print(f"  Effect size: {effect_label}")
    print(f"  Significant at alpha=0.05: {'YES' if result['significant'] else 'NO'}")
    print(f"  Better variant: {result['better']}")


class AblationStudy:
    """Track and present ablation study results.

    Usage:
        ablation = AblationStudy("ResNet Improvement Ablation")
        ablation.add_result("Baseline (no changes)", accuracy=0.9125)
        ablation.add_result("+ Stochastic depth", accuracy=0.9189)
        ablation.add_result("+ SE blocks", accuracy=0.9201)
        ablation.add_result("+ Both", accuracy=0.9215)
        ablation.report()
    """

    def __init__(self, name: str):
        self.name = name
        self.results: list[dict] = []

    def add_result(self, config_name: str, **metrics):
        self.results.append({"config": config_name, **metrics})

    def report(self):
        """Print formatted ablation table."""
        if not self.results:
            print("No ablation results recorded.")
            return

        metrics = [k for k in self.results[0] if k != "config"]

        print(f"\n{'='*60}")
        print(f"Ablation Study: {self.name}")
        print(f"{'='*60}")

        header = f"{'Configuration':<35}" + "".join(f"{m:>12}" for m in metrics)
        print(header)
        print("-" * len(header))

        for result in self.results:
            row = f"{result['config']:<35}"
            for metric in metrics:
                val = result.get(metric, 0)
                row += f"{val:>12.4f}"
            print(row)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"name": self.name, "results": self.results}, f, indent=2)
