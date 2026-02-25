"""
RLHF Evaluation Script
========================

Evaluates and compares base, RLHF-tuned, and DPO-tuned language models.
Supports qualitative generation comparison and quantitative metrics.

Usage:
    python evaluate.py --compare
    python evaluate.py --model checkpoints/ppo_model.pt --generate
    python evaluate.py --kl_frontier
"""

import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from model import RewardModel
from data import TEST_PROMPTS
from utils import (
    get_device,
    generate_text,
    compute_log_probs,
    compute_perplexity,
    compute_distinct_ngrams,
)


# ---------------------------------------------------------------------------
# Argument parsing (pre-written)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLHF Evaluation")
    parser.add_argument("--compare", action="store_true", help="Compare all models side-by-side")
    parser.add_argument("--generate", action="store_true", help="Generate samples from a model")
    parser.add_argument("--kl_frontier", action="store_true", help="Plot KL-reward frontier")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--n_samples", type=int, default=3, help="Samples per prompt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading (pre-written)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, model_name: str, device: torch.device) -> GPT2LMHeadModel:
    """Load a fine-tuned GPT-2 model from checkpoint.

    Args:
        checkpoint_path: Path to the saved model state dict.
        model_name: Base model name for architecture.
        device: Torch device.

    Returns:
        Model loaded in eval mode.
    """
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation functions (stubbed)
# ---------------------------------------------------------------------------

def compare_models(
    checkpoint_dir: str,
    model_name: str,
    device: torch.device,
    n_samples: int = 3,
) -> None:
    """Compare base, RLHF, and DPO models side-by-side.

    For each test prompt, generates responses from all available models
    and presents them for qualitative comparison.

    Args:
        checkpoint_dir: Directory containing model checkpoints.
        model_name: Base model name.
        device: Torch device.
        n_samples: Number of responses per prompt per model.
    """
    # YOUR CODE HERE
    # 1. Load base model, RLHF model, and DPO model (if available)
    # 2. Load tokenizer
    # 3. For each test prompt:
    #    a. Generate n_samples responses from each model
    #    b. Print responses side-by-side
    # 4. Optionally score with reward model
    raise NotImplementedError("Implement compare_models")


def compute_quantitative_metrics(
    checkpoint_dir: str,
    model_name: str,
    device: torch.device,
    n_generations: int = 100,
) -> dict:
    """Compute quantitative metrics for all models.

    Metrics:
    - Mean reward model score
    - Mean response length (tokens)
    - Preference function agreement rate
    - Perplexity under the base model
    - Distinct n-gram diversity

    Args:
        checkpoint_dir: Directory containing model checkpoints.
        model_name: Base model name.
        device: Torch device.
        n_generations: Number of generations for evaluation.

    Returns:
        Dictionary mapping model name to metrics dict.
    """
    # YOUR CODE HERE
    # For each model:
    #   1. Generate n_generations responses
    #   2. Compute reward model scores
    #   3. Compute response lengths
    #   4. Check preference function agreement
    #   5. Compute perplexity under base model
    #   6. Compute distinct n-gram diversity
    # Return results as nested dict
    raise NotImplementedError("Implement compute_quantitative_metrics")


def plot_kl_reward_frontier(
    checkpoint_dir: str,
    model_name: str,
    device: torch.device,
) -> None:
    """Plot the KL-Reward frontier from KL sweep results.

    For each KL coefficient, loads the corresponding trained model and
    plots (KL divergence, reward) to show the fundamental tradeoff.

    Args:
        checkpoint_dir: Directory containing sweep results.
        model_name: Base model name.
        device: Torch device.
    """
    # YOUR CODE HERE
    # 1. Load models trained with different KL coefficients
    # 2. For each model, compute final KL divergence and final reward
    # 3. Plot KL vs reward with connected points
    # 4. Save plot
    raise NotImplementedError("Implement plot_kl_reward_frontier")


def investigate_reward_hacking(
    checkpoint_dir: str,
    model_name: str,
    device: torch.device,
) -> None:
    """Investigate potential reward hacking with low KL penalty.

    Examines the highest-reward generations from the low-beta model
    and checks for degenerate patterns.

    Args:
        checkpoint_dir: Directory containing model checkpoints.
        model_name: Base model name.
        device: Torch device.
    """
    # YOUR CODE HERE
    # 1. Load the model trained with lowest beta (e.g., 0.01)
    # 2. Generate many responses
    # 3. Sort by reward model score
    # 4. Examine top responses for degeneracy
    # 5. Compute perplexity of high-reward responses under base model
    # 6. Report findings
    raise NotImplementedError("Implement investigate_reward_hacking")


# ---------------------------------------------------------------------------
# Main (pre-written)
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = get_device()
    model_name = "gpt2"

    print(f"Device: {device}")

    if args.compare:
        compare_models(args.checkpoint_dir, model_name, device, args.n_samples)
    elif args.kl_frontier:
        plot_kl_reward_frontier(args.checkpoint_dir, model_name, device)
    elif args.generate and args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = load_model(args.model, model_name, device)

        print(f"Generating from: {args.model}")
        for prompt in TEST_PROMPTS:
            print(f"\nPrompt: {prompt}")
            for i in range(args.n_samples):
                response = generate_text(model, tokenizer, prompt, device=device)
                print(f"  [{i+1}] {response}")
    else:
        print("Use --compare, --generate (with --model), or --kl_frontier")


if __name__ == "__main__":
    main()
