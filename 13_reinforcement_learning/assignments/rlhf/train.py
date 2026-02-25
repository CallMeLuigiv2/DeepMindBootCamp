"""
RLHF Training Script
=====================

Three-phase training pipeline:
1. Generate synthetic preference data
2. Train a reward model on preferences
3. Fine-tune a language model with PPO (RLHF) or DPO

Usage:
    python train.py --phase generate_data --n_pairs 1000
    python train.py --phase reward_model --epochs 3
    python train.py --phase ppo --steps 200 --kl_coeff 0.1
    python train.py --phase dpo --epochs 3
    python train.py --phase ppo --kl_sweep 0.01,0.1,1.0
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from model import RewardModel, RLHFPolicy, compute_reward_model_loss, dpo_loss
from data import (
    PreferenceDataset,
    PromptDataset,
    TEST_PROMPTS,
    TRAINING_PROMPTS,
)
from utils import (
    set_seed,
    get_device,
    generate_text,
    compute_log_probs,
    synthetic_preference_brevity,
    synthetic_preference_positive,
    synthetic_preference_helpfulness,
    generate_preference_pairs,
    compute_kl_divergence,
)


# ---------------------------------------------------------------------------
# Configuration loading (pre-written)
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load training configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Argument parsing (pre-written)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLHF Training Pipeline")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["generate_data", "reward_model", "ppo", "dpo"],
        help="Training phase to run",
    )
    parser.add_argument("--n_pairs", type=int, default=None, help="Number of preference pairs to generate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=None, help="Number of PPO training steps")
    parser.add_argument("--kl_coeff", type=float, default=None, help="KL penalty coefficient")
    parser.add_argument("--kl_sweep", type=str, default=None, help="Comma-separated KL coefficients for sweep")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--preference_rule",
        type=str,
        default=None,
        choices=["brevity", "positive_sentiment", "helpfulness"],
        help="Synthetic preference rule",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading (pre-written)
# ---------------------------------------------------------------------------

def load_base_models(model_name: str, device: torch.device):
    """Load pretrained GPT-2 models and tokenizer.

    Args:
        model_name: Hugging Face model name.
        device: Torch device.

    Returns:
        (policy_model, ref_model, tokenizer) tuple.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ref_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    policy_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    return policy_model, ref_model, tokenizer


# ---------------------------------------------------------------------------
# Phase 1: Generate preference data (stubbed)
# ---------------------------------------------------------------------------

def generate_preference_data(config: dict, device: torch.device) -> None:
    """Generate synthetic preference dataset.

    For each prompt, generate multiple response pairs and apply the
    synthetic preference function to determine which is preferred.

    Args:
        config: Training configuration.
        device: Torch device.
    """
    model_name = config.get("model_name", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    # Select preference rule
    rule_name = config.get("preference_rule", "brevity")
    preference_fns = {
        "brevity": synthetic_preference_brevity,
        "positive_sentiment": synthetic_preference_positive,
        "helpfulness": synthetic_preference_helpfulness,
    }
    preference_fn = preference_fns[rule_name]

    n_pairs = config.get("n_pairs_per_prompt", 20)
    prompts = TRAINING_PROMPTS[: config.get("n_prompts", 50)]
    dataset_path = config.get("dataset_path", "data/preferences.json")

    print(f"Generating preference data:")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Pairs per prompt: {n_pairs}")
    print(f"  Preference rule: {rule_name}")

    # YOUR CODE HERE
    # For each prompt:
    #   For each pair:
    #     1. Generate two responses from the model using generate_text()
    #     2. Apply preference_fn(response_1, response_2) to get preference
    #     3. Store as {"prompt": ..., "chosen": ..., "rejected": ...}
    # Save the dataset to JSON using PreferenceDataset.save_to_json()
    # Print statistics: total pairs, avg response length, preference agreement
    raise NotImplementedError("Implement generate_preference_data")


# ---------------------------------------------------------------------------
# Phase 2: Train reward model (stubbed)
# ---------------------------------------------------------------------------

def train_reward_model(config: dict, device: torch.device) -> RewardModel:
    """Train the reward model on preference data.

    Uses Bradley-Terry loss: L = -E[log sigma(R(chosen) - R(rejected))].

    Args:
        config: Training configuration.
        device: Torch device.

    Returns:
        Trained RewardModel.
    """
    model_name = config.get("model_name", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    rm_config = config.get("reward_model", {})
    dataset_path = config.get("dataset_path", "data/preferences.json")

    # Load and split dataset
    dataset = PreferenceDataset.load_from_json(dataset_path, tokenizer, rm_config.get("max_length", 512))
    train_dataset, val_dataset = dataset.train_val_split(config.get("validation_split", 0.2))

    train_loader = DataLoader(train_dataset, batch_size=rm_config.get("batch_size", 8), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=rm_config.get("batch_size", 8))

    # Create reward model
    reward_model = RewardModel(model_name).to(device)
    optimizer = optim.AdamW(reward_model.parameters(), lr=rm_config.get("learning_rate", 1e-5))

    n_epochs = config.get("epochs") or rm_config.get("epochs", 3)

    print(f"Training reward model:")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples: {len(val_dataset)}")
    print(f"  Epochs: {n_epochs}")

    # YOUR CODE HERE
    # For each epoch:
    #   1. Training loop:
    #      - For each batch, compute Bradley-Terry loss using compute_reward_model_loss()
    #      - Backpropagate and update
    #      - Track training loss and accuracy
    #   2. Validation loop:
    #      - Compute loss and accuracy on validation set
    #   3. Print epoch summary (train loss, val loss, val accuracy)
    #   4. Save checkpoint if val accuracy improves
    #
    # After training:
    #   - Score the test prompts and print examples
    #   - Save the final model
    raise NotImplementedError("Implement train_reward_model")

    return reward_model


# ---------------------------------------------------------------------------
# Phase 3a: PPO fine-tuning (stubbed)
# ---------------------------------------------------------------------------

def train_ppo_rlhf(config: dict, device: torch.device) -> None:
    """Fine-tune the language model with PPO against the reward model.

    Three-step process at each iteration:
    1. Generate responses from the current policy
    2. Score responses with reward model and compute KL penalty
    3. Update policy with PPO

    Args:
        config: Training configuration.
        device: Torch device.
    """
    model_name = config.get("model_name", "gpt2")
    policy_model, ref_model, tokenizer = load_base_models(model_name, device)

    # Load trained reward model
    reward_model = RewardModel(model_name).to(device)
    rm_checkpoint = Path(config.get("checkpoint_dir", "checkpoints")) / "reward_model_best.pt"
    reward_model.load_state_dict(torch.load(rm_checkpoint, map_location=device, weights_only=True))
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    ppo_config = config.get("ppo", {})
    kl_coeff = config.get("kl_coeff") or ppo_config.get("kl_coeff", 0.1)
    n_steps = config.get("steps") or ppo_config.get("n_steps", 200)

    rlhf_policy = RLHFPolicy(
        policy_model, ref_model, reward_model, tokenizer,
        kl_coeff=kl_coeff, clip_eps=ppo_config.get("clip_eps", 0.2),
    )
    optimizer = optim.Adam(
        policy_model.parameters(),
        lr=config.get("lr") or ppo_config.get("learning_rate", 1e-6),
    )

    prompts = TRAINING_PROMPTS
    prompt_dataset = PromptDataset(prompts, tokenizer)
    prompt_loader = DataLoader(prompt_dataset, batch_size=ppo_config.get("batch_size", 8), shuffle=True)

    print(f"PPO RLHF Training:")
    print(f"  Steps: {n_steps}")
    print(f"  KL coefficient: {kl_coeff}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")

    # YOUR CODE HERE
    # For each training step:
    #   1. Sample a batch of prompts
    #   2. Generate responses using rlhf_policy.generate_responses()
    #   3. Compute rewards using rlhf_policy.compute_rewards_with_kl()
    #   4. Compute advantages (simple: advantage = reward - mean(reward))
    #   5. Compute old log probs under current policy
    #   6. PPO update using rlhf_policy.ppo_update()
    #
    # Track metrics:
    #   - Mean reward per step
    #   - Mean KL divergence per step
    #   - Reward model score (without KL penalty)
    #   - Sample generations at steps 0, 50, 100, 150, 200
    #
    # Save model and metrics at the end
    raise NotImplementedError("Implement train_ppo_rlhf")


# ---------------------------------------------------------------------------
# Phase 3b: DPO training (stubbed)
# ---------------------------------------------------------------------------

def train_dpo(config: dict, device: torch.device) -> None:
    """Train the language model with Direct Preference Optimization.

    DPO directly optimizes the policy on preference data without a
    separate reward model. Much simpler than PPO-based RLHF.

    Args:
        config: Training configuration.
        device: Torch device.
    """
    model_name = config.get("model_name", "gpt2")
    policy_model, ref_model, tokenizer = load_base_models(model_name, device)

    dpo_config = config.get("dpo", {})
    dataset_path = config.get("dataset_path", "data/preferences.json")
    beta = dpo_config.get("beta", 0.1)
    n_epochs = config.get("epochs") or dpo_config.get("epochs", 3)

    dataset = PreferenceDataset.load_from_json(dataset_path, tokenizer)
    train_dataset, val_dataset = dataset.train_val_split(config.get("validation_split", 0.2))
    train_loader = DataLoader(train_dataset, batch_size=dpo_config.get("batch_size", 4), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=dpo_config.get("batch_size", 4))

    optimizer = optim.AdamW(
        policy_model.parameters(),
        lr=config.get("lr") or dpo_config.get("learning_rate", 5e-7),
    )

    print(f"DPO Training:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Beta: {beta}")
    print(f"  Train examples: {len(train_dataset)}")

    # YOUR CODE HERE
    # For each epoch:
    #   1. Training loop:
    #      - For each batch, compute DPO loss using dpo_loss()
    #      - Backpropagate and update policy_model
    #      - Track loss, accuracy, reward margin
    #   2. Validation:
    #      - Compute loss and accuracy on val set
    #   3. Generate sample outputs
    #   4. Save checkpoint
    raise NotImplementedError("Implement train_dpo")


# ---------------------------------------------------------------------------
# KL coefficient sweep (pre-written structure)
# ---------------------------------------------------------------------------

def run_kl_sweep(config: dict, device: torch.device) -> None:
    """Run PPO training with multiple KL coefficients.

    Args:
        config: Training configuration.
        device: Torch device.
    """
    kl_coeffs_str = config.get("kl_sweep")
    if isinstance(kl_coeffs_str, str):
        kl_coeffs = [float(x.strip()) for x in kl_coeffs_str.split(",")]
    else:
        kl_coeffs = config.get("kl_sweep", {}).get("coefficients", [0.01, 0.1, 1.0])

    results = {}
    for kl_coeff in kl_coeffs:
        print(f"\n{'='*60}")
        print(f"KL Coefficient: {kl_coeff}")
        print(f"{'='*60}")
        sweep_config = {**config, "kl_coeff": kl_coeff}
        train_ppo_rlhf(sweep_config, device)
        # Results will be saved in the checkpoint directory

    print("\nKL sweep complete!")


# ---------------------------------------------------------------------------
# Main entry point (pre-written)
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.n_pairs:
        config["n_pairs_per_prompt"] = args.n_pairs // config.get("n_prompts", 50)
    if args.preference_rule:
        config["preference_rule"] = args.preference_rule
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs:
        config["epochs"] = args.epochs
    if args.steps:
        config["steps"] = args.steps
    if args.kl_coeff:
        config["kl_coeff"] = args.kl_coeff
    if args.kl_sweep:
        config["kl_sweep"] = args.kl_sweep
    if args.lr:
        config["lr"] = args.lr

    device = get_device()
    set_seed(config.get("seed", 42))

    print(f"Device: {device}")

    # Create directories
    for d in ["checkpoint_dir", "log_dir", "data_dir"]:
        os.makedirs(config.get(d, d), exist_ok=True)

    if args.phase == "generate_data":
        generate_preference_data(config, device)
    elif args.phase == "reward_model":
        train_reward_model(config, device)
    elif args.phase == "ppo":
        if args.kl_sweep:
            run_kl_sweep(config, device)
        else:
            train_ppo_rlhf(config, device)
    elif args.phase == "dpo":
        train_dpo(config, device)


if __name__ == "__main__":
    main()
