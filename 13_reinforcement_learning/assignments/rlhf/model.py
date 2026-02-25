"""
RLHF Model Architectures
==========================

Implements model components for the RLHF pipeline:
- RewardModel: scalar reward prediction for (prompt, response) pairs
- RLHFPolicy: PPO policy wrapper over a language model
- DPO loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel
from typing import Tuple, Optional


class RewardModel(nn.Module):
    """Reward model that predicts a scalar reward for (prompt, response) pairs.

    Uses a GPT-2 backbone with a linear head on top of the last token's
    hidden state to produce a scalar reward. Trained on preference data
    using the Bradley-Terry loss.

    Architecture:
        GPT-2 backbone -> last token hidden state -> Linear(n_embd, 1) -> reward

    Args:
        model_name: Hugging Face model name for the backbone (default: "gpt2").
    """

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()

        # YOUR CODE HERE
        # 1. Load GPT-2 backbone using GPT2Model.from_pretrained(model_name)
        #    (GPT2Model, NOT GPT2LMHeadModel -- we don't need the LM head)
        # 2. Create a reward head: nn.Linear(backbone.config.n_embd, 1)
        # 3. Optionally initialize the reward head with small weights
        raise NotImplementedError("Implement RewardModel.__init__")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward for input sequences.

        The reward is computed from the last non-padding token's hidden state.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            rewards: Tensor of shape (batch_size,) -- scalar reward per example.
        """
        # YOUR CODE HERE
        # 1. Pass input_ids through the backbone to get hidden states
        # 2. Find the last non-padding token position for each example
        #    (use attention_mask to find the last 1 in each row)
        # 3. Extract the hidden state at that position
        # 4. Pass through reward_head to get scalar reward
        # 5. Squeeze to shape (batch_size,)
        raise NotImplementedError("Implement RewardModel.forward")


def compute_reward_model_loss(
    reward_model: RewardModel,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Compute the Bradley-Terry preference loss for the reward model.

    Loss = -E[log sigma(R(x, y_chosen) - R(x, y_rejected))]

    The reward model should assign higher reward to the chosen (preferred)
    response than to the rejected response.

    Args:
        reward_model: The reward model being trained.
        chosen_ids: Token IDs for preferred responses (batch_size, seq_len).
        rejected_ids: Token IDs for dispreferred responses (batch_size, seq_len).
        chosen_mask: Attention mask for chosen responses.
        rejected_mask: Attention mask for rejected responses.

    Returns:
        loss: Scalar Bradley-Terry loss.
        accuracy: Fraction of pairs where R(chosen) > R(rejected).
    """
    # YOUR CODE HERE
    # 1. Compute rewards for chosen and rejected responses
    # 2. Compute loss = -mean(log(sigmoid(r_chosen - r_rejected)))
    # 3. Compute accuracy = mean((r_chosen > r_rejected).float())
    raise NotImplementedError("Implement compute_reward_model_loss")


class RLHFPolicy:
    """PPO policy wrapper for RLHF language model fine-tuning.

    Wraps a GPT-2 language model as an RL policy where:
    - States = token sequences generated so far
    - Actions = next token predictions
    - Rewards = reward model score - KL penalty

    Args:
        policy_model: The GPT-2 model being fine-tuned.
        ref_model: Frozen reference model for KL computation.
        reward_model: Trained reward model for scoring responses.
        tokenizer: Tokenizer for encoding/decoding.
        kl_coeff: Beta coefficient for the KL penalty.
        clip_eps: PPO clipping parameter.
    """

    def __init__(
        self,
        policy_model: GPT2LMHeadModel,
        ref_model: GPT2LMHeadModel,
        reward_model: RewardModel,
        tokenizer,
        kl_coeff: float = 0.1,
        clip_eps: float = 0.2,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.kl_coeff = kl_coeff
        self.clip_eps = clip_eps

    def generate_responses(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate responses from the policy model.

        Uses sampling (not greedy) to maintain exploration during RL training.

        Args:
            prompt_ids: Prompt token IDs (batch_size, prompt_len).
            attention_mask: Attention mask for prompts.
            max_length: Maximum generation length (prompt + response).
            temperature: Sampling temperature.

        Returns:
            full_ids: Token IDs for (prompt + response) (batch_size, seq_len).
            response_mask: Mask indicating response tokens (batch_size, seq_len).
        """
        # YOUR CODE HERE
        # Use policy_model.generate() with do_sample=True, temperature=temperature
        # Return the full sequence and a mask for response-only tokens
        raise NotImplementedError("Implement RLHFPolicy.generate_responses")

    def compute_rewards_with_kl(
        self,
        full_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RLHF rewards with KL penalty.

        reward = R_phi(prompt, response) - beta * KL(pi_theta || pi_ref)

        Per-token KL: KL_t = log pi_theta(a_t|s_t) - log pi_ref(a_t|s_t)

        Args:
            full_ids: Token IDs for (prompt + response).
            attention_mask: Attention mask.
            prompt_length: Length of the prompt (to separate prompt from response).

        Returns:
            total_reward: reward model score - KL penalty (batch_size,).
            rm_score: raw reward model score (batch_size,).
            kl: KL divergence (batch_size,).
        """
        # YOUR CODE HERE
        # 1. Compute reward model score for the full sequence
        # 2. Compute per-token log probs under policy and reference models
        # 3. Compute KL = mean over response tokens of (log_pi - log_ref)
        # 4. total_reward = rm_score - kl_coeff * kl
        raise NotImplementedError("Implement RLHFPolicy.compute_rewards_with_kl")

    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """Perform one PPO update step.

        Args:
            states: Input token sequences.
            actions: Token actions taken.
            old_log_probs: Log probabilities under the old policy.
            advantages: Advantage estimates.
            optimizer: Optimizer for the policy model.

        Returns:
            Dictionary with policy_loss, clip_fraction, approx_kl.
        """
        # YOUR CODE HERE
        # 1. Compute new log_probs under current policy
        # 2. Compute ratio = exp(new_log_prob - old_log_prob)
        # 3. Compute clipped surrogate: min(r*A, clip(r,1-eps,1+eps)*A)
        # 4. Compute loss = -mean(clipped_surrogate)
        # 5. Backpropagate and update
        # 6. Return metrics
        raise NotImplementedError("Implement RLHFPolicy.ppo_update")


def dpo_loss(
    policy_model: GPT2LMHeadModel,
    ref_model: GPT2LMHeadModel,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, float, float]:
    """Compute the Direct Preference Optimization (DPO) loss.

    DPO bypasses the reward model entirely. Instead, it directly optimizes
    the policy on preference data using the implicit reward:
        r(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x))

    Loss = -E[log sigma(beta * (log(pi/pi_ref)(y_w) - log(pi/pi_ref)(y_l)))]

    Args:
        policy_model: The model being trained.
        ref_model: Frozen reference model.
        chosen_ids: Token IDs for preferred responses.
        rejected_ids: Token IDs for dispreferred responses.
        chosen_mask: Attention mask for chosen responses.
        rejected_mask: Attention mask for rejected responses.
        beta: Temperature parameter controlling deviation from reference.

    Returns:
        loss: Scalar DPO loss.
        accuracy: Fraction where implicit reward of chosen > rejected.
        reward_margin: Mean difference in implicit rewards (chosen - rejected).
    """
    # YOUR CODE HERE
    # 1. Compute log-probs of chosen/rejected under policy and reference models
    #    - For each model: run forward pass, get logits, compute per-token log-probs
    #    - Sum log-probs over response tokens only
    # 2. Compute log-ratios: log(pi/pi_ref) for chosen and rejected
    # 3. DPO loss = -mean(log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))))
    # 4. Accuracy = mean((log_ratio_chosen > log_ratio_rejected).float())
    # 5. Reward margin = mean(log_ratio_chosen - log_ratio_rejected)
    raise NotImplementedError("Implement dpo_loss")
