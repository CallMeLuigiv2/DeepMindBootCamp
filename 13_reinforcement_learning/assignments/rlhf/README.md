# Assignment 4: RLHF and Reward Modeling

## Overview

Build the complete Reinforcement Learning from Human Feedback (RLHF) pipeline -- the
technology that transforms a pretrained language model into an aligned assistant. Train
a reward model on preference data, use PPO to optimize a language model against that
reward model, and implement DPO as a simpler alternative.

This is where the entire RL module converges: the Bellman equation, Q-Learning, policy
gradients, PPO -- everything comes together to solve the most important applied problem
in modern AI: making language models behave the way humans want.

## Objectives

- Load and generate text from a pretrained GPT-2 model
- Create a synthetic preference dataset with a defined preference function
- Train a reward model using the Bradley-Terry preference loss
- Fine-tune GPT-2 with PPO against the reward model (RLHF)
- Implement Direct Preference Optimization (DPO) as an alternative to PPO
- Analyze KL-reward tradeoff and investigate reward hacking
- Compare base model, RLHF-tuned model, and DPO-tuned model

## Key Equations

**Bradley-Terry Reward Model Loss:**

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$$

where $y_w$ is the preferred response, $y_l$ is the dispreferred response.

**RLHF Reward with KL Penalty:**

$$R(x, y) = r_\phi(x, y) - \beta \, D_{KL}\left[\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)\right]$$

**Per-Token KL Divergence:**

$$KL_t = \log \pi_\theta(a_t | s_t) - \log \pi_{ref}(a_t | s_t)$$

**DPO Loss:**

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right) \right]$$

**PPO Clipped Surrogate (applied to language model tokens):**

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

## Setup

```bash
# From this directory
pip install -e ../../../

# Or install requirements directly
pip install -r requirements.txt
```

**Note:** A GPU is strongly recommended for this assignment. CPU training is possible
with GPT-2 small (~124M parameters) but will be slow.

## How to Run

```bash
# Phase 1: Generate preference dataset
python train.py --phase generate_data --n_pairs 1000

# Phase 2: Train reward model
python train.py --phase reward_model --epochs 3

# Phase 3: PPO fine-tuning (RLHF)
python train.py --phase ppo --steps 200 --kl_coeff 0.1

# Train DPO as alternative
python train.py --phase dpo --epochs 3

# Evaluate and compare models
python evaluate.py --compare

# KL coefficient sweep
python train.py --phase ppo --kl_sweep 0.01,0.1,1.0
```

## File Descriptions

| File | Description |
|------|-------------|
| `config.yaml` | Hyperparameters: SFT epochs, reward model, PPO, DPO, KL coefficients |
| `model.py` | `RewardModel` (stubbed), RLHF policy wrapper (stubbed), DPO loss (stubbed) |
| `data.py` | `PreferenceDataset` (pre-written), `PromptDataset` (pre-written), preference generation |
| `train.py` | Three-phase training: reward model, PPO fine-tuning, DPO (stubbed loops) |
| `evaluate.py` | Model comparison, reward analysis, generation quality metrics (stubbed) |
| `utils.py` | Preference data generation, KL divergence, response sampling (fully implemented) |
| `notebooks/analysis.ipynb` | Reward model curves, RLHF training, KL-reward frontier, model comparison |

## Three-Phase Pipeline

1. **Phase 1 -- Data Generation**: Generate synthetic preference pairs using a defined preference rule.
2. **Phase 2 -- Reward Model**: Train a reward model to predict human preferences using Bradley-Terry loss.
3. **Phase 3a -- PPO Fine-Tuning**: Use PPO to optimize the language model against the reward model with KL penalty.
4. **Phase 3b -- DPO (Alternative)**: Train directly on preference pairs without a separate reward model.

## Stretch Goals

- Real human preferences (recruit annotators for 100 pairs)
- Constitutional AI self-critique pipeline
- Scaling to GPT-2 Medium (345M parameters)
- Multi-objective RLHF (helpfulness + brevity Pareto frontier)
