# Assignment 3: Policy Gradients and PPO

## Overview

Build the entire policy gradient pipeline, from the simplest algorithm (REINFORCE) through
the industry standard (PPO). Implement each method, observe its strengths and weaknesses,
and understand precisely why each innovation was necessary.

PPO is the algorithm used to align Gemini, ChatGPT, and virtually every other frontier
language model via RLHF. When you implement PPO, you are implementing the algorithm that
connects reinforcement learning to the language model revolution.

## Objectives

- Implement REINFORCE and observe high-variance gradient estimates
- Add a value baseline for variance reduction (REINFORCE with Baseline)
- Implement Advantage Actor-Critic (A2C) with TD advantages and entropy bonus
- Implement full PPO with clipped surrogate objective, GAE, and parallel environments
- Compare all four methods on CartPole-v1 and LunarLander-v2
- Analyze sample efficiency, gradient variance, and wall-clock performance

## Key Equations

**Policy Gradient Theorem (REINFORCE):**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \, G_t \right]$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return from timestep $t$.

**Advantage Actor-Critic (A2C):**

$$A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**Generalized Advantage Estimation (GAE):**

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**PPO Clipped Surrogate Objective:**

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$.

**PPO Total Loss:**

$$L = -L^{CLIP} + c_1 \, L^{VF} - c_2 \, H[\pi_\theta]$$

## Setup

```bash
# From this directory
pip install -e ../../../

# Or install requirements directly
pip install -r requirements.txt
```

## How to Run

```bash
# Train REINFORCE on CartPole-v1
python train.py --method reinforce --env CartPole-v1

# Train PPO on CartPole-v1
python train.py --method ppo --env CartPole-v1 --total_timesteps 100000

# Train PPO on LunarLander-v2
python train.py --method ppo --env LunarLander-v2 --total_timesteps 500000

# Run method comparison (5 seeds each)
python train.py --compare --seeds 5

# Evaluate a trained agent
python evaluate.py --checkpoint checkpoints/ppo_best.pt --env CartPole-v1
```

## File Descriptions

| File | Description |
|------|-------------|
| `config.yaml` | Hyperparameters: clip ratio, entropy coeff, GAE lambda, learning rate schedules |
| `model.py` | `PolicyNetwork`, `ValueNetwork`, `ActorCritic` classes (stubbed) |
| `data.py` | `RolloutBuffer` (pre-written), GAE computation (stubbed) |
| `train.py` | REINFORCE, A2C, and PPO training loops (stubbed), argparse/logging (pre-written) |
| `evaluate.py` | Model loading, evaluation, rendering (stubbed) |
| `utils.py` | Advantage helpers, policy entropy, explained variance (fully implemented) |
| `notebooks/analysis.ipynb` | Method comparison plots, variance analysis, ablation results |

## Progressive Development

1. **Part 1 -- REINFORCE**: Simplest policy gradient. High variance, slow convergence.
2. **Part 2 -- REINFORCE with Baseline**: Value baseline reduces gradient variance.
3. **Part 3 -- A2C**: TD advantages + entropy bonus. Step-by-step updates.
4. **Part 4 -- PPO**: Clipped surrogate, GAE, parallel envs. Industry standard.
5. **Part 5 -- Method Comparison**: Fair comparison across all methods with 5 seeds.

## Stretch Goals

- Continuous action spaces (Gaussian policy for Pendulum-v1)
- PPO ablation study (normalization, entropy, clipping, epoch count)
- RLHF on a tiny language model (bridge to Assignment 4)
