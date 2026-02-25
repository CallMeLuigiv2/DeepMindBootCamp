# Assignment 2: Deep Q-Network from Scratch

## Overview

Build a Deep Q-Network (DQN) from the ground up, progressively adding the innovations
that made the original DeepMind paper work. Start with a naive approach that fails,
then add experience replay, a target network, and Double DQN to make it succeed.

## Objectives

- Implement a Q-network that maps states to Q-values for all actions
- Observe why naive DQN fails (correlated data + moving targets)
- Add experience replay to break temporal correlation
- Add a target network to stabilize training targets
- Implement Double DQN to reduce Q-value overestimation
- Train and evaluate on CartPole-v1 and LunarLander-v2
- Conduct ablation study to quantify each component's contribution

## Key Equations

**Bellman Optimality (Q-Learning Update):**

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**DQN Loss (Huber Loss on TD Error):**

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ L_\delta \left( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \right) \right]$$

where $L_\delta$ is the Huber loss:

$$L_\delta(x) = \begin{cases} \frac{1}{2}x^2 & \text{if } |x| \leq \delta \\ \delta(|x| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

**Double DQN Target (decoupled selection and evaluation):**

$$y = r + \gamma \, Q\!\left(s', \arg\max_{a'} Q(s', a'; \theta); \, \theta^{-}\right)$$

**Soft Target Update:**

$$\theta^{-} \leftarrow \tau \, \theta + (1 - \tau) \, \theta^{-}$$

## Setup

```bash
# From this directory
pip install -e ../../../

# Or install requirements directly
pip install -r requirements.txt
```

## How to Run

```bash
# Train DQN on CartPole-v1
python train.py --env CartPole-v1 --variant dqn

# Train Double DQN on LunarLander-v2
python train.py --env LunarLander-v2 --variant double_dqn --episodes 1500

# Evaluate a trained agent
python evaluate.py --checkpoint checkpoints/best_model.pt --env LunarLander-v2

# Run ablation study
python train.py --env LunarLander-v2 --ablation
```

## File Descriptions

| File | Description |
|------|-------------|
| `config.yaml` | Hyperparameters: epsilon schedule, gamma, buffer size, network architecture |
| `model.py` | `QNetwork` and `DuelingQNetwork` classes (stubbed) |
| `data.py` | `ReplayBuffer` (pre-written), `PrioritizedReplayBuffer` (stubbed), environment wrappers |
| `train.py` | Training script with argparse, epsilon-greedy schedule, training loop (stubbed) |
| `evaluate.py` | Model loading, evaluation episodes, rendering (stubbed) |
| `utils.py` | Episode reward tracking, epsilon schedule, moving average (fully implemented) |
| `notebooks/analysis.ipynb` | Visualization of learning curves, Q-value analysis, ablation results |

## Progressive Development

1. **Part 1 -- Naive DQN**: No replay, no target network. Observe instability.
2. **Part 2 -- Add Replay**: Break temporal correlation. Observe improved stability.
3. **Part 3 -- Add Target Network**: Stabilize TD targets. Full DQN.
4. **Part 4 -- Double DQN**: Reduce overestimation bias.
5. **Part 5 -- LunarLander**: Apply to a harder environment.
6. **Part 6 -- Ablation Study**: Quantify each component's contribution.

## Stretch Goals

- Dueling DQN architecture (separate value and advantage streams)
- Prioritized Experience Replay (proportional prioritization by TD error)
- Rainbow (combine Double, Dueling, PER)
- Atari Pong (convolutional Q-network with frame stacking)
