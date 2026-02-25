# Assignment 3: Policy Gradients and PPO

## Overview

In this assignment, you will build the entire policy gradient pipeline, from the simplest algorithm (REINFORCE) through the industry standard (PPO). You will implement each method, observe its strengths and weaknesses, and understand precisely why each innovation was necessary. By the end, you will have a working PPO implementation that you could extend to train real-world agents.

This is the most important assignment in the module. PPO is the algorithm used to align Gemini, ChatGPT, and virtually every other frontier language model via RLHF. When you implement PPO, you are implementing the algorithm that connects reinforcement learning to the language model revolution.

**Estimated time:** 16-22 hours

**Prerequisites:** Module 13 Sessions 1-5, PyTorch fluency, understanding of policy gradient theorem and the PPO clipped objective.

---

## Part 1: REINFORCE -- The Simplest Policy Gradient

### 1.1 Policy Network

Implement a policy network for discrete action spaces:

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        """
        Architecture:
        - Linear(state_dim, hidden_dim) + ReLU
        - Linear(hidden_dim, hidden_dim) + ReLU
        - Linear(hidden_dim, n_actions)

        Output: logits for each action (use Categorical distribution)
        """
        ...

    def forward(self, state):
        """Return a torch.distributions.Categorical distribution."""
        ...
```

### 1.2 REINFORCE Implementation

Implement the REINFORCE algorithm:

```python
class REINFORCE:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        """Sample action from the policy. Return (action, log_prob)."""
        ...

    def update(self, log_probs, rewards):
        """REINFORCE update after a complete episode.

        1. Compute returns G_t = sum_{k=t}^T gamma^{k-t} * r_k for each t.
        2. Compute policy gradient loss: L = -sum_t log pi(a_t|s_t) * G_t
        3. Backpropagate and update.
        """
        ...
```

**Training:**
- Train on CartPole-v1 for 1000 episodes.
- Plot reward per episode (smoothed with a rolling window of 50).
- Run 5 random seeds and plot mean +/- standard deviation.

**Expected observation:** REINFORCE learns, but the reward curve is **noisy** -- it oscillates significantly between episodes. This is the high variance problem.

### 1.3 Variance Analysis

To visualize the variance problem:
- At episode 500, collect 20 episodes of data.
- For each episode, compute the policy gradient estimate.
- Print the mean and standard deviation of the gradient norms across the 20 episodes.
- The standard deviation should be large relative to the mean -- this is why REINFORCE converges slowly.

---

## Part 2: REINFORCE with Baseline -- Variance Reduction

### 2.1 Value Network (Baseline)

Add a value network to serve as a baseline:

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        """Predicts V(s) -- the expected return from state s."""
        ...

    def forward(self, state):
        """Return a scalar value estimate."""
        ...
```

### 2.2 REINFORCE with Baseline

Modify REINFORCE to subtract the baseline:

```python
class REINFORCEWithBaseline:
    def __init__(self, state_dim, n_actions, lr_policy=1e-3,
                 lr_value=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, n_actions)
        self.value_net = ValueNetwork(state_dim)
        ...

    def update(self, states, log_probs, rewards):
        """
        1. Compute returns G_t.
        2. Compute baselines b_t = V(s_t).
        3. Compute advantages A_t = G_t - b_t.
        4. Policy loss: L_pi = -sum_t log pi(a_t|s_t) * A_t
        5. Value loss: L_V = sum_t (G_t - V(s_t))^2
        6. Update both networks.
        """
        ...
```

**Training:**
- Train on CartPole-v1 for 1000 episodes, same 5 seeds as Part 1.
- Plot on the same axes as Part 1.
- Repeat the variance analysis from Part 1.3. The gradient variance should be **significantly lower**.

**Key insight:** the baseline does not change the expected gradient (prove this in your writeup by showing E[nabla log pi * b(s)] = 0). It only reduces variance, making learning faster and more stable.

---

## Part 3: Actor-Critic (A2C)

### 3.1 Actor-Critic Network

Implement a shared-backbone actor-critic network:

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        """
        Shared layers -> Actor head (action logits)
                      -> Critic head (state value)
        """
        ...

    def forward(self, state):
        """Return (action_distribution, value_estimate)."""
        ...
```

### 3.2 A2C Implementation

Implement Advantage Actor-Critic:

```python
class A2C:
    def __init__(self, state_dim, n_actions, lr=3e-4, gamma=0.99,
                 entropy_coef=0.01, value_coef=0.5):
        self.model = ActorCritic(state_dim, n_actions)
        ...

    def update(self, states, actions, rewards, next_states, dones):
        """A2C update using one-step TD advantages.

        1. Compute advantages: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        2. Actor loss: -mean(log pi(a_t|s_t) * A_t.detach())
        3. Critic loss: mean(A_t^2)  -- or MSE between V(s) and returns
        4. Entropy bonus: -mean(entropy of pi(.|s_t))
        5. Total loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
        """
        ...
```

**Critical detail:** the advantages must be `.detach()`ed before being used in the actor loss. The advantage is a signal to the actor, not something the actor should differentiate through.

**Training:**
- Train on CartPole-v1 for 1000 episodes.
- A2C should learn faster and more stably than REINFORCE with baseline, because it updates every step (not just at episode end) and uses TD advantages (lower variance than MC returns).

### 3.3 Entropy Bonus Analysis

Train A2C with three entropy coefficients: 0.0, 0.01, 0.1.
- Plot learning curves for all three.
- For each, plot the policy entropy over training.
- With entropy_coef=0.0, the policy may converge prematurely to a deterministic (suboptimal) policy. With 0.01, it should explore adequately. With 0.1, it may explore too much and learn slowly.

---

## Part 4: PPO -- The Full Implementation

This is the main event. Implement PPO with all the components that make it work in practice.

### 4.1 PPO Architecture

Use the same ActorCritic network from Part 3, but with separate actor and critic networks (not shared backbone). Use orthogonal weight initialization:

```python
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=64):
        """
        Separate actor and critic networks.
        Use nn.Tanh activations (standard for PPO).
        Initialize with orthogonal initialization (gain=sqrt(2) for hidden,
        gain=0.01 for policy output, gain=1.0 for value output).
        """
        ...

    def get_action_and_value(self, state, action=None):
        """
        Returns: (action, log_prob, entropy, value)
        If action is provided, compute log_prob for that action
        (used during PPO update phase).
        """
        ...

    def get_value(self, state):
        """Return V(s)."""
        ...
```

### 4.2 GAE Computation

Implement Generalized Advantage Estimation:

```python
def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """Compute GAE advantages.

    A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Args:
        rewards: list of rewards from the rollout
        values: list of V(s_t) estimates
        dones: list of done flags
        next_value: V(s_T) for the state after the last step
        gamma: discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: list of GAE advantage estimates
        returns: advantages + values (used as targets for the critic)
    """
    ...
```

**Verification:** for a simple case (3 steps, known rewards and values), compute GAE by hand and verify your implementation matches.

### 4.3 Parallel Environment Collection

Use gymnasium's vectorized environments:

```python
import gymnasium as gym

def make_env(env_name):
    def _make():
        env = gym.make(env_name)
        return env
    return _make

# Create N parallel environments
envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1") for _ in range(8)])
```

### 4.4 PPO Training Loop

Implement the complete PPO training loop:

```python
class PPOTrainer:
    def __init__(self, env_name="CartPole-v1", n_envs=8, n_steps=128,
                 n_epochs=4, mini_batch_size=32, lr=2.5e-4,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
                 total_timesteps=100000):
        ...

    def collect_rollout(self):
        """Collect n_steps transitions from n_envs parallel environments.

        Store: states, actions, log_probs, rewards, dones, values.
        At the end, compute next_value = V(s_T) for GAE.
        """
        ...

    def update(self, rollout_data):
        """PPO update: multiple epochs of mini-batch gradient descent.

        For each epoch:
            Shuffle the data.
            Split into mini-batches.
            For each mini-batch:
                1. Compute new log_probs and values for the stored (state, action) pairs.
                2. Compute ratio: r_t = exp(new_log_prob - old_log_prob)
                3. Clipped surrogate: L_CLIP = min(r*A, clip(r,1-eps,1+eps)*A)
                4. Value loss: MSE(V_new, returns)
                5. Entropy bonus: H[pi]
                6. Total loss = -L_CLIP + value_coef * L_VF - entropy_coef * H
                7. Gradient step with max_grad_norm clipping.
        """
        ...

    def train(self):
        """Main loop: collect rollout -> compute GAE -> PPO update -> repeat."""
        ...
```

**Implementation checklist** (each of these matters):
- [ ] Parallel environments (n_envs=8)
- [ ] GAE advantage estimation (lambda=0.95)
- [ ] Advantage normalization (zero mean, unit variance per batch)
- [ ] Clipped surrogate objective (eps=0.2)
- [ ] Multiple epochs per rollout (n_epochs=4)
- [ ] Mini-batch updates within each epoch
- [ ] Entropy bonus (coef=0.01)
- [ ] Value function loss (coef=0.5)
- [ ] Gradient norm clipping (max_norm=0.5)
- [ ] Orthogonal weight initialization
- [ ] Learning rate linear annealing (optional but recommended)

### 4.5 Training on CartPole-v1

Train PPO on CartPole-v1:
- Total timesteps: 100,000 (this should be more than enough).
- PPO should solve CartPole (average reward >= 475 over 100 episodes) in under 50,000 timesteps.
- Plot: reward per episode (smoothed), policy loss, value loss, entropy, clip fraction (how often the ratio is clipped).

### 4.6 Training on LunarLander-v2

Train PPO on LunarLander-v2:
- Total timesteps: 500,000 to 1,000,000.
- Adjust hyperparameters if needed (larger network, different learning rate).
- PPO should achieve average reward >= 200.
- Plot the same metrics as CartPole.

---

## Part 5: Method Comparison

### 5.1 Reward Curves

Train all four methods on CartPole-v1 with 5 random seeds each:
1. REINFORCE
2. REINFORCE with Baseline
3. A2C
4. PPO

Plot all four on the same axes (mean +/- standard deviation across seeds). X-axis: total environment timesteps (not episodes, since episodes have different lengths). Y-axis: average reward (rolling window of 50 episodes).

### 5.2 Comparison Table

| Method | Avg Reward (final 100 ep) | Timesteps to Solve | Wall-Clock Time | Gradient Variance |
|--------|--------------------------|--------------------|-----------------|--------------------|
| REINFORCE | | | | |
| REINFORCE + Baseline | | | | |
| A2C | | | | |
| PPO | | | | |

### 5.3 Sample Efficiency Analysis

For each method, report the number of environment timesteps needed to first achieve an average reward of 400 over 50 consecutive episodes. PPO should be the most sample-efficient due to data reuse (multiple epochs per rollout).

### 5.4 Written Analysis

Write a 500-700 word analysis covering:
1. **REINFORCE vs REINFORCE with Baseline**: quantify the variance reduction. How much faster does the baseline version converge?
2. **REINFORCE with Baseline vs A2C**: what is the effect of using TD advantages instead of MC returns? What is the effect of step-by-step updates instead of episode-level updates?
3. **A2C vs PPO**: what does the clipping mechanism provide? Show the clip fraction over training -- when is clipping active?
4. **Sample efficiency**: why is PPO more sample-efficient than A2C? (Answer: PPO reuses each batch for multiple gradient steps.)
5. **Wall-clock time**: which method is fastest in wall-clock time? Why? (PPO may be slower per timestep due to multiple epochs, but faster overall due to better sample efficiency.)

---

## Deliverables

1. **Code**: complete, runnable implementations of REINFORCE, REINFORCE with Baseline, A2C, and PPO.
2. **Plots**: all reward curves, loss curves, entropy curves, clip fraction curves. Method comparison plots with error bars across seeds.
3. **Comparison table**: filled in with experimental results.
4. **Written analysis**: 500-700 words covering the questions in Part 5.4.
5. **GAE verification**: hand-computed GAE example matching your implementation.

## Evaluation Criteria

- **Correctness** (30%): REINFORCE converges on CartPole. PPO solves CartPole efficiently and achieves good performance on LunarLander. GAE computation is correct.
- **Progressive development** (15%): clear narrative from REINFORCE through PPO, with each step motivated by a specific problem.
- **Implementation completeness** (20%): PPO includes all checklist items (parallel envs, GAE, clipping, entropy, normalization, etc.).
- **Analysis and comparison** (25%): fair comparison across methods, variance quantification, sample efficiency analysis, thoughtful written discussion.
- **Code quality** (10%): clean, modular code. The four methods should share common components where possible.

## Stretch Goals

1. **Continuous Action Spaces**: implement PPO for continuous action spaces. The policy network outputs the mean and log-std of a Gaussian distribution. Use `torch.distributions.Normal`. Train on Gymnasium Pendulum-v1 (action: torque in [-2, 2]) or MountainCarContinuous-v0. This requires changing the policy network, the action sampling, and the log-probability computation.

2. **PPO Ablation Study**: systematically ablate PPO components on LunarLander:
   - No advantage normalization
   - No entropy bonus
   - No gradient clipping
   - n_epochs = 1 (single epoch)
   - n_epochs = 20 (too many epochs -- policy diverges)
   - clip_eps = 0.05 (too conservative)
   - clip_eps = 0.5 (too permissive)
   Present results in a table. This teaches you which details matter most.

3. **RLHF on a Tiny Language Model**: implement a minimal RLHF pipeline using PPO to fine-tune a small character-level language model. The "reward" is a simple rule (e.g., reward proportional to the number of vowels in the generated text, or reward for generating text of a certain length). This is a conceptual bridge to Assignment 4. The key insight: the LM is the policy, tokens are actions, and the reward comes at the end of generation.
