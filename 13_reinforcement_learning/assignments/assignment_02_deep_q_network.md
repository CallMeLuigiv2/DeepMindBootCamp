# Assignment 2: Deep Q-Network from Scratch

## Overview

In this assignment, you will build a Deep Q-Network (DQN) from the ground up, progressively adding the innovations that made the original DeepMind paper work. You will start with a naive approach that fails, then add experience replay and a target network to make it succeed. You will then implement Double DQN to address overestimation bias. By the end, you will have a deep and practical understanding of why DQN's innovations were necessary and how they work together.

This is the assignment where tabular RL meets deep learning. The Q-table becomes a neural network, and the challenges that arise -- correlated data, moving targets, overestimation -- are problems you will encounter in every deep RL system.

**Estimated time:** 12-16 hours

**Prerequisites:** Module 13 Sessions 1-3, PyTorch fluency, understanding of Q-Learning from Assignment 1.

---

## Part 1: The Naive Approach (Observe Failure)

### 1.1 Q-Network

Implement a Q-network: a neural network that takes a state vector as input and outputs Q-values for all actions.

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        """
        Architecture:
        - Linear(state_dim, hidden_dim) + ReLU
        - Linear(hidden_dim, hidden_dim) + ReLU
        - Linear(hidden_dim, n_actions)

        Output: Q(s, a) for all actions a
        """
        ...
```

### 1.2 Naive DQN (No Replay, No Target Network)

Implement DQN without experience replay or a target network. Train on CartPole-v1.

```python
class NaiveDQN:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.q_network = QNetwork(state_dim, n_actions)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        ...

    def select_action(self, state):
        """Epsilon-greedy selection."""
        ...

    def update(self, state, action, reward, next_state, done):
        """Single-transition update. No replay buffer.

        Loss: (r + gamma * max_a' Q(s', a'; theta) - Q(s, a; theta))^2

        WARNING: This uses the SAME network for both the target and the
        prediction. The target moves with every update.
        """
        ...
```

**Requirements:**
- Train for 500 episodes on CartPole-v1.
- Plot the reward per episode. You should observe **unstable, oscillating performance**. The agent may learn briefly, then forget and collapse.
- Record the maximum reward achieved and the reward at the end of training.

**Why it fails:**
1. Consecutive transitions are correlated (frame t and frame t+1 are nearly identical). The network overfits to recent experience.
2. The TD target uses the same network being updated. Each gradient step changes the target, creating a moving target that prevents convergence.

Write a brief explanation of why you observe instability. Refer to both problems explicitly.

---

## Part 2: Add Experience Replay

### 2.1 Replay Buffer

Implement an experience replay buffer:

```python
class ReplayBuffer:
    def __init__(self, capacity=100000):
        """Store transitions (s, a, r, s', done) in a fixed-size buffer.
        When full, overwrite the oldest transitions.
        """
        ...

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        ...

    def sample(self, batch_size):
        """Sample a random mini-batch of transitions.

        Returns:
            states: numpy array (batch_size, state_dim)
            actions: numpy array (batch_size,)
            rewards: numpy array (batch_size,)
            next_states: numpy array (batch_size, state_dim)
            dones: numpy array (batch_size,)
        """
        ...

    def __len__(self):
        ...
```

### 2.2 DQN with Replay (No Target Network)

Add the replay buffer to DQN. Now instead of updating on single transitions, sample a mini-batch from the buffer:

```python
class DQNWithReplay:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99,
                 epsilon=1.0, buffer_size=100000, batch_size=64):
        self.q_network = QNetwork(state_dim, n_actions)
        self.replay_buffer = ReplayBuffer(buffer_size)
        ...

    def update(self):
        """Sample mini-batch from replay buffer and update.

        Only update if the buffer has at least batch_size transitions.
        """
        ...
```

**Requirements:**
- Train for 500 episodes on CartPole-v1.
- Start updating only after the buffer has at least 1000 transitions.
- Plot the reward per episode on the same axes as Part 1.
- You should observe **improved stability** compared to the naive approach, but possibly still some oscillation.

**What replay fixes:** it breaks temporal correlation by sampling non-consecutive transitions. Each mini-batch is a diverse, decorrelated sample.

**What replay does NOT fix:** the target still uses the same network ($\theta$). It is still a moving target.

---

## Part 3: Add Target Network

### 3.1 Full DQN

Add a target network: a separate copy of the Q-network whose parameters are updated less frequently.

```python
class DQN:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99,
                 epsilon=1.0, buffer_size=100000, batch_size=64,
                 target_update_freq=100):
        # Online network: trained with gradient descent
        self.q_network = QNetwork(state_dim, n_actions)
        # Target network: frozen copy, updated periodically
        self.target_network = QNetwork(state_dim, n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        ...

    def update(self):
        """DQN update with target network.

        The TD target uses the TARGET network:
        y = r + gamma * max_a' Q(s', a'; theta^-)

        The loss is:
        L = (y - Q(s, a; theta))^2

        After every target_update_freq steps, copy theta to theta^-.
        """
        ...
```

**Requirements:**
- Train for 500 episodes on CartPole-v1.
- Try target_update_freq values of 10, 100, and 1000. Which works best?
- Plot reward curves for all three update frequencies on the same axes.
- Also plot on the same axes as Parts 1 and 2, creating a clear comparison of: Naive vs Replay-only vs Full DQN.

**What the target network fixes:** the TD target is now stable for target_update_freq steps. The network can make progress toward a fixed target before it changes.

### 3.2 Soft Target Updates

As an alternative to hard updates (copy every C steps), implement soft updates:

```python
def soft_update(self, tau=0.005):
    """Exponential moving average update:
    theta^- = tau * theta + (1 - tau) * theta^-
    """
    for target_param, online_param in zip(
        self.target_network.parameters(), self.q_network.parameters()
    ):
        target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
```

Call this after every gradient step (instead of periodic hard updates). Compare to hard updates. Which is smoother?

---

## Part 4: Double DQN

### 4.1 The Overestimation Problem

Before implementing the fix, demonstrate the problem:
- During training of standard DQN, log the mean Q-value predicted by the network for states in the replay buffer.
- Also log the actual mean return achieved from those states (by running the greedy policy and measuring returns).
- Plot both on the same axes. The predicted Q-values should be **higher** than the actual returns. This is overestimation.

### 4.2 Double DQN Implementation

Implement Double DQN. The only change is in the target computation:

```python
# Standard DQN target:
# y = r + gamma * max_a' Q(s', a'; theta^-)

# Double DQN target:
# a* = argmax_a' Q(s', a'; theta)          <-- online network selects
# y = r + gamma * Q(s', a*; theta^-)        <-- target network evaluates
```

```python
class DoubleDQN(DQN):
    def compute_target(self, next_states, rewards, dones):
        """Double DQN: decouple action selection from evaluation.

        Use online network to SELECT the best action.
        Use target network to EVALUATE that action.
        """
        with torch.no_grad():
            # Online network selects actions
            best_actions = self.q_network(next_states).argmax(dim=1)
            # Target network evaluates those actions
            next_q = self.target_network(next_states).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)
        return targets
```

### 4.3 Comparison

Train both standard DQN and Double DQN on CartPole-v1 for 500 episodes. Compare:
- Reward curves
- Q-value overestimation (predicted Q vs actual return)
- Final performance

Double DQN should show **less overestimation** and **more stable** learning.

---

## Part 5: LunarLander-v2

### 5.1 Training on a Harder Environment

Apply your best DQN variant (Double DQN with replay and target network) to LunarLander-v2.

**Environment details:**
- State: 8-dimensional continuous vector (position, velocity, angle, angular velocity, leg contacts)
- Actions: 4 discrete (do nothing, fire left, fire main, fire right)
- Reward: landing on the pad gives +100 to +140, crashing gives -100, each leg contact +10, firing engines costs fuel
- Solved: average reward >= 200 over 100 consecutive episodes

**Hyperparameter adjustments for LunarLander:**
- Increase training episodes to 1000-2000.
- Use a larger replay buffer (100K+).
- Try learning rates: 5e-4, 1e-3, 5e-3.
- Try target update frequencies: 50, 100, 500.
- Network: two hidden layers of 128 or 256 units.

### 5.2 Metrics to Track

Plot the following over training:
1. **Reward per episode** (with a smoothed rolling average).
2. **Mean Q-value** for a fixed set of 100 states sampled from the replay buffer at episode 100 (track how Q-values for these states evolve).
3. **Replay buffer size** over time.
4. **Epsilon** over time.
5. **Loss** (DQN loss per update step, smoothed).

### 5.3 Render Trained Agent

After training, render the agent playing several episodes (use `env = gym.make("LunarLander-v2", render_mode="human")` or record to video). Describe the agent's strategy in words.

---

## Part 6: Ablation Study

Run the following ablation experiments on LunarLander-v2 and present results in a table:

| Configuration | Avg Reward (last 100 ep) | Max Reward | Episodes to Solve |
|---|---|---|---|
| Full Double DQN | | | |
| DQN (no Double) | | | |
| No target network | | | |
| No experience replay | | | |
| No replay + no target | | | |
| Buffer size = 1000 | | | |
| Buffer size = 100000 | | | |

This table should make clear the contribution of each component.

---

## Deliverables

1. **Code**: complete, runnable implementations of NaiveDQN, DQNWithReplay, DQN, DoubleDQN.
2. **Plots**: all visualizations described above, clearly labeled and on shared axes where specified.
3. **Ablation table**: the comparison table from Part 6.
4. **Written explanations**: brief paragraphs explaining (a) why the naive approach fails, (b) what replay fixes and does not fix, (c) what the target network fixes, (d) why Double DQN reduces overestimation.

## Evaluation Criteria

- **Correctness** (35%): DQN solves CartPole-v1 reliably. Double DQN shows reduced overestimation. LunarLander achieves reasonable performance.
- **Progressive development** (20%): clear demonstration of each component's contribution -- the narrative of "naive fails, replay helps, target stabilizes, Double DQN refines."
- **Analysis and visualization** (25%): clear plots showing the effect of each innovation. Ablation study is complete and informative.
- **Code quality** (20%): modular code with the DQN variants sharing common components. Clean separation of concerns.

## Stretch Goals

1. **Dueling DQN**: implement the dueling architecture (separate value and advantage streams). Compare to standard DQN on LunarLander.

2. **Prioritized Experience Replay**: implement proportional prioritization. Sample transitions with probability proportional to $|\text{TD error}|^\alpha$. Add importance sampling weights. Compare replay buffer utilization and learning speed.

3. **Rainbow**: combine Double DQN, Dueling DQN, and Prioritized Experience Replay into a single agent. This is a subset of the full Rainbow paper (Hessel et al., 2018), which combines six DQN improvements.

4. **Atari Pong**: if you have a GPU, try training DQN on PongNoFrameskip-v4. You will need:
   - Frame preprocessing (grayscale, resize to 84x84, frame stacking of 4 frames)
   - Convolutional Q-network (3 conv layers + 2 FC layers, as in the original DQN paper)
   - Larger replay buffer (100K+ transitions)
   - More training steps (1M+ environment steps)
   - This is a significant undertaking but deeply satisfying. Watching your DQN learn to play Pong is one of the milestone experiences in RL.
