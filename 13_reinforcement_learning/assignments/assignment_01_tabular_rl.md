# Assignment 1: Tabular Reinforcement Learning

## Overview

In this assignment, you will build the foundations of reinforcement learning from scratch. You will implement a GridWorld environment, solve it exactly with Dynamic Programming, then learn to solve it from experience using Q-Learning and SARSA. By the end, you will have a visceral understanding of the Bellman equation, the difference between on-policy and off-policy learning, and the exploration-exploitation tradeoff.

This is where RL begins. Every algorithm you encounter in later assignments -- DQN, PPO, RLHF -- is a descendant of the tabular methods you build here. Master the tabular case and the deep RL extensions will be natural.

**Estimated time:** 10-14 hours

**Prerequisites:** Module 13 Sessions 1-2, understanding of Bellman equations and TD learning.

---

## Part 1: Implement a GridWorld Environment

### 1.1 The Environment

Build a GridWorld environment from scratch (do not use Gymnasium for this part -- you are building the environment itself).

**Specification:**
- Grid size: configurable (default 5x5)
- States: each cell is a state, identified by (row, col) or a flat index
- Actions: 4 discrete actions -- up (0), down (1), left (2), right (3)
- Transitions: deterministic. Moving into a wall keeps the agent in the same cell.
- Start state: top-left corner (0, 0)
- Goal state: bottom-right corner (4, 4)
- Reward: -1 per step (encourages the agent to reach the goal quickly), 0 at the goal
- Terminal: the episode ends when the agent reaches the goal

**Required interface:**
```python
class GridWorld:
    def __init__(self, size=5):
        ...

    def reset(self) -> int:
        """Reset to start state. Return initial state."""
        ...

    def step(self, action: int) -> tuple:
        """Take action. Return (next_state, reward, done)."""
        ...

    def get_transition_prob(self, state, action):
        """Return list of (next_state, reward, probability) tuples.
        Needed for Dynamic Programming methods."""
        ...

    def state_to_rowcol(self, state) -> tuple:
        """Convert flat state index to (row, col)."""
        ...

    def render(self):
        """Print the grid with agent position marked."""
        ...
```

**Verification:**
- Reset returns state 0. Step with action "down" from state 0 returns state 5 (second row).
- Moving "up" from state 0 returns state 0 (wall -- stay in place).
- Reaching the goal returns reward 0 and done=True. All other steps return reward -1.

### 1.2 Stochastic Extension

Add an optional stochastic mode: with probability 0.8, the action succeeds. With probability 0.2, a random action is taken instead. This models a "slippery floor" and makes the problem harder. Update `get_transition_prob` accordingly.

---

## Part 2: Dynamic Programming -- Solve the GridWorld Exactly

### 2.1 Value Iteration

Implement Value Iteration to find the optimal value function V* and optimal policy pi*:

```python
def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Args:
        env: GridWorld environment
        gamma: discount factor
        theta: convergence threshold

    Returns:
        V: optimal value function (array of size n_states)
        policy: optimal policy (array of size n_states, mapping state -> action)
        n_iterations: number of iterations to converge
    """
```

**Requirements:**
- Iterate the Bellman optimality equation until the maximum change in V is below theta.
- Extract the greedy policy from the converged V.
- Print the number of iterations to convergence.
- Run on both the deterministic and stochastic GridWorld.

### 2.2 Policy Iteration

Implement Policy Iteration:

```python
def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    Returns:
        V: optimal value function
        policy: optimal policy
        n_iterations: number of policy improvement steps
    """
```

**Requirements:**
- Start with a random policy (uniform over actions).
- Alternate between policy evaluation (solve for V^pi) and policy improvement (make pi greedy w.r.t. V^pi).
- Stop when the policy does not change.
- Verify that Value Iteration and Policy Iteration produce the same optimal policy.

### 2.3 Visualization

- **Value function heatmap**: display V*(s) as a color-coded grid. Cells near the goal should have higher values (less negative).
- **Optimal policy arrows**: display the optimal action at each cell as an arrow (up, down, left, right).
- Use matplotlib for both visualizations.

---

## Part 3: Q-Learning -- Learn from Experience

### 3.1 Q-Learning Agent

Implement a tabular Q-Learning agent:

```python
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        ...

    def select_action(self, state) -> int:
        """Epsilon-greedy action selection."""
        ...

    def update(self, state, action, reward, next_state, done):
        """Q-Learning update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        ...

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        ...
```

### 3.2 Training Loop

```python
def train_q_learning(env, agent, n_episodes=5000):
    """Train the Q-Learning agent.

    Returns:
        rewards_per_episode: list of total rewards for each episode
        steps_per_episode: list of steps taken per episode
    """
```

**Requirements:**
- Train for 5000 episodes.
- Track total reward and number of steps per episode.
- After training, extract the learned policy: pi(s) = argmax_a Q(s, a).
- Compare the learned policy to the optimal policy from Value Iteration. Are they the same?

### 3.3 Epsilon Decay Analysis

Run Q-Learning with three different epsilon schedules:
1. **Constant epsilon** = 0.1 (no decay)
2. **Linear decay**: epsilon decreases linearly from 1.0 to 0.01 over all episodes
3. **Exponential decay**: epsilon *= 0.995 after each episode

Plot the learning curves (cumulative reward vs episode) for all three. Which converges fastest? Which produces the best final policy?

---

## Part 4: SARSA -- On-Policy Learning

### 4.1 SARSA Agent

Implement a SARSA agent:

```python
class SARSAAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        ...

    def select_action(self, state) -> int:
        ...

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        """
        ...
```

Note the key difference: SARSA's update method takes `next_action` as an argument, because it uses Q(s', a') instead of max_a' Q(s', a').

### 4.2 The Cliff-Walking Comparison

Implement a CliffWorld environment:
- Grid: 4 rows x 12 columns
- Start: bottom-left (3, 0)
- Goal: bottom-right (3, 11)
- Cliff: cells (3, 1) through (3, 10) along the bottom row
- Reward: -1 per step, -100 for stepping on the cliff (and reset to start), 0 at goal

Train both Q-Learning and SARSA on CliffWorld with epsilon=0.1 (constant -- do not decay, so the difference is visible).

**Expected behavior:**
- Q-Learning learns the **optimal path** (along the cliff edge) because it evaluates the greedy policy.
- SARSA learns the **safe path** (one row above the cliff) because it evaluates the policy it is actually following, which includes epsilon-greedy exploration.

**Visualization:**
- Plot both learned policies on the grid.
- Plot learning curves for both algorithms.
- Write a paragraph explaining WHY they differ. This is the key insight about on-policy vs off-policy learning.

---

## Part 5: Analysis and Visualization

### 5.1 Learned Value Function Heatmap

For Q-Learning's converged Q-table:
- Compute V(s) = max_a Q(s, a) for each state.
- Display as a heatmap. States near the goal should have high value; states far away should have low value.
- Compare to the DP-computed V* from Part 2.

### 5.2 Learned Policy Arrows

Display the learned policy (argmax_a Q(s,a)) as arrows on the grid. Compare to the DP-computed optimal policy.

### 5.3 Learning Dynamics

- Plot the **running average reward** (window of 100 episodes) vs episode number for Q-Learning and SARSA.
- Plot **epsilon** vs episode number (for the exponential decay schedule).
- Plot the **maximum Q-value** across all (s,a) pairs vs episode number. This should stabilize as learning converges.

### 5.4 Written Analysis

Write a 300-500 word analysis addressing:
1. Do Q-Learning and Value Iteration converge to the same policy? Why or why not?
2. How does the epsilon decay schedule affect convergence speed and final performance?
3. Why do Q-Learning and SARSA learn different policies on CliffWorld? Which would you prefer in a safety-critical application?
4. What happens when you increase the learning rate alpha? What happens when you decrease it?
5. How does the discount factor gamma affect the learned policy?

---

## Deliverables

1. **Code**: a Jupyter notebook or Python scripts containing all implementations.
2. **Visualizations**: all plots described above (heatmaps, arrow grids, learning curves).
3. **Written analysis**: the 300-500 word analysis from Part 5.4.
4. **Correctness verification**: demonstration that Value Iteration and Q-Learning produce the same optimal policy on the deterministic GridWorld.

## Evaluation Criteria

- **Correctness** (40%): Value Iteration converges to the correct V*. Q-Learning converges to a near-optimal policy. SARSA shows the expected safe-path behavior on CliffWorld.
- **Code quality** (20%): clean, documented code with clear separation of environment, agent, and training loop.
- **Visualizations** (20%): clear, labeled plots that communicate the learning dynamics effectively.
- **Analysis** (20%): demonstrates genuine understanding of on-policy vs off-policy, exploration vs exploitation, and the connection between DP and RL.

## Stretch Goals

1. **TD(lambda)**: implement Q-Learning with eligibility traces. Compare learning speed to standard Q-Learning for different lambda values (0, 0.5, 0.9, 1.0).
2. **Taxi-v3**: apply your Q-Learning agent to the Gymnasium Taxi-v3 environment. This is a larger state space (500 states) and requires discovering a multi-step strategy (pick up passenger, drive to destination, drop off). Report learning curves and final success rate.
3. **Expected SARSA**: implement Expected SARSA (use expected Q-value under the policy instead of a single sample). Compare to SARSA and Q-Learning. Expected SARSA should have lower variance than SARSA.
4. **Visualization of Q-value convergence**: create an animation showing how Q-values evolve over training episodes. Use matplotlib animation.
