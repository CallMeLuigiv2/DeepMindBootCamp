# Module 13: Reinforcement Learning -- Lesson Plan

## Weeks 17-18 | Teaching Agents to Act

Reinforcement learning is where DeepMind began, and it remains the intellectual core of the lab. AlphaGo defeated the world champion at Go. AlphaFold solved protein structure prediction using RL-inspired training. AlphaProof proved mathematical theorems by combining neural networks with Monte Carlo Tree Search. RLHF aligns Gemini with human preferences. SIMA agents navigate 3D worlds. Every one of these systems is built on the ideas in this module.

You have spent twelve modules learning to build neural networks that map inputs to outputs -- supervised learning. Now you learn something fundamentally different: how to train agents that act in environments, receive rewards, and improve through trial and error. In supervised learning, the labels are given. In reinforcement learning, the agent must discover them through interaction.

This is harder than supervised learning. The agent faces delayed rewards (a chess move may not pay off for 40 turns), partial observability (you cannot see the opponent's strategy), the exploration-exploitation dilemma (do you try something new or stick with what works?), and non-stationary data (the training distribution changes as the policy improves). These challenges make RL both more difficult and more powerful than supervised learning.

By the end of this module, you will be able to:
- Formalize sequential decision problems as Markov Decision Processes
- Derive and implement the core RL algorithms: Q-Learning, DQN, REINFORCE, PPO
- Explain precisely why experience replay, target networks, and clipped objectives are necessary
- Implement the full RLHF pipeline that connects RL to large language models
- Understand Monte Carlo Tree Search and how DeepMind combines it with neural networks

---

## Session 1: Foundations of Reinforcement Learning

**Duration**: 3.5 hours (1.5 hours lecture, 2 hours derivation and conceptual exercises)
**Date**: Week 17, Day 1

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Define the RL problem formally: agent, environment, state, action, reward, policy, value function.
2. Formalize sequential decision problems as Markov Decision Processes (MDPs) and state the Markov property.
3. Derive the Bellman equation for both state-value $V(s)$ and action-value $Q(s,a)$.
4. Explain why we discount future rewards and the role of $\gamma$.
5. Distinguish exploration from exploitation and describe $\epsilon$-greedy, UCB, and Thompson sampling strategies.
6. Navigate the RL taxonomy: model-free vs model-based, on-policy vs off-policy, value-based vs policy-based.
7. Draw the agent-environment interaction loop and the MDP diagram.

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 25 min | The RL problem: agent-environment loop, the reward hypothesis |
| 2 | 25 min | Markov Decision Processes: states, actions, transitions, rewards |
| 3 | 30 min | Whiteboard: Bellman equations -- state-value, action-value, optimality |
| 4 | 20 min | The discount factor: why gamma matters, effective horizon |
| 5 | 25 min | Exploration vs exploitation: epsilon-greedy, UCB, Thompson sampling |
| 6 | 25 min | The RL taxonomy: the map of all RL algorithms |
| 7 | 20 min | Connection to supervised learning: what changes, what stays the same |

### Core Concepts

**The RL Problem**

The fundamental setup: an **agent** interacts with an **environment** in discrete time steps. At each step $t$, the agent observes a **state** $s_t$, takes an **action** $a_t$ according to its **policy** $\pi(a \mid s)$, receives a **reward** $r_t$, and transitions to a new state $s_{t+1}$. The goal is to find a policy that maximizes the expected cumulative reward.

Draw the diagram:

```
        action a_t
Agent  ----------->  Environment
  ^                      |
  |    state s_{t+1}     |
  |    reward r_t        |
  +<---------------------+
```

The **reward hypothesis** (Sutton): all goals can be described as the maximization of expected cumulative reward. This is a strong claim. Discuss when it breaks down.

**Connecting to Supervised Learning**

In supervised learning, you have (input, label) pairs. The loss function tells you exactly how wrong you are on every example. In RL, there are no labels -- only rewards, which may be sparse (you only learn if you won or lost at the end of a chess game) and delayed (the action that caused you to lose happened 30 moves ago). The agent must assign credit to past actions -- this is the **credit assignment problem**.

**Markov Decision Process**

An MDP is defined by the tuple $(S, A, P, R, \gamma)$:
- $S$: the set of states
- $A$: the set of actions
- $P(s' \mid s, a)$: the transition probability -- probability of reaching state $s'$ given state $s$ and action $a$
- $R(s, a, s')$: the reward function
- $\gamma$: the discount factor, $0 \le \gamma < 1$

The **Markov property**: the future depends only on the current state, not the history. $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_1, a_1, \ldots, s_t, a_t)$. This is a strong assumption. When it does not hold (e.g., partially observable environments), we use POMDPs or frame stacking.

**The Bellman Equation**

Derived on the whiteboard. The value of a state is the expected total discounted reward from that state forward:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[r_t + \gamma \, V^\pi(s_{t+1}) \mid s_t = s\right]$$

The Q-value of a state-action pair:

$$Q^\pi(s, a) = \mathbb{E}\!\left[r_t + \gamma \, V^\pi(s_{t+1}) \mid s_t = s,\, a_t = a\right]$$

The Bellman optimality equations:

$$V^*(s) = \max_a Q^*(s, a)$$

$$Q^*(s, a) = \mathbb{E}\!\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

These recursive equations are the foundation of all value-based RL. Every algorithm in Sessions 2-3 is a different strategy for solving them.

**The Discount Factor**

Why $\gamma < 1$? Three reasons: (1) mathematical -- ensures the infinite sum converges, (2) practical -- immediate rewards are more certain than future ones, (3) computational -- $\gamma$ controls the effective horizon. With $\gamma = 0.99$, the effective horizon is about 100 steps. With $\gamma = 0.9$, about 10 steps.

**Exploration vs Exploitation**

- **Epsilon-greedy**: with probability $\epsilon$, take a random action; otherwise, take the greedy action. Simple, effective, but undirected exploration.
- **Upper Confidence Bound (UCB)**: choose actions that either have high estimated value OR have been tried few times. Balances optimism with uncertainty.
- **Thompson Sampling**: maintain a posterior distribution over Q-values, sample from it, act greedily on the sample. Principled Bayesian exploration.

**The RL Taxonomy**

- **Model-free vs Model-based**: does the agent learn a model of the environment (transition dynamics), or does it learn a policy/value function directly? Model-free is simpler but less sample-efficient.
- **On-policy vs Off-policy**: does the agent learn from data collected by its current policy (on-policy) or from data collected by any policy (off-policy)? Off-policy can reuse old data but is harder to stabilize.
- **Value-based vs Policy-based**: does the agent learn a value function and derive a policy (value-based), or directly learn a parameterized policy (policy-based)? Actor-critic methods combine both.

### Derivation Exercises

1. Derive the Bellman equation for $V^\pi(s)$ from the definition of return $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$.
2. Show that $V^\pi(s) = \sum_a \pi(a \mid s) \, Q^\pi(s,a)$. Explain what this means.
3. Derive the Bellman optimality equation for $Q^*(s,a)$ from the Bellman equation for $Q^\pi(s,a)$.
4. For a two-state MDP with known transitions and rewards, solve the Bellman equation by hand to find $V^*$.

### Coding Tasks

1. Implement a simple GridWorld environment class with states, actions, transitions, and rewards.
2. For a small MDP (4 states, 2 actions), compute $V^\pi$ exactly by solving the linear system of Bellman equations.
3. Implement $\epsilon$-greedy action selection with decaying $\epsilon$.

### Paper References

- Sutton and Barto, "Reinforcement Learning: An Introduction" (2018) -- the RL bible, Chapters 1-3
- Bellman, "Dynamic Programming" (1957) -- the origin of the Bellman equation
- Auer et al., "Finite-time Analysis of the Multiarmed Bandit Problem" (2002) -- UCB analysis

---

## Session 2: Value-Based Methods

**Duration**: 3.5 hours (1.5 hours lecture, 2 hours implementation)
**Date**: Week 17, Day 2

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Implement policy evaluation, policy iteration, and value iteration for tabular MDPs.
2. Explain the difference between Monte Carlo and Temporal Difference methods.
3. Derive the Q-Learning update rule and explain every term.
4. Implement Q-Learning and SARSA and explain why they differ.
5. Explain n-step returns and eligibility traces as a spectrum between MC and TD.

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 30 min | Dynamic programming: policy evaluation, policy iteration, value iteration |
| 2 | 25 min | Monte Carlo methods: first-visit, every-visit, importance sampling |
| 3 | 30 min | Temporal Difference learning: TD(0), the bootstrapping insight |
| 4 | 25 min | Whiteboard: Q-Learning and SARSA derivation and comparison |
| 5 | 20 min | n-step returns, TD(lambda), eligibility traces |
| 6 | 60 min | Implementation: Q-Learning on GridWorld |

### Core Concepts

**Dynamic Programming**

When the MDP is fully known (transition probabilities and rewards), we can solve for the optimal policy exactly.

*Policy Evaluation*: Given a fixed policy $\pi$, compute $V^\pi$ by iterating the Bellman equation:

$$V_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma \, V_k(s')\right]$$

Repeat until convergence. This is a system of linear equations solved iteratively.

*Policy Iteration*: Alternate between (1) policy evaluation (compute $V^\pi$) and (2) policy improvement (make $\pi$ greedy with respect to $V^\pi$). Converges to the optimal policy in a finite number of steps.

*Value Iteration*: Combine evaluation and improvement into a single step:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma \, V_k(s')\right]$$

This is the Bellman optimality equation applied iteratively.

**Monte Carlo Methods**

When the MDP is unknown (we can only sample episodes), we estimate values from complete episodes.

*First-visit MC*: For each state $s$, average the returns $G_t$ following the first visit to $s$ across many episodes.

*Every-visit MC*: Average returns following every visit to $s$.

MC methods have zero bias (they use actual returns) but high variance (returns vary widely across episodes). They require complete episodes.

*Importance Sampling*: For off-policy MC, correct for the difference between the behavior policy (which generated the data) and the target policy (which we are evaluating). The importance ratio is $\pi(a \mid s) / b(a \mid s)$ for each step.

**Temporal Difference Learning**

TD methods bootstrap: instead of waiting for the full return $G_t$, they use an estimate.

*TD(0) update*:

$$V(s_t) \leftarrow V(s_t) + \alpha \left[r_t + \gamma \, V(s_{t+1}) - V(s_t)\right]$$

The term $(r_t + \gamma \, V(s_{t+1}))$ is the **TD target**. The difference (TD target $- V(s_t)$) is the **TD error**. This updates after every step, not after complete episodes. Lower variance than MC (bootstrapping smooths estimates), but introduces bias (the target uses the current imperfect estimate).

**Q-Learning**

The most important tabular RL algorithm. Off-policy TD control.

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

Break down every term:
- $Q(s_t, a_t)$: the current estimate of the value of taking action $a_t$ in state $s_t$
- $\alpha$: the learning rate (how much to update)
- $r_t$: the immediate reward received
- $\gamma \max_{a'} Q(s_{t+1}, a')$: the discounted value of the best action in the next state (the bootstrap)
- $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$: the TD target -- "what we should be closer to"
- The parenthetical difference: the TD error -- "how wrong we were"

Q-Learning is **off-policy** because the target uses max (the optimal action), regardless of what action the agent actually takes. This means it can learn the optimal policy while following an exploratory policy.

**SARSA**

On-policy TD control.

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma \, Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]$$

The difference from Q-Learning: instead of $\max_{a'} Q(s_{t+1}, a')$, we use $Q(s_{t+1}, a_{t+1})$ -- the value of the action actually taken. SARSA learns the value of the policy it is following, including its exploration. On the cliff-walking problem, SARSA learns a safer path because it accounts for the possibility of random exploration near the cliff.

**n-step Returns and TD(lambda)**

MC uses the full return. TD(0) uses one step. n-step returns use n steps:

$$G_t^{(n)} = r_t + \gamma \, r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$$

TD($\lambda$) combines all n-step returns using exponential weighting ($\lambda$). When $\lambda=0$, this is TD(0). When $\lambda=1$, this is MC. Eligibility traces provide an efficient implementation: each state maintains a trace that decays by $\gamma \lambda$ and is incremented when visited.

### Derivation Exercises

1. Prove that Q-Learning converges to $Q^*$ under appropriate conditions (sketch the argument using contraction mappings).
2. Show that SARSA with a greedy policy ($\epsilon=0$) is equivalent to Q-Learning.
3. Derive the importance sampling ratio for off-policy MC evaluation.

### Coding Tasks

1. Implement Value Iteration for a 4x4 GridWorld. Visualize the converged value function.
2. Implement Q-Learning for the same GridWorld. Plot the learning curve (cumulative reward vs episodes).
3. Implement SARSA. Compare Q-Learning and SARSA on the cliff-walking problem.

### Paper References

- Watkins and Dayan, "Q-Learning" (1992) -- the original Q-Learning paper
- Rummery and Niranjan, "On-line Q-learning using connectionist systems" (1994) -- SARSA
- Sutton, "Learning to Predict by the Methods of Temporal Differences" (1988) -- TD learning
- Sutton and Barto, "Reinforcement Learning: An Introduction" (2018) -- Chapters 4-7, 12

---

## Session 3: Deep Q-Networks (DQN)

**Duration**: 3.5 hours (1.5 hours lecture, 2 hours implementation)
**Date**: Week 17, Day 3

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Explain why tabular Q-Learning fails for large or continuous state spaces.
2. Describe the two key innovations of DQN: experience replay and target networks.
3. Derive the DQN loss function and relate it to supervised regression.
4. Implement DQN from scratch in PyTorch.
5. Explain Double DQN, Dueling DQN, and Prioritized Experience Replay as improvements.

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 20 min | The curse of dimensionality: why tables fail, function approximation |
| 2 | 30 min | The DQN paper: experience replay, target networks, the loss function |
| 3 | 20 min | Whiteboard: DQN loss derivation and gradient computation |
| 4 | 20 min | Double DQN, Dueling DQN, Prioritized Experience Replay |
| 5 | 20 min | Practical DQN: hyperparameters, common failure modes, debugging tips |
| 6 | 80 min | Implementation: DQN on CartPole and LunarLander |

### Core Concepts

**Why Tabular Methods Fail**

A Q-table for Atari has 210 x 160 x 3 x 256 possible pixel configurations per frame. Even with frame stacking (4 frames), the state space is astronomically large. You cannot visit every state-action pair. The solution: use a neural network to approximate $Q(s,a;\theta)$, generalizing from seen states to unseen states.

But naively training a neural network on RL data is unstable. Two problems:
1. **Correlated data**: consecutive transitions $(s_t, a_t, r_t, s_{t+1})$ are highly correlated. Neural networks trained on correlated data overfit to recent experience and forget old experience.
2. **Moving target**: the TD target $r + \gamma \max Q(s', a'; \theta)$ depends on the same parameters $\theta$ we are updating. This creates a feedback loop: updating $\theta$ changes the targets, which changes the loss, which changes the update direction. Training oscillates or diverges.

**The DQN Paper (Mnih et al., 2015)**

Two innovations that solved both problems:

*Experience Replay*: Store transitions $(s, a, r, s')$ in a replay buffer. Sample random mini-batches from the buffer for training. This breaks temporal correlation (non-consecutive transitions in each batch) and reuses data (each transition is used in multiple updates). DeepMind's original buffer stored 1 million transitions.

*Target Network*: Maintain a separate copy of the Q-network, $\theta_{\text{target}}$, that is updated less frequently (every $C$ steps, copy $\theta$ to $\theta_{\text{target}}$). Use $\theta_{\text{target}}$ to compute TD targets:

$$\mathcal{L}(\theta) = \mathbb{E}\!\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta_{\text{target}}) - Q(s, a; \theta)\right)^2\right]$$

The target is now stable for C steps, breaking the moving target problem.

**The DQN Loss**

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\!\left[\left(y - Q(s, a; \theta)\right)^2\right]$$

$$y = \begin{cases} r + \gamma \max_{a'} Q(s', a'; \theta_{\text{target}}) & \text{(non-terminal)} \\ r & \text{(terminal)} \end{cases}$$

This is a regression loss. The "label" $y$ is the TD target, computed with the frozen target network. Gradient descent on this loss moves $Q(s,a;\theta)$ toward the target. Note: we do NOT backpropagate through the target $y$ -- it is treated as a fixed target.

**Double DQN (van Hasselt et al., 2016)**

Standard DQN overestimates Q-values because $\max_{a'} Q(s', a')$ is a biased estimator -- maximization over noisy estimates introduces upward bias. Double DQN decouples action selection from evaluation:

$$a^* = \arg\max_{a'} Q(s', a'; \theta)$$

$$y = r + \gamma \, Q(s', a^*; \theta_{\text{target}})$$

This reduces overestimation and improves performance.

**Dueling DQN (Wang et al., 2016)**

Separate the Q-network into two streams:
- **Value stream**: $V(s)$ -- how good is this state?
- **Advantage stream**: $A(s,a)$ -- how much better is this action than average?

$$Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')$$

The subtraction of $\text{mean}_a \, A(s,a)$ ensures identifiability. This architecture learns faster in states where the action choice does not matter much (V dominates).

**Prioritized Experience Replay (Schaul et al., 2016)**

Not all transitions are equally informative. Transitions with high TD error (the agent was very wrong) should be replayed more often. Sample from the replay buffer with probability proportional to $|\text{TD error}|^\alpha$. Use importance sampling weights to correct the bias introduced by non-uniform sampling.

### Derivation Exercises

1. Derive the gradient of the DQN loss with respect to $\theta$. Show that it is: $-2(y - Q(s,a;\theta)) \nabla_\theta Q(s,a;\theta)$.
2. Show mathematically why max over noisy estimates introduces upward bias (Jensen's inequality argument).
3. Derive the importance sampling correction for Prioritized Experience Replay.

### Coding Tasks

1. Implement a basic DQN for CartPole-v1. First without replay and target network -- observe instability.
2. Add experience replay buffer. Add target network. Observe stabilization.
3. Implement Double DQN. Compare Q-value estimates.
4. Train on LunarLander-v2 and plot reward curves.

### Paper References

- Mnih et al., "Human-level control through deep reinforcement learning" (2015) -- the DQN paper, Nature
- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013) -- the DQN workshop paper (first version)
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016) -- Double DQN
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning" (2016) -- Dueling DQN
- Schaul et al., "Prioritized Experience Replay" (2016) -- PER

---

## Session 4: Policy Gradient Methods

**Duration**: 3.5 hours (1.5 hours lecture, 2 hours implementation)
**Date**: Week 18, Day 1

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Explain why value-based methods struggle with continuous action spaces.
2. Derive the policy gradient theorem (REINFORCE) from scratch.
3. Explain the log-probability trick and why it enables gradient estimation.
4. Implement baseline subtraction for variance reduction and explain why it does not change the expected gradient.
5. Implement Actor-Critic (A2C) and explain how it combines value-based and policy-based methods.
6. Describe Generalized Advantage Estimation (GAE) and the bias-variance tradeoff.

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 20 min | Why policy gradients: continuous actions, stochastic policies, direct optimization |
| 2 | 40 min | Whiteboard: policy gradient theorem derivation, the log-trick |
| 3 | 20 min | REINFORCE algorithm: implementation, high variance problem |
| 4 | 25 min | Baseline subtraction: why it reduces variance, why it does not introduce bias |
| 5 | 25 min | Actor-Critic: the actor proposes, the critic evaluates |
| 6 | 20 min | GAE: the bias-variance tradeoff in advantage estimation |
| 7 | 60 min | Implementation: REINFORCE and A2C on CartPole |

### Core Concepts

**Why Policy Gradients?**

DQN outputs $Q(s,a)$ for each discrete action, then picks the argmax. But what if the action space is continuous (e.g., the torque applied to a robotic joint)? You cannot enumerate all possible torques. Policy gradient methods directly parameterize the policy $\pi_\theta(a \mid s)$ -- e.g., a neural network that outputs the mean and standard deviation of a Gaussian distribution over actions. The agent samples actions from this distribution.

Additional advantages: policy gradient methods can learn stochastic policies (useful when the optimal strategy involves randomization, as in rock-paper-scissors) and are often more stable in practice.

**The Policy Gradient Theorem**

We want to maximize the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[R(\tau)\right] \quad \text{where } R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$$

The key insight -- the **log-probability trick**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[R(\tau) \, \nabla_\theta \log P(\tau \mid \theta)\right]$$

Since $P(\tau \mid \theta) = P(s_0) \prod_t \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$, the log decomposes and the transition probabilities cancel:

$$\nabla_\theta \log P(\tau \mid \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

Therefore:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t\right]$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return from time $t$ onward. This is the **REINFORCE** estimator.

Interpretation: increase the log-probability of actions that led to high returns, decrease it for actions that led to low returns. The gradient points in the direction that increases the probability of good trajectories.

**Baseline Subtraction**

REINFORCE has high variance because $G_t$ fluctuates wildly across episodes. A **baseline** $b(s_t)$ reduces variance without changing the expected gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left(G_t - b(s_t)\right)\right]$$

Why no bias? Because $\mathbb{E}[\nabla_\theta \log \pi(a \mid s) \, b(s)] = b(s) \, \mathbb{E}[\nabla_\theta \log \pi(a \mid s)] = b(s) \, \nabla_\theta \sum_a \pi(a \mid s) = b(s) \, \nabla_\theta 1 = 0$.

The optimal baseline (minimizing variance) is close to $V^\pi(s_t)$. In practice, we learn a value function $V_\phi(s)$ as the baseline.

**Actor-Critic (A2C)**

Combine a policy (the **actor**) with a value function (the **critic**):
- The **actor** $\pi_\theta(a \mid s)$ selects actions.
- The **critic** $V_\phi(s)$ evaluates states and provides the baseline.

The advantage $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$ measures how much better action $a_t$ is compared to the average. We estimate it using the TD error:

$$A_t = r_t + \gamma \, V_\phi(s_{t+1}) - V_\phi(s_t)$$

The actor update uses the advantage as the "reward signal." The critic is trained to minimize the value prediction error.

**Generalized Advantage Estimation (GAE)**

GAE (Schulman et al., 2016) provides a smooth interpolation between the one-step TD advantage (low variance, high bias) and the full Monte Carlo advantage (high variance, low bias):

$$A_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l} \quad \text{where } \delta_t = r_t + \gamma \, V(s_{t+1}) - V(s_t)$$

$\lambda=0$ gives the one-step TD estimate. $\lambda=1$ gives the MC estimate. Typical values: $\lambda=0.95$.

### Derivation Exercises

1. Derive the policy gradient theorem from scratch, starting from $J(\theta) = \mathbb{E}[R(\tau)]$.
2. Prove that the baseline $b(s)$ does not change the expected gradient (the "baseline is unbiased" proof).
3. Show that the advantage $A(s,a) = Q(s,a) - V(s)$ and that the one-step TD error is an unbiased estimate of the advantage.
4. Derive the GAE formula by expanding the geometric sum of TD errors.

### Coding Tasks

1. Implement REINFORCE on CartPole. Plot learning curve. Observe high variance.
2. Add a learned baseline (a simple value network). Compare variance of policy gradient estimates.
3. Implement A2C with separate actor and critic networks. Compare to REINFORCE.

### Paper References

- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992) -- REINFORCE
- Sutton et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation" (2000) -- policy gradient theorem
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016) -- GAE
- Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016) -- A3C (asynchronous actor-critic)

---

## Session 5: PPO and Modern Policy Optimization

**Duration**: 3.5 hours (1.5 hours lecture, 2 hours implementation)
**Date**: Week 18, Day 2

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Explain why large policy updates are dangerous and the concept of trust regions.
2. Describe TRPO's KL-divergence constraint and why it is impractical.
3. Derive the PPO clipped surrogate objective and explain every term.
4. Implement a complete PPO agent with parallel environments, GAE, and the full loss function.
5. Explain why PPO is the most widely used RL algorithm in practice.

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 25 min | The trust region problem: why large updates are catastrophic |
| 2 | 20 min | TRPO: the theory, the KL constraint, why it is expensive |
| 3 | 35 min | Whiteboard: PPO clipped objective derivation and analysis |
| 4 | 20 min | The complete PPO loss: clipped surrogate + value loss + entropy bonus |
| 5 | 20 min | Practical PPO: parallel environments, normalization, hyperparameters |
| 6 | 70 min | Implementation: PPO on CartPole and a continuous control task |

### Core Concepts

**The Trust Region Problem**

Policy gradient updates can be catastrophically large. A single bad gradient step can ruin a policy that took millions of steps to learn. Why? Because the policy gradient is a local approximation -- it tells you the direction of improvement at the current policy, but says nothing about how far you can go. In supervised learning, you have a fixed dataset that anchors your loss. In RL, a bad policy generates bad data, which generates a worse policy -- a death spiral.

**TRPO (Schulman et al., 2015)**

TRPO constrains the policy update so the new policy stays close to the old policy:

$$\max_\theta \; \mathbb{E}\!\left[r_t(\theta) A_t\right] \quad \text{subject to } \mathbb{E}\!\left[D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}})\right] \le \delta$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ is the probability ratio.

The KL constraint ensures the policy does not change too much. But computing the constrained optimization requires second-order methods (computing the Fisher information matrix), which is expensive and complex to implement.

**PPO: The Practical Solution (Schulman et al., 2017)**

PPO replaces the hard KL constraint with a clipped objective that achieves a similar effect with first-order optimization:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[\min\!\left(r_t A_t,\; \text{clip}(r_t,\, 1-\epsilon,\, 1+\epsilon) A_t\right)\right]$$

How this works:
- $r_t = \pi_{\text{new}}(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$: the probability ratio. $r_t = 1$ means the new and old policies agree.
- $A_t$: the advantage estimate (from GAE).
- $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$: clips the ratio to the interval $[1-\epsilon,\, 1+\epsilon]$, typically $\epsilon=0.2$.
- $\min(\cdots)$: takes the pessimistic (lower) bound.

Analysis by case:
- If $A_t > 0$ (good action): we want to increase $r_t$ (make the action more likely). But the clip prevents $r_t$ from exceeding $1+\epsilon$. This stops the policy from changing too much.
- If $A_t < 0$ (bad action): we want to decrease $r_t$ (make the action less likely). But the clip prevents $r_t$ from going below $1-\epsilon$. Again, bounded change.

The min ensures we always take the more conservative estimate.

**The Complete PPO Loss**

$$\mathcal{L}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\theta) + c_2 H[\pi_\theta]$$

Three terms:
- $\mathcal{L}^{\text{CLIP}}$: the clipped surrogate objective (maximize).
- $\mathcal{L}^{\text{VF}}$: the value function loss, typically MSE between $V_\phi(s)$ and the actual return (minimize).
- $H[\pi_\theta]$: entropy bonus, encourages exploration by penalizing overly deterministic policies (maximize). This prevents premature convergence to a suboptimal deterministic policy.

**Practical PPO Details**

- **Parallel environments**: run N environments simultaneously to collect diverse experience. Use `gym.vector.SyncVectorEnv` or `AsyncVectorEnv`.
- **Advantage normalization**: normalize advantages across the batch to have zero mean and unit variance. This stabilizes training significantly.
- **Multiple epochs per batch**: unlike on-policy methods that use data once, PPO reuses each batch for K epochs (typically 3-10). The clipping prevents the policy from changing too much during these epochs.
- **Gradient clipping**: clip the gradient norm (typically max_norm=0.5).
- **Learning rate annealing**: linearly decay the learning rate over training.

### Derivation Exercises

1. Derive the PPO clipped objective from the TRPO surrogate objective. Show that clipping the ratio is a conservative approximation to the KL constraint.
2. For a Gaussian policy $\pi(a \mid s) = \mathcal{N}(\mu(s), \sigma^2)$, derive $r_t$ in closed form.
3. Show that when $\epsilon=0$, PPO reduces to no update ($r_t$ is clipped to exactly 1).

### Coding Tasks

1. Implement the PPO clipped surrogate loss in PyTorch.
2. Implement a full PPO training loop: parallel environments, GAE, mini-batch updates, entropy bonus.
3. Train on CartPole-v1, then LunarLander-v2 (discrete).
4. (Stretch) Train on Pendulum-v1 or MountainCarContinuous-v0 (continuous action space).

### Paper References

- Schulman et al., "Trust Region Policy Optimization" (2015) -- TRPO
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) -- PPO
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016) -- GAE (companion to PPO)
- Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO" (2020) -- the details that make PPO work
- Huang et al., "The 37 Implementation Details of Proximal Policy Optimization" (2022) -- a thorough analysis of PPO implementation details

---

## Session 6: RLHF and the LLM Connection

**Duration**: 3.5 hours (1.5 hours lecture, 2 hours discussion and exercises)
**Date**: Week 18, Day 3

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Describe the full RLHF pipeline: SFT, reward modeling, PPO fine-tuning.
2. Explain the Bradley-Terry model for learning from human preferences.
3. Derive the RLHF objective with KL penalty and explain each term.
4. Explain Direct Preference Optimization (DPO) as a simplification of RLHF.
5. Describe Monte Carlo Tree Search and how AlphaGo/AlphaProof combine it with neural networks.
6. Articulate how RL connects to every major DeepMind system.

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 25 min | The alignment problem: why supervised fine-tuning is not enough |
| 2 | 30 min | The RLHF pipeline: SFT, reward modeling, PPO optimization |
| 3 | 25 min | Whiteboard: Bradley-Terry model, reward model loss, KL-penalized objective |
| 4 | 20 min | DPO: eliminating the reward model, the DPO loss derivation |
| 5 | 15 min | Constitutional AI: self-critique and the future of alignment |
| 6 | 30 min | Monte Carlo Tree Search: selection, expansion, simulation, backpropagation |
| 7 | 25 min | The DeepMind synthesis: AlphaGo, AlphaProof, Gemini, SIMA |

### Core Concepts

**Why RLHF?**

Supervised fine-tuning (SFT) teaches a language model to follow instructions by training on (instruction, response) pairs. But SFT has a fundamental limitation: it treats all correct-looking responses equally and cannot capture nuanced human preferences. Is a concise answer better than a detailed one? Should the model be more cautious or more direct? These are not questions with single correct answers -- they are preference questions. RLHF lets us optimize for human preferences directly.

**The RLHF Pipeline**

Step 1: **Supervised Fine-Tuning (SFT)**. Start with a pretrained LLM. Fine-tune on high-quality (prompt, response) pairs to create a model that can follow instructions. This is the starting point for RLHF -- the SFT model is $\pi_{\text{ref}}$.

Step 2: **Reward Modeling**. Collect human preference data: for each prompt $x$, show two responses $(y_1, y_2)$ to a human annotator who indicates which is better. Train a reward model $R_\phi(x, y)$ using the Bradley-Terry model:

$$P(y_1 \succ y_2 \mid x) = \sigma\!\left(R_\phi(x, y_1) - R_\phi(x, y_2)\right)$$

Loss:

$$\mathcal{L}(\phi) = -\mathbb{E}\!\left[\log \sigma\!\left(R_\phi(x, y_w) - R_\phi(x, y_l)\right)\right]$$

where $y_w$ is the preferred (winning) response and $y_l$ is the dispreferred (losing) response.

Step 3: **PPO Fine-Tuning**. The language model is the policy. It generates responses (sequences of token actions) given prompts (states). The reward model scores the complete generation. Optimize:

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\!\left[R_\phi(x, y)\right] - \beta \, D_{\text{KL}}\!\left[\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\right]$$

The KL penalty keeps the fine-tuned model close to the SFT model, preventing reward hacking (the model finding degenerate outputs that exploit the reward model but are not actually good).

**Direct Preference Optimization (DPO)**

Rafailov et al. (2023) showed that the RLHF objective has a closed-form optimal solution, which allows training directly on preference data without a separate reward model:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \left(\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right)\right]$$

DPO is simpler (no reward model, no PPO loop), more stable, and often competitive with PPO-based RLHF. It has become the preferred approach for many teams.

**Constitutional AI**

Anthropic's approach: instead of collecting human preference data, generate critiques and revisions using the model itself, guided by a set of principles (a "constitution"). The model critiques its own outputs, revises them, and the revisions are used as preference data. This reduces the need for expensive human annotation.

**Monte Carlo Tree Search (MCTS)**

MCTS builds a search tree incrementally through four phases:
1. **Selection**: starting from the root, traverse the tree using a selection policy (UCB1) that balances exploration and exploitation.
2. **Expansion**: when a leaf node is reached, add one or more child nodes.
3. **Simulation (rollout)**: from the new node, simulate a random game to completion.
4. **Backpropagation**: propagate the result back up the tree, updating visit counts and value estimates.

The UCB formula for tree traversal:

$$\text{UCB}(s, a) = Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

**AlphaGo** replaced the simulation phase with a value network ($V_\theta(s)$ estimates the probability of winning from state $s$) and used a policy network ($\pi_\theta(a \mid s)$ estimates the probability of good moves) to guide selection. This combination of neural networks with MCTS was the key innovation.

**AlphaProof** applies the same paradigm to mathematical theorem proving: a language model proposes proof steps, MCTS searches the proof tree, and a value network evaluates partial proofs. This is RL applied to reasoning.

### Derivation Exercises

1. Derive the Bradley-Terry model from the assumption that response quality follows a latent score plus Gumbel noise.
2. Derive the DPO loss from the RLHF objective by substituting the closed-form optimal policy.
3. Show that the KL penalty in RLHF is equivalent to an entropy-regularized reward: $\text{effective reward} = R(x,y) - \beta \log(\pi_\theta(y \mid x) / \pi_{\text{ref}}(y \mid x))$.

### Coding Tasks

1. Implement a reward model training loop on synthetic preference data.
2. Implement the DPO loss function in PyTorch.
3. (Conceptual) Design the PPO training loop for a language model: what is the state, action, reward, episode?

### Paper References

- Ouyang et al., "Training language models to follow instructions with human feedback" (2022) -- InstructGPT, the RLHF paper
- Christiano et al., "Deep Reinforcement Learning from Human Preferences" (2017) -- original RLHF for RL agents
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023) -- DPO
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022) -- Constitutional AI
- Silver et al., "Mastering the game of Go with deep neural networks and tree search" (2016) -- AlphaGo
- Silver et al., "Mastering the game of Go without human knowledge" (2017) -- AlphaGo Zero
- Trinh et al., "Solving olympiad geometry without human demonstrations" (2024) -- AlphaGeometry
- DeepMind, "AlphaProof and AlphaGeometry 2" (2024) -- mathematical reasoning with RL
