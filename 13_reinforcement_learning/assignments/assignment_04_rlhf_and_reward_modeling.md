# Assignment 4: RLHF and Reward Modeling

## Overview

In this assignment, you will build the complete Reinforcement Learning from Human Feedback (RLHF) pipeline -- the technology that transforms a pretrained language model into an aligned assistant. You will train a reward model on preference data, use PPO to optimize a language model against that reward model, and implement DPO as a simpler alternative. By the end, you will understand at a practical level how Gemini, ChatGPT, and Claude are aligned with human preferences.

This is where the entire module converges. The Bellman equation, Q-Learning, policy gradients, PPO -- everything you have learned -- comes together to solve the most important applied problem in modern AI: making language models behave the way humans want.

**Estimated time:** 18-24 hours

**Prerequisites:** Module 13 Sessions 1-6, Module 7 (Transformers), PyTorch fluency, a working PPO implementation from Assignment 3.

**Compute requirements:** A GPU is strongly recommended. CPU training is possible with a small model (GPT-2 small, ~124M parameters) but will be slow.

---

## Part 1: The Base Language Model

### 1.1 Setup

Load a pretrained GPT-2 small model using Hugging Face Transformers:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # 124M parameters
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# The SFT/reference model (frozen after loading)
ref_model = GPT2LMHeadModel.from_pretrained(model_name)
ref_model.eval()

# The policy model (will be fine-tuned with RLHF)
policy_model = GPT2LMHeadModel.from_pretrained(model_name)
```

### 1.2 Baseline Generation

Generate responses from the base model for a set of test prompts:

```python
test_prompts = [
    "The best way to learn programming is",
    "In a healthy relationship, it is important to",
    "The meaning of life is",
    "When faced with a difficult problem, you should",
    "A good leader always",
]
```

Generate 3 completions per prompt (using sampling with temperature=0.7, max_length=100). Save these as baseline generations for later comparison.

### 1.3 Understanding the Model as a Policy

Write a function that computes the log-probability of a response under a given model:

```python
def compute_log_probs(model, input_ids, attention_mask=None):
    """Compute log P(response | prompt) under the model.

    Args:
        model: a GPT-2 model
        input_ids: token IDs for [prompt + response]
        attention_mask: attention mask

    Returns:
        log_probs: per-token log probabilities (only for response tokens)
        total_log_prob: sum of per-token log probabilities
    """
    ...
```

This is the bridge between language modeling and RL: the log-probability of the generated tokens under the policy is the quantity that REINFORCE and PPO optimize.

---

## Part 2: Synthetic Preference Dataset

### 2.1 Design the Preference Rule

Instead of collecting real human preferences (expensive and slow), create a synthetic preference function that simulates human preferences. Choose one or more of the following rules:

**Option A -- Brevity preference**: shorter responses are preferred (simulating a preference for conciseness).

```python
def synthetic_preference(response_1, response_2):
    """Return 1 if response_1 is preferred, 0 if response_2 is preferred."""
    # Prefer shorter responses
    if len(response_1.split()) < len(response_2.split()):
        return 1
    elif len(response_1.split()) > len(response_2.split()):
        return 0
    else:
        return random.choice([0, 1])
```

**Option B -- Positive sentiment preference**: responses with more positive words are preferred.

**Option C -- Helpfulness heuristic**: responses that contain specific helpful patterns (numbered lists, "Here are", "First,") are preferred over vague responses.

Choose one option and document your choice. The specific rule matters less than having a consistent, evaluable preference signal.

### 2.2 Generate the Dataset

Generate the preference dataset:

```python
def generate_preference_dataset(model, tokenizer, prompts, n_pairs_per_prompt=20):
    """Generate preference pairs for training the reward model.

    For each prompt:
        1. Generate two responses from the model.
        2. Apply the synthetic preference function to determine which is preferred.
        3. Store (prompt, chosen_response, rejected_response).

    Returns:
        dataset: list of dicts with keys "prompt", "chosen", "rejected"
    """
    ...
```

**Requirements:**
- Use at least 50 diverse prompts (you can use a mix of hand-written prompts and prompts from a dataset like Anthropic HH-RLHF or OpenAssistant).
- Generate at least 20 preference pairs per prompt (1000+ total pairs).
- Save the dataset to disk (JSON format).
- Report statistics: average response length for chosen vs rejected, agreement rate with the preference function.

---

## Part 3: Train the Reward Model

### 3.1 Reward Model Architecture

Implement a reward model based on GPT-2:

```python
class RewardModel(nn.Module):
    def __init__(self, model_name="gpt2"):
        """A reward model that outputs a scalar reward for (prompt, response) pairs.

        Architecture:
        - GPT-2 backbone (can be initialized from pretrained weights)
        - Linear head on top of the last token's hidden state -> scalar reward
        """
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.backbone.config.n_embd, 1)
        ...

    def forward(self, input_ids, attention_mask=None):
        """Compute reward for a (prompt + response) input.

        Returns: reward (scalar per batch element)
        """
        ...
```

### 3.2 Bradley-Terry Training

Train the reward model using the Bradley-Terry preference loss:

```python
def train_reward_model(reward_model, dataset, n_epochs=3, batch_size=8, lr=1e-5):
    """Train the reward model on preference data.

    Loss: L = -E[log sigma(R(x, y_chosen) - R(x, y_rejected))]

    This is binary cross-entropy: the reward model should assign
    higher reward to the chosen response than the rejected response.
    """
    ...
```

**Requirements:**
- Train for 3-5 epochs.
- Track and plot: training loss, validation loss (hold out 20% of data), accuracy (% of pairs where R(chosen) > R(rejected)).
- The accuracy should reach 80%+ on the validation set.
- After training, compute rewards for the test prompts from Part 1. Do the reward values align with your preference function? Print a few examples with their rewards.

### 3.3 Reward Model Analysis

- Generate 10 responses of varying quality for a single prompt. Score them with the reward model. Rank by reward score. Does the ranking match your preference function?
- Plot the distribution of rewards for chosen vs rejected responses. The chosen distribution should be shifted higher.
- Identify failure cases: examples where the reward model disagrees with the preference function. Why might this happen?

---

## Part 4: PPO Fine-Tuning

### 4.1 The RLHF Training Loop

Implement PPO for language model fine-tuning:

```python
class RLHFTrainer:
    def __init__(self, policy_model, ref_model, reward_model, tokenizer,
                 lr=1e-6, kl_coef=0.1, clip_eps=0.2, n_epochs=4,
                 batch_size=8, max_gen_length=100):
        """
        policy_model: the LM being fine-tuned (pi_theta)
        ref_model: the frozen reference model (pi_ref)
        reward_model: the trained reward model (R_phi)
        tokenizer: the tokenizer
        kl_coef: beta -- weight of the KL penalty
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        ...

    def generate_responses(self, prompts):
        """Generate responses from the policy model.

        Use sampling (not greedy) to maintain exploration.
        Return: generated token IDs, attention masks.
        """
        ...

    def compute_rewards(self, prompt_ids, response_ids):
        """Compute the RLHF reward for each generation.

        reward = R_phi(prompt, response) - beta * KL(pi_theta || pi_ref)

        The KL penalty can be computed per-token:
        KL_t = log pi_theta(a_t | s_t) - log pi_ref(a_t | s_t)
        """
        ...

    def ppo_update(self, states, actions, old_log_probs, rewards, advantages):
        """PPO update step.

        Compute ratio r_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t).
        Compute clipped surrogate loss.
        Update policy model parameters.
        """
        ...

    def train_step(self, prompts):
        """One step of RLHF training.

        1. Generate responses from the current policy.
        2. Compute rewards (reward model score - KL penalty).
        3. Compute advantages (GAE or simple return-baseline).
        4. PPO update.
        """
        ...

    def train(self, prompts, n_steps=200):
        """Full RLHF training loop."""
        ...
```

### 4.2 Training

Train the policy model with RLHF:
- Use at least 200 training steps.
- At each step, sample a batch of prompts and generate responses.
- Track and plot:
  - Mean reward per step
  - Mean KL divergence per step
  - Reward model score (without KL penalty)
  - Sample generations at steps 0, 50, 100, 150, 200

**Important details:**
- Use a low learning rate (1e-6 to 5e-6). Language models are sensitive to large updates.
- The KL coefficient ($\beta$) is critical. Start with $\beta=0.1$ and adjust.
- Generate with temperature=0.7 or nucleus sampling (top_p=0.9).
- Freeze the reference model and reward model. Only the policy model is updated.

### 4.3 KL Penalty Sweep

Train three models with different KL coefficients:
- $\beta = 0.01$ (weak penalty -- the model can deviate far from the reference)
- $\beta = 0.1$ (moderate penalty)
- $\beta = 1.0$ (strong penalty -- the model stays very close to the reference)

For each, compare:
- Final reward
- Final KL divergence
- Sample generations
- Diversity of outputs (measure with distinct n-grams or self-BLEU)

**Expected behavior:**
- Low beta: high reward but potentially degenerate outputs (reward hacking). The model may learn to produce a narrow set of high-reward responses.
- High beta: outputs stay close to the base model. Reward improvement is limited.
- Medium beta: the sweet spot -- meaningful reward improvement while maintaining output quality.

---

## Part 5: Compare Base vs SFT vs RLHF

### 5.1 Qualitative Comparison

For each of the test prompts from Part 1, generate 3 responses from:
1. The base GPT-2 model
2. The reference (SFT) model (same as base in this case, unless you did additional SFT)
3. The RLHF-tuned model

Present the responses side-by-side. Annotate which responses best match the preference function.

### 5.2 Quantitative Comparison

Compute the following metrics for 100 generations from each model:
- Mean reward model score
- Mean response length (in tokens)
- Preference function agreement rate (% of responses that match the desired preference)
- Perplexity under the base model (measures how "natural" the text is)

Present in a table:

| Metric | Base GPT-2 | RLHF-tuned ($\beta$=0.01) | RLHF-tuned ($\beta$=0.1) | RLHF-tuned ($\beta$=1.0) |
|--------|-----------|------------------------|------------------------|------------------------|
| Reward Score | | | | |
| Response Length | | | | |
| Preference Agreement | | | | |
| Base Perplexity | | | | |

---

## Part 6: Direct Preference Optimization (DPO)

### 6.1 DPO Implementation

Implement DPO as an alternative to PPO-based RLHF:

```python
def dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids,
             chosen_mask, rejected_mask, beta=0.1):
    """DPO loss: train directly on preference data without a reward model.

    L_DPO = -E[log sigma(beta * (log(pi/pi_ref)(y_w) - log(pi/pi_ref)(y_l)))]

    Args:
        policy_model: the model being trained
        ref_model: the frozen reference model
        chosen_ids: token IDs for the preferred response
        rejected_ids: token IDs for the dispreferred response
        beta: temperature parameter

    Returns:
        loss: scalar DPO loss
        accuracy: fraction where implicit reward of chosen > rejected
        reward_margin: mean difference in implicit rewards
    """
    ...


def train_dpo(policy_model, ref_model, preference_dataset, tokenizer,
              n_epochs=3, batch_size=4, lr=5e-7, beta=0.1):
    """Train a language model with DPO.

    Much simpler than RLHF: no reward model, no PPO loop.
    Just a supervised loss on preference pairs.
    """
    ...
```

### 6.2 DPO Training

Train a DPO model on the same preference dataset used for the reward model:
- Train for 3 epochs.
- Track: DPO loss, accuracy, reward margin, KL divergence from the reference model.
- Generate sample outputs at each epoch.

### 6.3 PPO-RLHF vs DPO Comparison

Compare the PPO-RLHF model (from Part 4) with the DPO model:

| Metric | PPO-RLHF | DPO |
|--------|----------|-----|
| Reward Score | | |
| Preference Agreement | | |
| Training Time (wall-clock) | | |
| Number of Models Trained | 3 (RM + policy + ref) | 2 (policy + ref) |
| Implementation Complexity | Higher | Lower |
| KL from Reference | | |

Write a 200-300 word comparison discussing the practical tradeoffs between PPO-RLHF and DPO.

---

## Part 7: Analysis

### 7.1 Reward Hacking Investigation

With the lowest KL penalty ($\beta=0.01$), investigate reward hacking:
- Examine the highest-reward generations. Do they look good to a human, or has the model found degenerate patterns?
- Compute the perplexity of high-reward generations under the base model. Very high perplexity (very unnatural text) with high reward indicates hacking.
- If you observe reward hacking, describe the specific pattern the model exploits.

### 7.2 KL-Reward Frontier

Plot the **KL-Reward frontier**: for each KL coefficient ($\beta$=0.01, 0.05, 0.1, 0.2, 0.5, 1.0), plot a point at (final KL, final reward). Connect the points. This curve shows the fundamental tradeoff: more KL divergence buys more reward, but at the cost of moving further from the base model.

### 7.3 Written Analysis

Write a 500-700 word analysis addressing:
1. How does the KL penalty affect output quality? What happens with too little or too much?
2. Did you observe reward hacking? If so, what pattern did the model exploit?
3. How does DPO compare to PPO-RLHF in your experiments? When might you prefer one over the other?
4. What are the limitations of using a synthetic preference function? How would real human preferences differ?
5. Reflect on the RLHF pipeline as a whole. What are the potential failure modes? Where could things go wrong in a production system?

---

## Deliverables

1. **Code**: complete implementations of the reward model, PPO-RLHF training loop, and DPO loss.
2. **Preference dataset**: the generated preference dataset (saved as JSON).
3. **Plots**: reward model training curves, RLHF training curves (reward, KL, sample generations), KL-Reward frontier, comparison plots.
4. **Comparison tables**: base vs RLHF vs DPO with quantitative metrics.
5. **Sample generations**: side-by-side comparisons of base, RLHF, and DPO model outputs.
6. **Written analysis**: 500-700 words covering the questions in Part 7.3.

## Evaluation Criteria

- **Reward model** (15%): correctly trained, achieves >80% validation accuracy, produces sensible reward rankings.
- **PPO-RLHF** (25%): correct implementation, reward increases over training, KL penalty works as expected, no training collapse.
- **DPO** (15%): correct implementation, loss decreases, model improves on preference metric.
- **KL analysis** (15%): clear demonstration of the KL-reward tradeoff with multiple beta values, reward hacking investigation.
- **Comparison and analysis** (20%): thorough comparison of PPO vs DPO, base vs RLHF, quantitative metrics and qualitative examples.
- **Code quality** (10%): clean, documented, modular code.

## Stretch Goals

1. **Real Human Preferences**: use a small set of real human preferences instead of synthetic ones. Recruit 2-3 friends or colleagues to annotate 100 preference pairs. Train a reward model on real preferences and compare to the synthetic version. How well do synthetic preferences approximate real ones?

2. **Constitutional AI Self-Critique**: implement a simplified version of Constitutional AI:
   - Generate a response.
   - Ask the model to critique its own response (given a principle like "Be helpful and harmless").
   - Ask the model to revise its response based on the critique.
   - Use the (original, revised) pair as a preference pair for DPO training.
   This eliminates the need for human annotation entirely.

3. **Scaling to a Larger Model**: repeat the experiments with GPT-2 Medium (345M parameters) instead of GPT-2 Small. Does RLHF work better with a larger base model? Is the reward model more or less reliable?

4. **Multi-Objective RLHF**: train two reward models (e.g., one for helpfulness and one for brevity). Optimize the policy against a weighted combination of both rewards. Explore the Pareto frontier by varying the weights.
