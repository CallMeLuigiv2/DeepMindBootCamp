# Module 11: Reading Research Papers — Comprehensive Notes

> "The ability to pick up an unfamiliar paper on Monday, understand it by Wednesday, and
> implement it by Friday is the single most valuable skill at DeepMind. Everything else
> can be taught. This cannot be faked."

---

## Table of Contents

1. [The Three-Pass Method — Detailed](#the-three-pass-method)
2. [Paper Reading Template](#paper-reading-template)
3. [Common Mathematical Notation in ML Papers](#common-mathematical-notation)
4. [Landmark Papers Reading List](#landmark-papers-reading-list)
5. [Paper-to-Code Translation Guide](#paper-to-code-translation-guide)
6. [How to Write a Paper Review](#how-to-write-a-paper-review)

---

## The Three-Pass Method

### First Pass (5-10 minutes) — Triage

**Goal:** Decide if this paper is worth your time.

**Steps:**
1. Read the title. What field is this? What is the claimed contribution?
2. Read the abstract. Extract: problem, method (one line), result (one line).
3. Read all section headings. Map the structure of the argument.
4. Look at every figure and table. Read every caption. Figures contain the paper's real
   substance. A paper with bad figures is usually a bad paper.
5. Read the conclusion. What do the authors claim they have accomplished?
6. Scan the references. Do you recognize key papers? Are important works cited?

**Questions to answer after Pass 1:**
- [ ] What is this paper about? (one sentence)
- [ ] What type of paper is this? (new method / empirical study / theory / survey / position)
- [ ] What is the claimed contribution?
- [ ] Is this relevant to my current work or interests?
- [ ] Do the results look significant from the figures/tables?
- [ ] Should I do a second pass? (Yes / No — justify)

**Decision framework for continuing:**
- If the paper is directly relevant to your project: always do Pass 2.
- If the paper is in your area but not directly relevant: do Pass 2 if figures look
  interesting or if the paper is from a top venue.
- If the paper is outside your area: only do Pass 2 if the core idea seems transferable.
- If you cannot understand the abstract: you may lack prerequisite knowledge. Find a survey
  paper in that area first.

### Second Pass (45-90 minutes) — Understanding

**Goal:** Understand the paper's arguments and contributions without mastering every detail.

**Steps:**
1. Read the introduction carefully. Identify:
   - The problem statement (what gap in knowledge or capability exists)
   - The positioning (how this relates to prior work)
   - The contribution claims (usually a bulleted list)
   - The key insight (the one sentence that captures the idea)

2. Read the method section with the following strategy:
   - First read: text only, skip equations. Get the intuition.
   - Second read: text and equations together. For each equation, note what it computes.
   - Build a notation table as you go.
   - Draw the computational pipeline.
   - Identify the loss function — this reveals what the model optimizes.

3. Read the experiments section critically:
   - What datasets are used? Are they standard?
   - What baselines are compared? Are they recent and relevant?
   - What metrics are reported? Are they standard for this task?
   - Look at tables: where does the proposed method win? Where does it lose?
   - Look for ablation tables: what happens when you remove each component?
   - Are error bars or confidence intervals reported?

4. Read the related work:
   - This is your map of the field. Mark papers to follow up on.
   - Identify the two or three most relevant prior works.
   - Note how this paper differentiates itself from each.

5. Note things you do not understand:
   - Circle or highlight specific passages.
   - Write margin notes: "Why?", "How?", "Contradicts X?"
   - These become your agenda for Pass 3 or for discussion with others.

**Questions to answer after Pass 2:**
- [ ] Can I summarize this paper to a colleague? (Try it out loud.)
- [ ] What is the main contribution? (state it precisely)
- [ ] What are the 2-3 key technical components of the method?
- [ ] Which experiment is most convincing? Which is weakest?
- [ ] What are the paper's assumptions? Are they reasonable?
- [ ] What are 3 strengths and 3 weaknesses?
- [ ] What do I not understand? (Be specific.)
- [ ] What 2-3 papers from the references should I read next?

### Third Pass (4+ hours) — Mastery

**Goal:** Understand the paper so deeply that you could re-create it from scratch.

**Steps:**
1. Re-derive every key equation from first principles.
   - Start from the stated assumptions.
   - Derive each result step by step.
   - If a step says "it follows that..." — prove it.
   - If you cannot derive a result, either you are missing something or the paper is wrong.
     Both happen. Check cited references for the missing step.

2. Challenge every assumption:
   - "Why did they use this particular loss function?"
   - "What happens if this assumption is violated?"
   - "Why this architecture and not a simpler one?"
   - "Is this hyperparameter choice justified or arbitrary?"

3. Mentally implement the method:
   - Walk through the algorithm step by step with a concrete example.
   - What are the tensor shapes at each stage?
   - Where are the computational bottlenecks?
   - What numerical issues might arise?

4. Identify what the paper does NOT say:
   - What experiments are missing?
   - What failure modes are not discussed?
   - What happens at scale (larger or smaller)?
   - What are the computational costs?
   - What societal implications are not addressed?

5. Connect to other work:
   - How does this relate to papers you have already read?
   - Does it contradict anything you know?
   - Does it suggest new approaches to problems you care about?
   - What would the next paper in this line of work look like?

**Questions to answer after Pass 3:**
- [ ] Can I reconstruct the key algorithm from memory?
- [ ] Can I identify the most critical assumptions and what happens when they fail?
- [ ] Can I propose a concrete improvement or extension?
- [ ] Can I implement this method from scratch?
- [ ] Do I understand why the authors made each design choice?
- [ ] Can I explain this paper's contribution in the context of the field?

---

## Paper Reading Template

Use this template for every paper you read past the first pass. A separate, standalone version
is provided in `paper_reading_template.md`.

```
================================================================
PAPER READING TEMPLATE
================================================================

CITATION
--------
Title:
Authors:
Year:
Venue (journal/conference):
URL/DOI:
Date read:

ONE-SENTENCE SUMMARY
--------------------
[Summarize the entire paper in one sentence. If you cannot do this, you do not
understand the paper yet.]

PROBLEM
-------
[What problem does this paper address? Why does it matter? Who cares?]

KEY INSIGHT
-----------
[What is the single core idea that makes this work? One paragraph maximum.
This should be the thing that, if you only remembered one thing about this
paper, would capture its essence.]

METHOD
------
[How does the approach work? Include:]
- High-level description (2-3 sentences)
- Key equations (write them out, explain each symbol)
- Architecture diagram (sketch or describe)
- Loss function and what it optimizes
- Key design choices and their justification

TRAINING DETAILS
----------------
- Dataset(s):
- Preprocessing:
- Architecture specifics:
- Optimizer and learning rate:
- Batch size:
- Training duration (epochs/steps):
- Compute used (GPUs, time):
- Key hyperparameters:

RESULTS
-------
- Main findings:
- Comparison to baselines (which baselines, by how much):
- Ablation results (what matters, what doesn't):
- Key figures/tables to remember:

STRENGTHS (at least 3)
----------------------
1.
2.
3.

WEAKNESSES (at least 3)
-----------------------
1.
2.
3.

QUESTIONS
---------
[Things you don't understand or want to discuss:]
1.
2.
3.

CONNECTIONS
-----------
[How does this relate to other papers you've read?]
- Builds on:
- Competes with:
- Inspired:
- Contradicts:

IMPLEMENTATION NOTES
--------------------
[Key details you would need to implement this:]
- Non-obvious implementation details:
- Potential numerical issues:
- Things the paper doesn't specify:
- Estimated implementation effort:

RATING: __ / 10
----------------
[Justify your rating in 2-3 sentences. Consider: novelty, rigor,
significance, clarity, reproducibility.]
================================================================
```

---

## Common Mathematical Notation in ML Papers

This reference table covers the notation you will encounter in 90% of ML papers. When a paper
uses non-standard notation, build a custom table for that paper.

### Parameters and Variables

| Symbol | Name | Meaning | PyTorch Equivalent |
|--------|------|---------|-------------------|
| $\theta$ | theta | Model parameters (all weights and biases) | `model.parameters()` |
| $W$ | weight matrix | Linear transformation weights | `nn.Linear(in, out).weight` |
| $b$ | bias vector | Additive bias term | `nn.Linear(in, out).bias` |
| $x$ | input | Input data or features | `x` (tensor) |
| $y$ | target/label | Ground truth labels | `y` (tensor) |
| $\hat{y}$, $f(x)$ | prediction | Model output | `model(x)` |
| $h$ | hidden state | Hidden representation | intermediate tensor |
| $z$ | latent variable | Latent representation (in generative models) | `z = encoder(x)` |
| $a$ | activation | Pre or post activation value | intermediate tensor |
| $d$ | dimension | Dimensionality of a vector or space | `x.shape[-1]` |
| $N$, $n$ | count | Number of samples, features, etc. | `len(dataset)` |
| $B$ | batch size | Number of samples per batch | `batch_size` |
| $T$ | sequence length | Number of time steps | `seq_len` |

### Loss Functions and Optimization

| Symbol | Name | Meaning | PyTorch Equivalent |
|--------|------|---------|-------------------|
| $\mathcal{L}$ | loss | Loss function value | `loss = criterion(y_hat, y)` |
| $J(\theta)$ | cost function | Average loss over dataset | `loss.mean()` |
| $\nabla$ | nabla/del | Gradient operator | `loss.backward()` |
| $\nabla_\theta \mathcal{L}$ | gradient of $\mathcal{L}$ w.r.t. $\theta$ | Parameter gradients | `param.grad` |
| $\frac{\partial \mathcal{L}}{\partial w}$ | partial derivative | Gradient w.r.t. specific parameter | `w.grad` |
| $\eta$, $\alpha$ | learning rate | Step size for gradient descent | `lr` in optimizer |
| $\lambda$ | regularization coeff. | Weight decay or penalty strength | `weight_decay` in optimizer |
| $\arg\min_\theta$ | argmin | Value of $\theta$ that minimizes | `optimizer.step()` (iterative) |
| $\arg\max_\theta$ | argmax | Value of $\theta$ that maximizes | same, with negated loss |

### Probability and Statistics

| Symbol | Name | Meaning | PyTorch Equivalent |
|--------|------|---------|-------------------|
| $P(x)$, $p(x)$ | probability | Probability of $x$ | `torch.distributions` |
| $P(y \mid x)$ | conditional prob. | Probability of $y$ given $x$ | `model(x)` (after softmax) |
| $\mathbb{E}[x]$, $\mathbb{E}_p[x]$ | expectation | Expected value of $x$ under $p$ | `x.mean()` (empirical) |
| $\text{Var}(x)$ | variance | Variance of $x$ | `x.var()` |
| $\sigma$ | sigma | Standard deviation or sigmoid | `x.std()` or `torch.sigmoid(x)` |
| $\mu$ | mu | Mean | `x.mean()` |
| $\mathcal{N}(\mu, \sigma^2)$ | normal dist. | Gaussian distribution | `torch.randn(shape) * sigma + mu` |
| $D_{\text{KL}}(p \| q)$ | KL divergence | Divergence between distributions | `F.kl_div(q.log(), p)` |
| $H(p)$ | entropy | Shannon entropy | `-(p * p.log()).sum()` |
| $\log p(x)$ | log-likelihood | Log of probability | `dist.log_prob(x)` |
| $\prod$ | product | Product over terms | `torch.prod(x)` |
| $\sum$ | summation | Sum over terms | `torch.sum(x)` or `x.sum()` |

### Linear Algebra

| Symbol | Name | Meaning | PyTorch Equivalent |
|--------|------|---------|-------------------|
| $\|x\|$, $\|x\|_2$ | L2 norm | Euclidean norm | `torch.norm(x)` or `x.norm()` |
| $\|x\|_1$ | L1 norm | Sum of absolute values | `x.abs().sum()` |
| $\|x\|_F$ | Frobenius norm | Matrix norm | `torch.norm(x, 'fro')` |
| $x^T$ | transpose | Transpose of $x$ | `x.T` or `x.transpose(-2,-1)` |
| $x \cdot y$, $x^T y$ | dot product | Inner product | `torch.dot(x, y)` or `x @ y` |
| $X W$ | matrix multiply | Matrix multiplication | `torch.matmul(X, W)` or `X @ W` |
| $\text{diag}(x)$ | diagonal | Diagonal matrix from vector | `torch.diag(x)` |
| $I$ | identity | Identity matrix | `torch.eye(n)` |
| $\det(A)$ | determinant | Matrix determinant | `torch.det(A)` |
| $\text{tr}(A)$ | trace | Sum of diagonal elements | `torch.trace(A)` |
| $x \odot y$ | element-wise product | Hadamard product | `x * y` |
| $x \oplus y$ | element-wise sum | (less common) element-wise add | `x + y` |

### Common Functions

| Symbol | Name | Meaning | PyTorch Equivalent |
|--------|------|---------|-------------------|
| $\sigma(x)$ | sigmoid | $\frac{1}{1+e^{-x}}$ | `torch.sigmoid(x)` |
| $\tanh(x)$ | tanh | Hyperbolic tangent | `torch.tanh(x)` |
| $\text{ReLU}(x)$ | ReLU | $\max(0, x)$ | `F.relu(x)` |
| $\text{softmax}(x)_i$ | softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | `F.softmax(x, dim=-1)` |
| $\log \text{softmax}(x)$ | log softmax | Numerically stable $\log(\text{softmax})$ | `F.log_softmax(x, dim=-1)` |
| $\mathbb{1}[\text{condition}]$ | indicator | 1 if condition true, else 0 | `(condition).float()` |
| $\|x - y\|^2$ | squared distance | Squared Euclidean distance | `((x - y) ** 2).sum()` |
| $\max(0, m - s)$ | hinge | Hinge loss component | `F.relu(m - s)` |

### Attention and Transformer Notation

| Symbol | Name | Meaning | PyTorch Equivalent |
|--------|------|---------|-------------------|
| $Q$ | queries | Query matrix | `Q = x @ W_Q` |
| $K$ | keys | Key matrix | `K = x @ W_K` |
| $V$ | values | Value matrix | `V = x @ W_V` |
| $d_k$ | key dimension | Dimension of key vectors | `d_k = K.shape[-1]` |
| $d_{\text{model}}$ | model dimension | Hidden size of transformer | `d_model` (config) |
| $\text{Attention}(Q,K,V)$ | scaled dot-product | $\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | See implementation below |
| MultiHead | multi-head attention | Concatenated attention heads | `nn.MultiheadAttention` |
| PE | positional encoding | Position information | Sinusoidal or learned |
| [CLS] | classification token | Special token for classification | `cls_token` |
| Mask | attention mask | Prevents attending to certain positions | `attn_mask` |

### Subscript and Superscript Conventions

| Notation | Meaning | Example |
|----------|---------|---------|
| $x_i$ | $i$-th element of $x$ | `x[i]` |
| $x^{(t)}$ | $x$ at time step or iteration $t$ | `x_t` (in code) |
| $x^{(l)}$ | $x$ at layer $l$ | `layer_outputs[l]` |
| $W_{ij}$ | Weight from neuron $j$ to neuron $i$ | `W[i, j]` |
| $\theta_t$ | Parameters at optimization step $t$ | `optimizer.param_groups` |
| $h_t$ | Hidden state at time $t$ | `hidden_states[t]` |
| $x_{1:T}$ | Sequence from position 1 to $T$ | `x[:, :T]` (batch first) |

### Key Identities to Know

These come up repeatedly in derivations:

```
Softmax gradient:     d softmax(x)_i / d x_j = softmax(x)_i * (delta_ij - softmax(x)_j)
Cross-entropy:        L = -sum_i y_i * log(y_hat_i)
KL divergence:        KL(p || q) = sum_i p_i * log(p_i / q_i)
Bayes' rule:          P(z|x) = P(x|z) * P(z) / P(x)
Reparameterization:   z = mu + sigma * epsilon, where epsilon ~ N(0, I)
Log-sum-exp trick:    log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
Chain rule:           dL/dW = dL/dy * dy/dz * dz/dW
```

---

## Landmark Papers Reading List

These papers form the intellectual backbone of modern deep learning. Read them in order within
each section. Each entry includes the year, title, and why it matters.

### Foundations

1. **Rumelhart, Hinton, Williams (1986)** — "Learning representations by back-propagating errors"
   The paper that made backpropagation practical. Without this, none of deep learning exists.
   Read this to understand the core algorithm that trains every neural network.

2. **Hornik, Stinchcombe, White (1989)** — "Multilayer feedforward networks are universal approximators"
   Proves that a single hidden layer network can approximate any continuous function. Important
   for understanding what neural networks can theoretically represent vs. what they learn.

3. **LeCun, Bottou, Bengio, Haffner (1998)** — "Gradient-based learning applied to document recognition"
   The LeNet paper. Introduced convolutional neural networks for practical use. Established
   the pattern: convolution, pooling, fully connected. Still the template for CNN design.

### Convolutional Neural Networks

4. **Krizhevsky, Sutskever, Hinton (2012)** — "ImageNet classification with deep convolutional neural networks" (AlexNet)
   The paper that launched the deep learning revolution. Won ImageNet 2012 by a huge margin
   using GPUs. This is the moment the field changed.

5. **Simonyan, Zisserman (2014)** — "Very deep convolutional networks for large-scale image recognition" (VGGNet)
   Showed that depth matters. Used only 3x3 convolutions stacked deeply. Simple, clean
   architecture that became a standard feature extractor.

6. **He, Zhang, Ren, Sun (2015)** — "Deep residual learning for image recognition" (ResNet)
   Solved the degradation problem with skip connections. Made training very deep networks
   (100+ layers) possible. Residual connections now appear in almost every architecture.

7. **Ioffe, Szegedy (2015)** — "Batch normalization: accelerating deep network training by reducing internal covariate shift"
   Introduced batch normalization. The original justification (reducing internal covariate
   shift) has been challenged, but the technique remains essential. Read both this paper and
   "How Does Batch Normalization Help Optimization?" (Sanity et al., 2018) for the full story.

### Sequence Models

8. **Hochreiter, Schmidhuber (1997)** — "Long short-term memory" (LSTM)
   Solved the vanishing gradient problem in RNNs with gating mechanisms. Dominated sequence
   modeling for 20 years until transformers. Understanding gates is essential background.

9. **Bahdanau, Cho, Bengio (2014)** — "Neural machine translation by jointly learning to align and translate"
   Introduced the attention mechanism. Before this, encoder-decoder models compressed
   everything into a single vector. Attention lets the decoder look at all encoder states.
   This is the direct ancestor of the transformer.

10. **Vaswani, Shazeer, Parmar et al. (2017)** — "Attention is all you need" (Transformer)
    Eliminated recurrence entirely in favor of self-attention. Perhaps the most influential
    ML paper of the decade. Every modern language model, and increasingly vision models,
    use this architecture. Read this paper three times.

### Generative Models

11. **Kingma, Welling (2013)** — "Auto-encoding variational Bayes" (VAE)
    Introduced the variational autoencoder. Key ideas: the reparameterization trick for
    training through stochastic nodes, and the ELBO objective. Foundation for probabilistic
    deep learning.

12. **Goodfellow, Pouget-Abadie et al. (2014)** — "Generative adversarial nets" (GAN)
    Introduced adversarial training: a generator and discriminator competing. Elegant idea,
    notoriously difficult to train in practice. Spawned an enormous literature on stabilizing
    and improving GANs.

13. **Ho, Jain, Abbeel (2020)** — "Denoising diffusion probabilistic models" (DDPM)
    Made diffusion models practical for image generation. Simpler than GANs to train, better
    quality than VAEs. The foundation for DALL-E 2, Stable Diffusion, and the current
    generation of image models.

14. **Song, Ermon (2019)** — "Generative modeling by estimating gradients of the data distribution" (Score-based)
    Score matching perspective on diffusion. Together with DDPM, provides the two main
    viewpoints on diffusion models. Read this after DDPM for a deeper understanding.

### Modern Architecture and Scaling

15. **Devlin, Chang, Lee, Toutanova (2018)** — "BERT: Pre-training of deep bidirectional transformers for language understanding"
    Introduced masked language modeling for pretraining. Showed that bidirectional pretraining
    plus fine-tuning could achieve state-of-the-art on many NLP tasks. Defined the pretrain-
    then-fine-tune paradigm.

16. **Brown, Mann, Ryder et al. (2020)** — "Language models are few-shot learners" (GPT-3)
    Demonstrated that scaling language models to 175B parameters unlocks few-shot and zero-shot
    abilities. Changed the paradigm from fine-tuning to prompting. The paper that launched
    the current era of large language models.

17. **Dosovitskiy, Beyer, Kolesnikov et al. (2020)** — "An image is worth 16x16 words" (Vision Transformer)
    Applied the transformer to vision by splitting images into patches. Showed that the
    inductive biases of CNNs (locality, translation equivariance) are not necessary when
    you have enough data.

18. **Kaplan, McCandlish, Henighan et al. (2020)** — "Scaling laws for neural language models"
    Empirically characterized how loss scales with model size, dataset size, and compute.
    Showed smooth power-law relationships. This paper guides how labs allocate compute and
    influenced the push toward ever-larger models.

19. **Hoffmann, Borgeaud, Mensch et al. (2022)** — "Training compute-optimal large language models" (Chinchilla)
    Showed that most large language models are undertrained — you should scale data as much
    as model size. Changed how the field thinks about the compute-optimal frontier.

### Optimization

20. **Kingma, Ba (2014)** — "Adam: a method for stochastic optimization"
    The Adam optimizer. Combines momentum with adaptive learning rates. The default optimizer
    for most deep learning. Understand how it works and when it fails.

21. **Loshchilov, Hutter (2017)** — "Decoupled weight decay regularization" (AdamW)
    Fixed a subtle bug in how Adam interacts with L2 regularization. AdamW is now the standard
    optimizer for transformer training. The paper is a lesson in how small details matter.

22. **Loshchilov, Hutter (2016)** — "SGDR: Stochastic gradient descent with warm restarts" (Cosine annealing)
    Introduced cosine learning rate schedules and warm restarts. These schedules are now
    standard in transformer training.

23. **Frankle, Carlin (2018)** — "The lottery ticket hypothesis: finding sparse, trainable neural networks"
    Showed that dense networks contain sparse subnetworks ("winning tickets") that can match
    the full network's performance. Challenged the assumption that you need all parameters.
    Important for understanding overparameterization.

### Understanding Deep Learning

24. **Santurkar, Tsipras, Ilyas, Madry (2018)** — "How does batch normalization help optimization?"
    Challenged the original batch norm paper's explanation. Showed that batch norm smooths the
    loss landscape rather than reducing internal covariate shift. A model for how to do
    rigorous empirical investigation.

25. **Hinton, Vinyals, Dean (2015)** — "Distilling the knowledge in a neural network"
    Introduced knowledge distillation: training a small network to mimic a large one. The
    "dark knowledge" in soft labels carries more information than hard labels. Foundational
    for model compression.

26. **Zhang, Bengio, Hardt, Recht, Vinyals (2016)** — "Understanding deep learning requires rethinking generalization"
    Showed that deep networks can memorize random labels, challenging classical generalization
    theory. If a network can fit noise, why does it generalize to signal? This paper reshaped
    theoretical deep learning research.

### Efficient Training and Adaptation

27. **Hu, Shen, Wallis et al. (2021)** — "LoRA: Low-rank adaptation of large language models"
    Introduced low-rank adaptation for efficient fine-tuning. Instead of updating all
    parameters, add small low-rank matrices. Reduced fine-tuning parameters by 10,000x with
    comparable performance. Essential for practical LLM adaptation.

28. **Houlsby, Giurgiu, Jastrzebski et al. (2019)** — "Parameter-efficient transfer learning for NLP"
    Introduced adapter layers for efficient fine-tuning. Add small bottleneck layers between
    frozen pretrained layers. Predates LoRA and provides complementary perspective.

---

## Paper-to-Code Translation Guide

### The Systematic Process

#### Phase 1: Understand Before You Code (2-4 hours)

Do NOT open your IDE until you have completed these steps.

**Step 1.1: Identify the core algorithm.**
Read the method section and extract the algorithm. Strip away the motivation, the related work,
the theoretical analysis. What are the actual computational steps?

Write them as a numbered list:
```
1. Encode input x into representation h
2. Apply self-attention: compute Q, K, V from h
3. Compute attention weights: softmax(QK^T / sqrt(d_k))
4. Apply attention to values: weights @ V
5. Project output
6. Add residual connection and layer norm
...
```

**Step 1.2: Map all notation to concrete types.**
Build a complete notation table:
```
Symbol  | Meaning                | Shape           | Type
--------+------------------------+-----------------+-------
x       | input tokens           | (B, T)          | int64
E       | embedding matrix       | (V, d_model)    | float32
h       | hidden states          | (B, T, d_model) | float32
Q, K, V | query, key, value      | (B, T, d_k)     | float32
W_Q     | query projection       | (d_model, d_k)  | float32 (param)
```

**Step 1.3: Write detailed pseudocode.**
```
function TransformerBlock(x):
    # x shape: (batch, seq_len, d_model)

    # Multi-head attention sublayer
    residual = x
    x = LayerNorm(x)                    # (batch, seq_len, d_model)
    Q = x @ W_Q                         # (batch, seq_len, d_k)
    K = x @ W_K                         # (batch, seq_len, d_k)
    V = x @ W_V                         # (batch, seq_len, d_v)
    attn = softmax(Q @ K.T / sqrt(d_k)) # (batch, seq_len, seq_len)
    x = attn @ V                        # (batch, seq_len, d_v)
    x = x @ W_O                         # (batch, seq_len, d_model)
    x = x + residual                    # residual connection

    # Feed-forward sublayer
    residual = x
    x = LayerNorm(x)
    x = ReLU(x @ W_1 + b_1) @ W_2 + b_2
    x = x + residual

    return x
```

**Step 1.4: List every unknown detail.**
Things the paper does not specify that you need to implement:
```
UNKNOWN: Weight initialization scheme
UNKNOWN: Dropout rate in attention (mentioned but value unclear)
UNKNOWN: Whether layer norm is pre-norm or post-norm
UNKNOWN: Exact learning rate schedule warmup steps
UNKNOWN: Gradient clipping threshold
UNKNOWN: Data preprocessing / tokenization details
```

For each unknown, check: (1) the appendix, (2) supplementary material, (3) official code
if available, (4) follow-up papers, (5) blog posts by the authors.

#### Phase 2: Implement Bottom-Up (4-20 hours)

**Step 2.1: Implement the smallest testable component.**

Start with the simplest building block. For a transformer, this might be scaled dot-product
attention:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, heads, seq_len, d_k)
        K: (batch, heads, seq_len, d_k)
        V: (batch, heads, seq_len, d_v)
        mask: (batch, 1, 1, seq_len) or None
    Returns:
        output: (batch, heads, seq_len, d_v)
        attn_weights: (batch, heads, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (B, H, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
    output = attn_weights @ V                  # (B, H, T, d_v)

    return output, attn_weights
```

**Step 2.2: Test each component immediately.**

```python
# Test with known values
B, H, T, d_k = 2, 4, 8, 64
Q = torch.randn(B, H, T, d_k)
K = torch.randn(B, H, T, d_k)
V = torch.randn(B, H, T, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)

# Shape checks
assert output.shape == (B, H, T, d_k), f"Expected {(B, H, T, d_k)}, got {output.shape}"
assert weights.shape == (B, H, T, T), f"Expected {(B, H, T, T)}, got {weights.shape}"

# Attention weights should sum to 1 along last dimension
assert torch.allclose(weights.sum(-1), torch.ones(B, H, T), atol=1e-6)

# Test with mask: masked positions should get zero attention
mask = torch.ones(B, 1, 1, T)
mask[:, :, :, -2:] = 0  # Mask last two positions
_, masked_weights = scaled_dot_product_attention(Q, K, V, mask)
assert torch.allclose(masked_weights[:, :, :, -2:], torch.zeros(B, H, T, 2), atol=1e-6)

print("All tests passed.")
```

**Step 2.3: Build up component by component.**

Order of implementation for a Transformer:
1. Scaled dot-product attention (test)
2. Multi-head attention (test)
3. Position-wise feed-forward network (test)
4. Positional encoding (test, visualize)
5. Encoder layer = attention + FFN + residual + norm (test)
6. Full encoder = N encoder layers (test)
7. Decoder layer (test)
8. Full decoder (test)
9. Full transformer = encoder + decoder + embeddings + output projection (test)
10. Training loop with loss, optimizer, scheduler

**Step 2.4: Overfit a single batch first.**

Before training on the full dataset, verify your model can memorize a single batch:

```python
# Create a tiny dataset (e.g., 8 examples)
small_batch = next(iter(dataloader))
model.train()

for step in range(1000):
    loss = compute_loss(model, small_batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Loss should approach 0. If it doesn't, you have a bug.
```

**Step 2.5: Scale up gradually.**

```
1. Overfit 1 batch     -> verifies forward/backward pass works
2. Overfit 10 batches  -> verifies model has sufficient capacity
3. Train on 1% of data -> verifies data pipeline works
4. Train on 10% of data -> verifies training dynamics are reasonable
5. Train on full data   -> final experiment
```

At each step, check: Is the loss decreasing? Is the learning rate appropriate? Are gradients
flowing (not vanishing or exploding)?

#### Phase 3: Validate and Document (2-4 hours)

**Step 3.1: Compare to paper's reported results.**

Create a comparison table:
```
Metric          | Paper  | Our Reproduction | Difference
----------------+--------+------------------+-----------
Top-1 Accuracy  | 76.3%  | 74.8%           | -1.5%
Top-5 Accuracy  | 93.1%  | 92.6%           | -0.5%
Training Loss   | 2.1    | 2.3              | +0.2
BLEU Score      | 28.4   | 27.1             | -1.3
```

**Step 3.2: Investigate discrepancies.**

If your results do not match, systematically check:
1. Data preprocessing — different tokenization, normalization, or augmentation
2. Hyperparameters — learning rate, batch size, weight decay
3. Training duration — papers often train longer than they state
4. Implementation details — initialization, dropout placement, norm type
5. Compute differences — mixed precision, gradient accumulation, distributed training
6. Random seed — some results are seed-sensitive

**Step 3.3: Document everything.**

Write a reproduction report covering:
- Exact implementation choices made and why
- Every place where the paper was ambiguous and how you resolved it
- Results comparison table
- Hypotheses for any discrepancies
- What you learned from the process

### Common Pitfalls

**Pitfall 1: Shape errors.**
The most common bug. Add shape assertions liberally:
```python
assert x.shape == (batch_size, seq_len, d_model), \
    f"Expected shape {(batch_size, seq_len, d_model)}, got {x.shape}"
```

**Pitfall 2: Wrong dimension in operations.**
`softmax(x, dim=-1)` vs `softmax(x, dim=-2)` — one is correct, one silently produces garbage.
Always verify which dimension should sum to 1.

**Pitfall 3: Forgetting to detach.**
When a tensor should not receive gradients (e.g., targets in contrastive learning), forgetting
`.detach()` causes subtle training issues.

**Pitfall 4: Pre-norm vs post-norm.**
`LayerNorm(x + Sublayer(x))` (post-norm, original transformer) vs
`x + Sublayer(LayerNorm(x))` (pre-norm, most modern implementations).
Many papers say one but implement the other.

**Pitfall 5: Transposing the wrong way.**
`x.transpose(1, 2)` vs `x.permute(0, 2, 1)` — know when each is appropriate and what the
resulting shapes are.

**Pitfall 6: Off-by-one in masking.**
Causal masks, padding masks, and their combination. Test with small examples where you can
verify the output by hand.

**Pitfall 7: Learning rate schedule mismatches.**
Papers describe warmup + cosine decay, but the exact number of warmup steps, minimum learning
rate, and decay formula vary. Small differences here cause large differences in results.

**Pitfall 8: Evaluation mode.**
Forgetting `model.eval()` and `torch.no_grad()` during evaluation. Batch norm and dropout
behave differently in train vs eval mode.

---

## How to Write a Paper Review

### Why Learn This

Writing reviews makes you a better reader. When you know what reviewers look for, you read
papers more critically. When you can articulate strengths and weaknesses, you understand the
paper more deeply.

At DeepMind, you will review papers for internal reading groups and eventually for conferences.
This is a core professional skill.

### The Structure

#### 1. Summary (1 paragraph, 4-6 sentences)

Demonstrate that you read and understood the paper. Do NOT copy the abstract. Write the summary
in your own words. Include:
- The problem addressed
- The proposed approach (one sentence)
- The key result
- The main limitation or assumption

Example:
> "This paper addresses the problem of training very deep neural networks, which suffer from
> degradation (accuracy saturates then degrades with depth). The authors propose residual
> learning, where layers learn residual functions with reference to their inputs using shortcut
> connections. On ImageNet, a 152-layer ResNet achieves 3.57% top-5 error, winning the ILSVRC
> 2015 classification competition. The approach assumes that residual functions are easier to
> optimize than unreferenced functions, which is empirically validated but not theoretically
> proven."

#### 2. Strengths (3-5 bullet points)

Be specific and generous. Acknowledge genuine contributions. Each strength should be a
complete thought, not a fragment.

Good:
- "The experimental evaluation is thorough: the authors test on CIFAR-10, CIFAR-100, and
  ImageNet with consistent improvements, and provide ablation studies showing the effect of
  depth on both plain and residual networks (Table 1 and Figure 4)."

Bad:
- "Good experiments." (too vague)
- "Novel approach." (says nothing specific)

#### 3. Weaknesses (3-5 bullet points)

Be specific and constructive. Every weakness should suggest how it could be addressed.

Good:
- "The paper lacks theoretical justification for why residual learning should be easier
  to optimize. The 'identity mapping is easier to learn' argument (Section 3.1) is intuitive
  but not proven. Including an analysis of the loss landscape smoothness (e.g., as in
  Li et al., 2018) would strengthen this claim."

Bad:
- "Not enough theory." (too vague)
- "I don't believe the results." (not constructive)

#### 4. Questions for the Authors (2-5)

These should be genuine questions that would help you evaluate the paper. They might
request clarification, additional experiments, or responses to concerns.

Examples:
- "How sensitive are the results to the placement of batch normalization? Have you tested
  pre-activation residual blocks (identity mapping after addition)?"
- "Table 3 shows diminishing returns beyond 100 layers on CIFAR-10. Is there a practical
  limit to depth, and does this depend on the dataset size?"
- "The paper does not discuss computational cost. How does wall-clock training time compare
  between a 34-layer plain network and a 34-layer ResNet?"

#### 5. Minor Issues

Typos, grammatical errors, formatting issues, unclear figures. These should not affect your
overall assessment but are helpful for the authors.

#### 6. Overall Assessment

A numerical score and 2-3 sentence justification. Common scales:

**Conference scale (e.g., NeurIPS):**
- 1-3: Reject (fundamental flaws)
- 4-5: Borderline reject (interesting but significant weaknesses)
- 6-7: Borderline accept (solid contribution with some issues)
- 8-10: Accept (clear contribution, well-executed)

**Your justification should reference the four dimensions:**
> "I recommend acceptance (7/10). This paper makes a significant contribution by solving the
> degradation problem in deep networks through a simple and elegant mechanism (novelty: high,
> significance: high). The experimental evaluation is thorough (rigor: high), though the lack
> of theoretical justification is a notable gap. The approach is easy to reproduce given the
> detail in the paper and appendix (reproducibility: high)."

### Review Checklist

Before submitting a review, verify:

- [ ] Summary is in my own words and demonstrates understanding
- [ ] I identified at least 3 genuine strengths
- [ ] I identified at least 3 specific weaknesses with suggestions for improvement
- [ ] My questions are genuine and would help evaluate the paper
- [ ] I assessed novelty, rigor, reproducibility, and significance
- [ ] My tone is professional and respectful
- [ ] My criticisms are specific enough for the authors to act on
- [ ] I did not let personal bias (e.g., disliking the approach) override evidence
- [ ] I declared any potential conflicts of interest

### Common Review Mistakes

1. **Reviewing the paper you wish they had written, not the paper they wrote.** Evaluate the
   paper's contribution on its own terms.

2. **Rejecting for missing experiments that would require unreasonable compute.** Not every
   lab has 1,000 GPUs. Evaluate whether the experiments support the claims, not whether they
   are maximally comprehensive.

3. **Conflating "I don't understand X" with "X is wrong."** If something is unclear, ask for
   clarification. The gap might be in your knowledge, not the paper.

4. **Ignoring the supplementary material.** Many of your concerns may be addressed in the
   appendix. Read it before writing your review.

5. **Anchoring on one weakness.** A paper can have a serious weakness and still make a
   contribution worth publishing. Evaluate the whole picture.

---

## Quick Reference: Reading a Paper in Practice

Here is the actual workflow, from discovering a paper to implementing it:

```
Day 0: Discovery
    - See paper on arXiv/Twitter/reading group
    - First pass: 5-10 minutes
    - Decision: read further or skip

Day 1: Understanding
    - Second pass: 1 hour
    - Fill out paper reading template
    - Identify 2-3 follow-up papers to read

Day 2-3: Deep dive (if implementing)
    - Third pass: 4+ hours
    - Map all equations to pseudocode
    - List all implementation unknowns
    - Check appendix and supplementary material

Day 4-7: Implementation (if reproducing)
    - Phase 1: Understand (no code)
    - Phase 2: Implement bottom-up
    - Phase 3: Validate and document

Day 8+: Analysis
    - Write reproduction report
    - Identify limitations and possible improvements
    - Present to reading group or mentor
```

This workflow becomes faster with practice. After reading 50 papers, your first-pass takes
3 minutes. After reading 100 papers, you can do a productive second pass in 30 minutes.
After implementing 10 papers, the paper-to-code translation becomes almost automatic.

The goal is not speed for its own sake. The goal is to efficiently allocate your finite
reading time to the papers that matter most for your work. Every paper you skip wisely is
time saved for a paper that will change how you think.
