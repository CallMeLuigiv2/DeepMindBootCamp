# Module 4: Neural Networks Deep Dive — Lesson Plan

## Weeks 6-7 | The Foundation of Everything That Follows

This is the most important module in the entire course. Every architecture you will ever
encounter — transformers, diffusion models, reinforcement learning agents — is built from the
pieces covered here. If you understand these six sessions deeply, you can read any modern paper
and know what is happening under the hood. If you skim them, you will be guessing for the rest
of your career.

We do not "cover" topics here. We derive them, implement them, break them, and rebuild them.

---

## Session 1: The Perceptron to the MLP

**Duration**: 3 hours (1.5 lecture/derivation, 1.5 coding)

### Learning Objectives
By the end of this session, the apprentice will:
1. Explain why a single neuron is a linear classifier and what that means geometrically.
2. Derive the decision boundary of a single perceptron in 2D and nD.
3. Articulate precisely why each major activation function was invented and when to use it.
4. State the universal approximation theorem, what it guarantees, and what it does NOT guarantee.
5. Make informed decisions about network width vs depth for a given problem.
6. Explain representation learning: what intermediate layers actually compute.

### Outline

#### Part 1: The Single Neuron (30 min)
- A neuron computes: $z = \mathbf{w}^T \mathbf{x} + b$, then $a = \sigma(z)$.
- Geometrically: w defines a hyperplane in input space, b shifts it.
- Whiteboard derivation: draw the decision boundary for a 2D perceptron with specific weights.
  - Example: $\mathbf{w} = [2, -1]$, $b = 0.5$. Draw the line $2x_1 - x_2 + 0.5 = 0$.
  - Show which side is "class 1" depending on the activation function.
- The XOR problem: why a single neuron cannot solve it. Draw it. This is not a curiosity — it
  is the fundamental reason we need depth.

#### Part 2: Activation Functions — The Full Story (40 min)
- Whiteboard derivations for each:

| Function | Formula | Derivative | Range | Why It Was Invented |
|----------|---------|-----------|-------|---------------------|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $\sigma(1-\sigma)$ | $(0,1)$ | Biological plausibility, smooth step |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | $(-1,1)$ | Zero-centered outputs fix slow convergence |
| ReLU | $\max(0,z)$ | $0$ if $z<0$, $1$ if $z>0$ | $[0,\infty)$ | Sparse activation, no vanishing gradient for $z>0$ |
| LeakyReLU | $\max(\alpha z, z)$ | $\alpha$ if $z<0$, $1$ if $z>0$ | $(-\infty,\infty)$ | Fix "dead neuron" problem of ReLU |
| GELU | $z \cdot \Phi(z)$ | $\Phi(z) + z \cdot \phi(z)$ | $\approx(-0.17, \infty)$ | Stochastic regularization built into activation |
| Swish | $z \cdot \sigma(z)$ | $\text{swish}(z) + \sigma(z)(1-\text{swish}(z))$ | $\approx(-0.28, \infty)$ | Discovered via NAS, smooth non-monotonic |

- The historical arc: sigmoid dominated (1980s-2000s) -> tanh was better (zero-centered) ->
  ReLU changed everything (2010, Nair & Hinton) -> variants fixed ReLU's problems ->
  GELU/Swish found via search and theory.
- Key insight: the "right" activation function changed as we understood training dynamics better.

#### Part 3: Universal Approximation Theorem (20 min)
- Statement: A feedforward network with a single hidden layer containing a finite number of
  neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary
  accuracy, given a non-polynomial activation function.
- What this MEANS: MLPs are expressive enough. In principle.
- What this DOES NOT MEAN:
  - It says nothing about how MANY neurons you need (could be astronomically large).
  - It says nothing about whether gradient descent can FIND the right weights.
  - It says nothing about generalization to unseen data.
  - It does not mean a single hidden layer is optimal — depth gives exponential efficiency.
- Analogy: "Any speech can be written using the English alphabet" is true but tells you nothing
  about how to write a good speech.

#### Part 4: Width vs Depth (20 min)
- Width: more neurons per layer = richer representations at each level.
- Depth: more layers = hierarchical composition of features.
- The depth efficiency result: there exist functions computable by depth-k networks of
  polynomial size that require exponential width in depth-(k-1) networks.
- Practical heuristics:
  - Start with 2-3 hidden layers.
  - Make hidden layers wider than you think (then regularize).
  - Deeper networks need residual connections (foreshadow Module 5).
- The representation learning perspective: layer 1 learns edges, layer 2 learns textures,
  layer 3 learns parts, layer 4 learns objects. Each layer builds on the last.

### Coding Exercises (90 min)
1. **Perceptron from scratch** (20 min): Implement a single neuron in NumPy. Train it on a
   linearly separable 2D dataset. Visualize the decision boundary evolving during training.
2. **Activation function zoo** (20 min): Implement all six activation functions and their
   derivatives. Plot them. Plot their gradients. Verify derivatives numerically.
3. **MLP forward pass** (25 min): Build a 3-layer MLP forward pass in NumPy. Use it to
   classify the two-moons dataset. Visualize the decision boundary.
4. **Width vs depth experiment** (25 min): For the two-moons dataset, compare: (a) 1 hidden
   layer with 256 neurons, (b) 4 hidden layers with 16 neurons each, (c) 8 hidden layers with
   8 neurons each. Same total parameter count. Compare decision boundaries and training curves.

### Key Takeaways
- A single neuron draws a hyperplane. An MLP composes hyperplanes into arbitrary regions.
- Activation functions are not interchangeable. Each solves a specific problem.
- The universal approximation theorem is an existence result, not a construction.
- Depth gives exponential efficiency over width, but is harder to train.

---

## Session 2: Backpropagation — The Whole Story

**Duration**: 3.5 hours (2 lecture/derivation, 1.5 coding)

This is the session that separates practitioners who understand deep learning from those who
use it as a black box. We will derive every gradient by hand.

### Learning Objectives
1. Explain the forward pass as function composition and the backward pass as the chain rule.
2. Compute gradients by hand for a 3-layer MLP with specific numbers.
3. Explain the Jacobian perspective on backpropagation.
4. Analyze the computational cost of backprop vs numerical differentiation.
5. Derive mathematically why gradients vanish or explode in deep networks.

### Outline

#### Part 1: Forward Pass as Function Composition (20 min)
- For a 3-layer network:
  - z1 = W1 @ x + b1
  - a1 = relu(z1)
  - z2 = W2 @ a1 + b2
  - a2 = relu(z2)
  - z3 = W3 @ a2 + b3
  - y_hat = softmax(z3)
  - L = cross_entropy(y_hat, y)
- This is just: $L = f_6(f_5(f_4(f_3(f_2(f_1(\mathbf{x}))))))$
- Draw the computational graph explicitly on the whiteboard.

#### Part 2: Backward Pass — Full Derivation (50 min)
- Whiteboard derivation with CONCRETE dimensions:
  - Input: $\mathbf{x} \in \mathbb{R}^{784}$ (MNIST)
  - Layer 1: $W_1 \in \mathbb{R}^{128 \times 784}$, $\mathbf{b}_1 \in \mathbb{R}^{128}$
  - Layer 2: $W_2 \in \mathbb{R}^{64 \times 128}$, $\mathbf{b}_2 \in \mathbb{R}^{64}$
  - Layer 3: $W_3 \in \mathbb{R}^{10 \times 64}$, $\mathbf{b}_3 \in \mathbb{R}^{10}$
- Derive $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_3} = \hat{\mathbf{y}} - \mathbf{y}$ (for softmax + cross-entropy).
- Then: $\frac{\partial \mathcal{L}}{\partial W_3} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_3} \mathbf{a}_2^T$ (outer product).
- Then: $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_3} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_3}$.
- Then: $\frac{\partial \mathcal{L}}{\partial \mathbf{a}_2} = W_3^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}_3}$.
- Then: $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_2} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_2} \odot \text{relu}'(\mathbf{z}_2)$ (element-wise).
- Continue recursively.
- Write every single step. Use specific numbers for a mini-example (2-2-1 network) to
  verify understanding.

#### Part 3: The Jacobian Perspective (20 min)
- Each layer's backward pass computes a vector-Jacobian product (VJP).
- The Jacobian of layer $i$: $J_i = \frac{\partial \mathbf{a}_i}{\partial \mathbf{z}_i} \cdot \frac{\partial \mathbf{z}_i}{\partial \mathbf{a}_{i-1}} = \text{diag}(\sigma'(\mathbf{z}_i)) \cdot W_i$.
- Total gradient: product of all Jacobians from output to input.
- This is why understanding matrix norms matters: if $\|J_i\| > 1$ for many layers, gradients
  explode. If $\|J_i\| < 1$ for many layers, gradients vanish.

#### Part 4: Computational Efficiency (15 min)
- Numerical differentiation: perturb each parameter, recompute loss. For N parameters,
  need N+1 forward passes. For a ResNet-50 with 25M parameters: 25 million forward passes.
- Backpropagation: ONE forward pass + ONE backward pass (roughly 2-3x the cost of forward).
- This is the reason deep learning is possible at all.
- Relate to forward-mode vs reverse-mode automatic differentiation.

#### Part 5: Vanishing and Exploding Gradients (35 min)
- The mathematical argument:
  - $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \left(\prod_{i=1}^{L} J_i\right) \frac{\partial \mathcal{L}}{\partial \mathbf{a}_L}$
  - If each $\|J_i\| \sim r$, then $\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}}\right\| \sim r^L$.
  - $r < 1$: vanishing (exponentially). $r > 1$: exploding (exponentially).
- With sigmoid: $\sigma'(z) \in (0, 0.25]$. So max gradient per layer is $0.25 \cdot \|W_i\|$.
  Unless $\|W_i\| > 4$, gradients shrink every layer.
- With ReLU: $\text{relu}'(z) \in \{0, 1\}$. No shrinkage for active neurons. But dead neurons contribute
  zero gradient permanently.
- This drove 15 years of research:
  - Better activations (ReLU, 2010)
  - Better initialization (Xavier 2010, He 2015)
  - Normalization (BatchNorm 2015)
  - Residual connections (ResNet 2015)
  - Each technique addresses the same core problem differently.

### Coding Exercises (90 min)
1. **Backprop by hand, verified by code** (40 min): Implement a 3-layer MLP in NumPy.
   Manually compute gradients using the chain rule. Compare with PyTorch autograd. Difference
   should be < 1e-6.
2. **Gradient flow visualization** (25 min): Train a 10-layer MLP with sigmoid activations.
   Plot the gradient magnitude at each layer over training. Watch them vanish. Switch to ReLU
   and repeat.
3. **Numerical vs analytical gradients** (25 min): Implement numerical differentiation
   (finite differences). Time it vs backprop for networks of increasing size. Plot the
   computational cost ratio.

### Key Takeaways
- Backpropagation is just the chain rule applied to a computational graph.
- It converts an O(N) problem (per-parameter numerical gradients) into O(1) additional passes.
- Vanishing/exploding gradients are a mathematical inevitability of deep function composition.
- Every major architecture innovation in 2010-2020 was fundamentally about gradient flow.

---

## Session 3: Weight Initialization

**Duration**: 2.5 hours (1.5 lecture/derivation, 1 coding)

### Learning Objectives
1. Explain why zero initialization breaks symmetry requirements.
2. Derive Xavier/Glorot initialization from first principles.
3. Derive Kaiming/He initialization for ReLU networks.
4. Articulate how BatchNorm reduces sensitivity to initialization.
5. Apply correct initialization in practice.

### Outline

#### Part 1: Why Zeros Don't Work (15 min)
- If all weights are zero, all neurons compute the same function.
- Gradients are identical for all neurons in a layer.
- Update is identical. Symmetry is never broken.
- The network effectively has one neuron per layer regardless of width.
- Demonstrate with code: train a 3-layer MLP with zero init. Show it does not learn.

#### Part 2: Why Random Isn't Enough (15 min)
- Too small (e.g., $\mathcal{N}(0, 0.001)$): activations shrink to zero layer by layer. Gradients vanish.
- Too large (e.g., $\mathcal{N}(0, 1)$): activations saturate. Gradients also vanish (for sigmoid/tanh)
  or explode (for ReLU).
- The Goldilocks problem: we need the VARIANCE to be just right.

#### Part 3: Xavier/Glorot Initialization — Full Derivation (30 min)
- Goal: keep the variance of activations constant across layers.
- For layer $i$: $\mathbf{z}_i = W_i \mathbf{a}_{i-1}$.
- $\text{Var}(z_i) = n_{i-1} \cdot \text{Var}(W_i) \cdot \text{Var}(a_{i-1})$.
  (Assumes inputs and weights are independent, zero-mean.)
- For $\text{Var}(z_i) = \text{Var}(a_{i-1})$, we need $\text{Var}(W_i) = 1/n_{i-1}$.
- For the backward pass: we similarly need $\text{Var}(W_i) = 1/n_i$.
- Compromise: $\text{Var}(W_i) = \frac{2}{n_{i-1} + n_i}$.
- Xavier initialization: $W \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$ or $\mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\; \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right)$.
- This assumes linear or tanh activation (derivative near 1 at initialization).

#### Part 4: Kaiming/He Initialization — For ReLU (20 min)
- ReLU zeros out half the activations. $\text{Var}(\text{relu}(z)) = \text{Var}(z)/2$.
- Correcting: $\text{Var}(W_i) = 2/n_{i-1}$.
- He initialization: $W \sim \mathcal{N}(0, 2/n_{\text{in}})$.
- For LeakyReLU with slope $\alpha$: $\text{Var}(W_i) = \frac{2}{(1+\alpha^2) \cdot n_{i-1}}$.
- Whiteboard: show variance propagation through 10 layers with Xavier vs He for ReLU network.

#### Part 5: LSUV — Layer-Sequential Unit Variance (10 min)
- Data-driven initialization: feed a batch through the network, normalize each layer's
  output to have unit variance.
- Iterative: adjust weights until Var(output) ~ 1 for each layer.
- Advantage: works for any activation function without derivation.

#### Part 6: BatchNorm and Initialization (15 min)
- BatchNorm explicitly normalizes activations to zero mean, unit variance.
- This means the network is less sensitive to initialization — BatchNorm "fixes" bad init.
- But initialization still matters: it affects the initial loss landscape and early training.
- Practical guideline: use He init + BatchNorm for most architectures.

#### Part 7: Practical Guidelines (15 min)
- ReLU/LeakyReLU: He initialization.
- Sigmoid/Tanh: Xavier initialization.
- GELU/Swish: He initialization (close enough to ReLU).
- If using BatchNorm or LayerNorm: initialization is less critical but still use the above.
- Bias terms: initialize to zero (almost always).
- Output layer bias: initialize to the log-prior of classes for classification.
- Embedding layers: N(0, 1) or N(0, 1/sqrt(d)).

### Coding Exercises (60 min)
1. **Variance propagation experiment** (25 min): Build a 50-layer MLP (no activation, just
   linear layers). Initialize with different scales. Plot activation variance across layers.
   Find the scale that keeps variance constant. Verify it matches Xavier.
2. **Initialization shootout** (20 min): Train the same architecture on MNIST with: zero init,
   small random, large random, Xavier, He. Plot training curves. Show failure modes.
3. **LSUV implementation** (15 min): Implement LSUV. Apply it to a 20-layer network. Show
   activation variance before and after.

### Key Takeaways
- Initialization is a solved problem in practice, but understanding WHY it works is essential.
- The core insight: keep activation variance ~1 across layers at initialization.
- He init for ReLU, Xavier for tanh/sigmoid, and BatchNorm as insurance.
- Bad initialization does not just slow training — it can completely prevent it.

---

## Session 4: Regularization Arsenal

**Duration**: 3 hours (1.5 lecture, 1.5 coding)

### Learning Objectives
1. Explain the bias-variance tradeoff in the context of neural networks.
2. Implement and analyze L1/L2 regularization, dropout, and batch normalization.
3. Compare normalization techniques: BatchNorm, LayerNorm, GroupNorm.
4. Design a regularization strategy for a given problem.

### Outline

#### Part 1: The Overfitting Problem (15 min)
- Neural networks have more parameters than data points. They can memorize anything.
- Zhang et al. (2017): neural networks can fit random labels with 100% training accuracy.
- Regularization = any technique that improves generalization without improving training loss.
- The fundamental tension: expressiveness vs generalization.

#### Part 2: L1 and L2 Regularization (25 min)
- L2 (weight decay): $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \sum w^2$.
  - Gradient: $\frac{\partial \mathcal{L}}{\partial w} \mathrel{+}= \lambda w$.
  - Effect: pulls weights toward zero. Larger weights are penalized more.
  - Bayesian interpretation: Gaussian prior on weights.
  - In practice: $\lambda = 10^{-4}$ to $10^{-2}$ is typical.
- L1 regularization: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \sum |w|$.
  - Gradient: $\frac{\partial \mathcal{L}}{\partial w} \mathrel{+}= \lambda \cdot \text{sign}(w)$.
  - Effect: drives weights to exactly zero. Produces sparse networks.
  - Bayesian interpretation: Laplace prior on weights.
- L2 vs L1: L2 shrinks all weights, L1 prunes. L2 is almost always preferred for neural nets.
- Weight decay vs L2 in Adam: they are NOT the same. Loshchilov & Hutter (2019) showed
  decoupled weight decay (AdamW) is correct.

#### Part 3: Dropout (25 min)
- During training: randomly set each neuron's output to zero with probability p.
- The ensemble interpretation: training 2^n different subnetworks (one per dropout mask).
  At test time, we approximate the ensemble average.
- Inverted dropout: scale activations by $1/(1-p)$ during training so test time needs no change.
- Whiteboard: derive the expected value of a neuron's output with inverted dropout.
- Spatial dropout: for CNNs, drop entire feature maps instead of individual neurons.
  - Why: adjacent pixels are correlated, so standard dropout is ineffective.
- Practical notes:
  - p = 0.5 for hidden layers, p = 0.1-0.3 for input layer.
  - Dropout + BatchNorm interaction: order matters and can cause issues. Use one or the other.
  - Dropout is used less in modern architectures (replaced by other regularization).

#### Part 4: Batch Normalization (30 min)
- Forward pass (training):
  1. Compute batch mean: $\mu_B = \frac{1}{m} \sum_i x_i$.
  2. Compute batch variance: $\sigma^2_B = \frac{1}{m} \sum_i (x_i - \mu_B)^2$.
  3. Normalize: $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$.
  4. Scale and shift: $y_i = \gamma \hat{x}_i + \beta$.
  - $\gamma$ and $\beta$ are LEARNABLE parameters.
- Forward pass (evaluation): use running mean and running variance (exponential moving average
  from training).
- Why gamma and beta? Without them, normalization would force all layers to have N(0,1)
  activations, which limits the network's representational power. Gamma and beta let the
  network undo the normalization if that is optimal.
- WHY it works (multiple theories):
  - Original paper: reduces "internal covariate shift." Subsequent work showed this is not
    the main mechanism.
  - Smooths the loss landscape (Santurkar et al., 2018). This is the current best explanation.
  - Acts as a regularizer (noise from batch statistics).
  - Allows higher learning rates.
- Practical impact: almost always improves training speed. Sometimes improves final accuracy.
  Batch size matters — small batches mean noisy statistics.

#### Part 5: Layer Normalization and Group Normalization (15 min)
- **LayerNorm**: normalize across features (not batch). Used in transformers.
  - Advantage: no dependence on batch size. Works for sequence models.
- **GroupNorm**: normalize within groups of channels.
  - Advantage: works with small batch sizes (important for detection, segmentation).
- **InstanceNorm**: normalize per channel per sample. Used in style transfer.
- Comparison table: what each normalizes over, when to use it.

#### Part 6: Data Augmentation as Regularization (10 min)
- Artificially increasing the effective dataset size.
- For images: flips, rotations, crops, color jitter, Cutout, Mixup, CutMix.
- For text: synonym replacement, back-translation, dropout on embeddings.
- This is often the single most effective regularization technique.

#### Part 7: Early Stopping (10 min)
- Monitor validation loss. Stop when it starts increasing.
- Patience: how many epochs to wait before stopping.
- Equivalent to L2 regularization in linear models (provable result).
- Always use it. There is no reason not to.

### Coding Exercises (90 min)
1. **Memorization demonstration** (15 min): Train a large MLP on 1000 MNIST samples with
   random labels. Show it achieves 100% training accuracy.
2. **L2 regularization sweep** (20 min): Train on the 1000-sample subset with L2 lambda
   in {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}. Plot train/val accuracy for each.
3. **Dropout experiment** (20 min): Same setup with dropout rates {0, 0.1, 0.3, 0.5, 0.7, 0.9}.
   Plot results. Show how test-time behavior differs from training.
4. **BatchNorm implementation** (20 min): Implement BatchNorm from scratch in NumPy. Verify
   against PyTorch's nn.BatchNorm1d. Test both training and eval modes.
5. **Regularization combination** (15 min): Find the best combination of techniques for the
   1000-sample problem. Report your best validation accuracy and the recipe.

### Key Takeaways
- Every regularization technique addresses the same problem (overfitting) differently.
- L2 regularization and data augmentation are almost always helpful.
- BatchNorm is primarily about training dynamics, not regularization.
- The best regularization strategy depends on the data, the architecture, and the scale.
- When in doubt: data augmentation + weight decay + early stopping.

---

## Session 5: Optimization Deep Dive

**Duration**: 3 hours (1.5 lecture/derivation, 1.5 coding)

### Learning Objectives
1. Describe the loss landscape of deep networks and why optimization is hard.
2. Derive SGD with momentum, Nesterov momentum, and Adam from first principles.
3. Explain learning rate schedules and warmup.
4. Articulate the generalization debate: do optimizers affect what solution we find?

### Outline

#### Part 1: Loss Landscapes (20 min)
- The loss function L(theta) lives in a space with millions of dimensions.
- Local minima, saddle points, flat regions, sharp valleys.
- Key insight: in high dimensions, saddle points are far more common than local minima.
  A random critical point with N dimensions has a ~50% chance of being a saddle point
  for each eigenvalue of the Hessian. Local minima require ALL eigenvalues positive.
- Visualizing loss landscapes: Li et al. (2018) random 2D projections.

#### Part 2: SGD with Momentum (25 min)
- Vanilla SGD: $\theta_{t+1} = \theta_t - \eta \cdot g_t$.
  - Problem: oscillates in narrow valleys, slow in flat regions.
- Momentum: the physics analogy.
  - $v_{t+1} = \beta v_t + g_t$
  - $\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}$
  - $\beta$ is typically 0.9 (heavy ball) or 0.99 (very heavy ball).
  - Analogy: a ball rolling down a hill accumulates velocity.
- Effect: smooths out oscillations, accelerates in consistent gradient directions.
- Whiteboard derivation: show how momentum helps on a 2D quadratic with different curvatures
  along each axis.

#### Part 3: Nesterov Momentum (15 min)
- "Look ahead" before computing gradient:
  - $v_{t+1} = \beta v_t + \nabla f(\theta_t - \eta \beta v_t)$
  - $\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}$
- Intuition: compute the gradient at the position you are about to be at, not where you are.
- Slightly better convergence guarantees. Modest practical improvement.

#### Part 4: Adaptive Methods (40 min)
- **AdaGrad**: accumulate squared gradients, divide learning rate by their square root.
  - $s_t = s_{t-1} + g_t^2$
  - $\theta_{t+1} = \theta_t - \eta \cdot g_t / (\sqrt{s_t} + \epsilon)$
  - Problem: learning rate monotonically decreases. Eventually stops learning.
- **RMSProp**: fix AdaGrad with exponential moving average.
  - $s_t = \beta \cdot s_{t-1} + (1-\beta) \cdot g_t^2$
  - $\theta_{t+1} = \theta_t - \eta \cdot g_t / (\sqrt{s_t} + \epsilon)$
  - $\beta = 0.99$ typically.
- **Adam**: combine momentum with RMSProp.
  - $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ (first moment)
  - $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ (second moment)
  - $\hat{m}_t = m_t / (1 - \beta_1^t)$ (bias correction)
  - $\hat{v}_t = v_t / (1 - \beta_2^t)$ (bias correction)
  - $\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
- Whiteboard: derive the bias correction. At $t=0$, $m_0 = 0$. After one step,
  $m_1 = (1-\beta_1)g_1$. The expected value is $(1-\beta_1)\mathbb{E}[g]$, not $\mathbb{E}[g]$. Dividing by
  $(1-\beta_1^t)$ corrects this. Same for $v_t$.
- **AdamW**: decouple weight decay from the gradient-based update. Loshchilov & Hutter (2019).
  - Instead of: gradient += lambda * theta (L2 regularization on the loss)
  - Do: theta -= lr * lambda * theta (direct weight decay, separate from Adam update)
  - These are identical for SGD but different for Adam because Adam scales gradients.

#### Part 5: Learning Rate Schedules (20 min)
- Step decay: reduce LR by factor every N epochs.
- Cosine annealing: LR follows a cosine curve from max to min.
- Warmup: start with very small LR, linearly increase to target over K steps.
  - Why: at initialization, gradients are large and noisy. Large LR causes instability.
  - Especially important for Adam (adaptive methods are sensitive to early gradients).
- Cyclical learning rates (Smith, 2017): oscillate LR between bounds.
  - Can escape sharp minima. Sometimes finds better solutions.
- One-cycle policy: warmup to max LR, then cosine decay. Very effective in practice.

#### Part 6: The Generalization Debate (20 min)
- Sharp minima vs flat minima:
  - Keskar et al. (2017): large batch SGD finds sharp minima, which generalize poorly.
  - Dinh et al. (2017): sharpness can be reparameterized — the definition is not invariant.
  - Current view: the relationship is real but subtle.
- SGD noise as regularization:
  - Small batch SGD has inherent gradient noise.
  - This noise helps escape sharp minima and find flatter regions.
  - This is why SGD often generalizes better than Adam (but Adam trains faster).
- Practical resolution: Adam for fast convergence, SGD+momentum for best final accuracy
  (in vision). Transformers almost exclusively use Adam/AdamW.

### Coding Exercises (90 min)
1. **Optimizer from scratch** (30 min): Implement SGD, SGD+momentum, and Adam in NumPy.
   Train on a 2D Rosenbrock function. Visualize the optimization trajectories.
2. **Learning rate finder** (20 min): Implement the LR range test (Smith, 2017): train for
   one epoch with LR increasing exponentially from 1e-7 to 10. Plot loss vs LR. The optimal
   LR is where the loss decreases fastest.
3. **Optimizer comparison on MNIST** (20 min): Train the same MLP with SGD, SGD+momentum,
   Adam, and AdamW. Plot training loss and validation accuracy.
4. **Learning rate schedule comparison** (20 min): For the best optimizer above, compare:
   constant LR, step decay, cosine annealing, one-cycle. Plot results.

### Key Takeaways
- SGD with momentum is the workhorse. Adam is the convenience optimizer.
- Adaptive methods (Adam) adapt per-parameter learning rates based on gradient history.
- Learning rate is the most important hyperparameter. Use a schedule.
- The optimizer you choose affects not just convergence speed but what solution you find.
- Default starting point: AdamW with $\eta = 3 \times 10^{-4}$, cosine schedule with warmup.

---

## Session 6: Debugging Neural Networks

**Duration**: 3 hours (1 lecture, 2 hands-on debugging)

### Learning Objectives
1. Apply a systematic debugging protocol to a non-training network.
2. Use the learning rate finder to set learning rate.
3. Visualize gradient flow and diagnose problems from it.
4. Diagnose common failure modes from loss curves alone.
5. Debug a deliberately broken network (live exercise).

### Outline

#### Part 1: The Systematic Approach (30 min)
- The debugging protocol (in order):
  1. **Verify your data**: look at actual samples. Check labels. Check preprocessing. Compute
     class balance. This catches 50% of bugs.
  2. **Simplify the problem**: use a tiny subset. If 10 samples, can you overfit to 100%?
  3. **Overfit one batch**: take ONE batch and train on it repeatedly. Loss should go to ~0.
     If not, your model or training loop has a bug.
  4. **Gradually scale**: increase data, add regularization.
  5. **Use proven architectures first**: do not innovate on architecture AND data simultaneously.
- Why this order matters: each step narrows the possible failure modes.

#### Part 2: The Learning Rate Finder (15 min)
- Algorithm: exponentially increase LR from very small to very large over one epoch.
- Plot loss vs LR. The optimal LR is roughly 10x smaller than where the loss starts
  increasing.
- Implement and demonstrate live.

#### Part 3: Gradient Flow Visualization (20 min)
- For each layer, plot the ratio of gradient magnitude to parameter magnitude.
- Healthy: ratios are roughly 1e-3 across all layers.
- Vanishing: ratios decrease exponentially with depth.
- Exploding: ratios increase exponentially with depth.
- Dead ReLU detection: count the fraction of neurons that have zero gradient across a batch.
  If > 50%, you have a problem.

#### Part 4: Loss Curve Diagnosis (25 min)
- **Loss not decreasing at all**: learning rate too high or too low, bug in loss function,
  bug in data pipeline (labels shuffled relative to inputs).
- **Loss decreasing then NaN**: learning rate too high, numerical overflow, missing gradient
  clipping.
- **Training loss good, validation loss bad**: overfitting. Add regularization or more data.
- **Training loss bad, validation loss also bad**: underfitting. Bigger model, train longer,
  check for bugs.
- **Loss oscillating wildly**: learning rate too high, batch size too small, data issue.
- **Loss plateau then sudden drop**: learning rate schedule kicked in, or the network just
  needed more time.
- Show examples of each with real loss curves.

#### Part 5: Common Failure Modes and Fixes (30 min)
- **Incorrect data preprocessing**: e.g., not normalizing inputs, wrong channel order.
- **Incorrect loss function**: e.g., using cross-entropy without softmax, or with it twice.
- **Forgetting model.train() / model.eval()**: BatchNorm and Dropout behave differently.
- **Not zeroing gradients**: gradients accumulate across batches by default in PyTorch.
- **Data leakage**: training on test data unknowingly.
- **Learning rate too high for fine-tuning**: pre-trained models need 10-100x smaller LR.
- **Batch size too large**: reduces gradient noise, can hurt generalization.
- **Incorrect tensor shapes**: broadcasting hides dimension mismatches.

### Hands-On Debugging Lab (120 min)
The apprentice receives 4 "broken" training scripts. Each has 1-3 bugs. They must:
1. Identify the bug(s) from symptoms (loss curves, gradient plots, predictions).
2. Fix the bug(s).
3. Get the network training correctly.

Bugs include:
- Bug set 1: Labels not matching images (data loader bug), learning rate 100x too high.
- Bug set 2: Missing model.eval() for validation, forgetting optimizer.zero_grad().
- Bug set 3: Dead ReLU (bad initialization + large learning rate), no gradient clipping
  causing NaN.
- Bug set 4: Subtle — using L2 regularization in Adam instead of weight decay (AdamW). The
  network trains but generalizes poorly.

### Key Takeaways
- Always start with the data. Most bugs are data bugs.
- Overfit one batch first. This is the most underrated debugging tool.
- Gradient flow visualization tells you immediately if something is wrong.
- Loss curves are information-rich. Learn to read them.
- The difference between a working network and a broken one is often ONE line of code.

---

## Assessment Criteria

At the end of this module, the apprentice should be able to:
1. Derive backpropagation for a 3-layer MLP from scratch, on paper, without references.
2. Implement an MLP from scratch in NumPy that trains on MNIST.
3. Diagnose a broken training run from loss curves and gradient plots within 15 minutes.
4. Choose appropriate initialization, regularization, and optimizer for a new problem,
   and explain why.
5. Write a training loop in PyTorch with all best practices (gradient clipping, LR scheduling,
   early stopping, proper eval mode, etc.).

This module is the foundation. Do not proceed to convolutional networks or transformers until
every concept here is second nature.
