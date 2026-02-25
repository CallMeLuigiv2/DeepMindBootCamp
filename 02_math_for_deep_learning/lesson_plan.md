# Module 02: Math for Deep Learning — Week 3 Lesson Plan

## Philosophy

You already know basic math. This module is not a math refresher — it is a re-orientation.
Every concept here is taught through the lens of deep learning. We do not prove theorems
for elegance; we prove them because understanding the proof tells you why your model is
not converging, why your gradients are exploding, or why your VAE produces blurry images.

The goal by end of week: you should look at a neural network and see linear algebra,
calculus, and probability happening simultaneously at every layer, at every step of training.

---

## Session 1: Linear Algebra for Deep Learning

**Duration:** 3 hours (with breaks)

### Learning Objectives

By the end of this session, the apprentice will be able to:

1. Interpret vectors not just as lists of numbers, but as points in a learned representation space.
2. Read a matrix and immediately see a linear transformation — rotation, scaling, projection.
3. Explain why matrix multiplication is the dominant operation in neural networks.
4. Perform eigendecomposition and connect it to PCA and to what layers learn.
5. Use SVD for dimensionality reduction and data compression.
6. Apply norms correctly and know when to use L1 vs L2 vs Frobenius.
7. Predict NumPy/PyTorch broadcasting behavior without running the code.

### Topics and Time Allocation

| Time | Topic | Key Insight |
|------|-------|-------------|
| 0:00–0:25 | Vectors as points in representation space | An embedding IS a vector. Word2Vec, image features — all vectors. |
| 0:25–0:50 | Dot products as similarity | Cosine similarity, attention scores — all dot products. |
| 0:50–1:00 | **Break** | |
| 1:00–1:35 | Matrices as transformations | A linear layer $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ is a transformation of space. Visualize 2D. |
| 1:35–2:05 | Matrix multiplication as composition | Stacking layers = composing transformations. Why depth matters. |
| 2:05–2:15 | **Break** | |
| 2:15–2:35 | Eigenvalues, eigenvectors, and PCA | Eigenvectors = directions that survive a transformation. PCA = finding the axes of maximum variance. |
| 2:35–2:50 | Rank, null space, and SVD | Rank = information capacity. SVD = optimal low-rank approximation. Image compression demo. |
| 2:50–3:00 | Norms (L1, L2, Frobenius) and broadcasting | L1 = sparsity pressure. L2 = weight decay. Broadcasting = how tensors align in real code. |

### Intuition-Building Exercises

1. **Transformation Gallery** (15 min): Given a set of 2D points forming the letter "F", apply rotation, scaling, shearing, and projection matrices. Sketch the results by hand before running code. The "F" is asymmetric so you can see exactly what each transformation does.

2. **Dot Product as Detector** (10 min): Create a "template" vector and compute its dot product with several candidate vectors. Observe that the dot product is highest when the candidate matches the template. This is exactly what a neuron does — it is a learned template detector.

3. **Eigenvalue Intuition** (15 min): Apply a 2x2 matrix to random vectors repeatedly (power iteration). Watch most vectors converge to the dominant eigenvector direction. This is why PCA works.

4. **Broadcasting Prediction** (10 min): Given 5 broadcasting scenarios with shape annotations, predict the output shape and values before running. Then verify. At least one should be a common trap.

### Key Connections to Deep Learning

- **Vectors as embeddings**: Every hidden layer produces a vector. That vector IS the network's learned representation. Word2Vec showed that vector arithmetic captures semantics ($\text{king} - \text{man} + \text{woman} \approx \text{queen}$). This is not a trick — it is a consequence of linear structure in the embedding space.

- **Matrix multiplication as the forward pass**: The single most important operation in deep learning. A linear layer with 784 inputs and 256 outputs performs a 256x784 matrix multiplication billions of times during training. Understanding this operation at a mechanical level is non-negotiable.

- **Eigendecomposition and what layers learn**: The weight matrix of a trained layer has eigenvectors. Those eigenvectors correspond to "directions" in input space that the layer amplifies or suppresses. This connects directly to PCA, to understanding batch normalization (which whitens activations, i.e., makes the covariance matrix an identity), and to spectral normalization in GANs.

- **SVD and compression**: SVD gives you the best rank-k approximation to any matrix. This is the theoretical foundation for low-rank adaptation (LoRA) in large language models — instead of fine-tuning all parameters, you learn a low-rank update.

- **Norms as regularization**: $L_2$ norm of weights = weight decay. $L_1$ norm = sparsity. Frobenius norm of weight matrices appears in spectral normalization. Every regularization technique is, at its core, a constraint on some norm.

---

## Session 2: Calculus and Optimization

**Duration:** 3 hours (with breaks)

### Learning Objectives

By the end of this session, the apprentice will be able to:

1. Compute derivatives analytically and interpret them as local linear approximations.
2. Compute partial derivatives and gradients for multivariate functions.
3. Apply the chain rule — THE mechanism of backpropagation — fluently.
4. Explain when and why Jacobians and Hessians matter in practice.
5. Use Taylor expansion to justify why gradient descent works.
6. Distinguish convex from non-convex landscapes and understand the implications.
7. Explain saddle points, why they dominate in high dimensions, and how SGD noise helps.

### Topics and Time Allocation

| Time | Topic | Key Insight |
|------|-------|-------------|
| 0:00–0:20 | Derivatives as local linear approximation | $f(x + h) \approx f(x) + f'(x) \cdot h$. This is the entire basis of gradient-based optimization. |
| 0:20–0:45 | Partial derivatives and gradients | The gradient is a vector. It points uphill. We go downhill. That is gradient descent. |
| 0:45–0:55 | **Break** | |
| 0:55–1:35 | The chain rule — THE foundation of backprop | $\frac{d}{dx} f(g(h(x))) = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$. Every layer contributes one factor. This IS backpropagation. |
| 1:35–1:55 | Jacobians and Hessians | Jacobian = gradient for vector-valued functions. Hessian = curvature. Second-order methods use curvature but are expensive. |
| 1:55–2:05 | **Break** | |
| 2:05–2:25 | Taylor expansion and why SGD works | First-order Taylor = linear approximation. SGD works because loss landscapes are locally smooth enough that linear approximations hold for small steps. |
| 2:25–2:45 | Convexity, local minima, and loss landscapes | Convex = one minimum. Neural network losses are non-convex but "benign" — most local minima are nearly as good as the global one. |
| 2:45–3:00 | Saddle points and SGD noise | In high-D, saddle points vastly outnumber local minima. Stochastic noise in SGD helps escape them. This is a feature, not a bug. |

### Intuition-Building Exercises

1. **Derivative as Zoom** (10 min): Plot a function and zoom in at a point. As you zoom in, the curve looks more and more like a straight line. The derivative IS the slope of that line. This is not an approximation trick — it is the definition.

2. **Gradient Compass** (15 min): For $f(x, y) = x^2 + 3y^2$, compute the gradient at several points. Draw arrows on a contour plot. Observe: the gradient always points perpendicular to contour lines, toward steeper ascent. Negative gradient = the direction you'd walk to descend fastest.

3. **Chain Rule by Hand** (20 min): For a 3-layer network with specific weights and a single input, compute the forward pass, then compute all gradients by hand using the chain rule. Verify every intermediate value with PyTorch autograd. This exercise alone is worth more than a textbook chapter.

4. **Saddle Point Escape** (15 min): Run gradient descent on $f(x, y) = x^2 - y^2$ starting near the origin. Watch it get stuck at the saddle point. Now add noise (simulating SGD). Watch it escape. The noise is doing something useful.

### Key Connections to Deep Learning

- **Gradient descent is literally the training algorithm**: Every time you call `loss.backward()` and `optimizer.step()`, you are computing a gradient and taking a step in the negative gradient direction. If you do not understand gradients, you do not understand training.

- **The chain rule IS backpropagation**: Backpropagation is not a special algorithm. It is the chain rule applied systematically from output to input, with intermediate results cached (the forward pass). Understanding this transforms backprop from a black box into an obvious consequence of calculus.

- **Learning rate and Taylor expansion**: The Taylor expansion says $f(x - \eta \nabla f) \approx f(x) - \eta \|\nabla f\|^2$. This is only accurate for small $\eta$. Too large a learning rate breaks the linear approximation, and the loss increases instead of decreasing. This is why learning rate tuning matters.

- **Saddle points and the blessing of dimensionality**: In a network with millions of parameters, the loss landscape is a surface in million-dimensional space. At a critical point, each direction is independently either a minimum or maximum direction. The probability that ALL directions are minima (a true local minimum) is exponentially small. Most critical points are saddle points. SGD's noise makes it naturally unstable at saddle points, which is why it works better than full-batch gradient descent in practice.

---

## Session 3: Probability and Information Theory

**Duration:** 3 hours (with breaks)

### Learning Objectives

By the end of this session, the apprentice will be able to:

1. Manipulate random variables, expectations, and variances fluently.
2. Apply Bayes' theorem and connect it to the generative vs discriminative model distinction.
3. Identify and use common distributions (Gaussian, Bernoulli, Categorical) in DL contexts.
4. Derive MLE for simple models and explain MAP as regularized MLE.
5. Compute and interpret KL divergence and connect it to the VAE objective.
6. Derive cross-entropy loss from maximum likelihood estimation.
7. Explain entropy, mutual information, and their roles in DL.

### Topics and Time Allocation

| Time | Topic | Key Insight |
|------|-------|-------------|
| 0:00–0:20 | Random variables, expectation, variance | Every prediction a network makes is a point estimate of an expectation. Variance $=$ uncertainty. |
| 0:20–0:45 | Bayes' theorem and generative vs discriminative | Discriminative models learn $P(y \mid x)$ directly. Generative models learn $P(x \mid y)P(y)$ and invert with Bayes. |
| 0:45–0:55 | **Break** | |
| 0:55–1:20 | Common distributions in DL | Gaussian: weight init, VAEs, noise. Bernoulli: binary classification, dropout. Categorical: multiclass output (softmax). |
| 1:20–1:50 | MLE and MAP estimation | MLE = find parameters that make the data most likely. MAP = MLE + prior = MLE + regularization. L2 regularization IS a Gaussian prior on weights. |
| 1:50–2:00 | **Break** | |
| 2:00–2:25 | KL divergence | $D_{KL}(p \| q)$ = "extra bits needed when you use $q$ to encode data from $p$." Asymmetric. The core of the VAE loss. |
| 2:25–2:45 | Entropy and cross-entropy | Entropy = optimal encoding length. Cross-entropy = encoding length when you use the wrong distribution. Cross-entropy loss = negative log-likelihood. Same thing. |
| 2:45–3:00 | Information theory: mutual information | MI = how much knowing X tells you about Y. Feature selection, representation learning, InfoGAN. |

### Intuition-Building Exercises

1. **Bayes in Action** (10 min): Medical test example with concrete numbers. Prior probability of disease = 0.1%. Test accuracy = 99%. You test positive. What is the actual probability you have the disease? Work through Bayes' theorem. Be surprised. Understand why priors matter.

2. **MLE by Hand** (15 min): Given 10 coin flips (7 heads, 3 tails), derive the MLE estimate for the bias parameter. Then add a Beta prior and derive the MAP estimate. Watch how the prior "regularizes" the estimate toward $0.5$. This is exactly what weight decay does to neural network parameters.

3. **KL Divergence Asymmetry** (15 min): For two simple distributions $p$ and $q$, compute $D_{KL}(p \| q)$ and $D_{KL}(q \| p)$. They are different. Understand why: $D_{KL}(p \| q)$ penalizes $q$ for putting low probability where $p$ has high probability (mode-covering). $D_{KL}(q \| p)$ penalizes $q$ for putting high probability where $p$ has low probability (mode-seeking). This asymmetry determines the behavior of VAEs vs GANs.

4. **Cross-Entropy Loss Derivation** (10 min): Start with MLE for a categorical distribution. Take the negative log. Arrive at cross-entropy. Realize you have been using this loss function since day one. Now you know why.

### Key Connections to Deep Learning

- **MLE = minimizing cross-entropy**: When you train a classifier with cross-entropy loss, you are doing maximum likelihood estimation. This is not a coincidence — cross-entropy loss was derived FROM MLE. Understanding this means you understand why we use this loss and when to use alternatives.

- **MAP = MLE + regularization**: Adding an $L_2$ penalty to your loss is mathematically identical to placing a Gaussian prior on your weights and doing MAP estimation instead of MLE. Regularization is not a hack — it is a principled statement about your prior beliefs.

- **KL divergence and VAEs**: The VAE loss has two terms: reconstruction loss (how well can you reconstruct the input?) and KL divergence (how close is your learned latent distribution to the prior?). Without understanding KL divergence, the VAE loss is a mystery. With it, the loss is a direct consequence of variational inference.

- **Entropy and information bottleneck**: The information bottleneck theory suggests that neural networks learn by first fitting the data (increasing mutual information between hidden layers and labels) and then compressing (decreasing mutual information between hidden layers and input). Entropy quantifies this compression.

- **Gaussian distributions everywhere**: Weight initialization (Xavier, He) uses Gaussian distributions with specific variances. The reparameterization trick in VAEs samples from a Gaussian. Gaussian noise is added in diffusion models. Batch normalization makes activations approximately Gaussian. Understanding the Gaussian distribution is not optional.

---

## Weekly Assessment Criteria

By the end of Week 3, the apprentice should demonstrate:

1. **Mechanical fluency**: Can compute matrix multiplications, derivatives, chain rule applications, and probability calculations by hand without errors.

2. **Conceptual understanding**: Can explain, without equations, why the chain rule is backpropagation, why cross-entropy is the right loss, and why KL divergence appears in VAEs.

3. **Code fluency**: Can implement the core operations in NumPy and PyTorch, and can verify mathematical derivations with code.

4. **Connection to DL**: For every concept covered, can name at least one specific place it appears in deep learning practice.

## Resources

### Primary References
- Goodfellow, Bengio, Courville: *Deep Learning*, Chapters 2–4 (the math chapters)
- 3Blue1Brown: *Essence of Linear Algebra* (YouTube series — best visual intuition available)
- 3Blue1Brown: *Essence of Calculus* (YouTube series)

### Supplementary
- Gilbert Strang: *Linear Algebra and Its Applications*
- Boyd & Vandenberghe: *Convex Optimization* (Chapters 1–3 for convexity intuition)
- Cover & Thomas: *Elements of Information Theory* (for KL divergence and entropy depth)
- Petersen & Pedersen: *The Matrix Cookbook* (reference for matrix calculus identities)
