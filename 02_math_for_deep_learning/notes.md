# Module 02: Math for Deep Learning — Reference Notes

These are your reference notes for the mathematical foundations of deep learning.
Every concept follows the same structure: Intuition, Math, Code, DL Connection.
Bookmark this document. You will come back to it.

---

## Part 1: Linear Algebra for Deep Learning

---

### 1.1 Vectors — Points in Representation Space

#### Intuition

A vector is a list of numbers. But that framing is useless for deep learning.

Think of a vector as a **point in space**. A 2D vector is a point on a plane. A 3D vector
is a point in a room. A 768-dimensional vector is a point in the space where BERT
representations live.

When a neural network processes an image and produces a 512-dimensional feature vector,
that vector IS the network's understanding of the image. Similar images produce nearby
vectors. Dissimilar images produce distant vectors. The entire game of representation
learning is about producing vectors where geometric proximity equals semantic similarity.

#### The Math

A vector $\mathbf{x}$ in $\mathbb{R}^n$:

$$\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$$

Key operations:
- Addition: $\mathbf{x} + \mathbf{y} = [x_1 + y_1, x_2 + y_2, \ldots, x_n + y_n]^T$
- Scalar multiplication: $c \mathbf{x} = [c x_1, c x_2, \ldots, c x_n]^T$
- These two operations define a **vector space**. All of linear algebra follows.

#### Code

```python
import numpy as np
import torch

# A vector as a point in representation space
image_features = np.array([0.2, -0.5, 0.8, 0.1, -0.3])
text_features  = np.array([0.3, -0.4, 0.7, 0.2, -0.2])

# Euclidean distance — are these representations similar?
distance = np.linalg.norm(image_features - text_features)
print(f"Distance: {distance:.4f}")  # Small = similar

# In PyTorch — this is what happens inside every layer
x = torch.tensor([1.0, 2.0, 3.0])
print(x.shape)  # torch.Size([3]) — a point in 3D space
```

#### DL Connection

Every hidden layer in a neural network produces a vector. The dimensionality of that
vector is the "width" of the layer. A layer with 256 neurons produces a 256-dimensional
vector. The magic of deep learning is that these vectors, through training, come to
encode meaningful structure: nearby vectors represent similar concepts.

Word2Vec demonstrated this spectacularly: the vector for "king" minus "man" plus "woman"
produces a vector closest to "queen." This works because the learned vector space has
linear structure that mirrors semantic relationships.

---

### 1.2 Dot Products — Similarity as Computation

#### Intuition

The dot product of two vectors answers the question: **how much do these vectors agree?**

If two vectors point in the same direction, their dot product is large and positive.
If they point in opposite directions, it is large and negative. If they are perpendicular,
it is zero — they have nothing in common.

A single neuron computes a dot product between its weight vector and its input, then adds
a bias. The weight vector is a **learned template**. The dot product measures how well the
input matches the template. High match = high activation.

#### The Math

For vectors $\mathbf{x}, \mathbf{w} \in \mathbb{R}^n$:

$$\mathbf{x} \cdot \mathbf{w} = \sum_{i=1}^{n} x_i w_i = \|\mathbf{x}\| \|\mathbf{w}\| \cos(\theta)$$

where $\theta$ is the angle between the vectors.

Cosine similarity normalizes this:

$$\text{cos\_sim}(\mathbf{x}, \mathbf{w}) = \frac{\mathbf{x} \cdot \mathbf{w}}{\|\mathbf{x}\| \|\mathbf{w}\|}$$

This removes the effect of magnitude and purely measures directional agreement.

#### Code

```python
import numpy as np

# A neuron is a learned dot product
input_vector = np.array([1.0, 0.5, -0.3, 0.8])
weights       = np.array([0.2, 0.7, -0.1, 0.4])
bias          = 0.1

# The neuron's pre-activation output
z = np.dot(input_vector, weights) + bias
print(f"Neuron output (pre-activation): {z:.4f}")

# Cosine similarity — used in contrastive learning (CLIP, SimCLR)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = np.array([1.0, 0.0, 1.0])
doc_a = np.array([1.0, 0.0, 0.9])   # Similar to query
doc_b = np.array([0.0, 1.0, 0.0])   # Orthogonal to query

print(f"Similarity to doc_a: {cosine_similarity(query, doc_a):.4f}")  # ~0.99
print(f"Similarity to doc_b: {cosine_similarity(query, doc_b):.4f}")  # ~0.00
```

#### DL Connection

Dot products are everywhere:

- **Neurons**: Each neuron computes $\mathbf{w}^T \mathbf{x} + b$. The dot product $\mathbf{w}^T \mathbf{x}$ measures input-template similarity.
- **Attention**: In transformers, attention scores are computed as dot products between query and key vectors: $\text{score} = QK^T$. High dot product = "this token should attend to that token."
- **Contrastive learning**: CLIP trains by maximizing the dot product between matching image-text pairs and minimizing it for non-matching pairs.
- **Softmax classification**: The logit for class $k$ is $\mathbf{w}_k^T \mathbf{x} + b_k$ — a dot product between the input representation and the class-specific weight vector.

---

### 1.3 Matrices as Transformations

#### Intuition

Forget "a matrix is a grid of numbers." A matrix is a **machine that transforms space**.

Take a 2D plane with points on it. Multiply every point by a 2x2 matrix, and the entire
plane warps. Straight lines stay straight (it is a linear transformation), but they can
rotate, stretch, compress, shear, or reflect.

A neural network layer does exactly this. The weight matrix W transforms the input
vector from one space to another. A layer with 784 inputs and 256 outputs has a
256x784 weight matrix that takes points in 784-dimensional space and maps them to
points in 256-dimensional space.

#### The Math

A matrix $A$ of shape $(m, n)$ maps vectors from $\mathbb{R}^n$ to $\mathbb{R}^m$:

$$\mathbf{y} = A\mathbf{x}$$

where $A$ is $(m \times n)$, $\mathbf{x}$ is $(n \times 1)$, $\mathbf{y}$ is $(m \times 1)$.

Specific 2D transformations:

Rotation by angle $\theta$:

$$R = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}$$

Scaling by factors $s_x, s_y$:

$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

Shearing (horizontal) by factor $k$:

$$H = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$

#### Code

```python
import numpy as np

# A linear layer IS a matrix transformation
# Input: 784-dim (flattened 28x28 MNIST image)
# Output: 256-dim (hidden representation)

np.random.seed(42)
W = np.random.randn(256, 784) * 0.01  # Weight matrix (Xavier-like init)
b = np.zeros(256)                       # Bias vector
x = np.random.randn(784)                # One MNIST image (flattened)

# The forward pass of a linear layer
y = W @ x + b  # Matrix-vector multiplication + bias
print(f"Input shape:  {x.shape}")   # (784,)
print(f"Output shape: {y.shape}")   # (256,)
# We just transformed a point in 784D space to a point in 256D space.

# 2D visualization: rotation matrix
theta = np.pi / 4  # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])
rotated = R @ point
print(f"Original: {point}")
print(f"Rotated:  {rotated}")  # [0.707, 0.707] — 45 degrees from x-axis

# The key insight: the COLUMNS of the matrix tell you where the
# basis vectors end up.
print(f"Column 0 of R (where [1,0] goes): {R[:, 0]}")
print(f"Column 1 of R (where [0,1] goes): {R[:, 1]}")
```

#### DL Connection

The weight matrix of a linear layer is a learned transformation. During training,
gradient descent adjusts this matrix so that the transformation maps inputs to
representations that are useful for the task.

Key insight: the **columns** of the weight matrix tell you where each input dimension
gets mapped. The **rows** tell you what each output neuron is looking for. Reading
weight matrices this way is a powerful debugging tool.

In convolutional neural networks, the "matrix" is actually a convolution kernel, but
the principle is the same: a learned linear transformation of the input. Convolution
is just matrix multiplication with a specific sparse, weight-sharing structure.

---

### 1.4 Matrix Multiplication as Composition of Transformations

#### Intuition

If matrix $A$ represents one transformation and matrix $B$ represents another, then
the matrix product $AB$ represents doing $B$ first, then $A$.

A deep neural network is literally a composition of transformations. Layer 1 applies
$W_1$, then layer 2 applies $W_2$, then layer 3 applies $W_3$. The combined linear effect
(ignoring nonlinearities for a moment) is $W_3 W_2 W_1$.

This is why depth matters: each layer adds another transformation, and compositions
of simple transformations can produce complex ones.

#### The Math

For $A$ of shape $(m, k)$ and $B$ of shape $(k, n)$:

$$C = AB \quad \text{where } C \text{ is } (m, n)$$

$$C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}$$

The inner dimensions must match. The result has the outer dimensions.

Properties:
- NOT commutative: $AB \neq BA$ in general
- Associative: $(AB)C = A(BC)$
- Distributive: $A(B + C) = AB + AC$

#### Code

```python
import numpy as np

# A 3-layer network (linear parts only) as composed transformations
np.random.seed(42)
W1 = np.random.randn(128, 784)  # Layer 1: 784 -> 128
W2 = np.random.randn(64, 128)   # Layer 2: 128 -> 64
W3 = np.random.randn(10, 64)    # Layer 3: 64 -> 10

x = np.random.randn(784)  # Input

# Sequential application (what actually happens in a forward pass)
h1 = W1 @ x
h2 = W2 @ h1
y  = W3 @ h2

# Composed transformation (equivalent for the linear case)
W_composed = W3 @ W2 @ W1  # Shape: (10, 784)
y_composed = W_composed @ x

print(f"Sequential result: {y[:3]}")
print(f"Composed result:   {y_composed[:3]}")
# These are identical! (Up to floating point)

# This is WHY we need nonlinear activations.
# Without them, any deep linear network collapses to a single matrix.
# Depth without nonlinearity is useless.
```

#### DL Connection

This is the most important insight about depth and nonlinearity:

**Without activation functions, a 100-layer network is no more powerful than a single
layer.** Because matrix multiplication composes linearly, $W_{100} \cdots W_2 W_1$ is
just one big matrix. The nonlinear activations (ReLU, GELU, etc.) between layers are
what make depth meaningful. They break the linearity and allow the network to learn
non-linear decision boundaries.

The computational cost of matrix multiplication ($O(n^3)$ for square matrices, or $O(mkn)$
in general) dominates neural network training. This is why GPUs matter — they are designed
for massively parallel matrix multiplications. Understanding the shapes and costs of these
multiplications is essential for building efficient models.

---

### 1.5 Eigenvalues and Eigenvectors — PCA Connection

#### Intuition

An eigenvector of a matrix A is a direction that A does not rotate — only stretches
or compresses. The eigenvalue is the stretching factor.

Think of it this way: most vectors, when you multiply them by A, end up pointing in
a completely different direction. But eigenvectors are special. They come out pointing
the same way they went in, just scaled. If the eigenvalue is 2, the vector doubles
in length. If it is 0.5, it halves. If it is negative, it flips.

For PCA: the eigenvectors of the data's covariance matrix point in the directions of
maximum variance. The eigenvalues tell you how much variance each direction captures.
PCA says: keep the directions with the biggest eigenvalues, discard the rest. You have
found the most informative axes of your data.

#### The Math

For a square matrix $A$, the eigenvector $\mathbf{v}$ and eigenvalue $\lambda$ satisfy:

$$A\mathbf{v} = \lambda \mathbf{v}$$

Equivalently: $(A - \lambda I)\mathbf{v} = 0$

This has a non-trivial solution when:

$$\det(A - \lambda I) = 0 \quad \text{(the characteristic equation)}$$

For PCA specifically:
1. Center the data: $X_{\text{centered}} = X - \text{mean}(X)$
2. Compute the covariance matrix: $C = \frac{1}{n} X_{\text{centered}}^T X_{\text{centered}}$
3. Find eigenvectors of $C$ — these are the principal components
4. Project data onto the top-$k$ eigenvectors for dimensionality reduction

#### Code

```python
import numpy as np

# Eigendecomposition
A = np.array([[3, 1],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")    # [4, 2]
print(f"Eigenvectors:\n{eigenvectors}") # Columns are eigenvectors

# Verify: A @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = A @ v
    lam_v = lam * v
    print(f"A @ v_{i} = {Av}, lambda_{i} * v_{i} = {lam_v}")
    # They match.

# PCA from scratch
np.random.seed(42)
# Generate correlated 2D data
mean = [0, 0]
cov = [[3, 2],
       [2, 1.5]]
X = np.random.multivariate_normal(mean, cov, size=200)

# Step 1: Center
X_centered = X - X.mean(axis=0)

# Step 2: Covariance matrix
C = (X_centered.T @ X_centered) / (len(X) - 1)

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(C)

# Sort by eigenvalue (descending)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\nPCA Results:")
print(f"Variance explained by PC1: {eigenvalues[0]:.2f}")
print(f"Variance explained by PC2: {eigenvalues[1]:.2f}")
print(f"PC1 direction: {eigenvectors[:, 0]}")

# Step 4: Project onto first principal component
X_projected = X_centered @ eigenvectors[:, 0:1]  # Reduce 2D -> 1D
print(f"Original shape: {X.shape}, Projected shape: {X_projected.shape}")
```

#### DL Connection

- **PCA on features**: Running PCA on the activations of a neural network layer reveals which directions in feature space carry the most information. This is a diagnostic tool for understanding what the network has learned.

- **Covariance and Batch Normalization**: Batch normalization standardizes activations to have zero mean and unit variance. This is equivalent to making the diagonal of the covariance matrix all ones. The full version — whitening — would make the entire covariance matrix the identity, decorrelating all features. Some normalization methods (like Decorrelated Batch Normalization) do exactly this using eigendecomposition.

- **Weight matrix spectrum**: The eigenvalues (or singular values) of a layer's weight matrix tell you about training dynamics. If the largest eigenvalue is much bigger than the smallest, the layer has a large "condition number" and gradients through it may be unstable. Spectral normalization in GANs constrains the largest singular value to be 1, stabilizing training.

---

### 1.6 SVD — The Swiss Army Knife of Linear Algebra

#### Intuition

The Singular Value Decomposition (SVD) factorizes any matrix into three parts:
$A = U \Sigma V^T$

Think of it as decomposing a transformation into three steps:
1. $V^T$: Rotate the input
2. $\Sigma$: Scale along each axis (the singular values)
3. $U$: Rotate the output

The singular values (diagonal of $\Sigma$) tell you how important each axis is.
If you keep only the top-$k$ singular values and zero out the rest, you get the
**best rank-$k$ approximation** to the original matrix. This is optimal in a precise
mathematical sense (Eckart-Young theorem).

#### The Math

For any matrix $A$ of shape $(m, n)$:

$$A = U \Sigma V^T$$

where:
- $U$ is $(m, m)$ orthogonal — left singular vectors
- $\Sigma$ is $(m, n)$ diagonal — singular values (non-negative, sorted descending)
- $V^T$ is $(n, n)$ orthogonal — right singular vectors

The rank-$k$ approximation:

$$A_k = U_{:,:k} \, \Sigma_{:k,:k} \, V_{:k,:}$$

This minimizes $\|A - A_k\|_F$ over all rank-$k$ matrices.

#### Code

```python
import numpy as np

# SVD for image compression
# Simulate a grayscale image as a matrix
np.random.seed(42)
image = np.random.randn(100, 100)  # In practice: load a real image

U, s, Vt = np.linalg.svd(image, full_matrices=False)
# U: (100, 100), s: (100,), Vt: (100, 100)

# Compress by keeping only top-k singular values
for k in [5, 10, 20, 50]:
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    # Original storage: 100 * 100 = 10,000 values
    # Compressed storage: 100*k + k + k*100 = k*(201) values
    compression_ratio = (100 * 100) / (k * (100 + 1 + 100))
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    print(f"k={k:2d}: compression ratio = {compression_ratio:.1f}x, "
          f"relative error = {error:.4f}")

# Cumulative explained variance (like PCA)
explained = np.cumsum(s**2) / np.sum(s**2)
print(f"\nSingular values needed for 90% of 'energy': "
      f"{np.searchsorted(explained, 0.9) + 1}")
print(f"Singular values needed for 99% of 'energy': "
      f"{np.searchsorted(explained, 0.99) + 1}")
```

#### DL Connection

- **LoRA (Low-Rank Adaptation)**: The key insight behind LoRA is that weight updates during fine-tuning are approximately low-rank. Instead of updating a full $(d \times d)$ weight matrix, LoRA learns two small matrices $A$ $(d \times r)$ and $B$ $(r \times d)$ where $r \ll d$, and adds their product $BA$ to the frozen weights. This is directly motivated by SVD — the update lives in a low-rank subspace.

- **Truncated SVD for embeddings**: Large embedding matrices (e.g., vocabulary embeddings with 50,000 x 768 entries) can be compressed using SVD, reducing memory and computation with minimal quality loss.

- **Understanding model capacity**: The effective rank of weight matrices in trained networks is often much lower than their nominal rank. This suggests the network is not using its full capacity, which motivates techniques like pruning and distillation.

---

### 1.7 Norms — Measuring Size

#### Intuition

A norm measures the "size" of a vector or matrix. Different norms measure size in
different ways, and each way has a specific meaning in deep learning.

- **L1 norm** (Manhattan distance): Sum of absolute values. Promotes sparsity — it pushes
  small values to exactly zero. Used in L1 regularization (Lasso).

- **L2 norm** (Euclidean distance): Square root of sum of squares. The "natural" distance.
  Used in L2 regularization (weight decay) and virtually all distance computations.

- **Frobenius norm**: The L2 norm generalized to matrices. Square root of sum of all
  squared entries. Used in spectral normalization and matrix approximation error.

#### The Math

$$\|x\|_1 = \sum_i |x_i|$$

$$\|x\|_2 = \sqrt{\sum_i x_i^2}$$

$$\|x\|_\infty = \max_i |x_i|$$

$$\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{trace}(A^T A)}$$

#### Code

```python
import numpy as np

x = np.array([3.0, -4.0, 0.0, 1.0, -0.5])

l1 = np.linalg.norm(x, ord=1)     # = 8.5
l2 = np.linalg.norm(x, ord=2)     # = 5.148
linf = np.linalg.norm(x, ord=np.inf)  # = 4.0

print(f"L1 norm:   {l1:.3f}")
print(f"L2 norm:   {l2:.3f}")
print(f"L-inf norm: {linf:.3f}")

# L1 vs L2 regularization effect
# Simulate gradient descent with L1 vs L2 penalty
w = np.array([0.3, -0.1, 0.05, -0.02, 0.001])
lr = 0.01
lambda_reg = 0.1

# L2 regularization shrinks all weights proportionally
w_after_l2 = w - lr * lambda_reg * 2 * w  # Gradient of ||w||_2^2 is 2w
print(f"\nOriginal weights: {w}")
print(f"After L2 step:    {w_after_l2}")
# All weights shrink, but none become exactly zero.

# L1 regularization pushes small weights to zero
w_after_l1 = w - lr * lambda_reg * np.sign(w)  # Gradient of ||w||_1 is sign(w)
print(f"After L1 step:    {w_after_l1}")
# The smallest weight (0.001) gets pushed negative — in practice, you'd clip to zero.
# L1 produces SPARSE solutions.
```

#### DL Connection

- **L2 regularization = weight decay**: Adding $\lambda \|W\|_2^2$ to the loss penalizes large weights. The gradient of this term is $2\lambda W$, which shrinks all weights toward zero. This is called "weight decay" because it decays the weights at each step.

- **L1 regularization = sparsity**: L1 pushes weights to exactly zero, producing sparse networks. This is the mathematical foundation of pruning.

- **Gradient clipping**: When gradients have large L2 norm (gradient explosion), we clip them: if $\|g\|_2 > \text{threshold}$, replace $g$ with $g \cdot \text{threshold} / \|g\|_2$. This rescales the gradient to have a fixed norm without changing its direction.

- **Spectral norm**: The largest singular value of a matrix is its spectral norm $\|A\|_2$. Spectral normalization in GANs divides weight matrices by their spectral norm, ensuring the discriminator is Lipschitz-continuous with constant 1.

---

### 1.8 Broadcasting Rules

#### Intuition

Broadcasting is how NumPy and PyTorch handle operations between tensors of different shapes.
Instead of requiring exact shape matches, broadcasting "virtually expands" smaller tensors
to match larger ones, without actually copying data.

The rules are simple but tricky:
1. Align shapes from the right
2. Dimensions match if they are equal OR one of them is 1
3. A dimension of size 1 is "broadcast" (repeated) to match the other

Getting broadcasting wrong is one of the most common sources of silent bugs. The code
runs fine, produces no errors, but computes the wrong thing because shapes were
misaligned.

#### The Math

```
Shape alignment (right to left):
(3, 4) + (4,)    -> (3, 4) + (1, 4) -> (3, 4)    OK
(3, 4) + (3, 1)  -> (3, 4)                        OK
(3, 4) + (3,)    -> ERROR: 4 != 3                  FAIL
(2, 3, 4) + (3, 4) -> (2, 3, 4) + (1, 3, 4) -> (2, 3, 4)  OK
```

#### Code

```python
import numpy as np

# Common DL broadcasting patterns

# Pattern 1: Adding bias to a batch
batch = np.random.randn(32, 256)  # (batch_size, features)
bias = np.random.randn(256)        # (features,)
result = batch + bias              # (32, 256) + (256,) -> (32, 256)
# Each row gets the same bias added. Correct.

# Pattern 2: Scaling features
scale = np.random.randn(1, 256)    # (1, features)
result = batch * scale             # (32, 256) * (1, 256) -> (32, 256)
# Each feature gets scaled by the same factor across the batch.

# TRAP: A common bug
labels = np.array([0, 1, 2])              # Shape: (3,)
predictions = np.array([[0.1], [0.9], [0.2]])  # Shape: (3, 1)
# labels + predictions -> (3,) + (3, 1) -> (1, 3) + (3, 1) -> (3, 3) !!!
# You expected (3,) but got (3, 3). Silent bug.
result = labels + predictions
print(f"Expected shape: (3,), Got shape: {result.shape}")
print(result)
# Fix: ensure consistent shapes
labels_fixed = labels.reshape(-1, 1)  # (3, 1)
result_fixed = labels_fixed + predictions  # (3, 1) + (3, 1) -> (3, 1)
print(f"Fixed shape: {result_fixed.shape}")
```

#### DL Connection

Broadcasting is not abstract — it happens in every forward pass:
- Adding a bias vector to every sample in a batch
- Applying layer normalization (subtracting mean, dividing by std — both computed per-sample)
- Masking attention scores (the mask broadcasts over the batch and head dimensions)
- Computing pairwise distances between two sets of vectors

When debugging shape errors in PyTorch, the first thing to check is broadcasting.
Print shapes obsessively until the bug is found.

---

## Part 2: Calculus and Optimization

---

### 2.1 Derivatives as Local Linear Approximation

#### Intuition

The derivative of $f$ at point $x$ tells you: if I nudge $x$ by a tiny amount $h$, how much
does $f$ change?

$$f(x + h) \approx f(x) + f'(x) \cdot h$$

This is not an approximation technique. This IS the definition of the derivative.
The derivative is the slope of the best linear approximation to $f$ near $x$.

If you zoom into any smooth function at any point, it looks like a straight line.
The derivative gives you the slope of that line.

#### The Math

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

Key derivatives you must know cold:

$$\begin{aligned}
\frac{d}{dx} [x^n] &= n x^{n-1} \\
\frac{d}{dx} [e^x] &= e^x \\
\frac{d}{dx} [\ln(x)] &= \frac{1}{x} \\
\frac{d}{dx} [\sin(x)] &= \cos(x) \\
\frac{d}{dx} [\sigma(x)] &= \sigma(x)(1 - \sigma(x)) \\
\frac{d}{dx} [\text{ReLU}(x)] &= \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases} \quad \text{(set to 0 at 0 by convention)} \\
\frac{d}{dx} [\tanh(x)] &= 1 - \tanh^2(x)
\end{aligned}$$

#### Code

```python
import numpy as np

def numerical_derivative(f, x, h=1e-7):
    """Central difference — more accurate than forward difference."""
    return (f(x + h) - f(x - h)) / (2 * h)

# Example: derivative of sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative_analytical(x):
    s = sigmoid(x)
    return s * (1 - s)

x = 1.0
numerical = numerical_derivative(sigmoid, x)
analytical = sigmoid_derivative_analytical(x)
print(f"Numerical derivative:  {numerical:.8f}")
print(f"Analytical derivative: {analytical:.8f}")
print(f"Difference:            {abs(numerical - analytical):.2e}")
# They match to ~10 decimal places. This is how you verify gradients.
```

#### DL Connection

Gradient checking — comparing analytical gradients (from backprop) to numerical gradients
(from the formula above) — is the standard technique for verifying that your backward
pass is implemented correctly. If you ever implement a custom autograd function, you must
gradient-check it. PyTorch provides `torch.autograd.gradcheck` for this purpose.

---

### 2.2 Partial Derivatives and Gradients

#### Intuition

For a function of multiple variables f(x, y, z), the partial derivative with respect
to x tells you: if I nudge ONLY x and hold y, z fixed, how much does f change?

The gradient is the vector of ALL partial derivatives. It points in the direction of
steepest ascent. To minimize a function, go in the opposite direction. That is gradient
descent.

Think of yourself standing on a hilly landscape. The gradient tells you which way is
uphill. You want to go downhill. So you walk in the negative gradient direction.

#### The Math

For $f: \mathbb{R}^n \to \mathbb{R}$:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^T$$

Properties:
- The gradient is perpendicular to the level curves (contour lines) of $f$
- The gradient points in the direction of maximum increase of $f$
- The magnitude of the gradient tells you how steep that maximum increase is

#### Code

```python
import numpy as np

# The gradient of a quadratic function
# f(x, y) = x^2 + 3*y^2
# grad f = [2x, 6y]

def f(xy):
    x, y = xy
    return x**2 + 3 * y**2

def grad_f(xy):
    x, y = xy
    return np.array([2*x, 6*y])

point = np.array([2.0, 1.0])
print(f"f({point}) = {f(point)}")
print(f"grad f({point}) = {grad_f(point)}")

# Verify numerically
def numerical_gradient(f, x, h=1e-7):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

print(f"Numerical gradient: {numerical_gradient(f, point)}")

# Gradient descent
point = np.array([4.0, 3.0])
lr = 0.1
for step in range(20):
    if step % 5 == 0:
        print(f"Step {step:2d}: point = {point}, f = {f(point):.4f}")
    point = point - lr * grad_f(point)
print(f"Step 20: point = {point}, f = {f(point):.6f}")
# Converges toward [0, 0], the minimum.
```

#### Gradient Ascent on a Contour Plot (ASCII Visualization)

```
    y
    |
  3 |   . . . . . . . . . .       Contour lines of f(x,y) = x^2 + 3y^2
    |  .                     .
  2 | .   _______________     .    The ellipses are level curves.
    |.   /               \    .    The gradient at point P = (2,1) is [4, 6].
  1 |   /    _________    \   .    It points AWAY from the minimum.
    |  |    /         \    |  .    We go in the OPPOSITE direction.
  0 +--|---|---- * ----|---|--.--> x
    |  |    \_________/    |  .         * = minimum at (0,0)
 -1 |   \                 /   .         P = point (2,1)
    |    \_______________/    .         arrow from P points toward (-4, -6)
 -2 |  .                     .                = negative gradient direction
    |   . . . . . . . . . .
 -3 |
    +--+--+--+--+--+--+--+--+
   -4 -3 -2 -1  0  1  2  3  4
```

The gradient [4, 6] at point (2, 1) points "uphill." Gradient descent follows
[-4, -6], which points toward the minimum at the origin.

#### DL Connection

**This is training.** Every call to `loss.backward()` computes the gradient of the loss
with respect to every parameter. Every call to `optimizer.step()` takes a step in the
negative gradient direction. The entire training loop is:

```python
for batch in dataloader:
    loss = model(batch)          # Forward pass: compute loss
    loss.backward()              # Backward pass: compute gradients
    optimizer.step()             # Update: step in negative gradient direction
    optimizer.zero_grad()        # Reset gradients for next iteration
```

Every line here is a direct consequence of the math above.

---

### 2.3 The Chain Rule — THIS IS BACKPROPAGATION

#### Intuition

The chain rule says: if you have a composition of functions, the derivative of the
whole thing is the product of the derivatives of each piece.

This is not a minor convenience. This IS backpropagation. When you have a neural
network with layers $f_1, f_2, f_3$, and a loss $\mathcal{L}$, computing $\frac{d\mathcal{L}}{dx}$ requires:

$$\frac{d\mathcal{L}}{dx} = \frac{d\mathcal{L}}{df_3} \cdot \frac{df_3}{df_2} \cdot \frac{df_2}{df_1} \cdot \frac{df_1}{dx}$$

Each factor corresponds to one layer. Backpropagation simply computes these factors
from right to left (output to input), caching intermediate results from the forward
pass to avoid redundant computation.

There is no "backpropagation algorithm" separate from the chain rule. Backpropagation
IS the chain rule, applied to a computation graph, with efficient reuse of intermediate
values.

#### The Math

For the composition $y = f(g(x))$:

$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$

For a deeper composition $y = f(g(h(x)))$:

$$\frac{dy}{dx} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

In Leibniz notation (easier to see the "chain"):

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dv} \cdot \frac{dv}{dx}$$

where $u = g(h(x))$, $v = h(x)$.

The intermediate variables cancel like fractions. That is the chain.

#### Code — A Complete Backpropagation Example

```python
import numpy as np

# A concrete 3-layer network, fully worked out.
# Architecture: input(1) -> hidden1(2) -> hidden2(2) -> output(1)
# Activation: sigmoid everywhere
# Loss: MSE

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Fixed weights for reproducibility
W1 = np.array([[0.5], [0.3]])    # (2, 1)
b1 = np.array([0.1, -0.2])       # (2,)
W2 = np.array([[0.4, -0.6], [0.7, 0.2]])  # (2, 2)
b2 = np.array([0.0, 0.1])        # (2,)
W3 = np.array([[0.8, -0.5]])     # (1, 2)
b3 = np.array([0.05])            # (1,)

x = np.array([1.5])              # Input
y_true = np.array([0.8])         # Target

# ============ FORWARD PASS ============
# Layer 1
z1 = W1 @ x + b1                  # Pre-activation: (2,)
a1 = sigmoid(z1)                   # Activation: (2,)

# Layer 2
z2 = W2 @ a1 + b2                 # Pre-activation: (2,)
a2 = sigmoid(z2)                   # Activation: (2,)

# Layer 3 (output)
z3 = W3 @ a2 + b3                 # Pre-activation: (1,)
y_pred = sigmoid(z3)               # Prediction: (1,)

# Loss (MSE)
loss = 0.5 * (y_pred - y_true)**2

print("=== FORWARD PASS ===")
print(f"z1 = {z1}, a1 = {a1}")
print(f"z2 = {z2}, a2 = {a2}")
print(f"z3 = {z3}, y_pred = {y_pred}")
print(f"loss = {loss}")

# ============ BACKWARD PASS (Chain Rule) ============
# Start from the loss, work backward.

# dL/dy_pred = y_pred - y_true
dL_dy_pred = y_pred - y_true

# dL/dz3 = dL/dy_pred * dy_pred/dz3 = dL/dy_pred * sigmoid'(z3)
dL_dz3 = dL_dy_pred * sigmoid_deriv(z3)

# dL/dW3 = dL/dz3 * dz3/dW3 = dL/dz3 * a2^T  (outer product)
dL_dW3 = dL_dz3.reshape(-1, 1) @ a2.reshape(1, -1)
dL_db3 = dL_dz3

# dL/da2 = W3^T @ dL/dz3  (backpropagate through the linear layer)
dL_da2 = W3.T @ dL_dz3

# dL/dz2 = dL/da2 * sigmoid'(z2)  (element-wise)
dL_dz2 = dL_da2 * sigmoid_deriv(z2)

# dL/dW2 = dL/dz2 @ a1^T
dL_dW2 = dL_dz2.reshape(-1, 1) @ a1.reshape(1, -1)
dL_db2 = dL_dz2

# dL/da1 = W2^T @ dL/dz2
dL_da1 = W2.T @ dL_dz2

# dL/dz1 = dL/da1 * sigmoid'(z1)
dL_dz1 = dL_da1 * sigmoid_deriv(z1)

# dL/dW1 = dL/dz1 @ x^T
dL_dW1 = dL_dz1.reshape(-1, 1) @ x.reshape(1, -1)
dL_db1 = dL_dz1

print("\n=== BACKWARD PASS (manual chain rule) ===")
print(f"dL/dW3 = {dL_dW3}")
print(f"dL/dW2 = {dL_dW2}")
print(f"dL/dW1 = {dL_dW1}")

# ============ VERIFY WITH PYTORCH ============
import torch

W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)
W3_t = torch.tensor(W3, dtype=torch.float64, requires_grad=True)
b3_t = torch.tensor(b3, dtype=torch.float64, requires_grad=True)
x_t = torch.tensor(x, dtype=torch.float64)
y_true_t = torch.tensor(y_true, dtype=torch.float64)

z1_t = W1_t @ x_t + b1_t
a1_t = torch.sigmoid(z1_t)
z2_t = W2_t @ a1_t + b2_t
a2_t = torch.sigmoid(z2_t)
z3_t = W3_t @ a2_t + b3_t
y_pred_t = torch.sigmoid(z3_t)
loss_t = 0.5 * (y_pred_t - y_true_t)**2

loss_t.backward()

print("\n=== PYTORCH AUTOGRAD (verification) ===")
print(f"dL/dW3 = {W3_t.grad.numpy()}")
print(f"dL/dW2 = {W2_t.grad.numpy()}")
print(f"dL/dW1 = {W1_t.grad.numpy()}")
print("\nManual and autograd gradients should match exactly.")
```

#### DL Connection

This is it. This is the entire mechanism by which neural networks learn. Everything else
(optimizers, learning rate schedules, batch normalization, skip connections) is an
engineering improvement on top of this fundamental procedure:

1. Forward pass: compute the output and cache intermediate values
2. Backward pass: apply the chain rule from output to input
3. Update: move parameters in the negative gradient direction

When you understand this at a mechanical level — when you can compute gradients by
hand for a small network and verify them — you understand backpropagation. Everything
else is implementation detail.

**Why skip connections (ResNets) work, through the lens of the chain rule:**

In a plain network: $\frac{d\mathcal{L}}{dx} = \frac{d\mathcal{L}}{df_n} \cdot \frac{df_n}{df_{n-1}} \cdots \frac{df_2}{df_1}$

If any $\frac{df_i}{df_{i-1}}$ is small (saturated activation, poorly conditioned layer), the
product shrinks exponentially. This is the vanishing gradient problem.

With a skip connection: $f_i(x) = g_i(x) + x$, so $\frac{df_i}{dx} = \frac{dg_i}{dx} + I$. The identity
term $I$ means the gradient ALWAYS has a path that bypasses the layer. The product can
never vanish completely. This is why ResNets can train with 1000+ layers.

---

### 2.4 Jacobians and Hessians

#### Intuition

The Jacobian generalizes the gradient to **vector-valued functions**. If $f$ maps $\mathbb{R}^n$ to
$\mathbb{R}^m$, the Jacobian is the $m \times n$ matrix of all partial derivatives. Row $i$, column $j$ tells
you: how does the $i$-th output change when you nudge the $j$-th input?

The Hessian is the matrix of **second derivatives** of a scalar function. It tells you
about **curvature**: is the function bowl-shaped (positive definite Hessian = convex)?
Saddle-shaped (indefinite Hessian)? The Hessian contains the information that second-order
optimization methods like Newton's method use.

#### The Math

Jacobian of $f: \mathbb{R}^n \to \mathbb{R}^m$:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

Hessian of $f: \mathbb{R}^n \to \mathbb{R}$:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & & \ddots \end{bmatrix}$$

#### Code

```python
import torch

# Jacobian computation
def f(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]**2])

x = torch.tensor([2.0, 3.0], requires_grad=True)
J = torch.autograd.functional.jacobian(f, x)
print(f"Jacobian:\n{J}")
# Row 0: [2*x0, 1] = [4, 1]
# Row 1: [x1^2, 2*x0*x1] = [9, 12]

# Hessian computation
def g(x):
    return x[0]**2 + 3*x[0]*x[1] + x[1]**3

H = torch.autograd.functional.hessian(g, x)
print(f"\nHessian:\n{H}")
# [[2, 3], [3, 6*x1]] = [[2, 3], [3, 18]]
```

#### DL Connection

- **Jacobians in backprop**: The backward pass through each layer actually multiplies by the transpose of the layer's Jacobian. For a simple linear layer $y = Wx$, the Jacobian is $W$, and backprop multiplies by $W^T$. For nonlinear layers, the Jacobian includes the activation derivative.

- **Hessians and second-order methods**: Newton's method uses the Hessian to take better steps: instead of moving in the gradient direction with a fixed learning rate, it solves $H \cdot \text{step} = -\text{gradient}$. This accounts for curvature and can converge much faster. But computing and inverting the Hessian is $O(n^2)$ in storage and $O(n^3)$ in computation, which is prohibitive for large networks. Approximate methods (L-BFGS, K-FAC) approximate the Hessian cheaply.

- **Hessian eigenvalues and loss landscape**: The eigenvalues of the Hessian at a critical point determine its nature. All positive = local minimum. All negative = local maximum. Mixed = saddle point. Research has shown that the Hessian of neural network losses has a few large eigenvalues and many near-zero eigenvalues, suggesting the loss landscape is nearly flat in most directions.

---

### 2.5 Taylor Expansion — Why Gradient Descent Works

#### Intuition

The Taylor expansion approximates a function near a point using derivatives:

- 0th order: $f(x+h) \approx f(x)$ — the function is approximately constant
- 1st order: $f(x+h) \approx f(x) + f'(x)h$ — the function is approximately linear
- 2nd order: $f(x+h) \approx f(x) + f'(x)h + \frac{1}{2}f''(x)h^2$ — approximately quadratic

Gradient descent uses the 1st-order approximation. It says: near the current point,
the loss is approximately linear, and the gradient tells me the slope. I take a small
step downhill (negative gradient direction).

This works as long as the step is small enough that the linear approximation is valid.
If the step is too large, the approximation breaks down and the loss might increase.
This is why learning rate matters.

#### The Math

Taylor expansion of $f$ around point $a$:

$$f(a + h) = f(a) + f'(a)h + \frac{1}{2}f''(a)h^2 + \frac{1}{6}f'''(a)h^3 + \cdots$$

For gradient descent with learning rate $\eta$:

$$f(x - \eta \nabla f) \approx f(x) - \eta \|\nabla f\|^2 + O(\eta^2)$$

The first-order term is always negative (we are subtracting a positive quantity).
So for small enough $\eta$, the loss ALWAYS decreases. This is the fundamental guarantee
of gradient descent.

But if $\eta$ is too large, the $O(\eta^2)$ term (which depends on curvature/Hessian)
dominates and the loss can increase.

#### Code

```python
import numpy as np

# Demonstrate: GD works for small lr, fails for large lr
def f(x):
    return x**4 - 3*x**2 + 2  # Non-convex function

def grad_f(x):
    return 4*x**3 - 6*x

x_small_lr = 2.0
x_large_lr = 2.0

print("Small learning rate (0.01) — converges:")
for i in range(20):
    x_small_lr -= 0.01 * grad_f(x_small_lr)
    if i % 5 == 0:
        print(f"  Step {i}: x = {x_small_lr:.4f}, f(x) = {f(x_small_lr):.4f}")

print("\nLarge learning rate (0.5) — diverges:")
for i in range(10):
    x_large_lr -= 0.5 * grad_f(x_large_lr)
    if i % 2 == 0:
        print(f"  Step {i}: x = {x_large_lr:.4f}, f(x) = {f(x_large_lr):.4f}")
    if abs(x_large_lr) > 1e6:
        print("  DIVERGED.")
        break
```

#### DL Connection

The learning rate is the single most important hyperparameter in deep learning. The Taylor
expansion tells you why:

- Too small: convergence is guaranteed but painfully slow (you are taking tiny steps)
- Too large: the linear approximation breaks down and training diverges
- Just right: you descend quickly without overshooting

Learning rate schedules (warmup, cosine decay, etc.) are heuristics for navigating this
tradeoff. Warmup starts with a small lr to find a good region, then increases. Cosine
decay starts large and progressively shrinks to fine-tune near a minimum.

Adaptive methods (Adam, RMSProp) effectively use per-parameter learning rates by
normalizing gradients by their running variance. This is a cheap approximation to
second-order information (curvature), letting you take larger steps in flat directions
and smaller steps in steep directions.

---

### 2.6 Convexity and the Loss Landscape

#### Intuition

A function is convex if a line segment between any two points on the function lies above
the function. Visually: it is bowl-shaped. A convex function has exactly one minimum, and
gradient descent is guaranteed to find it.

Neural network losses are NOT convex. They have many local minima, saddle points, and
flat regions. But empirically, training works anyway. Why?

The key insight: in high dimensions, most local minima have loss values very close to the
global minimum. The loss landscape is "benign" — you do not need to find the global minimum
to get a good model. Any local minimum that gradient descent settles into is typically
good enough.

#### The Math

A function $f$ is convex if for all $x, y$ and all $t \in [0, 1]$:

$$f(tx + (1-t)y) \leq tf(x) + (1-t)f(y)$$

Equivalently, the Hessian is positive semi-definite everywhere:

$$H(x) \succeq 0 \quad \text{for all } x \quad \text{(all eigenvalues of } H \text{ are non-negative)}$$

For convex functions:
- Every local minimum is a global minimum
- Gradient descent converges to the global minimum (with appropriate learning rate)

For non-convex functions (neural networks):
- Multiple local minima exist
- Gradient descent converges to SOME local minimum (or saddle point)
- No guarantee it is the global minimum

#### DL Connection

Although neural network losses are non-convex, several factors make optimization tractable:

1. **Overparameterization**: Networks with more parameters than training samples often have
   loss landscapes where ALL local minima achieve zero training loss. The problem is not
   finding A minimum — it is that there are infinitely many, and generalization depends on
   WHICH one you find.

2. **SGD implicit regularization**: The noise in SGD biases it toward "flat" minima — minima
   where the loss does not change much in nearby directions. Flat minima tend to generalize
   better than sharp minima because they are more robust to perturbation.

3. **Loss of convexity but gain of expressiveness**: If we restricted ourselves to convex
   losses (e.g., linear models with MSE loss), we would have easy optimization but limited
   modeling power. The non-convexity IS the source of neural networks' expressiveness.

---

### 2.7 Saddle Points and the Blessing of High Dimensions

#### Intuition

A saddle point is a critical point (gradient = 0) that is a minimum in some directions
and a maximum in others. Think of a mountain pass: you are at the lowest point along the
ridge, but the highest point along the valley.

In high dimensions, saddle points vastly outnumber local minima. Here is why: at a
critical point, each of the $n$ directions is independently either a minimum direction
(positive Hessian eigenvalue) or a maximum direction (negative Hessian eigenvalue). For
the point to be a true local minimum, ALL $n$ directions must be minima. If each direction
has a 50% chance of being a minimum, the probability of a true local minimum is $2^{-n}$.
For $n = 1{,}000{,}000$ (a small network), this is astronomically unlikely.

The good news: SGD naturally escapes saddle points. The stochastic noise pushes the
parameter along the escape directions (the maximum directions of the saddle point). Full
batch gradient descent would get stuck; SGD does not.

#### Code

```python
import numpy as np

# f(x, y) = x^2 - y^2: a saddle point at the origin
def f(x, y):
    return x**2 - y**2

# Gradient descent from near the saddle point
x, y = 0.01, 0.01
lr = 0.1

print("Pure gradient descent near saddle point:")
for i in range(20):
    grad_x = 2 * x
    grad_y = -2 * y
    x -= lr * grad_x
    y -= lr * grad_y
    if i % 5 == 0:
        print(f"  Step {i:2d}: ({x:.6f}, {y:.6f}), f = {f(x, y):.6f}")
# x shrinks toward 0 (minimum direction), y GROWS (maximum direction).
# GD moves TOWARD the saddle in x, AWAY in y.
# Starting very near origin with perfect symmetry = slow progress.

# With noise (simulating SGD):
x, y = 0.01, 0.01
print("\nSGD (with noise) near saddle point:")
np.random.seed(42)
for i in range(20):
    grad_x = 2 * x + np.random.normal(0, 0.05)   # Noisy gradient
    grad_y = -2 * y + np.random.normal(0, 0.05)   # Noisy gradient
    x -= lr * grad_x
    y -= lr * grad_y
    if i % 5 == 0:
        print(f"  Step {i:2d}: ({x:.6f}, {y:.6f}), f = {f(x, y):.6f}")
# The noise kicks y away from 0, and GD amplifies the escape.
# SGD escapes saddle points faster.
```

#### DL Connection

This is one of the deepest insights about why deep learning works:

1. The loss landscape of neural networks has mostly saddle points, not local minima.
2. SGD, through its stochastic noise, naturally escapes saddle points.
3. This is a FEATURE of stochastic optimization, not a limitation.
4. Full-batch gradient descent (no noise) can get stuck at saddle points for very long.

This also explains why larger batch sizes sometimes hurt generalization: less noise
means slower escape from saddle points and flat regions, leading to sharper minima
that generalize worse.

---

## Part 3: Probability and Information Theory

---

### 3.1 Random Variables, Expectation, and Variance

#### Intuition

A random variable is a quantity whose value is uncertain. Every time you sample a batch,
every time you apply dropout, every time you initialize weights — you are drawing from
a random variable.

The expectation (mean) tells you the average outcome. The variance tells you how spread
out the outcomes are — how uncertain you should be.

A neural network's prediction is essentially a point estimate of an expectation. When a
classifier outputs [0.9, 0.1] for [cat, dog], it is saying: the expected outcome is 90%
cat. The spread (variance) tells you how confident it is, but standard networks do not
directly output variance — Bayesian neural networks and ensemble methods do.

#### The Math

For a discrete random variable $X$ with values $x_i$ and probabilities $P(X = x_i)$:

$$\mathbb{E}[X] = \sum_i x_i P(X = x_i)$$

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

$$\text{std}(X) = \sqrt{\text{Var}(X)}$$

Properties:

$$\begin{aligned}
\mathbb{E}[aX + b] &= a\mathbb{E}[X] + b \\
\text{Var}(aX + b) &= a^2 \text{Var}(X) \\
\mathbb{E}[X + Y] &= \mathbb{E}[X] + \mathbb{E}[Y] \quad \text{(always)} \\
\text{Var}(X + Y) &= \text{Var}(X) + \text{Var}(Y) \quad \text{(if independent)}
\end{aligned}$$

#### Code

```python
import numpy as np

# Empirical expectation and variance
np.random.seed(42)
samples = np.random.normal(loc=5.0, scale=2.0, size=10000)

print(f"Empirical mean: {samples.mean():.4f} (true: 5.0)")
print(f"Empirical var:  {samples.var():.4f} (true: 4.0)")
print(f"Empirical std:  {samples.std():.4f} (true: 2.0)")

# In DL: weight initialization variance matters enormously
# Xavier/Glorot initialization: Var(w) = 2 / (fan_in + fan_out)
fan_in, fan_out = 784, 256
xavier_std = np.sqrt(2.0 / (fan_in + fan_out))
weights = np.random.normal(0, xavier_std, size=(fan_out, fan_in))
print(f"\nXavier init for (784 -> 256):")
print(f"  Std of weights: {weights.std():.6f}")
print(f"  Theoretical:    {xavier_std:.6f}")

# Why? If weights have the wrong variance, activations either explode or vanish
# as they pass through layers. Xavier init maintains variance across layers.
```

#### DL Connection

- **Weight initialization**: The variance of initial weights determines whether activations
  grow, shrink, or stay stable as they pass through layers. Xavier initialization sets
  $\text{Var}(w) = \frac{2}{\text{fan\_in} + \text{fan\_out}}$ to maintain activation variance. He initialization sets
  $\text{Var}(w) = \frac{2}{\text{fan\_in}}$, which accounts for ReLU zeroing out half the activations.

- **Batch statistics**: Batch normalization computes the mean and variance of activations
  across a batch, then normalizes. Understanding expectation and variance at a mechanical
  level is required to understand why this works.

- **Dropout**: During training, dropout sets each activation to zero with probability $p$.
  The expectation of the output changes. To compensate, surviving activations are scaled
  by $\frac{1}{1-p}$. This "inverted dropout" ensures that the expected value of each activation
  is unchanged.

---

### 3.2 Bayes' Theorem — Generative vs Discriminative

#### Intuition

Bayes' theorem tells you how to update your beliefs when you see new evidence.

You start with a prior belief P(A) — your belief about A before seeing any data. You
observe evidence B. The likelihood P(B|A) tells you how likely the evidence is if A is
true. Bayes' theorem gives you the posterior P(A|B) — your updated belief about A after
seeing the evidence.

In machine learning:
- **Discriminative models** learn P(y|x) directly. Given input x, what is the label y?
  Most neural networks are discriminative.
- **Generative models** learn P(x|y) and P(y) — the joint distribution P(x, y) — and use
  Bayes' theorem to infer P(y|x) if needed. VAEs, GANs, diffusion models are generative.

#### The Math

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

$$\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$

In classification context:

$$P(y=k \mid x) = \frac{P(x \mid y=k) \cdot P(y=k)}{P(x)}$$

#### Code

```python
# Bayes in action: medical test example
# Disease prevalence: 0.1%
# Test sensitivity (true positive rate): 99%
# Test specificity (true negative rate): 99%

p_disease = 0.001
p_no_disease = 1 - p_disease
p_positive_given_disease = 0.99      # Sensitivity
p_positive_given_no_disease = 0.01   # 1 - Specificity

# P(positive) by total probability
p_positive = (p_positive_given_disease * p_disease +
              p_positive_given_no_disease * p_no_disease)

# Bayes' theorem
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"P(disease | positive test) = {p_disease_given_positive:.4f}")
print(f"That is only {p_disease_given_positive*100:.1f}%!")
# Despite a 99% accurate test, you only have a ~9% chance of actually being sick.
# The prior (0.1% prevalence) dominates.
# This is why priors matter. This is why regularization matters.
```

#### DL Connection

- **Generative vs discriminative**: A discriminative classifier (standard neural network) learns $P(y|x)$ directly by minimizing cross-entropy. A generative model learns $P(x|y)$ or $P(x)$ — the distribution of the data itself. VAEs learn $P(x)$ by modeling a latent variable $z$: $P(x) = \int P(x|z)P(z)\,dz$. Diffusion models learn to reverse a noise process, effectively modeling $P(x)$.

- **Priors = regularization**: In Bayesian statistics, a prior $P(\theta)$ expresses your belief about parameters before seeing data. A Gaussian prior $P(\theta) = \mathcal{N}(0, \sigma^2)$ says you believe weights should be small. The MAP objective $\max P(\theta|\text{data}) = \max P(\text{data}|\theta)P(\theta)$ becomes: maximize log-likelihood minus $\frac{1}{2\sigma^2}\|\theta\|^2$. That L2 penalty IS L2 regularization. The prior IS the regularizer.

---

### 3.3 Common Distributions in Deep Learning

#### Intuition

Three distributions appear everywhere in deep learning. Know them cold.

- **Gaussian (Normal)**: The bell curve. Appears in weight initialization, noise injection,
  VAE latent spaces, and virtually all continuous probability models. The Central Limit
  Theorem tells you why: the sum of many independent random things is approximately
  Gaussian, regardless of what those things are.

- **Bernoulli**: A coin flip. Output is 0 or 1. Appears in binary classification (the
  output of a sigmoid), in dropout (each neuron is kept or dropped), and in binary cross-entropy.

- **Categorical**: A dice roll. Output is one of K classes. Appears in multiclass
  classification (the output of softmax), in language models (next token prediction), and
  in cross-entropy loss.

#### The Math

$$P(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \quad \text{(Gaussian)}$$

$$P(x \mid p) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\} \quad \text{(Bernoulli)}$$

$$P(x=k \mid p_1, \ldots, p_K) = p_k, \quad \sum_k p_k = 1 \quad \text{(Categorical)}$$

#### Code

```python
import numpy as np

# Gaussian: weight initialization
np.random.seed(42)
fan_in = 512
he_std = np.sqrt(2.0 / fan_in)  # He initialization for ReLU
weights = np.random.normal(0, he_std, size=(256, fan_in))
print(f"He init: mean = {weights.mean():.6f}, std = {weights.std():.6f}")
print(f"Expected: mean = 0, std = {he_std:.6f}")

# Bernoulli: dropout
p_keep = 0.8
activations = np.random.randn(1, 256)  # Some layer's output
mask = np.random.binomial(1, p_keep, size=activations.shape)
dropped = activations * mask / p_keep  # Scale to maintain expectation
print(f"\nDropout: {(mask == 0).sum()} out of {mask.size} neurons dropped")
print(f"Mean before dropout: {activations.mean():.4f}")
print(f"Mean after dropout:  {dropped.mean():.4f}")  # Approximately the same

# Categorical: softmax output
logits = np.array([2.0, 1.0, 0.5, -1.0])
exp_logits = np.exp(logits - logits.max())  # Subtract max for numerical stability
probs = exp_logits / exp_logits.sum()
print(f"\nSoftmax probabilities: {probs}")
print(f"Sum: {probs.sum():.4f}")
# This IS a categorical distribution over 4 classes.
```

#### DL Connection

- **Gaussian in VAEs**: The VAE encoder outputs $\mu$ and $\sigma$ for a Gaussian distribution in latent space. The reparameterization trick samples $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$. This allows gradients to flow through the sampling operation.

- **Bernoulli in binary classification**: A binary classifier with sigmoid output models $P(y=1|x) = \sigma(\text{logit})$. The loss is negative log-likelihood of a Bernoulli: $-[y\log(p) + (1-y)\log(1-p)]$ = binary cross-entropy.

- **Categorical in language models**: A language model with softmax output models $P(\text{next\_token} = k \mid \text{context}) = \text{softmax}(\text{logits})_k$. The loss is negative log-likelihood of a categorical: $-\log(p_k)$ where $k$ is the true next token = cross-entropy.

---

### 3.4 Maximum Likelihood Estimation (MLE) and MAP

#### Intuition

MLE asks: what parameters make the observed data most probable?

You have data $D = \{x_1, \ldots, x_n\}$ and a model with parameters $\theta$. The likelihood
is $P(D \mid \theta)$ — the probability of observing this data given these parameters. MLE
finds the $\theta$ that maximizes this.

In practice, we maximize the log-likelihood (equivalent because log is monotonic):

$$\theta_{\text{MLE}} = \arg\max \sum_i \log P(x_i \mid \theta)$$

Minimizing the negative log-likelihood = maximizing the likelihood. Every loss function
in deep learning is, at its core, a negative log-likelihood for some probabilistic model.

MAP adds a prior: $\theta_{\text{MAP}} = \arg\max [\log P(D \mid \theta) + \log P(\theta)]$.
The prior term is regularization.

#### The Math

MLE:

$$\begin{aligned}
\theta_{\text{MLE}} &= \arg\max_\theta P(D \mid \theta) \\
&= \arg\max_\theta \sum_{i=1}^{n} \log P(x_i \mid \theta) \\
&= \arg\min_\theta -\sum_{i=1}^{n} \log P(x_i \mid \theta) \quad \text{[negative log-likelihood]}
\end{aligned}$$

MAP:

$$\begin{aligned}
\theta_{\text{MAP}} &= \arg\max_\theta P(\theta \mid D) \\
&= \arg\max_\theta [\log P(D \mid \theta) + \log P(\theta)] \\
&= \arg\min_\theta \left[-\sum_{i=1}^{n} \log P(x_i \mid \theta) - \log P(\theta)\right]
\end{aligned}$$

If $P(\theta) = \mathcal{N}(0, \sigma^2 I)$, then:

$$-\log P(\theta) = \frac{1}{2\sigma^2} \|\theta\|^2 + \text{const}$$

MAP with a Gaussian prior = MLE with L2 regularization. The prior IS the regularizer.

#### Code

```python
import numpy as np

# MLE for a Gaussian: estimate mean and variance from data
np.random.seed(42)
true_mu, true_sigma = 3.0, 2.0
data = np.random.normal(true_mu, true_sigma, size=100)

# MLE estimates (can be derived by setting d/d(theta) log P = 0)
mu_mle = data.mean()
sigma_mle = data.std()  # Note: MLE uses 1/n, not 1/(n-1)

print(f"True mu = {true_mu}, MLE mu = {mu_mle:.4f}")
print(f"True sigma = {true_sigma}, MLE sigma = {sigma_mle:.4f}")

# MLE for Bernoulli: coin flip
flips = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])  # 7 heads, 3 tails
p_mle = flips.mean()  # MLE for Bernoulli parameter
print(f"\nCoin flip MLE: p = {p_mle}")  # 0.7

# MAP with Beta prior (adds "pseudo-counts")
alpha_prior, beta_prior = 2, 2  # Mild prior toward 0.5
n_heads, n_tails = flips.sum(), len(flips) - flips.sum()
p_map = (n_heads + alpha_prior - 1) / (len(flips) + alpha_prior + beta_prior - 2)
print(f"Coin flip MAP (Beta(2,2) prior): p = {p_map:.4f}")
# MAP is pulled toward 0.5 by the prior. More data = less prior influence.
```

#### DL Connection

When you train a neural network with cross-entropy loss, you are doing MLE. When you add
L2 regularization, you are doing MAP with a Gaussian prior. This is not an analogy — it
is literally the same mathematical objective.

Understanding this unifies many apparently different techniques:
- Cross-entropy loss = negative log-likelihood for categorical distribution
- MSE loss = negative log-likelihood for Gaussian distribution (with fixed variance)
- L2 regularization = Gaussian prior on weights
- L1 regularization = Laplace prior on weights
- Dropout = approximate Bayesian inference (debated, but influential framing)

---

### 3.5 KL Divergence — The Core of VAEs

#### Intuition

KL divergence measures how different two probability distributions are. Specifically,
$D_{\text{KL}}(p \| q)$ measures: if data comes from distribution $p$, how many extra bits do you need
if you encode it using distribution $q$ instead of $p$?

Key properties:
- $D_{\text{KL}}(p \| q) \geq 0$ always (Gibbs' inequality)
- $D_{\text{KL}}(p \| q) = 0$ if and only if $p = q$
- $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ — it is NOT symmetric

The asymmetry matters. $D_{\text{KL}}(p \| q)$ penalizes $q$ for putting LOW probability where $p$ has
HIGH probability. This makes it "mode-covering" — $q$ tries to cover all modes of $p$.
$D_{\text{KL}}(q \| p)$ penalizes $q$ for putting HIGH probability where $p$ has LOW probability. This
makes it "mode-seeking" — $q$ concentrates on the highest-probability regions of $p$.

#### The Math

For discrete distributions:

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log\frac{p(x)}{q(x)} = \sum_x p(x) [\log p(x) - \log q(x)] = \mathbb{E}_p\left[\log p(x) - \log q(x)\right]$$

For continuous distributions:

$$D_{\text{KL}}(p \| q) = \int p(x) \log\frac{p(x)}{q(x)}\,dx$$

For two Gaussians (this is the form used in VAEs):

$$D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

For the VAE case (comparing learned posterior to standard normal prior):

$$D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = -\frac{1}{2} \sum_j \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$

This is the formula you will see in every VAE implementation.

#### Code

```python
import numpy as np

# KL divergence between two discrete distributions
p = np.array([0.4, 0.3, 0.2, 0.1])  # True distribution
q = np.array([0.25, 0.25, 0.25, 0.25])  # Approximation (uniform)

kl_pq = np.sum(p * np.log(p / q))
kl_qp = np.sum(q * np.log(q / p))

print(f"KL(p || q) = {kl_pq:.4f}")
print(f"KL(q || p) = {kl_qp:.4f}")
print(f"Asymmetric! KL(p||q) != KL(q||p)")

# KL divergence for Gaussians (VAE case)
# Encoder outputs mu and log_var for each latent dimension
mu = np.array([0.5, -0.3, 0.8])
log_var = np.array([-0.2, 0.1, -0.5])
sigma_sq = np.exp(log_var)

# KL(q(z|x) || p(z)) where q = N(mu, sigma^2), p = N(0, 1)
kl_per_dim = -0.5 * (1 + log_var - mu**2 - sigma_sq)
kl_total = kl_per_dim.sum()

print(f"\nVAE KL divergence:")
print(f"  Per dimension: {kl_per_dim}")
print(f"  Total: {kl_total:.4f}")

# As mu -> 0 and sigma -> 1, KL -> 0 (posterior matches prior)
mu_zero = np.array([0.0, 0.0, 0.0])
log_var_zero = np.array([0.0, 0.0, 0.0])
kl_zero = -0.5 * (1 + log_var_zero - mu_zero**2 - np.exp(log_var_zero))
print(f"  KL when mu=0, sigma=1: {kl_zero.sum():.4f}")  # 0.0
```

#### DL Connection

The VAE loss (Evidence Lower Bound, or ELBO) has exactly two terms:

$$\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z)) = \text{Reconstruction loss} - \text{KL divergence}$$

- **Reconstruction loss**: How well can the decoder reconstruct the input from the latent code? This pulls the latent codes to be informative.
- **KL divergence**: How close is the encoder's output distribution to the prior N(0,1)? This pulls the latent codes to be "regular" — to fill the latent space smoothly.

The tension between these two terms is the heart of VAE training. Too much weight on
reconstruction = overfitting, irregular latent space. Too much weight on KL = "posterior
collapse," where the encoder ignores the input and outputs the prior.

The $\beta$-VAE multiplies the KL term by $\beta > 1$, increasing the pressure for a regular
latent space, which encourages disentangled representations.

---

### 3.6 Cross-Entropy from Maximum Likelihood — The Full Derivation

#### Intuition

Cross-entropy loss is not arbitrary. It is the mathematically correct loss for
classification, derived directly from maximum likelihood estimation.

Here is the full story: you model each class probability as $p_k = \text{softmax}(z_k)$. The
data says the true class is $y$. The likelihood of observing class $y$ is $p_y$. You want to
maximize this likelihood. Taking the negative log gives you $-\log(p_y)$ = cross-entropy.

Cross-entropy is not "a loss that works well empirically." It is the unique loss that
corresponds to MLE for a categorical distribution parameterized by softmax.

#### The Math — Full Derivation

Step 1: Model the output as a categorical distribution.

$$P(y = k \mid x; \theta) = \text{softmax}(z(x; \theta))_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$$

where $z = f(x; \theta)$ are the logits (network output before softmax).

Step 2: Write the likelihood for one sample $(x, y)$.

$$P(y \mid x; \theta) = \prod_k P(y=k \mid x; \theta)^{\mathbb{1}[y=k]} = \text{softmax}(z)_y$$

(Only the term for the true class $y$ survives because of the indicator.)

Step 3: Write the log-likelihood for the full dataset.

$$\log \mathcal{L}(\theta) = \sum_{i=1}^{N} \log P(y_i \mid x_i; \theta) = \sum_{i=1}^{N} \log \text{softmax}(z_i)_{y_i}$$

Step 4: Minimize the negative log-likelihood.

$$\text{NLL}(\theta) = -\sum_{i=1}^{N} \log \text{softmax}(z_i)_{y_i}$$

Step 5: Recognize this as cross-entropy.

$$H(p_{\text{true}}, p_{\text{model}}) = -\sum_k p_{\text{true}}(k) \log p_{\text{model}}(k)$$

For one-hot $p_{\text{true}}$ (true class has probability 1, others 0):

$$H = -\log p_{\text{model}}(y_{\text{true}}) = \text{NLL for one sample}$$

Cross-entropy = negative log-likelihood. They are the same thing.

#### Code

```python
import numpy as np

# The full derivation in code
logits = np.array([2.0, 1.0, 0.1])  # Raw network output for 3 classes
true_class = 0                        # True label

# Step 1: Softmax (numerically stable)
logits_shifted = logits - logits.max()
exp_logits = np.exp(logits_shifted)
probs = exp_logits / exp_logits.sum()
print(f"Softmax probabilities: {probs}")

# Step 2: Negative log-likelihood
nll = -np.log(probs[true_class])
print(f"NLL (= cross-entropy): {nll:.4f}")

# Step 3: Cross-entropy formula (one-hot encoding)
one_hot = np.array([1, 0, 0])  # True distribution
cross_entropy = -np.sum(one_hot * np.log(probs))
print(f"Cross-entropy:         {cross_entropy:.4f}")
print(f"They are the same: {np.isclose(nll, cross_entropy)}")

# Step 4: Verify with PyTorch
import torch
import torch.nn.functional as F

logits_t = torch.tensor(logits, dtype=torch.float32)
target_t = torch.tensor(true_class)
ce_loss = F.cross_entropy(logits_t.unsqueeze(0), target_t.unsqueeze(0))
print(f"\nPyTorch cross_entropy: {ce_loss.item():.4f}")
# Same value. Because it IS the same thing.
```

#### DL Connection

Every classifier you train with `F.cross_entropy` is doing maximum likelihood estimation.
The cross-entropy loss is telling you: "here are the parameters that make the observed
labels most probable under the model's predicted distribution."

This also explains why cross-entropy works better than MSE for classification. MSE treats
the output as a Gaussian, which does not match the categorical nature of class labels.
Cross-entropy treats the output as a categorical distribution, which is correct. Using the
wrong loss = using the wrong probabilistic model = suboptimal training.

For binary classification: binary cross-entropy = NLL for a Bernoulli distribution,
parameterized by sigmoid output.

---

### 3.7 Entropy — The Minimum Description Length

#### Intuition

Entropy measures the **average surprise** of a random variable. If an outcome is very
predictable, there is low surprise and low entropy. If an outcome is very uncertain,
there is high surprise and high entropy.

Formally: entropy is the minimum number of bits needed, on average, to encode samples
from the distribution.

- A fair coin has entropy 1 bit (you need 1 bit per flip).
- A biased coin (99% heads) has entropy ~0.08 bits (most flips are predictable).
- A fair die has entropy $\log_2(6) \approx 2.58$ bits.

#### The Math

$$H(X) = -\sum_x P(x) \log P(x)$$

(Using $\log$ base 2 for bits, natural $\log$ for nats)

Properties:
- $H(X) \geq 0$
- $H(X) = 0$ if and only if $X$ is deterministic
- $H(X)$ is maximized when $X$ is uniform
- $H(X) \leq \log(K)$ where $K$ is the number of possible outcomes

Cross-entropy:

$$H(p, q) = -\sum_x p(x) \log q(x) = H(p) + D_{\text{KL}}(p \| q)$$

Cross-entropy = entropy + KL divergence. Since $D_{\text{KL}} \geq 0$, cross-entropy $\geq$ entropy.
Equality when $q = p$ (the model perfectly matches the true distribution).

#### Code

```python
import numpy as np

def entropy(p):
    """Compute entropy in nats. Filter out zeros to avoid log(0)."""
    p = p[p > 0]
    return -np.sum(p * np.log(p))

# Fair coin: maximum entropy for 2 outcomes
fair_coin = np.array([0.5, 0.5])
print(f"Fair coin entropy: {entropy(fair_coin):.4f} nats = "
      f"{entropy(fair_coin)/np.log(2):.4f} bits")

# Biased coin: lower entropy
biased_coin = np.array([0.99, 0.01])
print(f"Biased coin entropy: {entropy(biased_coin):.4f} nats = "
      f"{entropy(biased_coin)/np.log(2):.4f} bits")

# Confident classifier: low entropy output = confident
confident = np.array([0.95, 0.03, 0.02])
print(f"\nConfident classifier entropy: {entropy(confident):.4f}")

# Uncertain classifier: high entropy output = uncertain
uncertain = np.array([0.35, 0.33, 0.32])
print(f"Uncertain classifier entropy: {entropy(uncertain):.4f}")

# Maximum entropy (uniform): upper bound
uniform = np.array([1/3, 1/3, 1/3])
print(f"Maximum entropy (uniform):    {entropy(uniform):.4f}")
```

#### DL Connection

- **Entropy as confidence**: The entropy of a classifier's output distribution measures its uncertainty. Low entropy = confident prediction. High entropy = uncertain. This is used in active learning (query the samples where the model is most uncertain) and in confidence calibration.

- **Cross-entropy loss and entropy**: Since $H(p, q) = H(p) + D_{\text{KL}}(p\|q)$, minimizing cross-entropy is equivalent to minimizing $D_{\text{KL}}(p\|q)$ (entropy of the true labels is constant). We are training the model to minimize the "distance" between its predicted distribution and the true distribution.

- **Label smoothing**: Instead of one-hot labels (entropy = 0), label smoothing uses soft labels like [0.9, 0.05, 0.05] (entropy > 0). This prevents the model from becoming overconfident and improves generalization.

---

### 3.8 Mutual Information

#### Intuition

Mutual information $I(X; Y)$ measures how much knowing $X$ tells you about $Y$ (and vice
versa — it is symmetric).

- If $X$ and $Y$ are independent: $I(X; Y) = 0$. Knowing $X$ tells you nothing about $Y$.
- If $X$ completely determines $Y$: $I(X; Y) = H(Y)$. Knowing $X$ tells you everything about $Y$.

Think of it as the "information overlap" between two random variables.

#### The Math

$$\begin{aligned}
I(X; Y) &= H(X) + H(Y) - H(X, Y) \\
&= H(X) - H(X|Y) \\
&= H(Y) - H(Y|X) \\
&= D_{\text{KL}}(P(X,Y) \| P(X)P(Y))
\end{aligned}$$

The last form is illuminating: mutual information is the KL divergence between the
joint distribution and the product of marginals. It measures how far $X$ and $Y$ are from
being independent.

#### Code

```python
import numpy as np

# Mutual information from a joint distribution
# X = weather {sunny, rainy}, Y = umbrella {yes, no}
joint = np.array([[0.3, 0.2],   # P(sunny, umbrella), P(sunny, no umbrella)
                  [0.05, 0.45]]) # P(rainy, umbrella), P(rainy, no umbrella)

# Marginals
p_x = joint.sum(axis=1)  # P(sunny), P(rainy)
p_y = joint.sum(axis=0)  # P(umbrella), P(no umbrella)

print(f"P(weather): {p_x}")
print(f"P(umbrella): {p_y}")

# Mutual information
mi = 0
for i in range(2):
    for j in range(2):
        if joint[i, j] > 0:
            mi += joint[i, j] * np.log(joint[i, j] / (p_x[i] * p_y[j]))

print(f"\nMutual Information I(weather; umbrella) = {mi:.4f} nats")
print(f"H(weather) = {-np.sum(p_x * np.log(p_x)):.4f}")
print(f"H(umbrella) = {-np.sum(p_y * np.log(p_y)):.4f}")
# Knowing about umbrellas tells you something about the weather, and vice versa.
```

#### DL Connection

- **Information Bottleneck Theory**: This theory (Tishby et al.) proposes that deep learning works by finding a representation $T$ that maximizes $I(T; Y)$ (information about the label) while minimizing $I(T; X)$ (information about the input). The network compresses the input, keeping only what is relevant for the task.

- **InfoGAN**: Uses mutual information maximization between latent codes and generated outputs to learn disentangled representations without supervision.

- **Contrastive learning**: Methods like SimCLR and CLIP can be interpreted as maximizing a lower bound on mutual information between different views of the same data point.

- **Feature selection**: Mutual information between features and labels is a principled criterion for selecting the most informative features.

---

## Summary: The Mathematical Toolbox

Here is the complete picture of how these tools connect:

```
LINEAR ALGEBRA                 CALCULUS                    PROBABILITY
  |                              |                            |
  Vectors = representations      Derivatives = gradients      Distributions = outputs
  Matrices = transformations     Chain rule = backprop         MLE = training objective
  Eigenvalues = PCA, spectrum    Taylor = why GD works         KL = VAE loss
  SVD = compression, LoRA        Hessian = curvature           Entropy = cross-entropy loss
  Norms = regularization         Saddle points = landscape     Bayes = generative models
  |                              |                            |
  +----------- NEURAL NETWORKS: all three, simultaneously ---+
```

A forward pass is a sequence of matrix multiplications (linear algebra) followed by
nonlinear activations (analyzed with calculus), producing a probability distribution
(probability theory) over outputs.

A backward pass is the chain rule (calculus) applied to the computation graph, producing
gradients of the loss (a probability-theoretic quantity) with respect to weight matrices
(linear algebra).

Training is optimization (calculus) of a likelihood objective (probability) over
parameters that are matrices (linear algebra).

These are not three separate subjects. In deep learning, they are one subject.
