# Assignment 01: Linear Algebra for Deep Learning — Hands-On Exercises

## Overview

This assignment has one goal: make linear algebra feel like a tool you reach for
naturally when thinking about neural networks. By the end, you should see matrix
multiplications when you look at a forward pass, eigenvectors when you think about
PCA, and broadcasting rules when you write tensor code.

Every exercise has a "why this matters" section. If you skip it, you are missing the
point. The mechanical computation is necessary but not sufficient.

**Estimated time:** 6-8 hours
**Language:** Python (NumPy, with PyTorch for verification)
**Submission:** Jupyter notebook (.ipynb) with all code, outputs, and written answers.

---

## Exercise 1: Matrix Operations from Scratch

**Why this matters:** Every neural network forward pass is a sequence of matrix operations.
If you cannot do these by hand, you cannot debug them in code.

### Task

Implement the following operations using **only basic NumPy** (no `np.linalg`, no `np.matmul`
for the core implementation — use loops or element-wise operations). Then verify your results
against the built-in NumPy functions.

#### 1a. Dot Product

```python
def dot_product(a, b):
    """
    Compute the dot product of two 1D arrays.
    Do NOT use np.dot, np.inner, or the @ operator.

    Args:
        a: 1D numpy array of shape (n,)
        b: 1D numpy array of shape (n,)
    Returns:
        Scalar dot product
    """
    # Your implementation here
    pass
```

Test with:
```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
assert np.isclose(dot_product(a, b), np.dot(a, b))
```

#### 1b. Matrix-Vector Multiplication

```python
def matvec(A, x):
    """
    Compute A @ x using only dot products (your function from 1a).

    Args:
        A: 2D numpy array of shape (m, n)
        x: 1D numpy array of shape (n,)
    Returns:
        1D numpy array of shape (m,)
    """
    # Your implementation here
    pass
```

**Think about this:** Each element of the output is a dot product between one ROW of A
and the vector x. This is exactly what a single linear layer does: each neuron computes
a dot product between its weight row and the input.

#### 1c. Matrix-Matrix Multiplication

```python
def matmul(A, B):
    """
    Compute A @ B using only your matvec function.

    Args:
        A: 2D numpy array of shape (m, k)
        B: 2D numpy array of shape (k, n)
    Returns:
        2D numpy array of shape (m, n)
    """
    # Your implementation here
    pass
```

Test with random matrices of various sizes: (3, 4) @ (4, 5), (10, 20) @ (20, 3), etc.

#### 1d. Outer Product

```python
def outer_product(a, b):
    """
    Compute the outer product a @ b^T.

    Args:
        a: 1D numpy array of shape (m,)
        b: 1D numpy array of shape (n,)
    Returns:
        2D numpy array of shape (m, n)
    """
    # Your implementation here
    pass
```

**Why this matters:** The gradient of a linear layer's weight matrix is an outer product:
dL/dW = (dL/dz) outer (input). This is how weight gradients are computed in backprop.

#### 1e. Transpose

```python
def transpose(A):
    """
    Compute the transpose of A without using .T or np.transpose.

    Args:
        A: 2D numpy array of shape (m, n)
    Returns:
        2D numpy array of shape (n, m)
    """
    # Your implementation here
    pass
```

### Deliverable for Exercise 1

- All five functions implemented and tested
- For each function, include a comment explaining the computational complexity (Big O)
- Write a brief paragraph (3-4 sentences): How does the cost of matrix multiplication
  scale with the dimensions? Why does this matter for choosing the width of neural
  network layers?

---

## Exercise 2: Visualizing 2D Linear Transformations

**Why this matters:** The weight matrix of a linear layer transforms input space into
output space. You need to be able to "see" what a matrix does to build intuition about
what layers learn.

### Task

Create a visualization function that shows the effect of a 2x2 matrix on a set of 2D
points. Use the letter "F" as your test shape (it is asymmetric, so you can see rotations
and reflections clearly).

#### 2a. Setup

```python
import matplotlib.pyplot as plt
import numpy as np

def make_F():
    """
    Create a set of 2D points forming the letter F.
    Returns: numpy array of shape (N, 2)
    """
    points = []
    # Vertical bar
    for y in np.linspace(0, 2, 20):
        points.append([0, y])
    # Top horizontal bar
    for x in np.linspace(0, 1.5, 15):
        points.append([x, 2])
    # Middle horizontal bar
    for x in np.linspace(0, 1.0, 10):
        points.append([x, 1])
    return np.array(points)

def plot_transformation(original, transformed, title, ax):
    """
    Plot original points in blue and transformed points in red.
    Draw arrows showing the mapping for a few selected points.
    Include the origin and axis lines for reference.
    """
    # Your implementation here
    pass
```

#### 2b. Apply and Visualize These Transformations

Create a 2x3 grid of subplots showing the original F and the result of applying each
of these matrices:

1. **Rotation by 45 degrees:**
   ```
   R = [[cos(pi/4), -sin(pi/4)],
        [sin(pi/4),  cos(pi/4)]]
   ```

2. **Non-uniform scaling (stretch x by 2, compress y by 0.5):**
   ```
   S = [[2.0, 0.0],
        [0.0, 0.5]]
   ```

3. **Horizontal shear:**
   ```
   H = [[1.0, 0.5],
        [0.0, 1.0]]
   ```

4. **Reflection across the x-axis:**
   ```
   M = [[1.0,  0.0],
        [0.0, -1.0]]
   ```

5. **Projection onto the x-axis:**
   ```
   P = [[1.0, 0.0],
        [0.0, 0.0]]
   ```

6. **A "neural network layer" — random matrix (use a fixed seed):**
   ```python
   np.random.seed(42)
   W = np.random.randn(2, 2) * 0.5
   ```

For each transformation, label the plot with:
- The matrix
- The determinant (det < 0 means reflection, det = 0 means projection/rank reduction)
- What the transformation "does" in one phrase

#### 2c. Composition of Transformations

Apply rotation(30 degrees) followed by scaling(2, 0.5) in two ways:
1. Sequentially: first rotate, then scale
2. As a single matrix: S @ R

Verify they produce the same result. Then try the other order: R @ S. Show that the
result is different (matrix multiplication is not commutative).

### Written Question

In a neural network, the weight matrix W of a linear layer performs a transformation.
The bias vector b performs a translation. Together, y = Wx + b is an **affine**
transformation, not a linear one (because of the bias). Why is the bias necessary?
What can affine transformations do that pure linear transformations cannot? (Hint: think
about what happens at the origin.)

### Deliverable for Exercise 2

- Complete visualization code with all 6 transformations plotted
- Composition comparison plot showing S @ R vs R @ S
- Written answer to the affine transformation question (4-6 sentences)

---

## Exercise 3: PCA from Scratch

**Why this matters:** PCA is the most fundamental dimensionality reduction technique. It
finds the axes of maximum variance in your data using eigendecomposition. Understanding PCA
from scratch means understanding eigenvalues, covariance matrices, and projection — all of
which appear throughout deep learning.

### Task

#### 3a. Implement PCA

```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None      # Principal component directions
        self.mean = None            # Data mean (for centering)
        self.explained_variance = None

    def fit(self, X):
        """
        Fit PCA on data X.

        Steps:
        1. Center the data (subtract mean)
        2. Compute the covariance matrix
        3. Compute eigenvalues and eigenvectors of the covariance matrix
        4. Sort by eigenvalue (descending)
        5. Store the top n_components eigenvectors

        Args:
            X: numpy array of shape (n_samples, n_features)
        """
        # Your implementation here
        pass

    def transform(self, X):
        """
        Project data onto the principal components.

        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            numpy array of shape (n_samples, n_components)
        """
        # Your implementation here
        pass

    def inverse_transform(self, X_projected):
        """
        Reconstruct data from the projected representation.

        Args:
            X_projected: numpy array of shape (n_samples, n_components)
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        # Your implementation here
        pass

    def explained_variance_ratio(self):
        """Return the proportion of variance explained by each component."""
        # Your implementation here
        pass
```

#### 3b. Verify Against sklearn

```python
from sklearn.decomposition import PCA as SklearnPCA

# Generate test data
np.random.seed(42)
X = np.random.multivariate_normal(
    mean=[2, 3],
    cov=[[3, 2], [2, 2]],
    size=200
)

# Your PCA
my_pca = PCA(n_components=2)
my_pca.fit(X)

# sklearn PCA
sk_pca = SklearnPCA(n_components=2)
sk_pca.fit(X)

# Compare explained variance ratios (should match up to sign ambiguity)
print("My PCA variance ratio:", my_pca.explained_variance_ratio())
print("sklearn variance ratio:", sk_pca.explained_variance_ratio_)
```

Note: eigenvectors are unique only up to sign (v and -v are both eigenvectors). Your
components might be negated relative to sklearn. That is fine.

#### 3c. Visualize PCA on 2D Data

Using the correlated 2D data from 3b:
1. Plot the original data points
2. Overlay the principal component directions as arrows, scaled by the eigenvalues
3. Plot the data projected onto each principal component
4. Show the reconstruction from only the first principal component

This visualization should make it visually obvious that PCA finds the axis of maximum
spread.

### Deliverable for Exercise 3

- Complete PCA class implementation
- Verification against sklearn (printed comparison)
- 2D visualization with principal component arrows and projections

---

## Exercise 4: PCA on MNIST

**Why this matters:** MNIST images are 784-dimensional vectors. PCA reduces them to
a manageable number of dimensions while preserving most of the information. This
exercise connects eigendecomposition to real data and real compression.

### Task

#### 4a. Load and Prepare MNIST

```python
from sklearn.datasets import fetch_openml

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float64)  # (70000, 784)
y = mnist.target.astype(int)       # (70000,)

# Use a subset for speed
X_subset = X[:10000]
y_subset = y[:10000]
```

#### 4b. Apply Your PCA

1. Fit your PCA implementation on the MNIST subset with n_components = 50
2. Plot the cumulative explained variance ratio. How many components are needed
   to capture 90% of the variance? 95%? 99%?
3. Print the exact numbers.

#### 4c. Visualize Principal Components as Images

The principal components are 784-dimensional vectors. Reshape them to 28x28 and display
them as images.

```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    # Reshape the i-th principal component to 28x28 and display
    component_image = my_pca.components[i].reshape(28, 28)
    ax.imshow(component_image, cmap='RdBu_r')  # Red-blue colormap
    ax.set_title(f'PC {i+1}')
    ax.axis('off')
plt.suptitle('Top 10 Principal Components of MNIST')
plt.tight_layout()
plt.savefig('mnist_principal_components.png', dpi=150)
plt.show()
```

**Written question:** What do the principal components look like? Do they resemble
specific digits, or do they look like something else? Why?

#### 4d. Reconstruction at Different Compression Levels

For a few sample digits, show the reconstruction using k = {5, 10, 20, 50, 100, 200}
principal components. Display them side by side with the original.

```python
# Select one example of each digit 0-9
sample_indices = []
for digit in range(10):
    idx = np.where(y_subset == digit)[0][0]
    sample_indices.append(idx)

# For each compression level, show reconstructions
for n_comp in [5, 10, 20, 50, 100, 200]:
    pca_k = PCA(n_components=n_comp)
    pca_k.fit(X_subset)
    X_proj = pca_k.transform(X_subset[sample_indices])
    X_recon = pca_k.inverse_transform(X_proj)

    # Plot original and reconstructed side by side
    # Your plotting code here
```

**Written question:** At what number of components do the reconstructions become
"good enough" to recognize the digit? How does this compare to the original 784
dimensions? What does this tell you about the intrinsic dimensionality of MNIST?

#### 4e. 2D Scatter Plot

Project MNIST onto just 2 principal components and create a scatter plot colored by
digit label. Use a colormap with 10 distinct colors.

**Written question:** Which digits are well-separated in PCA space? Which overlap?
Why might this be? (Think about what visual features PCA captures.)

### Deliverable for Exercise 4

- Cumulative variance plot with 90/95/99% thresholds marked
- Principal component images (top 10)
- Reconstruction comparison grid
- 2D scatter plot colored by digit
- Written answers to all three questions (3-5 sentences each)

---

## Exercise 5: SVD-Based Image Compression

**Why this matters:** SVD gives the best low-rank approximation to any matrix. This is the
theoretical foundation for techniques like LoRA (Low-Rank Adaptation) used in fine-tuning
large language models. Understanding SVD on images makes the abstract concept concrete.

### Task

#### 5a. Load an Image and Apply SVD

```python
from PIL import Image
import requests
from io import BytesIO

# Load a grayscale image (use any image you like, or generate one)
# Option 1: Use a sample image
img = np.random.RandomState(42).rand(256, 256)  # Placeholder — use a real image

# Option 2: Load from file
# img = np.array(Image.open('your_image.jpg').convert('L')) / 255.0

# Apply SVD
U, s, Vt = np.linalg.svd(img, full_matrices=False)

print(f"Image shape: {img.shape}")
print(f"U shape: {U.shape}, s shape: {s.shape}, Vt shape: {Vt.shape}")
print(f"Top 10 singular values: {s[:10]}")
```

#### 5b. Reconstruct at Different Ranks

```python
def svd_compress(U, s, Vt, k):
    """Reconstruct image using top-k singular values."""
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

def compression_ratio(original_shape, k):
    """Compute the compression ratio for rank-k approximation."""
    m, n = original_shape
    original_size = m * n
    compressed_size = k * (m + 1 + n)  # U columns + singular values + V rows
    return original_size / compressed_size

# Show reconstructions for various ranks
ranks = [1, 5, 10, 20, 50, 100]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Plot original
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

for i, k in enumerate(ranks):
    ax = axes.flat[i + 1]
    reconstructed = svd_compress(U, s, Vt, k)
    ratio = compression_ratio(img.shape, k)
    error = np.linalg.norm(img - reconstructed) / np.linalg.norm(img)
    ax.imshow(reconstructed, cmap='gray')
    ax.set_title(f'k={k}, ratio={ratio:.1f}x\nerror={error:.3f}')
    ax.axis('off')

# Last subplot: singular value spectrum
ax = axes.flat[-1]
ax.semilogy(s)
ax.set_xlabel('Index')
ax.set_ylabel('Singular Value (log scale)')
ax.set_title('Singular Value Spectrum')

plt.tight_layout()
plt.savefig('svd_compression.png', dpi=150)
plt.show()
```

#### 5c. Analyze the Singular Value Spectrum

1. Plot the singular values on a log scale
2. Plot the cumulative "energy" (sum of s_i^2 up to index k, divided by total sum of s_i^2)
3. Find the rank needed to capture 90%, 95%, and 99% of the energy

**Written question:** Natural images have rapidly decaying singular values. What does
this mean about the "true dimensionality" of images? How does this connect to why
neural networks can compress images effectively (autoencoders, VAEs)?

#### 5d. Compare Compression on Different Image Types

Apply SVD compression to:
1. A natural photograph (smooth gradients, complex textures)
2. A simple geometric pattern (e.g., stripes, checkerboard)
3. A pure noise image (np.random.randn)

Compare the singular value spectra and the visual quality at the same rank.

**Written question:** Which image type compresses best? Which compresses worst? Why?
(Hint: think about the rank of each image type.)

### Deliverable for Exercise 5

- Reconstruction grid at different ranks
- Singular value spectrum and cumulative energy plots
- Comparison across image types
- Written answers (3-5 sentences each)

---

## Exercise 6: Broadcasting — Predict Then Verify

**Why this matters:** Broadcasting bugs are silent. The code runs, the shapes are wrong,
and the results are garbage. You must be able to predict broadcasting behavior without
running the code.

### Task

For each of the following operations, FIRST write down:
1. The output shape
2. Whether it will succeed or raise an error
3. If it succeeds, describe in one sentence what the operation computes

THEN run the code and check your predictions.

#### Scenario 1
```python
A = np.ones((3, 4))
b = np.ones((4,))
result = A + b
# Your prediction: shape = ?, succeeds = ?
```

#### Scenario 2
```python
A = np.ones((3, 4))
b = np.ones((3,))
result = A + b
# Your prediction: shape = ?, succeeds = ?
```

#### Scenario 3
```python
A = np.ones((3, 4))
b = np.ones((3, 1))
result = A + b
# Your prediction: shape = ?, succeeds = ?
```

#### Scenario 4
```python
A = np.ones((2, 3, 4))
b = np.ones((3, 4))
result = A * b
# Your prediction: shape = ?, succeeds = ?
```

#### Scenario 5
```python
A = np.ones((2, 3, 4))
b = np.ones((2, 1, 4))
result = A - b
# Your prediction: shape = ?, succeeds = ?
```

#### Scenario 6 (The Trap)
```python
logits = np.random.randn(32, 10)     # (batch, classes)
labels = np.array([3, 5, 1, ...])     # (batch,) — 32 values
result = logits - labels
# Your prediction: shape = ?, succeeds = ?
# If this succeeds but gives the wrong thing, explain what went wrong.
```

#### Scenario 7 (Real DL Pattern)
```python
# Batch normalization: normalize each feature across the batch
activations = np.random.randn(32, 256)  # (batch, features)
mean = activations.mean(axis=0)          # What shape?
std = activations.std(axis=0)            # What shape?
normalized = (activations - mean) / std  # Does this broadcast correctly?
# Your prediction for each step.
```

#### Scenario 8 (Attention Pattern)
```python
# Scaled dot-product attention: mask broadcasting
scores = np.random.randn(8, 16, 64, 64)  # (batch, heads, seq, seq)
mask = np.ones((1, 1, 64, 64))            # Causal mask
masked_scores = scores + mask              # Does this work?
# Your prediction: shape = ?, succeeds = ?
```

### Deliverable for Exercise 6

- Written predictions for all 8 scenarios (shape, success/failure, description)
- Code output confirming or correcting your predictions
- For any wrong predictions, a brief explanation of what you misunderstood

---

## Grading Criteria

| Criterion | Weight | What We Look For |
|-----------|--------|------------------|
| Correctness | 30% | All implementations produce correct results, verified against reference. |
| Code Quality | 20% | Clean, documented code. Functions have docstrings. Variable names are meaningful. |
| Visualizations | 20% | Clear, labeled plots. Axes labeled. Titles informative. Colormaps appropriate. |
| Written Analysis | 20% | Demonstrates understanding of WHY, not just WHAT. Connects to deep learning. |
| Completeness | 10% | All exercises attempted. All deliverables present. |

---

## Stretch Goals

If you finish early and want to push further:

1. **Implement power iteration** to find the dominant eigenvector of a matrix without
   using np.linalg.eig. Use it to compute PCA one component at a time (with deflation).
   This is how PCA is computed in practice for very large datasets.

2. **Implement randomized SVD**: Instead of full SVD (expensive for large matrices), use
   random projections to find an approximate SVD. This is the algorithm behind
   sklearn's `randomized_svd` and is crucial for large-scale applications.

3. **Measure the effective rank** of weight matrices in a pre-trained neural network
   (e.g., a small ResNet from torchvision). Compute the singular values of each layer's
   weight matrix and plot the spectrum. How many singular values are "significant"?
   What does this tell you about the capacity utilization of the network?

4. **Implement the NumPy broadcasting algorithm from scratch**: Write a function that
   takes two shapes as tuples and returns the broadcast output shape (or raises an error).
   This forces you to internalize the rules completely.
