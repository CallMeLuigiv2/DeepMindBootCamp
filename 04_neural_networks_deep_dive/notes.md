# Module 4: Neural Networks Deep Dive — Reference Notes

## The Foundation Document

Everything in modern deep learning — transformers, diffusion models, large language models,
reinforcement learning agents — is built from the components described in this document.
This is not a survey. This is the reference you will return to repeatedly for the rest of
the course and your career.

For each concept: intuition first, then the math, then the code, then what goes wrong.

---

## Table of Contents

1. [From Perceptrons to MLPs](#1-from-perceptrons-to-mlps)
2. [Activation Functions: The Complete Guide](#2-activation-functions-the-complete-guide)
3. [Backpropagation: Full Derivation](#3-backpropagation-full-derivation)
4. [Vanishing and Exploding Gradients](#4-vanishing-and-exploding-gradients)
5. [Weight Initialization](#5-weight-initialization)
6. [Batch Normalization](#6-batch-normalization)
7. [The Regularization Toolbox](#7-the-regularization-toolbox)
8. [Optimization: From SGD to Adam](#8-optimization-from-sgd-to-adam)
9. [Learning Rate Schedules](#9-learning-rate-schedules)
10. [The Debugging Checklist](#10-the-debugging-checklist)

---

## 1. From Perceptrons to MLPs

### Intuition

A single neuron is the simplest possible classifier. It takes a weighted sum of its inputs,
adds a bias, and passes the result through a nonlinear function. Geometrically, it draws a
single hyperplane in input space and says "everything on this side is class A, everything on
that side is class B."

An MLP stacks many neurons into layers. The first layer draws many hyperplanes. The second
layer combines those hyperplanes into more complex regions. The third layer combines those
regions further. By the time you have three or four layers, the network can carve input space
into arbitrarily complex decision regions.

This is representation learning: each layer transforms the data into a new representation
where the task becomes progressively easier.

### The Math

A single neuron computes:

```
z = w^T x + b       (linear transformation)
a = sigma(z)         (nonlinear activation)
```

where w in R^n is the weight vector, b in R is the bias, x in R^n is the input, and sigma
is the activation function.

The decision boundary is the set {x : w^T x + b = 0}, which is a hyperplane in R^n.

An L-layer MLP computes:

```
z^(1) = W^(1) x + b^(1)
a^(1) = sigma(z^(1))
z^(2) = W^(2) a^(1) + b^(2)
a^(2) = sigma(z^(2))
...
z^(L) = W^(L) a^(L-1) + b^(L)
y_hat = output_activation(z^(L))
```

For classification, the output activation is typically softmax:

```
softmax(z)_i = exp(z_i) / sum_j(exp(z_j))
```

### The Universal Approximation Theorem

**Statement**: A feedforward network with one hidden layer containing a finite number of
neurons, with a non-polynomial activation function, can approximate any continuous function
on a compact subset of R^n to arbitrary accuracy (Cybenko 1989, Hornik 1991).

**What it means**: MLPs are universal function approximators. They are expressive enough to
represent any continuous mapping.

**What it does NOT mean**:
- It does not say how many neurons you need. The required width may be exponential.
- It does not say gradient descent can find the right weights.
- It does not say the network will generalize from finite training data.
- It does not say one hidden layer is optimal. Depth gives exponential efficiency over width
  for many function classes.

### Code: Basic MLP in PyTorch

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        layers = []
        prev_dim = input_dim

        act_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
        }[activation]

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Example: MNIST classifier
model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
```

### What Can Go Wrong

- **Too narrow**: Network cannot represent the function. Training and validation loss both
  plateau at a high value.
- **Too deep without skip connections**: Gradients vanish or explode. Training loss does not
  decrease. (See Section 4.)
- **No activation functions**: The entire network collapses to a single linear transformation.
  W^(L) ... W^(2) W^(1) = one matrix. Depth is useless without nonlinearity.

---

## 2. Activation Functions: The Complete Guide

### The Historical Arc

Each activation function was invented to solve a specific problem with its predecessors.
Understanding this arc is understanding the evolution of deep learning.

### Comparison Table

| Function | Formula | Derivative | Output Range | Pros | Cons |
|----------|---------|-----------|-------------|------|------|
| Sigmoid | 1/(1+e^{-z}) | sigma*(1-sigma) | (0, 1) | Smooth, bounded, probabilistic interpretation | Vanishing gradients, not zero-centered, exp() is expensive |
| Tanh | (e^z-e^{-z})/(e^z+e^{-z}) | 1 - tanh^2(z) | (-1, 1) | Zero-centered, stronger gradients than sigmoid | Still vanishing gradients for large |z| |
| ReLU | max(0, z) | 0 if z<0, 1 if z>0 | [0, inf) | No vanishing gradient for z>0, sparse, fast | Dead neurons (zero gradient for z<0), not zero-centered |
| LeakyReLU | max(alpha*z, z) | alpha if z<0, 1 if z>0 | (-inf, inf) | No dead neurons, fast | Requires choosing alpha (typically 0.01) |
| GELU | z * Phi(z) | Phi(z) + z*phi(z) | ~(-0.17, inf) | Smooth approximation to ReLU, works well in transformers | Slightly more expensive than ReLU |
| Swish | z * sigmoid(z) | swish(z) + sigmoid(z)*(1-swish(z)) | ~(-0.28, inf) | Smooth, non-monotonic, found via NAS | More expensive than ReLU |

Where Phi(z) is the CDF of the standard normal, and phi(z) is its PDF.

### Sigmoid (1980s-2000s)

**Intuition**: A smooth step function. Squashes any real number into (0, 1). Originally
motivated by biological neurons (firing rate interpretation).

**The math**:
```
sigma(z) = 1 / (1 + exp(-z))
sigma'(z) = sigma(z) * (1 - sigma(z))
```

**The problem**: When z is very positive or very negative, sigma'(z) is near zero. The
maximum value of sigma'(z) is 0.25 (at z=0). This means gradients SHRINK by at least 4x
at every layer. For a 10-layer network: 0.25^10 = 9.5e-8. The gradient is essentially zero.

**When to use**: Output layer for binary classification. Nowhere else.

### Tanh (Improvement over Sigmoid)

**Intuition**: Zero-centered sigmoid. Outputs range from -1 to +1 instead of 0 to 1.

**Why it was better**: With sigmoid, all outputs are positive. This means gradients for the
weights of the next layer are all the same sign, causing zig-zag optimization dynamics.
Tanh's zero-centered outputs fix this.

**The math**:
```
tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
tanh'(z) = 1 - tanh^2(z)
```

Note: tanh(z) = 2*sigmoid(2z) - 1. They are the same family.

**The remaining problem**: tanh'(z) still saturates to 0 for large |z|. Maximum is 1 (at
z=0), which is better than sigmoid's 0.25, but still vanishes with depth.

### ReLU (2010 — The Game Changer)

**Intuition**: "If the input is positive, pass it through. If negative, output zero."

**The math**:
```
relu(z) = max(0, z)
relu'(z) = 1 if z > 0, 0 if z < 0  (undefined at z=0, typically set to 0)
```

**Why it changed everything**: For active neurons (z > 0), the gradient is exactly 1. No
shrinkage. No vanishing gradient. This is why deep networks (10+ layers) became trainable.

**The problem — dead neurons**: If a neuron's input is always negative (e.g., due to a large
negative bias), its gradient is always zero. It never updates. It is permanently dead.

**How common**: In practice, 10-40% of neurons can die during training. This is not
necessarily catastrophic (it creates sparsity), but a network with >50% dead neurons is
underperforming.

### LeakyReLU (Fixing Dead Neurons)

**The math**:
```
leaky_relu(z) = z if z > 0, alpha * z if z < 0   (alpha = 0.01 typically)
leaky_relu'(z) = 1 if z > 0, alpha if z < 0
```

**Why**: Dead neurons get a small gradient (alpha) instead of zero. They can recover.

**PReLU**: Same as LeakyReLU but alpha is a learnable parameter. The network decides the
optimal slope.

### GELU (2016 — The Transformer Activation)

**Intuition**: "How much of the input to pass through" is a probabilistic decision. GELU
scales the input by the probability that a Gaussian random variable is less than that input.

**The math**:
```
gelu(z) = z * Phi(z) = z * 0.5 * (1 + erf(z / sqrt(2)))
```

Approximate: gelu(z) ~ 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))

**Why it works in transformers**: It provides a smooth, non-zero-everywhere gradient that
includes a form of stochastic regularization (the probabilistic gating).

### Swish (2017 — Discovered by Neural Architecture Search)

**The math**:
```
swish(z) = z * sigmoid(z)
swish'(z) = swish(z) + sigmoid(z) * (1 - swish(z))
```

**Notable property**: Non-monotonic. Slightly negative outputs for slightly negative inputs.
This can help optimization by allowing small negative gradients.

### Code: All Activation Functions

```python
import torch
import torch.nn.functional as F
import numpy as np

# In NumPy (for from-scratch implementations)
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

# In PyTorch
# sigmoid: torch.sigmoid(z) or nn.Sigmoid()
# tanh: torch.tanh(z) or nn.Tanh()
# relu: F.relu(z) or nn.ReLU()
# leaky_relu: F.leaky_relu(z, 0.01) or nn.LeakyReLU(0.01)
# gelu: F.gelu(z) or nn.GELU()
# swish/silu: F.silu(z) or nn.SiLU()
```

### Practical Guidelines

- **Default choice**: ReLU. It is fast, well-understood, and works in most settings.
- **If dead neurons are a problem**: LeakyReLU or GELU.
- **For transformers**: GELU (it is the standard).
- **For the output layer**: sigmoid for binary, softmax for multiclass, linear for regression.
- **Never use sigmoid or tanh in hidden layers** unless you have a specific reason and
  understand the vanishing gradient consequences.

---

## 3. Backpropagation: Full Derivation

This section is the most important in the entire document. We will derive backpropagation
for a concrete 3-layer MLP, step by step, with no shortcuts.

### The Network

```
Input:   x in R^3        (3 features)
Layer 1: W1 in R^{4x3}, b1 in R^4, activation: ReLU
Layer 2: W2 in R^{4x4}, b2 in R^4, activation: ReLU
Layer 3: W3 in R^{2x4}, b3 in R^2, activation: softmax
Loss:    cross-entropy
```

### Forward Pass — Every Computation

```
Step 1: z1 = W1 @ x + b1          (shape: 4)
Step 2: a1 = relu(z1)             (shape: 4)
Step 3: z2 = W2 @ a1 + b2         (shape: 4)
Step 4: a2 = relu(z2)             (shape: 4)
Step 5: z3 = W3 @ a2 + b3         (shape: 2)
Step 6: y_hat = softmax(z3)       (shape: 2)
Step 7: L = -sum(y * log(y_hat))  (scalar, where y is one-hot target)
```

This is function composition: L = loss(softmax(W3 @ relu(W2 @ relu(W1 @ x + b1) + b2) + b3), y)

### Backward Pass — Every Gradient

We apply the chain rule, working backward from the loss.

#### Step 1: dL/dz3 (Output layer gradient)

For softmax + cross-entropy combined, this simplifies to:

```
dL/dz3 = y_hat - y                (shape: 2)
```

**Derivation of this simplification**:

Let y_hat_i = exp(z3_i) / sum_j(exp(z3_j)) and L = -sum_i(y_i * log(y_hat_i)).

For the case where y is one-hot with y_k = 1:
L = -log(y_hat_k) = -log(exp(z3_k) / sum_j(exp(z3_j)))
L = -z3_k + log(sum_j(exp(z3_j)))

dL/dz3_i = -delta_{ik} + exp(z3_i) / sum_j(exp(z3_j))
         = -delta_{ik} + y_hat_i
         = y_hat_i - y_i

This is one of the most elegant results in neural network math.

#### Step 2: Gradients for W3 and b3

```
z3 = W3 @ a2 + b3

dL/dW3 = dL/dz3 @ a2^T            (shape: 2x4)
dL/db3 = dL/dz3                   (shape: 2)
```

**Why?** By the chain rule: dL/dW3_{ij} = dL/dz3_i * dz3_i/dW3_{ij} = dL/dz3_i * a2_j.
In matrix form: dL/dW3 = dL/dz3 @ a2^T (outer product).

For the bias: dL/db3_i = dL/dz3_i * dz3_i/db3_i = dL/dz3_i * 1 = dL/dz3_i.

#### Step 3: Propagate gradient to a2

```
dL/da2 = W3^T @ dL/dz3            (shape: 4)
```

**Why?** z3 = W3 @ a2 + b3. So dz3_i/da2_j = W3_{ij}. Therefore:
dL/da2_j = sum_i(dL/dz3_i * dz3_i/da2_j) = sum_i(dL/dz3_i * W3_{ij}).
In matrix form: dL/da2 = W3^T @ dL/dz3.

#### Step 4: Propagate through ReLU

```
dL/dz2 = dL/da2 * relu'(z2)       (shape: 4, element-wise multiplication)
```

Where relu'(z2_j) = 1 if z2_j > 0, 0 if z2_j <= 0.

**Critical**: This is where dead neurons show up. If z2_j <= 0, then dL/dz2_j = 0, and
NO gradient flows through that neuron to earlier layers.

#### Step 5: Gradients for W2 and b2

```
dL/dW2 = dL/dz2 @ a1^T            (shape: 4x4)
dL/db2 = dL/dz2                   (shape: 4)
```

Same pattern as Step 2.

#### Step 6: Propagate gradient to a1

```
dL/da1 = W2^T @ dL/dz2            (shape: 4)
```

#### Step 7: Propagate through ReLU

```
dL/dz1 = dL/da1 * relu'(z1)       (shape: 4)
```

#### Step 8: Gradients for W1 and b1

```
dL/dW1 = dL/dz1 @ x^T             (shape: 4x3)
dL/db1 = dL/dz1                   (shape: 4)
```

### The Pattern

For any layer i:
```
dL/dW_i = dL/dz_i @ a_{i-1}^T
dL/db_i = dL/dz_i
dL/da_{i-1} = W_i^T @ dL/dz_i
dL/dz_{i-1} = dL/da_{i-1} * sigma'(z_{i-1})
```

This is the same four equations repeated for every layer. Backpropagation is just this pattern
applied recursively from the output to the input.

### Concrete Numerical Example

Let us trace through with actual numbers for a tiny 2-2-1 network.

```
Network: 2 inputs, 2 hidden neurons (ReLU), 1 output (sigmoid)
Loss: binary cross-entropy

W1 = [[0.1, 0.3],    b1 = [0.0, 0.0]
      [0.2, 0.4]]

W2 = [[0.5, 0.6]]    b2 = [0.0]

Input: x = [1.0, 2.0]
Target: y = 1
```

**Forward pass**:
```
z1 = W1 @ x + b1
z1[0] = 0.1*1.0 + 0.3*2.0 + 0.0 = 0.7
z1[1] = 0.2*1.0 + 0.4*2.0 + 0.0 = 1.0

a1 = relu(z1) = [0.7, 1.0]   (both positive, so unchanged)

z2 = W2 @ a1 + b2
z2[0] = 0.5*0.7 + 0.6*1.0 + 0.0 = 0.95

y_hat = sigmoid(z2) = sigmoid(0.95) = 1/(1+exp(-0.95)) = 0.7211

L = -(y*log(y_hat) + (1-y)*log(1-y_hat))
  = -(1*log(0.7211) + 0*log(0.2789))
  = -log(0.7211) = 0.3267
```

**Backward pass**:
```
dL/dz2 = y_hat - y = 0.7211 - 1 = -0.2789

dL/dW2 = dL/dz2 @ a1^T
dL/dW2[0,0] = -0.2789 * 0.7 = -0.1952
dL/dW2[0,1] = -0.2789 * 1.0 = -0.2789

dL/db2 = dL/dz2 = [-0.2789]

dL/da1 = W2^T @ dL/dz2
dL/da1[0] = 0.5 * (-0.2789) = -0.1395
dL/da1[1] = 0.6 * (-0.2789) = -0.1674

dL/dz1 = dL/da1 * relu'(z1)
relu'(z1) = [1, 1]   (both z1 values were positive)
dL/dz1 = [-0.1395, -0.1674]

dL/dW1 = dL/dz1 @ x^T
dL/dW1[0,0] = -0.1395 * 1.0 = -0.1395
dL/dW1[0,1] = -0.1395 * 2.0 = -0.2789
dL/dW1[1,0] = -0.1674 * 1.0 = -0.1674
dL/dW1[1,1] = -0.1674 * 2.0 = -0.3347

dL/db1 = dL/dz1 = [-0.1395, -0.1674]
```

**Verification with PyTorch**:
```python
import torch
import torch.nn as nn

# Set up the exact same network
W1 = torch.tensor([[0.1, 0.3], [0.2, 0.4]], requires_grad=True)
b1 = torch.tensor([0.0, 0.0], requires_grad=True)
W2 = torch.tensor([[0.5, 0.6]], requires_grad=True)
b2 = torch.tensor([0.0], requires_grad=True)

x = torch.tensor([1.0, 2.0])
y = torch.tensor([1.0])

# Forward
z1 = W1 @ x + b1
a1 = torch.relu(z1)
z2 = W2 @ a1 + b2
y_hat = torch.sigmoid(z2)
loss = -torch.sum(y * torch.log(y_hat) + (1-y) * torch.log(1-y_hat))

# Backward
loss.backward()

print(f"dL/dW1 = {W1.grad}")   # Should match our hand computation
print(f"dL/db1 = {b1.grad}")
print(f"dL/dW2 = {W2.grad}")
print(f"dL/db2 = {b2.grad}")
```

The values should match to 4+ decimal places. If they do not, the hand computation has an
error.

### How This Maps to PyTorch's Autograd

When you call `loss.backward()` in PyTorch, it does exactly what we did above:

1. It traverses the computational graph backward (from loss to inputs).
2. At each operation, it computes the local Jacobian and multiplies by the incoming gradient.
3. It accumulates gradients in each tensor's `.grad` attribute.
4. The chain rule is applied automatically via vector-Jacobian products (VJPs).

Each PyTorch operation (addition, multiplication, relu, etc.) has a registered backward
function. When the forward pass runs, PyTorch builds a graph of these operations. The
backward pass traverses this graph in reverse topological order.

### The Jacobian Perspective

For a layer that computes a = f(z), the Jacobian is:

```
J = da/dz    (matrix of partial derivatives)
```

For element-wise activations, J is diagonal: J_{ij} = f'(z_i) if i=j, 0 otherwise.

For a linear layer z = W @ a_prev, the Jacobian with respect to a_prev is W.

The full gradient of the loss with respect to the input is:

```
dL/dx = J_1^T @ J_2^T @ ... @ J_L^T @ dL/da_L
```

This is a product of L Jacobians. The norm of this product determines whether gradients
vanish or explode.

### Computational Cost

- **Forward pass**: O(sum of layer parameters). For a network with P total parameters,
  roughly O(P).
- **Backward pass**: also O(P). Each operation's VJP has the same cost as the operation.
  Total: 2-3x the cost of a forward pass (extra memory for storing intermediates).
- **Numerical differentiation**: For each parameter, perturb it and recompute the loss.
  Cost: O(P) per parameter, so O(P^2) total. For a network with 1 million parameters,
  backprop is 1 million times faster.

This is why deep learning is computationally feasible.

---

## 4. Vanishing and Exploding Gradients

### Intuition

Consider a 50-layer network. The gradient of the loss with respect to the first layer's
weights must flow backward through all 50 layers. At each layer, the gradient is multiplied
by the layer's Jacobian. If each multiplication shrinks the gradient by even a small factor,
50 multiplications will reduce it to near zero. If each multiplication grows the gradient,
50 multiplications will make it astronomically large.

This is not a bug. It is a mathematical consequence of the chain rule applied to deep
function compositions. It is the core problem that drove 15 years of deep learning research.

### The Math

The gradient of the loss with respect to the first layer's activations is:

```
dL/da_0 = (product from i=L to 1 of J_i^T) @ dL/da_L
```

where J_i = diag(sigma'(z_i)) @ W_i is the Jacobian of layer i.

The norm of this product:

```
||dL/da_0|| <= (product of ||J_i||) * ||dL/da_L||
```

If each ||J_i|| = r, then:

```
||dL/da_0|| ~ r^L * ||dL/da_L||
```

- If r < 1: the gradient decays exponentially with depth. **Vanishing gradients.**
- If r > 1: the gradient grows exponentially with depth. **Exploding gradients.**
- Only if r = 1 exactly do gradients remain stable. This is very hard to maintain.

### With Sigmoid Activation

```
sigmoid'(z) in (0, 0.25]    (maximum at z=0)
```

So the Jacobian's diagonal entries are at most 0.25. The spectral norm of J_i is bounded
by 0.25 * ||W_i||. Unless ||W_i|| > 4, the gradient shrinks at every layer.

For a 10-layer network with sigmoid and ||W_i|| = 1:
```
||gradient|| ~ 0.25^10 * ||output gradient|| = 9.5e-8 * ||output gradient||
```

The first layer effectively receives zero gradient. It cannot learn.

### With ReLU Activation

```
relu'(z) = 1 if z > 0, 0 if z <= 0
```

For active neurons, the gradient multiplier is exactly 1. No shrinkage. But:
- Dead neurons (z <= 0) pass zero gradient. If a neuron is dead, it stays dead.
- The Jacobian's spectral norm is approximately ||W_i|| (for the active subspace).
  If ||W_i|| > 1, gradients can still explode. If ||W_i|| < 1, they can still vanish.

### Solutions (Each Addresses the Same Problem Differently)

1. **Better activations** (ReLU, LeakyReLU): Remove gradient saturation for positive inputs.
2. **Better initialization** (Xavier, He): Set ||W_i|| ~ 1 at initialization so r ~ 1.
3. **Normalization** (BatchNorm, LayerNorm): Continuously re-normalize activations during
   training, preventing them from drifting to extreme values.
4. **Residual connections** (ResNet): y = f(x) + x. The gradient flows through the skip
   connection with multiplier 1, bypassing the vanishing/exploding problem entirely.
5. **Gradient clipping**: If ||gradient|| > threshold, scale it down. Prevents explosion
   but does not help vanishing.

### Code: Visualizing Gradient Flow

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_gradient_flow(model, loss):
    """Visualize gradient magnitudes across layers after loss.backward()."""
    loss.backward()

    layers = []
    avg_grads = []
    max_grads = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(layers)), avg_grads, alpha=0.7, label='Mean gradient')
    plt.bar(range(len(layers)), max_grads, alpha=0.3, label='Max gradient')
    plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
    plt.xlabel('Layer')
    plt.ylabel('Gradient magnitude')
    plt.title('Gradient Flow')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
```

### What Can Go Wrong

- **Vanishing symptoms**: Loss decreases extremely slowly or plateaus. Early layers have
  near-zero gradient norms. Weight histograms show early layers barely changing.
- **Exploding symptoms**: Loss suddenly becomes NaN or inf. Gradient norms are huge (1e6+).
  Weights become very large.
- **Dead ReLU symptoms**: Gradient is exactly zero for many neurons. Check by computing
  `(activation > 0).float().mean()` — if this is below 0.5 for any layer, you have a
  problem.

---

## 5. Weight Initialization

### Intuition

At the start of training, you must set all the weights to some values. If you choose badly,
the network cannot train at all. If you choose well, training starts smoothly and converges
faster.

The core idea: at initialization, we want the variance of activations to stay approximately
constant across layers. If variance grows, activations explode. If variance shrinks,
activations vanish. Either way, gradients break.

### Why Zeros Do Not Work

If W = 0 for all layers:
- All neurons in a layer compute the same output (zero, or the bias).
- All neurons receive the same gradient.
- All neurons update identically.
- Symmetry is never broken. The network has effectively one neuron per layer.

### Xavier/Glorot Initialization (2010)

**Derivation**:

Consider layer z = W @ a_prev, where a_prev has n_in elements.

Assume W and a_prev are independent with zero mean. Then:

```
Var(z_j) = sum_{k=1}^{n_in} Var(W_{jk}) * Var(a_prev_k)
         = n_in * Var(W) * Var(a_prev)
```

(Using the identity Var(AB) = Var(A)*Var(B) when E[A]=E[B]=0.)

For Var(z) = Var(a_prev), we need:

```
n_in * Var(W) = 1
Var(W) = 1 / n_in
```

For the backward pass, a similar analysis gives:

```
Var(W) = 1 / n_out
```

The compromise (Xavier initialization):

```
Var(W) = 2 / (n_in + n_out)
```

As a normal distribution: W ~ N(0, 2/(n_in + n_out)).
As a uniform distribution: W ~ U(-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))).

**Assumption**: This derivation assumes the activation function is linear near zero
(like tanh at initialization). It is correct for tanh and sigmoid.

### Kaiming/He Initialization (2015)

**Derivation**:

ReLU zeros out half the activations. So Var(relu(z)) = Var(z) / 2.

Correcting the Xavier derivation:

```
Var(z) = n_in * Var(W) * Var(a_prev)
Var(a) = Var(relu(z)) = Var(z) / 2 = n_in * Var(W) * Var(a_prev) / 2
```

For Var(a) = Var(a_prev):

```
n_in * Var(W) / 2 = 1
Var(W) = 2 / n_in
```

He initialization: W ~ N(0, 2/n_in).

For LeakyReLU with slope alpha:

```
Var(W) = 2 / ((1 + alpha^2) * n_in)
```

### Code: Initialization in PyTorch

```python
import torch.nn as nn
import torch.nn.init as init

# Xavier (Glorot) initialization
layer = nn.Linear(784, 256)
init.xavier_uniform_(layer.weight)    # Uniform variant
init.xavier_normal_(layer.weight)     # Normal variant
init.zeros_(layer.bias)

# Kaiming (He) initialization
layer = nn.Linear(784, 256)
init.kaiming_uniform_(layer.weight, nonlinearity='relu')
init.kaiming_normal_(layer.weight, nonlinearity='relu')
init.zeros_(layer.bias)

# Apply to entire model
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

model = MLP(784, [256, 128], 10)
model.apply(init_weights)
```

### LSUV: Layer-Sequential Unit Variance

A data-driven approach. Algorithm:
1. Initialize all layers with orthogonal initialization.
2. For each layer, from first to last:
   a. Pass a batch of data through the network up to this layer.
   b. Compute the variance of the layer's output.
   c. Scale the layer's weights by 1/sqrt(variance) so output variance becomes 1.
   d. Repeat until variance is close to 1 (typically 1-3 iterations).

This works for any activation function without needing a closed-form derivation.

### Practical Guidelines

| Activation | Initialization | Formula |
|-----------|---------------|---------|
| ReLU, LeakyReLU | Kaiming (He) | W ~ N(0, 2/n_in) |
| Sigmoid, Tanh | Xavier (Glorot) | W ~ N(0, 2/(n_in+n_out)) |
| GELU, Swish | Kaiming (He) | W ~ N(0, 2/n_in) |
| Any (with BatchNorm) | Kaiming or Xavier | Less sensitive, but still use correct init |
| Bias terms | Zeros | b = 0 |
| Output bias (classification) | Log-prior | b_k = log(n_k / N) for class k |

### What Can Go Wrong

- **Activation collapse**: With too-small initialization, all activations in later layers
  are near zero. Loss does not decrease. Check by printing `a.std()` for each layer.
- **Activation explosion**: With too-large initialization, activations overflow to NaN.
  Check by printing `a.abs().max()` for each layer.
- **Symmetric neurons**: If you accidentally initialize all weights identically (not just
  zeros, but any identical values), you get the symmetry problem.
- **Forgetting to initialize**: PyTorch's default initialization for `nn.Linear` is
  Kaiming uniform, which is usually fine. But custom layers may not have proper defaults.

---

## 6. Batch Normalization

### Intuition

Imagine you are trying to learn to hit a moving target. Every time you adjust your aim,
someone moves the target. That is what a hidden layer experiences during training: its
inputs change every time the preceding layers update their weights. Batch normalization
stabilizes the target by normalizing the inputs to each layer.

More precisely: BatchNorm forces each layer's inputs to have zero mean and unit variance
(within a batch), then allows the network to learn optimal mean and variance via parameters
gamma and beta.

### The Full Forward Pass

**Training mode**:

Given a mini-batch of inputs {x_1, ..., x_m} to a layer:

```
Step 1: Batch mean
  mu_B = (1/m) * sum_{i=1}^{m} x_i

Step 2: Batch variance
  sigma^2_B = (1/m) * sum_{i=1}^{m} (x_i - mu_B)^2

Step 3: Normalize
  x_hat_i = (x_i - mu_B) / sqrt(sigma^2_B + epsilon)

Step 4: Scale and shift (learnable)
  y_i = gamma * x_hat_i + beta
```

where gamma and beta are learnable parameters (one per feature/channel), and epsilon is a
small constant (e.g., 1e-5) for numerical stability.

During training, also maintain running statistics:
```
running_mean = momentum * running_mean + (1 - momentum) * mu_B
running_var  = momentum * running_var  + (1 - momentum) * sigma^2_B
```
(PyTorch default momentum = 0.1.)

**Evaluation mode**:

Use the running statistics instead of batch statistics:
```
x_hat_i = (x_i - running_mean) / sqrt(running_var + epsilon)
y_i = gamma * x_hat_i + beta
```

This is critical: at test time, the output for a single input should not depend on what
other inputs are in the batch.

### Why Gamma and Beta?

If we only normalized (steps 1-3), we would force every layer's inputs to be N(0,1). This
constrains the network's representational power. Maybe the network needs inputs with mean 5
and variance 2 for optimal performance.

Gamma and beta let the network undo the normalization if needed. If gamma = sqrt(sigma^2_B)
and beta = mu_B, the transformation is the identity. So BatchNorm can never hurt
representational power — in the worst case, it learns to do nothing.

### Why It Works: Multiple Theories

**Theory 1: Internal Covariate Shift (original paper, Ioffe & Szegedy 2015)**
- As earlier layers update, the distribution of inputs to later layers changes.
- This forces each layer to constantly re-adapt, slowing training.
- BatchNorm stabilizes these distributions.
- **Status**: Subsequent work showed this is not the main mechanism. Santurkar et al. (2018)
  showed that BatchNorm does not reduce internal covariate shift.

**Theory 2: Loss Landscape Smoothing (Santurkar et al. 2018)**
- BatchNorm makes the loss landscape significantly smoother.
- Smoother landscape = more predictable gradients = can use larger learning rates.
- This is currently the most supported explanation.

**Theory 3: Implicit Regularization**
- Batch statistics add noise to the forward pass (different batches give different means and
  variances). This noise acts as a regularizer, similar to dropout.
- This is why BatchNorm with very large batch sizes (where statistics are nearly exact)
  provides less regularization.

**Theory 4: Reparameterization**
- BatchNorm effectively reparameterizes the loss function in a way that makes optimization
  easier, regardless of the underlying mechanism.

### The Backward Pass Through BatchNorm

This is more complex than a linear layer. The key gradients:

```
dL/dgamma = sum_{i=1}^{m} dL/dy_i * x_hat_i
dL/dbeta  = sum_{i=1}^{m} dL/dy_i

dL/dx_hat_i = dL/dy_i * gamma

dL/dsigma^2_B = sum_{i=1}^{m} dL/dx_hat_i * (x_i - mu_B) * (-1/2) * (sigma^2_B + epsilon)^{-3/2}

dL/dmu_B = sum_{i=1}^{m} dL/dx_hat_i * (-1/sqrt(sigma^2_B + epsilon))
         + dL/dsigma^2_B * (-2/m) * sum_{i=1}^{m} (x_i - mu_B)

dL/dx_i = dL/dx_hat_i * (1/sqrt(sigma^2_B + epsilon))
        + dL/dsigma^2_B * (2/m) * (x_i - mu_B)
        + dL/dmu_B * (1/m)
```

This complexity is why autograd is valuable. You implement BatchNorm's forward pass, and
PyTorch handles the backward pass automatically.

### Code: BatchNorm from Scratch and in PyTorch

```python
import numpy as np

class BatchNorm1d:
    """BatchNorm from scratch (NumPy)."""
    def __init__(self, num_features, momentum=0.1, epsilon=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def forward(self, x):
        """x: shape (batch_size, num_features)"""
        if self.training:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.x_hat = (x - mu) / np.sqrt(var + self.epsilon)

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            # Cache for backward
            self.mu = mu
            self.var = var
            self.x = x
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        return self.gamma * self.x_hat + self.beta
```

```python
import torch.nn as nn

# PyTorch BatchNorm
bn = nn.BatchNorm1d(num_features=128)

# IMPORTANT: BatchNorm behaves differently in training vs eval
model.train()   # Uses batch statistics, updates running stats
model.eval()    # Uses running statistics, no updates

# Common bug: forgetting model.eval() at test time
# Results: noisy, inconsistent predictions
```

### Placement: Before or After Activation?

The original paper (Ioffe & Szegedy) recommends BEFORE the activation function:
```
z = Linear(x) -> z_bn = BatchNorm(z) -> a = ReLU(z_bn)
```

Some practitioners place it AFTER the activation. Both work. The convention has shifted
toward before, but this is not a settled question.

### What Can Go Wrong

- **Forgetting model.eval()**: The most common BatchNorm bug. At test time, batch statistics
  are noisy (especially with batch size 1). Running statistics are stable. Always call
  model.eval() before inference.
- **Batch size too small**: With batch size 1, batch statistics are meaningless (variance is
  zero). Use at least 16-32 samples per batch, or use LayerNorm/GroupNorm instead.
- **Dropout + BatchNorm interaction**: Dropout changes the scale of activations at training
  time, which disrupts BatchNorm's statistics. Place dropout after BatchNorm, or avoid using
  both together.
- **Freezing running stats during fine-tuning**: When fine-tuning a pre-trained model on a
  new domain, the running statistics may not match the new data. Consider resetting or
  recomputing them.

---

## 7. The Regularization Toolbox

### Intuition

Regularization is any technique that improves generalization (test performance) without
necessarily improving training performance. Neural networks have far more parameters than
data points and can memorize anything, including random noise. Regularization prevents this.

### L2 Regularization (Weight Decay)

**The math**:
```
L_total = L_data + (lambda/2) * sum(w_ij^2)

dL_total/dw = dL_data/dw + lambda * w
```

**Effect**: Every gradient update includes a term that pulls weights toward zero. Larger
weights are pulled more strongly. This encourages the network to use small weights, which
corresponds to simpler (smoother) functions.

**Bayesian interpretation**: L2 regularization is equivalent to a Gaussian prior on the
weights: P(w) ~ N(0, 1/lambda). MAP estimation with this prior gives L2 regularization.

**In PyTorch**:
```python
# For SGD: weight_decay parameter IS L2 regularization
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# For Adam: use AdamW for proper decoupled weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

**AdamW vs Adam with weight_decay**: In Adam, the gradient is scaled by adaptive learning
rates. Adding L2 to the loss means the regularization term is also scaled, which is
undesirable. AdamW applies weight decay DIRECTLY to the weights, bypassing the adaptive
scaling. This is the correct approach for Adam-family optimizers.

### L1 Regularization

**The math**:
```
L_total = L_data + lambda * sum(|w_ij|)
dL_total/dw = dL_data/dw + lambda * sign(w)
```

**Effect**: Drives weights to exactly zero, producing sparse networks. Unlike L2, which
shrinks all weights proportionally, L1 eliminates some weights entirely.

**When to use**: When you want feature selection (some inputs are truly irrelevant). Rarely
used alone in modern deep learning; L2 is preferred.

### Dropout

**How it works**:
- During training: for each forward pass, randomly set each neuron's activation to zero with
  probability p (independently). Scale remaining activations by 1/(1-p).
- During evaluation: use all neurons with no scaling.

**The ensemble interpretation**: With n neurons, there are 2^n possible dropout masks. Each
mask defines a different sub-network. Training with dropout trains all 2^n sub-networks
simultaneously (with shared weights). At test time, using all neurons (with scaling)
approximates the ensemble average of all sub-networks.

**Inverted dropout** (the standard implementation):
```python
# Training
mask = (torch.rand_like(a) > p).float()
a = a * mask / (1 - p)    # Scale up to maintain expected value

# Evaluation (nothing changes)
a = a
```

By scaling during training, we avoid having to scale during evaluation. This is why it is
called "inverted" — the original dropout paper scaled at test time.

**Code**:
```python
import torch
import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)  # No dropout on output layer
        return x
```

**Typical rates**: p=0.5 for hidden layers (original paper), p=0.1-0.3 for input layer.
Modern practice: p=0.1 is common, especially with other regularization.

### Data Augmentation

**Intuition**: If you cannot get more data, make the data you have look more varied. Each
augmented sample encodes a prior: "the label should not change under this transformation."

**For images**:
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Advanced techniques**:
- **Cutout**: Randomly mask a square region of the image.
- **Mixup**: Blend two images and their labels: x = lambda*x1 + (1-lambda)*x2,
  y = lambda*y1 + (1-lambda)*y2. Lambda ~ Beta(alpha, alpha).
- **CutMix**: Paste a patch from one image onto another. Mix labels proportionally.

Data augmentation is often the single most effective regularization technique.

### Early Stopping

**Algorithm**:
1. Track validation loss every epoch.
2. Maintain a counter of epochs since the last improvement.
3. If the counter exceeds `patience`, stop training.
4. Restore the weights from the best epoch.

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
```

**Why it works**: Training beyond the point where validation loss starts increasing means
you are memorizing noise. Stopping early limits the effective model complexity.

**Always use early stopping.** There is no downside.

### Practical Regularization Strategy

For a new problem, apply regularization in this order:

1. **Data augmentation**: Always. The most effective and cheapest technique.
2. **Weight decay**: Always. Use lambda=1e-4 as default, tune if needed.
3. **Early stopping**: Always. No reason not to.
4. **Dropout**: If still overfitting. Start with p=0.1, increase if needed.
5. **BatchNorm**: Primarily for training speed, but has mild regularization effect.
6. **Label smoothing**: If training on noisy labels.

---

## 8. Optimization: From SGD to Adam

### Intuition

Training a neural network is optimization: find the parameters theta that minimize the loss
function L(theta). The loss landscape has millions of dimensions, is non-convex, and is
expensive to evaluate. We need efficient algorithms that work despite these challenges.

### Vanilla SGD

The simplest optimizer:
```
theta_{t+1} = theta_t - lr * g_t
```
where g_t = gradient of L with respect to theta on a mini-batch.

**Problem**: In a narrow valley (think of a steep-sided trough), SGD oscillates across the
valley walls while making slow progress along the bottom.

### SGD with Momentum

**The physics analogy**: Imagine a ball rolling down a hill. It accumulates velocity in the
direction of consistent gradient, and the velocity smooths out oscillations.

```
v_{t+1} = beta * v_t + g_t             (update velocity)
theta_{t+1} = theta_t - lr * v_{t+1}   (update position)
```

beta is typically 0.9. This means the effective gradient is an exponential moving average
of past gradients.

**Effect**:
- Along the bottom of the valley: gradients are consistent, velocity builds up, progress
  accelerates.
- Across the valley: gradients oscillate (positive then negative), velocity averages to
  near zero, oscillations are damped.

### Nesterov Momentum

**The improvement**: Instead of computing the gradient at your current position, compute it
at the position you are about to move to:

```
theta_lookahead = theta_t - lr * beta * v_t         (look ahead)
g_t = gradient at theta_lookahead                    (compute gradient there)
v_{t+1} = beta * v_t + g_t
theta_{t+1} = theta_t - lr * v_{t+1}
```

**Intuition**: "If I am going to move in this direction anyway, I might as well compute the
gradient at where I will end up, not where I am now." This gives slightly better convergence.

### AdaGrad

**The idea**: Different parameters may need different learning rates. Frequently updated
parameters should have smaller learning rates. Rarely updated parameters should have larger
learning rates.

```
s_t = s_{t-1} + g_t^2                           (accumulate squared gradients)
theta_{t+1} = theta_t - lr * g_t / (sqrt(s_t) + epsilon)
```

s_t grows monotonically, so the effective learning rate always decreases.

**Problem**: The learning rate eventually goes to zero and the optimizer stops learning. This
makes AdaGrad unsuitable for deep learning (training runs are long).

### RMSProp

**Fix**: Use an exponential moving average instead of a sum:

```
s_t = beta * s_{t-1} + (1 - beta) * g_t^2       (EMA of squared gradients)
theta_{t+1} = theta_t - lr * g_t / (sqrt(s_t) + epsilon)
```

beta = 0.99 typically. This forgets old gradients, so the learning rate does not go to zero.

### Adam: Adaptive Moment Estimation

**The full algorithm** (Kingma & Ba, 2015):

Adam combines momentum (first moment) with RMSProp (second moment), plus bias correction.

```
Hyperparameters: lr (learning rate), beta1=0.9, beta2=0.999, epsilon=1e-8

Initialize: m_0 = 0, v_0 = 0, t = 0

For each step:
  t = t + 1
  g_t = gradient at theta_{t-1}

  # Update biased first moment estimate (momentum)
  m_t = beta1 * m_{t-1} + (1 - beta1) * g_t

  # Update biased second moment estimate (RMSProp)
  v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

  # Bias correction
  m_hat_t = m_t / (1 - beta1^t)
  v_hat_t = v_t / (1 - beta2^t)

  # Parameter update
  theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

### Why Bias Correction Matters

At t=0, m_0 = 0 and v_0 = 0. After the first step:

```
m_1 = beta1 * 0 + (1 - beta1) * g_1 = (1 - beta1) * g_1
```

The expected value is E[m_1] = (1 - beta1) * E[g_1]. But we want E[m_1] = E[g_1].

The bias correction fixes this:
```
m_hat_1 = m_1 / (1 - beta1^1) = (1 - beta1) * g_1 / (1 - beta1) = g_1
```

At step t:
```
E[m_t] = (1 - beta1^t) * E[g]
E[m_hat_t] = E[m_t] / (1 - beta1^t) = E[g]
```

The bias diminishes as t grows (since beta1^t -> 0), so the correction matters most in
early training. Without correction, the first few updates are too small.

The same logic applies to v_t: without correction, v_hat is initialized too small, causing
the denominator to be too small and the effective learning rate to be too large. This can
cause instability in early training.

### AdamW: Decoupled Weight Decay

The standard Adam with L2 regularization adds lambda*w to the gradient:

```
g_t = gradient + lambda * theta_{t-1}
```

But Adam SCALES the gradient by 1/sqrt(v_hat). This means the regularization strength
varies across parameters depending on their gradient history. Parameters with large
gradients get less effective regularization. This is undesirable.

AdamW applies weight decay DIRECTLY to the parameters, after the Adam update:

```
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon) - lr * lambda * theta_{t-1}
```

This ensures every parameter gets the same regularization strength.

**Always use AdamW instead of Adam when you want weight decay.**

### Code: Adam from Scratch

```python
import numpy as np

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0

        # Initialize moment estimates
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1

        for key in self.params:
            g = grads[key]

            # Update moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g**2

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Parameter update (AdamW style)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.params[key] -= self.lr * self.weight_decay * self.params[key]
```

### SGD vs Adam: When to Use Which

| Criterion | SGD + Momentum | Adam / AdamW |
|-----------|---------------|--------------|
| Convergence speed | Slower | Faster |
| Final accuracy (vision) | Often better | Sometimes worse |
| Hyperparameter sensitivity | High (LR matters a lot) | Lower (more forgiving) |
| Per-parameter adaptation | No | Yes |
| Memory | 1x parameters (velocity) | 2x parameters (m and v) |
| Default for vision | Yes (with cosine LR) | Growing |
| Default for NLP/transformers | No | Yes (AdamW) |
| Default for starting a project | No | Yes (easier to get working) |

**Practical advice**: Start with AdamW (lr=3e-4, weight_decay=0.01). If you need to squeeze
out the last fraction of accuracy, try SGD+momentum with careful LR tuning.

### What Can Go Wrong

- **Learning rate too high**: Loss oscillates or explodes. For Adam, start with 3e-4. For
  SGD, start with 0.1 and use a LR finder.
- **Learning rate too low**: Loss decreases very slowly. Training takes forever but
  eventually gets there.
- **Wrong optimizer for the task**: Using SGD without LR scheduling on a transformer will
  fail. Using Adam on a ResNet may slightly underperform SGD+momentum.
- **Not using bias correction in Adam**: The first few updates are too small, causing slow
  start. All modern implementations include it, but custom implementations sometimes forget.
- **Using Adam with L2 instead of AdamW with weight_decay**: Reduces effectiveness of
  regularization for parameters with large adaptive learning rates.

---

## 9. Learning Rate Schedules

### Why Schedules Matter

The optimal learning rate changes during training. At the start, large LR explores the
landscape quickly. Later, small LR fine-tunes the solution. A schedule automates this.

### Step Decay

Reduce LR by a factor every N epochs:
```
lr = initial_lr * gamma^(epoch // step_size)
```
Example: lr=0.1, gamma=0.1, step_size=30. LR is 0.1 for epochs 0-29, 0.01 for 30-59,
0.001 for 60+.

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### Cosine Annealing

LR follows a cosine curve from the initial value to near zero:
```
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

### Warmup

Start with a very small LR and linearly increase to the target over K steps:
```
lr_t = target_lr * min(1, t / warmup_steps)
```

**Why**: At initialization, the model's gradients are large and noisy. A large LR causes
instability. Warming up allows the model to "settle in" before using the full learning rate.

**Especially important for**:
- Transformers (large models with LayerNorm are sensitive to early updates).
- Adam (the adaptive learning rate estimates are noisy at the start).
- Large batch training (gradient estimates are more accurate, so the model can take larger
  steps, but needs warmup to avoid early instability).

### One-Cycle Policy

Combine warmup and cosine decay:
1. Warmup: linearly increase LR from lr_max/div_factor to lr_max.
2. Decay: cosine anneal from lr_max to lr_max/final_div_factor.
3. Also anneal momentum inversely (high momentum when LR is low, low momentum when LR is
   high).

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=1000,
    pct_start=0.3,     # 30% warmup
    div_factor=25,      # initial_lr = max_lr / 25
    final_div_factor=1e4  # final_lr = initial_lr / 1e4
)
```

### The LR Range Test (Learning Rate Finder)

Algorithm to find a good starting LR:
1. Start with a very small LR (1e-7).
2. For one epoch, exponentially increase LR after each batch.
3. Record the loss at each step.
4. Plot loss vs LR.
5. The optimal LR is roughly where the loss is decreasing fastest (steepest descent),
   typically 10x less than where the loss starts to diverge.

```python
def lr_range_test(model, train_loader, criterion, start_lr=1e-7, end_lr=10, num_steps=100):
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_multiplier = (end_lr / start_lr) ** (1 / num_steps)

    lrs, losses = [], []
    current_lr = start_lr

    for i, (inputs, targets) in enumerate(train_loader):
        if i >= num_steps:
            break

        optimizer.param_groups[0]['lr'] = current_lr

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lrs.append(current_lr)
        losses.append(loss.item())
        current_lr *= lr_multiplier

    return lrs, losses
```

### Code: Complete Training Loop with Schedule

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

model = MLP(784, [256, 128], 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = OneCycleLR(optimizer, max_lr=3e-3, total_steps=len(train_loader) * num_epochs)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()  # Update LR after each batch for OneCycleLR

    # Validation
    model.eval()
    with torch.no_grad():
        # Compute validation metrics
        pass
```

---

## 10. The Debugging Checklist

This is the protocol you follow when your network is not training. It is systematic. It is
ordered by likelihood of the problem. Follow it in order.

### Step 1: Verify Your Data

**Time allocation**: Always check this first. 5-15 minutes.

- **Visually inspect** a random sample of inputs and labels. Do the labels match? Is the
  preprocessing correct? Are images right-side-up? Is text tokenized correctly?
- **Check class balance**: Is the dataset heavily imbalanced? If one class is 95% of the
  data, the network can achieve 95% accuracy by always predicting that class.
- **Check for data leakage**: Is any training data in the test set? Are features derived
  from the target?
- **Check normalization**: Are inputs scaled to a reasonable range (e.g., [0,1] or [-1,1])?
  Neural networks struggle with inputs on vastly different scales.

**This step catches 50% of all training bugs.**

### Step 2: Overfit One Batch

**Time allocation**: 5-10 minutes.

- Take a SINGLE mini-batch (e.g., 32 samples).
- Train on this batch ONLY, for hundreds of iterations.
- The training loss should go to near zero.
- If it does not: there is a bug in your model, loss function, or training loop.

```python
# Overfit one batch test
inputs, targets = next(iter(train_loader))
model = create_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")

# Loss should be < 0.01 after 1000 steps
```

**If this fails**: Your bug is in the model or training loop, not in the data.

### Step 3: Check the Training Loop

- Is `optimizer.zero_grad()` called BEFORE or at the start of each batch? If you forget it,
  gradients accumulate across batches.
- Is `loss.backward()` called BEFORE `optimizer.step()`?
- Is `model.train()` called before training and `model.eval()` before validation?
- Are you detaching tensors correctly when computing metrics?

### Step 4: Check the Loss Function

- For classification: are you using `nn.CrossEntropyLoss`? It expects RAW LOGITS, not
  softmax probabilities. If you apply softmax to the model output AND use CrossEntropyLoss,
  you are applying softmax twice.
- For binary classification: use `nn.BCEWithLogitsLoss` (applies sigmoid internally) rather
  than `nn.BCELoss` (expects probabilities).
- Check that the loss at initialization is correct:
  - For K-class classification with random weights, initial loss should be approximately
    -log(1/K) = log(K). For 10 classes: log(10) = 2.3.
  - If your initial loss is much higher or lower, something is wrong.

### Step 5: Use the Learning Rate Finder

If the model can overfit one batch but does not train on the full dataset:
- Run the LR range test.
- Set LR to roughly 10x below where loss starts diverging.
- If no LR works, the problem is elsewhere.

### Step 6: Visualize Gradient Flow

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        param_norm = param.norm().item()
        ratio = grad_norm / (param_norm + 1e-8)
        print(f"{name:30s} | grad: {grad_norm:.6f} | param: {param_norm:.6f} | ratio: {ratio:.6f}")
```

**Healthy**: Ratios are roughly 1e-3 across all layers.
**Vanishing**: Ratios decrease exponentially toward earlier layers.
**Exploding**: Ratios are very large (>1) or you see NaN.
**Dead ReLUs**: Gradient is exactly 0 for a layer.

### Step 7: Check for Dead ReLUs

```python
def check_dead_relu(model, inputs):
    """Count the fraction of dead ReLU neurons in each layer."""
    model.eval()
    hooks = []
    activations = {}

    def hook_fn(name):
        def fn(module, input, output):
            activations[name] = output
        return fn

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        model(inputs)

    for name, act in activations.items():
        dead_frac = (act == 0).float().mean().item()
        print(f"{name}: {dead_frac*100:.1f}% dead neurons")

    for h in hooks:
        h.remove()
```

If > 50% of neurons are dead in any layer, try: smaller learning rate, LeakyReLU, better
initialization.

### Step 8: Diagnose from Loss Curves

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss not decreasing | LR too low, bug in training loop | Increase LR, check loop |
| Loss oscillating wildly | LR too high | Decrease LR by 10x |
| Loss goes to NaN | LR too high, numerical overflow | Decrease LR, add gradient clipping |
| Train good, val bad | Overfitting | Add regularization, more data |
| Train bad, val bad | Underfitting | Bigger model, train longer, check data |
| Loss plateau then drop | Normal; or LR schedule kicked in | Be patient, or check schedule |
| Loss drops then rises | LR too high for this phase | Use LR schedule with decay |
| Val loss tracks train then diverges | Classic overfitting onset | Apply early stopping |

### Step 9: Common Silent Bugs

These bugs do not cause errors but produce poor results:

- **Wrong image channel order**: Model expects RGB, data is BGR (or vice versa). The model
  trains but accuracy is lower than expected.
- **Integer division in Python 3**: `7 / 2 = 3.5` but `7 // 2 = 3`. Check your index
  computations.
- **Broadcasting errors**: `a.shape = (32, 10)`, `b.shape = (10,)`. `a + b` works due to
  broadcasting, but `b.shape = (32,)` would also broadcast along a different dimension
  without error. Check shapes explicitly.
- **Not shuffling training data**: If data is sorted by class, the model sees all of class 0,
  then all of class 1, etc. This causes poor training. Always shuffle.
- **Augmentation applied to validation/test**: Data augmentation should only be applied to
  training data. If applied to validation, your validation metrics are noisy and unreliable.

### The Golden Rule

When something goes wrong, **simplify** first. Reduce the model to 1 layer. Reduce the
data to 10 samples. Reduce the input to the simplest case. Find the simplest configuration
that reproduces the problem. Then fix it. Then scale back up.

Do not try to debug a 50-layer network on a million-sample dataset. Debug a 1-layer network
on 10 samples. The bug will be the same, but it will be visible.

---

## Appendix A: PyTorch Best Practices Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

def train(model, train_loader, val_loader, config):
    """
    Complete training loop with all best practices.

    config: dict with keys:
        lr, weight_decay, epochs, patience, grad_clip_norm
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    for epoch in range(config['epochs']):
        # --- Training ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config['grad_clip_norm']
            )

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss /= total
        val_acc = correct / total

        # --- LR Schedule ---
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

        # --- Logging ---
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

    # Restore best model
    model.load_state_dict(best_model_state)
    return model, history
```

---

## Appendix B: Key Equations Reference Card

```
FORWARD PASS
  z = W @ a_prev + b
  a = activation(z)
  y_hat = softmax(z_output)
  L = -sum(y * log(y_hat))

BACKWARD PASS
  dL/dz_output = y_hat - y                    (softmax + cross-entropy)
  dL/dW_i = dL/dz_i @ a_{i-1}^T              (weight gradient)
  dL/db_i = dL/dz_i                           (bias gradient)
  dL/da_{i-1} = W_i^T @ dL/dz_i              (propagate to previous layer)
  dL/dz_{i-1} = dL/da_{i-1} * sigma'(z_{i-1}) (through activation)

INITIALIZATION
  Xavier:  W ~ N(0, 2/(n_in + n_out))         (for tanh/sigmoid)
  He:      W ~ N(0, 2/n_in)                   (for ReLU)

BATCH NORMALIZATION
  mu_B = mean(x, axis=batch)
  sigma^2_B = var(x, axis=batch)
  x_hat = (x - mu_B) / sqrt(sigma^2_B + eps)
  y = gamma * x_hat + beta

ADAM
  m_t = beta1 * m_{t-1} + (1-beta1) * g_t
  v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
  m_hat = m_t / (1 - beta1^t)
  v_hat = v_t / (1 - beta2^t)
  theta -= lr * m_hat / (sqrt(v_hat) + eps)

ADAMW (add after Adam update)
  theta -= lr * lambda * theta
```

---

## Appendix C: Recommended Reading

- **Backpropagation**: Rumelhart, Hinton, Williams (1986). "Learning representations by
  back-propagating errors." The original paper.
- **Batch Normalization**: Ioffe & Szegedy (2015). "Batch Normalization: Accelerating Deep
  Network Training by Reducing Internal Covariate Shift."
- **BatchNorm analysis**: Santurkar et al. (2018). "How Does Batch Normalization Help
  Optimization?"
- **Adam**: Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization."
- **AdamW**: Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization."
- **He Initialization**: He et al. (2015). "Delving Deep into Rectifiers."
- **Xavier Initialization**: Glorot & Bengio (2010). "Understanding the difficulty of
  training deep feedforward neural networks."
- **Dropout**: Srivastava et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks
  from Overfitting."
- **GELU**: Hendrycks & Gimpel (2016). "Gaussian Error Linear Units."
- **Residual Connections**: He et al. (2016). "Deep Residual Learning for Image Recognition."
- **Loss Landscape Visualization**: Li et al. (2018). "Visualizing the Loss Landscape of
  Neural Nets."
- **Zhang et al. (2017)**: "Understanding deep learning requires rethinking generalization."
  The paper showing networks can memorize random labels.
