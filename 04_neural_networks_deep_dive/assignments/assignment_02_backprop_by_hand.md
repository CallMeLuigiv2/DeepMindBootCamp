# Assignment 2: Backpropagation By Hand

## Overview

This is a pencil-and-paper assignment. You will compute the forward pass and backward pass
of a small neural network entirely by hand, with specific numerical values. No code until the
verification step.

The purpose is to make backpropagation viscerally concrete. When you have computed
dL/dW1[0,0] by multiplying specific numbers through the chain rule, you will never again
think of backprop as a black box.

**Estimated time**: 6-8 hours.

---

## Part 1: Forward Pass By Hand

### The Network

Consider a 3-layer network with the following architecture:

- Input: x in R^2 (2 features)
- Hidden layer 1: 3 neurons, ReLU activation
- Hidden layer 2: 2 neurons, ReLU activation
- Output layer: 2 neurons, softmax activation
- Loss: cross-entropy

### The Specific Weights

```
Layer 1 (2 -> 3):
W1 = [[ 0.1,  0.2],
      [-0.3,  0.4],
      [ 0.5, -0.1]]
b1 = [0.1, -0.2, 0.0]

Layer 2 (3 -> 2):
W2 = [[ 0.3, -0.2,  0.1],
      [ 0.4,  0.1, -0.3]]
b2 = [0.0, 0.1]

Layer 3 (2 -> 2):
W3 = [[ 0.2,  0.5],
      [-0.1,  0.3]]
b3 = [0.1, -0.1]
```

### The Input

```
x = [1.0, 2.0]
y_true = [1, 0]  (class 0, one-hot encoded)
```

### Task

Compute every intermediate value in the forward pass. Show all work.

1. **Layer 1 pre-activation**: z1 = W1 @ x + b1. Write out each element.
2. **Layer 1 post-activation**: a1 = relu(z1). Write out each element.
3. **Layer 2 pre-activation**: z2 = W2 @ a1 + b2. Write out each element.
4. **Layer 2 post-activation**: a2 = relu(z2). Write out each element.
5. **Output pre-activation**: z3 = W3 @ a2 + b3. Write out each element.
6. **Output probabilities**: y_hat = softmax(z3). Write out each element.
7. **Loss**: L = -sum(y_true * log(y_hat)). Compute the scalar value.

**Example for z1[0]**:
```
z1[0] = W1[0,0]*x[0] + W1[0,1]*x[1] + b1[0]
      = 0.1 * 1.0 + 0.2 * 2.0 + 0.1
      = 0.1 + 0.4 + 0.1
      = 0.6
```

Continue this for every value. Yes, it is tedious. That is the point.

---

## Part 2: Backward Pass By Hand

Now compute the gradients for every parameter in the network, working backward from the loss.

### Step 1: Output Layer Gradient

Compute dL/dz3 = y_hat - y_true. Write out each element.

### Step 2: Gradients for W3 and b3

```
dL/dW3 = dL/dz3 @ a2^T    (outer product, shape 2x2)
dL/db3 = dL/dz3            (shape 2)
```

Compute each element of dL/dW3 explicitly.

### Step 3: Propagate to Layer 2

```
dL/da2 = W3^T @ dL/dz3     (shape 2)
dL/dz2 = dL/da2 * relu'(z2) (element-wise, shape 2)
```

Where relu'(z) = 1 if z > 0, 0 if z <= 0. Check the values of z2 you computed in the
forward pass to determine which neurons are active.

### Step 4: Gradients for W2 and b2

```
dL/dW2 = dL/dz2 @ a1^T     (shape 2x3)
dL/db2 = dL/dz2             (shape 2)
```

Compute each element explicitly.

### Step 5: Propagate to Layer 1

```
dL/da1 = W2^T @ dL/dz2      (shape 3)
dL/dz1 = dL/da1 * relu'(z1)  (element-wise, shape 3)
```

Again, check which neurons in z1 are active.

### Step 6: Gradients for W1 and b1

```
dL/dW1 = dL/dz1 @ x^T       (shape 3x2)
dL/db1 = dL/dz1              (shape 3)
```

Compute each element explicitly.

### Summary Table

Fill in this table with all your computed gradients:

| Parameter | Shape | Gradient (your computation) |
|-----------|-------|-----------------------------|
| W1[0,0] | scalar | |
| W1[0,1] | scalar | |
| W1[1,0] | scalar | |
| W1[1,1] | scalar | |
| W1[2,0] | scalar | |
| W1[2,1] | scalar | |
| b1[0] | scalar | |
| b1[1] | scalar | |
| b1[2] | scalar | |
| W2[0,0] | scalar | |
| W2[0,1] | scalar | |
| W2[0,2] | scalar | |
| W2[1,0] | scalar | |
| W2[1,1] | scalar | |
| W2[1,2] | scalar | |
| b2[0] | scalar | |
| b2[1] | scalar | |
| W3[0,0] | scalar | |
| W3[0,1] | scalar | |
| W3[1,0] | scalar | |
| W3[1,1] | scalar | |
| b3[0] | scalar | |
| b3[1] | scalar | |

---

## Part 3: Verification with PyTorch

Write a PyTorch script that:

1. Creates the exact same network with the exact same weights.
2. Passes the exact same input through the network.
3. Computes the loss.
4. Calls `loss.backward()`.
5. Prints all gradients.
6. Compares to your hand-computed values.

```python
import torch
import torch.nn as nn

# Create the network
# YOUR CODE: define layers with the exact weights from the problem

# Set requires_grad=True for all parameters
# Forward pass
# Compute loss
# Backward pass
# Print all gradients
# Compare to your hand-computed values

# IMPORTANT: PyTorch's nn.Linear stores weights as (out_features, in_features)
# and computes output = input @ weight.T + bias.
# Make sure your weight matrices are set up correctly.
```

Every gradient you computed by hand should match PyTorch to at least 4 decimal places.
If they do not match, your hand computation has an error. Find it and fix it.

---

## Part 4: Backpropagation Through Batch Normalization

Now add BatchNorm to the network and repeat. This is substantially harder.

### The Modified Network

Insert BatchNorm after each hidden layer (before the activation):

- z1 = W1 @ x + b1
- z1_bn = BatchNorm(z1)  (with gamma1, beta1)
- a1 = relu(z1_bn)
- z2 = W2 @ a1 + b2
- z2_bn = BatchNorm(z2)  (with gamma2, beta2)
- a2 = relu(z2_bn)
- z3 = W3 @ a2 + b3
- y_hat = softmax(z3)

### BatchNorm Parameters

For simplicity, use a batch of 2 samples:

```
x_batch = [[1.0, 2.0],
           [1.5, 0.5]]
y_batch = [[1, 0],
           [0, 1]]

gamma1 = [1.0, 1.0, 1.0]  (scale, per-neuron)
beta1  = [0.0, 0.0, 0.0]  (shift, per-neuron)
gamma2 = [1.0, 1.0]
beta2  = [0.0, 0.0]
```

### BatchNorm Forward Pass

For each neuron j in the layer, across the batch:
1. Compute batch mean: mu_j = (1/m) * sum_i(z_j^(i))
2. Compute batch variance: sigma^2_j = (1/m) * sum_i((z_j^(i) - mu_j)^2)
3. Normalize: z_hat_j^(i) = (z_j^(i) - mu_j) / sqrt(sigma^2_j + epsilon)
4. Scale and shift: z_bn_j^(i) = gamma_j * z_hat_j^(i) + beta_j

Use epsilon = 1e-5.

### Task

1. Compute the full forward pass for both samples in the batch.
2. Compute the backward pass for at least the BatchNorm layer in layer 1.
   The BatchNorm backward pass involves:
   - dL/dgamma_j = sum_i(dL/dz_bn_j^(i) * z_hat_j^(i))
   - dL/dbeta_j = sum_i(dL/dz_bn_j^(i))
   - dL/dz_j^(i) = (gamma_j / sqrt(sigma^2_j + epsilon)) * (dL/dz_bn_j^(i) - (1/m)*dL/dbeta_j - (z_hat_j^(i)/m)*dL/dgamma_j)
3. Verify with PyTorch.

This is challenging. The point is to see that BatchNorm's backward pass is more complex
than a simple linear layer, which is why understanding autograd matters.

---

## Part 5: Essay — "Why Backpropagation Is Efficient"

Write a short essay (500-800 words) answering the question: **Why is backpropagation efficient?**

Your essay must address:

1. **The naive alternative**: To compute dL/dw for each of N parameters using numerical
   differentiation, we need N+1 forward passes (or 2N for central differences). For a model
   with 1 million parameters, that is 1 million forward passes per gradient computation.
   State the total computational complexity.

2. **Backpropagation's complexity**: One forward pass + one backward pass. The backward pass
   has roughly the same cost as the forward pass (2-3x due to storing intermediates). State
   the total computational complexity.

3. **The ratio**: For a model with N parameters, backprop is O(N) times faster than numerical
   differentiation. For modern models with billions of parameters, this is the difference
   between "feasible" and "impossible."

4. **Why it works**: The key insight is that the chain rule allows us to reuse intermediate
   computations. When computing dL/dW1, we reuse dL/dz2 which was already computed for
   dL/dW2. This sharing of computation is what makes it efficient.

5. **Forward mode vs reverse mode**: Briefly explain the difference. Why is reverse mode
   (backprop) preferred when we have many parameters and one scalar loss?

---

## Deliverables

1. **Handwritten or typed computation sheets** for Parts 1 and 2 (PDF or in a notebook).
   Every intermediate value must be shown. No skipping steps.
2. **PyTorch verification script** for Part 3 (working code in a notebook).
3. **BatchNorm forward/backward computation** for Part 4 (at least the forward pass and
   BatchNorm backward pass for one layer). PyTorch verification.
4. **Essay** for Part 5 (500-800 words, in the same notebook or separate document).

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Forward pass correctness | 20% | All intermediate values are correct (verified by PyTorch). |
| Backward pass correctness | 30% | All gradients are correct (verified by PyTorch). No steps skipped. |
| BatchNorm computation | 20% | Forward pass and at least partial backward pass through BatchNorm. Verified by PyTorch. |
| Essay quality | 20% | Clear explanation of computational complexity. Forward vs reverse mode discussed. Specific numbers used. |
| Presentation | 10% | Work is organized, legible, and easy to follow. Summary table is filled in. |

---

## Common Mistakes to Watch For

1. **Forgetting to check ReLU activation**: If z <= 0, the gradient is zero. Dead neurons
   propagate zero gradient. Check your z values from the forward pass.
2. **Transposition errors**: dL/dW = dL/dz @ a_prev^T, NOT a_prev @ dL/dz^T. Dimensions
   must work out correctly.
3. **Softmax gradient**: The gradient of cross-entropy with softmax simplifies to
   (y_hat - y_true) ONLY when combined. Do not compute them separately.
4. **BatchNorm with batch size 2**: The variance computation with m=2 produces high variance
   estimates. This is normal for the exercise; in practice, use larger batches.
5. **PyTorch weight convention**: `nn.Linear(in, out)` stores weight as shape (out, in).
   Make sure your hand-computation matches this convention.
