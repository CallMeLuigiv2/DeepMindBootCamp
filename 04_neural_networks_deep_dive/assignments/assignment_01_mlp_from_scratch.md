# Assignment 1: MLP From Scratch

## Overview

You will build a fully functional multi-layer perceptron from scratch using only NumPy. No
PyTorch, no TensorFlow, no automatic differentiation. You will implement every computation
yourself: the forward pass, the backward pass (manual gradient computation via the chain rule),
parameter updates, and training loop.

Then you will implement the exact same architecture in PyTorch and verify that your gradients
match. This is how you prove to yourself that you truly understand backpropagation.

**Estimated time**: 8-12 hours.

---

## Part 1: NumPy MLP Implementation

### Requirements

Build a class `MLP` that supports:

1. **Arbitrary architecture**: The constructor takes a list of layer sizes.
   Example: `MLP([784, 128, 64, 10])` creates a 3-layer network (input 784, hidden 128,
   hidden 64, output 10).

2. **At least 3 activation functions**: Implement ReLU, Sigmoid, and Tanh. Each must include
   both the forward function and its derivative.

3. **Forward pass**: Given input x, compute the output through all layers. Store all
   intermediate values (pre-activations z and post-activations a) because you will need them
   for the backward pass.

4. **Backward pass**: Given the loss gradient dL/dy_hat, compute gradients for ALL parameters
   (every W and b matrix) using the chain rule. This is the core of the assignment. You must
   implement this yourself — no autograd.

5. **Parameter update**: Implement vanilla SGD. Update each parameter using:
   $W \mathrel{-}= \eta \cdot dW$

6. **Loss function**: Implement softmax + cross-entropy loss for classification.
   - Forward: convert logits to probabilities via softmax, compute cross-entropy.
   - Backward: the gradient of softmax + cross-entropy simplifies to (y_hat - y_true).
   - You must derive this simplification and include the derivation in your notebook.

### Skeleton Code

```python
import numpy as np

class MLP:
    def __init__(self, layer_sizes, activation='relu'):
        """
        layer_sizes: list of ints, e.g. [784, 128, 64, 10]
        activation: 'relu', 'sigmoid', or 'tanh'
        """
        self.num_layers = len(layer_sizes) - 1
        self.params = {}
        self.activation_name = activation

        # Initialize weights using He initialization for ReLU, Xavier for others
        for i in range(self.num_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            if activation == 'relu':
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = np.sqrt(2.0 / (n_in + n_out))
            self.params[f'W{i}'] = np.random.randn(n_out, n_in) * scale
            self.params[f'b{i}'] = np.zeros((n_out, 1))

    def activation(self, z):
        """Apply activation function."""
        # YOUR CODE: implement relu, sigmoid, tanh
        pass

    def activation_derivative(self, z):
        """Compute derivative of activation function."""
        # YOUR CODE
        pass

    def softmax(self, z):
        """Numerically stable softmax."""
        # YOUR CODE: subtract max for numerical stability
        pass

    def forward(self, x):
        """
        Forward pass. Store all intermediates.
        x: shape (n_features, batch_size)
        Returns: y_hat (probabilities), shape (n_classes, batch_size)
        """
        self.cache = {}
        # YOUR CODE: loop through layers, apply linear transform + activation
        # Store z_i and a_i for each layer in self.cache
        pass

    def compute_loss(self, y_hat, y_true):
        """
        Cross-entropy loss.
        y_hat: predicted probabilities, shape (n_classes, batch_size)
        y_true: one-hot encoded, shape (n_classes, batch_size)
        Returns: scalar loss
        """
        # YOUR CODE
        pass

    def backward(self, y_hat, y_true):
        """
        Backward pass. Compute gradients for all parameters.
        Returns: dictionary of gradients {dW0, db0, dW1, db1, ...}
        """
        grads = {}
        m = y_true.shape[1]  # batch size

        # Output layer gradient (softmax + cross-entropy)
        # dz = y_hat - y_true

        # YOUR CODE: propagate gradients backward through all layers
        # For each layer i (from last to first):
        #   dW_i = (1/m) * dz_i @ a_{i-1}.T
        #   db_i = (1/m) * sum(dz_i, axis=1, keepdims=True)
        #   da_{i-1} = W_i.T @ dz_i
        #   dz_{i-1} = da_{i-1} * activation_derivative(z_{i-1})
        pass

    def update_params(self, grads, learning_rate):
        """Vanilla SGD update."""
        for i in range(self.num_layers):
            self.params[f'W{i}'] -= learning_rate * grads[f'dW{i}']
            self.params[f'b{i}'] -= learning_rate * grads[f'db{i}']

    def train_step(self, x, y, learning_rate):
        """One training step: forward, loss, backward, update."""
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y)
        grads = self.backward(y_hat, y)
        self.update_params(grads, learning_rate)
        return loss
```

### Training on MNIST

1. Load MNIST (you may use `sklearn.datasets.fetch_openml` or download it directly).
2. Preprocess: normalize pixels to [0, 1] or standardize to zero mean, unit variance.
3. One-hot encode labels.
4. Split into train (55,000), validation (5,000), test (10,000).
5. Train for 50+ epochs with mini-batch SGD (batch size 64 or 128).
6. Track training loss, training accuracy, and validation accuracy each epoch.
7. **Target: >95% test accuracy.**

---

## Part 2: PyTorch Verification

### Gradient Matching

1. Create the EXACT same architecture in PyTorch (`nn.Linear` layers with the same
   activation functions).
2. Copy your NumPy weights into the PyTorch model (use `model.layer.weight.data = ...`).
3. Run ONE forward+backward pass on the same batch of data.
4. Compare your NumPy gradients to PyTorch's `.grad` attributes.
5. The maximum absolute difference should be < 1e-5.

```python
import torch
import torch.nn as nn

class PyTorchMLP(nn.Module):
    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # no activation on last layer
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Copy weights from NumPy model to PyTorch model
def copy_weights(numpy_mlp, pytorch_mlp):
    """Copy weights from your NumPy MLP to the PyTorch MLP."""
    # YOUR CODE
    pass

# Compare gradients
def compare_gradients(numpy_grads, pytorch_model):
    """Compare NumPy gradients to PyTorch gradients. Report max difference."""
    # YOUR CODE
    pass
```

### Training Comparison

1. Train BOTH models on MNIST for the same number of epochs with the same hyperparameters.
2. Plot training loss curves for both on the same graph.
3. They should be nearly identical (small differences due to floating point are acceptable).
4. Report final test accuracy for both.

---

## Part 3: Analysis

Write a short report (1-2 pages in your notebook) covering:

1. **Gradient derivation**: Show the full chain rule derivation for $\frac{\partial \mathcal{L}}{\partial W_1}$ (the first layer
   weights) for your specific architecture. Every step.

2. **The softmax-cross-entropy gradient simplification**: Derive why $\frac{\partial \mathcal{L}}{\partial \mathbf{z}}$ (for the output
   layer) simplifies to $\hat{\mathbf{y}} - \mathbf{y}_{\text{true}}$. This is one of the most elegant results in
   neural network math.

3. **Numerical gradient checking**: For a small network (e.g., [4, 3, 2]), compute gradients
   numerically using finite differences:

   $$\frac{\partial \mathcal{L}}{\partial w_{ij}} \approx \frac{\mathcal{L}(w_{ij} + \epsilon) - \mathcal{L}(w_{ij} - \epsilon)}{2\epsilon}$$

   Compare to your analytical gradients. Report the relative error.

4. **What you learned**: What was the hardest part? What confused you? What clicked?

---

## Deliverables

1. A Jupyter notebook containing:
   - Complete MLP implementation in NumPy (all methods filled in, well-documented).
   - Training on MNIST with >95% test accuracy.
   - PyTorch verification with gradient comparison.
   - Training curve comparison plot.
   - Written analysis (gradient derivation, softmax simplification, numerical check).
2. The notebook must run end-to-end without errors.
3. All plots must have labeled axes, titles, and legends.

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Correctness | 30% | Forward pass, backward pass, and gradient computation are correct. Verified by matching PyTorch to < 1e-5. |
| MNIST accuracy | 15% | Achieves >95% test accuracy with the NumPy implementation. |
| Code quality | 15% | Clean, well-documented code. Meaningful variable names. No unnecessary complexity. |
| Gradient derivation | 20% | Full chain rule derivation is correct and clearly presented. Softmax simplification is derived. |
| Analysis | 10% | Thoughtful comparison of NumPy vs PyTorch. Numerical gradient check is correct. |
| Plots | 10% | Training curves, gradient comparison, and any additional visualizations are clear and informative. |

---

## Stretch Goals

For those who want to go further:

1. **Add momentum**: Implement SGD with momentum. Compare training curves with and without
   momentum.
2. **Add learning rate scheduling**: Implement step decay or cosine annealing. Show the effect.
3. **Add L2 regularization**: Implement weight decay in your NumPy MLP. Show the effect on
   overfitting with a small dataset.
4. **Batch normalization from scratch**: Add BatchNorm layers to your NumPy MLP. This is
   quite challenging — the backward pass through BatchNorm is non-trivial.
5. **Mini-batch implementation details**: Implement proper data shuffling, batching for
   non-divisible dataset sizes, and training/evaluation mode.
6. **Speed comparison**: Time your NumPy implementation vs PyTorch for 100 training steps.
   Explain the speed difference (hint: BLAS optimizations, GPU if available).

---

## Hints

- **Numerical stability in softmax**: Always subtract the max of $\mathbf{z}$ before exponentiating.
  $\exp(z - \max(z))$ prevents overflow.
- **Numerical stability in cross-entropy**: Add a small $\epsilon$ inside the log.
  $\log(\hat{y} + 10^{-8})$ prevents $\log(0)$.
- **Matrix dimensions**: The most common bug is transposing a matrix that should not be
  transposed, or vice versa. Write out dimensions for every operation.
- **Gradient checking**: If your gradients do not match PyTorch, start with a TINY network
  (e.g., [2, 2, 1]) and trace through the computation by hand. The bug is usually in the
  backward pass.
- **Batch dimension**: Be consistent about whether batch is the first or second dimension.
  The skeleton above uses batch as the second dimension (columns), which matches mathematical
  convention but differs from PyTorch's convention (batch first).
