# Assignment 02: Autograd Deep Dive — Understanding the Engine

**Module 3, Session 2 | Estimated time: 5-7 hours**

---

## Objective

Autograd is the engine that makes deep learning practical. Most practitioners treat it as a
black box — they call `loss.backward()` and trust that gradients appear. You will not be most
practitioners. By the end of this assignment, you will be able to trace gradient flow through
any computational graph, implement custom backward passes, and debug gradient-related issues
with confidence.

---

## Part 1: Manual Backward Pass (60 minutes)

### Exercise 1.1: Linear Model

Consider the following computation:

```python
import torch

# Parameters
w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2, 2)
b = torch.tensor([0.5, -0.5], requires_grad=True)                # (2,)

# Input and target
x = torch.tensor([1.0, -1.0])    # (2,)
target = torch.tensor([2.0, 0.0]) # (2,)

# Forward pass
z = x @ w + b          # (2,)
loss = ((z - target) ** 2).mean()  # scalar

# Backward pass
loss.backward()
```

**Tasks:**

1. **On paper**, compute the forward pass step by step. What is `z`? What is `loss`?
2. **On paper**, compute `dloss/dw` and `dloss/db` using the chain rule. Show every step.
3. Draw the computational graph as ASCII art, labeling each node with its `grad_fn`.
4. Run the code and verify your hand-computed gradients match `w.grad` and `b.grad`.
5. If they do not match, find your error. This is the exercise.

### Exercise 1.2: Non-Linear Computation

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b           # 6.0
d = torch.sin(c)    # sin(6.0)
e = d ** 2          # sin(6.0)^2
f = e + a           # sin(6.0)^2 + 2.0
loss = f * b        # (sin(6.0)^2 + 2.0) * 3.0
```

**Tasks:**

1. Compute the value of every intermediate variable.
2. Draw the computational graph.
3. Compute `dloss/da` and `dloss/db` by hand using the chain rule. You will need:
   - `d/dx[sin(x)] = cos(x)`
   - `d/dx[x^2] = 2x`
   - The chain rule applied at every node.
4. Verify against PyTorch's autograd.

### Exercise 1.3: Branching Graph

```python
x = torch.tensor(3.0, requires_grad=True)

# x is used in TWO branches
a = x ** 2      # branch 1: 9.0
b = x ** 3      # branch 2: 27.0

loss = a + b     # 36.0
```

**Tasks:**

1. Compute `dloss/dx` by hand. Note that `x` participates in two paths.
2. Explain why gradients from both branches are **summed** at `x`.
3. Verify with PyTorch.
4. Now add a third branch: `c = torch.exp(x)`. Recompute and verify.

---

## Part 2: Custom Autograd Function (90 minutes)

### Exercise 2.1: Implement Swish Activation

The Swish activation function is defined as:

```
swish(x) = x * sigmoid(x)
```

Its derivative is:

```
swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
         = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
```

**Tasks:**

1. Implement Swish as a `torch.autograd.Function` with explicit `forward` and `backward` methods.
2. Use `ctx.save_for_backward` to save what you need for the backward pass.
3. Apply your Swish function to a tensor and call `.backward()`.
4. Verify correctness using `torch.autograd.gradcheck` with float64 tensors.
5. Compare the output and gradients to a "naive" implementation:
   `y = x * torch.sigmoid(x)` (which uses PyTorch's built-in autograd).

```python
class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: implement
        pass

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: implement
        pass

swish = SwishFunction.apply
```

### Exercise 2.2: Implement Straight-Through Estimator (STE)

The STE is used when you need to backpropagate through a non-differentiable operation (like
rounding or sign). The forward pass applies the non-differentiable function, but the backward
pass pretends it is the identity function.

```
Forward: y = sign(x)  (or round, or threshold)
Backward: dy/dx = 1   (pretend it was identity)
```

**Tasks:**

1. Implement the STE as a `torch.autograd.Function`.
2. The forward pass should return `torch.sign(x)`.
3. The backward pass should pass `grad_output` through unchanged.
4. Demonstrate that gradients flow through the STE by using it in a small computation
   and calling `.backward()`.
5. Explain in a comment: where would you use this in practice? (Hint: quantization, binary
   neural networks.)

### Exercise 2.3: Implement a Custom Clamp Function

Implement `torch.clamp(x, min_val, max_val)` as a custom autograd Function.

**Key insight:** The gradient of clamp is:
- 1 where `min_val < x < max_val` (the value was unchanged)
- 0 where `x <= min_val` or `x >= max_val` (the value was clamped)

**Tasks:**

1. Implement the custom Clamp function.
2. Verify with `gradcheck`.
3. Compare against `torch.clamp` to ensure identical forward and backward behavior.

---

## Part 3: Gradient Accumulation Experiments (45 minutes)

### Exercise 3.1: Observing Accumulation

```python
w = torch.tensor(1.0, requires_grad=True)
```

1. Compute `y1 = w * 2` and call `y1.backward()`. Print `w.grad`.
2. Without zeroing, compute `y2 = w * 3` and call `y2.backward()`. Print `w.grad`.
3. Without zeroing, compute `y3 = w * 5` and call `y3.backward()`. Print `w.grad`.
4. Explain the values you observe.
5. Zero the gradient and repeat step 3. Verify the gradient is now correct.

### Exercise 3.2: Simulating Large Batch Training

Gradient accumulation is used to simulate large batch sizes when GPU memory is limited.

**Task:** Write a training loop that:

1. Uses a batch size of 8 for the DataLoader.
2. Accumulates gradients over 4 batches before calling `optimizer.step()`.
3. Scales the loss by `1/accumulation_steps` to normalize.
4. Effectively simulates a batch size of 32.

Use a simple linear model on synthetic data. Compare the final model weights when training with:
- True batch size 32 (no accumulation)
- Simulated batch size 32 (batch 8, accumulate 4)

They should converge to similar (not identical, due to BatchNorm/dropout-free architecture)
results.

### Exercise 3.3: When Accumulation Goes Wrong

Create a deliberate example where forgetting `zero_grad()` causes training to diverge. Plot
the loss curve for both the correct and incorrect versions. This should make the importance of
zeroing gradients viscerally obvious.

---

## Part 4: Gradient Checking (45 minutes)

Numerical gradient checking verifies that your analytical gradients (from autograd) are correct.
The idea: approximate the gradient using finite differences.

```
df/dx ≈ (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
```

### Exercise 4.1: Implement Gradient Checking

```python
def numerical_gradient(func, x, eps=1e-5):
    """
    Compute the numerical gradient of func at x using central differences.

    Args:
        func: a function that takes a tensor and returns a scalar
        x: the point at which to compute the gradient (1D tensor)
        eps: the finite difference step size

    Returns:
        A tensor of the same shape as x containing numerical gradients
    """
    # TODO: implement
    pass
```

**Tasks:**

1. Implement `numerical_gradient` using the central difference formula.
2. Test it on simple functions:
   - `f(x) = x^2` at `x = 3.0` (expected gradient: 6.0)
   - `f(x) = sin(x)` at `x = pi/4` (expected gradient: cos(pi/4))
   - `f(x) = x.sum()` for a vector (expected gradient: all ones)
3. Test it on a multi-layer computation:
   ```python
   def complex_func(x):
       return (x ** 3 + torch.sin(x * 2)).sum()
   ```
4. Compare your numerical gradients to PyTorch's autograd gradients. Report the maximum
   absolute difference. It should be on the order of 1e-5 or smaller.

### Exercise 4.2: Verify Your Custom Functions

Use both `numerical_gradient` and `torch.autograd.gradcheck` to verify the custom autograd
functions from Part 2 (Swish, STE, Clamp).

For the STE, note that `gradcheck` will fail because the forward pass (sign) is not
differentiable. Explain why this is expected and why the STE is still useful.

---

## Part 5: Gradient Clipping from Scratch (30 minutes)

### Exercise 5.1: Implement clip_grad_norm

```python
def clip_grad_norm(parameters, max_norm):
    """
    Clip gradients by total norm.

    Compute the total L2 norm of all parameter gradients. If it exceeds max_norm,
    scale all gradients by (max_norm / total_norm) so the total norm equals max_norm.

    Args:
        parameters: iterable of tensors with .grad attributes
        max_norm: maximum allowed total gradient norm

    Returns:
        The total gradient norm (before clipping)
    """
    # TODO: implement
    pass
```

**Tasks:**

1. Implement `clip_grad_norm`.
2. Verify it matches `torch.nn.utils.clip_grad_norm_` on several test cases.
3. Create a scenario with exploding gradients (e.g., a deep linear network with large weights).
4. Show that clipping prevents the gradient explosion.

### Exercise 5.2: Implement clip_grad_value

```python
def clip_grad_value(parameters, clip_value):
    """
    Clip gradients by value.

    Clamp each gradient element to [-clip_value, clip_value].

    Args:
        parameters: iterable of tensors with .grad attributes
        clip_value: maximum absolute value for any gradient element
    """
    # TODO: implement
    pass
```

Verify against `torch.nn.utils.clip_grad_value_`.

---

## Part 6: Visualizing the Computational Graph (30 minutes)

### Option A: Using torchviz (Recommended)

Install torchviz: `pip install torchviz`

```python
from torchviz import make_dot

x = torch.randn(1, 10)
model = SimpleNet(10, 32, 5)  # define or import your model
y = model(x)
loss = y.sum()

dot = make_dot(loss, params=dict(model.named_parameters()), show_attrs=True)
dot.render('computational_graph', format='png')
```

**Tasks:**

1. Visualize the computational graph for a simple 2-layer network.
2. Visualize the graph for a network with a skip connection.
3. Compare the two graphs. Explain how the skip connection appears in the graph.
4. Visualize the graph for the Swish custom function. Can you see where your custom backward
   will be called?

### Option B: Manual ASCII Graph (If torchviz is unavailable)

For each of the computations in Part 1, draw a detailed ASCII graph showing:
- Every tensor (with its value and whether it requires grad)
- Every operation (with its `grad_fn` name)
- The flow of gradients in the backward pass (with numerical values)

Example format:
```
x=3.0 (leaf, requires_grad=True)
 |
 v
[PowBackward0]  (x^2)
 |
 v
a=9.0 (non-leaf, grad_fn=PowBackward0)
 |
 ...
```

---

## Part 7: Advanced Autograd Concepts (45 minutes)

### Exercise 7.1: Higher-Order Gradients

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 3  # y = x^3, dy/dx = 3x^2, d2y/dx2 = 6x

# First derivative
grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = {grad_y}")  # should be 27.0

# Second derivative
grad_grad_y = torch.autograd.grad(grad_y, x)[0]
print(f"d2y/dx2 = {grad_grad_y}")  # should be 18.0
```

**Tasks:**

1. Verify the above computation by hand.
2. Explain what `create_graph=True` does and why it is needed for higher-order gradients.
3. Compute the third derivative. What do you expect?
4. Implement Newton's method for finding the minimum of `f(x) = (x - 3)^4` using second-order
   gradients. Start from `x = 0.0`.

### Exercise 7.2: detach() vs no_grad() Experiment

```python
x = torch.randn(3, requires_grad=True)
```

1. Compute `y = x * 2` inside a `torch.no_grad()` block. Does `y` require grad? Does `y`
   have a `grad_fn`?
2. Compute `y = x * 2` outside the block, then `z = y.detach()`. Does `z` require grad?
   Does `z` have a `grad_fn`?
3. Compute `w = z * 3`. Call `w.sum().backward()`. Does `x` get a gradient? Why or why not?
4. Summarize: when would you use `no_grad()` vs `detach()`? Give a concrete use case for each.

### Exercise 7.3: retain_graph

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
```

1. Call `y.backward()`. Then call `y.backward()` again. What happens? Why?
2. Use `retain_graph=True` on the first call. Does the second call work now?
3. Explain why retaining the graph is expensive in terms of memory.
4. Describe a real scenario where `retain_graph=True` is necessary (Hint: multiple losses,
   GANs).

---

## Deliverables

Submit a Python script or Jupyter notebook containing:

1. All hand-computed gradients (Parts 1.1, 1.2, 1.3) with verification against PyTorch.
2. All custom autograd functions (Swish, STE, Clamp) with `gradcheck` verification.
3. The gradient accumulation experiments with plots showing correct vs incorrect behavior.
4. The `numerical_gradient` implementation with verification.
5. The `clip_grad_norm` and `clip_grad_value` implementations with verification.
6. Computational graph visualizations (torchviz or ASCII).
7. Higher-order gradient and detach/no_grad experiments.
8. A "Key Insights" section summarizing what you learned about how autograd works internally.

---

## Evaluation Criteria

- **Correctness:** Hand-computed gradients match PyTorch. Custom functions pass `gradcheck`.
- **Depth of Understanding:** Explanations show you understand *why*, not just *how*.
- **Graph Visualization:** Computational graphs are accurate and clearly labeled.
- **Gradient Checking:** Numerical gradients are within expected tolerance of analytical ones.
- **Code Quality:** Clean, well-commented code. No unnecessary complexity.

---

## Stretch Goals

1. **Implement a simple reverse-mode autodiff engine from scratch** in pure Python (no PyTorch).
   Support: add, multiply, power, sin/cos. Build a computational graph, then traverse it
   backward. This is the deepest way to understand autograd.

2. **Implement gradient checkpointing manually:** Write a function that takes a sequence of
   layers, runs the forward pass but only stores activations at "checkpoints" (e.g., every
   3rd layer). During backward, recompute the missing activations. Compare memory usage
   to the standard approach.

3. **Explore Jacobians and Hessians:** Use `torch.autograd.functional.jacobian` and
   `torch.autograd.functional.hessian` to compute full Jacobian and Hessian matrices for a
   small network. Visualize them as heatmaps. Discuss what the Hessian eigenvalues tell you
   about the loss landscape.

4. **Profile autograd overhead:** Compare the wall-clock time of a forward pass with
   `requires_grad=True` vs `requires_grad=False`. How much overhead does graph construction add?
   Test with models of increasing depth.
