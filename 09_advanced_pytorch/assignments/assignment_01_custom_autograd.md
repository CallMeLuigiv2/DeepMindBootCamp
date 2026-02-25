# Assignment 1: Custom Autograd Mastery

## Overview

The autograd engine is the foundation of everything PyTorch does. Most users never need to touch it because composing existing differentiable operations is sufficient. But when you encounter a non-differentiable operation, a numerically unstable gradient, or a custom CUDA kernel, you need to write your own backward pass. This assignment ensures you can do that correctly and confidently.

By the end of this assignment, you will have implemented custom autograd functions that are production-quality: verified, efficient, and correct.

---

## Part 1: Custom Swish/SiLU with Learnable Beta

### Background

The SiLU (Sigmoid Linear Unit) activation is defined as `f(x) = x * sigmoid(x)`. A parameterized variant uses a learnable scaling parameter: `f(x) = x * sigmoid(beta * x)`, where `beta` is a learnable scalar.

PyTorch's built-in `torch.nn.SiLU` does not support a learnable `beta`. You will implement this as a custom autograd Function.

### Task

1. **Derive the gradients analytically.** Given `f(x) = x * sigmoid(beta * x)`, compute:
   - `df/dx` (the gradient with respect to the input)
   - `df/dbeta` (the gradient with respect to the parameter)

   Show your derivation in a comment or markdown cell. The chain rule through sigmoid is essential here: `sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))`.

2. **Implement `ParameterizedSwish` as a `torch.autograd.Function`.**
   - `forward(ctx, x, beta)` should compute `x * sigmoid(beta * x)`.
   - `backward(ctx, grad_output)` should return the analytically derived gradients for both `x` and `beta`.
   - Use `ctx.save_for_backward` correctly.

3. **Wrap it in an `nn.Module`** called `LearnableSwish`:
   - The module should have a single learnable parameter `beta`, initialized to 1.0.
   - The forward method should call `ParameterizedSwish.apply`.

4. **Verify correctness:**
   - Compare the output of your implementation against a pure-PyTorch computation: `x * torch.sigmoid(beta * x)`. They should match to floating-point precision.
   - Compare the gradients by running `.backward()` on both and checking `.grad` values.
   - Run `torch.autograd.gradcheck` with float64 inputs. It must pass.

### Deliverables

- `custom_swish.py` containing the `ParameterizedSwish` Function and `LearnableSwish` Module.
- A test script or notebook that demonstrates correctness (output match, gradient match, gradcheck).

---

## Part 2: Straight-Through Estimator

### Background

Many useful operations are non-differentiable: rounding, thresholding, argmax, sampling from a categorical distribution. The Straight-Through Estimator (STE) is a technique that assigns a surrogate gradient to these operations. In the forward pass, the non-differentiable operation is applied normally. In the backward pass, the gradient is passed through as if the operation were the identity function (or a clamped identity).

This is used in:
- Quantization-aware training (rounding weights/activations to fixed-point)
- Binary neural networks (sign function for binary weights)
- Discrete VAEs (Gumbel-Softmax with hard sampling)

### Task

1. **Implement `HardThresholdSTE` as a `torch.autograd.Function`.**
   - Forward: output is 1 where input >= 0, else 0 (a step function).
   - Backward: pass the gradient through unchanged (identity).

2. **Implement `ClampedSTE`**, a variant where the backward pass only passes gradients through for inputs in the range `[-1, 1]`. Outside this range, the gradient is zero. This prevents unbounded gradient flow for inputs far from the threshold.

3. **Train a binary activation network:**
   - Build a small network (3 linear layers, hidden sizes 256 and 128) for MNIST classification.
   - Replace ReLU activations with your `HardThresholdSTE`.
   - Train for 10 epochs on MNIST.
   - Report the final accuracy. It should be above 90% (not as good as continuous activations, but surprisingly good given that all hidden activations are binary).

4. **Compare STE variants:**
   - Train the same network with `HardThresholdSTE` and `ClampedSTE`.
   - Plot training loss curves for both.
   - Which converges faster? Which reaches higher accuracy? Explain why.

### Deliverables

- `ste.py` containing both STE implementations.
- Training script or notebook showing MNIST training with binary activations.
- Loss curves and final accuracy for both STE variants.

---

## Part 3: Custom Loss Function

### Background

Sometimes you need a loss function with a non-trivial backward pass that differs from what autograd would compute. Examples include:
- Numerically stable implementations (log-sum-exp trick)
- Losses with discontinuities where you want a smooth surrogate gradient
- Losses involving non-differentiable operations (like IoU for bounding boxes)

### Task

1. **Implement `AsymmetricMSELoss` as a custom autograd Function.**
   - This loss penalizes overestimation and underestimation differently.
   - Forward: `L = mean(alpha * max(0, y_pred - y_true)^2 + beta * max(0, y_true - y_pred)^2)` where `alpha` and `beta` are constants (e.g., alpha=1.0, beta=2.0 penalizes underestimation more).
   - Backward: Compute the correct gradient of this loss with respect to `y_pred`. `y_true` does not require a gradient.

2. **Verify with gradcheck.** Use float64 inputs.

3. **Use the custom loss** to train a simple regression model on synthetic data where underestimation is more costly than overestimation (e.g., predicting resource usage where underestimation causes outages). Show that with `beta > alpha`, the model's predictions are biased upward compared to standard MSE.

### Deliverables

- `asymmetric_loss.py` containing the custom loss function.
- Training comparison: standard MSE vs asymmetric MSE on synthetic data.
- Visualization showing the prediction bias.

---

## Part 4: Performance Benchmark

### Task

1. **Benchmark your `ParameterizedSwish` custom autograd Function** against a pure-PyTorch implementation: `x * torch.sigmoid(beta * x)`.
   - Measure forward pass time and backward pass time for various input sizes (1K, 10K, 100K, 1M elements).
   - Use `torch.cuda.Event` for GPU timing or `time.perf_counter` for CPU.
   - Run each measurement at least 100 times and report mean and standard deviation.

2. **Analyze the results:**
   - Is the custom Function faster, slower, or the same as the pure-PyTorch version?
   - Explain why. (Hint: for a composition of standard operations, the custom Function is rarely faster — PyTorch's autograd is already very efficient. The value of custom Functions is correctness, not performance.)

3. **Memory comparison:**
   - Measure peak memory usage for both implementations using `torch.cuda.max_memory_allocated()`.
   - Does the custom Function use less memory? (It might, because you choose exactly what to save for backward.)

### Deliverables

- Benchmark script with timing results.
- Table or chart comparing performance.
- Written analysis of results (2-3 paragraphs).

---

## Evaluation Criteria

### Passing

- All custom autograd Functions have correct forward and backward implementations.
- `torch.autograd.gradcheck` passes for all Functions with float64 inputs.
- The STE-based binary network trains and achieves above 90% accuracy on MNIST.
- The asymmetric loss produces a measurable prediction bias compared to standard MSE.
- Code is clean, well-commented, and follows PyTorch conventions.

### Distinction

All of the above, plus:
- Benchmark is thorough with proper statistical analysis (mean, std, multiple runs).
- Written analysis demonstrates understanding of why custom Functions do or do not provide performance benefits.
- Clear visualization of results.

---

## Stretch Goals

1. **Higher-order gradients:** Modify your `ParameterizedSwish` to support `torch.autograd.gradgradcheck` (double backward). This requires implementing `backward` in a way that is itself differentiable. The trick: inside `backward`, use PyTorch operations on tensors with `requires_grad=True` rather than manual gradient formulas with `.data`. Verify with `torch.autograd.gradgradcheck`.

2. **Custom function for log-sum-exp:** Implement a numerically stable log-sum-exp as a custom autograd Function. The forward should use the max-subtraction trick: `log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))`. The backward should return the correct gradient (softmax). Verify it is more numerically stable than the naive implementation by testing with large input values (e.g., `torch.tensor([1000.0, 1001.0, 1002.0])`).

3. **Custom CUDA kernel in autograd:** If you have CUDA experience, implement a simple custom operation (e.g., fused add + multiply) as a CUDA kernel, wrap it in a custom autograd Function, and benchmark against the pure-PyTorch equivalent.
