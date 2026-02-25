# Assignment 1: Convolution from Scratch

## Overview

You will implement the 2D convolution operation from first principles using only NumPy. No
scipy, no PyTorch convolution functions. This forces you to understand every detail of the
convolution operation — the sliding window, multi-channel computation, stride, padding, and
most importantly, the backward pass.

By the end of this assignment, you will be able to explain exactly what happens inside
`nn.Conv2d` during both the forward and backward pass, why PyTorch is orders of magnitude
faster than your implementation, and what learned filters actually look like.

**Estimated time:** 8-12 hours

---

## Part 1: Single-Channel Convolution (Forward Pass)

### Task

Implement a function `conv2d_forward` that performs 2D convolution on a single-channel input
with a single filter.

```python
def conv2d_forward(input, kernel, stride=1, padding=0):
    """
    Perform 2D convolution (cross-correlation).

    Args:
        input: numpy array of shape (H, W)
        kernel: numpy array of shape (K_h, K_w)
        stride: int, stride of the convolution
        padding: int, zero-padding added to both sides

    Returns:
        output: numpy array of shape (H_out, W_out)
        where H_out = (H + 2*padding - K_h) // stride + 1
              W_out = (W + 2*padding - K_w) // stride + 1
    """
```

### Requirements

1. Support arbitrary kernel sizes (not just 3x3).
2. Support stride > 1.
3. Support zero-padding.
4. Compute the output using explicit loops (no vectorized tricks yet — understand the
   algorithm first).

### Verification

Test against `torch.nn.functional.conv2d`:

```python
import torch
import torch.nn.functional as F
import numpy as np

# Generate test data
np.random.seed(42)
input_np = np.random.randn(8, 8).astype(np.float32)
kernel_np = np.random.randn(3, 3).astype(np.float32)

# Your implementation
output_np = conv2d_forward(input_np, kernel_np, stride=1, padding=1)

# PyTorch reference
input_torch = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)
kernel_torch = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
output_torch = F.conv2d(input_torch, kernel_torch, padding=1).squeeze().numpy()

# Compare
print(f"Max absolute difference: {np.max(np.abs(output_np - output_torch))}")
assert np.allclose(output_np, output_torch, atol=1e-5), "Outputs do not match!"
print("PASSED: Single-channel convolution matches PyTorch")
```

Run this verification for at least 5 different configurations:
- kernel 3x3, stride 1, padding 0
- kernel 3x3, stride 1, padding 1 (same padding)
- kernel 5x5, stride 1, padding 2
- kernel 3x3, stride 2, padding 1
- kernel 7x7, stride 2, padding 3

---

## Part 2: Multi-Channel Convolution (Forward Pass)

### Task

Extend your implementation to handle batched, multi-channel inputs and multiple output filters.

```python
def conv2d_forward_multichannel(input, weight, bias=None, stride=1, padding=0):
    """
    Perform multi-channel 2D convolution.

    Args:
        input: numpy array of shape (N, C_in, H, W) — batch of images
        weight: numpy array of shape (C_out, C_in, K_h, K_w) — filters
        bias: numpy array of shape (C_out,) or None
        stride: int
        padding: int

    Returns:
        output: numpy array of shape (N, C_out, H_out, W_out)
    """
```

### Requirements

1. Handle batched inputs (N > 1).
2. Handle multiple input channels (C_in > 1).
3. Handle multiple output channels (C_out > 1).
4. Support optional bias.

### Verification

```python
# Test: match PyTorch nn.Conv2d
N, C_in, H, W = 2, 3, 16, 16
C_out, K = 8, 3

np.random.seed(42)
input_np = np.random.randn(N, C_in, H, W).astype(np.float32)

conv_torch = torch.nn.Conv2d(C_in, C_out, K, padding=1)
weight_np = conv_torch.weight.detach().numpy()
bias_np = conv_torch.bias.detach().numpy()

output_np = conv2d_forward_multichannel(input_np, weight_np, bias_np, stride=1, padding=1)
output_torch = conv_torch(torch.from_numpy(input_np)).detach().numpy()

print(f"Max absolute difference: {np.max(np.abs(output_np - output_torch))}")
assert np.allclose(output_np, output_torch, atol=1e-5), "Multi-channel outputs do not match!"
print("PASSED: Multi-channel convolution matches PyTorch")
```

---

## Part 3: Backward Pass

### Task

Implement the backward pass for your convolution operation. This is the hard part and the
most educational.

```python
def conv2d_backward(d_output, input, weight, stride=1, padding=0):
    """
    Backward pass for 2D convolution.

    Args:
        d_output: numpy array of shape (N, C_out, H_out, W_out) — gradient of loss
                  with respect to the output
        input: numpy array of shape (N, C_in, H, W) — the original input (saved from forward)
        weight: numpy array of shape (C_out, C_in, K_h, K_w) — the filters
        stride: int
        padding: int

    Returns:
        d_input: numpy array of shape (N, C_in, H, W) — gradient w.r.t. input
        d_weight: numpy array of shape (C_out, C_in, K_h, K_w) — gradient w.r.t. weights
        d_bias: numpy array of shape (C_out,) — gradient w.r.t. bias
    """
```

### The Math You Need

The gradient computations for a convolution layer:

**Gradient w.r.t. bias:**

$$\frac{\partial \mathcal{L}}{\partial b_{c_{out}}} = \sum_{n, h, w} \frac{\partial \mathcal{L}}{\partial \text{output}}[n, c_{out}, h, w]$$

This is straightforward: the bias adds to every spatial position, so its gradient is the
sum over all positions.

**Gradient w.r.t. weights:**

$$\frac{\partial \mathcal{L}}{\partial W[c_{out}, c_{in}, m, n]} = \sum_{\text{batch}, h, w} \frac{\partial \mathcal{L}}{\partial \text{output}}[\text{batch}, c_{out}, h, w] \cdot \text{input\_padded}[\text{batch}, c_{in}, h \cdot s+m, w \cdot s+n]$$

This is a convolution of the input with d_output.

**Gradient w.r.t. input:**

$\frac{\partial \mathcal{L}}{\partial \text{input}}$ is a "full" convolution of d_output with the flipped weight.
For each position in d_input, accumulate contributions from all output positions
whose receptive field includes that input position.
This is the trickiest part. Think about it carefully: each input element contributes to
multiple output elements (all the output positions whose kernel window overlaps with that
input position). The gradient flows back from all those output positions.

### Requirements

1. Implement all three gradients (d_input, d_weight, d_bias).
2. Support stride and padding.
3. Handle the multi-channel, batched case.

### Verification

This is CRITICAL. Use PyTorch autograd to verify your gradients.

```python
def verify_backward():
    """Verify backward pass against PyTorch autograd."""
    N, C_in, H, W = 2, 3, 8, 8
    C_out, K = 4, 3
    stride, padding = 1, 1

    np.random.seed(42)
    input_np = np.random.randn(N, C_in, H, W).astype(np.float32)

    # PyTorch reference
    input_torch = torch.from_numpy(input_np).requires_grad_(True)
    conv = torch.nn.Conv2d(C_in, C_out, K, stride=stride, padding=padding)
    weight_np = conv.weight.detach().numpy()
    bias_np = conv.bias.detach().numpy()

    output_torch = conv(input_torch)
    d_output_np = np.random.randn(*output_torch.shape).astype(np.float32)
    output_torch.backward(torch.from_numpy(d_output_np))

    d_input_ref = input_torch.grad.numpy()
    d_weight_ref = conv.weight.grad.numpy()
    d_bias_ref = conv.bias.grad.numpy()

    # Your implementation
    d_input, d_weight, d_bias = conv2d_backward(
        d_output_np, input_np, weight_np, stride=stride, padding=padding
    )

    # Compare
    print(f"d_input max diff:  {np.max(np.abs(d_input - d_input_ref)):.2e}")
    print(f"d_weight max diff: {np.max(np.abs(d_weight - d_weight_ref)):.2e}")
    print(f"d_bias max diff:   {np.max(np.abs(d_bias - d_bias_ref)):.2e}")

    assert np.allclose(d_input, d_input_ref, atol=1e-5), "d_input mismatch!"
    assert np.allclose(d_weight, d_weight_ref, atol=1e-5), "d_weight mismatch!"
    assert np.allclose(d_bias, d_bias_ref, atol=1e-5), "d_bias mismatch!"
    print("PASSED: All backward gradients match PyTorch!")

verify_backward()
```

Also test with stride=2 and different padding values. Edge cases matter.

---

## Part 4: Benchmarking

### Task

Compare the speed of your NumPy implementation against PyTorch's `nn.Conv2d`.

```python
import time

def benchmark(input_shape, C_out, K, stride=1, padding=0, n_runs=10):
    """Benchmark your conv vs PyTorch conv."""
    N, C_in, H, W = input_shape

    input_np = np.random.randn(*input_shape).astype(np.float32)
    weight_np = np.random.randn(C_out, C_in, K, K).astype(np.float32)
    bias_np = np.random.randn(C_out).astype(np.float32)

    # Your implementation
    start = time.time()
    for _ in range(n_runs):
        _ = conv2d_forward_multichannel(input_np, weight_np, bias_np, stride, padding)
    numpy_time = (time.time() - start) / n_runs

    # PyTorch CPU
    input_torch = torch.from_numpy(input_np)
    conv_torch = torch.nn.Conv2d(C_in, C_out, K, stride=stride, padding=padding)
    conv_torch.weight.data = torch.from_numpy(weight_np)
    conv_torch.bias.data = torch.from_numpy(bias_np)

    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = conv_torch(input_torch)
    pytorch_time = (time.time() - start) / n_runs

    speedup = numpy_time / pytorch_time
    print(f"Input: {input_shape}, Conv({C_in}, {C_out}, {K})")
    print(f"  NumPy:   {numpy_time*1000:.2f} ms")
    print(f"  PyTorch: {pytorch_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.1f}x")
    return numpy_time, pytorch_time
```

### Run benchmarks at these scales

```python
# Small (should be manageable for NumPy)
benchmark((1, 1, 28, 28), 6, 5)        # LeNet-like
benchmark((1, 3, 32, 32), 32, 3, padding=1)  # CIFAR-like

# Medium (NumPy will be slow)
benchmark((1, 64, 56, 56), 64, 3, padding=1)  # ResNet-like

# Large (NumPy will be very slow — reduce n_runs)
benchmark((1, 3, 224, 224), 64, 7, stride=2, padding=3)  # ImageNet first conv
```

### Analysis Questions

Answer these in your submission:

1. **What is the speedup factor?** Typical: PyTorch is 100-1000x faster on CPU.

2. **Why is PyTorch so much faster?** Write at least a paragraph. Key reasons:
   - PyTorch uses im2col: reshapes the convolution into a matrix multiplication, which is
     highly optimized via BLAS (MKL, OpenBLAS).
   - Your implementation uses Python loops, which are ~100x slower than C loops.
   - PyTorch uses SIMD instructions (SSE, AVX) for vectorized computation.
   - PyTorch has been extensively profiled and optimized over years.
   - On GPU: massive parallelism (thousands of cores) via cuDNN.

3. **Could you make your implementation faster without calling PyTorch?** Discuss:
   - im2col + np.dot (convert conv to matmul)
   - Using Cython or Numba to JIT-compile the loops
   - Using numpy vectorization (np.lib.stride_tricks)

---

## Part 5: Filter Visualization

### Task

Load a pretrained CNN and visualize what its filters have learned.

```python
import torchvision.models as models
import matplotlib.pyplot as plt

# Load pretrained model
model = models.resnet18(weights='IMAGENET1K_V1')

# Extract first conv layer weights
# Shape: (64, 3, 7, 7) — 64 filters, each 3-channel, 7x7
first_conv_weights = model.conv1.weight.detach().numpy()

# Visualize the 64 filters as RGB images
fig, axes = plt.subplots(8, 8, figsize=(16, 16))
for i, ax in enumerate(axes.flat):
    if i < 64:
        # Normalize each filter to [0, 1] for display
        w = first_conv_weights[i]
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        # Transpose from (3, 7, 7) to (7, 7, 3) for display
        ax.imshow(w.transpose(1, 2, 0))
    ax.axis('off')
plt.suptitle("ResNet-18 First Layer Filters (64 filters, 7x7x3)")
plt.tight_layout()
plt.savefig('first_layer_filters.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Analysis

1. **What patterns do you see?** You should observe oriented edges at various angles, color
   blobs, and gradient detectors. These are similar to Gabor filters and the features found
   in the primary visual cortex (V1) of biological vision systems.

2. **Visualize deeper layer filters.** Extract weights from `model.layer2[0].conv1` (shape
   will be something like (128, 64, 3, 3)). Can you interpret these 64-channel filters? Why
   or why not? (Answer: deeper filters operate on abstract feature maps, not RGB images, so
   they are not directly interpretable as images.)

3. **Visualize activations.** Feed a specific image through the network and plot the
   intermediate feature maps at layers 1, 2, 3, and 4. Which feature maps activate for
   specific parts of the image?

---

## Deliverables

Submit a Jupyter notebook (or Python script + PDF report) containing:

1. **conv2d_forward** — single-channel convolution with tests passing
2. **conv2d_forward_multichannel** — multi-channel batched convolution with tests passing
3. **conv2d_backward** — backward pass with gradient verification passing
4. **Benchmarks** — timing comparison table at multiple scales
5. **Analysis** — written answers to the three benchmark questions (minimum 200 words total)
6. **Filter visualizations** — first-layer filters and activation maps with written analysis

### Grading Criteria

| Component | Weight | Criteria |
|-----------|--------|----------|
| Forward pass correctness | 25% | All test configurations pass (atol < 1e-5) |
| Backward pass correctness | 30% | All gradient checks pass for stride=1 and stride=2 |
| Benchmarking | 15% | Complete timing table + thoughtful analysis of WHY |
| Filter visualization | 15% | Clear visualizations with written interpretation |
| Code quality | 15% | Clean, commented, well-organized code |

### Common Pitfalls

- **Off-by-one errors in the output size calculation.** Use the formula. Do not guess.
- **Forgetting to pad the input before applying the kernel.** Padding happens BEFORE the
  sliding window.
- **Getting the backward pass dimensions wrong.** Draw the computation graph on paper first.
  Label every tensor's shape. The shapes must be consistent.
- **Confusing convolution with cross-correlation.** PyTorch implements cross-correlation
  (no kernel flip). Your implementation should also use cross-correlation to match.

---

## Stretch Goals

1. **im2col optimization:** Implement the im2col transformation that converts convolution
   into matrix multiplication. Benchmark it against your loop-based implementation and
   against PyTorch.

2. **Grouped convolution:** Extend your implementation to support `groups > 1`. Verify
   against `nn.Conv2d(groups=...)`. Then implement depthwise separable convolution using
   groups=C_in.

3. **Dilated convolution:** Add a `dilation` parameter to your implementation. Verify against
   `nn.Conv2d(dilation=...)`.

4. **Transposed convolution:** Implement the transposed convolution operation. Verify against
   `nn.ConvTranspose2d`.

5. **Gradient checking via finite differences:** Implement numerical gradient checking
   (perturb each weight by epsilon, measure change in output) and verify it matches your
   analytical backward pass. This is the gold standard for verifying gradients.

   ```python
   def numerical_gradient(f, x, epsilon=1e-5):
       """Compute gradient numerically via finite differences."""
       grad = np.zeros_like(x)
       for idx in np.ndindex(x.shape):
           x_plus = x.copy()
           x_plus[idx] += epsilon
           x_minus = x.copy()
           x_minus[idx] -= epsilon
           grad[idx] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
       return grad
   ```
