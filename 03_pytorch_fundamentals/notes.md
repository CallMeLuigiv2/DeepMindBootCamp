# Module 3: PyTorch Fundamentals — Reference Notes

**Your PyTorch Bible. Keep this open while you work.**

---

## Table of Contents

1. [Tensors: The Foundation](#1-tensors-the-foundation)
2. [Autograd: The Engine of Learning](#2-autograd-the-engine-of-learning)
3. [nn.Module: Building Blocks](#3-nnmodule-building-blocks)
4. [Data Pipeline](#4-data-pipeline)
5. [The Training Loop](#5-the-training-loop)
6. [Device Management](#6-device-management)
7. [Debugging and Common Mistakes](#7-debugging-and-common-mistakes)
8. [Performance Tips](#8-performance-tips)

---

## 1. Tensors: The Foundation

### 1.1 What Is a Tensor, Really?

A tensor is not just a "multi-dimensional array." That description is technically accurate but
misses the point. A PyTorch tensor is a **typed, device-aware, strided view over a contiguous
block of memory** that optionally participates in **automatic differentiation**.

Four properties define a tensor:
- **Shape** (`.shape` or `.size()`): the dimensions.
- **Dtype** (`.dtype`): the numerical type of each element.
- **Device** (`.device`): where the data lives (CPU RAM or GPU VRAM).
- **Layout** (`.stride()`): how logical indices map to physical memory offsets.

When you "transpose" a tensor, no data moves. PyTorch just changes the strides.

### 1.2 The Dtype System

```python
import torch

# Common dtypes
torch.float32   # (or torch.float)  — default for most operations
torch.float64   # (or torch.double) — rarely needed, slower on GPU
torch.float16   # (or torch.half)   — mixed precision training
torch.bfloat16  # — better dynamic range than float16, preferred for training
torch.int64     # (or torch.long)   — default for indices and labels
torch.int32     # (or torch.int)    — sometimes used for indices
torch.bool      # — masks

# Creating with specific dtype
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
y = x.to(torch.float16)   # explicit cast
z = x.half()               # shorthand for float16
w = x.double()             # shorthand for float64

# Check dtype
print(x.dtype)  # torch.float32

# GOTCHA: torch.tensor([1, 2, 3]) creates int64, not float32
# Use torch.tensor([1.0, 2.0, 3.0]) or specify dtype explicitly
```

**When dtype matters:**
- Model parameters: usually float32 (or bfloat16 for mixed precision).
- Labels for CrossEntropyLoss: must be int64 (long).
- Boolean masks: torch.bool.
- Mixed precision training: float16 or bfloat16 for forward pass, float32 for master weights.

### 1.3 Tensor Creation Methods

```python
# From Python data
a = torch.tensor([1, 2, 3])           # copies data
b = torch.as_tensor([1, 2, 3])        # shares memory if possible

# From NumPy (shares memory!)
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(np_arr)          # shares memory with numpy array
# Modifying t will modify np_arr and vice versa

# Zeros, ones, random
z = torch.zeros(3, 4)                 # shape (3, 4), float32
o = torch.ones(3, 4)
r = torch.randn(3, 4)                 # standard normal
u = torch.rand(3, 4)                  # uniform [0, 1)
e = torch.empty(3, 4)                 # uninitialized (fast but dangerous)

# Ranges
seq = torch.arange(0, 10, 2)          # [0, 2, 4, 6, 8]
lin = torch.linspace(0, 1, 5)         # [0.0, 0.25, 0.5, 0.75, 1.0]

# Like-tensors (same shape/dtype/device as another tensor)
x = torch.randn(3, 4, device='cuda')
y = torch.zeros_like(x)               # same shape, dtype, device

# Identity matrix
eye = torch.eye(4)                    # 4x4 identity

# IMPORTANT: torch.tensor() always copies data.
# torch.as_tensor() and torch.from_numpy() share memory when possible.
```

### 1.4 Indexing and Slicing

```python
x = torch.arange(24).reshape(4, 6)

# Basic indexing (returns views, not copies)
x[0]          # first row
x[:, 0]       # first column
x[1:3, 2:5]   # rows 1-2, columns 2-4

# Advanced (integer array) indexing (returns copies)
idx = torch.tensor([0, 2, 3])
x[idx]         # rows 0, 2, 3

# Boolean masking
mask = x > 10
x[mask]        # all elements > 10 (flattened)

# torch.where — element-wise conditional
result = torch.where(x > 10, x, torch.zeros_like(x))

# Fancy indexing for gather/scatter patterns
# Select element [i, j] for each pair
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([1, 3, 5])
x[rows, cols]  # elements at (0,1), (1,3), (2,5)

# GOTCHA: Basic indexing returns a view (shares memory).
# Advanced indexing returns a copy (new memory).
# This matters when you assign to a slice.
```

### 1.5 View vs Reshape vs Contiguous

This is one of the most misunderstood topics in PyTorch. Read this carefully.

**Memory Layout and Strides:**

```python
x = torch.arange(12).reshape(3, 4)
# x is stored in memory as: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# x.stride() = (4, 1)
# To access x[i, j], PyTorch computes: offset = i * 4 + j * 1

print(x.stride())         # (4, 1)
print(x.is_contiguous())  # True

# Transpose changes strides, NOT data
y = x.t()  # or x.T
print(y.stride())         # (1, 4)  — strides are swapped
print(y.is_contiguous())  # False — row elements are not adjacent in memory

# Visual: x in memory is [0,1,2,3,4,5,6,7,8,9,10,11]
# x sees: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]  (stride 4,1)
# y sees: [[0,4,8], [1,5,9], [2,6,10], [3,7,11]] (stride 1,4)
# Same memory, different views!
```

**The Three Operations:**

```python
# view() — REQUIRES contiguous memory. Returns a view (no copy).
a = torch.arange(12)
b = a.view(3, 4)         # works: a is contiguous
c = b.t()                # c is NOT contiguous
# c.view(12)             # ERROR: cannot view non-contiguous tensor

# contiguous() — copies data into contiguous memory IF needed.
d = c.contiguous()       # now d is contiguous (new memory)
e = d.view(12)           # works

# reshape() — returns a view if possible, copies if necessary.
f = c.reshape(12)        # works: reshapes may copy internally
# You do NOT know if f shares memory with c or not!
```

**When to use which:**

| Operation | Guarantees | When to use |
|-----------|-----------|-------------|
| `view()` | No copy, always a view | When you know data is contiguous and you want zero-cost reshaping |
| `reshape()` | Works always, may copy | When you do not care about memory sharing |
| `contiguous()` | Makes data contiguous | Before `view()` on non-contiguous tensors |

**DeepMind-level insight:** In performance-critical code, prefer `view()` because it guarantees
no copy. If `view()` fails, you have a non-contiguous tensor — investigate why, rather than
blindly calling `reshape()`. Understanding your memory layout prevents subtle bugs.

### 1.6 Broadcasting Rules

Broadcasting allows operations between tensors of different shapes. The rules are:

1. If tensors have different numbers of dimensions, pad the shorter one with 1s on the left.
2. Dimensions are compatible if they are equal or one of them is 1.
3. The output shape takes the maximum along each dimension.

```python
# Example 1: Adding a bias to each row
x = torch.randn(3, 4)    # shape (3, 4)
b = torch.randn(4)       # shape (4,) -> broadcast as (1, 4)
y = x + b                # shape (3, 4) — b is added to every row

# Example 2: Outer product via broadcasting
a = torch.tensor([1, 2, 3]).reshape(3, 1)    # shape (3, 1)
b = torch.tensor([10, 20, 30]).reshape(1, 3) # shape (1, 3)
c = a * b                                     # shape (3, 3) — outer product

# Example 3: Batch operations
batch = torch.randn(32, 3, 4)  # shape (32, 3, 4)
scale = torch.randn(3, 1)      # shape (3, 1) -> broadcast as (1, 3, 1)
result = batch * scale          # shape (32, 3, 4)

# GOTCHA: Unintended broadcasting
x = torch.randn(3, 1)  # shape (3, 1)
y = torch.randn(1, 4)  # shape (1, 4)
z = x + y               # shape (3, 4) — is this what you intended?

# EXERCISE: Predict the output shape before running
# a: (5, 3, 1) + b: (1, 4) -> ?
# Answer: (5, 3, 1) + (1, 1, 4) -> (5, 3, 4)
```

**Common mistake:** Broadcasting silently succeeds when you have a shape bug. If you expected
`(3, 4) + (3, 4)` but accidentally got `(3, 1) + (1, 4)`, PyTorch will not warn you. Always
verify shapes with assertions in your code.

### 1.7 In-Place Operations

```python
x = torch.randn(3, 4, requires_grad=True)

# In-place operations end with underscore
# x.add_(1)    # THIS WILL BREAK AUTOGRAD
# x.mul_(2)    # THIS WILL BREAK AUTOGRAD

# Why? Autograd needs the original values to compute gradients.
# In-place ops modify the data that autograd saved for backward.
# PyTorch will raise a RuntimeError if it detects this.

# Safe in-place ops (when autograd is not involved):
with torch.no_grad():
    x.add_(1)     # fine: no gradient tracking

# Or on tensors that don't require grad:
y = torch.randn(3, 4)
y.add_(1)         # fine: y doesn't participate in autograd

# RULE: Never use in-place operations on tensors that require gradients
# during the forward pass. The memory savings are not worth the bugs.
```

---

## 2. Autograd: The Engine of Learning

### 2.1 The Computational Graph

Every operation on tensors with `requires_grad=True` builds a directed acyclic graph (DAG).
This graph records what operations produced each tensor, enabling automatic gradient computation.

```python
# A concrete example
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)   # input, no grad needed
b = torch.tensor(1.0, requires_grad=True)

# Forward pass: y = w * x + b
y = w * x + b  # y = 2*3 + 1 = 7

# The computational graph looks like this:
#
#    w (leaf, requires_grad=True)
#    |
#    v
#   [Mul] <--- x (leaf, requires_grad=False)
#    |
#    v
#   mul_result (non-leaf, grad_fn=MulBackward)
#    |
#    v
#   [Add] <--- b (leaf, requires_grad=True)
#    |
#    v
#    y (non-leaf, grad_fn=AddBackward)
#    |
#    v
#   [MSE or other loss]
#    |
#    v
#   loss (non-leaf, grad_fn=...)

# Backward pass: compute gradients
loss = (y - 5) ** 2   # loss = (7 - 5)^2 = 4
loss.backward()

# Gradients:
# dloss/dy = 2 * (y - 5) = 2 * 2 = 4
# dy/dw = x = 3
# dy/db = 1
# dloss/dw = dloss/dy * dy/dw = 4 * 3 = 12
# dloss/db = dloss/dy * dy/db = 4 * 1 = 4

print(w.grad)  # tensor(12.)
print(b.grad)  # tensor(4.)
```

**ASCII art of the computational graph and backward pass:**

```
FORWARD PASS (left to right):              BACKWARD PASS (right to left):

 w=2.0 ---\                                w.grad=12 <--\
            [*] --> (w*x=6) --\                          [*] <-- d(w*x)/dw=x=3
 x=3.0 ---/                   \                         /
                                [+] --> y=7 --> [(y-5)^2] --> loss=4
 b=1.0 -----------------------/            \   /
                                             [+] <-- d(y)/db=1
                                b.grad=4 <--/

 Gradient flow:
 dloss/dloss = 1
 dloss/d(y-5)^2 = 1
 dloss/dy = 2*(y-5) = 4
 dloss/dw = dloss/dy * dy/dw = 4 * 3 = 12
 dloss/db = dloss/dy * dy/db = 4 * 1 = 4
```

### 2.2 Key Autograd Concepts

```python
# requires_grad: tells PyTorch to track operations
w = torch.randn(3, 4, requires_grad=True)
print(w.requires_grad)  # True

# .grad: stores the gradient after backward()
# Only populated for LEAF tensors
loss.backward()
print(w.grad)  # tensor of shape (3, 4)

# .grad_fn: the function that created this tensor
y = w * 2
print(y.grad_fn)  # <MulBackward0 object>

# .is_leaf: True for tensors created directly (not by operations)
print(w.is_leaf)  # True
print(y.is_leaf)  # False

# Non-leaf tensors do NOT retain gradients by default
# Use .retain_grad() if you need them (rare)
y.retain_grad()
```

### 2.3 Gradient Accumulation and Zeroing

```python
w = torch.tensor(2.0, requires_grad=True)

# First backward
y1 = w * 3
y1.backward()
print(w.grad)  # tensor(3.)

# Second backward WITHOUT zeroing
y2 = w * 5
y2.backward()
print(w.grad)  # tensor(8.)  <-- 3 + 5, gradients ACCUMULATED!

# This is BY DESIGN. Useful for:
# - Gradient accumulation across mini-batches (simulating larger batch sizes)
# - But it means you MUST zero gradients before each optimization step

# The correct pattern:
optimizer.zero_grad()     # or model.zero_grad()
loss = compute_loss(...)
loss.backward()
optimizer.step()

# Alternative: set gradients to None (slightly more efficient)
optimizer.zero_grad(set_to_none=True)
# This sets .grad to None instead of zero tensor, saving memory
```

### 2.4 detach() and torch.no_grad()

```python
# torch.no_grad(): context manager that disables gradient tracking
# Use for: inference, evaluation, anything where you don't need gradients
with torch.no_grad():
    predictions = model(test_data)
    # No computational graph is built
    # Operations are faster and use less memory

# detach(): creates a tensor that shares data but doesn't track gradients
# Use for: stopping gradient flow at a specific point
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.detach()  # z shares data with y but has requires_grad=False
# Gradients will NOT flow from z back to x

# Common use case: using a model's output as input to another model
# without backpropagating through the first model
features = encoder(x).detach()  # stop gradients here
output = decoder(features)       # only train decoder

# IMPORTANT DIFFERENCE:
# torch.no_grad() — affects ALL operations in the block
# detach() — affects ONE specific tensor
```

### 2.5 Leaf vs Non-Leaf Tensors

```python
# Leaf tensors: created directly by the user, not by operations
a = torch.randn(3, requires_grad=True)   # leaf
b = torch.randn(3, requires_grad=False)  # leaf (but won't get gradients)

# Non-leaf tensors: created by operations on other tensors
c = a + b                                 # non-leaf (grad_fn=AddBackward)

# KEY RULE: Only leaf tensors retain .grad after backward()
# Non-leaf tensor gradients are computed but immediately discarded

# Why? Memory. In a deep network, intermediate activations vastly outnumber
# parameters. Keeping all intermediate gradients would be prohibitive.

# If you really need a non-leaf gradient (for debugging/visualization):
c.retain_grad()
loss = c.sum()
loss.backward()
print(c.grad)  # now available

# GOTCHA: Parameters in nn.Module are always leaf tensors
# (created by nn.Parameter, which sets requires_grad=True)
```

### 2.6 Custom Autograd Functions

```python
import torch
from torch.autograd import Function

class Swish(Function):
    @staticmethod
    def forward(ctx, x):
        # ctx is a context object for saving information for backward
        sigmoid_x = torch.sigmoid(x)
        ctx.save_for_backward(x, sigmoid_x)
        return x * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is the gradient flowing from downstream
        x, sigmoid_x = ctx.saved_tensors
        # d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        grad_input = grad_output * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
        return grad_input

# Usage
swish = Swish.apply
x = torch.randn(5, requires_grad=True)
y = swish(x)
y.sum().backward()
print(x.grad)

# Verify with gradcheck (uses numerical differentiation)
from torch.autograd import gradcheck
x = torch.randn(5, dtype=torch.float64, requires_grad=True)  # float64 for precision
assert gradcheck(swish, (x,), eps=1e-6)
```

**DeepMind-level insight:** Custom autograd functions are how you implement operations that
PyTorch does not natively support or where you need a numerically stable backward pass that
differs from the naive chain rule application. Research code uses these regularly.

### 2.7 Gradient Checkpointing (Concept)

```python
# Problem: Deep models have many intermediate activations stored for backward.
# A model with 100 layers stores ~100 activation tensors in memory.

# Solution: Gradient checkpointing trades compute for memory.
# Instead of storing all activations, recompute them during backward.

from torch.utils.checkpoint import checkpoint

class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(256, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)

    def forward(self, x):
        # Checkpoint layer2: its activations won't be stored
        # They'll be recomputed during backward
        x = self.layer1(x)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = self.layer3(x)
        return x

# Trade-off: ~33% more compute, ~33% less memory
# Essential for training very large models (e.g., transformers with many layers)
```

---

## 3. nn.Module: Building Blocks

### 3.1 The Module Abstraction

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # ALWAYS call super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate
model = SimpleNet(784, 128, 10)

# ALWAYS call model(x), never model.forward(x)
# __call__ runs hooks, applies decorators, etc.
x = torch.randn(32, 784)
output = model(x)  # shape: (32, 10)
```

### 3.2 Parameters and Named Parameters

```python
# List all parameters
for param in model.parameters():
    print(param.shape, param.requires_grad)

# Named parameters (useful for debugging and selective optimization)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
# Output:
# fc1.weight: torch.Size([128, 784])
# fc1.bias: torch.Size([128])
# fc2.weight: torch.Size([10, 128])
# fc2.bias: torch.Size([10])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params}, Trainable: {trainable_params}")

# Children (immediate sub-modules)
for name, child in model.named_children():
    print(f"{name}: {child}")

# Modules (all modules recursively, including self)
for name, module in model.named_modules():
    print(f"{name}: {module.__class__.__name__}")
```

### 3.3 Common Layers

```python
# Linear (fully connected)
linear = nn.Linear(in_features=784, out_features=256, bias=True)
# Weight shape: (out_features, in_features) = (256, 784)
# NOTE: weight is (out, in), not (in, out)!

# Conv2d
conv = nn.Conv2d(
    in_channels=3,      # e.g., RGB
    out_channels=16,    # number of filters
    kernel_size=3,      # 3x3 kernel
    stride=1,
    padding=1           # 'same' padding for stride=1
)
# Output size formula: (W - K + 2P) / S + 1
# With W=32, K=3, P=1, S=1: (32 - 3 + 2) / 1 + 1 = 32 (same size)

# BatchNorm2d
bn = nn.BatchNorm2d(num_features=16)  # must match number of channels
# Has DIFFERENT behavior in train vs eval mode!
# Train: uses batch statistics, updates running mean/var
# Eval: uses running mean/var (accumulated during training)

# Dropout
dropout = nn.Dropout(p=0.5)
# Train: randomly zeros elements with probability p
# Eval: does nothing (identity function)
# CRITICAL: call model.eval() during inference!

# MaxPool2d
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# Halves spatial dimensions: (B, C, H, W) -> (B, C, H/2, W/2)

# ReLU / GELU / SiLU
relu = nn.ReLU(inplace=False)   # inplace=True saves memory but can break autograd
gelu = nn.GELU()                 # smoother, used in transformers
silu = nn.SiLU()                 # also called Swish: x * sigmoid(x)

# LayerNorm
ln = nn.LayerNorm(normalized_shape=256)
# Normalizes over the last N dimensions. Common in transformers.
```

### 3.4 Sequential vs Custom Modules

```python
# Sequential: for simple stacks of layers
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# When to use Sequential:
# - The forward pass is a simple chain: layer1 -> layer2 -> ... -> layerN
# - No skip connections, no branching, no custom logic

# When to use a custom Module:
# - Skip connections (ResNet)
# - Multiple inputs or outputs
# - Conditional logic in forward pass
# - Anything that isn't a simple chain

# ModuleList: for dynamic lists of modules
class DynamicNet(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        # DO NOT use a plain Python list! Parameters won't be registered.

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

# ModuleDict: for named dynamic modules
class FlexibleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleDict({
            'encoder': nn.Linear(784, 256),
            'decoder': nn.Linear(256, 784),
        })
```

### 3.5 Buffers vs Parameters

```python
class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        # Buffer: part of state but NOT a learnable parameter
        # Saved in state_dict, moved with .to(device), but not in parameters()
        self.register_buffer('running_mean', torch.zeros(in_features))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            self.running_mean = (self.running_mean * self.count + x.mean(0)) / (self.count + 1)
            self.count += 1
        return self.linear(x - self.running_mean)

# Buffers vs Parameters:
# Parameters: requires_grad=True, returned by .parameters(), optimized
# Buffers: requires_grad=False, NOT in .parameters(), but in state_dict
# Both are moved to device with .to() and saved/loaded with state_dict
```

### 3.6 State Dict and Model Saving

```python
# CORRECT way to save a model
torch.save(model.state_dict(), 'model_weights.pth')

# CORRECT way to load a model
model = SimpleNet(784, 128, 10)  # create the architecture first
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()  # set to evaluation mode

# Inspect state dict
for key, tensor in model.state_dict().items():
    print(f"{key}: {tensor.shape}")

# Saving everything needed to resume training
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'best_val_loss': best_val_loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Loading a checkpoint
checkpoint = torch.load('checkpoint.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# WRONG way to save (do NOT do this):
# torch.save(model, 'model.pth')  # saves the entire object, fragile
# This breaks if you rename the module file or change the class definition

# Handling mismatched keys
model.load_state_dict(state_dict, strict=False)
# strict=False ignores missing and unexpected keys
# Useful for transfer learning when you've changed some layers
```

### 3.7 Train vs Eval Mode

```python
# train() and eval() affect modules with mode-dependent behavior:
# - Dropout: active in train, disabled in eval
# - BatchNorm: uses batch stats in train, running stats in eval

model.train()  # set to training mode
# ... training loop ...

model.eval()   # set to evaluation mode
with torch.no_grad():  # also disable gradient computation
    predictions = model(test_data)

# COMMON BUG: Forgetting model.eval() during validation
# This causes:
# 1. Dropout to randomly zero activations (noisy predictions)
# 2. BatchNorm to use batch stats instead of population stats
# 3. Results that are different every time you run evaluation
# This is one of the top 5 most common bugs in deep learning code.
```

---

## 4. Data Pipeline

### 4.1 The Dataset Class

```python
from torch.utils.data import Dataset, DataLoader

# Map-style Dataset: random access via __getitem__
class RegressionDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.x = torch.randn(num_samples, 10)
        self.y = self.x @ torch.randn(10) + 0.1 * torch.randn(num_samples)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Usage
dataset = RegressionDataset(1000)
x, y = dataset[0]  # single sample

# Image Dataset with transforms
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

### 4.2 DataLoader — The Full API

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=64,          # samples per batch
    shuffle=True,           # shuffle at every epoch (MUST for training)
    num_workers=4,          # parallel data loading processes
    pin_memory=True,        # pre-allocate page-locked memory for GPU transfer
    drop_last=True,         # drop the last incomplete batch
    prefetch_factor=2,      # batches to prefetch per worker (default 2)
    persistent_workers=True # keep workers alive between epochs (saves startup time)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,         # can use larger batch for validation (no gradients)
    shuffle=False,          # NEVER shuffle validation data
    num_workers=4,
    pin_memory=True,
    drop_last=False,        # evaluate ALL samples
)

# Iterating
for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(device)       # move to GPU
    targets = targets.to(device)
    # ... training step ...
```

### 4.3 DataLoader Internals: What Actually Happens

**num_workers explained:**

```
num_workers=0 (default):
  Main process does EVERYTHING:
  [Load sample] -> [Collate] -> [To GPU] -> [Forward] -> [Backward]
  Data loading blocks the training loop.

num_workers=4:
  Worker 0: [Load batch N+1] ------>
  Worker 1: [Load batch N+2] ------>
  Worker 2: [Load batch N+3] ------>
  Worker 3: [Load batch N+4] ------>
  Main:     [Train on batch N] -> [Take next ready batch] -> [Train]

  Workers load data in parallel while the main process trains.
  The GPU is never idle waiting for data (if workers are fast enough).
```

**pin_memory explained:**

```
Without pin_memory:
  [CPU RAM] -> [Copy to pinned memory] -> [DMA transfer to GPU]

With pin_memory=True:
  [Pinned CPU RAM] -> [DMA transfer to GPU]

Pinned (page-locked) memory cannot be swapped to disk by the OS.
This makes CPU -> GPU transfers faster because the DMA controller
can access the memory directly without involving the CPU.
Always use pin_memory=True when training on GPU.
```

**collate_fn explained:**

```python
# Default collation: stack samples into a batch tensor
# [tensor(3,), tensor(3,), tensor(3,)] -> tensor(3, 3)

# Custom collation: for variable-length data
def custom_collate_fn(batch):
    """Pad variable-length sequences to the maximum length in the batch."""
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)

    # Pad sequences
    padded = torch.zeros(len(sequences), max_len)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return padded, labels, lengths

loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)
```

### 4.4 Transforms and Data Augmentation

```python
from torchvision import transforms

# Training transforms: augmentation + normalization
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),        # random crop and resize
    transforms.RandomHorizontalFlip(p=0.5),   # 50% chance of horizontal flip
    transforms.ColorJitter(                    # random color changes
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),
    transforms.RandomRotation(10),             # random rotation up to 10 degrees
    transforms.ToTensor(),                     # PIL Image -> tensor, scales to [0, 1]
    transforms.Normalize(                      # channel-wise normalization
        mean=[0.485, 0.456, 0.406],           # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    ),
])

# Validation transforms: NO augmentation, only resize + normalize
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# CRITICAL: Training and validation MUST use different transforms.
# Augmentation is for training only. Validation must be deterministic.
```

### 4.5 Standard Datasets

```python
from torchvision import datasets

# CIFAR-10
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=val_transform
)

# Fashion-MNIST
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
)

# Create train/val split from training data
from torch.utils.data import random_split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # reproducible split
)
```

### 4.6 Windows-Specific DataLoader Notes

```python
# On Windows, num_workers > 0 requires the __main__ guard
# because Windows uses 'spawn' (not 'fork') for multiprocessing

if __name__ == '__main__':
    train_loader = DataLoader(dataset, batch_size=32, num_workers=4)
    for batch in train_loader:
        # ... training ...
        pass

# If you get "RuntimeError: An attempt has been made to start a new process..."
# you forgot the __main__ guard.

# If workers crash silently, try reducing num_workers.
# Start with num_workers=2 on Windows.
```

---

## 5. The Training Loop

### 5.1 The Canonical Training Loop

This is the most important code in this entire module. Memorize it.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ============================================================
# SETUP
# ============================================================

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = SimpleNet(input_dim=784, hidden_dim=256, output_dim=10).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()  # for classification (expects raw logits)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ============================================================
# TRAINING LOOP
# ============================================================

num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):

    # ---- TRAINING PHASE ----
    model.train()                          # set training mode (Dropout, BatchNorm)
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.to(device)         # move data to GPU
        targets = targets.to(device)

        optimizer.zero_grad()              # STEP 1: clear old gradients

        outputs = model(inputs)            # STEP 2: forward pass

        loss = criterion(outputs, targets) # STEP 3: compute loss

        loss.backward()                    # STEP 4: compute gradients (backward pass)

        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()                   # STEP 5: update parameters

        # Track metrics
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss /= total
    train_acc = 100.0 * correct / total

    # ---- VALIDATION PHASE ----
    model.eval()                           # set evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():                 # disable gradient computation
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= total
    val_acc = 100.0 * correct / total

    # ---- SCHEDULER STEP ----
    scheduler.step()                       # update learning rate

    # ---- LOGGING ----
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # ---- CHECKPOINTING ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, 'best_model.pth')
        print(f"  Saved best model (val_loss: {val_loss:.4f})")
```

### 5.2 Why the Order Matters

```
zero_grad() -> forward -> loss -> backward() -> step()

Q: Why zero_grad BEFORE forward?
A: It doesn't strictly need to be before forward. It needs to be before backward().
   But placing it at the start of the loop makes the intent clear: "start fresh."

Q: Why backward() before step()?
A: backward() computes the gradients (.grad attributes). step() reads those
   gradients to update parameters. Without backward(), gradients are stale or zero.

Q: What if I forget zero_grad()?
A: Gradients accumulate across batches. Your effective gradient becomes an
   ever-growing sum, and training diverges.

Q: What if I call step() before backward()?
A: Parameters are updated using whatever .grad currently holds (likely zeros
   on the first step, or stale gradients from a previous batch). Training fails.
```

### 5.3 Loss Functions

```python
# REGRESSION
mse_loss = nn.MSELoss()          # Mean Squared Error
# Input: (N, *), Target: (N, *)
# Use for: continuous value prediction (regression)

l1_loss = nn.L1Loss()            # Mean Absolute Error
# More robust to outliers than MSE

smooth_l1 = nn.SmoothL1Loss()    # Huber loss
# L1 for large errors, L2 for small errors. Best of both worlds.

# CLASSIFICATION (multi-class, single label)
ce_loss = nn.CrossEntropyLoss()
# Input: (N, C) raw logits (NOT softmax!)
# Target: (N,) class indices (long tensor)
# Combines LogSoftmax + NLLLoss internally
# COMMON BUG: applying softmax before CrossEntropyLoss

# CLASSIFICATION (binary or multi-label)
bce_loss = nn.BCEWithLogitsLoss()
# Input: (N, *) raw logits (NOT sigmoid!)
# Target: (N, *) float tensor with values in [0, 1]
# Combines Sigmoid + BCELoss internally
# Use for: binary classification, multi-label classification

# GOTCHA: Do NOT use nn.BCELoss() directly. It requires sigmoid output
# and is numerically unstable. Always use BCEWithLogitsLoss.

# Example: choosing the right loss
# Task: classify images into 10 classes
criterion = nn.CrossEntropyLoss()
logits = model(images)               # shape: (B, 10), raw scores
loss = criterion(logits, labels)      # labels shape: (B,), dtype=long

# Task: multi-label classification (image can have multiple tags)
criterion = nn.BCEWithLogitsLoss()
logits = model(images)               # shape: (B, num_tags), raw scores
loss = criterion(logits, tag_labels)  # tag_labels shape: (B, num_tags), float

# Task: regression (predict house prices)
criterion = nn.MSELoss()
predictions = model(features)        # shape: (B, 1)
loss = criterion(predictions, prices) # prices shape: (B, 1)
```

### 5.4 Optimizers

```python
# SGD: the fundamental optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,                # learning rate
    momentum=0.9,           # momentum (almost always use 0.9)
    weight_decay=1e-4       # L2 regularization
)
# Good: often generalizes better. Bad: requires careful LR tuning.

# Adam: adaptive learning rates per parameter
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,                # learning rate (1e-3 is a good default)
    betas=(0.9, 0.999),     # decay rates for first and second moment
    eps=1e-8,               # numerical stability
    weight_decay=0          # NOTE: L2 regularization in Adam is not ideal
)
# Good: fast convergence, less sensitive to LR. Bad: can overfit.

# AdamW: Adam with DECOUPLED weight decay (the correct way)
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2       # decoupled weight decay (not L2 regularization)
)
# This is the default choice for most modern deep learning.
# Use AdamW unless you have a specific reason not to.

# Per-parameter options (different LR for different parts of the model)
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-4},    # pretrained: low LR
    {'params': model.classifier.parameters(), 'lr': 1e-3},  # new: higher LR
], weight_decay=1e-2)
```

### 5.5 Learning Rate Schedulers

```python
# StepLR: reduce LR by a factor every N epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# LR * 0.1 every 10 epochs. Simple but coarse.

# CosineAnnealingLR: smooth cosine decay to zero
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# Smoothly decays LR from initial value to 0 over T_max epochs.
# Very popular. Good default choice.

# OneCycleLR: ramp up then decay (super-convergence)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader)
)
# IMPORTANT: Call scheduler.step() after EVERY BATCH, not every epoch!
# This is different from other schedulers.

# ReduceLROnPlateau: reduce LR when metric plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # reduce when metric stops decreasing
    factor=0.1,          # multiply LR by this factor
    patience=5,          # wait this many epochs before reducing
    verbose=True
)
# IMPORTANT: Call scheduler.step(val_loss), not scheduler.step()
# This scheduler watches a metric, so you must pass it.

# When to call scheduler.step():
# StepLR, CosineAnnealing: once per epoch, after validation
# OneCycleLR: once per batch, after optimizer.step()
# ReduceLROnPlateau: once per epoch, passing the monitored metric
```

### 5.6 Early Stopping

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=7, min_delta=0.0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# Usage
early_stopping = EarlyStopping(patience=10)
for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    early_stopping(val_loss, model)
    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model after training
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
```

### 5.7 Gradient Clipping

```python
# Clip by norm (most common): scale gradients so their total norm <= max_norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# This scales ALL gradients proportionally if the total norm exceeds max_norm.
# Preserves gradient direction.

# Clip by value: clamp each gradient element to [-clip_value, clip_value]
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
# This can change gradient direction. Less commonly used.

# When to use gradient clipping:
# - RNNs/LSTMs (prone to exploding gradients)
# - Transformers (large models can have gradient spikes)
# - Any time you see NaN loss or loss spikes during training

# Where in the training loop:
loss.backward()                                                # compute gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip AFTER backward
optimizer.step()                                               # update AFTER clipping
```

### 5.8 Gradient Accumulation

```python
# Simulate a larger batch size when GPU memory is limited.
# Instead of batch_size=64 (which may not fit), use batch_size=16
# and accumulate gradients over 4 steps.

accumulation_steps = 4
optimizer.zero_grad()

for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps     # scale loss
    loss.backward()                       # accumulate gradients

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()                  # update parameters
        optimizer.zero_grad()             # reset gradients

# Effective batch size = batch_size * accumulation_steps = 16 * 4 = 64
```

---

## 6. Device Management

### 6.1 Device-Agnostic Code

```python
# The standard pattern
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device
model = SimpleNet(784, 256, 10).to(device)

# Move data to device (in the training loop)
inputs = inputs.to(device)
targets = targets.to(device)

# For Apple Silicon Macs
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# Check GPU info
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.3f} GB")
```

### 6.2 Common CUDA Errors

```python
# ERROR: "RuntimeError: Expected all tensors to be on the same device"
# CAUSE: model is on GPU but data is on CPU (or vice versa)
# FIX: make sure both model and data are on the same device
model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)

# ERROR: "RuntimeError: CUDA out of memory"
# CAUSE: batch size too large, model too large, or memory leak
# FIX:
# 1. Reduce batch size
# 2. Use gradient accumulation
# 3. Use mixed precision training (torch.cuda.amp)
# 4. Check for memory leaks: are you storing tensors in a list that grows?
#    Common mistake: storing loss tensors instead of loss.item()
losses = []
# WRONG: losses.append(loss)           # keeps entire computation graph!
# RIGHT: losses.append(loss.item())    # stores just the number

# ERROR: "RuntimeError: CUDA error: device-side assert triggered"
# CAUSE: often an out-of-bounds index in an embedding or CrossEntropyLoss
# FIX: check that your labels are in [0, num_classes - 1]
# Set CUDA_LAUNCH_BLOCKING=1 to get a better error message:
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Clearing GPU cache
torch.cuda.empty_cache()  # releases unused cached memory
# This does NOT free memory used by tensors. It only releases
# cached memory back to CUDA. Useful before large allocations.
```

### 6.3 Non-Blocking Transfers

```python
# When using pin_memory=True in DataLoader, use non_blocking=True for transfers
inputs = inputs.to(device, non_blocking=True)
targets = targets.to(device, non_blocking=True)
# This allows the CPU-to-GPU transfer to happen asynchronously
# while the CPU continues with other work.
# Only works with pinned memory (page-locked memory).
```

---

## 7. Debugging and Common Mistakes

### 7.1 The Top 10 PyTorch Bugs

```
1. Forgetting model.eval() during validation/inference
   Symptom: noisy, inconsistent predictions
   Fix: model.eval() + torch.no_grad()

2. Applying softmax before CrossEntropyLoss
   Symptom: model trains but accuracy is lower than expected
   Fix: pass raw logits to CrossEntropyLoss

3. Forgetting to zero gradients
   Symptom: loss decreases then diverges or oscillates
   Fix: optimizer.zero_grad() at the start of each step

4. Data and model on different devices
   Symptom: RuntimeError about tensor devices
   Fix: .to(device) for both model and all data tensors

5. Storing loss tensors instead of loss.item()
   Symptom: GPU memory grows every iteration until OOM
   Fix: use loss.item() when logging, not the tensor itself

6. Wrong label dtype for CrossEntropyLoss
   Symptom: RuntimeError about expected Long tensor
   Fix: labels must be torch.long (int64)

7. Not shuffling training data
   Symptom: model trains slowly or converges to poor solution
   Fix: shuffle=True in training DataLoader

8. Forgetting to call scheduler.step()
   Symptom: learning rate never changes
   Fix: call scheduler.step() at the right time (per-epoch or per-batch)

9. Using the wrong dimension in loss/softmax
   Symptom: loss is computed incorrectly, model doesn't learn
   Fix: check dim argument (usually dim=1 for batch of vectors)

10. Not setting random seeds for reproducibility
    Symptom: different results every run
    Fix: set seeds for torch, numpy, random, and CUDA
```

### 7.2 Shape Debugging

```python
# Add shape assertions liberally during development
def forward(self, x):
    assert x.shape[1] == 3, f"Expected 3 channels, got {x.shape[1]}"

    x = self.conv1(x)
    assert x.shape == (x.shape[0], 16, 32, 32), f"Unexpected shape: {x.shape}"

    # ... more layers ...
    return x

# Print shapes at every layer (during debugging)
def forward(self, x):
    print(f"Input: {x.shape}")
    x = self.conv1(x)
    print(f"After conv1: {x.shape}")
    x = self.pool(x)
    print(f"After pool: {x.shape}")
    x = x.view(x.size(0), -1)
    print(f"After flatten: {x.shape}")
    return x

# Use torchsummary or torchinfo for model architecture overview
# pip install torchinfo
from torchinfo import summary
summary(model, input_size=(1, 3, 32, 32))
```

### 7.3 NaN Debugging

```python
# If loss becomes NaN, check these in order:

# 1. Learning rate too high
# Try reducing by 10x

# 2. Log of zero or negative number
# Check if you're computing log(softmax(x)) instead of log_softmax(x)

# 3. Division by zero
# Check any normalization layers, check loss function inputs

# 4. Exploding gradients
# Add gradient clipping: clip_grad_norm_(model.parameters(), max_norm=1.0)

# 5. NaN in input data
# Check your data pipeline:
for batch in train_loader:
    inputs, targets = batch
    assert not torch.isnan(inputs).any(), "NaN in inputs!"
    assert not torch.isinf(inputs).any(), "Inf in inputs!"
    break

# Detect NaN during training
torch.autograd.set_detect_anomaly(True)  # slow but catches NaN source
# This will give you a stack trace pointing to where the NaN was produced.
# Disable in production (significant overhead).
```

---

## 8. Performance Tips

### 8.1 General Optimization

```python
# 1. Use torch.backends.cudnn.benchmark for fixed input sizes
torch.backends.cudnn.benchmark = True
# Lets cuDNN auto-tune convolution algorithms for your specific input sizes.
# Set to True when input sizes don't change between batches.
# Set to False when input sizes vary (e.g., NLP with variable sequence lengths).

# 2. Use non_blocking transfers with pinned memory
inputs = inputs.to(device, non_blocking=True)

# 3. Use torch.no_grad() for inference
# This disables gradient tracking, saving memory and compute.

# 4. Use AMP (Automatic Mixed Precision) for faster training
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')
for inputs, targets in train_loader:
    optimizer.zero_grad()
    with autocast('cuda'):  # use float16 where safe
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()  # scale loss to prevent underflow
    scaler.step(optimizer)
    scaler.update()

# 5. Profile your code to find bottlenecks
# Is the bottleneck data loading or training?
import time
for batch in train_loader:
    t0 = time.time()
    inputs = inputs.to(device)
    # ... forward + backward ...
    torch.cuda.synchronize()  # wait for GPU to finish
    t1 = time.time()
    print(f"Step time: {t1-t0:.3f}s")
```

### 8.2 Memory Optimization

```python
# 1. Use loss.item() not loss when logging
total_loss += loss.item()  # Python float, no graph
# NOT: total_loss += loss   # keeps entire graph in memory!

# 2. Delete intermediate tensors
del intermediate_output
torch.cuda.empty_cache()

# 3. Use gradient checkpointing for deep models
from torch.utils.checkpoint import checkpoint_sequential
# Re-computes activations during backward instead of storing them.

# 4. Reduce batch size and use gradient accumulation
# See Section 5.8

# 5. Use in-place operations ONLY outside of autograd
# During data preprocessing (before the model), in-place ops are fine.
```

### 8.3 Reproducibility

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # For fully deterministic behavior (may reduce performance):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)

# DataLoader with reproducible shuffling
g = torch.Generator()
g.manual_seed(42)
train_loader = DataLoader(
    dataset, batch_size=32, shuffle=True,
    worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
    generator=g
)
```

### 8.4 TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_01')

for epoch in range(num_epochs):
    # ... training ...

    # Log scalars
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

    # Log model weights histogram
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    # Log images (e.g., sample predictions)
    if epoch % 10 == 0:
        writer.add_images('Predictions', sample_images, epoch)

writer.close()

# Launch TensorBoard:
# tensorboard --logdir=runs
```

---

## Quick Reference Card

```
TENSOR CREATION        | torch.zeros, ones, randn, rand, arange, linspace, from_numpy
TENSOR INFO            | .shape, .dtype, .device, .stride(), .is_contiguous()
RESHAPING              | .view() (no copy), .reshape() (may copy), .contiguous()
INDEXING               | Basic: view; Advanced (int array): copy; Boolean: copy
AUTOGRAD               | requires_grad, backward(), .grad, detach(), no_grad()
MODULE BASICS          | __init__, forward, parameters(), named_parameters()
SAVING/LOADING         | torch.save(model.state_dict()), model.load_state_dict()
COMMON LAYERS          | Linear, Conv2d, BatchNorm2d, Dropout, ReLU, MaxPool2d
DATA PIPELINE          | Dataset(__len__, __getitem__) + DataLoader(batch, shuffle, workers)
LOSS FUNCTIONS         | MSELoss (regression), CrossEntropyLoss (classification), BCEWithLogitsLoss
OPTIMIZERS             | SGD, Adam, AdamW (default choice)
LR SCHEDULERS          | StepLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
TRAINING ORDER         | zero_grad -> forward -> loss -> backward -> clip -> step
DEBUGGING              | model.eval(), torch.no_grad(), set_detect_anomaly(True)
DEVICE                 | device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPRODUCIBILITY        | manual_seed, cudnn.deterministic, cudnn.benchmark=False
```

---

*These notes are your reference. Return to them when you are unsure. The code examples are
designed to be copy-pasted and modified. Master each section before moving to the next.*
