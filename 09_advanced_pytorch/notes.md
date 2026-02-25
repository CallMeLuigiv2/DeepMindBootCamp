# Advanced PyTorch — Expert Reference Notes

This document is your reference manual for the parts of PyTorch that most people never learn. Everything here is battle-tested — these are the techniques used in production research at organizations like DeepMind, FAIR, and Google Brain.

---

## Table of Contents

1. [Custom Autograd Functions](#1-custom-autograd-functions)
2. [Hooks, Debugging, and Internals](#2-hooks-debugging-and-internals)
3. [Distributed Training](#3-distributed-training)
4. [Advanced Training Techniques](#4-advanced-training-techniques)
5. [Production PyTorch](#5-production-pytorch)

---

## 1. Custom Autograd Functions

### When You Need This

- You have an operation whose derivative PyTorch cannot compute automatically (e.g., a non-differentiable operation where you want to supply a surrogate gradient).
- You need to implement a numerically stable backward pass that differs from what automatic differentiation would produce (e.g., log-sum-exp).
- You are wrapping a C/CUDA kernel that PyTorch does not know about.
- You need a straight-through estimator for quantization or discrete decisions.
- You want to implement a custom operation for meta-learning that requires higher-order gradients.

You do NOT need this when:
- You are composing existing differentiable operations. PyTorch's autograd handles composition perfectly.
- You think it will be faster. In most cases, PyTorch's fused kernels for standard operations are already optimal.

### The API

```python
class torch.autograd.Function:
    @staticmethod
    def forward(ctx, *args, **kwargs) -> Tensor:
        """
        Performs the forward computation.

        Parameters:
            ctx: A context object. Use ctx.save_for_backward() to stash tensors
                 needed in backward. Use ctx.mark_non_differentiable() for outputs
                 that should not receive gradients.
            *args: Input tensors and other arguments.

        Returns:
            Output tensor(s).
        """
        pass

    @staticmethod
    def backward(ctx, *grad_outputs) -> tuple:
        """
        Computes gradients of the forward operation.

        Parameters:
            ctx: The same context object from forward.
            *grad_outputs: Gradient of the loss with respect to each output
                           of forward. One grad_output per output tensor.

        Returns:
            Tuple of gradients with respect to each input of forward.
            Return None for inputs that do not require gradients.
            The number of returned values MUST match the number of inputs
            to forward (excluding ctx).
        """
        pass
```

Key context methods:
- `ctx.save_for_backward(*tensors)` — Save tensors for use in backward. ONLY use this method for tensors. It handles memory management correctly.
- `ctx.saved_tensors` — Retrieve saved tensors in backward. Returns a tuple in the same order they were saved.
- `ctx.needs_input_grad` — A tuple of booleans indicating which inputs need gradients.
- `ctx.mark_non_differentiable(*tensors)` — Mark output tensors that should not receive gradients.

### Code: Complete Custom Autograd Function Template

```python
import torch
from torch.autograd import Function

class MyCustomFunction(Function):
    """
    Template for a custom autograd function.

    Mathematical operation: output = f(input, param)
    Backward: d_loss/d_input = d_loss/d_output * df/d_input
              d_loss/d_param = d_loss/d_output * df/d_param
    """

    @staticmethod
    def forward(ctx, input_tensor, param_tensor):
        # Step 1: Save tensors needed for backward.
        # ONLY save what you need. Do not save the output if you can
        # recompute it from saved inputs.
        ctx.save_for_backward(input_tensor, param_tensor)

        # Step 2: You can also save non-tensor data on ctx directly.
        # This is for scalars, shapes, flags — NOT tensors.
        ctx.some_flag = True

        # Step 3: Compute the forward pass.
        output = input_tensor * torch.sigmoid(param_tensor * input_tensor)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Step 1: Retrieve saved tensors.
        input_tensor, param_tensor = ctx.saved_tensors

        # Step 2: Compute gradients.
        # grad_output is d_loss/d_output. We need to compute:
        #   d_loss/d_input = grad_output * d_output/d_input
        #   d_loss/d_param = grad_output * d_output/d_param

        sigmoid_val = torch.sigmoid(param_tensor * input_tensor)
        sigmoid_deriv = sigmoid_val * (1 - sigmoid_val)

        # d_output/d_input = sigmoid(beta*x) + x * sigmoid'(beta*x) * beta
        grad_input = grad_output * (sigmoid_val + input_tensor * sigmoid_deriv * param_tensor)

        # d_output/d_param = x * sigmoid'(beta*x) * x = x^2 * sigmoid'(beta*x)
        grad_param = grad_output * (input_tensor * sigmoid_deriv * input_tensor)

        # Step 3: Return one gradient per input to forward (excluding ctx).
        # Order must match the order of inputs in forward.
        return grad_input, grad_param


# Usage:
# Always use .apply(), never call forward() directly.
my_fn = MyCustomFunction.apply
output = my_fn(input_tensor, param_tensor)
```

### Code: Straight-Through Estimator

The Straight-Through Estimator (STE) is used when the forward pass contains a non-differentiable operation (like rounding, thresholding, or argmax) but you want gradients to flow through it anyway. The key insight: in the backward pass, pretend the non-differentiable operation was the identity function.

This is used extensively in:
- Quantization-aware training (rounding to fixed-point in forward, identity gradient in backward)
- Binary neural networks (sign function in forward, identity in backward)
- Discrete VAEs (categorical sampling in forward, continuous relaxation in backward)

```python
class StraightThroughEstimator(Function):
    """
    Forward: threshold at 0 (hard step function).
    Backward: pass gradient through unchanged (identity).

    This is mathematically wrong — the derivative of a step function is zero
    (or a Dirac delta at 0). But it works in practice because it provides a
    useful learning signal.
    """

    @staticmethod
    def forward(ctx, input_tensor):
        # Hard threshold: output is 1 where input >= 0, else 0.
        output = (input_tensor >= 0).float()
        # We do NOT need to save anything for backward because the STE
        # backward is just the identity — it does not depend on the input.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Pass the gradient through unchanged.
        # This is the "straight-through" part.
        return grad_output


# A more practical variant: STE with clamped gradient.
# Only pass gradients through where |input| <= 1 (the "saturated" STE).
class ClampedSTE(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return (input_tensor >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        # Only pass gradient where |input| <= 1.
        # This prevents gradients from growing without bound for inputs
        # far from the threshold.
        grad_input = grad_output * (input_tensor.abs() <= 1).float()
        return grad_input


# Usage: train a network with binary activations.
class BinaryNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)
        self.ste = StraightThroughEstimator.apply

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.ste(self.fc1(x))  # Binary activation
        x = self.ste(self.fc2(x))  # Binary activation
        x = self.fc3(x)            # Linear output for classification
        return x
```

### Code: Gradient Checking

```python
from torch.autograd import gradcheck

# gradcheck compares analytical gradients (from your backward method)
# against numerical gradients (finite differences).
# It requires float64 inputs for numerical stability.

input_tensor = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
param_tensor = torch.randn(1, dtype=torch.float64, requires_grad=True)

# gradcheck returns True if gradients match, raises an error otherwise.
test = gradcheck(MyCustomFunction.apply, (input_tensor, param_tensor), eps=1e-6, atol=1e-4)
print(f"Gradient check passed: {test}")

# For functions with non-continuous derivatives (like STE), gradcheck
# will fail because the numerical gradient does not match the surrogate.
# This is expected. You should still test the non-STE parts.
```

### Gotchas

1. **Forgetting @staticmethod**: Both `forward` and `backward` MUST be static methods. If you forget, you get a cryptic error about the number of arguments.
2. **Saving tensors outside ctx.save_for_backward**: If you do `ctx.my_tensor = some_tensor`, PyTorch cannot manage its memory. It will not be freed when it should be. Always use `save_for_backward` for tensors.
3. **Wrong number of return values in backward**: You must return exactly one gradient per input to forward (excluding ctx). If forward takes 3 inputs, backward must return 3 values (use None for non-differentiable inputs).
4. **In-place modifications**: Never modify inputs in-place in forward. It breaks gradient computation. PyTorch has version tracking that will catch this, but the error message is confusing.
5. **Calling .apply() vs .forward()**: Always use `MyFunction.apply(...)`, never `MyFunction.forward(ctx, ...)`. The `.apply()` method sets up the autograd graph correctly.

### DeepMind Perspective

Custom autograd functions are the foundation of several research directions: quantization-aware training, neural architecture search with discrete decisions, reinforcement learning with non-differentiable environments (REINFORCE uses a form of STE), and meta-learning (MAML requires second-order gradients through the optimization process). If you cannot write a custom backward pass, you cannot implement these techniques from scratch.

---

## 2. Hooks, Debugging, and Internals

### When You Need This

- You need to extract intermediate features from a pretrained model without modifying its code.
- You are debugging vanishing or exploding gradients and need to see gradient magnitudes at every layer.
- You need to modify gradients during backpropagation (gradient reversal, gradient penalty).
- You are implementing pruning, quantization, or other techniques that need to observe or modify weights/activations during training.
- You want to log activation statistics for monitoring training health.

### The API

```python
# --- Forward Hooks ---
# Called AFTER forward() completes.
# Signature: hook(module, input, output) -> None or modified output
handle = module.register_forward_hook(hook_fn)

# --- Forward Pre-Hooks ---
# Called BEFORE forward(). Can modify the input.
# Signature: hook(module, input) -> None or modified input
handle = module.register_forward_pre_hook(hook_fn)

# --- Backward Hooks (Full) ---
# Called AFTER backward() completes for the module.
# Signature: hook(module, grad_input, grad_output) -> None or modified grad_input
handle = module.register_full_backward_hook(hook_fn)

# --- Tensor Hooks ---
# Called when gradient is computed for a specific tensor.
# Signature: hook(grad) -> None or modified grad
handle = tensor.register_hook(hook_fn)

# --- Removing Hooks ---
handle.remove()  # Always clean up when done.
```

### Code: Five Practical Hook Patterns

**Pattern 1: Extracting Intermediate Features for Transfer Learning**

```python
import torch
import torchvision.models as models

class FeatureExtractor:
    """
    Extract features from any intermediate layer of a pretrained model
    without modifying the model code.
    """

    def __init__(self, model, layer_names):
        self.model = model
        self.features = {}
        self._hooks = []

        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def __call__(self, x):
        self.features = {}
        _ = self.model(x)
        return self.features

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# Usage:
model = models.resnet50(pretrained=True)
model.eval()

extractor = FeatureExtractor(model, ['layer1', 'layer2', 'layer3', 'layer4'])

dummy_input = torch.randn(1, 3, 224, 224)
features = extractor(dummy_input)

for name, feat in features.items():
    print(f"{name}: {feat.shape}")
# layer1: torch.Size([1, 256, 56, 56])
# layer2: torch.Size([1, 512, 28, 28])
# layer3: torch.Size([1, 1024, 14, 14])
# layer4: torch.Size([1, 2048, 7, 7])

extractor.close()  # Always clean up.
```

**Pattern 2: Per-Layer Gradient Clipping**

```python
def clip_gradient_hook(max_norm):
    """
    Create a hook that clips gradients for a specific module.
    Unlike torch.nn.utils.clip_grad_norm_ which clips globally,
    this clips per-layer.
    """
    def hook(module, grad_input, grad_output):
        clipped = []
        for g in grad_input:
            if g is not None:
                clipped.append(torch.clamp(g, -max_norm, max_norm))
            else:
                clipped.append(g)
        return tuple(clipped)
    return hook


# Apply per-layer clipping to specific layers:
model = torch.nn.Sequential(
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 10),
)

# Clip gradients in the first linear layer to [-0.5, 0.5]
handle = model[0].register_full_backward_hook(clip_gradient_hook(0.5))
```

**Pattern 3: Activation Statistics for Debugging**

```python
class ActivationMonitor:
    """
    Log activation statistics during training to detect:
    - Dead ReLUs (mean activation near 0, fraction of zeros high)
    - Exploding activations (mean or std growing over time)
    - Saturated sigmoids/tanhs (outputs near -1/1)
    """

    def __init__(self, model):
        self.stats = {}
        self._hooks = []

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.ReLU, torch.nn.Linear,
                                   torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                if isinstance(output, torch.Tensor):
                    self.stats[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'fraction_zero': (output == 0).float().mean().item(),
                        'fraction_negative': (output < 0).float().mean().item(),
                    }
        return hook

    def report(self):
        for name, s in self.stats.items():
            print(f"{name:30s} | mean={s['mean']:+.4f} std={s['std']:.4f} "
                  f"zeros={s['fraction_zero']:.2%} neg={s['fraction_negative']:.2%}")

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
```

**Pattern 4: Gradient Reversal for Domain Adaptation**

```python
class GradientReversalFunction(torch.autograd.Function):
    """
    In the forward pass: identity.
    In the backward pass: negate the gradient and scale by lambda.

    Used in domain adaptation (DANN - Domain-Adversarial Neural Networks).
    The domain classifier receives reversed gradients, so the feature
    extractor learns domain-invariant representations.
    """

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_val=1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


# Usage in a DANN architecture:
class DANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
        )
        self.class_classifier = torch.nn.Linear(256, 10)
        self.domain_classifier = torch.nn.Sequential(
            GradientReversalLayer(lambda_val=1.0),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        class_pred = self.class_classifier(features)
        domain_pred = self.domain_classifier(features)
        return class_pred, domain_pred
```

**Pattern 5: Pruning-Related Hooks**

```python
class MagnitudePruner:
    """
    Zero out the smallest N% of weights after each forward pass.
    Uses forward pre-hooks to apply the mask before computation.
    """

    def __init__(self, model, prune_ratio=0.2):
        self.masks = {}
        self._hooks = []

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # Compute the mask based on weight magnitude.
                self._compute_mask(name, module, prune_ratio)
                # Register a hook to apply the mask before every forward pass.
                hook = module.register_forward_pre_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _compute_mask(self, name, module, prune_ratio):
        with torch.no_grad():
            weight = module.weight.data.abs()
            threshold = torch.quantile(weight.flatten(), prune_ratio)
            self.masks[name] = (weight > threshold).float()

    def _make_hook(self, name):
        def hook(module, input):
            module.weight.data *= self.masks[name]
        return hook

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
```

### Gotchas

1. **Memory leaks from hooks**: Every hook holds a reference to the closure and any captured variables. If you capture the model itself inside a hook closure, you create a reference cycle. Always use `handle.remove()` when done.
2. **Hook ordering**: Hooks fire in the order they were registered. If you register two forward hooks on the same module, the first one fires first. This matters if one hook modifies the output and another reads it.
3. **register_backward_hook is deprecated**: Use `register_full_backward_hook` instead. The old API had issues with modules that have multiple inputs or outputs.
4. **Hooks slow down training**: Each hook adds overhead. Forward hooks are cheap (just a function call). Backward hooks can be expensive if they do computation. Profile before shipping.
5. **Tensor hooks vs module hooks**: Tensor hooks (`tensor.register_hook(fn)`) fire when the gradient for that specific tensor is computed. Module hooks fire when the module's forward/backward completes. Use tensor hooks when you need gradient-level control; use module hooks when you need input/output level control.

### PyTorch Internals: The Dispatcher

The PyTorch dispatcher is the routing system that decides which implementation of an operator to call. When you write `torch.add(a, b)`, the dispatcher looks at:
- The device (CPU, CUDA, MPS)
- The dtype
- Whether autograd is needed
- Whether the tensors are vmap-batched
- Whether torch.compile is active

It then routes to the correct kernel. You almost never interact with the dispatcher directly, but understanding it explains why PyTorch can run the same code on CPU and GPU without changes.

### torch.compile

`torch.compile` is PyTorch 2.0's flagship feature. It works in three stages:

1. **TorchDynamo** (graph capture): Intercepts Python bytecode execution and captures a computation graph. It handles most Python control flow by "guarding" on conditions and creating specialized graphs.
2. **AOTAutograd** (automatic differentiation): Takes the captured graph and generates the backward graph ahead of time.
3. **TorchInductor** (code generation): Takes the forward and backward graphs and generates optimized Triton (GPU) or C++ (CPU) code with operator fusion.

```python
import torch

model = MyModel()
compiled_model = torch.compile(model)  # Default settings

# With options:
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",   # Minimize overhead (uses CUDA graphs)
    # mode="max-autotune",    # Try all possible optimizations (slow compile)
    # mode="default",         # Balanced (the default)
    dynamic=False,            # Set True if input shapes change
    fullgraph=False,          # Set True to error on graph breaks
)

# Usage is identical to the uncompiled model.
output = compiled_model(input_tensor)
```

When torch.compile helps:
- Compute-bound operations (large matrix multiplies, convolutions)
- Models with many small operations that can be fused (element-wise chains)
- Training loops with stable shapes

When it does not help:
- Data-loading bound workloads (compile does not speed up I/O)
- Models with heavy dynamic control flow (causes graph breaks and recompilation)
- Very small models where compilation overhead dominates

Common issues:
- **Graph breaks**: When Dynamo encounters code it cannot trace (e.g., data-dependent control flow, calls to non-PyTorch libraries), it "breaks" the graph and falls back to eager execution. Use `fullgraph=True` to catch these.
- **Dynamic shapes**: If input shapes change, Dynamo recompiles. Use `dynamic=True` to generate shape-generic code, at some performance cost.
- **Compilation time**: The first call to a compiled model is slow (seconds to minutes). Subsequent calls are fast.

### DeepMind Perspective

Hooks are essential for research tooling. At DeepMind, hooks are used to build experiment monitoring infrastructure, implement gradient manipulation techniques like gradient reversal and gradient penalty, extract features for auxiliary losses, and debug training instabilities. The ability to observe and modify the computation graph without changing the model code is what makes rapid experimentation possible.

`torch.compile` is increasingly important for training efficiency. DeepMind's JAX ecosystem has had compilation (via XLA) from the start. PyTorch's `torch.compile` is catching up, and understanding it is essential for competitive training speeds.

---

## 3. Distributed Training

### When You Need This

- Your model takes too long to train on a single GPU.
- You need to train on more data per unit time (data parallelism).
- Your model does not fit on a single GPU (model parallelism, FSDP).
- You are doing hyperparameter sweeps and want to use multiple GPUs efficiently.

### The API: DistributedDataParallel

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- Initialize Process Group ---
dist.init_process_group(
    backend="nccl",           # "nccl" for GPU, "gloo" for CPU
    # init_method, rank, and world_size are set by torchrun automatically.
)

# --- Get Local Rank ---
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# --- Wrap Model ---
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])
# After wrapping, model.module gives you the original model.

# --- Distributed Sampler ---
sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True,
)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# --- Training Loop ---
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # CRITICAL: ensures different shuffling each epoch.
    for batch in loader:
        # Forward, backward, step — same as single GPU.
        pass

# --- Cleanup ---
dist.destroy_process_group()
```

Launch with:
```bash
torchrun --nproc_per_node=4 train.py
```

### Code: Complete DistributedDataParallel Training Script

```python
"""
Complete DDP training script.
Launch: torchrun --nproc_per_node=NUM_GPUS train_ddp.py

Every line is annotated to explain what it does and why.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def setup():
    """
    Initialize the distributed process group.
    torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, and MASTER_ADDR/MASTER_PORT
    as environment variables. init_process_group reads them automatically
    when using the "env://" init method (the default).
    """
    dist.init_process_group(backend="nccl")  # Use "gloo" if no GPU.


def cleanup():
    """Destroy the process group. Call this at the end of training."""
    dist.destroy_process_group()


def get_rank():
    """Global rank of this process (0 to world_size - 1)."""
    return dist.get_rank()


def get_local_rank():
    """
    Local rank: which GPU on this machine this process should use.
    Set by torchrun as an environment variable.
    """
    return int(os.environ["LOCAL_RANK"])


def get_world_size():
    """Total number of processes across all machines."""
    return dist.get_world_size()


def create_model(local_rank):
    """
    Create model and move to the correct GPU.
    Each process creates its own copy. DDP handles synchronization.
    """
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model = model.to(local_rank)

    # Wrap in DDP. device_ids tells DDP which GPU this process uses.
    # DDP inserts hooks into the model that trigger allreduce during backward.
    # After every backward pass, all processes have identical gradients.
    model = DDP(model, device_ids=[local_rank])

    return model


def create_dataloader(local_rank):
    """
    Create a DataLoader with DistributedSampler.
    The sampler ensures each process sees a unique subset of the data.
    """
    # Dummy dataset for demonstration.
    X = torch.randn(10000, 784)
    y = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(X, y)

    # DistributedSampler splits the dataset across processes.
    # Process 0 gets indices [0, 4, 8, ...], process 1 gets [1, 5, 9, ...], etc.
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,  # Shuffle within each process's partition.
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,     # Sampler is mutually exclusive with shuffle=True.
        num_workers=2,
        pin_memory=True,     # Speeds up CPU-to-GPU transfer.
    )

    return loader, sampler


def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save checkpoint from rank 0 only.
    All processes have identical parameters (DDP guarantees this),
    so saving from one process is sufficient.
    """
    if get_rank() == 0:
        # model.module gives the unwrapped model (without DDP wrapper).
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filepath)


def load_checkpoint(model, optimizer, filepath, local_rank):
    """
    Load checkpoint on all ranks.
    map_location ensures tensors are loaded to the correct GPU.
    """
    # map_location is critical: without it, all processes load to GPU 0.
    checkpoint = torch.load(filepath, map_location=f'cuda:{local_rank}')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def train():
    setup()

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    model = create_model(local_rank)
    loader, sampler = create_dataloader(local_rank)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        # CRITICAL: set_epoch ensures different shuffling each epoch.
        # Without this, every epoch sees data in the same order.
        sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(local_rank)
            y_batch = y_batch.to(local_rank)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)

            # DDP hooks trigger allreduce here. After backward(),
            # all processes have identical averaged gradients.
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Only print from rank 0 to avoid duplicate output.
        if get_rank() == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save final checkpoint.
    save_checkpoint(model, optimizer, num_epochs, "checkpoint.pt")

    cleanup()


if __name__ == "__main__":
    train()
```

### FSDP (Fully Sharded Data Parallel)

When your model does not fit on a single GPU, DDP is not enough because every GPU holds a full copy of the model. FSDP shards the model parameters, gradients, and optimizer states across GPUs.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Basic FSDP wrapping:
model = MyLargeModel()
model = FSDP(model)

# With configuration:
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # Maximum memory savings.
    # ShardingStrategy.SHARD_GRAD_OP = shard gradients and optimizer only.
    # ShardingStrategy.NO_SHARD = equivalent to DDP.
)
```

Use FSDP when:
- Model parameters alone exceed one GPU's memory.
- You are training models with billions of parameters.
- You need to maximize memory efficiency at the cost of communication overhead.

### Gotchas

1. **Forgetting sampler.set_epoch()**: Without this, the data order is identical every epoch. This is a silent bug — training works but converges worse.
2. **Saving from all ranks**: If every process saves a checkpoint, you write N identical files and waste I/O. Always guard with `if get_rank() == 0`.
3. **map_location on load**: Without `map_location=f'cuda:{local_rank}'`, all processes load tensors to GPU 0, causing an OOM.
4. **Printing from all ranks**: Every `print()` statement runs on every process. Guard with `if get_rank() == 0` or use logging.
5. **Unused parameters**: If some parameters are not used in every forward pass (e.g., conditional branches), DDP will hang because it expects allreduce for every parameter. Use `find_unused_parameters=True`, but be aware of the performance cost.

### DeepMind Perspective

At DeepMind, virtually all large-scale training uses distributed training. Data parallelism is the default for models that fit on a single GPU. For large language models and large-scale RL agents, model parallelism and pipeline parallelism are combined with data parallelism. Understanding the communication patterns (allreduce, all-gather, reduce-scatter) and their costs is essential for reasoning about training efficiency at scale.

---

## 4. Advanced Training Techniques

### Gradient Accumulation

**When you need this:** You want to train with a large effective batch size but your GPU memory only supports small batches. Common in NLP where batch sizes of 2048+ tokens are standard but a single GPU can only fit 32.

```python
accumulation_steps = 8  # Effective batch size = 32 * 8 = 256
optimizer.zero_grad()

for i, (X_batch, y_batch) in enumerate(loader):
    output = model(X_batch)
    loss = criterion(output, y_batch)

    # Normalize loss by accumulation steps.
    # This ensures the total gradient magnitude is the same as a single
    # large batch, not 8x larger.
    loss = loss / accumulation_steps
    loss.backward()  # Gradients accumulate in .grad attributes.

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights with accumulated gradients.
        optimizer.zero_grad()  # Reset gradients for next accumulation cycle.
```

**Key insight:** `loss.backward()` ADDS to `.grad`, it does not replace it. This is what makes accumulation work. Normalizing the loss by `accumulation_steps` ensures mathematical equivalence with a single large batch.

### Mixed Precision Training

**When you need this:** You want to speed up training and reduce memory usage on GPUs with Tensor Cores (V100, A100, H100, and newer).

**Why it works:**
- float16 operations are 2-8x faster than float32 on modern GPUs with Tensor Cores.
- float16 tensors use half the memory, allowing larger batches or models.
- float16 has limited range (max ~65504, min positive ~6e-8). Small gradients underflow to zero.
- Solution: loss scaling. Multiply the loss by a large constant before backward (so gradients are larger and do not underflow), then divide the gradients by the same constant before the optimizer step.

```python
from torch.cuda.amp import autocast, GradScaler

# GradScaler manages loss scaling automatically.
# It starts with a large scale factor, and if it detects inf/nan gradients
# (indicating the scale is too large), it reduces the scale and skips the step.
scaler = GradScaler()

for X_batch, y_batch in loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad()

    # autocast automatically runs operations in float16 where safe,
    # and keeps float32 where needed (e.g., loss computation, batch norm).
    with autocast():
        output = model(X_batch)
        loss = criterion(output, y_batch)

    # scaler.scale(loss) multiplies loss by the scale factor.
    # backward() computes scaled gradients.
    scaler.scale(loss).backward()

    # scaler.step() does two things:
    # 1. Unscales the gradients (divides by scale factor).
    # 2. If no inf/nan gradients, calls optimizer.step().
    # 3. If inf/nan detected, skips the step.
    scaler.step(optimizer)

    # Update the scale factor for next iteration.
    scaler.update()
```

**Operations that are safe in float16:** Matrix multiplication, convolution, linear layers.
**Operations that need float32:** Loss computation, softmax, batch normalization, layer normalization, any operation involving summation over many elements (risk of overflow/underflow).
`autocast` handles this mapping automatically.

### Gradient Checkpointing

**When you need this:** Your model is so deep that intermediate activations consume too much memory. Gradient checkpointing trades compute for memory: instead of storing all activations, it recomputes them during the backward pass.

```python
from torch.utils.checkpoint import checkpoint

class DeepResNet(nn.Module):
    def __init__(self, num_blocks=50):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(256, 256) for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        for block in self.blocks:
            # Without checkpointing: all intermediate activations stored.
            # x = block(x)

            # With checkpointing: activations for this block are recomputed
            # during backward instead of stored.
            x = checkpoint(block, x, use_reentrant=False)

        return self.classifier(x)
```

**Memory savings:** Roughly proportional to the number of checkpointed segments. For N layers, without checkpointing you store N activation tensors. With checkpointing, you store sqrt(N) (if you checkpoint every sqrt(N) layers).

**Compute cost:** Approximately 33% more compute (one extra forward pass through checkpointed segments during backward).

**use_reentrant=False:** Always use this. The old reentrant implementation (`use_reentrant=True`) has subtle bugs with multiple uses and does not support all autograd features. The non-reentrant version is correct and will become the default.

### torch.compile Deep Dive

```python
import torch

# Basic usage:
model = MyModel()
compiled = torch.compile(model)

# Benchmark:
import time

# Warm up (first call triggers compilation):
compiled(dummy_input)

# Benchmark compiled:
start = time.time()
for _ in range(100):
    compiled(dummy_input)
torch.cuda.synchronize()
compiled_time = time.time() - start

# Benchmark eager:
start = time.time()
for _ in range(100):
    model(dummy_input)
torch.cuda.synchronize()
eager_time = time.time() - start

print(f"Eager: {eager_time:.3f}s, Compiled: {compiled_time:.3f}s, "
      f"Speedup: {eager_time / compiled_time:.2f}x")
```

**What torch.compile does internally:**

1. TorchDynamo hooks into Python's frame evaluation to capture the computation graph from your Python code.
2. It creates "guards" — conditions under which the captured graph is valid (e.g., input shape, dtype, device).
3. AOTAutograd generates the backward graph ahead of time.
4. TorchInductor generates optimized Triton kernels (GPU) or C++ code (CPU), with operator fusion.

**When it helps:**
- Element-wise operations that can be fused (e.g., a chain of add, mul, activation).
- Large compute-bound operations (big matrix multiplies benefit from better scheduling).
- Models with many small operations (reduces Python overhead and kernel launch overhead).

**When it does not help:**
- I/O-bound workloads (data loading is the bottleneck).
- Very dynamic models with data-dependent control flow (causes recompilation).
- Tiny models where compilation overhead dominates.

**Common compilation errors and fixes:**

```python
# Problem: Graph break due to data-dependent control flow.
def forward(self, x):
    if x.sum() > 0:  # Data-dependent! Dynamo cannot trace this statically.
        return x * 2
    else:
        return x * 3

# Fix: Use torch.where instead of Python if.
def forward(self, x):
    return torch.where(x.sum() > 0, x * 2, x * 3)


# Problem: Dynamic input shapes cause recompilation.
# Fix: Use dynamic=True.
compiled = torch.compile(model, dynamic=True)


# Problem: Calling non-PyTorch code inside the model.
def forward(self, x):
    x_np = x.numpy()  # Graph break! numpy is not traceable.
    return torch.from_numpy(np.sin(x_np))

# Fix: Use PyTorch operations.
def forward(self, x):
    return torch.sin(x)
```

### Training Techniques from Papers

**Exponential Moving Average (EMA):**

```python
class EMA:
    """
    Maintains an exponential moving average of model parameters.
    Used for evaluation — the EMA model often generalizes better than
    the trained model because it averages over the training trajectory.

    Used in: virtually every modern generative model (diffusion, GANs),
    and in semi-supervised learning.
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights as a copy of model weights.
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Call after each optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self):
        """Replace model weights with EMA weights for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
```

**Label Smoothing:**

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Instead of one-hot targets [0, 0, 1, 0, ...], use
    [eps/K, eps/K, 1-eps+eps/K, eps/K, ...] where K is the number of classes.

    This prevents the model from becoming overconfident and improves
    generalization. Standard in ImageNet training and machine translation.
    """

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)

        # NLL loss for the true class.
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)

        # KL divergence from uniform distribution (encourages uncertainty).
        smooth_loss = -log_probs.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
```

**Stochastic Depth:**

```python
class StochasticDepthResBlock(nn.Module):
    """
    During training, randomly skip this block with probability drop_prob.
    During evaluation, scale the output by (1 - drop_prob) to compensate.

    This is a form of regularization that effectively trains an ensemble
    of networks of different depths. Used in EfficientNet, Vision Transformers.
    """

    def __init__(self, in_channels, out_channels, drop_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop_prob = drop_prob

    def forward(self, x):
        identity = x

        if self.training and torch.rand(1).item() < self.drop_prob:
            # Skip this block entirely during training.
            return identity

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if not self.training:
            # Scale output during evaluation to match expected value.
            out = out * (1 - self.drop_prob)

        return out + identity
```

### Gotchas

1. **Gradient accumulation with mixed precision**: The `scaler.scale(loss / accumulation_steps).backward()` is correct. Do not unscale before accumulation is complete.
2. **Gradient checkpointing with DDP**: Works, but you must use `use_reentrant=False` to avoid issues with DDP's gradient synchronization hooks.
3. **EMA and distributed training**: Only update EMA on rank 0 if you want to save memory. Or update on all ranks for consistency.
4. **Label smoothing value**: 0.1 is standard. Higher values (0.3+) hurt performance. Lower values (0.01) have negligible effect.
5. **torch.compile and gradient checkpointing**: These compose, but compilation may be slower because the recomputation creates additional graph complexity.

### DeepMind Perspective

At DeepMind, a standard training configuration for a large model combines DDP (or FSDP for very large models), mixed precision, gradient accumulation, gradient checkpointing, EMA, and often stochastic depth or dropout. These are not optional extras — they are the baseline. Understanding how they compose and interact is a core competency. When training runs cost hundreds of thousands of dollars in compute, a 10% efficiency improvement from proper mixed precision or compilation is not a nice-to-have — it is a requirement.

---

## 5. Production PyTorch

### Model Export: ONNX

**When you need this:** You want to run your model outside of Python, on a different framework, or on specialized hardware.

```python
import torch
import torch.onnx

model = MyModel()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,          # Store trained weights in the ONNX file.
    opset_version=17,            # ONNX operator set version.
    do_constant_folding=True,    # Fold constant operations for optimization.
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={               # Allow variable batch size.
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    },
)

# Verify with ONNX Runtime:
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
ort_outputs = session.run(None, ort_inputs)

# Compare outputs:
torch_output = model(dummy_input).detach().numpy()
np.testing.assert_allclose(torch_output, ort_outputs[0], rtol=1e-5, atol=1e-5)
print("ONNX output matches PyTorch output.")
```

### Quantization

**When you need this:** You want to reduce model size (4x) and inference latency (2-4x) with minimal accuracy loss.

**Dynamic Quantization** (simplest, no calibration data needed):

```python
import torch.quantization

model_fp32 = MyModel()
model_fp32.eval()

# Quantize Linear and LSTM layers to INT8.
# Weights are quantized statically. Activations are quantized dynamically
# at inference time.
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # Which layer types to quantize.
    dtype=torch.qint8,
)

# Compare sizes:
import os
torch.save(model_fp32.state_dict(), "model_fp32.pt")
torch.save(model_int8.state_dict(), "model_int8.pt")
print(f"FP32 size: {os.path.getsize('model_fp32.pt') / 1e6:.1f} MB")
print(f"INT8 size: {os.path.getsize('model_int8.pt') / 1e6:.1f} MB")
```

**Static Quantization** (better accuracy, requires calibration data):

```python
# Step 1: Prepare the model for quantization by inserting observers.
model_fp32.eval()
model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
model_prepared = torch.quantization.prepare(model_fp32)

# Step 2: Calibrate with representative data.
# The observers collect statistics about activation ranges.
with torch.no_grad():
    for X_batch, _ in calibration_loader:
        model_prepared(X_batch)

# Step 3: Convert to quantized model.
model_quantized = torch.quantization.convert(model_prepared)
```

**Quantization-Aware Training (QAT)** (best accuracy, most effort):

```python
# Insert fake quantization modules that simulate INT8 during training.
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('x86')
model_qat = torch.quantization.prepare_qat(model)

# Train as normal — the fake quantization modules add noise that
# teaches the model to be robust to quantization.
for epoch in range(num_epochs):
    for X_batch, y_batch in loader:
        output = model_qat(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Convert to actual quantized model for deployment.
model_qat.eval()
model_quantized = torch.quantization.convert(model_qat)
```

### Profiling with torch.profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("model_inference"):
        output = model(input_tensor)

# Print a summary sorted by CUDA time:
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export a Chrome trace (open in chrome://tracing):
prof.export_chrome_trace("trace.json")

# For training profiling with scheduling:
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,     # Skip the first step (warmup).
        warmup=1,   # Warmup the profiler.
        active=3,   # Profile 3 steps.
        repeat=1,   # Do this once.
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
) as prof:
    for step, (X_batch, y_batch) in enumerate(loader):
        if step >= 5:
            break
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()  # Signal the profiler that a step is complete.
```

### Custom Kernels: Triton Overview

When PyTorch's built-in operations are not fast enough, you can write custom GPU kernels. Triton is an accessible alternative to raw CUDA.

```python
import triton
import triton.language as tl

@triton.jit
def fused_add_relu_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + ReLU: output = max(x + y, 0)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = tl.maximum(x + y, 0.0)

    tl.store(output_ptr + offsets, output, mask=mask)


def fused_add_relu(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_add_relu_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# Benchmark:
x = torch.randn(1_000_000, device='cuda')
y = torch.randn(1_000_000, device='cuda')

# PyTorch (two separate kernels: add, then relu):
torch_result = torch.relu(x + y)

# Triton (one fused kernel):
triton_result = fused_add_relu(x, y)

assert torch.allclose(torch_result, triton_result)
```

When you need custom kernels:
- You have a sequence of element-wise operations that PyTorch cannot fuse.
- You need a specialized memory access pattern (e.g., custom attention).
- `torch.compile` does not generate efficient code for your use case.
- You are squeezing the last 10-20% of performance out of a critical operation.

When you do not:
- For standard operations (matmul, conv, attention). PyTorch's built-in kernels are highly optimized.
- When `torch.compile` already fuses your operations effectively.

### Gotchas

1. **ONNX export with dynamic control flow**: ONNX is a static graph format. If your model has data-dependent control flow, you need to use TorchScript scripting or restructure the model.
2. **Quantization accuracy drop**: Dynamic quantization on Linear layers is almost free in terms of accuracy. Static quantization on Conv layers may need calibration tuning. QAT recovers accuracy but requires retraining.
3. **Profiler overhead**: The profiler itself adds overhead. Profile representative workloads and ignore the absolute numbers — focus on relative comparisons between operations.
4. **Triton and autograd**: Triton kernels do not automatically participate in autograd. You need to wrap them in a custom autograd Function if you want gradients.

### DeepMind Perspective

Production deployment of research models is an active area. At DeepMind, models are typically exported to specialized serving infrastructure. The key insight: the model architecture should be designed with export in mind from the start. If you cannot export your model, it cannot leave the research lab. Quantization is increasingly important for deploying large models efficiently. Profiling is not something you do once — it is a continuous practice, especially as models and hardware evolve.

---

## Summary: The Expert's Checklist

Before you call yourself an expert PyTorch user, you should be able to:

1. Write a custom autograd Function with correct backward and verify it with gradcheck.
2. Use hooks to inspect and modify any part of the forward or backward pass without changing the model.
3. Write a distributed training script that works with torchrun and scales to multiple GPUs.
4. Set up mixed precision training, gradient accumulation, and gradient checkpointing — and explain the tradeoffs.
5. Export a model to ONNX, quantize it, and profile it.
6. Use `torch.compile` and understand when it helps and when it does not.
7. Read a paper and implement every training trick it describes, from scratch.

There are no shortcuts. The only way to learn this is to write the code, break it, debug it, and write it again.
