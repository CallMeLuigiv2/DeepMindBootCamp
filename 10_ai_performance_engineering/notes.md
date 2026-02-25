# Module 10: AI Performance Engineering -- Comprehensive Reference Notes

## Preamble

These notes are structured around a single principle: every optimization must be justified by measurement. For each technique, we present the problem, how to measure it, the solution, expected impact, and working code with before/after benchmarks.

This is a reference you will return to throughout your career. Bookmark it.

---

## Table of Contents

1. [Profiling Cookbook](#1-profiling-cookbook)
2. [Memory Budget Calculator](#2-memory-budget-calculator)
3. [GPU Memory Optimization Techniques](#3-gpu-memory-optimization-techniques)
4. [Mixed Precision Training -- Complete Guide](#4-mixed-precision-training----complete-guide)
5. [Post-Training Quantization](#5-post-training-quantization)
6. [DataLoader Optimization Checklist](#6-dataloader-optimization-checklist)
7. [torch.compile Practical Guide](#7-torchcompile-practical-guide)
8. [Operator Fusion and Custom Kernels](#8-operator-fusion-and-custom-kernels)
9. [End-to-End Optimization Strategy](#9-end-to-end-optimization-strategy)

---

## 1. Profiling Cookbook

### The Problem

You have a training pipeline and it feels slow. Maybe GPU utilization hovers around 40%. Maybe training an epoch takes an hour when you expected twenty minutes. But you do not know *why*. Without measurement, every optimization you try is a guess. Most guesses are wrong.

### How to Measure

#### Step 1: Quick GPU Utilization Check

Before touching the profiler, get a rough sense of GPU utilization:

```bash
# In a separate terminal during training:
watch -n 0.5 nvidia-smi

# Or for a cleaner view:
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

If GPU utilization is consistently below 70%, you likely have a non-GPU bottleneck. If it is above 90%, your GPU is the bottleneck (which is the healthy state -- it means nothing else is slowing you down).

#### Step 2: Measure DataLoader Throughput Independently

```python
import time
import torch
from torch.utils.data import DataLoader

def measure_dataloader_throughput(dataloader, n_batches=100):
    """Measure raw DataLoader throughput without any model computation."""
    start = time.perf_counter()
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
    elapsed = time.perf_counter() - start
    batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
    throughput = n_batches * batch_size / elapsed
    print(f"DataLoader throughput: {throughput:.0f} samples/sec ({elapsed:.2f}s for {n_batches} batches)")
    return throughput
```

Compare this to your model's training throughput. If they are close, the DataLoader is the bottleneck.

#### Step 3: Time Each Phase of a Training Step

```python
import time
import torch

def timed_training_step(model, dataloader, criterion, optimizer, device, n_steps=20):
    """Time each component of a training step."""
    model.train()
    timings = {"data_loading": [], "to_device": [], "forward": [], "backward": [], "optimizer": []}

    data_iter = iter(dataloader)
    for step in range(n_steps):
        # Data loading
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = next(data_iter)
        inputs, labels = batch
        t1 = time.perf_counter()
        timings["data_loading"].append(t1 - t0)

        # Transfer to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        timings["to_device"].append(t2 - t1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        timings["forward"].append(t3 - t2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        timings["backward"].append(t4 - t3)

        # Optimizer step
        optimizer.step()
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        timings["optimizer"].append(t5 - t4)

    # Print summary (skip first 5 steps for warmup)
    print("\n--- Training Step Timing Breakdown ---")
    total = 0
    for phase, times in timings.items():
        avg = sum(times[5:]) / len(times[5:])
        total += avg
        print(f"  {phase:20s}: {avg*1000:8.2f} ms")
    print(f"  {'TOTAL':20s}: {total*1000:8.2f} ms")
    print(f"  Throughput: {inputs.shape[0] / total:.0f} samples/sec")
    return timings
```

**Key: always call `torch.cuda.synchronize()` before taking timestamps.** CUDA operations are asynchronous. Without synchronization, you are timing the kernel *launch*, not the kernel *execution*.

#### Step 4: Full torch.profiler Session

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_training(model, dataloader, criterion, optimizer, device, n_steps=10):
    """Full profiling with torch.profiler."""
    model.train()
    data_iter = iter(dataloader)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(n_steps):
            inputs, labels = next(data_iter)
            inputs, labels = inputs.to(device), labels.to(device)

            with record_function("forward"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            with record_function("backward"):
                optimizer.zero_grad()
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

            prof.step()

    # Print the top operations by CUDA time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Print the top operations by CPU time
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Export Chrome trace for visualization
    prof.export_chrome_trace("trace.json")
    print("\nTrace exported to trace.json -- open in chrome://tracing")
    return prof
```

#### Step 5: Reading the Profiler Output

The `key_averages().table()` output looks like:

```
---------------------------------  --------  --------  --------  --------
                             Name  Self CPU   CPU total  Self CUDA  CUDA total
---------------------------------  --------  --------  --------  --------
                     aten::conv2d     0.5ms     15.2ms     12.3ms     12.3ms
                        aten::mm     0.2ms      8.7ms      8.1ms      8.1ms
                aten::batch_norm     0.3ms      6.1ms      5.8ms      5.8ms
            aten::_log_softmax     0.1ms      1.2ms      1.0ms      1.0ms
                      backward     0.8ms     32.5ms     28.1ms     28.1ms
---------------------------------  --------  --------  --------  --------
```

**How to interpret:**
- **Self CPU time high, Self CUDA time low**: CPU-bound operation. The GPU is waiting.
- **Self CUDA time high**: GPU-bound operation. This is where compute time goes.
- **CPU total >> Self CPU**: the operation calls sub-operations that are expensive.
- **Many small CUDA operations**: kernel launch overhead may dominate. Consider `torch.compile`.
- **Large gap between CPU total and CUDA total**: possible host-device synchronization or data transfer.

#### Identifying Specific Bottleneck Types

**(a) CPU Bottleneck Signature:**
- nvidia-smi shows GPU utilization < 50%.
- In profiler: CPU time dominates. Large gaps between CUDA kernels in the trace viewer.
- Common causes: data loading, preprocessing, Python overhead.
- Fix: optimize DataLoader, move work to GPU, use `torch.compile`.

**(b) Data Loading Bottleneck Signature:**
- DataLoader throughput (measured independently) is close to or lower than training throughput.
- In the trace viewer: long gaps at the beginning of each step (before any CUDA kernels).
- The `__getitem__` or `collate_fn` shows high CPU time.
- Fix: increase `num_workers`, use `pin_memory`, change data format, precompute.

**(c) GPU Idle Time Signature:**
- In the trace viewer: gaps between CUDA kernels.
- CUDA stream has visible empty periods.
- Often caused by: CPU work between GPU operations, host-device synchronization (`.item()`, `.cpu()`), small kernels with high launch overhead.
- Fix: remove sync points, use `torch.compile`, use CUDA graphs.

**(d) Memory-Bound vs Compute-Bound:**
- Compute arithmetic intensity: FLOPs / bytes_accessed.
- For your GPU, find the roofline crossover point: peak_FLOPS / memory_bandwidth.
  - A100: 312 TFLOPS (TF32) / 2 TB/s = 156 FLOP/byte. Operations with AI < 156 are memory-bound.
  - For FP16: 624 TFLOPS / 2 TB/s = 312 FLOP/byte.
- Large matmuls (m, k, n > 512) are typically compute-bound. Element-wise operations (activation, normalization) are almost always memory-bound.
- For memory-bound operations, operator fusion helps. For compute-bound operations, use Tensor Cores (mixed precision).

### The Solution: A Complete Profiling Script

```python
"""
Complete profiling script for a PyTorch training pipeline.
Run this BEFORE any optimization work.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# ---- Configuration ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 4
N_PROFILING_STEPS = 20
N_WARMUP_STEPS = 5

# ---- Setup ----
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)

model = models.resnet18(num_classes=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ---- Phase 1: DataLoader throughput ----
print("=" * 60)
print("PHASE 1: DataLoader Throughput")
print("=" * 60)
start = time.perf_counter()
for i, (inputs, _) in enumerate(dataloader):
    if i >= 100:
        break
elapsed = time.perf_counter() - start
dl_throughput = 100 * BATCH_SIZE / elapsed
print(f"DataLoader: {dl_throughput:.0f} samples/sec")

# ---- Phase 2: Training step breakdown ----
print("\n" + "=" * 60)
print("PHASE 2: Training Step Breakdown")
print("=" * 60)
model.train()
data_iter = iter(dataloader)
timings = {"data": [], "transfer": [], "forward": [], "backward": [], "optim": []}

for step in range(N_PROFILING_STEPS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    inputs, labels = next(data_iter)
    t1 = time.perf_counter()
    inputs = inputs.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    t5 = time.perf_counter()

    if step >= N_WARMUP_STEPS:
        timings["data"].append(t1 - t0)
        timings["transfer"].append(t2 - t1)
        timings["forward"].append(t3 - t2)
        timings["backward"].append(t4 - t3)
        timings["optim"].append(t5 - t4)

total_avg = 0
for phase, times in timings.items():
    avg_ms = sum(times) / len(times) * 1000
    total_avg += avg_ms
    print(f"  {phase:12s}: {avg_ms:8.2f} ms ({avg_ms/total_avg*100 if total_avg > 0 else 0:5.1f}%)")

train_throughput = BATCH_SIZE / (total_avg / 1000)
print(f"\n  Total step:   {total_avg:.2f} ms")
print(f"  Throughput:   {train_throughput:.0f} samples/sec")
print(f"  DataLoader:   {dl_throughput:.0f} samples/sec")
if dl_throughput < train_throughput * 1.5:
    print("  WARNING: DataLoader may be a bottleneck!")

# ---- Phase 3: Memory profile ----
print("\n" + "=" * 60)
print("PHASE 3: GPU Memory")
print("=" * 60)
torch.cuda.reset_peak_memory_stats()
inputs, labels = next(data_iter)
inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
outputs = model(inputs)
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

peak_mem = torch.cuda.max_memory_allocated() / 1024**2
current_mem = torch.cuda.memory_allocated() / 1024**2
reserved_mem = torch.cuda.memory_reserved() / 1024**2
print(f"  Peak allocated:    {peak_mem:.1f} MB")
print(f"  Current allocated: {current_mem:.1f} MB")
print(f"  Reserved:          {reserved_mem:.1f} MB")

# ---- Phase 4: torch.profiler ----
print("\n" + "=" * 60)
print("PHASE 4: Detailed Profiler")
print("=" * 60)
data_iter = iter(dataloader)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step in range(5):
        inputs, labels = next(data_iter)
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("\nTop 15 operations by CUDA time:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

prof.export_chrome_trace("profiling_trace.json")
print("\nChrome trace exported to profiling_trace.json")
```

### Expected Impact

Profiling itself does not speed anything up. But it tells you *where* to optimize. In practice, most engineers discover that 70-80% of their training time is being wasted on something they did not expect. Common discoveries:
- DataLoader with num_workers=0 is 3-5x slower than num_workers=4.
- A stray `.item()` call in the training loop forces CPU-GPU sync every step.
- Augmentation on CPU takes longer than the model forward pass.

---

## 2. Memory Budget Calculator

### The Problem

You want to train a model but do not know if it will fit in GPU memory. Or you want to find the maximum batch size for your GPU. Guessing leads to OOM errors and wasted time.

### How to Measure

```python
def calculate_memory_budget(model, batch_size, seq_len=None, optimizer_type="adam",
                            precision="fp32", gradient_checkpointing=False):
    """
    Calculate GPU memory budget for training.

    Returns a dict with memory breakdown in MB.
    """
    bytes_per_param = 4 if precision == "fp32" else 2  # fp16 or bf16

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Parameter memory
    param_mem = n_params * bytes_per_param

    # Gradient memory (only for trainable params)
    grad_mem = n_trainable * bytes_per_param

    # Optimizer state memory
    if optimizer_type == "adam" or optimizer_type == "adamw":
        # Adam stores m (first moment) and v (second moment) in fp32 always
        optim_mem = n_trainable * 4 * 2  # Two fp32 buffers
        if precision != "fp32":
            # Also stores fp32 master copy of weights
            optim_mem += n_trainable * 4
    elif optimizer_type == "sgd":
        optim_mem = 0  # Vanilla SGD has no state
    elif optimizer_type == "sgd_momentum":
        optim_mem = n_trainable * 4  # One buffer for momentum
    else:
        optim_mem = n_trainable * 4 * 2  # Conservative estimate

    # Activation memory (estimated via forward hook)
    activation_mem = _estimate_activation_memory(model, batch_size, seq_len,
                                                  bytes_per_param, gradient_checkpointing)

    # Convert to MB
    to_mb = lambda x: x / (1024 ** 2)

    budget = {
        "parameters": to_mb(param_mem),
        "gradients": to_mb(grad_mem),
        "optimizer_states": to_mb(optim_mem),
        "activations_estimated": to_mb(activation_mem),
        "total_estimated": to_mb(param_mem + grad_mem + optim_mem + activation_mem),
        "n_params_millions": n_params / 1e6,
    }

    return budget


def _estimate_activation_memory(model, batch_size, seq_len, bytes_per_element, checkpointing):
    """Estimate activation memory by running a forward pass with hooks."""
    import torch

    total_activation_bytes = 0
    activation_sizes = []

    def hook_fn(module, input, output):
        nonlocal total_activation_bytes
        if isinstance(output, torch.Tensor):
            size = output.nelement() * output.element_size()
            total_activation_bytes += size
            activation_sizes.append((module.__class__.__name__, size))
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    size = o.nelement() * o.element_size()
                    total_activation_bytes += size
                    activation_sizes.append((module.__class__.__name__, size))

    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    # Run a dummy forward pass
    device = next(model.parameters()).device
    try:
        if seq_len:
            dummy_input = torch.randn(batch_size, seq_len, dtype=torch.float32, device=device)
        else:
            # Attempt to infer input shape from first layer
            dummy_input = torch.randn(batch_size, 3, 32, 32, dtype=torch.float32, device=device)
        with torch.no_grad():
            model(dummy_input)
    except Exception:
        # If the dummy input fails, return a rough estimate
        n_params = sum(p.numel() for p in model.parameters())
        total_activation_bytes = n_params * bytes_per_element * batch_size // 10  # rough

    for h in hooks:
        h.remove()

    if checkpointing:
        # Rough estimate: checkpointing reduces activation memory by ~60%
        total_activation_bytes = int(total_activation_bytes * 0.4)

    return total_activation_bytes


def print_memory_budget(budget):
    """Pretty-print the memory budget."""
    print(f"\n{'=' * 50}")
    print(f"GPU MEMORY BUDGET")
    print(f"{'=' * 50}")
    print(f"  Model parameters:   {budget['n_params_millions']:.1f}M")
    print(f"  Parameter memory:   {budget['parameters']:.1f} MB")
    print(f"  Gradient memory:    {budget['gradients']:.1f} MB")
    print(f"  Optimizer states:   {budget['optimizer_states']:.1f} MB")
    print(f"  Activations (est):  {budget['activations_estimated']:.1f} MB")
    print(f"  {'─' * 40}")
    print(f"  TOTAL (estimated):  {budget['total_estimated']:.1f} MB")
    print(f"  TOTAL (estimated):  {budget['total_estimated']/1024:.2f} GB")
    print(f"{'=' * 50}")
```

### Worked Example: ResNet-50

```
Model: ResNet-50 (25.6M parameters)
Batch size: 32, Input: 3x224x224, Precision: fp32, Optimizer: Adam

Parameters:   25.6M * 4 bytes     =   102.4 MB
Gradients:    25.6M * 4 bytes     =   102.4 MB
Adam states:  25.6M * 4 * 2 bytes =   204.8 MB  (m and v buffers)
Activations:  ~batch_size dependent = ~2,500 MB  (at batch=32)
──────────────────────────────────────────────────
TOTAL (estimated):                 ≈ 2,910 MB  (2.84 GB)

With fp16 mixed precision:
Parameters:   102.4 MB  (master copy in fp32)
Gradients:    51.2 MB   (fp16)
Adam states:  204.8 MB + 102.4 MB = 307.2 MB  (fp32 states + fp32 master weights)
Activations:  ~1,250 MB (fp16, half of fp32)
──────────────────────────────────────────────────
TOTAL (estimated):                 ≈ 1,711 MB  (1.67 GB)  -- 41% reduction
```

### Worked Example: Small Transformer

```
Model: Transformer (d_model=512, n_heads=8, n_layers=6, vocab=30000, seq_len=512)

Embedding:  30000 * 512                     = 15.4M params
Per layer:  4 * 512 * 512 (attention)       = 1.05M
            + 512 * 2048 + 2048 * 512 (FFN) = 2.10M
            + norms, biases                 ≈ 0.01M
            ≈ 3.16M per layer * 6 layers    = 18.9M
Output:     512 * 30000                      = 15.4M
Total: approximately 49.7M parameters

FP32, Adam, batch_size=32, seq_len=512:
Parameters:   49.7M * 4       =   199 MB
Gradients:    49.7M * 4       =   199 MB
Adam states:  49.7M * 4 * 2   =   398 MB
Activations:  Per layer, the attention matrix alone is
              batch * n_heads * seq * seq * 4 bytes
              = 32 * 8 * 512 * 512 * 4 = 268 MB (per layer!)
              Plus QKV, FFN activations...
              Total ≈ 2,000 - 4,000 MB
──────────────────────────────────────────────────
TOTAL:                         ≈ 2,800 - 4,800 MB  (2.7 - 4.7 GB)

With Flash Attention:
Activation memory drops dramatically -- attention matrices are never materialized.
Per-layer activation memory drops from ~268MB to ~negligible for attention.
Total activations ≈ 800 - 1,500 MB
TOTAL:                         ≈ 1,600 - 2,100 MB  (1.6 - 2.1 GB)
```

### Expected Impact

Knowing your memory budget lets you:
- Choose the maximum batch size without trial and error.
- Decide whether gradient checkpointing or mixed precision is needed before training starts.
- Plan multi-GPU strategies based on whether the model fits on one GPU.
- Avoid the "run, OOM, reduce batch size, repeat" cycle that wastes hours.

---

## 3. GPU Memory Optimization Techniques

### 3.1 Gradient Checkpointing

**The Problem:** Activations from the forward pass must be stored for the backward pass. For deep models, this dominates GPU memory.

**How to Measure:**
```python
# Compare peak memory with and without checkpointing
torch.cuda.reset_peak_memory_stats()
# ... run training step ...
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak memory: {peak:.2f} GB")
```

**The Solution:**
```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerBlockWithCheckpointing(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Checkpoint the attention block
        x = x + checkpoint(self._attention_block, x, use_reentrant=False)
        # Checkpoint the FFN block
        x = x + checkpoint(self._ffn_block, x, use_reentrant=False)
        return x

    def _attention_block(self, x):
        return self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]

    def _ffn_block(self, x):
        return self.ffn(self.norm2(x))
```

**Expected Impact:**
- Memory reduction: 50-70% for activations (varies by model depth).
- Speed overhead: 20-33% slower (one extra forward pass per checkpointed segment).
- Net effect: you can increase batch size by 2-3x within the same memory, which often *more* than compensates for the per-step slowdown.

### 3.2 Gradient Accumulation

**The Problem:** You want effective batch size 256 but can only fit 32 in GPU memory.

**How to Measure:** Compare final model accuracy and per-step gradient norms between true batch=256 and accumulated batch=32x8.

**The Solution:**
```python
def train_with_gradient_accumulation(model, dataloader, criterion, optimizer,
                                      device, accumulation_steps=8):
    """Training loop with gradient accumulation."""
    model.train()
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass -- divide loss by accumulation steps to average
        outputs = model(inputs)
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            # Optional: clip gradients before stepping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining steps
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Expected Impact:**
- Memory: no change per micro-batch. Enables training with effective batch sizes that would otherwise OOM.
- Speed: no speedup per effective batch (same total forward/backward passes). But enables optimal batch sizes.
- Accuracy: mathematically equivalent to large batch (except for BatchNorm statistics, which see the micro-batch).

### 3.3 In-Place Operations and Memory Cleanup

**The Problem:** Unnecessary tensor allocations and retained references waste GPU memory.

**The Solution:**
```python
# Bad: creates a new tensor
x = x + residual
x = torch.relu(x)

# Good: in-place operations save memory
x += residual      # or x.add_(residual)
x = torch.relu_(x) # Note: be careful with autograd -- only safe in specific cases

# Bad: retains computational graph
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)  # Retains the entire graph!

# Good: detach before storing
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.detach().item())  # Stores a Python float, no graph

# Release memory when switching between training and inference
model.eval()
torch.cuda.empty_cache()  # Returns cached memory to CUDA allocator
```

**Expected Impact:**
- In-place operations: 5-15% memory reduction, depends on model.
- Fixing memory leaks (retained graphs): can prevent unbounded memory growth.
- `empty_cache()`: does not reduce memory usage during training but helps when transitioning between tasks.

---

## 4. Mixed Precision Training -- Complete Guide

### The Problem

Training in float32 uses more memory than necessary and does not fully utilize Tensor Cores on modern GPUs. Tensor Cores on A100 deliver 312 TFLOPS for TF32 but 624 TFLOPS for FP16 -- a 2x throughput gap.

### How to Measure

```python
import time
import torch

def benchmark_precision(model, input_shape, device, n_iterations=100):
    """Compare FP32 vs FP16 forward pass speed."""
    model = model.to(device)
    x = torch.randn(*input_shape, device=device)

    # FP32 benchmark
    model.float()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - start) / n_iterations

    # FP16 with autocast benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                _ = model(x)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / n_iterations

    print(f"FP32: {fp32_time*1000:.2f} ms/iter")
    print(f"FP16: {fp16_time*1000:.2f} ms/iter")
    print(f"Speedup: {fp32_time/fp16_time:.2f}x")
```

### The Solution: Complete Mixed Precision Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_mixed_precision(model, train_loader, val_loader, criterion,
                          optimizer, device, n_epochs=10, use_bf16=False):
    """
    Complete mixed precision training loop with annotations.

    Key components:
    1. autocast: automatically casts operations to lower precision where safe.
    2. GradScaler: scales loss to prevent gradient underflow in FP16.
       (Not needed for BF16 because BF16 has the same exponent range as FP32.)
    """
    # Choose dtype -- BF16 is preferred when available (A100+, TPU)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # GradScaler is only needed for FP16 (FP16 has limited exponent range)
    # For BF16, gradients rarely underflow, so scaling is unnecessary
    use_scaler = not use_bf16
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ---- Forward pass in mixed precision ----
            # autocast automatically handles which ops run in which precision:
            #   - matmul, conv2d, linear -> FP16/BF16 (Tensor Core accelerated)
            #   - softmax, layer_norm, loss -> FP32 (numerically sensitive)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # ---- Backward pass ----
            optimizer.zero_grad()

            # scaler.scale(loss) multiplies loss by a large factor (e.g., 65536)
            # This scales all gradients up, preventing FP16 underflow.
            # For BF16, scaler is disabled and this is a no-op.
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer) divides gradients by the scale factor.
            # This must happen before gradient clipping.
            scaler.unscale_(optimizer)

            # Now gradients are in their true magnitude -- safe to clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step() checks for inf/NaN gradients.
            # If found: skips optimizer.step() and reduces the scale factor.
            # If not: calls optimizer.step() normally.
            scaler.step(optimizer)

            # scaler.update() adjusts the scale factor:
            # - If no overflow for N consecutive steps, increase scale (more aggressive).
            # - If overflow occurred, scale was already reduced.
            scaler.update()

            total_loss += loss.detach().item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # ---- Validation ----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # autocast during inference too -- for consistent behavior and speed
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                for inputs, labels in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{n_epochs} -- Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model
```

### Which Operations Are Safe in FP16 vs Require FP32

```
Operations SAFE in FP16 (benefit from Tensor Cores):
  - torch.nn.Linear (matrix multiply)
  - torch.nn.Conv1d/Conv2d/Conv3d
  - torch.bmm, torch.matmul, torch.mm
  - torch.nn.functional.linear
  - torch.nn.GRU, torch.nn.LSTM (the matmul parts)

Operations that MUST remain in FP32 (numerically sensitive):
  - torch.nn.LayerNorm, torch.nn.BatchNorm*
  - torch.nn.functional.softmax
  - torch.nn.functional.cross_entropy
  - torch.nn.functional.binary_cross_entropy
  - torch.nn.functional.log_softmax
  - Reductions: torch.sum, torch.mean over large tensors
  - torch.exp, torch.log (can overflow/underflow easily)
  - Loss functions in general

autocast handles this automatically. You do NOT need to manually cast tensors.
```

### The Loss Scaling Mechanism -- Why It Is Necessary

```
The FP16 gradient underflow problem:
  - FP16 minimum positive normal: ~6.1e-5
  - FP16 minimum subnormal: ~5.96e-8
  - Many gradients during training are in the range 1e-5 to 1e-7.
  - In FP16, these become zero. The model stops learning.

The fix -- dynamic loss scaling:
  1. Multiply loss by a large scale factor S (start with S = 2^16 = 65536).
  2. All gradients are now S times larger (chain rule).
  3. After backward(), divide all gradients by S to get true values.
  4. If any gradient is inf/NaN (overflow): skip this step, set S = S / 2.
  5. If no overflow for 2000 consecutive steps: set S = S * 2.

This is what GradScaler does automatically.

Why BF16 does not need this:
  - BF16 has 8 exponent bits, same as FP32.
  - BF16 range: ~1e-38 to ~3e38 (same as FP32).
  - Gradients of 1e-7 are representable. No underflow.
  - BF16 has less precision (7 mantissa bits vs 23 for FP32), but range is what
    matters for gradient flow.
```

### Expected Impact

```
Typical results for mixed precision training:

                     FP32        FP16 (AMP)     BF16 (AMP)
Training speed:      1.0x        1.5-2.0x       1.5-2.0x
GPU memory:          1.0x        0.5-0.7x       0.5-0.7x
Accuracy:            baseline    within 0.1%    within 0.1%
Tensor Core usage:   no          yes            yes

The speedup comes from:
  1. Tensor Cores operating on FP16/BF16 at 2x throughput.
  2. Halved memory bandwidth for activations (half the bytes to read/write).
  3. Reduced GPU memory allows larger batch sizes.

When to use which:
  - FP16 + GradScaler: works on all GPUs with Tensor Cores (V100+).
  - BF16: preferred on A100, H100, and newer GPUs. Simpler (no scaler needed).
  - FP32: only when numerical precision is critical (rare in practice).
```

---

## 5. Post-Training Quantization

### The Problem

Your trained model is too slow for inference. You need faster inference without retraining.

### How to Measure

```python
import time
import torch

def benchmark_inference(model, input_tensor, n_iterations=1000, warmup=100):
    """Benchmark model inference speed."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Benchmark
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(input_tensor)
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / n_iterations * 1000
    return avg_ms
```

### The Solution: Dynamic Quantization

```python
import torch
import torch.nn as nn
import torch.quantization

def apply_dynamic_quantization(model):
    """
    Dynamic quantization: weights are quantized to INT8 at load time.
    Activations are quantized on-the-fly during inference.

    Best for: models dominated by Linear layers (Transformers, LSTMs).
    No calibration data needed.
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
        dtype=torch.qint8,
    )
    return quantized_model

# Usage and benchmark
model_fp32 = MyModel()
model_fp32.load_state_dict(torch.load("model.pth"))
model_fp32.eval()

model_int8 = apply_dynamic_quantization(model_fp32)

# Compare model sizes
import os
torch.save(model_fp32.state_dict(), "model_fp32.pth")
torch.save(model_int8.state_dict(), "model_int8.pth")
size_fp32 = os.path.getsize("model_fp32.pth") / 1024 / 1024
size_int8 = os.path.getsize("model_int8.pth") / 1024 / 1024
print(f"FP32 model size: {size_fp32:.1f} MB")
print(f"INT8 model size: {size_int8:.1f} MB")
print(f"Compression: {size_fp32/size_int8:.1f}x")
```

### The Solution: Static Quantization

```python
import torch
import torch.nn as nn
from torch.quantization import (
    get_default_qconfig, prepare, convert, QuantStub, DeQuantStub
)

class QuantizableModel(nn.Module):
    """Model wrapper with quantization stubs."""
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)      # Quantize input
        x = self.model(x)
        x = self.dequant(x)    # Dequantize output
        return x

def apply_static_quantization(model, calibration_loader, n_calibration_batches=100):
    """
    Static quantization: both weights and activations are quantized.
    Activation ranges are determined by running calibration data.

    Better than dynamic because activation quantization params are precomputed.
    Requires representative calibration data.
    """
    # Wrap model with quant/dequant stubs
    quantizable = QuantizableModel(model)
    quantizable.eval()

    # Set quantization config
    quantizable.qconfig = get_default_qconfig('x86')  # or 'qnnpack' for ARM

    # Insert observers that record activation ranges
    prepared = prepare(quantizable)

    # Run calibration data through the model
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_loader):
            if i >= n_calibration_batches:
                break
            prepared(inputs)

    # Convert to quantized model using observed ranges
    quantized = convert(prepared)

    return quantized
```

### Expected Impact

```
Typical quantization results (CPU inference, Transformer-based model):

                     FP32      Dynamic INT8    Static INT8
Inference speed:     1.0x      1.5-2.0x        2.0-3.0x
Model size:          1.0x      0.25x           0.25x
Accuracy:            baseline  -0.1 to -1.0%   -0.1 to -0.5%
Calibration needed:  no        no              yes (100+ samples)

Notes:
  - INT8 speedup is primarily on CPU. GPU INT8 requires specific kernel support.
  - Accuracy degradation varies by model. Well-trained models degrade less.
  - Static quantization is generally better than dynamic because activation ranges
    are precomputed, avoiding per-inference overhead.
  - For best results with > 1% accuracy loss, consider quantization-aware training.
```

---

## 6. DataLoader Optimization Checklist

### The Problem

The GPU is idle between training steps because data loading cannot keep up. This is the most common bottleneck in training pipelines and the easiest to fix.

### How to Measure

```python
import time

# Step 1: Measure raw DataLoader throughput
def measure_dl_throughput(dataset, batch_size, num_workers, pin_memory=False,
                          persistent_workers=False, prefetch_factor=2, n_batches=200):
    """Measure DataLoader throughput for a specific configuration."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    # Warmup
    data_iter = iter(loader)
    for _ in range(min(10, n_batches)):
        next(data_iter)

    start = time.perf_counter()
    for i, _ in enumerate(loader):
        if i >= n_batches:
            break
    elapsed = time.perf_counter() - start

    throughput = n_batches * batch_size / elapsed
    return throughput

# Step 2: Sweep num_workers
for nw in [0, 1, 2, 4, 8, 12, 16]:
    tp = measure_dl_throughput(dataset, batch_size=64, num_workers=nw)
    print(f"  num_workers={nw:2d}: {tp:.0f} samples/sec")
```

### The Solution: Systematic Optimization

Each step below is independent. Apply them in order, measuring after each.

#### Step 1: Set num_workers > 0

```python
# Before (default):
loader = DataLoader(dataset, batch_size=64, shuffle=True)
# num_workers defaults to 0: all loading happens in the main process.

# After:
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
# Rule of thumb: start with 4 * num_GPUs, then sweep.
```
**Typical improvement: 2-5x for image datasets.**

#### Step 2: Enable pin_memory

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,
                    pin_memory=True)

# And use non_blocking transfer:
for inputs, labels in loader:
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
```
**Typical improvement: 10-30% for GPU training.**

Pin memory allocates tensors in page-locked CPU memory, enabling asynchronous (overlapped) CPU-to-GPU transfers.

#### Step 3: Enable persistent_workers

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,
                    pin_memory=True, persistent_workers=True)
```
**Typical improvement: eliminates per-epoch worker startup cost (1-10 seconds per epoch).**

Without this, worker processes are spawned and destroyed every epoch. With large datasets or heavy initialization, this overhead adds up.

#### Step 4: Tune prefetch_factor

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,
                    pin_memory=True, persistent_workers=True, prefetch_factor=4)
```
**Typical improvement: 5-15% when workers are fast but there is a gap between batches.**

Default is 2 (each worker prefetches 2 batches). Increasing to 3-4 can smooth out variance. Cost: more CPU memory.

#### Step 5: Move augmentation to GPU

```python
import kornia.augmentation as K

# Before: CPU augmentation in __getitem__
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# After: minimal CPU transform, heavy augmentation on GPU
cpu_transform = transforms.Compose([
    transforms.ToTensor(),  # Only convert to tensor on CPU
])

gpu_transform = K.AugmentationSequential(
    K.RandomCrop((32, 32), padding=4),
    K.RandomHorizontalFlip(p=0.5),
    K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    data_keys=["input"],
)

# In training loop:
for inputs, labels in loader:
    inputs = inputs.to(device)
    inputs = gpu_transform(inputs)  # Augmentation happens on GPU
```
**Typical improvement: 1.5-3x when CPU augmentation is the bottleneck.**

#### Step 6: Use efficient data format

```python
# Convert dataset to LMDB for fast random access
import lmdb
import pickle

def create_lmdb_dataset(dataset, lmdb_path, map_size=1e11):
    """Convert a PyTorch dataset to LMDB format."""
    env = lmdb.open(lmdb_path, map_size=int(map_size))
    with env.begin(write=True) as txn:
        for i in range(len(dataset)):
            img, label = dataset[i]
            txn.put(str(i).encode(), pickle.dumps((img, label)))
        txn.put(b'length', str(len(dataset)).encode())
    env.close()

class LMDBDataset(torch.utils.data.Dataset):
    """Dataset that reads from LMDB."""
    def __init__(self, lmdb_path, transform=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = int(txn.get(b'length').decode())
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode()))
        img, label = data
        if self.transform:
            img = self.transform(img)
        return img, label
```
**Typical improvement: 1.5-3x for datasets stored as many small files (e.g., ImageNet).**

#### Step 7: Precompute expensive preprocessing

If your preprocessing is deterministic (e.g., tokenization, feature extraction), compute it once and save the results. Do not redo it every epoch.

```python
# Precompute and save
processed_data = []
for img, label in raw_dataset:
    processed = expensive_preprocessing(img)
    processed_data.append((processed, label))
torch.save(processed_data, "preprocessed_data.pt")

# During training, load the preprocessed data
data = torch.load("preprocessed_data.pt")
```
**Typical improvement: depends on preprocessing cost. Can be 10x for heavy preprocessing.**

### Expected Impact: Cumulative Results

```
Typical cumulative speedup (ImageNet-style dataset, single GPU):

Optimization                   Throughput     Cumulative Speedup
─────────────────────────────────────────────────────────────────
Baseline (nw=0, no pin)        200 img/s      1.0x
+ num_workers=4                700 img/s      3.5x
+ pin_memory                   850 img/s      4.3x
+ persistent_workers           870 img/s      4.4x
+ prefetch_factor=4            900 img/s      4.5x
+ non_blocking transfer        920 img/s      4.6x
+ GPU augmentation             1200 img/s     6.0x
+ LMDB data format             1500 img/s     7.5x

These numbers are illustrative. Your mileage will vary based on:
  - Dataset size and image resolution
  - Augmentation complexity
  - Storage type (HDD vs SSD vs NVMe)
  - CPU core count
  - GPU speed (a faster GPU makes data bottlenecks more visible)
```

---

## 7. torch.compile Practical Guide

### The Problem

Eager-mode PyTorch dispatches each operation individually, incurring Python overhead and kernel launch overhead for every op. Memory-bound operations (element-wise, normalization) suffer because they read and write intermediate tensors to GPU memory unnecessarily.

### How to Measure

```python
import time
import torch

def benchmark_compiled_vs_eager(model, input_tensor, device, n_iterations=200, warmup=50):
    """Compare eager vs compiled model speed."""
    model = model.to(device).eval()
    x = input_tensor.to(device)

    # Eager benchmark
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(x)
    torch.cuda.synchronize()
    eager_time = (time.perf_counter() - start) / n_iterations

    # Compiled benchmark
    compiled_model = torch.compile(model)
    with torch.no_grad():
        # First few calls trigger compilation -- longer warmup needed
        for _ in range(warmup + 10):
            _ = compiled_model(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = compiled_model(x)
    torch.cuda.synchronize()
    compiled_time = (time.perf_counter() - start) / n_iterations

    print(f"Eager:    {eager_time*1000:.2f} ms/iter")
    print(f"Compiled: {compiled_time*1000:.2f} ms/iter")
    print(f"Speedup:  {eager_time/compiled_time:.2f}x")
    return eager_time, compiled_time
```

### The Solution

#### Basic Usage

```python
import torch

model = MyModel()
model = model.to(device)

# Option 1: Compile the whole model (recommended starting point)
compiled_model = torch.compile(model)

# Option 2: Compile with specific mode
compiled_model = torch.compile(model, mode="reduce-overhead")  # For small models
compiled_model = torch.compile(model, mode="max-autotune")     # For best runtime speed

# Option 3: Compile individual functions
@torch.compile
def custom_loss(pred, target):
    return torch.nn.functional.cross_entropy(pred, target) + 0.1 * pred.norm()

# Training loop is identical -- compiled model is a drop-in replacement
for inputs, labels in train_loader:
    outputs = compiled_model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### Compilation Modes

```
mode="default":
  - Good balance of compile time and runtime.
  - Compile time: 30s - 2min for typical models.
  - Runtime speedup: 1.3-2x.
  - Best for: most use cases.

mode="reduce-overhead":
  - Uses CUDA graphs to reduce kernel launch overhead.
  - Compile time: similar to default.
  - Runtime speedup: 1.5-3x, especially for small/medium models.
  - Best for: models where kernel launch overhead dominates (small batch sizes,
    many small operations).
  - Limitation: CUDA graphs require static input shapes.

mode="max-autotune":
  - Tries many kernel configurations and picks the fastest.
  - Compile time: 5-30 minutes (!) for large models.
  - Runtime speedup: best possible, often 2-3x.
  - Best for: production inference, benchmarking.
```

#### Identifying and Fixing Graph Breaks

Graph breaks are the primary obstacle to getting good performance from `torch.compile`. A graph break splits your model into multiple independently compiled subgraphs, reducing optimization opportunities.

```bash
# Identify graph breaks by setting the environment variable:
TORCH_LOGS="graph_breaks" python train.py

# Or for more verbose output:
TORCH_LOGS="graph_breaks,recompiles" python train.py
```

Common causes and fixes:

```python
# CAUSE 1: print statements in the model
class BadModel(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        print(f"Shape after layer1: {x.shape}")  # GRAPH BREAK!
        x = self.layer2(x)
        return x

# FIX: remove prints, or use torch._dynamo.config.verbose = True for debugging

# CAUSE 2: data-dependent control flow
class BadModel2(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        if x.sum().item() > 0:  # GRAPH BREAK! .item() forces CPU sync
            x = self.branch_a(x)
        else:
            x = self.branch_b(x)
        return x

# FIX: use torch.where or torch.cond for data-dependent branching
class FixedModel2(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        a = self.branch_a(x)
        b = self.branch_b(x)
        mask = (x.sum() > 0).float()
        return mask * a + (1 - mask) * b

# CAUSE 3: calling .numpy() or unsupported operations
class BadModel3(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        x_np = x.detach().cpu().numpy()  # GRAPH BREAK!
        # ... some numpy operation ...
        return x

# FIX: stay in PyTorch tensors throughout the forward pass

# CAUSE 4: dynamic shapes (different input shapes across iterations)
# This causes recompilation, not a graph break per se, but equally damaging.
# FIX: pad inputs to fixed sizes, or use dynamic=True:
compiled_model = torch.compile(model, dynamic=True)
```

#### Enforcing Full Graph Compilation

```python
# During development, use fullgraph=True to catch graph breaks as errors:
compiled_model = torch.compile(model, fullgraph=True)
# This will raise an error if any graph break is detected.
# Remove fullgraph=True for production if you cannot eliminate all breaks.
```

#### The Compilation Cache

```python
# Compiled kernels are cached to avoid recompilation:
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/path/to/cache"

# The cache persists across runs. First run compiles (slow), subsequent runs
# load from cache (fast).

# To inspect generated code:
# TORCH_LOGS="output_code" python train.py
# This prints the Triton/C++ kernels that Inductor generates.
```

### Models That Work Well With torch.compile

```
Works very well (1.5-3x speedup):
  - Standard architectures: ResNet, Vision Transformer, BERT, GPT-2
  - Models with simple control flow
  - Models with many element-wise operations (benefits from fusion)
  - Inference workloads (static shapes, no training-specific dynamism)

Works okay (1.2-1.5x speedup):
  - Models with some dynamic shapes
  - Models with conditional computation
  - Training loops (some overhead from gradient tracking)

May have issues:
  - Models with heavy use of custom Python logic in forward()
  - Models that call external libraries (not PyTorch) in forward()
  - Models with highly dynamic control flow
  - Very small models (compilation overhead dominates)
```

### Expected Impact

```
Typical speedups from torch.compile:

Model               Batch Size   Eager    Compiled   Speedup
──────────────────────────────────────────────────────────────
ResNet-50 (train)     64         45ms     28ms       1.6x
ResNet-50 (infer)     64         12ms     6ms        2.0x
ViT-B/16 (train)      32         68ms     38ms       1.8x
BERT-base (train)     32         52ms     31ms       1.7x
GPT-2 (infer)         1          25ms     11ms       2.3x
Small MLP (infer)     256        0.8ms    0.3ms      2.7x

Notes:
  - Speedups are GPU-dependent. Larger speedups on newer GPUs.
  - First iteration is slow (compilation). Amortized over training.
  - reduce-overhead mode gives best results for inference.
  - max-autotune gives best results when you can afford compile time.
```

---

## 8. Operator Fusion and Custom Kernels

### The Problem

Memory-bound operations -- element-wise activations, normalization, residual additions -- are limited by GPU memory bandwidth, not compute. Each operation reads its input from GPU memory and writes its output back. A sequence of N element-wise operations does N memory round trips.

### How Fusion Helps

```
Without fusion (3 separate kernels):
  Kernel 1: Read x from memory, compute bias_add, write result to memory
  Kernel 2: Read result from memory, compute batch_norm, write result to memory
  Kernel 3: Read result from memory, compute relu, write result to memory
  Total memory traffic: 6 reads + 3 writes = 9 memory operations

With fusion (1 kernel):
  Kernel 1: Read x from memory, compute bias_add + batch_norm + relu, write result
  Total memory traffic: 1 read + 1 write = 2 memory operations

For memory-bound operations, this is a 4.5x reduction in memory traffic.
torch.compile does this automatically for many common patterns.
```

### Writing Custom Triton Kernels

When `torch.compile` does not fuse what you need, or when you need a custom fused operation, Triton lets you write GPU kernels in Python.

```python
import torch
import triton
import triton.language as tl

# Example: Fused dropout + residual add + layer norm
# Three operations that are commonly applied sequentially.

@triton.jit
def fused_add_dropout_layernorm_kernel(
    x_ptr, residual_ptr, output_ptr, weight_ptr, bias_ptr,
    n_cols, eps, dropout_p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: output = LayerNorm(dropout(x, p) + residual)"""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load inputs
    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)

    # Dropout
    random = tl.rand(seed, row_idx * n_cols + col_offsets)
    dropout_mask = random > dropout_p
    x = tl.where(dropout_mask, x / (1.0 - dropout_p), 0.0)

    # Residual add
    x = x + residual

    # Layer norm
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    x_norm = x_centered / tl.sqrt(var + eps)

    # Scale and shift
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    output = x_norm * weight + bias

    # Store
    tl.store(output_ptr + row_idx * n_cols + col_offsets, output, mask=mask)


def fused_add_dropout_layernorm(x, residual, weight, bias, dropout_p=0.1, eps=1e-5):
    """Python wrapper for the Triton kernel."""
    assert x.shape == residual.shape
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    seed = torch.randint(0, 2**31, (1,)).item()

    fused_add_dropout_layernorm_kernel[(n_rows,)](
        x, residual, output, weight, bias,
        n_cols, eps, dropout_p, seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

### A Simple Triton Kernel -- Fused GELU

```python
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # GELU approximation
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    output = 0.5 * x * (1.0 + tl.libdevice.tanh(inner))

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_gelu(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

# Benchmark
x = torch.randn(4096, 4096, device='cuda')
import time

# PyTorch GELU
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    y1 = torch.nn.functional.gelu(x)
torch.cuda.synchronize()
pytorch_time = time.perf_counter() - start

# Triton GELU
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    y2 = triton_gelu(x)
torch.cuda.synchronize()
triton_time = time.perf_counter() - start

print(f"PyTorch GELU: {pytorch_time:.3f}s")
print(f"Triton GELU:  {triton_time:.3f}s")
print(f"Max diff: {(y1 - y2).abs().max().item():.2e}")
```

### Flash Attention: The Gold Standard

```
Standard attention:
  Q, K, V: (batch, heads, seq_len, head_dim)

  Step 1: S = Q @ K^T            -> (batch, heads, seq_len, seq_len)  -- O(n^2) memory!
  Step 2: P = softmax(S / sqrt(d)) -> (batch, heads, seq_len, seq_len)  -- O(n^2) memory!
  Step 3: O = P @ V              -> (batch, heads, seq_len, head_dim)

  Memory: O(n^2) for the attention matrix S.
  For n=4096, heads=32, batch=8, FP16: 8 * 32 * 4096 * 4096 * 2 = 8 GB!

Flash Attention (Dao et al., 2022):
  - Tiles Q, K, V into blocks that fit in GPU SRAM (shared memory).
  - Computes attention block-by-block using the "online softmax" trick.
  - Never materializes the full n x n attention matrix.

  Memory: O(n) -- only the output and log-sum-exp accumulators.
  Speed: 2-4x faster due to reduced HBM (GPU DRAM) access.

  The key insight: reading/writing from HBM is slow. SRAM is fast but small.
  Flash Attention restructures the computation to maximize SRAM reuse.

Usage in PyTorch:
  # PyTorch >= 2.0 automatically uses Flash Attention when possible
  from torch.nn.functional import scaled_dot_product_attention

  output = scaled_dot_product_attention(
      query, key, value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=True,  # For autoregressive models
  )
  # PyTorch selects the best backend: Flash Attention, Memory-Efficient Attention,
  # or the math fallback, depending on input shapes and GPU capability.
```

### Expected Impact

```
Operator fusion (via torch.compile):
  - Element-wise operations: 1.5-3x speedup.
  - Batch norm + activation: 1.3-1.8x speedup.
  - Entire model: 1.3-2x overall (the compute-bound parts do not speed up).

Custom Triton kernels:
  - For specific fused operations: 1.5-4x over unfused PyTorch.
  - Diminishing returns if torch.compile already fuses the pattern.

Flash Attention:
  - Speed: 2-4x faster than standard attention.
  - Memory: O(n) vs O(n^2). Enables 4-16x longer sequences.
  - This is the single most impactful kernel optimization for Transformer models.
```

---

## 9. End-to-End Optimization Strategy

### The Problem

You have a training pipeline. It is slow. You want to make it fast. But there are a dozen possible optimizations and you cannot try them all. What order do you apply them in?

### The Strategy

The optimization workflow, in order of priority:

```
1. PROFILE (always first)
   - Run the profiling cookbook from Section 1.
   - Identify the primary bottleneck: data, CPU, GPU compute, memory.

2. FIX THE DATA PIPELINE (if data-bottlenecked)
   - Apply the DataLoader checklist from Section 6.
   - Target: GPU utilization > 90% during training.
   - Expected impact: 2-5x for naive pipelines.

3. ENABLE MIXED PRECISION (almost always)
   - Add autocast + GradScaler (or autocast + bfloat16).
   - Expected impact: 1.5-2x speed, 30-50% memory reduction.
   - Risk: essentially zero for standard architectures.

4. APPLY torch.compile (almost always)
   - compiled_model = torch.compile(model)
   - Debug graph breaks if speedup is less than expected.
   - Expected impact: 1.3-2x speed.

5. OPTIMIZE MEMORY (if memory-constrained)
   - Gradient checkpointing, gradient accumulation, Flash Attention.
   - Allows larger batch sizes or longer sequences.
   - Expected impact: 2-3x more batch size within same memory.

6. TUNE BATCH SIZE (after memory optimizations)
   - Larger batch sizes = better GPU utilization (up to a point).
   - Use gradient accumulation to achieve desired effective batch size.
   - Watch for generalization impact at very large batch sizes.

7. CUSTOM KERNELS (last resort, high effort)
   - Write Triton kernels for specific bottleneck operations.
   - Only if profiling shows a specific operation is the bottleneck
     and torch.compile does not optimize it adequately.
   - Expected impact: 1.5-4x for the specific operation.
```

### Combined Impact

```
Typical combined improvement from all techniques:

Technique                 Speedup     Cumulative
──────────────────────────────────────────────────
Baseline                  1.0x        1.0x
DataLoader optimization   2.5x        2.5x
Mixed precision           1.7x        4.3x
torch.compile             1.5x        6.4x
Larger batch (memory opt) 1.3x        8.3x
──────────────────────────────────────────────────
Total:                                ~8x

This is a realistic combined improvement for a naive training pipeline.
Your specific numbers will vary. Some pipelines are already well-optimized
in some areas but not others. The key is to profile and target the actual
bottleneck at each step.
```

### The Performance Engineering Mindset

Performance optimization is a discipline, not a one-time activity. Every time you change your model architecture, dataset, or training configuration, the bottleneck may shift. The engineer who profiles regularly will always outperform the one who "optimizes" by cargo-culting.

Rules to internalize:
1. **Measure before optimizing.** Always.
2. **Optimize the bottleneck.** A 2x speedup to a non-bottleneck operation gives 0x overall improvement.
3. **One change at a time.** Otherwise, you cannot attribute the improvement.
4. **Re-profile after each change.** The bottleneck shifts.
5. **Know your hardware.** Peak FLOPS, memory bandwidth, CPU cores, disk speed. These set the ceiling.
6. **The goal is not the fastest possible code. The goal is the fastest possible experimental iteration.** Time saved on training is time available for trying the next idea.

---

## Appendix A: Quick Reference -- GPU Specifications

```
GPU          FP32 TFLOPS   FP16 TFLOPS   BF16 TFLOPS   Memory    Bandwidth
─────────────────────────────────────────────────────────────────────────────
RTX 3090       35.6          71            35.6          24 GB     936 GB/s
RTX 4090       82.6          165           82.6          24 GB     1008 GB/s
A100 (80G)     19.5          312 (TF32)    312           80 GB     2039 GB/s
H100 (80G)     67            989 (TF32)    989           80 GB     3350 GB/s
V100 (32G)     15.7          125           N/A           32 GB     900 GB/s

Note: "TFLOPS" for Tensor Core operations (FP16/BF16/TF32) includes sparsity
and tensor core acceleration. Actual achievable throughput is typically 60-80%
of peak.

The roofline crossover (arithmetic intensity at which compute becomes the
bottleneck rather than memory):
  A100: 312 / 2.039 = 153 FLOP/byte (TF32)
  H100: 989 / 3.350 = 295 FLOP/byte (TF32)

Operations with arithmetic intensity below this are memory-bound.
Operations above it are compute-bound.
```

## Appendix B: Quick Reference -- Common Operation FLOP Counts

```
Operation                          FLOPs (approximate)
────────────────────────────────────────────────────────────
Linear(in, out), batch B:         2 * B * in * out
Conv2d(C_in, C_out, k, k):       2 * B * C_out * H_out * W_out * C_in * k * k
BatchNorm(C):                     ~5 * B * C * H * W  (mean, var, norm, scale, shift)
LayerNorm(d):                     ~5 * B * seq_len * d
Self-Attention(d, n_heads, seq):  ~8 * B * seq^2 * d  (QKV proj + attn + output proj)
GELU/ReLU/Sigmoid(N):            ~N to ~10N depending on activation
Softmax(N):                       ~5 * N  (exp, sum, divide)
```

## Appendix C: Debugging Checklist for Common Performance Issues

```
Symptom: GPU utilization < 50%
  -> Check DataLoader throughput.
  -> Check for .item() or .cpu() in training loop.
  -> Check for excessive printing or logging.

Symptom: OOM (Out of Memory)
  -> Reduce batch size.
  -> Enable gradient checkpointing.
  -> Enable mixed precision.
  -> Use gradient accumulation instead of large batches.
  -> Check for memory leaks (tensors appended to lists without .detach()).

Symptom: Training speed does not improve with torch.compile
  -> Check for graph breaks: TORCH_LOGS="graph_breaks"
  -> Ensure input shapes are static (or use dynamic=True).
  -> Try mode="reduce-overhead" for small models.

Symptom: Mixed precision training produces NaN loss
  -> GradScaler is handling this (it skips steps with NaN).
  -> If persistent: reduce learning rate, add gradient clipping.
  -> Check if custom loss function needs explicit FP32 casting.
  -> Try bfloat16 instead of float16.

Symptom: Training is slower with more num_workers
  -> CPU contention. Reduce num_workers.
  -> Check if workers use too much memory (OOM on CPU).
  -> On Windows, multiprocessing overhead can dominate for small datasets.

Symptom: Model accuracy degrades with quantization
  -> Try static quantization with more calibration data.
  -> Try quantization-aware training.
  -> Check if specific layers are quantization-sensitive (skip those).
  -> Use per-channel quantization instead of per-tensor.
```
