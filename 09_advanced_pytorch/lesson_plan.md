# Week 14 — Advanced PyTorch

## Module Overview

You can write training loops, build models, and use optimizers. That makes you competent. This week makes you dangerous. We are going to open the hood on PyTorch and understand the machinery that most practitioners never touch: custom autograd, hooks, distributed training, mixed precision, compilation, and production deployment. These are the skills that separate someone who uses PyTorch from someone who bends it to their will.

After this week, there should be no black boxes left.

---

## Session 1: Custom Autograd Functions

**Duration:** 2.5 hours

### Objectives

By the end of this session, the apprentice will be able to:

1. Explain when and why a custom backward pass is necessary, and when it is not.
2. Implement a custom autograd Function with correct forward and backward methods.
3. Use `ctx.save_for_backward` correctly, understanding what can and cannot be saved.
4. Implement a Straight-Through Estimator for non-differentiable operations.
5. Verify custom gradients using `torch.autograd.gradcheck`.
6. Implement functions that support higher-order gradients (double backward).

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| Lecture | 30 min | Why custom autograd exists. The `torch.autograd.Function` class. The contract between `forward` and `backward`. When you need this vs when you do not. |
| Live Coding | 40 min | Implement a custom activation function (Swish with learnable beta). Walk through `ctx`, `save_for_backward`, `@staticmethod` requirements. Run `gradcheck`. |
| Lecture | 20 min | The Straight-Through Estimator — what it is, why it matters (quantization, discrete decisions, binary networks). The gradient of a step function is zero everywhere, so we lie about it. |
| Live Coding | 30 min | Implement STE. Train a tiny network with binary activations using STE. Show that it actually works. |
| Exercise | 20 min | Implement a custom loss function with a non-trivial backward pass (e.g., a margin-based ranking loss). |
| Discussion | 10 min | When you actually need custom autograd in research. Double backward for meta-learning. Performance considerations. |

### Live-Coding Exercises

1. **Custom Swish Function**: Implement `forward` and `backward` for `x * sigmoid(beta * x)`. Compare output and gradients against `torch.nn.SiLU`. Verify with `gradcheck`.
2. **Straight-Through Estimator**: Implement a hard threshold in forward, identity gradient in backward. Train a network with binary weights.
3. **Custom Loss with Backward**: Implement a loss function where the backward pass involves a conditional branch (e.g., asymmetric loss).

### Key Takeaways

- Custom autograd is the escape hatch for when automatic differentiation cannot express what you need. It is not a performance optimization — it is a correctness tool.
- `ctx.save_for_backward` exists because PyTorch manages memory for you. If you stash tensors in `self` or a closure, you break the memory management contract and create leaks.
- `gradcheck` is not optional. Every custom Function must be verified numerically before you trust it.
- The Straight-Through Estimator is a principled hack — it is wrong, but it is useful. Understand when that tradeoff is acceptable.

---

## Session 2: Hooks, Debugging, and Internals

**Duration:** 2.5 hours

### Objectives

By the end of this session, the apprentice will be able to:

1. Register and use forward hooks, backward hooks, and full-backward hooks on modules and tensors.
2. Use hooks for feature extraction, gradient inspection, gradient modification, and activation logging.
3. Explain the difference between module hooks and tensor hooks.
4. Understand what the PyTorch dispatcher does at a high level.
5. Explain the difference between TorchScript and `torch.compile`, and when each is appropriate.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| Lecture | 25 min | What hooks are and why they exist. Forward hooks vs backward hooks. Module hooks vs tensor hooks. The hook signature. When hooks fire in the computation graph. |
| Live Coding | 35 min | Five hook patterns: feature extraction, gradient visualization, per-layer gradient clipping, activation statistics, gradient reversal. |
| Exercise | 20 min | Build a gradient flow visualizer — hook into every layer, collect gradient norms, plot them. |
| Lecture | 20 min | PyTorch internals: the dispatcher, operator registration, how `torch.compile` works (dynamo captures the graph, inductor generates code). TorchScript vs `torch.compile`. |
| Live Coding | 20 min | Compile a model, inspect what happens, show a case where it speeds things up and a case where it breaks. |
| Discussion | 10 min | When to use hooks in production vs only in debugging. The overhead of hooks. Cleanup with `handle.remove()`. |

### Live-Coding Exercises

1. **Feature Extractor**: Use forward hooks to pull intermediate representations from a pretrained ResNet. Show that you get the same result as manually slicing the model.
2. **Gradient Flow Plot**: Register backward hooks on every layer. Collect gradient norms. Plot them after one training step. Identify a vanishing gradient.
3. **Gradient Reversal Layer**: Implement domain adaptation's gradient reversal trick using a backward hook — negate gradients flowing through a specific layer.
4. **Activation Statistics Logger**: Log mean, std, min, max of activations at every layer during training. Detect a dead ReLU.
5. **torch.compile Demo**: Compile a transformer block. Benchmark compiled vs eager. Show a dynamic-shape case that fails and how to fix it with `dynamic=True`.

### Key Takeaways

- Hooks are the X-ray vision of PyTorch. They let you observe and modify the forward and backward passes without changing the model code.
- Always remove hooks when you are done. A forgotten hook is a memory leak and a performance drain.
- `torch.compile` is the future of PyTorch performance. It is not magic — it captures a computation graph and generates optimized code. Understanding its limitations is as important as understanding its strengths.
- TorchScript is legacy for most use cases. Prefer `torch.compile` for performance and ONNX for export.

---

## Session 3: Distributed Training

**Duration:** 2.5 hours

### Objectives

By the end of this session, the apprentice will be able to:

1. Explain data parallelism, model parallelism, and pipeline parallelism — when each is appropriate.
2. Explain why `DistributedDataParallel` is superior to `DataParallel`.
3. Write a complete distributed training script that works with `torchrun`.
4. Use `DistributedSampler` correctly and understand its interaction with shuffling and epochs.
5. Save and load checkpoints correctly in a distributed setting.
6. Describe FSDP and when it is necessary.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| Lecture | 30 min | Why distribute: data-parallel (same model, split data) vs model-parallel (split model). Ring-allreduce algorithm. Why `DataParallel` is bad (GIL, asymmetric memory, no multi-node). `DistributedDataParallel` architecture. |
| Live Coding | 45 min | Write a complete DDP training script from scratch. Initialize process group. Wrap model. Use DistributedSampler. Train. Save/load checkpoints from rank 0 only. Launch with torchrun. |
| Lecture | 15 min | FSDP: sharding optimizer states, gradients, and parameters across GPUs. When you need it (model does not fit on one GPU). Pipeline parallelism concept. |
| Exercise | 25 min | Take a single-GPU training script and convert it to DDP. Test by simulating 2 processes on 1 GPU. |
| Discussion | 10 min | Multi-GPU best practices. Communication overhead. Gradient compression. When not to distribute. |

### Live-Coding Exercises

1. **DDP From Scratch**: Write a full distributed training script. Initialize `nccl` (or `gloo` for CPU), wrap model in DDP, use `DistributedSampler`, train for a few epochs. Show that all processes produce the same final model.
2. **Checkpoint Management**: Save a checkpoint from rank 0 only. Load it on all ranks. Handle the `map_location` correctly.
3. **Single-GPU to Multi-GPU Conversion**: Take an existing training script. Add the 10 lines needed to make it distributed. Verify correctness.

### Key Takeaways

- `DistributedDataParallel` is always the right choice over `DataParallel`. There is no scenario where `DataParallel` is preferable.
- The distributed sampler ensures each process sees a unique subset of data. You must call `sampler.set_epoch(epoch)` each epoch or shuffling breaks.
- Only save checkpoints from rank 0. All ranks have identical parameters after each step due to allreduce.
- FSDP is for when your model does not fit on a single GPU. If it fits, DDP is simpler and sufficient.
- Distributed training introduces an entire class of bugs that only appear at scale. Test your distributed code with 2 processes on 1 GPU before scaling up.

---

## Session 4: Advanced Training Techniques

**Duration:** 2.5 hours

### Objectives

By the end of this session, the apprentice will be able to:

1. Implement gradient accumulation to simulate large batch sizes.
2. Use mixed precision training with `torch.cuda.amp` and explain why loss scaling is necessary.
3. Apply gradient checkpointing to reduce memory usage at the cost of compute.
4. Use `torch.compile` to accelerate training and understand its tradeoffs.
5. Implement training techniques from papers: EMA, label smoothing, stochastic depth.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| Lecture | 20 min | Gradient accumulation: why large batches matter, how to simulate them. Mixed precision: why float16 is faster, why it underflows, how loss scaling fixes it. Gradient checkpointing: the time-memory tradeoff. |
| Live Coding | 40 min | Implement gradient accumulation. Implement mixed precision training with autocast and GradScaler. Implement gradient checkpointing on a deep network. Measure memory before/after. |
| Lecture | 15 min | torch.compile deep dive: dynamo traces Python bytecode, inductor generates Triton/C++, when it helps (compute-bound), when it hurts (overhead on small models). |
| Live Coding | 25 min | Compile a model. Benchmark. Show compilation errors with dynamic control flow and how to fix them. |
| Exercise | 20 min | Implement EMA (exponential moving average of model weights), label smoothing, and stochastic depth. Combine with mixed precision in a single training loop. |
| Discussion | 10 min | How these techniques compose. The standard recipe at DeepMind: DDP + mixed precision + gradient accumulation + EMA. |

### Live-Coding Exercises

1. **Gradient Accumulation**: Train with effective batch size 256 using micro-batches of 32. Show that the final gradients are mathematically equivalent to a single batch of 256.
2. **Mixed Precision Training Loop**: Full training loop with `autocast`, `GradScaler`, `scaler.scale(loss).backward()`, `scaler.step()`, `scaler.update()`. Measure memory savings and speedup.
3. **Gradient Checkpointing**: Apply `torch.utils.checkpoint.checkpoint` to segments of a deep ResNet. Measure memory reduction and training time increase.
4. **Paper Techniques**: Implement EMA with configurable decay, label smoothing cross-entropy loss, stochastic depth in a ResBlock.

### Key Takeaways

- Gradient accumulation is free in terms of memory. It just takes more steps. Normalize your loss by the number of accumulation steps.
- Mixed precision gives you roughly 2x speedup and 2x memory savings on modern GPUs. Loss scaling is the only tricky part — `GradScaler` handles it automatically.
- Gradient checkpointing trades compute for memory. For very deep networks, it can be the difference between training and OOM.
- `torch.compile` is most effective on compute-bound operations. For small models or models with heavy data loading, the compilation overhead may not be worth it.
- EMA, label smoothing, and stochastic depth are standard tricks. Learn to implement them from the paper, not from a library.

---

## Session 5: Production PyTorch

**Duration:** 2.5 hours

### Objectives

By the end of this session, the apprentice will be able to:

1. Export models to ONNX format and explain why it matters.
2. Explain the difference between TorchScript tracing and scripting.
3. Apply post-training quantization (dynamic and static) to reduce model size and latency.
4. Use `torch.profiler` to identify performance bottlenecks.
5. Understand the landscape of custom kernel development (CUDA, Triton).

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| Lecture | 25 min | Model export landscape: ONNX (portable, wide runtime support), TorchScript (Python-free execution), torch.package (bundle everything). Quantization: why (4x smaller, 2-4x faster), how (dynamic, static, QAT). |
| Live Coding | 35 min | Export a model to ONNX. Run it with ONNX Runtime. Apply dynamic quantization. Apply static quantization with calibration. Compare sizes and latencies. |
| Lecture | 15 min | Profiling: torch.profiler, reading Chrome traces, identifying bottlenecks (CPU-bound vs GPU-bound vs memory-bound). Custom kernels: when you need them, Triton as an accessible alternative to raw CUDA. |
| Live Coding | 25 min | Profile a training step. Identify a bottleneck. Write a simple Triton kernel for a fused operation. Show the speedup. |
| Exercise | 20 min | Take a model through the full production pipeline: train, export to ONNX, quantize, profile, measure end-to-end latency. |
| Discussion | 10 min | The production ML stack at DeepMind. How research models become production models. The gap between a notebook and a deployed system. |

### Live-Coding Exercises

1. **ONNX Export**: Export a trained model. Load it with `onnxruntime`. Verify output matches PyTorch. Handle dynamic axes.
2. **Quantization Pipeline**: Apply dynamic quantization to a model. Then apply static quantization with calibration data. Compare file sizes and inference latency.
3. **Profiling**: Profile a training step with `torch.profiler`. Generate a Chrome trace. Identify the most expensive operations. Discuss optimization strategies.
4. **Triton Kernel** (demo): Write a simple fused kernel in Triton (e.g., fused add + ReLU). Benchmark against naive PyTorch.

### Key Takeaways

- ONNX is the lingua franca of model deployment. If your model cannot be exported to ONNX, you have a deployment problem.
- Quantization is the single most impactful optimization for inference. INT8 quantization gives you 4x size reduction with minimal accuracy loss for most models.
- Profiling before optimizing. Never guess where the bottleneck is. `torch.profiler` will tell you.
- Custom CUDA kernels are the nuclear option. Triton makes kernel development accessible to Python programmers. You probably will not need this often, but when you do, nothing else will suffice.
- The gap between a research model and a production model is enormous. This session bridges that gap.

---

## Module Assessment Criteria

The apprentice has mastered this module when they can:

1. Write a custom autograd Function from scratch, verify it with gradcheck, and explain every line.
2. Use hooks to extract features, visualize gradients, and modify gradient flow — without modifying the model.
3. Convert a single-GPU training script to multi-GPU DDP training in under 15 minutes.
4. Set up a training loop with mixed precision, gradient accumulation, and gradient checkpointing — and explain why each is there.
5. Export a model to ONNX, quantize it, profile it, and reason about production tradeoffs.

There is no partial credit at this level. Either you can do it or you cannot.
