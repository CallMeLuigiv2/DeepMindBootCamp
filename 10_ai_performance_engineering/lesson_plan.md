# Module 10: AI Performance Engineering -- Lesson Plan

## Week 15 | The Discipline That Separates Researchers from Engineers

At DeepMind, a single training run on a large model can cost hundreds of thousands of dollars. The difference between a well-optimized and a naive pipeline is not a minor convenience -- it is the difference between a project that ships and one that runs out of compute budget. A 2x speedup does not just save time; it lets you run twice as many experiments, explore twice as many hypotheses, and publish before your competitors.

Performance engineering is not an afterthought. It is a discipline. And like all disciplines, it starts with measurement.

The cardinal rule of this module: **measure before you optimize**. Every optimization you apply without first profiling is superstition. Every speedup you claim without a benchmark is fiction.

---

## Session 1: Profiling and Bottleneck Analysis

**Duration**: 3.5 hours (1.5 lecture/demonstration, 2 hands-on profiling)

Before you optimize anything, you must understand where time is being spent. The overwhelming majority of engineers who "optimize" their training pipelines are guessing. They apply mixed precision because someone told them to, or increase num_workers because a blog post said so. They have no idea whether data loading, GPU compute, or memory transfers are the actual bottleneck. This session teaches you to stop guessing and start measuring.

### Learning Objectives

By the end of this session, the apprentice will:
1. Profile a complete training step using `torch.profiler` and interpret every section of the output.
2. Distinguish between CPU-bound, GPU-bound, memory-bound, and data-loading-bound training pipelines.
3. Read and navigate Chrome trace viewer output to identify idle time, kernel launches, and memory transfers.
4. Apply the roofline model to determine whether an operation is compute-bound or memory-bound.
5. Count FLOPs for common operations (matmul, convolution) and relate them to hardware throughput.
6. Identify the top 5 most common bottlenecks in training pipelines and their signatures in profiler output.

### Outline

#### Part 1: The Cardinal Rule -- Measure First (20 min)
- The performance optimization workflow: **profile -> identify bottleneck -> optimize -> re-profile -> repeat**.
- Why intuition fails: the bottleneck is almost never where you think it is.
- Anecdote: a team spent a week optimizing their GPU kernels, only to discover that 60% of training time was spent in data preprocessing on CPU.
- Types of bottlenecks:
  - **CPU bottleneck**: data loading, preprocessing, augmentation, Python overhead.
  - **GPU compute bottleneck**: the GPU is running at full throughput (this is the _good_ kind of bottleneck).
  - **Memory transfer bottleneck**: CPU-to-GPU transfers, GPU memory bandwidth limits.
  - **GPU memory bottleneck**: running out of GPU RAM, forced to reduce batch size.
  - **I/O bottleneck**: disk read speed, network fetch for distributed storage.

#### Part 2: torch.profiler Deep Dive (40 min)
- Setting up the profiler:
  ```python
  with torch.profiler.profile(
      activities=[
          torch.profiler.ProfilerActivity.CPU,
          torch.profiler.ProfilerActivity.CUDA,
      ],
      schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
      on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
      record_shapes=True,
      profile_memory=True,
      with_stack=True,
  ) as prof:
      for step, (inputs, labels) in enumerate(train_loader):
          train_step(model, inputs, labels, optimizer)
          prof.step()
  ```
- Understanding the schedule: `wait` (skip steps to let CUDA warm up), `warmup` (profile but discard -- lets caches fill), `active` (actual profiling), `repeat`.
- Key output tables:
  - `prof.key_averages()` -- aggregated by operation type.
  - `prof.key_averages(group_by_input_shape=True)` -- disaggregated by tensor shapes.
  - Sorting by `self_cuda_time_total`, `cpu_time_total`, `cuda_memory_usage`.
- Reading the output: CPU time vs CUDA time, self time vs total time, count (how many times called).
- The Chrome trace viewer: open `chrome://tracing`, load the JSON trace, navigate the timeline.
  - CPU thread at top, CUDA stream at bottom.
  - Gaps in CUDA = GPU idle time = wasted compute.
  - Overlapping CPU and CUDA = good pipelining.
  - Long CPU bars before CUDA bars = CPU bottleneck.

#### Part 3: The Roofline Model (30 min)
- Every operation is either **compute-bound** or **memory-bound**.
- **Compute-bound**: the GPU ALUs are the bottleneck. More FLOPs would make it slower. Example: large matrix multiplications.
- **Memory-bound**: reading/writing data from GPU memory is the bottleneck. The ALUs are idle waiting for data. Example: element-wise operations, activations, normalization.
- The **arithmetic intensity** of an operation: FLOPs per byte of memory accessed.
  - Matrix multiply $(m \times k) \times (k \times n)$: $2mkn$ FLOPs, reads $(mk + kn) \times \text{bytes\_per\_element}$, writes $mn \times \text{bytes\_per\_element}$.
  - Element-wise ReLU on tensor of size $N$: $N$ FLOPs (comparisons), reads $N \times \text{bytes}$, writes $N \times \text{bytes}$.
- The roofline: plot throughput (FLOPS) vs arithmetic intensity (FLOP/byte).
  - Below the roofline = not hitting hardware limits, room for optimization.
  - On the memory-bound slope = need to reduce memory access.
  - On the compute-bound ceiling = need faster hardware or algorithmic improvement.
- GPU specs you must know: peak FLOPS (e.g., A100: 312 TFLOPS for TF32), memory bandwidth (e.g., A100: 2 TB/s).

#### Part 4: Common Bottleneck Patterns (20 min)
- **Pattern 1: Data starvation**. Symptom: GPU utilization is low (30-50%), CUDA timeline has gaps. Cause: DataLoader cannot feed data fast enough. Fix: increase `num_workers`, use `pin_memory`, prefetch, or change data format.
- **Pattern 2: CPU preprocessing**. Symptom: large CPU blocks before each CUDA kernel. Cause: augmentation or preprocessing happening on CPU. Fix: move to GPU (kornia, DALI) or precompute.
- **Pattern 3: Small kernel launches**. Symptom: many tiny CUDA kernels with gaps between them. Cause: Python overhead, many small operations. Fix: `torch.compile`, operator fusion.
- **Pattern 4: Memory thrashing**. Symptom: frequent allocations and deallocations visible in memory profile. Cause: creating new tensors every step. Fix: pre-allocate, in-place operations.
- **Pattern 5: Host-device sync**. Symptom: CPU blocks waiting on CUDA. Cause: calling `.item()`, `print(tensor)`, or other operations that force CPU-GPU synchronization. Fix: avoid sync points in the training loop.

### Profiling Exercises (120 min)

1. **Profile a baseline training loop** (30 min): Take a ResNet-18 training on CIFAR-10 with a deliberately untuned DataLoader (num_workers=0, no pin_memory). Profile 10 training steps. Identify the top 3 time-consuming operations. Calculate GPU utilization from the trace.

2. **Bottleneck identification challenge** (40 min): You are given descriptions of four different profiler outputs. For each, identify the bottleneck type and propose the fix:
   - Profile A: GPU utilization 25%, long gaps between CUDA kernels, CPU shows `__getitem__` taking 80% of time.
   - Profile B: GPU utilization 95%, no gaps, CUDA kernels dominated by `aten::mm` and `aten::conv2d`.
   - Profile C: GPU utilization 60%, many small CUDA kernels (< 10us each), significant Python overhead between them.
   - Profile D: GPU utilization 70%, large `aten::to` operations (CPU->CUDA copies) taking 30% of time.

3. **FLOPS counting exercise** (25 min): For a given Transformer block (d_model=512, n_heads=8, seq_len=128, ffn_dim=2048), calculate the FLOPs for: self-attention QKV projection, attention score computation, FFN layers. Compare your manual calculation to `torch.utils.flop_counter`.

4. **Build a profiling utility** (25 min): Write a reusable function `profile_training_step(model, dataloader, optimizer, n_steps=5)` that returns a structured summary: total time, data loading time, forward time, backward time, optimizer step time, GPU utilization estimate.

### Key Takeaways
- Profile first. Always. No exceptions.
- GPU utilization below 80% means you are leaving performance on the table.
- Most training pipelines are bottlenecked by data loading or CPU preprocessing, not GPU compute.
- The roofline model tells you whether optimization should target compute or memory.
- Host-device synchronization points are silent performance killers.

---

## Session 2: GPU Memory Optimization

**Duration**: 3.5 hours (1.5 lecture/demonstration, 2 hands-on)

GPU memory is the most constrained resource in deep learning. You cannot train a model that does not fit in memory. Understanding exactly where memory goes -- and how to reduce each component -- is the difference between training a model and not training it at all.

### Learning Objectives

By the end of this session, the apprentice will:
1. Enumerate every component of GPU memory usage during training and calculate its size.
2. Use `torch.cuda.memory_stats()` and `torch.cuda.memory_summary()` to diagnose memory usage.
3. Implement gradient checkpointing and explain the compute-memory tradeoff.
4. Apply gradient accumulation to simulate large batch sizes within memory constraints.
5. Explain Flash Attention and why it reduces memory from $O(n^2)$ to $O(n)$.
6. Make informed decisions about when to use float16 vs bfloat16 for activations.

### Outline

#### Part 1: Where Does GPU Memory Go? (30 min)
- The five components of GPU memory during training:
  1. **Model parameters**: each float32 parameter = 4 bytes. A 100M parameter model = 400 MB.
  2. **Gradients**: same size as parameters. 100M params = another 400 MB.
  3. **Optimizer states**: Adam stores first moment (m) and second moment (v) for each parameter. That is 2x parameter size = 800 MB for 100M params in fp32.
  4. **Activations**: intermediate outputs saved for the backward pass. This is the variable cost -- it scales with batch size, sequence length, and model depth.
  5. **Temporary buffers**: workspace for operations like convolution, intermediate computation results. Usually small but can spike.
- The memory formula:
  ```
  Total = params * (4 or 2) + gradients * (4 or 2) + optimizer_states + activations + buffers
  ```
- Worked example -- ResNet-50 (25.6M parameters):
  - Parameters: 25.6M * 4 bytes = 102 MB
  - Gradients: 102 MB
  - Adam states: 204 MB
  - Activations at batch_size=32: approximately 2-4 GB (dominates!)
  - Total: approximately 2.5-4.5 GB
- Worked example -- Small Transformer (d=512, L=6, vocab=30k, seq_len=512):
  - Parameters: approximately 40M * 4 = 160 MB
  - Gradients: 160 MB
  - Adam states: 320 MB
  - Activations at batch_size=32: approximately 3-6 GB
  - Total: approximately 4-7 GB
- Critical insight: **activations dominate memory** for most models. This is why batch size is the primary memory knob.

#### Part 2: Memory Profiling Tools (20 min)
- `torch.cuda.memory_allocated()` -- currently allocated memory.
- `torch.cuda.max_memory_allocated()` -- peak memory (the real constraint).
- `torch.cuda.memory_reserved()` -- memory held by the caching allocator (may be higher than allocated).
- `torch.cuda.memory_summary()` -- detailed breakdown including allocation counts, peak usage, and fragmentation.
- Memory snapshots for timeline analysis:
  ```python
  torch.cuda.memory._record_memory_history()
  # ... run your code ...
  torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
  ```
- Visualizing memory snapshots in TensorBoard or the PyTorch memory visualizer.

#### Part 3: Gradient Checkpointing (30 min)
- The tradeoff: normally, all activations from the forward pass are saved for the backward pass. Gradient checkpointing discards some activations and recomputes them during the backward pass.
- Implementation:
  ```python
  from torch.utils.checkpoint import checkpoint

  class CheckpointedBlock(nn.Module):
      def forward(self, x):
          # Instead of saving all intermediate activations:
          return checkpoint(self._inner_forward, x, use_reentrant=False)

      def _inner_forward(self, x):
          # ... expensive computation ...
          return output
  ```
- Memory savings: for a model with $L$ layers, naive approach stores $L$ activation tensors. With checkpointing every $\sqrt{L}$ layers, you store $\sqrt{L}$ tensors and recompute the rest. Memory goes from $O(L)$ to $O(\sqrt{L})$.
- Compute cost: one extra forward pass through the checkpointed segments. Typically 20-30% slower but uses 50-70% less activation memory.
- When to use it: when activations are the memory bottleneck and you want to increase batch size or model size.

#### Part 4: Gradient Accumulation (20 min)
- Problem: you want a batch size of 256 but can only fit 32 in memory.
- Solution: accumulate gradients over 8 mini-batches of 32, then take one optimizer step.
  ```python
  accumulation_steps = 8
  for i, (inputs, labels) in enumerate(train_loader):
      loss = model(inputs, labels) / accumulation_steps
      loss.backward()
      if (i + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```
- Important: divide loss by `accumulation_steps` to keep gradient magnitude consistent.
- This is mathematically equivalent to a larger batch (ignoring BatchNorm statistics).
- When to use: always, when the desired batch size exceeds GPU memory. The only cost is that training takes the same wall-clock time per effective batch -- there is no speedup from accumulation.

#### Part 5: Advanced Memory Techniques (30 min)
- **In-place operations**: operations like `relu_(x)` and `x.add_(y)` modify tensors in place, avoiding allocation. But be careful: in-place ops on tensors that require grad can cause errors. Use only where safe.
- **Clearing cache**: `torch.cuda.empty_cache()` releases unused cached memory back to CUDA. Useful when switching between models or inference modes. Do not call it in the training loop -- it will slow things down.
- **Flash Attention**: standard attention computes the full $N \times N$ attention matrix, requiring $O(N^2)$ memory. Flash Attention tiles the computation and never materializes the full matrix. Memory drops to $O(N)$. This is not an approximation -- it is exact attention, just computed more efficiently.
  ```python
  # PyTorch >= 2.0 has built-in scaled_dot_product_attention with Flash Attention
  from torch.nn.functional import scaled_dot_product_attention
  output = scaled_dot_product_attention(query, key, value, is_causal=True)
  ```
- **CPU offloading**: move optimizer states or even model parameters to CPU when not needed on GPU. Used in large model training (FSDP, DeepSpeed ZeRO). Adds communication overhead but enables training models that do not fit on a single GPU.
- **Activation dtype**: storing activations in float16 or bfloat16 halves activation memory. Combined with mixed precision training (Session 3), this is usually free performance.

### Hands-On Exercises (120 min)

1. **Memory budget calculator** (30 min): Write a function that takes a model and batch size as input and returns the estimated memory breakdown: parameter memory, gradient memory, optimizer state memory, activation memory (estimated via a forward pass with hooks). Compare your estimate to `torch.cuda.max_memory_allocated()`.

2. **Gradient checkpointing experiment** (35 min): Take a 12-layer Transformer. Measure peak memory and training speed with: (a) no checkpointing, (b) checkpointing every layer, (c) checkpointing every 3 layers. Plot the memory-vs-speed tradeoff curve. Find the maximum batch size for each configuration.

3. **Gradient accumulation validation** (25 min): Train the same model with batch_size=128 (one step) and batch_size=16 with 8 accumulation steps. Verify that the gradients after accumulation are identical (within floating point tolerance). Compare final trained model accuracy.

4. **Memory leak detection** (30 min): Write a training loop with a deliberate memory leak (e.g., appending loss tensors to a Python list without detaching). Profile the memory growth. Then fix it and verify memory stays constant.

### Key Takeaways
- Activations are the largest memory consumer during training. Everything else is fixed per model.
- Gradient checkpointing trades 20-30% compute for 50-70% memory savings. Use it when memory is the constraint.
- Gradient accumulation is free in terms of optimization quality. It just does not make training faster.
- Flash Attention is not optional for long sequences. It reduces memory from quadratic to linear.
- Always profile memory with `torch.cuda.memory_summary()` before and after optimization.

---

## Session 3: Mixed Precision and Quantization

**Duration**: 3.5 hours (1.5 lecture/demonstration, 2 hands-on)

Modern GPUs have specialized hardware (Tensor Cores) that operate on lower-precision numbers dramatically faster than full float32. Mixed precision training exploits this for a near-free 1.5-2x training speedup. Quantization goes further, reducing model precision for inference to get 2-4x speedups with minimal accuracy loss. This session covers both.

### Learning Objectives

By the end of this session, the apprentice will:
1. Explain the numerical properties of float32, float16, bfloat16, and int8, and when each is appropriate.
2. Implement mixed precision training with `torch.amp.autocast` and `torch.amp.GradScaler`.
3. Explain the loss scaling mechanism and why it is necessary for float16 training.
4. Apply post-training quantization (dynamic and static) to a trained model.
5. Describe quantization-aware training and when it is needed.
6. Articulate the accuracy-vs-speed tradeoff for each precision level.

### Outline

#### Part 1: Number Formats -- What You Must Know (25 min)
- **float32 (FP32)**: 1 sign, 8 exponent, 23 mantissa bits. Range: approximately 1e-38 to 3e38. Precision: approximately 7 decimal digits. The default for everything.
- **float16 (FP16)**: 1 sign, 5 exponent, 10 mantissa bits. Range: approximately 6e-8 to 65504. Precision: approximately 3 decimal digits. The Tensor Core format.
  - Problem: limited range. Gradients can be as small as 1e-7 or smaller -- below FP16's minimum representable value. They become zero. This is **underflow**.
  - Problem: limited precision. Accumulating many small values loses accuracy.
- **bfloat16 (BF16)**: 1 sign, 8 exponent, 7 mantissa bits. Range: same as float32 (!). Precision: approximately 2 decimal digits.
  - Advantage over FP16: same exponent range as FP32, so underflow is extremely rare. No loss scaling needed.
  - Disadvantage: less precision than FP16. But for neural network training, range matters more than precision.
  - Available on: A100, H100, newer GPUs, TPUs.
- **int8 (INT8)**: 8-bit integers. Range: -128 to 127 (signed) or 0-255 (unsigned). Used for inference quantization.
  - Requires careful mapping (quantization) from floating-point range to integer range.
  - Tensor Cores support INT8 operations with very high throughput.

#### Part 2: Mixed Precision Training (40 min)
- The core idea: keep a **master copy** of weights in float32. Run forward and backward passes in float16 (or bfloat16) for speed. Accumulate gradients in float32 for accuracy. Update the float32 master weights.
- Which operations are safe in float16:
  - **Safe**: matmul, conv2d, linear layers, BMM -- these benefit from Tensor Cores.
  - **Unsafe (keep in float32)**: softmax, layer normalization, loss computation, reductions (sum, mean over large tensors), anything with a large dynamic range.
- `torch.amp.autocast` handles this automatically:
  ```python
  with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
      output = model(input)
      loss = criterion(output, target)
  ```
- **Loss scaling**: the GradScaler mechanism.
  - Problem: in FP16, small gradients underflow to zero.
  - Solution: multiply the loss by a large number (e.g., 1024) before `.backward()`. This scales all gradients up, preventing underflow. Then divide the gradients by the same number before the optimizer step.
  - Dynamic loss scaling: start with a large scale. If gradients overflow (inf/NaN), skip the optimizer step and reduce the scale. If no overflow for N consecutive steps, increase the scale.
  ```python
  scaler = torch.amp.GradScaler()
  with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
      output = model(input)
      loss = criterion(output, target)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```
- BFloat16 training: same pattern but `dtype=torch.bfloat16`. GradScaler is typically not needed because bfloat16 has the same exponent range as float32.

#### Part 3: Post-Training Quantization (25 min)
- **Dynamic quantization**: quantize weights to INT8 at model load time. Activations are quantized on-the-fly during inference. No calibration data needed.
  ```python
  quantized_model = torch.quantization.quantize_dynamic(
      model, {nn.Linear}, dtype=torch.qint8
  )
  ```
  - Speedup: 1.5-2x for CPU inference on models dominated by linear layers.
  - Accuracy: usually < 1% degradation for well-trained models.
- **Static quantization**: both weights and activations are quantized using calibration data. Activation ranges are determined by running representative inputs through the model.
  - Steps: (1) insert quantization observers, (2) run calibration data, (3) convert to quantized model.
  - Better than dynamic because activation quantization parameters are precomputed.
  - Requires representative calibration data (typically 100-1000 samples).

#### Part 4: Quantization-Aware Training (20 min)
- When post-training quantization degrades accuracy too much, simulate quantization during training.
- Insert "fake quantization" nodes that quantize-then-dequantize during forward pass. The backward pass uses straight-through estimators (gradients pass through the quantization as if it were the identity).
- The model learns to be robust to quantization noise.
- Typically recovers most of the accuracy lost by post-training quantization.
- Trade-off: requires retraining (expensive) but produces the best quantized model.

#### Part 5: LLM Quantization Overview (20 min)
- Large language models have unique quantization challenges due to their size and the importance of preserving generation quality.
- **GPTQ**: post-training quantization method that quantizes weights to 4-bit or 3-bit using a layer-wise approach. Uses calibration data to minimize quantization error.
- **AWQ**: Activation-aware Weight Quantization. Identifies "salient" weights (those corresponding to large activations) and keeps them at higher precision.
- **GGML/GGUF**: quantization formats optimized for CPU inference of LLMs.
- The landscape is rapidly evolving. The key principle remains: measure accuracy vs. speed for your specific model and task.

### Hands-On Exercises (120 min)

1. **Number format exploration** (20 min): Write a script that demonstrates: (a) the smallest positive number representable in FP16, FP32, and BF16, (b) what happens when you add 1.0 + 1e-4 in each format, (c) what happens to a gradient of magnitude 1e-6 in FP16 vs BF16.

2. **Mixed precision training** (40 min): Take a standard ResNet-18 training pipeline on CIFAR-10. Implement mixed precision training with autocast + GradScaler. Measure: (a) training speed (samples/sec) before and after, (b) peak GPU memory before and after, (c) final accuracy (should be within 0.2% of baseline). Record all numbers.

3. **Post-training quantization** (30 min): Take a trained model (your ResNet-18 or a pre-trained one). Apply dynamic quantization. Apply static quantization with calibration. Compare: (a) inference speed (FP32 vs dynamic INT8 vs static INT8), (b) accuracy on test set, (c) model size on disk.

4. **The accuracy-speed frontier** (30 min): For a single model, plot the Pareto frontier: x-axis = inference latency, y-axis = accuracy. Points: FP32, FP16 (using autocast for inference), dynamic INT8, static INT8. Annotate each point with its model size.

### Key Takeaways
- Mixed precision training is nearly free: 1.5-2x speedup and 30-50% memory reduction with negligible accuracy impact.
- BFloat16 is preferable to FP16 when available -- it eliminates the need for loss scaling.
- Post-training quantization gives 1.5-3x inference speedup with minor accuracy loss.
- QAT recovers accuracy but requires retraining.
- Always benchmark accuracy AND speed. A fast model that gives wrong answers is useless.

---

## Session 4: Data Pipeline Optimization

**Duration**: 3 hours (1 lecture, 2 hands-on)

The fastest GPU in the world is useless if it spends half its time waiting for data. Data pipeline optimization is unglamorous work -- no one publishes papers about their DataLoader configuration -- but it is often the single largest source of training speedup available. The golden rule: the GPU should NEVER wait for data.

### Learning Objectives

By the end of this session, the apprentice will:
1. Profile a DataLoader to measure its throughput independently of the model.
2. Tune `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers` for maximum throughput.
3. Explain why different data formats (raw files, LMDB, HDF5, WebDataset, memory-mapped) have different performance characteristics.
4. Move data augmentation from CPU to GPU using kornia or NVIDIA DALI.
5. Build a data pipeline that saturates GPU utilization.

### Outline

#### Part 1: The DataLoader as Bottleneck (25 min)
- How PyTorch DataLoader works internally:
  - Main process sends indices to worker processes.
  - Workers call `__getitem__`, which reads from disk, decodes, transforms.
  - Workers send processed tensors back to the main process via shared memory (or a queue).
  - Main process collates into a batch.
- The fundamental constraint: if `data_loading_time > model_step_time`, the GPU is idle.
- How to measure DataLoader throughput in isolation:
  ```python
  import time
  start = time.perf_counter()
  for i, batch in enumerate(dataloader):
      if i == 100:
          break
  elapsed = time.perf_counter() - start
  throughput = 100 * batch_size / elapsed
  print(f"DataLoader throughput: {throughput:.0f} samples/sec")
  ```
- Compare this to your model's throughput. If DataLoader throughput < model throughput, you have a data bottleneck.

#### Part 2: DataLoader Tuning (30 min)
- **num_workers**: number of subprocesses for data loading.
  - `num_workers=0`: data loading in the main process. Always slow for non-trivial datasets.
  - Rule of thumb: `num_workers = 4 * num_GPUs`, but profile to find the optimal value.
  - Too many workers: CPU contention, memory overhead, diminishing returns.
  - Benchmark: sweep num_workers from 0 to 16, measure throughput.
- **pin_memory=True**: allocates tensors in pinned (page-locked) CPU memory.
  - Enables faster (asynchronous) CPU-to-GPU transfers via `tensor.to(device, non_blocking=True)`.
  - Almost always worth enabling. Cost: slightly more CPU memory.
- **prefetch_factor**: how many batches each worker prefetches.
  - Default is 2. Increasing to 3-4 can help if workers are fast but there is a gap between batches.
  - Cost: more CPU memory per worker.
- **persistent_workers=True**: keep worker processes alive between epochs.
  - Without this, workers are respawned every epoch (expensive startup cost).
  - Essential when using many workers or when dataset initialization is slow.
- **non_blocking transfers**: after pin_memory, use `data.to(device, non_blocking=True)` to overlap data transfer with computation.

#### Part 3: Efficient Data Formats (25 min)
- **Individual files (JPEG, PNG)**: the default. Each `__getitem__` opens a file, reads, decodes. Many small file reads are slow due to filesystem overhead (inode lookup, file handle creation).
- **LMDB (Lightning Memory-Mapped Database)**: stores all data in a single memory-mapped file. Random access is fast because the OS handles paging. Excellent for datasets that fit on disk.
- **HDF5**: similar concept to LMDB. Hierarchical structure. Good for structured scientific data. Watch out: default HDF5 does not handle concurrent reads well from multiple workers.
- **WebDataset**: stores data as tar archives. Designed for sequential (streaming) access, ideal for very large datasets or cloud storage. Used widely at Google/DeepMind.
- **Memory-mapped NumPy/tensors**: `np.memmap` or `torch.Tensor` backed by a file. The OS loads pages on demand. Fast random access for pre-processed data.
- **TFRecord / RecordIO**: sequential binary formats. Good for large-scale distributed training.
- Decision guide: for datasets < 100GB on local SSD, LMDB is excellent. For cloud/distributed, WebDataset. For preprocessed numerical data, memory-mapped tensors.

#### Part 4: GPU-Accelerated Augmentation (20 min)
- Standard augmentation (torchvision.transforms) runs on CPU. For heavy augmentation pipelines (random crop, flip, color jitter, normalize), this can become the bottleneck.
- **Kornia**: differentiable augmentations on GPU tensors.
  ```python
  import kornia.augmentation as K
  transform = K.AugmentationSequential(
      K.RandomHorizontalFlip(p=0.5),
      K.RandomCrop((32, 32), padding=4),
      K.Normalize(mean, std),
      data_keys=["input"],
  )
  # Apply on GPU after transfer
  batch = transform(batch.to(device))
  ```
- **NVIDIA DALI**: high-performance data loading and augmentation pipeline. Can decode JPEG on GPU, apply augmentations on GPU, all in a fused pipeline. Significant speedup for image-heavy workloads.
- Trade-off: GPU augmentation uses some GPU compute and memory. But if CPU augmentation is the bottleneck, the trade-off is strongly positive.

#### Part 5: Putting It All Together (20 min)
- The optimization checklist (in order):
  1. Measure DataLoader throughput independently.
  2. Set `num_workers` (start at 4, sweep up).
  3. Enable `pin_memory=True`.
  4. Enable `persistent_workers=True`.
  5. Increase `prefetch_factor` to 3-4.
  6. Use `non_blocking=True` for `.to(device)`.
  7. If still bottlenecked: move augmentation to GPU.
  8. If still bottlenecked: switch data format (LMDB, WebDataset).
  9. If still bottlenecked: precompute expensive preprocessing.
- Each step typically gives 10-50% improvement. Combined: 2-5x improvement over a naive pipeline.

### Hands-On Exercises (120 min)

1. **DataLoader throughput benchmark** (25 min): For CIFAR-10 with standard augmentation, sweep `num_workers` from 0 to 16. Plot throughput vs. num_workers. Find the optimal value for your system. Then enable `pin_memory`, `persistent_workers`, increased `prefetch_factor`. Report cumulative improvement.

2. **Data format comparison** (35 min): Create an LMDB version of a subset of ImageNet (or CIFAR-100). Create a memory-mapped tensor version. Benchmark DataLoader throughput for: raw files, LMDB, memory-mapped. Report speedup and the disk space trade-off.

3. **GPU augmentation migration** (30 min): Take a training pipeline with heavy CPU augmentation. Move the augmentation to GPU using kornia. Measure: (a) DataLoader throughput, (b) total training throughput, (c) GPU memory impact.

4. **The zero-wait pipeline** (30 min): Starting from a deliberately slow DataLoader (num_workers=0, no pin_memory, heavy CPU augmentation on individual PNG files), apply every optimization from the checklist one at a time, measuring after each. Your goal: GPU utilization > 90% during training. Document each optimization and its impact.

### Key Takeaways
- A GPU that waits for data is a GPU that wastes money.
- num_workers, pin_memory, and persistent_workers are the low-hanging fruit. Always set them.
- Data format matters. Reading thousands of individual files is slow. Use LMDB or WebDataset for large datasets.
- GPU augmentation is underused. If CPU augmentation is your bottleneck, kornia or DALI will fix it.
- Profile the DataLoader separately from the model. Know your data throughput.

---

## Session 5: Compilation and Kernel Optimization

**Duration**: 3.5 hours (1.5 lecture/demonstration, 2 hands-on)

PyTorch's eager mode -- where each operation executes immediately -- is wonderful for debugging but leaves significant performance on the table. `torch.compile` (introduced in PyTorch 2.0) bridges this gap by capturing computational graphs and generating optimized kernels. This session covers compilation, operator fusion, and the basics of writing custom kernels when the compiler is not enough.

### Learning Objectives

By the end of this session, the apprentice will:
1. Explain how `torch.compile` works: TorchDynamo for graph capture, TorchInductor for code generation.
2. Apply `torch.compile` to a model and benchmark the speedup.
3. Identify and fix graph breaks that prevent full compilation.
4. Explain why operator fusion reduces memory traffic and improves performance.
5. Write a basic Triton kernel for a fused operation.
6. Describe Flash Attention as a case study in kernel engineering.

### Outline

#### Part 1: Why Compilation Matters (15 min)
- In eager mode, every operation is dispatched individually: Python -> C++ dispatch -> CUDA kernel launch -> execute -> return.
- For a model with 1000 operations per forward pass, that is 1000 kernel launches, 1000 round trips, 1000 memory reads and writes.
- Compilation captures the graph of operations and generates a single optimized kernel (or a small number of fused kernels).
- Benefit 1: **reduced kernel launch overhead** -- fewer launches, less Python overhead.
- Benefit 2: **operator fusion** -- combine multiple operations into one kernel, reducing memory reads/writes.
- Benefit 3: **memory planning** -- allocate all intermediate tensors upfront, reduce allocator overhead.
- Typical speedups: 1.3-2x for standard models, sometimes up to 3x for memory-bound models.

#### Part 2: torch.compile Deep Dive (40 min)
- The compilation stack:
  1. **TorchDynamo**: a Python bytecode analyzer that captures the computational graph from your Python code. It intercepts Python execution and records operations.
  2. **FX Graph**: the intermediate representation -- a directed acyclic graph of operations.
  3. **TorchInductor**: the default backend that takes the FX graph and generates optimized Triton kernels (for GPU) or C++/OpenMP code (for CPU).
- Using it:
  ```python
  model = torch.compile(model)  # That's it for the basic case
  ```
- Compilation modes:
  - `mode="default"`: good balance of compile time and runtime speed.
  - `mode="reduce-overhead"`: uses CUDA graphs to eliminate kernel launch overhead. Great for small models where launch overhead dominates.
  - `mode="max-autotune"`: tries many kernel variants and picks the fastest. Slower compile, faster runtime.
- **Graph breaks**: when Dynamo encounters code it cannot trace (e.g., data-dependent control flow, unsupported Python constructs), it inserts a "graph break" -- splitting the model into multiple subgraphs. Each break reduces optimization potential.
  - Debugging: `TORCH_LOGS="graph_breaks" python train.py` shows where breaks occur.
  - Common causes: `print()` statements, `if tensor.item() > 0:` (data-dependent), `.numpy()` calls, unsupported third-party library calls.
  - Fix: remove print statements from the model, avoid `.item()` in forward pass, use `fullgraph=True` to make graph breaks an error during development.
- The compilation cache: compiled kernels are cached. First run is slow (compilation), subsequent runs are fast. The cache persists across runs if `TORCHINDUCTOR_CACHE_DIR` is set.

#### Part 3: Operator Fusion -- Why It Matters (20 min)
- Consider a sequence: `y = relu(batch_norm(linear(x)))`.
- Without fusion: linear writes output to GPU memory, batch_norm reads it back, writes normalized result, relu reads it back, writes final output. Three memory round trips.
- With fusion: one kernel reads x, computes linear + batch_norm + relu, writes the final y. One memory round trip.
- For memory-bound operations (which most element-wise operations are), this is a massive win. Memory bandwidth, not compute, is the bottleneck.
- torch.compile performs fusion automatically for many common patterns.
- You can inspect the generated kernels:
  ```python
  # Set environment variable
  TORCH_LOGS="output_code" python train.py
  ```

#### Part 4: Writing Custom Triton Kernels (35 min)
- When `torch.compile` is not enough -- when you need a custom operation or the compiler misses an optimization opportunity -- you can write Triton kernels.
- Triton: a Python-like language for writing GPU kernels, developed by OpenAI. Much more accessible than CUDA C++.
- A simple example -- fused add + relu:
  ```python
  import triton
  import triton.language as tl

  @triton.jit
  def add_relu_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      block_start = pid * BLOCK_SIZE
      offsets = block_start + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      x = tl.load(x_ptr + offsets, mask=mask)
      y = tl.load(y_ptr + offsets, mask=mask)
      result = tl.maximum(x + y, 0.0)
      tl.store(output_ptr + offsets, result, mask=mask)
  ```
- Key concepts:
  - **Program ID**: each kernel instance is identified by a program ID. Different instances process different data blocks.
  - **Block size**: how many elements each kernel instance processes. Tuned for hardware.
  - **Masking**: handles the case where tensor size is not divisible by block size.
  - **Memory coalescing**: adjacent threads should access adjacent memory locations for maximum bandwidth.

#### Part 5: CUDA Kernel Concepts (15 min)
- For those who want to go deeper, a conceptual overview of CUDA:
  - **Thread blocks**: groups of threads (32 threads = 1 warp) that execute together.
  - **Shared memory**: fast on-chip memory shared within a thread block. Used for data reuse.
  - **Global memory**: GPU DRAM. High bandwidth but high latency. Coalesced access is critical.
  - **Occupancy**: the ratio of active warps to maximum warps. Higher occupancy can hide memory latency.
- You do not need to write CUDA kernels for most work. But understanding these concepts helps you reason about why certain operations are fast or slow.

#### Part 6: Flash Attention -- A Case Study (25 min)
- Standard attention: $Q K^T$ produces an $N \times N$ matrix (for sequence length $N$). This matrix is stored in GPU memory, softmax is applied, then it is multiplied by $V$.
  - Memory: $O(N^2)$ for the attention matrix.
  - For $N=8192$, that is $8192^2 \times 2$ bytes (FP16) = 128 MB per head per batch element.
- Flash Attention (Dao et al., 2022): tiles the $Q$, $K$, $V$ matrices into blocks that fit in SRAM (shared memory). Computes attention block by block, accumulating the softmax statistics online. Never materializes the full $N \times N$ matrix in GPU memory.
  - Memory: $O(N)$ -- only the output and the log-sum-exp statistics.
  - Speed: 2-4x faster than standard attention due to reduced memory traffic, despite doing some redundant computation.
- Why this is the gold standard of kernel engineering:
  - It does not approximate anything. The output is identical.
  - It exploits the memory hierarchy (SRAM vs DRAM) to reduce total data movement.
  - It required deep understanding of both the algorithm (online softmax) and the hardware (memory hierarchy, thread block scheduling).
- Lesson: the biggest performance wins come not from micro-optimizations but from rethinking algorithms with hardware constraints in mind.

### Hands-On Exercises (120 min)

1. **torch.compile benchmark** (30 min): Take a ResNet-50 and a small Transformer. Compile each with `mode="default"`, `mode="reduce-overhead"`, and `mode="max-autotune"`. Benchmark forward pass time and full training step time. Report the speedup for each mode and model. Note any graph breaks.

2. **Graph break debugging** (25 min): Write a model with 3 deliberate graph breaks (a print statement, a `.item()` call, a data-dependent if statement). Use `TORCH_LOGS="graph_breaks"` to identify them. Fix each one. Measure the speedup from eliminating graph breaks.

3. **Triton kernel: fused LayerNorm + GELU** (35 min): Write a Triton kernel that computes LayerNorm followed by GELU in a single pass. Benchmark against the unfused PyTorch version (two separate operations). Verify numerical correctness. Report the speedup.

4. **Flash Attention comparison** (30 min): Implement standard scaled dot-product attention (materializing the full attention matrix). Compare with `torch.nn.functional.scaled_dot_product_attention` (which uses Flash Attention). Benchmark for sequence lengths: 256, 512, 1024, 2048, 4096. Plot memory usage and speed for both implementations.

### Key Takeaways
- `torch.compile` is the single easiest performance optimization in modern PyTorch. Try it first.
- Graph breaks kill performance. Write "compilation-friendly" code: no prints, no `.item()`, no data-dependent control flow in the model.
- Operator fusion is the primary mechanism by which compilation improves speed -- it reduces memory traffic.
- Custom Triton kernels are for when the compiler is not enough. They are more accessible than CUDA but still require understanding of GPU execution.
- Flash Attention is the exemplar of hardware-aware algorithm design. Study it.

---

## Assessment Criteria

At the end of this module, the apprentice should be able to:
1. Profile any PyTorch training pipeline and identify the primary bottleneck within 15 minutes.
2. Calculate the GPU memory budget for a given model and batch size, to within 20% accuracy.
3. Implement mixed precision training from memory and explain every component.
4. Configure a DataLoader that achieves > 90% GPU utilization.
5. Apply `torch.compile` and debug graph breaks.
6. Write a simple Triton kernel for a fused operation.
7. Given a slow training pipeline, produce a systematic optimization plan with expected speedups.

Performance is not something you "add at the end." It is a discipline that informs how you design models, write code, and plan experiments from day one. A DeepMind engineer who cannot profile their code is like a surgeon who cannot read an X-ray.
