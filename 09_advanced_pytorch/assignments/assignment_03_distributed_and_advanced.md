# Assignment 3: Distributed Training and Advanced Techniques

## Overview

This is the capstone assignment. You will take a working single-GPU training pipeline and systematically upgrade it with every technique covered in this module: distributed training, gradient accumulation, mixed precision, gradient checkpointing, and `torch.compile`. Each technique is added incrementally so you understand its individual contribution. The final product is a training script that represents current best practice — the kind of code that runs at scale in production research.

---

## Part 1: Single-GPU to Multi-GPU with DDP

### Background

DistributedDataParallel (DDP) is the standard way to scale training across multiple GPUs. Unlike DataParallel, it uses one process per GPU, avoids the GIL, distributes memory evenly, and supports multi-node training. Converting a single-GPU script to DDP requires roughly 10-15 lines of code changes, but they must be exactly right.

### Task

1. **Start with the provided single-GPU training script** (or write your own):
   - A ResNet-18 (or equivalent) trained on CIFAR-10.
   - Standard training loop: forward, loss, backward, step.
   - Logging: print loss and accuracy each epoch.
   - Checkpoint saving at the end of training.

2. **Convert to DDP:**
   - Add `dist.init_process_group()` and `dist.destroy_process_group()`.
   - Get `local_rank` from `os.environ["LOCAL_RANK"]`.
   - Set `torch.cuda.set_device(local_rank)`.
   - Move the model to the correct GPU with `.to(local_rank)`.
   - Wrap the model in `DDP(model, device_ids=[local_rank])`.
   - Replace the standard DataLoader sampler with `DistributedSampler`.
   - Call `sampler.set_epoch(epoch)` at the start of each epoch.
   - Guard print statements and checkpoint saving with `if dist.get_rank() == 0`.

3. **Test your DDP script:**
   - If you have multiple GPUs: launch with `torchrun --nproc_per_node=2 train.py`.
   - If you have one GPU: launch with `torchrun --nproc_per_node=2 train.py` (two processes share one GPU — slower but tests correctness).
   - If you have no GPU: use `backend="gloo"` and run on CPU.

4. **Verify correctness:**
   - Train for 5 epochs with DDP and without DDP.
   - Compare final accuracy and loss. They should be similar (not identical due to different batching, but within 1-2% accuracy).
   - Verify that the saved checkpoint loads correctly on a single GPU.

5. **Checkpoint management:**
   - Save the checkpoint from rank 0 only: `model.module.state_dict()`.
   - Load the checkpoint on all ranks with correct `map_location`.
   - Demonstrate resuming training from a checkpoint.

### Deliverables

- `train_single_gpu.py` — the original single-GPU script.
- `train_ddp.py` — the converted DDP script.
- Written notes on every change made, explaining why each change is necessary.
- Evidence that the DDP script runs correctly (terminal output or logs).

---

## Part 2: Gradient Accumulation

### Background

Many important models require large batch sizes for training stability (e.g., BERT uses 256, large vision models use 1024+). When GPU memory only supports a small batch, gradient accumulation simulates a larger batch by accumulating gradients over multiple forward-backward passes before updating weights.

### Task

1. **Implement gradient accumulation** in your training loop:
   - Set `micro_batch_size = 32` and `accumulation_steps = 8` (effective batch size = 256).
   - Divide the loss by `accumulation_steps` before calling `.backward()`.
   - Only call `optimizer.step()` and `optimizer.zero_grad()` every `accumulation_steps` micro-batches.

2. **Verify mathematical equivalence:**
   - Train a small model on a small dataset for 1 step with batch size 256 (if memory permits).
   - Train the same model (same initialization) with batch size 32 and 8 accumulation steps.
   - Compare the resulting model parameters. They should be identical (up to floating-point precision).
   - If batch size 256 does not fit in memory, use a smaller effective batch size for this verification.

3. **Handle the edge case:** What happens when the dataset size is not evenly divisible by `micro_batch_size * accumulation_steps`? Implement proper handling: either drop the last incomplete accumulation cycle or adjust the normalization factor for the last cycle.

4. **Integrate with DDP:** Add gradient accumulation to your DDP training script. Ensure that gradient synchronization only happens on the accumulation boundary (use `model.no_sync()` context manager for intermediate steps to avoid unnecessary allreduce).

### Deliverables

- Modified training script with gradient accumulation.
- Verification that accumulated gradients match single-step gradients.
- Explanation of the `no_sync()` optimization for DDP.

---

## Part 3: Mixed Precision Training

### Background

Modern GPUs (V100, A100, H100) have Tensor Cores that operate on float16 data at 2-8x the speed of float32. Mixed precision training uses float16 for most operations and float32 where precision is critical. The challenge: float16 has limited range, and small gradients can underflow to zero. Loss scaling (multiplying the loss by a large constant before backward, then dividing gradients by the same constant before the optimizer step) prevents this.

### Task

1. **Implement mixed precision training** using `torch.cuda.amp`:
   - Wrap the forward pass in `torch.cuda.amp.autocast()`.
   - Use `torch.cuda.amp.GradScaler` for loss scaling.
   - Replace `loss.backward()` with `scaler.scale(loss).backward()`.
   - Replace `optimizer.step()` with `scaler.step(optimizer)`.
   - Add `scaler.update()` after each step.

2. **Measure the impact:**
   - Train the same model with and without mixed precision for 10 epochs.
   - Measure: training time per epoch, peak GPU memory usage, final accuracy.
   - Report the results in a table.
   - Expected: ~1.5-2x speedup, ~40-50% memory reduction, <0.5% accuracy difference.

3. **Understand what autocast does:**
   - Print the dtypes of intermediate tensors inside the model during a forward pass with autocast enabled.
   - Identify which operations run in float16 and which stay in float32.
   - Explain why loss computation should be in float32 (hint: log and exp operations lose precision in float16).

4. **Integrate with gradient accumulation:**
   - Combine mixed precision with gradient accumulation.
   - The correct pattern: `scaler.scale(loss / accumulation_steps).backward()` for each micro-batch, then `scaler.step(optimizer)` and `scaler.update()` on the accumulation boundary.

### Deliverables

- Training script with mixed precision.
- Performance comparison table (time, memory, accuracy).
- Written explanation of which operations use float16 vs float32 and why.

---

## Part 4: Gradient Checkpointing

### Background

During the forward pass, PyTorch saves intermediate activations (tensors needed for gradient computation) in memory. For deep networks, this memory usage can be enormous — often more than the model parameters and optimizer states combined. Gradient checkpointing discards some intermediate activations during forward and recomputes them during backward, trading roughly 33% more compute for significant memory savings.

### Task

1. **Implement gradient checkpointing** on a deep model:
   - Use a deep ResNet (50+ layers) or build a deep custom network.
   - Apply `torch.utils.checkpoint.checkpoint` to segments of the network.
   - Use `use_reentrant=False` (the correct, modern implementation).

2. **Measure memory savings:**
   - Train one step without checkpointing. Record peak GPU memory with `torch.cuda.max_memory_allocated()`.
   - Train one step with checkpointing. Record peak GPU memory.
   - Calculate the memory savings as a percentage.
   - Expected: 30-60% memory reduction depending on model depth.

3. **Measure compute overhead:**
   - Time 100 training steps without checkpointing.
   - Time 100 training steps with checkpointing.
   - Calculate the slowdown.
   - Expected: 20-40% increase in training time.

4. **Experiment with checkpointing granularity:**
   - Checkpoint every block vs every other block vs every 4th block.
   - Plot: memory usage vs training time for different granularities.
   - Find the sweet spot for your model.

### Deliverables

- Training script with gradient checkpointing.
- Memory and compute measurements.
- Granularity analysis with plot.

---

## Part 5: torch.compile

### Background

`torch.compile` is PyTorch 2.0's compilation system. It captures the computation graph from Python code (via TorchDynamo), generates an optimized backward graph (via AOTAutograd), and produces fused GPU kernels (via TorchInductor). The result: faster training and inference with zero code changes to the model.

### Task

1. **Apply torch.compile** to your model:
   ```python
   compiled_model = torch.compile(model)
   ```

2. **Benchmark before and after:**
   - Measure forward pass time, backward pass time, and total training step time.
   - Account for compilation overhead: the first call is slow (compilation happens), subsequent calls are fast (use cached compiled code).
   - Run enough iterations to amortize compilation time.
   - Expected: 10-30% speedup for compute-bound models on modern GPUs.

3. **Try different compilation modes:**
   - `mode="default"` — balanced.
   - `mode="reduce-overhead"` — uses CUDA graphs to reduce kernel launch overhead.
   - `mode="max-autotune"` — tries all possible optimizations (slow compilation, best runtime).
   - Compare the three modes on your model.

4. **Handle compilation errors:**
   - Introduce a deliberate graph break (e.g., a `print(x.shape)` inside the model).
   - Show the warning/error that torch.compile produces.
   - Fix it.
   - Try `fullgraph=True` to make graph breaks an error.

5. **Dynamic shapes:**
   - Test with variable batch sizes.
   - Show the recompilation that happens.
   - Fix with `dynamic=True`.

### Deliverables

- Benchmark results for each compilation mode.
- Documentation of a graph break and its fix.
- Analysis: when does torch.compile help the most for your model?

---

## Part 6: The Combined Pipeline

### Background

In production research, all of these techniques are used together. They compose, but there are subtle interactions. This part combines everything into a single, production-quality training script.

### Task

1. **Write a complete training script** that combines:
   - DistributedDataParallel (multi-GPU)
   - Mixed precision training (autocast + GradScaler)
   - Gradient accumulation (effective batch size 256 from micro-batch 32)
   - Gradient checkpointing (on the model backbone)
   - torch.compile (on the model)

2. **The script must:**
   - Initialize the distributed process group.
   - Create the model, wrap with DDP, compile.
   - Use DistributedSampler with set_epoch.
   - Use autocast and GradScaler.
   - Accumulate gradients with `model.no_sync()` for intermediate steps.
   - Apply gradient checkpointing to the backbone.
   - Save checkpoints from rank 0.
   - Log metrics (loss, accuracy, memory usage, throughput) from rank 0.
   - Clean up the process group on exit.
   - Handle keyboard interrupt gracefully.

3. **Train on CIFAR-10** with a ResNet-18 or similar model for 20 epochs. Report:
   - Final test accuracy.
   - Training time per epoch.
   - Peak memory usage per GPU.
   - Throughput (images per second).

4. **Ablation study:** Measure the contribution of each technique by disabling them one at a time:
   - All techniques enabled (baseline).
   - Disable torch.compile: how much slower?
   - Disable mixed precision: how much more memory, how much slower?
   - Disable gradient checkpointing: how much more memory?
   - Disable gradient accumulation: what effective batch size can you fit?

   Present results in a table.

### Deliverables

- `train_full.py` — the complete combined training script. Every line annotated.
- Ablation study results table.
- Written analysis (2-3 paragraphs) discussing which techniques provide the most benefit for this model and dataset, and how the picture would change for a larger model (e.g., a large language model).

---

## Evaluation Criteria

### Passing

- The DDP script runs correctly and produces reasonable accuracy.
- Gradient accumulation is mathematically verified.
- Mixed precision training shows measurable speedup and memory savings.
- Gradient checkpointing shows measurable memory savings.
- torch.compile shows measurable speedup.
- The combined script runs without errors and combines all techniques correctly.

### Distinction

All of the above, plus:
- The ablation study is thorough and the analysis is insightful.
- The combined script handles edge cases (keyboard interrupt, odd dataset sizes, checkpoint resume).
- The code is production-quality: proper logging, error handling, configuration via command-line arguments.
- Written analysis discusses how results would differ for larger models and datasets.

---

## Stretch Goals

1. **FSDP:** Replace DDP with FSDP in the combined script. This requires a model large enough that FSDP's memory savings are meaningful. Try with a larger model (e.g., ResNet-152 or a small transformer). Measure memory savings compared to DDP.

2. **ONNX export:** After training, export the model to ONNX format. Load it with ONNX Runtime and verify the output matches PyTorch. Measure inference latency with ONNX Runtime vs PyTorch. Handle dynamic batch sizes with `dynamic_axes`.

3. **Quantization:** Apply post-training dynamic quantization to the trained model. Measure the model size reduction and inference speedup. Compare accuracy before and after quantization. If accuracy degrades significantly, try quantization-aware training.

4. **Profiling:** Profile the combined training script with `torch.profiler`. Generate a Chrome trace. Identify the three most expensive operations. Propose optimizations for each.

5. **Custom learning rate scheduler:** Implement a cosine annealing scheduler with linear warmup (commonly used in modern training recipes). The learning rate should linearly increase from 0 to `lr_max` over the first N steps, then follow a cosine decay to `lr_min`. Integrate it into the combined script and show its effect on the training curve.
