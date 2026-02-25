# Assignment 01: Profiling and Bottleneck Analysis

**Module:** 10 -- AI Performance Engineering
**Session:** 1 -- Profiling and Bottleneck Analysis
**Estimated Time:** 8-10 hours
**Difficulty:** Intermediate-Advanced

---

## Overview

You will take a deliberately inefficient training pipeline, profile it systematically, identify every bottleneck, fix each one individually while measuring the improvement, and produce a professional-quality profiling report. This is the foundational skill of performance engineering: you cannot optimize what you have not measured.

By the end of this assignment, you will have the profiling instincts that let you look at any training pipeline and, within 15 minutes, know exactly where the time is going and what to fix first.

---

## Part 1: Build the Deliberately Slow Pipeline (15 points)

Create a training pipeline for ResNet-18 on CIFAR-10 that incorporates the following anti-patterns. Each one is a deliberate performance mistake that you will later identify and fix.

### Anti-Patterns to Implement

1. **DataLoader with num_workers=0**: all data loading happens in the main process, blocking GPU work.

2. **No pin_memory**: CPU-to-GPU transfers are synchronous and go through pageable memory.

3. **Heavy CPU augmentation**: apply an excessively heavy augmentation pipeline on CPU, including:
   - RandomResizedCrop with interpolation
   - ColorJitter with all four parameters (brightness, contrast, saturation, hue)
   - RandomRotation with 30-degree range
   - RandomAffine with translation and shear
   - GaussianBlur
   - RandomErasing
   - All the standard ones (horizontal flip, normalize)

4. **Synchronization point in the loop**: after each forward pass, call `loss.item()` and print the loss value every step. This forces a CPU-GPU synchronization on every iteration.

5. **No in-place operations**: use functional ReLU (not in-place), create new tensors for every operation.

6. **Unnecessary tensor copies**: explicitly call `.clone()` on the input tensors before passing them to the model.

7. **Logging full tensors**: every 10 steps, convert the model's output tensor to a NumPy array for "logging" (simulating a common mistake).

8. **Small batch size without justification**: use batch_size=8, which underutilizes the GPU.

**Requirements:**
- The pipeline must actually train (loss should decrease). It just needs to be slow.
- Add comments marking each anti-pattern so you can track them.
- Run the pipeline for 50 iterations and record the baseline throughput (samples/sec).

---

## Part 2: Profile the Baseline (25 points)

### 2A: Quick Metrics (5 points)

Without using the full profiler, collect these initial measurements:
- GPU utilization (from `nvidia-smi`). Record the average over 30 seconds of training.
- Peak GPU memory usage (from `torch.cuda.max_memory_allocated()`).
- Overall throughput: samples per second for 50 training steps.
- DataLoader throughput measured independently (iterate the DataLoader without any model computation for 100 batches).

Present these in a table:

```
| Metric                    | Value     |
|---------------------------|-----------|
| GPU Utilization (avg)     | ____%     |
| Peak GPU Memory           | ____ MB   |
| Training Throughput       | ____ samp/s |
| DataLoader Throughput     | ____ samp/s |
```

### 2B: Phase-by-Phase Timing (10 points)

Instrument the training loop to separately time each phase. Use `torch.cuda.synchronize()` before every timestamp (explain in a comment why this is necessary).

Phases to time:
1. Data loading (fetching the next batch from the DataLoader)
2. Data transfer (moving tensors to GPU)
3. Forward pass
4. Loss computation and synchronization
5. Backward pass
6. Optimizer step

Run for 50 steps (skip the first 10 for warmup). Present average times per phase in a table and as a bar chart or pie chart showing the percentage of total time spent in each phase.

**Critical question to answer:** Which phase dominates? Is this what you expected?

### 2C: Full torch.profiler Analysis (10 points)

Run the complete `torch.profiler` on 10 training steps. Export a Chrome trace.

From the profiler output:
1. List the top 10 operations by CUDA time. For each, note whether it is compute-bound or memory-bound (justify your reasoning).
2. List the top 10 operations by CPU time. Identify any surprises.
3. From the Chrome trace, screenshot or describe: (a) the overall pattern of CPU vs CUDA activity, (b) any visible gaps in the CUDA stream, (c) any unexpectedly long CPU operations.
4. Calculate the approximate GPU utilization from the trace: (total CUDA active time) / (total wall clock time).

---

## Part 3: Fix Each Bottleneck (40 points)

Fix each anti-pattern **one at a time**, measuring the impact of each fix independently. This is critical: if you fix everything at once, you learn nothing about which optimizations matter most.

### 3A: Fix Data Loading (8 points)

- Increase `num_workers` (try values 2, 4, 8).
- Enable `pin_memory=True`.
- Enable `persistent_workers=True`.
- Enable `non_blocking=True` for `.to(device)` calls.

For each change, measure:
- DataLoader throughput (independent measurement)
- Training throughput
- GPU utilization

Present results in a table showing the incremental improvement from each sub-fix.

### 3B: Fix CPU Augmentation (8 points)

Replace the heavy CPU augmentation with a lightweight augmentation pipeline:
- Keep only RandomCrop(32, padding=4) and RandomHorizontalFlip (the standard CIFAR-10 augmentation).
- Move normalize to a simple tensor operation.

Measure the impact on:
- DataLoader throughput
- Training throughput

Then, as a stretch: move augmentation to GPU using kornia and measure the additional improvement.

### 3C: Remove Synchronization Points (8 points)

- Remove the per-step `loss.item()` call. Instead, only call `.item()` every 50 steps (or accumulate losses using `.detach()`).
- Remove the NumPy conversion logging.
- Remove unnecessary `.clone()` calls.

Measure the impact on training throughput. Explain why removing `.item()` from the inner loop makes a difference (hint: CUDA asynchrony).

### 3D: Increase Batch Size (8 points)

- Increase batch_size from 8 to 32, 64, 128, and 256 (or until you run out of memory).
- For each batch size, measure training throughput (samples/sec) and GPU utilization.
- Plot throughput vs batch size. At what point do returns diminish?

### 3E: Apply Additional Optimizations (8 points)

Apply two more optimizations from the module material:
1. **Mixed precision training** (autocast + GradScaler): measure speed and memory impact.
2. **torch.compile**: compile the model and measure the speedup.

For each, measure training throughput, peak memory, and GPU utilization.

---

## Part 4: The Profiling Report (20 points)

Produce a structured performance engineering report. This is the kind of document you would present to a team lead at DeepMind to justify compute allocation or architectural decisions.

### Required Sections

**1. Executive Summary (3 points)**
- Baseline throughput, final throughput, total speedup.
- The single most impactful optimization and why.
- One-paragraph recommendation for anyone training a similar model.

**2. Before/After Measurements Table (5 points)**

```
| Metric              | Baseline | After DataLoader | After Augmentation | After Sync Removal | After Batch Size | After Mixed Prec | After Compile | Final |
|---------------------|----------|------------------|--------------------|--------------------|------------------|------------------|---------------|-------|
| Throughput (samp/s) |          |                  |                    |                    |                  |                  |               |       |
| GPU Utilization     |          |                  |                    |                    |                  |                  |               |       |
| Peak Memory (MB)    |          |                  |                    |                    |                  |                  |               |       |
| Data Load Time (ms) |          |                  |                    |                    |                  |                  |               |       |
| Forward Time (ms)   |          |                  |                    |                    |                  |                  |               |       |
| Backward Time (ms)  |          |                  |                    |                    |                  |                  |               |       |
| Optim Step Time(ms) |          |                  |                    |                    |                  |                  |               |       |
```

**3. Speedup Waterfall Chart (5 points)**

Create a waterfall chart (or stacked bar chart) showing the contribution of each optimization to the total speedup. The chart should make it visually obvious which optimizations mattered most.

**4. Bottleneck Analysis (4 points)**

For each optimization:
- What was the bottleneck before the fix?
- How did you identify it (which profiling metric)?
- What was the expected impact based on your analysis?
- What was the actual impact?
- If expected and actual differ, why?

**5. Lessons Learned (3 points)**

Write 3-5 paragraphs on what you learned about performance engineering from this exercise. Address:
- Which bottleneck was the most surprising?
- How does the order of optimizations matter?
- What would you profile first the next time you encounter a slow training pipeline?

---

## Target

**Your final optimized pipeline must achieve at least 3x throughput over the deliberately slow baseline.** Given the severity of the anti-patterns, 5-10x improvement is achievable. Document the exact speedup achieved.

---

## Deliverables

1. **Baseline script** (`baseline_slow.py`): The deliberately slow pipeline with all anti-patterns.
2. **Optimized script** (`optimized.py`): The final optimized pipeline.
3. **Profiling notebook** (`profiling_analysis.ipynb` or `.py`): All profiling code, measurements, and analysis.
4. **Report** (`profiling_report.md` or in the notebook): The structured report from Part 4.
5. **Chrome traces**: At least two trace files -- one for the baseline, one for the final optimized version.

---

## Evaluation Criteria

| Component | Points | Criteria |
|-----------|--------|----------|
| Slow pipeline construction | 15 | All 7 anti-patterns implemented, pipeline trains correctly |
| Quick metrics | 5 | All 4 metrics measured and tabulated correctly |
| Phase-by-phase timing | 10 | Correct synchronization, clear presentation, correct analysis |
| Full profiler analysis | 10 | Top ops identified, bottleneck types classified, trace interpreted |
| DataLoader fixes | 8 | Incremental measurements, optimal num_workers found |
| Augmentation fix | 8 | Throughput improvement measured, GPU augmentation attempted |
| Sync point removal | 8 | Impact measured, explanation of CUDA asynchrony correct |
| Batch size tuning | 8 | Sweep performed, diminishing returns identified |
| Additional optimizations | 8 | Mixed precision and compile applied, measured correctly |
| Profiling report | 20 | Complete, professional, accurate, includes waterfall chart |
| **Total** | **100** | |

**Passing score:** 70/100. To pass, you must achieve at least 3x speedup and identify the top 3 bottlenecks correctly.

---

## Stretch Goals

1. **Profile a Transformer model**: Replace ResNet-18 with a small Transformer (e.g., 6 layers, d_model=256) on a text classification task. Profile it and compare the bottleneck distribution to the CNN pipeline. Transformers have different bottleneck profiles -- the attention mechanism and its quadratic memory scaling introduce new considerations.

2. **Roofline analysis**: For the top 5 CUDA operations in your model, compute the arithmetic intensity and plot them on a roofline chart for your GPU. Classify each as compute-bound or memory-bound. Verify your classification against profiler data.

3. **Automated profiling tool**: Write a general-purpose `profile_pipeline(model, dataloader, optimizer)` function that automatically runs through the profiling cookbook and outputs a formatted report with recommendations. Make it reusable for any model.

4. **Multi-GPU profiling**: If you have access to multiple GPUs, profile a DataParallel or DistributedDataParallel setup. Identify new bottleneck types that arise from inter-GPU communication.

5. **Profile a real workload**: Take a model from a recent paper you have implemented (or from earlier modules in this course) and apply the full profiling methodology. Identify at least one non-obvious optimization opportunity and implement it.

---

*"The engineers who change the field are not the ones who build the biggest models. They are the ones who make the biggest models trainable."*
