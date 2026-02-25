# Assignment 03: End-to-End Optimization -- The "Make It Fast" Challenge

**Module:** 10 -- AI Performance Engineering
**Session:** All Sessions (Capstone Assignment)
**Estimated Time:** 12-16 hours
**Difficulty:** Advanced

---

## Overview

This is the culmination of the performance engineering module. You will take a naive training pipeline and apply every optimization technique you have learned -- systematically, one at a time, measuring each -- to produce the fastest possible training setup that maintains accuracy.

This assignment simulates a real scenario at a lab like DeepMind: you have a model, a dataset, a GPU budget, and a deadline. Your job is to make training as fast as possible so you can iterate on ideas before the budget runs out.

---

## The Task

### Model

A medium-sized Transformer encoder for text classification.

**Architecture specification:**
- Embedding dimension: 256
- Number of attention heads: 8
- Number of Transformer layers: 6
- FFN hidden dimension: 1024
- Vocabulary size: 30,000
- Maximum sequence length: 256
- Classification head: linear layer on the [CLS] token (or mean pooling over sequence positions)
- Dropout: 0.1

This model has approximately 15-20M parameters (compute the exact number as part of the assignment).

### Dataset

Use one of the following text classification datasets (all available through Hugging Face `datasets` or torchtext):
- **AG News** (4-class news classification, 120K training samples) -- recommended
- **IMDB** (binary sentiment, 25K training samples)
- **SST-2** (binary sentiment, 67K training samples)

You may use a simple tokenizer (character-level, word-level, or a pre-trained tokenizer like the one from `bert-base-uncased`). Tokenizer choice does not matter for this assignment -- what matters is the training speed.

### Naive Baseline

Start with a deliberately unoptimized (but functional) training script that includes:
- `DataLoader` with `num_workers=0`, no `pin_memory`
- Full FP32 precision
- No compilation
- Batch size = 16
- Standard PyTorch training loop with no special optimizations
- Raw text data loaded and tokenized on-the-fly in `__getitem__`

This baseline must train correctly: loss should decrease and accuracy should improve over 10 epochs.

---

## Part 1: Establish the Baseline (10 points)

### 1A: Implement the Model (5 points)

Implement the Transformer encoder from the specification above. You may use `nn.TransformerEncoder` or build from `nn.MultiheadAttention` blocks.

Calculate and report:
- Total parameter count
- Estimated memory budget (using the formulas from the notes): parameter memory, gradient memory, optimizer state memory (AdamW), estimated activation memory at batch_size=16

### 1B: Implement the Naive Baseline (5 points)

Write the training script with all the constraints listed above. Train for 10 epochs.

Record:
- Training throughput (samples/sec)
- Peak GPU memory (MB)
- GPU utilization (from nvidia-smi)
- Final test accuracy
- Total training wall-clock time

This is your baseline. All improvements will be measured relative to these numbers.

---

## Part 2: Optimize One Technique at a Time (50 points)

For each optimization below, start from the previous best configuration (not from the baseline). But also measure the standalone impact by comparing to the state just before this optimization was applied.

**Critical methodology:** For each optimization, record:
- Throughput (samples/sec)
- Peak GPU memory (MB)
- GPU utilization
- Test accuracy (train for the same number of epochs)
- The speedup from this specific optimization

### 2A: DataLoader Optimization (8 points)

Apply the full DataLoader optimization checklist:
1. Increase `num_workers` (sweep 2, 4, 8 -- find optimal).
2. Enable `pin_memory=True`.
3. Enable `persistent_workers=True`.
4. Use `non_blocking=True` for `.to(device)`.
5. Pre-tokenize the dataset (tokenize all text upfront and store as tensors, instead of tokenizing in `__getitem__`).

**Key measurement:** Profile the DataLoader throughput independently before and after. Is the DataLoader still the bottleneck after optimization?

### 2B: Mixed Precision Training (8 points)

Add mixed precision training:
- `torch.amp.autocast(device_type='cuda', dtype=torch.float16)` (or `bfloat16` if available).
- `torch.amp.GradScaler()` (for FP16).
- Apply to forward pass and loss computation.

**Key measurements:**
- Throughput improvement.
- Memory reduction (this should free up memory for a larger batch size in a later step).
- Accuracy must remain within 0.5% of the FP32 baseline.

### 2C: torch.compile (8 points)

Compile the model:
```python
model = torch.compile(model)
```

**Key measurements:**
- Throughput improvement after warmup (exclude compilation time).
- Check for graph breaks: `TORCH_LOGS="graph_breaks" python train.py`
- If graph breaks exist, fix them and measure the improvement.
- Try `mode="reduce-overhead"` and `mode="max-autotune"` -- report which gives the best result for this model.

### 2D: Gradient Accumulation (8 points)

Implement gradient accumulation to simulate a larger effective batch size:
- Keep micro-batch size at the current value.
- Accumulate over 4 and 8 micro-batches (effective batch sizes of 4x and 8x the micro-batch).
- Divide loss by accumulation steps.

**Key measurements:**
- Does throughput change? (It should not significantly -- accumulation does not speed up individual steps.)
- Does accuracy change with larger effective batch size? (It may -- document the effect.)
- What is the maximum effective batch size before accuracy degrades?

### 2E: Gradient Checkpointing (8 points)

Apply gradient checkpointing to the Transformer layers:
```python
from torch.utils.checkpoint import checkpoint

# In the Transformer forward pass, checkpoint each layer:
for layer in self.layers:
    x = checkpoint(layer, x, use_reentrant=False)
```

**Key measurements:**
- Peak memory reduction.
- Throughput impact (expect 20-30% slowdown).
- With the freed memory, increase batch size until you are back to the same peak memory as before checkpointing. Is net throughput (accounting for larger batch) better or worse?

### 2F: Memory Optimization (10 points)

Apply remaining memory optimizations:
1. Use Flash Attention via `torch.nn.functional.scaled_dot_product_attention`:
   ```python
   # Replace manual attention with:
   attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=False)
   ```
2. With the memory freed from all optimizations (mixed precision + checkpointing + Flash Attention), find the maximum batch size that fits in GPU memory.
3. Train at this maximum batch size. Report accuracy and throughput.

**Key measurements:**
- Peak memory before and after Flash Attention.
- Maximum batch size.
- Throughput at maximum batch size vs previous configurations.

---

## Part 3: Combined Optimization (15 points)

### 3A: All Techniques Together (10 points)

Combine ALL optimizations into a single, fully-optimized training script:
- Optimized DataLoader (pre-tokenized, num_workers tuned, pin_memory, persistent_workers)
- Mixed precision training (FP16 or BF16)
- torch.compile (best mode)
- Flash Attention
- Gradient checkpointing (if memory allows a significantly larger batch size)
- Optimal batch size (largest that fits in memory with all optimizations)
- Gradient accumulation (if an even larger effective batch helps accuracy)

Train for the full number of epochs.

**Record:**
- Final throughput (samples/sec)
- Total training wall-clock time
- Peak GPU memory
- GPU utilization
- Final test accuracy
- Total speedup over baseline

### 3B: Speedup Attribution (5 points)

Create a table that attributes the total speedup to each individual technique:

```
| Optimization            | Standalone Speedup | Cumulative Throughput | Cumulative Speedup |
|-------------------------|--------------------|-----------------------|--------------------|
| Baseline                | 1.0x               |        samp/s         | 1.0x               |
| + DataLoader            |     x               |        samp/s         |     x               |
| + Mixed Precision       |     x               |        samp/s         |     x               |
| + torch.compile         |     x               |        samp/s         |     x               |
| + Flash Attention       |     x               |        samp/s         |     x               |
| + Grad Checkpointing    |     x               |        samp/s         |     x               |
| + Batch Size Increase   |     x               |        samp/s         |     x               |
| **FINAL**               |                     |        samp/s         |     **x**           |
```

Create a waterfall chart visualizing the contribution of each technique to the total speedup.

---

## Part 4: Performance Engineering Report (25 points)

Write a professional performance engineering report. This is the document you would give to a principal engineer or research lead.

### Required Sections

**1. Executive Summary (3 points)**
- One paragraph: the task, the final result, the key finding.
- Headline numbers: baseline throughput, final throughput, total speedup, accuracy delta.

**2. Methodology (3 points)**
- How you measured each metric (include details about warmup, synchronization, number of repetitions).
- Why you applied optimizations in this specific order.
- Any control variables (same random seed, same number of epochs, same dataset split).

**3. Results (7 points)**

Present all results tables:
- Per-optimization comparison table (from Part 3B).
- Memory usage table across configurations.
- Accuracy table across configurations (proving that optimizations did not degrade accuracy).
- Waterfall chart.

**4. Analysis (7 points)**

For each optimization technique, write one paragraph covering:
- Why it works (the underlying mechanism).
- When it helps most and when it does not help.
- Whether it interacted positively or negatively with other techniques.
- Whether the observed speedup matched your expectation (and if not, why).

Specifically address:
- Did the order of optimizations matter? Would a different order yield the same final throughput?
- Which single optimization had the largest impact? Why?
- What was the most surprising result?
- Is GPU utilization now the bottleneck? If not, what is, and what would you try next?

**5. Accuracy Verification (3 points)**

Demonstrate that the optimized pipeline produces the same quality of model:
- Training loss curves: overlay baseline and optimized (they should be similar after accounting for batch size effects).
- Test accuracy: within 0.5% of baseline.
- If accuracy differs by more than 0.5%, explain why and what you would do to close the gap.

**6. Recommendations (2 points)**

Write a prioritized checklist for someone optimizing a similar training pipeline. What should they do first? What can they skip? What depends on their specific hardware?

---

## Target

**Your final optimized pipeline must achieve the fastest possible training time (for 10 epochs to a fixed accuracy threshold) while maintaining accuracy within 0.5% of the FP32 baseline.**

A good result is a 5-8x total speedup over the naive baseline. An excellent result exceeds 10x. The exact number depends on your hardware, but the methodology and documentation matter as much as the raw number.

---

## Deliverables

1. **Naive baseline script** (`baseline.py`): The unoptimized training script.
2. **Optimized script** (`optimized.py`): The fully optimized training script.
3. **Benchmark script** (`benchmark.py`): Script that runs each optimization configuration and collects all metrics.
4. **Results notebook** (`results.ipynb` or `results.py`): All tables, charts, and intermediate measurements.
5. **Performance report** (`performance_report.md` or in the notebook): The full report from Part 4.
6. **Model checkpoint**: Save both the baseline-trained and optimized-trained models for accuracy comparison.

---

## Evaluation Criteria

| Component | Points | Criteria |
|-----------|--------|----------|
| Model implementation | 5 | Correct architecture, parameter count reported |
| Naive baseline | 5 | Trains correctly, all metrics recorded |
| DataLoader optimization | 8 | Systematic sweep, pre-tokenization, throughput measured |
| Mixed precision | 8 | Correct implementation, speed and memory measured |
| torch.compile | 8 | Applied correctly, graph breaks addressed, modes compared |
| Gradient accumulation | 8 | Correct implementation, accuracy impact documented |
| Gradient checkpointing | 8 | Correct implementation, memory-batch tradeoff explored |
| Memory optimization | 10 | Flash Attention, max batch size found |
| Combined optimization | 10 | All techniques integrated, final numbers reported |
| Speedup attribution | 5 | Clear table and waterfall chart |
| Performance report | 25 | Professional, thorough, accurate, insightful |
| **Total** | **100** | |

**Passing score:** 70/100. To pass, you must demonstrate at least 4 different optimization techniques with measured speedups, achieve a total speedup of at least 3x, and maintain accuracy within 1% of baseline.

---

## Stretch Goals

1. **Custom Triton kernel**: Identify the most time-consuming fused operation in the Transformer (e.g., LayerNorm + GELU in the FFN) and write a custom Triton kernel for it. Benchmark against the unfused PyTorch version and against torch.compile's fusion. Document the speedup.

2. **FSDP (Fully Sharded Data Parallel)**: If you have access to multiple GPUs, apply FSDP to your training pipeline. Measure the scaling efficiency: does 2 GPUs give 2x throughput? What about 4 GPUs? Identify the communication overhead.

3. **Dynamic batching**: Implement a collation strategy that groups sequences of similar lengths into the same batch, minimizing padding. Measure the throughput improvement from reduced wasted computation on padding tokens.

4. **Profiling-guided optimization**: After applying all standard optimizations, run the full profiler on the optimized pipeline. Identify the new bottleneck. Is it compute-bound or memory-bound? What would be the next optimization to apply? Implement it if you can.

5. **INT8 training**: Using libraries like `bitsandbytes`, try INT8 optimizer states (8-bit Adam). This reduces optimizer memory by 4x with minimal accuracy impact. Combine with all other optimizations and report the new maximum batch size and throughput.

6. **Reproduce a benchmark**: Find a published training speed benchmark for a similar model (e.g., from MLPerf or a library's benchmark suite). Try to match or exceed their reported throughput. Document any discrepancies and hypothesize why.

7. **Cost analysis**: Compute the dollar cost of training your model to the target accuracy on a cloud GPU (e.g., A100 at $3/hour). Compare the cost using the naive baseline vs the fully optimized pipeline. How much money did your optimization save? Extrapolate: if this were a model that takes 1 week to train at baseline, how much would the optimization save?

---

## A Note on Methodology

In performance engineering, numbers without methodology are meaningless. For every measurement you report, the reader should be able to answer:
- How many times was this measured?
- Was there a warmup period?
- Were CUDA synchronization points used for timing?
- Were other processes running on the GPU?
- Is the number a mean, median, or single observation?
- What is the variance?

If you cannot answer these questions about your own measurements, the measurements are not trustworthy. At a lab like DeepMind, presenting unreproducible benchmarks would undermine your credibility. Get the methodology right.

---

*"The goal of performance engineering is not to write the fastest code. It is to make the fastest possible progress on the research question. Every minute saved on training is a minute gained for the next experiment."*
