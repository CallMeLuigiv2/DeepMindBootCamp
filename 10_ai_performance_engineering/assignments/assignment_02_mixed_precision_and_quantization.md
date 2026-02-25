# Assignment 02: Mixed Precision Training and Quantization

**Module:** 10 -- AI Performance Engineering
**Session:** 3 -- Mixed Precision and Quantization
**Estimated Time:** 8-10 hours
**Difficulty:** Intermediate-Advanced

---

## Overview

Mixed precision training and quantization are among the highest-impact, lowest-effort optimizations in the performance engineering toolkit. Mixed precision gives you a near-free 1.5-2x training speedup. Quantization gives you 2-4x inference speedup. Together, they represent the practical difference between a model that can be deployed and one that sits in a notebook.

In this assignment, you will implement both from scratch, measure everything, and build a deep understanding of when and why these techniques work -- and when they do not.

---

## Part 1: Mixed Precision Training (35 points)

### 1A: Baseline FP32 Training (5 points)

Set up a standard training pipeline. Use ResNet-18 on CIFAR-10 (or CIFAR-100 for more signal on accuracy impacts).

Your baseline must:
- Train for at least 20 epochs with a reasonable learning rate schedule (e.g., cosine annealing).
- Use AdamW optimizer with lr=1e-3.
- Record: final test accuracy, best test accuracy, training samples/sec, peak GPU memory.
- Save the trained model weights.

Present the baseline results:

```
| Metric                  | FP32 Baseline |
|-------------------------|---------------|
| Final test accuracy     |               |
| Best test accuracy      |               |
| Training throughput     |     samp/s    |
| Peak GPU memory         |     MB        |
| Training time (20 ep)   |     sec       |
```

### 1B: Implement Mixed Precision with FP16 (15 points)

Add mixed precision training using `torch.amp.autocast` and `torch.amp.GradScaler`.

**Requirements:**
1. Wrap the forward pass and loss computation in `autocast(device_type='cuda', dtype=torch.float16)`.
2. Use `GradScaler` for loss scaling: `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()`.
3. If using gradient clipping, call `scaler.unscale_(optimizer)` before `clip_grad_norm_`.
4. Use autocast during validation as well.

**Measurements (all mandatory):**
- Training throughput (samples/sec): compare to baseline.
- Peak GPU memory: compare to baseline.
- Final test accuracy: must be within 0.5% of the FP32 baseline.
- Training time for 20 epochs.
- Monitor the GradScaler's scale factor over training. Plot it. How often does it reduce (indicating overflow)?

**Analysis questions (answer each in 2-4 sentences):**
1. Why can matmul and conv2d run safely in FP16 while softmax and layer norm should not?
2. What would happen if you trained with FP16 but without the GradScaler? Try it for a few epochs and report what happens to the loss.
3. Why does mixed precision reduce memory usage? Which component of GPU memory (parameters, gradients, activations, optimizer states) benefits most?

### 1C: BFloat16 Comparison (10 points)

If your GPU supports BFloat16 (Ampere/A100 or newer -- check with `torch.cuda.is_bf16_supported()`), repeat the mixed precision experiment with BFloat16.

**Requirements:**
1. Use `autocast(device_type='cuda', dtype=torch.bfloat16)`.
2. Train *without* GradScaler (explain why it is not needed for BF16).
3. Measure the same metrics as 1B.

**Comparison table:**

```
| Metric                  | FP32      | FP16 + Scaler | BF16      |
|-------------------------|-----------|---------------|-----------|
| Final test accuracy     |           |               |           |
| Training throughput     |           |               |           |
| Peak GPU memory         |           |               |           |
| Training time (20 ep)   |           |               |           |
| Overflow events         | N/A       |               | N/A       |
```

If your GPU does not support BF16, explain why BF16 is preferred over FP16 when available (hint: exponent range), and fill in the BF16 column with "N/A -- not supported on this GPU."

### 1D: Numerical Exploration (5 points)

Write a short script that demonstrates the numerical properties of each format:

1. Show the smallest positive normal number in FP32, FP16, and BF16.
2. Demonstrate precision loss: compute `1.0 + x` for x = 1e-4, 1e-5, 1e-6 in each format. At what point does the addition have no effect?
3. Simulate gradient underflow: create a tensor with value 1e-6 (a realistic gradient magnitude). Cast it to FP16. What happens? Cast to BF16. What happens?
4. Show why loss scaling works: multiply the 1e-6 value by 1024 before casting to FP16. Is the value preserved after cast and divide-by-1024?

Present results in a clean table.

---

## Part 2: Post-Training Quantization (35 points)

### 2A: Dynamic Quantization (10 points)

Take the trained FP32 model from Part 1A (or a pre-trained ResNet-18 / BERT-base if you prefer a model with more Linear layers).

Apply dynamic quantization:
```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

**Measurements:**
- Inference speed (latency per batch) on CPU: FP32 vs INT8.
- Test accuracy: FP32 vs INT8.
- Model size on disk: FP32 vs INT8.

**Important note:** Dynamic quantization primarily speeds up CPU inference for models dominated by Linear layers. For CNNs, the impact is smaller because Conv2d is the dominant operation. If using ResNet, you may see modest speedup. If using a Transformer-based model, the speedup will be more dramatic. Explain why in your analysis.

### 2B: Static Quantization (15 points)

Apply static quantization to the same model. This requires:

1. **Prepare the model**: add QuantStub and DeQuantStub wrappers (or use the newer `torch.quantization.quantize_fx` API).
2. **Insert observers**: these record the activation ranges during calibration.
3. **Calibrate**: run 100-500 samples from the training set through the prepared model.
4. **Convert**: convert the model to a quantized version using the observed ranges.

**Measurements:**
- Inference speed on CPU: FP32 vs dynamic INT8 vs static INT8.
- Test accuracy: FP32 vs dynamic INT8 vs static INT8.
- Model size on disk for all three.

**Analysis questions:**
1. Why is static quantization generally faster than dynamic quantization? (Hint: when are activation quantization parameters computed?)
2. How sensitive is static quantization to the calibration dataset? Try calibrating with 10, 100, and 500 samples. Does accuracy change?
3. Observe which layers contribute most to accuracy degradation. You can do this by quantizing one layer at a time and measuring accuracy. Which layers are most sensitive?

### 2C: Transformer INT8 Quantization (10 points)

For this part, use a Transformer-based model. Options:
- A small BERT model (e.g., `bert-base-uncased` from Hugging Face) fine-tuned on a text classification task (SST-2 or similar).
- The small Transformer from your earlier modules.
- A pre-trained DistilBERT.

Apply INT8 quantization (dynamic is sufficient; static is a stretch goal):

**Measurements:**
- Inference latency per sample: FP32 vs INT8.
- Accuracy on the test set: FP32 vs INT8.
- Model size: FP32 vs INT8.
- Throughput (samples/sec) at batch sizes 1, 8, 32.

**Key observation to document:** Transformers are dominated by Linear layers (QKV projections, FFN layers). Dynamic quantization should produce larger speedups here than for CNNs. Verify this claim with your measurements.

---

## Part 3: The Comprehensive Comparison Table (15 points)

### 3A: Build the Table (10 points)

For a single model (ResNet-18, your Transformer, or both), create the following comprehensive comparison table:

```
| Configuration        | Train Speed | Infer Speed | Infer Latency | Peak Memory | Model Size | Accuracy |
|----------------------|-------------|-------------|---------------|-------------|------------|----------|
| FP32 (baseline)      |             |             |               |             |            |          |
| FP16 mixed precision |             |    (*)      |     (*)       |             |            |          |
| BF16 mixed precision |             |    (*)      |     (*)       |             |            |          |
| FP16 inference only  | N/A         |             |               |             |            |          |
| Dynamic INT8 (CPU)   | N/A         |             |               | N/A         |            |          |
| Static INT8 (CPU)    | N/A         |             |               | N/A         |            |          |
```

(*) For FP16/BF16 inference, use autocast during inference and measure on GPU.

**Units:**
- Train Speed: samples/sec
- Infer Speed: samples/sec (batch of 32)
- Infer Latency: ms per single sample
- Peak Memory: MB (GPU) or MB (CPU) as appropriate
- Model Size: MB on disk
- Accuracy: % on test set

### 3B: Plot the Pareto Frontier (5 points)

Create a scatter plot with:
- X-axis: inference latency (ms per sample)
- Y-axis: test accuracy (%)

Each point represents one configuration (FP32, FP16, BF16, dynamic INT8, static INT8). Label each point.

Draw the Pareto frontier: the set of configurations where no other configuration is both faster AND more accurate.

**Analysis:** Which configuration offers the best accuracy-speed tradeoff for: (a) a research setting where accuracy matters most, (b) a production setting where latency matters most, (c) an edge deployment where model size matters most?

---

## Part 4: Written Analysis (15 points)

Answer the following questions. Aim for depth, not length. Each answer should be 3-6 sentences with specific numbers from your experiments.

1. **Mixed precision tradeoffs**: You observed that mixed precision training gives approximately ___ speedup with approximately ___ memory reduction and approximately ___ accuracy change. Given these numbers, is there ever a reason NOT to use mixed precision? When?

2. **FP16 vs BF16**: Based on your numerical exploration (Part 1D) and training results, explain the practical difference between FP16 and BF16. Why is BF16 considered strictly better for training when hardware supports it?

3. **Quantization accuracy degradation**: Your dynamic INT8 model lost approximately ___% accuracy compared to FP32. Your static INT8 model lost approximately ___%. Why does quantization degrade accuracy? What happens to a weight value of 0.0037 when it is quantized to INT8 with a range of [-1, 1]?

4. **CNN vs Transformer quantization**: Compare the quantization speedups you observed for the CNN (ResNet) vs the Transformer. Which benefited more? Why? (Hint: think about which operations dominate each architecture.)

5. **The deployment decision**: You are deploying a model to production. You can serve it in FP32 on a GPU, FP16 on a GPU, or INT8 on a CPU. The GPU costs $3/hour and the CPU costs $0.50/hour. Based on your latency measurements, which option gives the best throughput per dollar? Show your calculation.

---

## Deliverables

1. **Training scripts**: `train_fp32.py`, `train_mixed_precision.py` (or a single configurable script).
2. **Quantization script**: `quantize_and_benchmark.py` with all quantization methods.
3. **Numerical exploration**: `numerical_precision.py` or notebook.
4. **Results notebook**: All tables, plots (Pareto frontier, GradScaler plot), and analysis.
5. **Comparison table**: The comprehensive table from Part 3A, clearly formatted.
6. **Written analysis**: Part 4 answers, in the notebook or as a separate markdown file.

---

## Evaluation Criteria

| Component | Points | Criteria |
|-----------|--------|----------|
| FP32 baseline | 5 | Trains correctly, all metrics recorded |
| FP16 mixed precision | 15 | Correct autocast + GradScaler, metrics measured, analysis questions |
| BF16 comparison | 10 | Correct implementation or thorough explanation, comparison table |
| Numerical exploration | 5 | All 4 demonstrations correct, clean presentation |
| Dynamic quantization | 10 | Correct implementation, CPU inference benchmarked |
| Static quantization | 15 | Correct calibration workflow, sensitivity analysis |
| Transformer INT8 | 10 | Correct quantization, batch size sweep |
| Comparison table | 10 | All configurations, correct units, complete |
| Pareto frontier plot | 5 | Correct plot, frontier identified, deployment analysis |
| Written analysis | 15 | Thoughtful, specific, uses experimental numbers |
| **Total** | **100** | |

**Passing score:** 70/100. To pass, you must demonstrate working mixed precision training with measurable speedup AND at least one form of post-training quantization with measurable inference improvement.

---

## Stretch Goals

1. **Quantization-Aware Training (QAT)**: Implement quantization-aware training for your model. Insert fake quantization nodes, train for 5-10 epochs, then convert to a fully quantized model. Compare the accuracy to post-training quantization. QAT should recover most of the accuracy lost by post-training quantization.

2. **BFloat16 everywhere**: If your hardware supports it, try an entire training run in BF16 (not mixed precision -- pure BF16 for everything including optimizer states). Measure accuracy impact. Modern research (e.g., from Google's TPU work) suggests this can work well.

3. **INT4 quantization**: Using a library like `bitsandbytes` or GPTQ, try 4-bit quantization on a Transformer model. How much accuracy is lost compared to INT8? How much additional speedup and size reduction do you get?

4. **Quantization sensitivity analysis**: For each layer in your model, apply quantization to only that layer (keep all others in FP32) and measure accuracy. Create a per-layer sensitivity heatmap. Use this to design a mixed-precision quantization strategy where sensitive layers stay in FP32 and others are INT8.

5. **FP8 exploration**: If you have access to an H100 GPU, explore FP8 (E4M3 and E5M2) training. These new formats are designed specifically for deep learning and are supported by the latest Transformer Engine library.

---

*"Precision is not accuracy. A float32 model that takes three times longer to train is not three times more accurate. Understand the numbers your model actually needs."*
