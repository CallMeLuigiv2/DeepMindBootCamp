# Assignment 1: Attention Mechanisms from Scratch

## Overview

You will implement the core attention mechanisms that power every modern Transformer — from scratch, using only PyTorch tensor operations. No `nn.MultiheadAttention`, no `nn.TransformerEncoder`. Every matrix multiplication, every softmax, every mask must be your code.

This is not a wrapper exercise. When you finish, you should be able to implement attention on a whiteboard from memory.

**Estimated time**: 8-12 hours
**Prerequisites**: Module 7 Sessions 1-2, strong PyTorch fluency

---

## Part 1: Scaled Dot-Product Attention

### Task

Implement the function `scaled_dot_product_attention(Q, K, V, mask=None)` using only:
- `torch.matmul` (or the `@` operator)
- `torch.softmax` (or `F.softmax`)
- Basic arithmetic operations (`/`, `**`)
- `torch.masked_fill` or equivalent for masking

### Requirements

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor, shape (batch, seq_len_q, d_k)
           or (batch, num_heads, seq_len_q, d_k) for multi-head usage
        K: Key tensor, shape (batch, seq_len_k, d_k)
           or (batch, num_heads, seq_len_k, d_k)
        V: Value tensor, shape (batch, seq_len_k, d_v)
           or (batch, num_heads, seq_len_k, d_v)
        mask: Optional boolean mask, shape broadcastable to (..., seq_len_q, seq_len_k)
              True/1 at positions to MASK (set to -inf before softmax)

    Returns:
        output: Weighted sum of values, same shape as Q but with d_v in last dim
        attention_weights: Softmax attention weights, shape (..., seq_len_q, seq_len_k)
    """
    pass
```

### Verification

1. **Shape test**: for Q (2, 5, 64), K (2, 10, 64), V (2, 10, 32), verify output shape is (2, 5, 32) and weights shape is (2, 5, 10).

2. **Row sum test**: verify that each row of attention_weights sums to 1.0 (within floating-point tolerance).

3. **Identity test**: if Q = K (queries equal keys), the diagonal of the attention weight matrix should have the highest values (each token attends most to itself, all else being equal).

4. **Worked example**: reproduce the worked example from the notes (3 tokens, d_k=4). Verify your output matches the hand-computed values.

---

## Part 2: Multi-Head Attention

### Task

Implement a complete `MultiHeadAttention` class from scratch.

### Requirements

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: Total model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        # Initialize W_Q, W_K, W_V, W_O as nn.Linear layers (without bias)
        # Compute d_k = d_model // num_heads
        pass

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  Optional, broadcastable to (batch, num_heads, seq_len_q, seq_len_k)

        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        # 1. Project Q, K, V using linear layers
        # 2. Reshape to (batch, num_heads, seq_len, d_k)
        # 3. Apply scaled_dot_product_attention per head (in parallel via batched matmul)
        # 4. Concatenate heads: (batch, seq_len, num_heads * d_k) = (batch, seq_len, d_model)
        # 5. Apply output projection W_O
        pass
```

### Verification

1. **Shape test**: d_model=512, num_heads=8. Input (4, 20, 512). Output should be (4, 20, 512).

2. **Parameter count**: verify your module has exactly 4 * d_model^2 parameters (no biases). For d_model=512: 1,048,576 parameters.

3. **Self-attention test**: call with query=key=value=x. Verify it runs without error and produces the correct output shape.

4. **Cross-attention test**: call with query=(batch, 10, d_model), key=value=(batch, 20, d_model). Verify output is (batch, 10, d_model) — the output sequence length matches the query length, not the key length.

---

## Part 3: The Copy Task

### Task

Train your multi-head attention on a simple copy task to verify that attention can learn to "attend to the right positions."

### Setup

```python
# Generate data: sequences of random integers
# Input:  [3, 7, 1, 5, 9] (random sequence)
# Target: [3, 7, 1, 5, 9] (same sequence — the model must copy it)

# The model must learn to attend position i of the output to position i of the input.
# After training, the attention weight matrix should be approximately the identity matrix.
```

### Implementation

1. Create a simple model: Embedding -> MultiHeadAttention (cross-attention) -> Linear -> Softmax
2. Generate random integer sequences of length 5-10
3. Train with cross-entropy loss
4. After training, extract and print the attention weights for a test sequence
5. The attention weights should show a clear diagonal pattern (position i attends to position i)

### Deliverable

- Training loss curve (should decrease to near zero)
- Attention weight visualization for 3 test sequences (should show near-diagonal pattern)
- Brief analysis: how many epochs did it take to converge? Did multi-head outperform single-head?

---

## Part 4: Attention Weight Visualization

### Task

Visualize attention patterns to build intuition about what attention learns.

### Requirements

1. Create a function that takes attention weights (num_heads, seq_len_q, seq_len_k) and token labels, and produces a matplotlib heatmap for each head.

2. Visualize the copy task attention (from Part 3) — should show diagonal patterns.

3. Create a more interesting test case: a sequence with repeated tokens, e.g., [A, B, C, A, B]. Train the copy task with this input. Do the attention weights correctly handle repeated tokens? Which head focuses on position, and which on content?

4. Create a "reverse" task: input [3, 7, 1, 5, 9], target [9, 5, 1, 7, 3]. Visualize the attention — it should show an anti-diagonal pattern.

### Deliverable

- Heatmap visualizations for at least 4 different scenarios
- Written analysis of what each attention head has learned in each scenario

---

## Part 5: Causal (Masked) Attention

### Task

Implement causal attention for autoregressive models.

### Requirements

1. Implement `create_causal_mask(seq_len)` that returns an upper-triangular mask.

2. Modify your `scaled_dot_product_attention` to accept this mask and apply it correctly (add -inf to masked positions before softmax).

3. **Verification**: for a 5-token sequence, print the attention weights with causal masking:
   - Token 0 should attend ONLY to token 0 (weight = 1.0)
   - Token 1 should attend to tokens 0-1 (weights sum to 1.0)
   - Token 4 should attend to tokens 0-4 (weights sum to 1.0)
   - All weights above the diagonal should be exactly 0.0

4. **Autoregressive generation test**: implement a simple autoregressive sequence generation function:
   ```python
   def generate(model, start_token, max_len):
       """Generate a sequence token by token using causal attention."""
       tokens = [start_token]
       for _ in range(max_len - 1):
           # Run model on current sequence with causal mask
           # Sample or argmax the next token
           # Append to tokens
           pass
       return tokens
   ```

### Deliverable

- Causal mask implementation and verification output
- Attention weight heatmaps showing the triangular pattern
- Working autoregressive generation function

---

## Part 6: Comparison with PyTorch's nn.MultiheadAttention

### Task

Verify that your implementation produces the same output as PyTorch's built-in `nn.MultiheadAttention`.

### Requirements

1. Create an instance of your `MultiHeadAttention` and PyTorch's `nn.MultiheadAttention` with the same hyperparameters (d_model=64, num_heads=4).

2. Copy the weights from PyTorch's module to yours (or vice versa). Note: PyTorch packs Q, K, V into a single `in_proj_weight` matrix — you will need to understand this layout.

   ```python
   # PyTorch's nn.MultiheadAttention stores weights as:
   # in_proj_weight: (3 * d_model, d_model) = [W_Q; W_K; W_V] stacked
   # out_proj.weight: (d_model, d_model) = W_O
   ```

3. Run both modules on the same input tensor. Compare outputs using `torch.allclose(output_yours, output_pytorch, atol=1e-6)`.

4. Test with and without masking.

### Deliverable

- Code that copies weights between the two implementations
- Assertion that outputs match (within floating-point tolerance) for at least 5 random inputs
- Brief discussion of any differences in API or convention (e.g., PyTorch uses (seq_len, batch, d_model) by default unless `batch_first=True`)

---

## Part 7: Speed Benchmark

### Task

Measure the computational cost of your attention implementation vs PyTorch's optimized version.

### Requirements

1. Benchmark both implementations for:
   - Sequence lengths: 64, 128, 256, 512, 1024
   - d_model = 256, num_heads = 8
   - Batch size = 32

2. Measure:
   - Forward pass time (average over 100 runs, excluding warmup)
   - Peak memory usage

3. Plot sequence length vs time for both implementations on the same graph.

4. Explain WHY PyTorch's implementation is faster (hint: fused kernels, memory access patterns, Flash Attention in recent PyTorch versions).

### Deliverable

- Benchmark results table
- Plot of sequence length vs forward pass time
- Written analysis explaining the performance difference

---

## Deliverables Summary

| Part | Deliverable | Weight |
|------|-------------|--------|
| 1 | `scaled_dot_product_attention` function + verification tests | 15% |
| 2 | `MultiHeadAttention` class + verification tests | 20% |
| 3 | Copy task: training code, loss curve, attention visualization | 15% |
| 4 | Attention visualization functions + analysis | 10% |
| 5 | Causal masking + autoregressive generation | 20% |
| 6 | PyTorch comparison + weight copying code | 10% |
| 7 | Speed benchmark + analysis | 10% |

Submit as a Jupyter notebook with clear sections, inline outputs, and written analysis.

---

## Grading Criteria

**Passing (70%+)**:
- Parts 1-2 are correct and pass all verification tests
- Part 3 shows a working copy task with decreasing loss
- Part 5 causal masking is correct

**Distinction (85%+)**:
- All of the above, plus:
- Part 4 attention visualizations are clear and well-analyzed
- Part 6 outputs match PyTorch within tolerance
- Part 7 benchmark is complete with reasonable analysis

**Exceptional (95%+)**:
- All of the above, plus:
- Copy task analysis includes multi-head vs single-head comparison
- Attention visualizations include creative test cases (repeated tokens, reversals, etc.)
- Benchmark analysis demonstrates understanding of GPU memory hierarchy

---

## Stretch Goals

For those who want to push further:

1. **Conceptual Flash Attention**: implement a simplified version of Flash Attention's tiling strategy. You do not need CUDA kernels — implement the tiling and online softmax in pure Python/PyTorch to demonstrate the algorithm. Compare memory usage (measured via `torch.cuda.max_memory_allocated()`) against your standard implementation.

2. **Additive (Bahdanau) Attention**: implement the original additive attention mechanism and compare it to dot-product attention on the copy task. Is one faster to converge? Is one more expressive?

3. **Linear Attention**: implement attention with a kernel approximation (e.g., using random features) that avoids the softmax and achieves O(n) complexity. Compare quality vs speed against standard attention.

4. **Attention with Relative Position Bias**: add a learnable relative position bias to your attention scores (as in T5 or ALiBi). Show that this improves performance on a task where position matters.

---

## Common Pitfalls

1. **Forgetting to scale**: without dividing by sqrt(d_k), training will be unstable. If your loss is not decreasing, check this first.

2. **Wrong mask convention**: some implementations use True for "attend" and False for "mask." Others use the opposite. Be consistent and document your convention.

3. **Transposing the wrong dimensions**: when reshaping for multi-head attention, it is easy to transpose (seq_len, num_heads) instead of getting (num_heads, seq_len). Print shapes at every step.

4. **Not using contiguous() before view()**: after transpose(), the tensor may not be contiguous in memory. Call `.contiguous()` before `.view()` to avoid runtime errors.

5. **PyTorch default shapes**: `nn.MultiheadAttention` expects (seq_len, batch, d_model) by default, not (batch, seq_len, d_model). Use `batch_first=True` or transpose accordingly.
