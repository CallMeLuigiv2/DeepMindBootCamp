# Assignment 01: Tensor Operations — Mastery Through Repetition

**Module 3, Session 1 | Estimated time: 4-6 hours**

---

## Objective

Tensors are the atoms of everything you will build. This assignment drills tensor operations
until they are reflexive. By the end, you should be able to write any tensor manipulation
without consulting documentation.

---

## Part 1: Tensor Creation and Properties (20 minutes)

Without looking at the docs, create the following tensors. After creating each one, print its
`shape`, `dtype`, `device`, and `stride()`.

1. A 3x4 tensor of zeros (float32).
2. A 5x5 identity matrix.
3. A 1D tensor containing integers from 0 to 99.
4. A 2x3x4 tensor of random values drawn from a standard normal distribution.
5. A tensor from the Python list `[[1, 2], [3, 4], [5, 6]]`.
6. A tensor of 50 evenly spaced values between -1 and 1.
7. A 4x4 tensor of ones on GPU (if available), otherwise CPU.
8. A tensor with the same shape and device as tensor (4), but filled with the value 7.
9. A tensor from a NumPy array. Modify the NumPy array and verify the tensor changed.
10. An uninitialized 1000x1000 tensor. Explain why this is dangerous.

---

## Part 2: Predict the Shape (Broadcasting Exercises)

For each pair of tensors below, **predict the output shape** of `a + b` before running the code.
Write your prediction, then verify. If the operation would fail, explain why.

```python
# 1
a = torch.randn(5, 3)
b = torch.randn(3)

# 2
a = torch.randn(2, 1, 4)
b = torch.randn(3, 1)

# 3
a = torch.randn(3, 1)
b = torch.randn(1, 4)

# 4
a = torch.randn(2, 3, 4)
b = torch.randn(2, 1, 4)

# 5
a = torch.randn(5, 3, 4, 1)
b = torch.randn(4, 2)

# 6
a = torch.randn(3, 4)
b = torch.randn(5, 4)

# 7
a = torch.randn(1, 5, 1, 3)
b = torch.randn(2, 1, 4, 3)

# 8
a = torch.randn(3, 1)
b = torch.randn(2, 1, 4)

# 9
a = torch.randn(6, 1, 3)
b = torch.randn(1, 5, 1)

# 10
a = torch.randn(2, 3)
b = torch.randn(4, 3)
```

For each, write:
- Your predicted shape (or "Error")
- The broadcasting rule that applies
- Verification code

---

## Part 3: Indexing and Slicing Challenges (30 minutes)

Using the tensor `x = torch.arange(60).reshape(3, 4, 5)`, write one-line expressions to extract:

1. The element at position [1, 2, 3].
2. The entire first "slice" along dimension 0 (shape should be 4x5).
3. The last element along dimension 2 for all positions (shape should be 3x4).
4. Every other element along dimension 2 (shape should be 3x4x3).
5. All elements where the value is greater than 40.
6. All elements where the value is divisible by 7.
7. The elements at positions [0, 2] along dimension 0 and [1, 3] along dimension 1.
8. Replace all elements greater than 50 with -1 (use boolean indexing).
9. Use `torch.where` to replace negative values with 0 and leave positive values unchanged,
   given `y = torch.randn(5, 5)`.
10. Use `torch.gather` to select specific elements along dimension 1 given an index tensor.

For each, state the expected output shape before running.

---

## Part 4: Memory Layout Exercises (45 minutes)

This is the section that separates those who use PyTorch from those who understand it.

### Exercise 4.1: Stride Investigation

```python
x = torch.arange(24).reshape(2, 3, 4)
```

1. What is `x.stride()`? Explain what each number means.
2. What is `x.T.stride()`? Is `x.T` contiguous?
3. Create `y = x.permute(2, 0, 1)`. What is `y.stride()`? Is `y` contiguous?
4. Create `z = x[:, ::2, :]`. What is `z.stride()`? Is `z` contiguous? Why?
5. For each non-contiguous tensor above, call `.contiguous()` and verify the stride changes.

### Exercise 4.2: View vs Reshape

For each of the following, predict whether `.view(desired_shape)` will work or raise an error.
If it will error, explain why and provide the fix.

```python
a = torch.arange(12).reshape(3, 4)
a.view(4, 3)      # predict: works or error?

b = a.t()
b.view(12)         # predict: works or error?

c = a[:, :2]
c.view(6)          # predict: works or error?

d = torch.arange(12).reshape(3, 4).contiguous()
d.view(2, 6)       # predict: works or error?

e = torch.arange(12).reshape(3, 4).permute(1, 0)
e.reshape(12)      # predict: works or error?
```

### Exercise 4.3: Storage

```python
x = torch.arange(12).reshape(3, 4)
y = x[1:3, 1:3]
```

1. Print `x.storage()` and `y.storage()`. Are they the same object?
2. What is `y.storage_offset()`? Explain what this number means.
3. Modify `y[0, 0] = 999`. Check `x`. What happened and why?
4. Call `z = y.clone()`. Modify `z[0, 0] = -1`. Check `y`. What happened and why?

---

## Part 5: Implement Matrix Multiplication from Scratch (30 minutes)

Implement matrix multiplication **without using `torch.mm`, `torch.matmul`, `@`, or any
built-in matrix multiplication function**. Use only element-wise operations and sum/reduce.

```python
def manual_matmul(A, B):
    """
    Multiply matrices A (m x n) and B (n x p) -> result (m x p).
    Use only element-wise operations, broadcasting, and reduction.
    Do NOT use torch.mm, torch.matmul, @, or torch.einsum.
    """
    # Your implementation here
    pass
```

Requirements:
1. Handle arbitrary shapes (as long as dimensions are compatible).
2. Raise a clear error if shapes are incompatible.
3. Verify your result matches `torch.mm` for several test cases.
4. Benchmark your implementation vs `torch.mm` for 1000x1000 matrices. Report the time
   difference.

Hint: Think about how broadcasting can help you compute all dot products simultaneously.

---

## Part 6: GPU vs CPU Benchmarking (30 minutes)

*Skip this section if you do not have a CUDA GPU. Document that you skipped it and why.*

Write a benchmarking script that measures the time for the following operations on both CPU and
GPU:

1. Matrix multiplication: `(1000, 1000) @ (1000, 1000)`
2. Element-wise operations: `torch.sin(x)` for x of shape `(10000, 10000)`
3. Reduction: `x.sum()` for x of shape `(10000, 10000)`
4. Small operation: `(10, 10) @ (10, 10)` — is GPU faster here?

For GPU timing, you must use `torch.cuda.synchronize()` before and after each operation, and
use `torch.cuda.Event` for accurate timing. Explain why `time.time()` alone is unreliable for
GPU timing.

```python
# GPU timing template
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# ... operation ...
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

Report your findings in a table:

| Operation | CPU Time (ms) | GPU Time (ms) | Speedup |
|-----------|---------------|---------------|---------|
| Large matmul | | | |
| Element-wise sin | | | |
| Reduction sum | | | |
| Small matmul | | | |

Explain when the GPU is faster and when it is not, and why.

---

## Part 7: 20+ Operations Without Docs (45 minutes)

Write a script that performs each of the following operations. Do not look at the documentation.
If you get stuck, mark the operation and look it up after attempting all of them.

1. Create a random 5x5 matrix and compute its transpose.
2. Compute the element-wise product of two 3x4 tensors.
3. Compute the dot product of two 1D tensors of length 10.
4. Stack three 3x3 tensors along a new dimension 0 (result: 3x3x3).
5. Concatenate three 3x3 tensors along dimension 1 (result: 3x9).
6. Compute the mean and standard deviation of a tensor along dimension 1.
7. Find the indices of the maximum values along dimension 0.
8. Sort a 2D tensor along dimension 1 and get the sorted values and indices.
9. Clamp all values in a tensor to the range [-1, 1].
10. Compute the L2 norm of each row in a 5x10 matrix.
11. Create a diagonal matrix from a 1D tensor.
12. Compute the element-wise absolute value of a tensor with negative values.
13. Apply a boolean mask to select elements and compute their sum.
14. Repeat a 2x3 tensor 4 times along dimension 0 and 2 times along dimension 1.
15. Flatten a 2x3x4 tensor to 1D.
16. Unsqueeze a 1D tensor of length 5 to shape (1, 5, 1).
17. Compute the cumulative sum along dimension 0 of a 4x4 matrix.
18. Find unique elements in a tensor and their counts.
19. Perform an outer product of two 1D tensors without using `torch.outer`.
20. Create a lower triangular mask for a 5x5 matrix.
21. Compute the top-3 values and indices of each row in a 10x20 matrix.
22. Use `torch.einsum` to compute a batched matrix multiplication.

After completing all operations, go back and look up any you struggled with. Write a brief note
about what you found confusing and what the correct approach is.

---

## Deliverables

Submit a single Python script or Jupyter notebook containing:

1. All exercises from Parts 1-7 with clear section headers.
2. All shape predictions written as comments before the verification code.
3. The `manual_matmul` function with correctness verification and benchmarks.
4. The GPU benchmarking results table (or a note explaining why it was skipped).
5. A "Lessons Learned" section at the end listing:
   - Operations you found most confusing.
   - Key insights about memory layout.
   - Any surprises from the benchmarking.

---

## Evaluation Criteria

- **Completeness:** All exercises attempted. No sections skipped without justification.
- **Correctness:** All predictions verified. All implementations produce correct results.
- **Understanding:** Explanations demonstrate comprehension, not just working code.
- **Memory layout:** You can explain strides, contiguity, and view vs reshape in your own words.
- **Broadcasting:** You can predict output shapes without running the code.

---

## Stretch Goals

1. **Implement batch matrix multiplication from scratch** (without `torch.bmm` or `@`), handling
   a batch dimension.

2. **Write a tensor operation profiler** that times each of the 22 operations from Part 7 and
   ranks them by speed. Explore which operations are memory-bound vs compute-bound.

3. **Implement Einstein summation from scratch** for a subset of operations (matrix multiply,
   trace, batch outer product) using only basic indexing and reduce operations.

4. **Explore memory fragmentation:** Allocate and free tensors of varying sizes on GPU. Monitor
   `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`. Explain why reserved
   memory can be much larger than allocated memory.
