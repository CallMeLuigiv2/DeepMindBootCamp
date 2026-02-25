# Assignment 01: RNN and LSTM from Scratch

## Overview

In this assignment, you will implement the fundamental recurrent architectures from raw tensor operations. No `nn.RNN`, no `nn.LSTM` — just you, matrix multiplications, and activation functions. By building these cells by hand, you will internalize the mechanics of gating, cell state updates, and sequential processing in a way that reading equations alone cannot provide.

You will then train your implementations on a character-level language model and observe, empirically, the vanishing gradient problem and how LSTM solves it.

**Estimated time**: 8-12 hours

**Prerequisites**: PyTorch fundamentals (Module 03), backpropagation (Module 04), understanding of the RNN and LSTM equations from this module's notes.

---

## Part 1: Implement a Vanilla RNN Cell (25%)

### Task

Implement a class `ManualRNNCell` that computes:

```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
```

### Requirements

1. Your class should accept `input_size` and `hidden_size` as constructor arguments.
2. Use Xavier initialization for weight matrices.
3. Initialize biases to zero.
4. Implement a `forward(self, x_t, h_prev)` method that:
   - Takes input `x_t` of shape `(batch_size, input_size)` and previous hidden state `h_prev` of shape `(batch_size, hidden_size)`.
   - Returns `h_t` of shape `(batch_size, hidden_size)`.
5. Implement a `parameters(self)` method returning a list of all learnable tensors (with `requires_grad=True`).

### Verification

After implementing your cell, verify it against `nn.RNNCell`:

```python
import torch
import torch.nn as nn

# Your implementation
manual_cell = ManualRNNCell(input_size=10, hidden_size=20)

# PyTorch's implementation
torch_cell = nn.RNNCell(input_size=10, hidden_size=20)

# Copy weights from your cell to PyTorch's cell (or vice versa)
# Then verify that both produce the same output for the same input
x = torch.randn(5, 10)
h = torch.randn(5, 20)

# Outputs should match within floating-point tolerance
manual_output = manual_cell.forward(x, h)
torch_output = torch_cell(x, h)
assert torch.allclose(manual_output, torch_output, atol=1e-6)
```

Write this verification as a test function.

---

## Part 2: Implement an LSTM Cell (30%)

### Task

Implement a class `ManualLSTMCell` that computes all four gates:

```
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)     # Forget gate
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)     # Input gate
g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)        # Candidate
C_t = f_t * C_{t-1} + i_t * g_t                # Cell state update
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)     # Output gate
h_t = o_t * tanh(C_t)                          # Hidden state
```

### Requirements

1. Constructor accepts `input_size` and `hidden_size`.
2. **Initialize the forget gate bias to 1.0.** All other biases to 0.0. This is critical — explain in a comment why you are doing this.
3. Use Xavier initialization for all weight matrices.
4. Implement `forward(self, x_t, h_prev, c_prev)` that:
   - Takes `x_t` (batch_size, input_size), `h_prev` (batch_size, hidden_size), `c_prev` (batch_size, hidden_size).
   - Returns `(h_t, c_t)` — both of shape (batch_size, hidden_size).
5. Implement `parameters(self)`.

### Verification

Verify against `nn.LSTMCell`. Note: PyTorch stores LSTM weights as a single concatenated matrix in the order `[W_ii, W_if, W_ig, W_io]` (input, forget, candidate/cell, output). You will need to carefully map your separate matrices to PyTorch's format to verify equivalence.

```python
# Verify your LSTM cell matches nn.LSTMCell
torch_cell = nn.LSTMCell(input_size=10, hidden_size=20)

# Copy weights appropriately
# ...

x = torch.randn(5, 10)
h = torch.randn(5, 20)
c = torch.randn(5, 20)

manual_h, manual_c = manual_cell.forward(x, h, c)
torch_h, torch_c = torch_cell(x, (h, c))

assert torch.allclose(manual_h, torch_h, atol=1e-6)
assert torch.allclose(manual_c, torch_c, atol=1e-6)
```

---

## Part 3: Character-Level Language Model (25%)

### Task

Build a character-level language model: given a sequence of characters, predict the next character. Train this model using both your ManualRNNCell and ManualLSTMCell.

### Dataset

Use a publicly available text corpus. Good options:
- Shakespeare's collected works (~1.1M characters): available at `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Any other plain text of at least 100K characters.

### Requirements

1. **Data preparation**:
   - Build a character vocabulary (char-to-index and index-to-char mappings).
   - Create training sequences of a fixed length (start with length 50).
   - Implement a simple data loader that yields batches of (input_sequence, target_sequence) pairs, where target is input shifted by one character.

2. **Model architecture**:
   - Embedding layer: `nn.Embedding(vocab_size, embed_size)` — you may use the PyTorch module for this.
   - Your manual RNN or LSTM cell, applied across all time steps in a loop.
   - Output projection: a linear layer mapping hidden_size to vocab_size.
   - Cross-entropy loss.

3. **Training loop**:
   - Implement truncated BPTT: process the text in chunks, detaching the hidden state between chunks.
   - Use Adam optimizer with learning rate 1e-3.
   - Implement gradient clipping with max_norm=5.0.
   - Train for at least 20 epochs.
   - Log training loss every 100 batches.

4. **Comparison experiments**:
   - Train the RNN model on sequences of length 25, 50, 100, and 200.
   - Train the LSTM model on the same sequence lengths.
   - For each, record the final training loss and validation loss.
   - Plot training curves (loss vs. epoch) for all configurations on a single plot.

### Expected Observations

- Both RNN and LSTM should work well on short sequences (length 25).
- On longer sequences (100+), the RNN's performance should degrade noticeably while the LSTM should continue to learn effectively.
- The generated text from the LSTM model should show more coherent long-range structure (matching quotes, maintaining paragraph structure) compared to the RNN.

---

## Part 4: Gradient Analysis (20%)

### Task

Empirically demonstrate the vanishing gradient problem in RNNs and show how LSTM mitigates it.

### Requirements

1. **Gradient norm tracking**:
   - During training, record the gradient norm of the loss with respect to the hidden state at each time step.
   - Specifically, after the forward pass, retain the hidden states at each time step (store them in a list). After `loss.backward()`, examine the `.grad` attribute of each hidden state.
   - Plot gradient norms as a function of time step distance from the loss (i.e., how far back in the sequence the gradient must travel).

2. **Gradient clipping experiment**:
   - Train the RNN model WITHOUT gradient clipping. Record any NaN losses or exploding gradients.
   - Train with gradient clipping (max_norm=5.0). Show that training stabilizes.
   - Plot the gradient norm before clipping over the course of training. Show how often the gradients would have been problematic without clipping.

3. **Visualization**:
   - Create a plot with two subplots:
     - Left: Gradient norms vs. time step distance for the vanilla RNN. Should show exponential decay.
     - Right: Gradient norms vs. time step distance for the LSTM. Should show much more stable gradients.
   - Title the plot "Vanishing Gradients: RNN vs LSTM."

### Expected Output

The RNN gradient plot should show a clear exponential decay — gradients from 50 steps back are orders of magnitude smaller than gradients from 5 steps back. The LSTM plot should show relatively stable gradient norms across all time steps.

---

## Deliverables

Submit a single Jupyter notebook (or Python script with clear sections) containing:

1. **ManualRNNCell implementation** with verification test.
2. **ManualLSTMCell implementation** with verification test.
3. **Character-level language model** with training loops for both RNN and LSTM.
4. **Training curves** comparing RNN vs LSTM across sequence lengths.
5. **Gradient analysis plots** showing vanishing gradients in RNN and stable gradients in LSTM.
6. **Generated text samples** from both models (at least 500 characters each).
7. **Brief written analysis** (1-2 paragraphs) interpreting your results: Why does the LSTM outperform the RNN on longer sequences? What do the gradient plots tell you?

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Correctness | 30% | RNN and LSTM cells produce correct outputs, verified against PyTorch |
| Training | 25% | Models train successfully, loss decreases, generated text is somewhat coherent |
| Gradient analysis | 25% | Clear demonstration of vanishing gradients and LSTM's advantage |
| Code quality | 10% | Clean, well-commented code with clear variable names |
| Analysis | 10% | Written interpretation shows understanding of the underlying phenomena |

---

## Stretch Goals

If you finish early and want to go deeper:

1. **Implement a GRU cell from scratch.** Add it to the comparison experiments. Does it match LSTM performance? Is it faster to train?

2. **Experiment with sequence length.** Find the approximate sequence length at which the vanilla RNN completely fails to learn long-range dependencies. Can you relate this to the eigenvalues of your W_hh matrix?

3. **Implement orthogonal initialization** for W_hh and compare training dynamics to Xavier initialization. Does orthogonal initialization help the vanilla RNN with longer sequences?

4. **Temperature sampling.** Implement temperature-controlled sampling for text generation. Show how temperature affects the diversity vs. coherence trade-off.

```python
def sample_with_temperature(logits, temperature=1.0):
    """Sample from logits with temperature scaling."""
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

5. **Implement layer normalization** within the LSTM cell (applying layer norm to each gate's pre-activation). Does this improve training stability on very long sequences?

---

## Common Pitfalls

- **Forgetting to detach the hidden state** in truncated BPTT. If you do not call `.detach()` on the hidden state between chunks, the computation graph grows without bound and you will run out of memory.

- **Wrong weight matrix shapes.** The concatenation `[h_{t-1}, x_t]` has size `hidden_size + input_size`. Make sure your weight matrices have the right number of columns.

- **Not initializing the forget gate bias.** If you leave it at 0, the sigmoid output starts at 0.5, which means the LSTM forgets half the cell state at every step from the beginning. This makes early training unnecessarily difficult.

- **Confusing PyTorch's weight layout.** When verifying against `nn.LSTMCell`, remember that PyTorch stores the gates in the order (input_gate, forget_gate, cell_gate, output_gate), which differs from the (forget, input, cell, output) order commonly used in textbooks.

---

## A Note on Why We Do This

You might wonder: if `nn.LSTM` exists and is faster, why implement it from scratch? Three reasons.

First, understanding. You cannot claim to understand the LSTM if you have not implemented every gate. The equations are deceptively simple — the understanding comes from wrestling with the shapes, the concatenations, and the initializations.

Second, debugging. When your model behaves unexpectedly, knowing exactly what happens inside the cell lets you diagnose problems. Is the forget gate saturated? Is the gradient flowing through the cell state? You can only answer these questions if you know the internals.

Third, foundations for research. Novel architectures are built by modifying existing ones. If you want to design a new gating mechanism or a new memory structure, you need to be comfortable building cells from scratch.

Build it once by hand. Then use `nn.LSTM` for everything else.
