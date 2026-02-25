# Module 06: Sequence Models — Comprehensive Notes

## Table of Contents

1. [Recurrent Neural Networks](#1-recurrent-neural-networks)
2. [Backpropagation Through Time](#2-backpropagation-through-time)
3. [Vanishing and Exploding Gradients](#3-vanishing-and-exploding-gradients)
4. [Long Short-Term Memory (LSTM)](#4-long-short-term-memory-lstm)
5. [LSTM Numerical Walkthrough](#5-lstm-numerical-walkthrough)
6. [Gated Recurrent Unit (GRU)](#6-gated-recurrent-unit-gru)
7. [Sequence-to-Sequence Models](#7-sequence-to-sequence-models)
8. [The Attention Mechanism](#8-the-attention-mechanism)
9. [Practical Considerations](#9-practical-considerations)

---

## 1. Recurrent Neural Networks

### Intuition

Think of the hidden state as a note passed between students sitting in a row. The first student reads the first word of a sentence, writes a summary on a note, and passes it to the next student. The second student reads both the second word *and* the note, writes a new summary, and passes it on. By the last student, the note should contain a summary of the entire sentence — but it is a fixed-size piece of paper, so information is inevitably lost along the way.

That note is the hidden state. The act of reading a word and updating the note is the recurrent computation. The fixed size of the paper is the fundamental limitation.

### The Math

The vanilla RNN is defined by:

```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
y_t = W_hy * h_t + b_y
```

Where:
- `x_t` is the input at time t (shape: [input_size])
- `h_t` is the hidden state at time t (shape: [hidden_size])
- `h_{t-1}` is the hidden state from the previous time step
- `y_t` is the output at time t (shape: [output_size])
- `W_xh` is the input-to-hidden weight matrix (shape: [hidden_size, input_size])
- `W_hh` is the hidden-to-hidden weight matrix (shape: [hidden_size, hidden_size])
- `W_hy` is the hidden-to-output weight matrix (shape: [output_size, hidden_size])
- `b_h`, `b_y` are biases
- `tanh` is the activation function (squashes values to [-1, 1])

The initial hidden state h_0 is typically a zero vector.

### Code: RNN Cell from Scratch

```python
import torch
import torch.nn as nn

class ManualRNNCell:
    """RNN cell implemented with raw tensor operations."""

    def __init__(self, input_size, hidden_size):
        # Xavier initialization
        scale_xh = (2.0 / (input_size + hidden_size)) ** 0.5
        scale_hh = (2.0 / (hidden_size + hidden_size)) ** 0.5

        self.W_xh = torch.randn(hidden_size, input_size) * scale_xh
        self.W_hh = torch.randn(hidden_size, hidden_size) * scale_hh
        self.b_h = torch.zeros(hidden_size)

        # Enable gradient tracking
        self.W_xh.requires_grad_(True)
        self.W_hh.requires_grad_(True)
        self.b_h.requires_grad_(True)

    def forward(self, x_t, h_prev):
        """
        x_t: (batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        Returns: h_t (batch_size, hidden_size)
        """
        h_t = torch.tanh(x_t @ self.W_xh.T + h_prev @ self.W_hh.T + self.b_h)
        return h_t

    def parameters(self):
        return [self.W_xh, self.W_hh, self.b_h]
```

### Code: Using PyTorch's nn.RNN

```python
# Single-layer RNN
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# Input: (batch_size, seq_len, input_size)
x = torch.randn(32, 50, 10)
h0 = torch.zeros(1, 32, 20)  # (num_layers, batch_size, hidden_size)

output, h_n = rnn(x, h0)
# output: (32, 50, 20) — hidden state at every time step
# h_n: (1, 32, 20) — final hidden state
```

### Historical Context

The idea of recurrent connections dates back to the 1980s (Jordan networks, Elman networks). The formulation above is the "Elman network" (Elman, 1990). It was simple, elegant, and worked — for short sequences. The limitations became clear as researchers tried to apply it to longer sequences, leading to the discoveries around vanishing gradients and eventually to the LSTM.

### Bidirectional RNNs

For tasks where the full sequence is available (classification, tagging — but NOT autoregressive generation), we can run two RNNs:

```
Forward:  h_t^f = RNN_f(x_t, h_{t-1}^f)    (left to right)
Backward: h_t^b = RNN_b(x_t, h_{t+1}^b)    (right to left)
Combined: h_t = [h_t^f ; h_t^b]             (concatenation)
```

The combined hidden state captures both past and future context.

```python
bi_rnn = nn.RNN(input_size=10, hidden_size=20, bidirectional=True, batch_first=True)
# Output hidden size is 2 * hidden_size = 40
```

### Stacked (Multi-Layer) RNNs

Stack L layers of RNNs. The hidden states of layer l become inputs to layer l+1:

```
h_t^1 = RNN_1(x_t, h_{t-1}^1)
h_t^2 = RNN_2(h_t^1, h_{t-1}^2)
...
h_t^L = RNN_L(h_t^{L-1}, h_{t-1}^L)
```

```python
stacked_rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=3, batch_first=True)
# h0 shape: (3, batch_size, 20) — one initial state per layer
```

In practice, 2-4 layers work well. Beyond that, training becomes difficult without skip connections.

---

## 2. Backpropagation Through Time

### Intuition

When you unroll an RNN across T time steps, you get a very deep feedforward network with shared weights. BPTT is just standard backpropagation applied to this unrolled graph. But depth T means the gradient must travel through T layers, and that is where problems begin.

### The Math: Full BPTT Derivation

Consider a simple RNN with loss computed at every time step:

```
h_t = tanh(W * h_{t-1} + U * x_t + b)
y_t = V * h_t
L = sum_{t=1}^{T} L_t(y_t, target_t)
```

We want dL/dW. By the chain rule:

```
dL/dW = sum_{t=1}^{T} dL_t/dW
```

For a single time step t:

```
dL_t/dW = dL_t/dy_t * dy_t/dh_t * dh_t/dW
```

But h_t depends on h_{t-1}, which depends on h_{t-2}, and so on. So:

```
dh_t/dW = partial(h_t)/partial(W) + partial(h_t)/partial(h_{t-1}) * dh_{t-1}/dW
```

Expanding recursively:

```
dL_t/dW = sum_{k=1}^{t} dL_t/dh_t * (prod_{j=k+1}^{t} dh_j/dh_{j-1}) * dh_k/dW
```

The critical term is the product of Jacobians:

```
prod_{j=k+1}^{t} dh_j/dh_{j-1}
```

Each Jacobian dh_j/dh_{j-1} = diag(1 - h_j^2) * W (for tanh activation, since tanh'(z) = 1 - tanh^2(z)).

When t - k is large (long-range dependencies), this product either vanishes or explodes.

### Truncated BPTT

Rather than backpropagating through all T steps, truncate after K steps:

```
dL_t/dW approx sum_{k=max(1,t-K)}^{t} dL_t/dh_t * (prod_{j=k+1}^{t} dh_j/dh_{j-1}) * dh_k/dW
```

In practice:
1. Process K steps forward.
2. Compute loss and backpropagate through only those K steps.
3. Detach the hidden state (stop gradient) and continue to the next K steps.

```python
# Truncated BPTT with chunk_size K
hidden = torch.zeros(batch_size, hidden_size)
for i in range(0, seq_len, K):
    chunk = x[:, i:i+K, :]
    hidden = hidden.detach()  # Stop gradient flow to previous chunks
    output, hidden = rnn_forward(chunk, hidden)
    loss = compute_loss(output, targets[:, i:i+K])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Typical values for K: 20-100 steps. This trades gradient accuracy for training stability.

### Historical Context

BPTT was described by Werbos (1990) and later formalized in the deep learning context by several researchers. The computational cost (O(T) memory for the full backward pass) motivated truncated BPTT and was one reason researchers sought architectures that could bypass the long gradient paths entirely.

---

## 3. Vanishing and Exploding Gradients

### Intuition

Imagine whispering a message through a chain of 100 people. Each person hears the message, transforms it slightly, and whispers it to the next. If each person tends to make the message slightly quieter (eigenvalues < 1), by person 100 the message is inaudible. If each person tends to make it slightly louder (eigenvalues > 1), by person 100 it is a deafening roar. Only if each person passes the message at exactly the same volume does it arrive intact — and that is nearly impossible to arrange.

The "message" is the gradient signal. The "people" are the time steps.

### The Math

Consider the gradient of h_T with respect to h_k:

```
dh_T/dh_k = prod_{t=k+1}^{T} dh_t/dh_{t-1}
           = prod_{t=k+1}^{T} diag(f'(z_t)) * W_hh
```

where f'(z_t) is the derivative of the activation function at time t.

For tanh, f'(z) = 1 - tanh^2(z), so each element is in (0, 1].

Let us consider the norm of this product. For simplicity, assume f' is approximately constant (say, around some average value). Then:

```
||dh_T/dh_k|| approx ||diag(f') * W_hh||^{T-k}
```

Let sigma_max be the largest singular value of (diag(f') * W_hh). Then:

- If sigma_max < 1: ||dh_T/dh_k|| shrinks exponentially as (T-k) grows. **Gradients vanish.**
- If sigma_max > 1: ||dh_T/dh_k|| grows exponentially. **Gradients explode.**

More precisely, consider the eigenvalue decomposition of W_hh = Q * Lambda * Q^{-1}. The product of T copies of W_hh is Q * Lambda^T * Q^{-1}. If the largest eigenvalue magnitude |lambda_max| < 1, then Lambda^T -> 0 as T -> infinity. If |lambda_max| > 1, Lambda^T -> infinity.

**Key insight**: There is no stable middle ground for vanilla RNNs. The eigenvalue must be *exactly* 1 for gradients to neither vanish nor explode, and this is a measure-zero set in parameter space. In practice, vanilla RNNs cannot learn dependencies beyond approximately 10-20 time steps.

### Gradient Clipping

Gradient clipping addresses **exploding** gradients (not vanishing ones):

```python
# Clip by global norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

Norm clipping rescales the entire gradient vector if its norm exceeds the threshold, preserving the direction. Value clipping clips each element independently, which can change the gradient direction.

Norm clipping is generally preferred. Typical threshold values: 1.0 to 10.0.

### Empirical Observation

You can verify this experimentally:

```python
# Track gradient norms at different time steps
def track_gradients(model, sequence, target):
    """Record gradient norms at each time step."""
    hidden = torch.zeros(1, hidden_size, requires_grad=True)
    hiddens = [hidden]

    for t in range(len(sequence)):
        hidden = torch.tanh(sequence[t] @ model.W_xh.T + hidden @ model.W_hh.T + model.b_h)
        hiddens.append(hidden)

    loss = criterion(hiddens[-1], target)
    loss.backward()

    # Gradient of loss w.r.t. hidden state at each time step
    grad_norms = [h.grad.norm().item() for h in hiddens if h.grad is not None]
    return grad_norms
```

For a vanilla RNN, you will see the gradient norms decay exponentially as you go further back in time.

---

## 4. Long Short-Term Memory (LSTM)

### Intuition

Think of the LSTM cell as a conveyor belt running through time. Items (information) ride on the belt. At each time step, three workers stand beside the belt:

1. **The Forgetter** (forget gate): removes items that are no longer needed. "We have left the topic of cats; clear the cat-related information."
2. **The Adder** (input gate + candidate): places new items on the belt. "We are now talking about dogs; add dog-related information."
3. **The Reader** (output gate): looks at the belt and selects what is relevant for the current output. "The question is about color, so read the color information from the belt."

The belt itself is the cell state. It flows through time with minimal interference — just addition and element-wise multiplication. This is why gradients can flow along it without vanishing.

### The Math

The LSTM cell computes the following at each time step:

```
Forget gate:    f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
Input gate:     i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
Candidate:      g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)
Cell update:    C_t = f_t * C_{t-1} + i_t * g_t
Output gate:    o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
Hidden state:   h_t = o_t * tanh(C_t)
```

Where:
- `[h_{t-1}, x_t]` denotes concatenation of the previous hidden state and current input.
- `*` denotes element-wise (Hadamard) multiplication.
- All gates use sigmoid activation (output in [0, 1], acting as soft switches).
- The candidate uses tanh (output in [-1, 1], creating new content).

Dimensions (assuming input_size=d, hidden_size=n):
- Each W matrix: [n, n+d] (because input is concatenation of h and x)
- Each bias: [n]
- Cell state C_t: [n]
- Hidden state h_t: [n]

### Why Each Gate Exists

**Forget gate (f_t)**: Controls what to erase from the cell state. When processing "The cat sat. The dog ran.", the forget gate should clear cat-related information when we encounter "The dog." A forget gate value of 0 means "erase this cell state element completely." A value of 1 means "keep it perfectly."

**Input gate (i_t)**: Controls what new information to write to the cell state. It works in conjunction with the candidate g_t. Even if the candidate generates useful new content, the input gate can suppress it if it is not the right time to write. Think of it as the gatekeeper deciding which new memories are worth storing.

**Candidate (g_t)**: Generates new content to potentially add to the cell state. It uses tanh to bound the values in [-1, 1]. The input gate then decides how much of this candidate actually gets written.

**Output gate (o_t)**: Controls what part of the cell state is exposed as the hidden state. The cell state might contain information about many things, but only some are relevant for the current output. The output gate selects what to expose. The cell state goes through tanh first (bounding it) and then the output gate filters it.

### The Gradient Highway

The cell state update equation is:

```
C_t = f_t * C_{t-1} + i_t * g_t
```

The gradient of C_T with respect to C_k is:

```
dC_T/dC_k = prod_{t=k+1}^{T} f_t
```

(Ignoring second-order terms through the gates, which are smaller.)

Each f_t has elements in (0, 1). If the forget gate learns to be close to 1 for important information, the gradient flows through time almost unimpeded:

```
dC_T/dC_k approx 1^{T-k} = 1
```

Compare this to the vanilla RNN where the gradient involves products of W_hh and activation derivatives — a chain of matrix multiplications that amplifies or attenuates exponentially.

This is why the LSTM solves vanishing gradients: the cell state provides a *linear* pathway (add and element-wise multiply, no matrix multiplication) through which gradients can flow.

### Code: LSTM Cell from Scratch

```python
class ManualLSTMCell:
    """LSTM cell implemented with raw tensor operations."""

    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        # Initialize all four gate weight matrices
        scale = (2.0 / (combined_size + hidden_size)) ** 0.5

        self.W_f = torch.randn(hidden_size, combined_size) * scale
        self.W_i = torch.randn(hidden_size, combined_size) * scale
        self.W_g = torch.randn(hidden_size, combined_size) * scale
        self.W_o = torch.randn(hidden_size, combined_size) * scale

        # Biases — NOTE: forget gate bias initialized to 1.0
        self.b_f = torch.ones(hidden_size)   # Important! Start by remembering.
        self.b_i = torch.zeros(hidden_size)
        self.b_g = torch.zeros(hidden_size)
        self.b_o = torch.zeros(hidden_size)

        # Enable gradients
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, x_t, h_prev, c_prev):
        """
        x_t:    (batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        c_prev: (batch_size, hidden_size)
        Returns: (h_t, c_t)
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([h_prev, x_t], dim=1)  # (batch, hidden+input)

        # Compute all four gates
        f_t = torch.sigmoid(combined @ self.W_f.T + self.b_f)  # Forget gate
        i_t = torch.sigmoid(combined @ self.W_i.T + self.b_i)  # Input gate
        g_t = torch.tanh(combined @ self.W_g.T + self.b_g)     # Candidate
        o_t = torch.sigmoid(combined @ self.W_o.T + self.b_o)  # Output gate

        # Cell state update
        c_t = f_t * c_prev + i_t * g_t

        # Hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def parameters(self):
        return [self.W_f, self.W_i, self.W_g, self.W_o,
                self.b_f, self.b_i, self.b_g, self.b_o]
```

### Code: Using PyTorch's nn.LSTM

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2,
               batch_first=True, dropout=0.1, bidirectional=False)

x = torch.randn(32, 50, 10)  # (batch, seq_len, input_size)
h0 = torch.zeros(2, 32, 20)  # (num_layers, batch, hidden_size)
c0 = torch.zeros(2, 32, 20)  # (num_layers, batch, hidden_size)

output, (h_n, c_n) = lstm(x, (h0, c0))
# output: (32, 50, 20) — hidden state at every time step (top layer)
# h_n: (2, 32, 20) — final hidden state for each layer
# c_n: (2, 32, 20) — final cell state for each layer
```

Note: PyTorch's LSTM internally concatenates all four gate matrices into one large matrix for efficiency:

```python
# Internally, PyTorch computes:
# gates = x @ W_ih.T + h @ W_hh.T + b_ih + b_hh
# Then splits into f, i, g, o
# The weight matrices are stored as:
# lstm.weight_ih_l0: shape (4*hidden_size, input_size)
# lstm.weight_hh_l0: shape (4*hidden_size, hidden_size)
```

### Historical Context

Hochreiter and Schmidhuber published the LSTM in 1997 ("Long Short-Term Memory," Neural Computation). The name is a play on "short-term memory" — the LSTM provides a mechanism for memory that is "long" in the short-term sense, meaning it can persist across many time steps within a sequence.

The LSTM was not widely adopted until the 2010s, when increased compute made it practical to train on large datasets. Between roughly 2013-2017, LSTMs dominated NLP: machine translation, language modeling, speech recognition, and text generation all used LSTM-based architectures. The era ended with the introduction of the Transformer in 2017.

---

## 5. LSTM Numerical Walkthrough

This section walks through ONE time step of an LSTM cell with concrete numbers. If you understand this, you understand the LSTM.

### Setup

Let hidden_size = 2 and input_size = 2 for simplicity.

Previous states:
```
h_{t-1} = [0.5, -0.3]     (previous hidden state)
C_{t-1} = [0.8, -0.5]     (previous cell state)
x_t     = [1.0, 0.2]      (current input)
```

Concatenated input:
```
[h_{t-1}, x_t] = [0.5, -0.3, 1.0, 0.2]
```

Weights (simplified small values for illustration):
```
W_f = [[0.1, 0.2, 0.3, 0.1],    b_f = [1.0, 1.0]  (initialized to 1!)
       [0.2, 0.1, 0.2, 0.3]]

W_i = [[0.3, 0.1, 0.2, 0.1],    b_i = [0.0, 0.0]
       [0.1, 0.3, 0.1, 0.2]]

W_g = [[0.2, 0.3, 0.1, 0.2],    b_g = [0.0, 0.0]
       [0.1, 0.2, 0.3, 0.1]]

W_o = [[0.1, 0.1, 0.3, 0.2],    b_o = [0.0, 0.0]
       [0.3, 0.2, 0.1, 0.1]]
```

### Step 1: Forget Gate — "What do I throw away?"

```
z_f = W_f * [h_{t-1}, x_t] + b_f

z_f[0] = 0.1*0.5 + 0.2*(-0.3) + 0.3*1.0 + 0.1*0.2 + 1.0
       = 0.05 - 0.06 + 0.3 + 0.02 + 1.0
       = 1.31

z_f[1] = 0.2*0.5 + 0.1*(-0.3) + 0.2*1.0 + 0.3*0.2 + 1.0
       = 0.10 - 0.03 + 0.2 + 0.06 + 1.0
       = 1.33

f_t = sigmoid(z_f) = [sigmoid(1.31), sigmoid(1.33)]
    = [0.787, 0.791]
```

**Interpretation**: The forget gate is close to 0.8 for both elements. This means "keep about 80% of the existing cell state." The bias initialization of 1.0 pushed the raw values above 1.0, ensuring the sigmoid outputs are above 0.5 — the network starts by retaining memories.

### Step 2: Input Gate — "What new information is worth storing?"

```
z_i = W_i * [h_{t-1}, x_t] + b_i

z_i[0] = 0.3*0.5 + 0.1*(-0.3) + 0.2*1.0 + 0.1*0.2 + 0.0
       = 0.15 - 0.03 + 0.2 + 0.02
       = 0.34

z_i[1] = 0.1*0.5 + 0.3*(-0.3) + 0.1*1.0 + 0.2*0.2 + 0.0
       = 0.05 - 0.09 + 0.1 + 0.04
       = 0.10

i_t = sigmoid(z_i) = [sigmoid(0.34), sigmoid(0.10)]
    = [0.584, 0.525]
```

**Interpretation**: The input gate is around 0.5-0.6, moderately open. It will allow some — but not all — of the candidate information through.

### Step 3: Candidate Cell State — "What could I add?"

```
z_g = W_g * [h_{t-1}, x_t] + b_g

z_g[0] = 0.2*0.5 + 0.3*(-0.3) + 0.1*1.0 + 0.2*0.2 + 0.0
       = 0.10 - 0.09 + 0.1 + 0.04
       = 0.15

z_g[1] = 0.1*0.5 + 0.2*(-0.3) + 0.3*1.0 + 0.1*0.2 + 0.0
       = 0.05 - 0.06 + 0.3 + 0.02
       = 0.31

g_t = tanh(z_g) = [tanh(0.15), tanh(0.31)]
    = [0.149, 0.300]
```

**Interpretation**: The candidate proposes new values of approximately [0.15, 0.30] for the cell state.

### Step 4: Cell State Update — "Update memory"

```
C_t = f_t * C_{t-1} + i_t * g_t

C_t[0] = 0.787 * 0.8 + 0.584 * 0.149
       = 0.630 + 0.087
       = 0.717

C_t[1] = 0.791 * (-0.5) + 0.525 * 0.300
       = -0.396 + 0.158
       = -0.238
```

**Interpretation**: The first cell state element went from 0.8 to 0.717 — it was mostly retained (forget gate was 0.787) and a small amount (0.087) was added. The second element went from -0.5 to -0.238 — it was partially retained and partially overwritten. The cell state has been *selectively* updated.

### Step 5: Output Gate — "What should I output?"

```
z_o = W_o * [h_{t-1}, x_t] + b_o

z_o[0] = 0.1*0.5 + 0.1*(-0.3) + 0.3*1.0 + 0.2*0.2 + 0.0
       = 0.05 - 0.03 + 0.3 + 0.04
       = 0.36

z_o[1] = 0.3*0.5 + 0.2*(-0.3) + 0.1*1.0 + 0.1*0.2 + 0.0
       = 0.15 - 0.06 + 0.1 + 0.02
       = 0.21

o_t = sigmoid(z_o) = [sigmoid(0.36), sigmoid(0.21)]
    = [0.589, 0.552]
```

### Step 6: Hidden State — "The final output"

```
h_t = o_t * tanh(C_t)
    = [0.589 * tanh(0.717), 0.552 * tanh(-0.238)]
    = [0.589 * 0.615, 0.552 * (-0.233)]
    = [0.362, -0.129]
```

**Interpretation**: The hidden state is the filtered view of the cell state. The output gate decided to expose about 55-59% of the tanh-squashed cell state. The hidden state [0.362, -0.129] is what downstream layers or the next time step will see.

### Summary Table

| Component | Value | Interpretation |
|-----------|-------|----------------|
| h_{t-1} | [0.500, -0.300] | Previous hidden state |
| C_{t-1} | [0.800, -0.500] | Previous cell state |
| x_t | [1.000, 0.200] | Current input |
| f_t | [0.787, 0.791] | Keep ~79% of old cell state |
| i_t | [0.584, 0.525] | Write ~53-58% of candidate |
| g_t | [0.149, 0.300] | Candidate values to add |
| C_t | [0.717, -0.238] | Updated cell state |
| o_t | [0.589, 0.552] | Expose ~55-59% of cell state |
| h_t | [0.362, -0.129] | New hidden state |

---

## 6. Gated Recurrent Unit (GRU)

### Intuition

The GRU is the LSTM's leaner sibling. It asks: do we really need separate cell state and hidden state? Do we need three gates? The GRU merges cell and hidden state into one and uses two gates instead of three. It often works just as well with fewer parameters.

### The Math

```
Reset gate:  r_t = sigmoid(W_r * [h_{t-1}, x_t] + b_r)
Update gate: z_t = sigmoid(W_z * [h_{t-1}, x_t] + b_z)
Candidate:   h_tilde_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)
Hidden:      h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
```

**Reset gate (r_t)**: Controls how much of the previous hidden state to use when computing the candidate. When r_t is close to 0, the candidate is computed mostly from the current input, effectively "resetting" the memory. This is useful when the current input represents a sharp context change.

**Update gate (z_t)**: Controls the interpolation between the old hidden state and the new candidate. This single gate plays the roles of both the forget gate and the input gate in LSTM. When z_t is close to 0, the hidden state stays the same (old information is retained). When z_t is close to 1, the hidden state is replaced with the candidate (new information dominates).

Note the elegant constraint: the coefficients (1 - z_t) and z_t sum to 1, so the update is a proper interpolation. No information is created or destroyed, only mixed.

### Code: GRU Cell from Scratch

```python
class ManualGRUCell:
    """GRU cell implemented with raw tensor operations."""

    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        scale = (2.0 / (combined_size + hidden_size)) ** 0.5

        self.W_r = torch.randn(hidden_size, combined_size) * scale
        self.W_z = torch.randn(hidden_size, combined_size) * scale
        self.W_h = torch.randn(hidden_size, combined_size) * scale

        self.b_r = torch.zeros(hidden_size)
        self.b_z = torch.zeros(hidden_size)
        self.b_h = torch.zeros(hidden_size)

        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, x_t, h_prev):
        """
        x_t:    (batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        Returns: h_t (batch_size, hidden_size)
        """
        combined = torch.cat([h_prev, x_t], dim=1)

        r_t = torch.sigmoid(combined @ self.W_r.T + self.b_r)  # Reset gate
        z_t = torch.sigmoid(combined @ self.W_z.T + self.b_z)  # Update gate

        # Candidate uses reset-gated hidden state
        combined_reset = torch.cat([r_t * h_prev, x_t], dim=1)
        h_tilde = torch.tanh(combined_reset @ self.W_h.T + self.b_h)

        # Interpolate between old and new
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        return h_t

    def parameters(self):
        return [self.W_r, self.W_z, self.W_h, self.b_r, self.b_z, self.b_h]
```

### LSTM vs GRU: Practical Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | 4 gate matrices | 3 gate matrices (~25% fewer) |
| States | Cell state + hidden state | Hidden state only |
| Gates | Forget, Input, Output | Reset, Update |
| Training speed | Slower | Faster |
| Performance | Slightly better on some long-range tasks | Comparable, sometimes better on small data |
| Memory | Higher | Lower |

**When to choose LSTM**: Tasks requiring fine-grained memory control, large datasets, very long sequences where precise forgetting matters.

**When to choose GRU**: Smaller datasets, faster experimentation, resource-constrained settings, comparable performance with fewer parameters.

**Honest advice**: Try both. The difference is usually small. The architecture choice matters less than hyperparameter tuning, data quality, and training strategy.

### Peephole Connections

A variant of LSTM where the gates can see the cell state directly:

```
f_t = sigmoid(W_f * [h_{t-1}, x_t] + W_pf * C_{t-1} + b_f)
i_t = sigmoid(W_i * [h_{t-1}, x_t] + W_pi * C_{t-1} + b_i)
o_t = sigmoid(W_o * [h_{t-1}, x_t] + W_po * C_t + b_o)
```

The W_p matrices are diagonal (element-wise connections from cell state to gates). This gives gates direct access to the memory contents. In practice, peephole connections provide marginal improvement and are rarely used.

### Historical Context

The GRU was introduced by Cho et al. (2014) in "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." It was proposed alongside the encoder-decoder architecture for machine translation, which we discuss next.

---

## 7. Sequence-to-Sequence Models

### Intuition

Think of a human interpreter translating a speech from French to English. The interpreter first *listens* to the entire French sentence (encoding), builds an understanding of it in their mind (context representation), and then *speaks* the English translation (decoding). The interpreter does not translate word-by-word — they understand the whole meaning first, then produce the translation in a different order with different words.

The encoder-decoder architecture mimics this process.

### The Architecture

**Encoder**: An LSTM (or GRU) reads the source sequence x_1, x_2, ..., x_S one token at a time. The final hidden state h_S (and cell state C_S for LSTM) is the "context vector" — a fixed-size summary of the entire source.

**Decoder**: Another LSTM, initialized with the encoder's final states, generates the target sequence y_1, y_2, ..., y_T one token at a time. At each step, it takes the previous output (or ground truth during training) as input and produces a probability distribution over the vocabulary.

```
Encoder:
  for s = 1 to S:
    h_s^enc, C_s^enc = LSTM_enc(x_s, h_{s-1}^enc, C_{s-1}^enc)
  context = h_S^enc  (and C_S^enc)

Decoder (initialized with context):
  h_0^dec = h_S^enc
  C_0^dec = C_S^enc
  for t = 1 to T:
    h_t^dec, C_t^dec = LSTM_dec(y_{t-1}, h_{t-1}^dec, C_{t-1}^dec)
    p(y_t) = softmax(W_vocab * h_t^dec + b_vocab)
```

### The Information Bottleneck

The entire source sentence — every word, every nuance of meaning, every grammatical relationship — must be compressed into the fixed-size context vector. For a 512-dimensional hidden state, that is 512 floating-point numbers encoding everything about a potentially 100-word sentence.

This is an information bottleneck. It works tolerably for short sentences (under 20 words) but degrades sharply for longer inputs. Sutskever et al. (2014) observed that reversing the source sentence helped — it reduced the average distance between corresponding words — but this was a band-aid.

The fundamental problem remains: a fixed-size vector cannot faithfully represent arbitrarily complex input.

### Teacher Forcing

During training, the decoder must decide what input to use at each step:

**Option 1 — Free-running (autoregressive)**: Use the decoder's own previous prediction as input. Problem: if the decoder makes a mistake at step t, all subsequent predictions are conditioned on that mistake. Errors compound rapidly ("exposure bias").

**Option 2 — Teacher forcing**: Use the ground-truth previous token as input, regardless of what the decoder predicted. Problem: during inference, ground truth is not available, so the model has never learned to recover from its own mistakes ("train-test mismatch").

**Practical approach — Scheduled sampling** (Bengio et al., 2015): Start training with teacher forcing (probability 1.0 of using ground truth), then gradually decrease this probability so the decoder learns to use its own predictions. A typical schedule linearly decays from 1.0 to 0.5 over the first half of training.

```python
def get_decoder_input(prev_prediction, ground_truth, teacher_forcing_ratio):
    """Choose between ground truth and own prediction."""
    if random.random() < teacher_forcing_ratio:
        return ground_truth
    else:
        return prev_prediction
```

### Beam Search Decoding

At inference time, we need to find the most likely output sequence. Greedy decoding (pick the highest-probability token at each step) is fast but suboptimal — locally optimal choices can lead to globally poor sequences.

Beam search maintains the top K (beam width) partial sequences at each step:

```
Beam search with K=3:

Step 1: Generate top 3 first words:
  "The" (0.4), "A" (0.3), "This" (0.2)

Step 2: Expand each, keep top 3 overall:
  "The cat" (0.4*0.5=0.20)
  "A small" (0.3*0.6=0.18)
  "The dog" (0.4*0.4=0.16)

Step 3: Expand each, keep top 3 overall:
  "A small cat" (0.18*0.7=0.126)
  "The cat sat" (0.20*0.6=0.120)
  "The dog ran" (0.16*0.7=0.112)

... continue until <EOS> token
```

Beam search with K=1 is greedy decoding. K=5-10 is typical in practice. Larger K gives better sequences but is K times more expensive.

```python
def beam_search(decoder, encoder_output, beam_width=5, max_len=50):
    """Basic beam search implementation."""
    # Each beam: (log_probability, sequence, hidden_state)
    beams = [(0.0, [BOS_TOKEN], initial_hidden)]

    for step in range(max_len):
        all_candidates = []
        for log_prob, seq, hidden in beams:
            if seq[-1] == EOS_TOKEN:
                all_candidates.append((log_prob, seq, hidden))
                continue

            # Get next token probabilities
            output, new_hidden = decoder(seq[-1], hidden, encoder_output)
            log_probs = torch.log_softmax(output, dim=-1)

            # Expand this beam with top K tokens
            top_k_probs, top_k_ids = log_probs.topk(beam_width)
            for i in range(beam_width):
                new_log_prob = log_prob + top_k_probs[i].item()
                new_seq = seq + [top_k_ids[i].item()]
                all_candidates.append((new_log_prob, new_seq, new_hidden))

        # Keep top K beams
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]

        # Stop if all beams have ended
        if all(beam[1][-1] == EOS_TOKEN for beam in beams):
            break

    return beams[0][1]  # Return best sequence
```

### Code: Simple Seq2Seq

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.embedding(src)           # (batch, src_len, embed_size)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt_token, hidden, cell):
        # tgt_token: (batch, 1)
        embedded = self.embedding(tgt_token)     # (batch, 1, embed_size)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # (batch, vocab_size)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size)
        enc_outputs, hidden, cell = self.encoder(src)

        # First decoder input is <BOS> token
        decoder_input = tgt[:, 0:1]

        for t in range(1, tgt_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = prediction

            if random.random() < teacher_forcing_ratio:
                decoder_input = tgt[:, t:t+1]  # Teacher forcing
            else:
                decoder_input = prediction.argmax(dim=1, keepdim=True)

        return outputs
```

### Historical Context

Sutskever, Vinyals, and Le published "Sequence to Sequence Learning with Neural Networks" in 2014. It was one of the landmark papers in deep learning for NLP. The key tricks: reversing the source sentence, using deep LSTMs (4 layers), and ensembling multiple models. They achieved state-of-the-art results on English-French translation.

But the information bottleneck was immediately apparent, and within a year, Bahdanau et al. would publish their attention mechanism — the insight that changed everything.

---

## 8. The Attention Mechanism

### Intuition

Without attention, the decoder is like a student taking an exam with only a one-paragraph summary of the textbook. With attention, the decoder has the full textbook open and can look up any page at any time.

More precisely: at each decoding step, the decoder asks "which parts of the input are relevant to what I am generating right now?" and constructs a custom context vector by focusing on those parts.

When translating "Le chat noir dort sur le tapis" to "The black cat sleeps on the rug," the decoder generating "black" should attend primarily to "noir," even though "noir" appeared at position 3 while "black" appears at position 2. Attention allows this non-monotonic alignment.

### The Information Flow

```
                    Attention
                   Mechanism
                      |
  Encoder states:  h1  h2  h3  h4  h5     (one per source word)
                    |   |   |   |   |
                    v   v   v   v   v
                  [alignment scores]
                    |   |   |   |   |
                    v   v   v   v   v
                  [  softmax  weights  ]
                    |   |   |   |   |
                    v   v   v   v   v
                  [weighted sum = context c_t]
                              |
                              v
                     Decoder state s_t -----> output y_t
```

### The Math: Bahdanau (Additive) Attention

Bahdanau et al. (2015) proposed the following:

At decoder time step t, with decoder hidden state s_{t-1} and encoder hidden states h_1, ..., h_S:

**Step 1: Compute alignment scores**
```
e_{t,j} = v^T * tanh(W_s * s_{t-1} + W_h * h_j)
```
for each encoder position j = 1, ..., S.

Here:
- W_s: (alignment_dim, decoder_hidden_size) — projects decoder state
- W_h: (alignment_dim, encoder_hidden_size) — projects encoder state
- v: (alignment_dim,) — learnable vector that produces a scalar score
- alignment_dim is a hyperparameter (often equal to hidden_size)

The score e_{t,j} measures how relevant encoder position j is to decoder step t.

**Step 2: Normalize with softmax**
```
alpha_{t,j} = exp(e_{t,j}) / sum_{k=1}^{S} exp(e_{t,k})
```

The attention weights alpha_{t,j} form a probability distribution over source positions. They sum to 1.

**Step 3: Compute context vector**
```
c_t = sum_{j=1}^{S} alpha_{t,j} * h_j
```

The context vector is a weighted average of all encoder hidden states, where the weights reflect relevance to the current decoding step.

**Step 4: Use context for prediction**
```
s_t = LSTM(y_{t-1}, s_{t-1}, c_t)
p(y_t) = softmax(W_out * [s_t; c_t] + b_out)
```

The context vector is concatenated with the decoder state and used for both the LSTM update and the output prediction.

### ASCII Diagram: Full Attention Flow

```
ENCODER                                      DECODER
                                              Step t
Source: x1    x2    x3    x4    x5
         |     |     |     |     |
         v     v     v     v     v
       [h1]  [h2]  [h3]  [h4]  [h5]         [s_{t-1}]
         |     |     |     |     |               |
         |     |     |     |     |      +--------+--------+
         |     |     |     |     |      |                  |
         v     v     v     v     v      v                  |
       +-------------------------------------------+       |
       |   Alignment Scores:                       |       |
       |   e_{t,1} = v^T tanh(W_s*s + W_h*h1)     |       |
       |   e_{t,2} = v^T tanh(W_s*s + W_h*h2)     |       |
       |   e_{t,3} = v^T tanh(W_s*s + W_h*h3)     |       |
       |   e_{t,4} = v^T tanh(W_s*s + W_h*h4)     |       |
       |   e_{t,5} = v^T tanh(W_s*s + W_h*h5)     |       |
       +-------------------------------------------+       |
         |     |     |     |     |                         |
         v     v     v     v     v                         |
       +-------------------------------------------+       |
       |   Softmax:                                |       |
       |   a1=0.05  a2=0.10  a3=0.70  a4=0.10     |       |
       |   a5=0.05                                 |       |
       +-------------------------------------------+       |
         |     |     |     |     |                         |
         v     v     v     v     v                         |
       +-------------------------------------------+       |
       |   Context vector:                         |       |
       |   c_t = 0.05*h1 + 0.10*h2 + 0.70*h3      |       |
       |       + 0.10*h4 + 0.05*h5                 |       |
       +-------------------------------------------+       |
                        |                                  |
                        v                                  v
                  +------------------------------------------+
                  |  Decoder LSTM:                            |
                  |  s_t = LSTM(y_{t-1}, s_{t-1}, c_t)       |
                  +------------------------------------------+
                                    |
                                    v
                            +---------------+
                            | Output: y_t   |
                            | p(y_t|...)    |
                            +---------------+
```

In this example, the decoder is heavily attending to position 3 (weight 0.70). If x3 is "noir" and the decoder is generating "black," this makes intuitive sense.

### Luong (Multiplicative) Attention

Luong et al. (2015) proposed simpler score functions:

**Dot product**:
```
e_{t,j} = s_t^T * h_j
```
Requires encoder and decoder hidden sizes to be equal. Very fast — just a dot product.

**General (bilinear)**:
```
e_{t,j} = s_t^T * W_a * h_j
```
W_a: (decoder_hidden_size, encoder_hidden_size) is a learnable matrix. Works with different hidden sizes.

**Concat** (same as Bahdanau):
```
e_{t,j} = v^T * tanh(W * [s_t; h_j])
```

Luong attention uses the current decoder state s_t (not s_{t-1} as in Bahdanau). The context is computed *after* the decoder LSTM step, not before. This is a subtle but notable difference.

| Score Function | Formula | Parameters | Speed |
|---|---|---|---|
| Dot | s^T h | 0 | Fastest |
| General | s^T W h | d_s * d_h | Medium |
| Additive (Bahdanau) | v^T tanh(W_s s + W_h h) | d_a*(d_s+d_h) + d_a | Slowest |

In practice, dot product attention scaled by sqrt(d) (as in Transformers) works very well and is the most common choice.

### Attention as Soft Lookup (Query-Key-Value)

Reframe attention using the terminology that will become central to Transformers:

- **Query (Q)**: What the decoder is looking for. Derived from s_t.
- **Key (K)**: What each encoder position offers as a "label." Derived from h_j.
- **Value (V)**: The actual content at each encoder position. Also derived from h_j (or a separate projection).

The attention computation is:
1. Compute similarity between query and each key: score(Q, K_j)
2. Normalize similarities: alpha_j = softmax(scores)
3. Weighted sum of values: output = sum(alpha_j * V_j)

This is a *differentiable dictionary lookup*. In a regular dictionary, you have an exact key match and retrieve one value. In attention, you have a soft match against all keys and retrieve a weighted combination of all values.

```
Regular dictionary:  query -> exact match -> one value
Attention:           query -> soft match  -> weighted sum of all values
```

The leap to the Transformer is: what if Q, K, and V all come from the *same* sequence? That is self-attention. And what if you remove the recurrence entirely and rely solely on attention to capture relationships between positions? That is the Transformer architecture.

### Code: Bahdanau Attention

```python
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, attention_size):
        super().__init__()
        self.W_h = nn.Linear(encoder_hidden_size, attention_size, bias=False)
        self.W_s = nn.Linear(decoder_hidden_size, attention_size, bias=False)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        """
        decoder_state:   (batch, decoder_hidden_size)
        encoder_outputs: (batch, src_len, encoder_hidden_size)
        Returns:
            context:   (batch, encoder_hidden_size)
            weights:   (batch, src_len)
        """
        # Project encoder and decoder states
        # (batch, src_len, attn_size)
        encoder_proj = self.W_h(encoder_outputs)
        # (batch, 1, attn_size)
        decoder_proj = self.W_s(decoder_state).unsqueeze(1)

        # Alignment scores: (batch, src_len, 1) -> (batch, src_len)
        scores = self.v(torch.tanh(encoder_proj + decoder_proj)).squeeze(-1)

        # Attention weights: (batch, src_len)
        weights = torch.softmax(scores, dim=1)

        # Context vector: (batch, encoder_hidden_size)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights
```

### Code: Luong (Dot Product) Attention

```python
class LuongDotAttention(nn.Module):
    def forward(self, decoder_state, encoder_outputs):
        """
        decoder_state:   (batch, hidden_size)
        encoder_outputs: (batch, src_len, hidden_size)
        Requires encoder and decoder hidden sizes to be equal.
        """
        # Alignment scores: (batch, src_len)
        scores = torch.bmm(
            encoder_outputs,
            decoder_state.unsqueeze(2)
        ).squeeze(2)

        # Attention weights
        weights = torch.softmax(scores, dim=1)

        # Context vector
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights
```

### Code: Seq2Seq with Attention

```python
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, attention_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(hidden_size, hidden_size, attention_size)
        # LSTM input is embedding + context
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)  # Concat hidden + context

    def forward(self, tgt_token, hidden, cell, encoder_outputs):
        """
        tgt_token:       (batch, 1)
        hidden:          (1, batch, hidden_size)
        cell:            (1, batch, hidden_size)
        encoder_outputs: (batch, src_len, hidden_size)
        """
        embedded = self.embedding(tgt_token)  # (batch, 1, embed_size)

        # Compute attention using previous hidden state
        context, weights = self.attention(
            hidden.squeeze(0),  # (batch, hidden_size)
            encoder_outputs
        )

        # Concatenate embedding and context as LSTM input
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # Prediction from hidden state and context
        prediction = self.fc(torch.cat([output.squeeze(1), context], dim=1))

        return prediction, hidden, cell, weights
```

### Visualizing Attention Weights

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, source_tokens, target_tokens):
    """
    attention_weights: (tgt_len, src_len) numpy array
    source_tokens: list of source words
    target_tokens: list of target words
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=source_tokens,
                yticklabels=target_tokens, cmap='viridis', ax=ax)
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('Attention Weights')
    plt.tight_layout()
    plt.show()

# Example output (conceptual):
#
#              Le    chat   noir   dort   sur    le    tapis
# The        [0.8   0.05   0.05  0.03   0.03  0.02   0.02]
# black      [0.02  0.05   0.85  0.03   0.02  0.01   0.02]
# cat        [0.03  0.80   0.07  0.03   0.03  0.02   0.02]
# sleeps     [0.02  0.03   0.03  0.82   0.05  0.02   0.03]
# on         [0.02  0.02   0.02  0.05   0.78  0.08   0.03]
# the        [0.02  0.02   0.02  0.03   0.05  0.80   0.06]
# rug        [0.02  0.02   0.02  0.03   0.03  0.05   0.83]
#
# The diagonal-ish pattern shows word alignment, but "black" aligns
# with "noir" (position 3->2), showing non-monotonic alignment.
```

### Why Attention Was Revolutionary

1. **Removed the information bottleneck**: The decoder accesses ALL encoder states, not just the final compressed one. A 100-word sentence has 100 * hidden_size numbers available, not just hidden_size.

2. **Created direct gradient paths**: Gradients flow directly from the decoder loss back to each encoder hidden state through the attention weights. No need to flow through the entire sequential chain. This dramatically helps with long-range learning.

3. **Provided interpretability**: The attention weights show exactly what the model is "looking at." You can visualize them as heatmaps and verify that the model is making sensible alignments. This was a rare gift in deep learning — a window into the model's reasoning.

4. **Enabled non-monotonic alignment**: The decoder can attend to any source position in any order. For translation, this is essential because word order differs between languages.

5. **Revealed the key operation**: Attention showed that the critical computation in sequence modeling is not recurrence but *relevance matching between positions*. This insight led directly to the question: "If attention is so powerful, do we need the RNN at all?"

The answer, as Vaswani et al. demonstrated in 2017, is no. The Transformer uses only attention (specifically, self-attention) and dispenses with recurrence entirely. This enables full parallelization across time steps during training and has led to the era of large language models.

### Historical Context

Bahdanau, Cho, and Bengio published "Neural Machine Translation by Jointly Learning to Align and Translate" in 2015 (submitted 2014). It was motivated by the observation that the basic encoder-decoder (Sutskever et al., 2014; Cho et al., 2014) performed poorly on long sentences.

The paper's title emphasizes "align" — attention was originally conceived as an alignment mechanism for machine translation, not as a general computation primitive. The realization that attention could be generalized beyond alignment, applied within a single sequence (self-attention), and used as the *sole* mechanism for sequence processing — that took another two years and culminated in the Transformer.

Luong, Pham, and Manning published "Effective Approaches to Attention-based Neural Machine Translation" in 2015, providing the simpler multiplicative attention variants and a systematic comparison.

---

## 9. Practical Considerations

### Weight Initialization

- **LSTM forget gate bias**: Initialize to 1.0 or even 2.0. This ensures the network starts by remembering and learns what to forget. Without this, the forget gate starts near 0.5, which means the network immediately loses half the cell state at each step. (Jozefowicz et al., 2015, "An Empirical Exploration of Recurrent Network Architectures")

- **Weight matrices**: Xavier/Glorot initialization or orthogonal initialization. Orthogonal initialization of W_hh is particularly helpful — it ensures the eigenvalues start near 1, which gives gradients the best chance of flowing well at the start of training.

```python
# Orthogonal initialization for recurrent weights
nn.init.orthogonal_(lstm.weight_hh_l0)

# Set forget gate bias to 1.0
# In PyTorch, the bias is stored as [b_ii, b_if, b_ig, b_io] for LSTM
# The forget gate is the second quarter
hidden_size = lstm.hidden_size
lstm.bias_ih_l0.data[hidden_size:2*hidden_size].fill_(1.0)
lstm.bias_hh_l0.data[hidden_size:2*hidden_size].fill_(0.0)  # Or also set to 1.0
```

### Regularization

- **Dropout**: Apply between LSTM layers, not within the recurrent connection. Applying dropout to the recurrent connection destroys the memory mechanism.

```python
# Dropout between layers (built-in)
lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.3)

# Variational dropout (same mask across time steps) — better for RNNs
# Not built into nn.LSTM; use a library or implement manually
```

- **Weight tying**: In language models, tie the input embedding matrix and the output projection matrix. This reduces parameters and often improves performance.

```python
# Weight tying
model.decoder_fc.weight = model.embedding.weight
```

- **L2 regularization / weight decay**: Use sparingly. The gating mechanism provides its own form of regularization.

### Sequence Packing

Variable-length sequences in a batch require padding. PyTorch provides utilities to avoid computing on padding tokens:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sort sequences by length (descending)
lengths = [len(seq) for seq in sequences]
sorted_indices = sorted(range(len(lengths)), key=lambda i: -lengths[i])

# Pad and pack
padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=True)

# Run through LSTM
output_packed, (h_n, c_n) = lstm(packed)

# Unpack
output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
```

Packing ensures the LSTM does not waste computation on padding tokens and avoids contaminating the hidden state with padding.

### Gradient Clipping in Practice

```python
# After loss.backward(), before optimizer.step()
max_norm = 5.0
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Log the gradient norm to monitor training
if total_norm > max_norm:
    print(f"Gradient clipped: {total_norm:.2f} -> {max_norm:.2f}")
```

Monitor the gradient norm over training. If it is frequently being clipped, the learning dynamics may be unstable. Consider reducing the learning rate.

### Learning Rate Schedules

RNN training benefits from:
- Starting with a reasonable learning rate (1e-3 for Adam, 1.0 for SGD)
- Reducing it when validation loss plateaus
- Using gradient clipping throughout

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
# After each epoch:
scheduler.step(val_loss)
```

### When RNNs Are Still the Right Choice

Despite the Transformer revolution, RNNs (particularly LSTMs) remain relevant for:

1. **Streaming / online inference**: When data arrives one step at a time and you need immediate predictions. RNNs naturally process one step at a time with O(1) memory. Transformers need to reprocess the entire context window.

2. **Resource-constrained environments**: RNNs have far fewer parameters for simple tasks. An LSTM with 256 hidden units might have ~250K parameters. A small Transformer might have 10M+.

3. **Very long sequences with simple patterns**: Time series forecasting with thousands of steps but simple temporal patterns.

4. **When training data is very limited**: Fewer parameters means less overfitting risk.

5. **Latency-critical applications**: The sequential nature of RNNs is a disadvantage for training (cannot parallelize), but for inference on one sequence, the computation per step is very small.

---

## Summary: The Arc from RNN to Transformer

| Architecture | Year | Solved | New Problem |
|---|---|---|---|
| Vanilla RNN | 1990 | Sequential processing | Vanishing gradients |
| LSTM | 1997 | Vanishing gradients | Still sequential, fixed-size state |
| GRU | 2014 | Simpler gating | Same limitations as LSTM |
| Seq2Seq | 2014 | Variable-length mapping | Information bottleneck |
| Attention | 2015 | Information bottleneck | Still needs recurrence (slow training) |
| Transformer | 2017 | Removes recurrence entirely | (New challenges: quadratic attention, position encoding) |

Each architecture in this table was a direct response to the limitations of the one above it. Understanding this progression is not just historical curiosity — it is the conceptual foundation for understanding why Transformers work the way they do.

When you study the Transformer next week, you will recognize:
- Multi-head attention is Bahdanau/Luong attention generalized and applied within a single sequence.
- Positional encoding is the replacement for the position information that recurrence provided for free.
- The feedforward layers in each Transformer block are the replacement for the recurrent processing.
- Layer normalization and residual connections are the stabilization techniques that replace the LSTM's gradient highway.

The Transformer did not come from nowhere. It came from a decade of struggling with the limitations you have now studied in detail.
