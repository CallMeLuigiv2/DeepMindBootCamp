# Module 7: Transformers and Attention — Comprehensive Notes

## The Definitive Reference

These notes cover the Transformer architecture in its entirety. This is the document you will return to repeatedly — when implementing, when debugging, when reading papers, when preparing for interviews. Every concept is presented with intuition, full mathematical derivation, PyTorch implementation, and paper lineage.

The Transformer is the single most important architecture in modern AI. Treat these notes accordingly.

---

## Table of Contents

1. [From Seq2Seq to Self-Attention](#1-from-seq2seq-to-self-attention)
2. [Scaled Dot-Product Attention](#2-scaled-dot-product-attention)
3. [Multi-Head Attention](#3-multi-head-attention)
4. [Positional Encoding](#4-positional-encoding)
5. [The Transformer Encoder Block](#5-the-transformer-encoder-block)
6. [The Transformer Decoder Block](#6-the-transformer-decoder-block)
7. [The Full Transformer Architecture](#7-the-full-transformer-architecture)
8. [Causal Masking](#8-causal-masking)
9. [Training Objectives](#9-training-objectives)
10. [The Residual Stream View](#10-the-residual-stream-view)
11. [Transformer Variants](#11-transformer-variants)
12. [Vision Transformers](#12-vision-transformers)
13. [Modern Efficiency Techniques](#13-modern-efficiency-techniques)
14. [Tokenization](#14-tokenization)
15. [Scaling Laws](#15-scaling-laws)

---

## 1. From Seq2Seq to Self-Attention

### Intuition

The story of attention begins with a problem: how do you compress an entire input sequence into a single fixed-length vector? Early seq2seq models (Sutskever et al., 2014) used an encoder RNN to read the input and produce a final hidden state, then a decoder RNN to generate the output conditioned on that state. For long sequences, this is an information bottleneck — the fixed vector cannot faithfully represent all the information in a 100-word sentence.

Bahdanau et al. (2014) proposed a solution: let the decoder look back at all encoder hidden states at each generation step, using a learned compatibility function to determine which states are most relevant. This is attention — a dynamic, content-based addressing mechanism.

But Bahdanau attention is cross-attention: queries from one sequence, keys and values from another. Vaswani et al. (2017) asked: what if we let a sequence attend to itself? Self-attention lets each token gather information from all other tokens in the same sequence, enabling the model to capture long-range dependencies without the sequential bottleneck of RNNs.

### The Math

**Bahdanau Attention:**

$$
\begin{aligned}
e_{ij} &= a(s_{i-1},\, h_j) & \text{(alignment score)} \\
\alpha_{ij} &= \text{softmax}_j(e_{ij}) & \text{(attention weight)} \\
c_i &= \sum_j \alpha_{ij} \cdot h_j & \text{(context vector)}
\end{aligned}
$$

Where $s_{i-1}$ is the decoder hidden state, $h_j$ is the $j$-th encoder hidden state, and $a(\cdot)$ is a learned alignment function (typically a small MLP).

**Self-Attention (simplified):**

$$
\begin{aligned}
e_{ij} &= x_i^\top \cdot x_j & \text{(dot-product similarity)} \\
\alpha_{ij} &= \text{softmax}_j(e_{ij}) & \text{(attention weight)} \\
z_i &= \sum_j \alpha_{ij} \cdot x_j & \text{(output)}
\end{aligned}
$$

The key innovation of Vaswani et al. was to separate the roles of each token into three learned projections: Query, Key, and Value.

### Paper Reference

- Sutskever et al., "Sequence to Sequence Learning with Neural Networks" (2014) — the seq2seq bottleneck
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014) — attention is born
- Vaswani et al., "Attention Is All You Need" (2017) — self-attention replaces recurrence entirely

---

## 2. Scaled Dot-Product Attention

This is the atomic unit of the Transformer. Master this completely.

### Intuition

Think of a database lookup. You have a query ("find me documents about neural networks"), a set of keys (document titles/tags), and corresponding values (document contents). The database compares your query against all keys, finds the best matches, and returns the corresponding values.

Attention does the same thing, but soft: instead of returning a single exact match, it returns a weighted average of ALL values, where the weights are determined by how well each key matches the query. Keys that match well get high weight; keys that do not match get near-zero weight.

Now, every token in the sequence plays all three roles simultaneously:
- **Query (Q)**: "What information am I looking for?"
- **Key (K)**: "What information do I contain? What should other tokens know about me?"
- **Value (V)**: "If another token decides to attend to me, what information do I actually provide?"

Q, K, and V are different linear projections of the same input — the model learns to separate "what to search for" from "what to advertise" from "what to deliver."

### The Math

Given input $X$ of shape (seq_len, $d_\text{model}$):

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

Where $W_Q \in \mathbb{R}^{d_\text{model} \times d_k}$, $W_K \in \mathbb{R}^{d_\text{model} \times d_k}$, $W_V \in \mathbb{R}^{d_\text{model} \times d_v}$.

The attention function:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

Step by step:

1. **Compute similarity scores**: $S = Q K^\top$
   - Shape: (seq_len, $d_k$) @ ($d_k$, seq_len) = (seq_len, seq_len)
   - $S[i][j]$ = dot product of $\text{query}_i$ and $\text{key}_j$ = how much token $i$ should attend to token $j$
   - This single matrix multiplication computes ALL pairwise similarities at once

2. **Scale**: $S_\text{scaled} = S / \sqrt{d_k}$
   - Why? See the variance argument below.

3. **Softmax**: $A = \text{softmax}(S_\text{scaled}, \text{dim}=-1)$
   - Shape: (seq_len, seq_len)
   - Each row sums to 1
   - $A[i][j]$ = the attention weight that token $i$ places on token $j$

4. **Weighted sum of values**: $\text{output} = A V$
   - Shape: (seq_len, seq_len) @ (seq_len, $d_v$) = (seq_len, $d_v$)
   - $\text{output}[i]$ = weighted average of all value vectors, weighted by attention weights from token $i$

### Why We Scale by $\sqrt{d_k}$ — The Variance Argument

This is not a minor detail. It is essential for training stability.

Assume $q$ and $k$ are vectors of dimension $d_k$, where each component is drawn independently from a distribution with mean 0 and variance 1. The dot product is:

$$
q \cdot k = \sum_{i=1}^{d_k} q_i \cdot k_i
$$

Each term $q_i \cdot k_i$ has:
- $\mathbb{E}[q_i \cdot k_i] = \mathbb{E}[q_i] \cdot \mathbb{E}[k_i] = 0 \times 0 = 0$
- $\text{Var}(q_i \cdot k_i) = \mathbb{E}[q_i^2] \cdot \mathbb{E}[k_i^2] = 1 \times 1 = 1$

Since the terms are independent:
- $\mathbb{E}[q \cdot k] = 0$
- $\text{Var}(q \cdot k) = d_k$

So the standard deviation of the dot product is $\sqrt{d_k}$. For $d_k = 64$ (typical), the dot products have standard deviation 8. Values of 8 or -8 push the softmax deep into saturation, where:

```
softmax([8, 0, 0, 0]) ~= [0.9997, 0.0001, 0.0001, 0.0001]
```

The gradients of saturated softmax are near zero — training stalls. By dividing by $\sqrt{d_k}$, we normalize the variance back to 1, keeping the softmax in a well-behaved regime:

```
softmax([1, 0, 0, 0]) ~= [0.46, 0.18, 0.18, 0.18]
```

Gradients flow freely. Training proceeds.

### Complete Worked Example

Let us work through attention with actual numbers. We have 3 tokens, $d_k = d_v = 4$.

**Input representations (after projection):**

```
Q = [[1, 0, 1, 0],    # Token 0's query
     [0, 1, 0, 1],    # Token 1's query
     [1, 1, 0, 0]]    # Token 2's query

K = [[1, 1, 0, 0],    # Token 0's key
     [0, 0, 1, 1],    # Token 1's key
     [1, 0, 1, 0]]    # Token 2's key

V = [[1, 0, 0, 1],    # Token 0's value
     [0, 1, 1, 0],    # Token 1's value
     [1, 1, 0, 0]]    # Token 2's value
```

**Step 1: Compute $Q K^\top$**

```
S = Q @ K^T

S[0][0] = 1*1 + 0*1 + 1*0 + 0*0 = 1    # Token 0 query vs Token 0 key
S[0][1] = 1*0 + 0*0 + 1*1 + 0*1 = 1    # Token 0 query vs Token 1 key
S[0][2] = 1*1 + 0*0 + 1*1 + 0*0 = 2    # Token 0 query vs Token 2 key

S[1][0] = 0*1 + 1*1 + 0*0 + 1*0 = 1
S[1][1] = 0*0 + 1*0 + 0*1 + 1*1 = 1
S[1][2] = 0*1 + 1*0 + 0*1 + 1*0 = 0

S[2][0] = 1*1 + 1*1 + 0*0 + 0*0 = 2
S[2][1] = 1*0 + 1*0 + 0*1 + 0*1 = 0
S[2][2] = 1*1 + 1*0 + 0*1 + 0*0 = 1

S = [[1, 1, 2],
     [1, 1, 0],
     [2, 0, 1]]
```

**Step 2: Scale by $\sqrt{d_k} = \sqrt{4} = 2$**

```
S_scaled = [[0.50, 0.50, 1.00],
            [0.50, 0.50, 0.00],
            [1.00, 0.00, 0.50]]
```

**Step 3: Softmax (row-wise)**

```
Row 0: exp([0.50, 0.50, 1.00]) = [1.649, 1.649, 2.718]
        sum = 6.016
        A[0] = [0.274, 0.274, 0.452]

Row 1: exp([0.50, 0.50, 0.00]) = [1.649, 1.649, 1.000]
        sum = 4.298
        A[1] = [0.384, 0.384, 0.233]

Row 2: exp([1.00, 0.00, 0.50]) = [2.718, 1.000, 1.649]
        sum = 5.367
        A[2] = [0.506, 0.186, 0.307]
```

**Interpretation of attention weights:**
- Token 0 attends most strongly to Token 2 (weight 0.452) — their Q-K similarity was highest
- Token 1 attends roughly equally to Tokens 0 and 1 (0.384 each), less to Token 2
- Token 2 attends most to Token 0 (0.506) — again matching the highest Q-K similarity

**Step 4: Weighted sum with V**

```
output[0] = 0.274 * [1,0,0,1] + 0.274 * [0,1,1,0] + 0.452 * [1,1,0,0]
          = [0.274, 0, 0, 0.274] + [0, 0.274, 0.274, 0] + [0.452, 0.452, 0, 0]
          = [0.726, 0.726, 0.274, 0.274]

output[1] = 0.384 * [1,0,0,1] + 0.384 * [0,1,1,0] + 0.233 * [1,1,0,0]
          = [0.384, 0, 0, 0.384] + [0, 0.384, 0.384, 0] + [0.233, 0.233, 0, 0]
          = [0.617, 0.617, 0.384, 0.384]

output[2] = 0.506 * [1,0,0,1] + 0.186 * [0,1,1,0] + 0.307 * [1,1,0,0]
          = [0.506, 0, 0, 0.506] + [0, 0.186, 0.186, 0] + [0.307, 0.307, 0, 0]
          = [0.813, 0.493, 0.186, 0.506]
```

Each output is a weighted blend of value vectors, with weights determined by query-key compatibility.

### Code

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention.

    Args:
        Q: Queries, shape (batch, ..., seq_len_q, d_k)
        K: Keys, shape (batch, ..., seq_len_k, d_k)
        V: Values, shape (batch, ..., seq_len_k, d_v)
        mask: Optional mask, shape broadcastable to (batch, ..., seq_len_q, seq_len_k)
              Positions with True (or 1) are MASKED (not attended to).

    Returns:
        output: shape (batch, ..., seq_len_q, d_v)
        attention_weights: shape (batch, ..., seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # Step 1: Compute scaled similarity scores
    # (batch, ..., seq_len_q, d_k) @ (batch, ..., d_k, seq_len_k)
    # -> (batch, ..., seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Step 2: Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf'))

    # Step 3: Softmax over key dimension
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### Paper Reference

- Vaswani et al., "Attention Is All You Need" (2017), Section 3.2.1

---

## 3. Multi-Head Attention

### Intuition

A single attention head computes one set of attention weights — one "view" of which tokens are relevant to which. But language is rich: token relationships include syntax (subject-verb agreement), semantics (word meaning similarity), coreference (pronoun resolution), positional proximity, and more. A single attention function cannot capture all these simultaneously.

Multi-head attention runs h parallel attention functions, each with its own learned projections. Each head can specialize in a different type of relationship. The outputs are concatenated and projected back to the model dimension.

Think of it as convening a panel of h experts, each examining the same data but looking for different patterns. Their findings are combined into a single report.

### The Math

Given input $X$ of shape (seq_len, $d_\text{model}$), with $h$ heads:

$$
d_k = d_v = d_\text{model} / h
$$

For each head $i$ ($i = 1, \ldots, h$):

$$
\begin{aligned}
Q_i &= X W_Q^i, \quad K_i = X W_K^i, \quad V_i = X W_V^i \\
\text{head}_i &= \text{Attention}(Q_i,\, K_i,\, V_i)
\end{aligned}
$$

Where $W_Q^i, W_K^i \in \mathbb{R}^{d_\text{model} \times d_k}$ and $W_V^i \in \mathbb{R}^{d_\text{model} \times d_v}$.

Concatenate all heads and project:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W_O
$$

Shape: (seq_len, $h \cdot d_v$) @ ($h \cdot d_v$, $d_\text{model}$) = (seq_len, $d_\text{model}$).

**Parameter count for multi-head attention:**

$$
\begin{aligned}
W_Q &: h \times (d_\text{model} \times d_k) = d_\text{model}^2 \\
W_K &: d_\text{model}^2 \\
W_V &: d_\text{model}^2 \\
W_O &: d_\text{model}^2 \\
\text{Total} &: 4 \cdot d_\text{model}^2
\end{aligned}
$$

For $d_\text{model} = 512$: $4 \times 512^2 = 1{,}048{,}576$ parameters per multi-head attention layer.

Note: the total parameter count is the SAME as a single-head attention with full $d_\text{model}$ dimensions for Q, K, V. Multi-head attention does not add parameters — it restructures the computation to enable parallel, specialized attention patterns.

### Explicit Parameter Dimensions ($d_\text{model}=512$, $h=8$)

```
d_k = d_v = 512 / 8 = 64

Per head:
  W_Q^i: (512, 64)   — 32,768 parameters
  W_K^i: (512, 64)   — 32,768 parameters
  W_V^i: (512, 64)   — 32,768 parameters

All heads combined (equivalent to one large matrix):
  W_Q: (512, 512)    — 262,144 parameters (8 heads * 32,768)
  W_K: (512, 512)    — 262,144 parameters
  W_V: (512, 512)    — 262,144 parameters

Output projection:
  W_O: (512, 512)    — 262,144 parameters

Total: 1,048,576 parameters
```

### What Different Heads Learn

Empirical research has shown that different attention heads specialize:
- Some heads attend to the previous token (bigram patterns)
- Some heads attend to the first token (global context)
- Some heads track syntactic dependencies (subject-verb over long distances)
- Some heads implement coreference resolution (pronoun -> antecedent)
- Some heads attend to separator/special tokens

This specialization emerges naturally from training — it is not explicitly programmed.

### Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention from scratch.

    Args:
        d_model: Total model dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Linear projections for Q, K, V, and output
        # We use single large matrices and reshape, rather than h separate matrices
        self.W_Q = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  (batch, 1, seq_len_q, seq_len_k) or broadcastable

        For self-attention: query = key = value = X
        For cross-attention: query = decoder, key = value = encoder output

        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # Step 1: Linear projections
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Step 2: Reshape to separate heads
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        # -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Scaled dot-product attention (per head, in parallel)
        # Q: (batch, num_heads, seq_len_q, d_k)
        # K: (batch, num_heads, seq_len_k, d_k)
        # scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)

        # (batch, num_heads, seq_len_q, seq_len_k) @ (batch, num_heads, seq_len_k, d_k)
        # -> (batch, num_heads, seq_len_q, d_k)
        context = torch.matmul(attention_weights, V)

        # Step 4: Concatenate heads
        # (batch, num_heads, seq_len_q, d_k) -> (batch, seq_len_q, num_heads, d_k)
        # -> (batch, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Step 5: Final linear projection
        output = self.W_O(context)  # (batch, seq_len_q, d_model)

        return output, attention_weights
```

### Paper Reference

- Vaswani et al., "Attention Is All You Need" (2017), Section 3.2.2
- Voita et al., "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting" (2019)

---

## 4. Positional Encoding

### Intuition

Self-attention is permutation equivariant: if you shuffle the input tokens, the output tokens are shuffled in the same way (each token's output is a weighted sum of value vectors, and the weights depend only on pairwise similarities, not positions). This means a Transformer without positional encoding cannot distinguish "the cat sat on the mat" from "mat the on sat cat the."

We must explicitly inject position information. There are three major approaches:

1. **Sinusoidal (fixed)**: Vaswani et al. (2017). Use sine and cosine functions at different frequencies.
2. **Learned**: BERT, GPT. Train a position embedding matrix.
3. **Rotary (RoPE)**: Su et al. (2021). Encode position through rotation in the complex plane.

### The Math — Sinusoidal Encoding

For position $\text{pos}$ and dimension $i$:

$$
\begin{aligned}
PE(\text{pos},\, 2i) &= \sin\!\left(\text{pos}\, / \, 10000^{2i / d_\text{model}}\right) \\
PE(\text{pos},\, 2i+1) &= \cos\!\left(\text{pos}\, / \, 10000^{2i / d_\text{model}}\right)
\end{aligned}
$$

The encoding for each position is a $d_\text{model}$-dimensional vector. Different dimensions oscillate at different frequencies:
- Dimension 0-1: frequency = 1 (wavelength = $2\pi$ positions, changes rapidly)
- Dimension 2-3: frequency = $1/10000^{2/d_\text{model}}$ (slightly slower)
- ...
- Dimension $d_\text{model}-2$, $d_\text{model}-1$: frequency = $1/10000$ (very slow, wavelength ~ 63,000 positions)

**The Fourier Intuition**: each pair of dimensions encodes position at a different "resolution." Low-indexed dimensions capture fine-grained position differences (nearby tokens have different encodings). High-indexed dimensions capture coarse position (only very distant tokens differ). Together, they create a unique fingerprint for each position — much like how a number can be uniquely represented in binary (each bit oscillates at a different frequency).

**The Relative Position Argument**: for any fixed offset $k$, there exists a linear transformation $M_k$ such that $PE(\text{pos} + k) = M_k \cdot PE(\text{pos})$. This is because:

$$
\begin{aligned}
\sin(a + b) &= \sin(a)\cos(b) + \cos(a)\sin(b) \\
\cos(a + b) &= \cos(a)\cos(b) - \sin(a)\sin(b)
\end{aligned}
$$

So a relative shift by $k$ positions corresponds to a rotation in each sin/cos pair. The model can learn to detect relative positions through linear operations on the encodings.

### Encoding Matrix for 10 Positions, d_model = 8

Here are the values for the first 10 positions (rounded to 3 decimal places):

```
         dim0    dim1    dim2    dim3    dim4    dim5    dim6    dim7
pos 0:  0.000   1.000   0.000   1.000   0.000   1.000   0.000   1.000
pos 1:  0.841   0.540   0.100   0.995   0.010   1.000   0.001   1.000
pos 2:  0.909  -0.416   0.198   0.980   0.020   1.000   0.002   1.000
pos 3:  0.141  -0.990   0.296   0.955   0.030   1.000   0.003   1.000
pos 4: -0.757  -0.654   0.389   0.921   0.040   0.999   0.004   1.000
pos 5: -0.959   0.284   0.479   0.878   0.050   0.999   0.005   1.000
pos 6: -0.279   0.960   0.565   0.825   0.060   0.998   0.006   1.000
pos 7:  0.657   0.754   0.644   0.765   0.070   0.998   0.007   1.000
pos 8:  0.989  -0.146   0.717   0.697   0.080   0.997   0.008   1.000
pos 9:  0.412  -0.911   0.783   0.622   0.090   0.996   0.009   1.000
```

Notice:
- dim0/dim1 oscillate rapidly (high frequency)
- dim6/dim7 barely change (low frequency)
- Each position has a unique pattern across all dimensions

### Learned Positional Embeddings

BERT and GPT simply learn a matrix of shape (max_seq_len, $d_\text{model}$). Position $i$ gets embedding $E_\text{pos}[i]$. Simple, effective, but cannot generalize beyond the maximum sequence length seen during training.

### Rotary Positional Embeddings (RoPE)

RoPE encodes position by rotating the query and key vectors. For a 2D subspace (dimensions $2i$ and $2i+1$), apply a rotation by angle $m \cdot \theta_i$, where $m$ is the position and $\theta_i = 10000^{-2i/d}$:

$$
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
$$

The inner product of rotated $q$ at position $m$ and rotated $k$ at position $n$ depends only on $m - n$ (the relative position):

$$
\langle q'_m,\, k'_n \rangle = \langle R(m)\, q,\, R(n)\, k \rangle = \langle q,\, R(m-n)\, k \rangle
$$

This gives relative position encoding without any additional parameters. RoPE is used in LLaMA, GPT-NeoX, PaLM, and most modern LLMs.

### Code — Sinusoidal Positional Encoding

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need."

    Args:
        d_model: Model dimension (e.g., 512)
        max_len: Maximum sequence length (e.g., 5000)
        dropout: Dropout rate applied after adding positional encoding
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Frequency term: 10000^(2i/d_model) for i = 0, 1, ..., d_model/2 - 1
        # We compute in log space for numerical stability:
        # exp(-2i/d_model * log(10000)) = 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 1, 3, 5, ...

        # Register as buffer (not a parameter — not trained)
        # Shape: (1, max_len, d_model) for easy broadcasting with (batch, seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)
        Returns:
            x + positional encoding, shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

### Code — Rotary Positional Embeddings (RoPE)

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply Rotary Position Embedding to query and key tensors.

    Args:
        q, k: (batch, num_heads, seq_len, d_k)
        cos, sin: (1, 1, seq_len, d_k) -- precomputed cos/sin for each position
    """
    # Split into pairs: (q0, q1), (q2, q3), ...
    q_even = q[..., 0::2]  # (batch, heads, seq_len, d_k/2)
    q_odd  = q[..., 1::2]

    k_even = k[..., 0::2]
    k_odd  = k[..., 1::2]

    cos_half = cos[..., 0::2]  # (1, 1, seq_len, d_k/2)
    sin_half = sin[..., 0::2]

    # Apply rotation
    q_rotated_even = q_even * cos_half - q_odd * sin_half
    q_rotated_odd  = q_even * sin_half + q_odd * cos_half
    k_rotated_even = k_even * cos_half - k_odd * sin_half
    k_rotated_odd  = k_even * sin_half + k_odd * cos_half

    # Interleave back
    q_rotated = torch.stack([q_rotated_even, q_rotated_odd], dim=-1).flatten(-2)
    k_rotated = torch.stack([k_rotated_even, k_rotated_odd], dim=-1).flatten(-2)

    return q_rotated, k_rotated


def precompute_rope(d_k, max_len, base=10000.0):
    """Precompute cos and sin for RoPE."""
    freqs = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k/2,)
    positions = torch.arange(max_len).float()  # (max_len,)
    angles = torch.outer(positions, freqs)  # (max_len, d_k/2)
    # Duplicate for paired dimensions
    angles = angles.repeat(1, 2)  # (max_len, d_k) -- not strictly needed, but matches shapes
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, max_len, d_k)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    return cos, sin
```

### Paper Reference

- Vaswani et al., "Attention Is All You Need" (2017), Section 3.5 — sinusoidal encoding
- Devlin et al., "BERT" (2019) — learned positional embeddings
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)

---

## 5. The Transformer Encoder Block

### Intuition

The encoder block is the fundamental building block of the Transformer encoder. It processes the input sequence and produces contextualized representations — representations that are aware of the entire input context.

Each encoder block has two sub-layers:
1. **Multi-head self-attention**: each token gathers information from all other tokens
2. **Position-wise feed-forward network (FFN)**: processes each token independently through a two-layer MLP

Both sub-layers use residual connections and layer normalization. The residual connections allow gradients to flow directly through the network. Layer normalization stabilizes training by normalizing activations.

### The Math

Input: $x$ of shape (batch, seq_len, $d_\text{model}$)

**Sub-layer 1: Multi-Head Self-Attention + Add & Norm**

$$
\begin{aligned}
\text{attn\_output} &= \text{MultiHeadAttention}(x, x, x) \quad \text{(Q=K=V=x, self-attention)} \\
x &= \text{LayerNorm}(x + \text{attn\_output}) \quad \text{(Residual + Norm)}
\end{aligned}
$$

**Sub-layer 2: Feed-Forward Network + Add & Norm**

$$
\begin{aligned}
\text{FFN}(x) &= \max(0,\, x W_1 + b_1)\, W_2 + b_2 \\
x &= \text{LayerNorm}(x + \text{FFN}(x)) \quad \text{(Residual + Norm)}
\end{aligned}
$$

FFN dimensions:
- $W_1 \in \mathbb{R}^{d_\text{model} \times d_{ff}}$, typically $d_{ff} = 4 \cdot d_\text{model}$
- $W_2 \in \mathbb{R}^{d_{ff} \times d_\text{model}}$
- This is applied position-wise: the same FFN is applied independently to each position

### Walk-Through With Explicit Tensor Shapes

Configuration: batch_size=2, seq_len=10, $d_\text{model}$=512, $d_{ff}$=2048, num_heads=8, $d_k$=64

```
Input x:                          (2, 10, 512)

--- Sub-layer 1: Multi-Head Self-Attention ---

Q = x @ W_Q:                     (2, 10, 512) @ (512, 512) -> (2, 10, 512)
K = x @ W_K:                     (2, 10, 512) @ (512, 512) -> (2, 10, 512)
V = x @ W_V:                     (2, 10, 512) @ (512, 512) -> (2, 10, 512)

Reshape to heads:
Q:                                (2, 10, 8, 64) -> transpose -> (2, 8, 10, 64)
K:                                (2, 10, 8, 64) -> transpose -> (2, 8, 10, 64)
V:                                (2, 10, 8, 64) -> transpose -> (2, 8, 10, 64)

Attention scores = Q @ K^T / 8:  (2, 8, 10, 64) @ (2, 8, 64, 10) -> (2, 8, 10, 10)
                                  (divided by sqrt(64) = 8)

Attention weights (softmax):      (2, 8, 10, 10)
                                  (each row sums to 1)

Context = weights @ V:            (2, 8, 10, 10) @ (2, 8, 10, 64) -> (2, 8, 10, 64)

Concatenate heads:                (2, 8, 10, 64) -> transpose -> (2, 10, 8, 64) -> (2, 10, 512)

Output projection = concat @ W_O: (2, 10, 512) @ (512, 512) -> (2, 10, 512)

Residual + LayerNorm:             LayerNorm((2, 10, 512) + (2, 10, 512)) -> (2, 10, 512)

--- Sub-layer 2: Feed-Forward Network ---

FFN layer 1 = ReLU(x @ W_1 + b_1): (2, 10, 512) @ (512, 2048) -> (2, 10, 2048) -> ReLU

FFN layer 2 = ffn1 @ W_2 + b_2:    (2, 10, 2048) @ (2048, 512) -> (2, 10, 512)

Residual + LayerNorm:               LayerNorm((2, 10, 512) + (2, 10, 512)) -> (2, 10, 512)

Output:                             (2, 10, 512)
```

The output shape is identical to the input shape. Encoder blocks are stackable.

### Pre-Norm vs Post-Norm

The original Transformer uses Post-Norm: $\text{LayerNorm}(x + \text{SubLayer}(x))$.

Many modern implementations use Pre-Norm: $x + \text{SubLayer}(\text{LayerNorm}(x))$.

Pre-Norm places the LayerNorm before the sub-layer, which makes the residual connection a clean identity path. This stabilizes training for deep models (no need for warmup in some cases) but may slightly reduce final performance. Most LLMs (GPT-3, LLaMA, PaLM) use Pre-Norm.

### Code

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.

    Args:
        d_model: Model dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        d_ff: Feed-forward inner dimension (e.g., 2048)
        dropout: Dropout rate
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            (batch, seq_len, d_model)
        """
        # Sub-layer 1: Multi-Head Self-Attention + Add & Norm (Post-Norm)
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Sub-layer 2: Feed-Forward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x
```

### Paper Reference

- Vaswani et al., "Attention Is All You Need" (2017), Section 3.1
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020) — Pre-Norm vs Post-Norm

---

## 6. The Transformer Decoder Block

### Intuition

The decoder block is more complex than the encoder block because it must do two things:
1. Attend to previously generated tokens (but NOT future tokens — causal constraint)
2. Attend to the encoder output (cross-attention — gathering information from the input)

This gives the decoder three sub-layers:
1. **Masked self-attention**: self-attention with a causal mask preventing attention to future positions
2. **Cross-attention**: queries from the decoder, keys and values from the encoder
3. **Feed-forward network**: same as in the encoder

### The Math

Input: $y$ (decoder input) of shape (batch, tgt_len, $d_\text{model}$)
Encoder output: $\text{enc\_out}$ of shape (batch, src_len, $d_\text{model}$)

**Sub-layer 1: Masked Multi-Head Self-Attention + Add & Norm**

$$
\begin{aligned}
\text{attn\_output} &= \text{MultiHeadAttention}(y, y, y, \text{mask}=\text{causal\_mask}) \\
y &= \text{LayerNorm}(y + \text{attn\_output})
\end{aligned}
$$

**Sub-layer 2: Cross-Attention + Add & Norm**

$$
\begin{aligned}
\text{cross\_output} &= \text{MultiHeadAttention}(y, \text{enc\_out}, \text{enc\_out}) \\
y &= \text{LayerNorm}(y + \text{cross\_output})
\end{aligned}
$$

**Sub-layer 3: Feed-Forward + Add & Norm**

$$
\begin{aligned}
\text{ffn\_output} &= \text{FFN}(y) \\
y &= \text{LayerNorm}(y + \text{ffn\_output})
\end{aligned}
$$

### Code

```python
class TransformerDecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block.

    Three sub-layers:
    1. Masked self-attention (causal)
    2. Cross-attention (decoder queries, encoder keys/values)
    3. Feed-forward network
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input, (batch, tgt_len, d_model)
            encoder_output: Encoder output, (batch, src_len, d_model)
            src_mask: Mask for encoder output (padding mask)
            tgt_mask: Causal mask for decoder (upper triangular)

        Returns:
            (batch, tgt_len, d_model)
        """
        # Sub-layer 1: Masked Self-Attention
        attn_output, _ = self.masked_self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Sub-layer 2: Cross-Attention
        # Queries from decoder, Keys and Values from encoder
        cross_output, _ = self.cross_attention(x, encoder_output, encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout2(cross_output))

        # Sub-layer 3: Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x
```

### Paper Reference

- Vaswani et al., "Attention Is All You Need" (2017), Section 3.1

---

## 7. The Full Transformer Architecture

### Intuition

The complete Transformer is an encoder-decoder architecture:

**Encoder**: stack of N encoder blocks. Takes the source sequence and produces contextualized representations. Each token's representation is enriched with information from all other source tokens.

**Decoder**: stack of N decoder blocks. Takes the target sequence (shifted right) and the encoder output. Generates the output sequence one token at a time during inference (but processes all positions in parallel during training with teacher forcing).

The input flow:
1. Source tokens -> embedding + positional encoding -> N encoder blocks -> encoder output
2. Target tokens -> embedding + positional encoding -> N decoder blocks (with cross-attention to encoder output) -> linear projection -> softmax -> output probabilities

### Code

```python
class Transformer(nn.Module):
    """
    Full Transformer (Encoder-Decoder) from "Attention Is All You Need."

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension (default 512)
        num_heads: Number of attention heads (default 8)
        num_layers: Number of encoder/decoder blocks (default 6)
        d_ff: FFN inner dimension (default 2048)
        max_len: Maximum sequence length (default 5000)
        dropout: Dropout rate (default 0.1)
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Scale embeddings by sqrt(d_model) as in the paper
        self.d_model = d_model
        self.scale = d_model ** 0.5

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        # (batch, src_len) -> (batch, src_len, d_model)
        x = self.src_embedding(src) * self.scale
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence given encoder output."""
        # (batch, tgt_len) -> (batch, tgt_len, d_model)
        x = self.tgt_embedding(tgt) * self.scale
        x = self.positional_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Full forward pass.

        Args:
            src: Source token IDs, (batch, src_len)
            tgt: Target token IDs, (batch, tgt_len)
            src_mask: Source padding mask
            tgt_mask: Target causal mask

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits

    @staticmethod
    def generate_causal_mask(seq_len):
        """Generate upper-triangular causal mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask  # 1 = masked, 0 = attend
```

### Parameter Count

For the base Transformer (d_model=512, d_ff=2048, N=6, h=8, vocab=37000):

```
Per encoder block:
  Multi-head attention: 4 * 512^2           =  1,048,576
  FFN: 512*2048 + 2048*512                  =  2,097,152
  LayerNorm (x2): 2 * 2 * 512              =      2,048
  Subtotal:                                 =  3,147,776

6 encoder blocks:                           = 18,886,656

Per decoder block:
  Masked self-attention: 4 * 512^2          =  1,048,576
  Cross-attention: 4 * 512^2                =  1,048,576
  FFN: 2 * 512 * 2048                       =  2,097,152
  LayerNorm (x3): 3 * 2 * 512              =      3,072
  Subtotal:                                 =  4,197,376

6 decoder blocks:                           = 25,184,256

Embeddings:
  Source: 37000 * 512                       = 18,944,000
  Target: 37000 * 512                       = 18,944,000

Output projection: 512 * 37000             = 18,944,000

Positional encoding (not learned):                    0

Total:                                     ~100,902,912 (~101M parameters)
```

Note: in the original paper, source and target embeddings and the output projection share weights, reducing the count significantly.

### Paper Reference

- Vaswani et al., "Attention Is All You Need" (2017), full paper

---

## 8. Causal Masking

### Intuition

In autoregressive generation, when predicting token $t$, the model should only see tokens $1$ through $t-1$ — not tokens $t+1$ onward (that would be cheating; those tokens have not been generated yet). During training with teacher forcing, we process all positions in parallel but must still prevent each position from attending to future positions.

The solution: add a mask to the attention scores before softmax. Positions that should not be attended to get a score of $-\infty$. After softmax, $\exp(-\infty) = 0$, so those positions contribute nothing to the weighted sum.

### The Math

For a sequence of length $n$, the causal mask is an upper-triangular matrix:

$$
\text{mask} = \begin{bmatrix} 0 & -\infty & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty & -\infty \\ 0 & 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}
$$

Position $i$ can attend to positions $j$ where $j \leq i$ (lower triangle + diagonal).

After adding the mask to the attention scores:

$$
\text{masked\_scores}[i][j] = \begin{cases} \text{scores}[i][j] & \text{if } j \leq i \quad (\text{mask is } 0) \\ -\infty & \text{if } j > i \quad (\text{mask is } -\infty) \end{cases}
$$

After softmax:

$$
\text{attention\_weights}[i][j] = \begin{cases} 0 & \text{if } j > i \quad (\exp(-\infty) = 0) \\ \propto \exp(\text{scores}[i][j]) & \text{if } j \leq i \end{cases}
$$

**Why -inf works and not just 0 or a large negative number:**

Using 0 does not work — the softmax would still assign non-zero weight to future positions. Using a large negative number (like $-10^9$) works in practice but is not as clean. Using $-\infty$ is mathematically exact: $\exp(-\infty) = 0$, guaranteeing that future positions get exactly zero attention weight.

In PyTorch, `float('-inf')` is a valid float value. Softmax handles it correctly: if all inputs to softmax are $-\infty$, the output is uniform (NaN in theory, but PyTorch handles it). In practice, at least one non-masked position always exists (the current position can attend to itself).

### Code

```python
def create_causal_mask(seq_len):
    """
    Create a causal (look-ahead) mask for autoregressive attention.

    Returns a mask where mask[i][j] = 1 if j > i (position i cannot attend to position j).
    Use with: scores.masked_fill(mask == 1, float('-inf'))

    Args:
        seq_len: Length of the sequence

    Returns:
        mask: (seq_len, seq_len), upper triangular with 1s above diagonal
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask


def create_causal_mask_additive(seq_len):
    """
    Alternative: create additive mask (add directly to scores before softmax).

    Returns:
        mask: (seq_len, seq_len), 0 on and below diagonal, -inf above
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask


# Usage in attention:
def masked_attention(Q, K, V, causal=True):
    """Attention with optional causal masking."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if causal:
        seq_len = Q.size(-2)
        mask = create_causal_mask_additive(seq_len).to(Q.device)
        scores = scores + mask

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights
```

### Visualization

For seq_len=5, the attention weight matrix after masking looks like this (example values):

```
Token 0 attends to: [1.00, 0.00, 0.00, 0.00, 0.00]  <- can only see itself
Token 1 attends to: [0.35, 0.65, 0.00, 0.00, 0.00]  <- sees tokens 0-1
Token 2 attends to: [0.20, 0.30, 0.50, 0.00, 0.00]  <- sees tokens 0-2
Token 3 attends to: [0.10, 0.15, 0.25, 0.50, 0.00]  <- sees tokens 0-3
Token 4 attends to: [0.08, 0.12, 0.20, 0.25, 0.35]  <- sees all tokens
```

The upper-right triangle is always exactly zero.

### Paper Reference

- Vaswani et al., "Attention Is All You Need" (2017), Section 3.1

---

## 9. Training Objectives

### 9.1 Autoregressive Language Modeling (GPT-style)

**Intuition**: predict the next word given all previous words. The model learns the distribution $P(x_t \mid x_1, \ldots, x_{t-1})$.

**The Math**:

Loss function (cross-entropy over the vocabulary at each position):

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})
$$

The model uses causal masking to ensure position $t$ only sees positions $1$ through $t-1$. During training, all positions are computed in parallel (teacher forcing). During inference, tokens are generated one at a time.

### 9.2 Masked Language Modeling (BERT-style)

**Intuition**: randomly mask some tokens and predict them from context. The model learns bidirectional representations.

**The Math**:

1. Select 15% of tokens for prediction
2. Of selected tokens: 80% are replaced with [MASK], 10% with a random token, 10% unchanged
3. Loss = cross-entropy only at the masked positions:

$$
\mathcal{L} = -\frac{1}{|M|} \sum_{t \in M} \log P(x_t \mid x_{\setminus M})
$$

Where $M$ is the set of masked positions and $x_{\setminus M}$ means all tokens except the masked ones.

The 80/10/10 split is important: if we always used [MASK], the model would never see [MASK] during fine-tuning, creating a train-test mismatch. The random token replacement and identity ensure the model cannot simply learn to ignore [MASK] tokens.

### 9.3 Learning Rate Warmup

**Intuition**: Transformers are unusually sensitive to learning rate, especially early in training. The Adam optimizer estimates second moments (squared gradients) from a running average. Early in training, these estimates are inaccurate (biased toward zero), leading to overly large parameter updates. Warmup gradually increases the learning rate, giving the optimizer time to build reliable estimates.

**The Noam Schedule** (from the original Transformer paper):

$$
\text{lr} = d_\text{model}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)
$$

This increases linearly for the first `warmup_steps` steps, then decays proportionally to the inverse square root of the step number. Peak learning rate occurs at $\text{step} = \text{warmup\_steps}$.

```python
class NoamScheduler:
    """Learning rate scheduler from 'Attention Is All You Need'."""
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
```

### 9.4 Label Smoothing

**Intuition**: instead of training with hard one-hot targets (all probability on the correct class), distribute a small amount of probability mass uniformly across all classes. This prevents the model from becoming overconfident and improves generalization.

**The Math**:

With label smoothing parameter $\epsilon$:

$$
y_\text{smooth}[i] = (1 - \epsilon) \cdot y_\text{onehot}[i] + \frac{\epsilon}{V}
$$

where $V$ is the vocabulary size and $y_\text{onehot}$ is the standard one-hot target.

For $\epsilon = 0.1$ and $V = 10000$, the correct class gets weight 0.9 and each incorrect class gets weight $0.1 / 10000 = 0.00001$.

Cross-entropy with smoothed labels:

$$
\begin{aligned}
\mathcal{L} &= -\sum_i y_\text{smooth}[i] \cdot \log p[i] \\
&= -(1 - \epsilon) \cdot \log p[\text{correct}] - \frac{\epsilon}{V} \sum_i \log p[i] \\
&= (1 - \epsilon) \cdot H(y, p) + \epsilon \cdot H(u, p)
\end{aligned}
$$

Where $H(y, p)$ is the standard cross-entropy and $H(u, p)$ is the cross-entropy with a uniform distribution.

### 9.5 The FFN as Key-Value Memories

**Intuition**: Geva et al. (2021) showed that FFN layers in Transformers can be interpreted as unnormalized key-value memories:

$$
\text{FFN}(x) = \text{ReLU}(x\, W_1)\, W_2
$$

- $W_1$ rows are "keys" — they match specific input patterns (activating for certain token/context combinations)
- $W_2$ columns are "values" — they encode the knowledge that should be added to the representation when the corresponding key matches
- ReLU acts as a sparse selection mechanism — only keys that match (positive activation) contribute

This means factual knowledge in language models (e.g., "Paris is the capital of France") is stored in the FFN weights. Editing these weights can modify the model's factual knowledge.

### Paper Reference

- Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018) — GPT
- Devlin et al., "BERT" (2019) — MLM, NSP
- Vaswani et al., "Attention Is All You Need" (2017) — warmup, label smoothing
- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (2021)

---

## 10. The Residual Stream View

### Intuition

The standard way to think about a Transformer: data flows through a pipeline of layers, each transforming the representation.

The residual stream view (Elhage et al., 2021, Anthropic): instead of a pipeline, think of a shared communication channel — the residual stream. Each attention head and MLP layer READS from this stream, performs a computation, and WRITES (adds) its result back to the stream. The final output is the accumulated sum of the initial embedding plus all the additions from every layer.

$$
x_\text{final} = x_0 + \text{attn}_1(x_0) + \text{mlp}_1(x_0 + \text{attn}_1(x_0)) + \text{attn}_2(\ldots) + \text{mlp}_2(\ldots) + \ldots
$$

**Why this view is powerful:**

1. **Superposition**: multiple features can coexist in the same residual stream because they are represented as directions in a high-dimensional space.

2. **Skip connections are the default path**: if a layer is not useful for a particular input, it can output near-zero and the information passes through unchanged. Layers do not need to learn the identity function — it is free.

3. **Composition**: attention head A in layer 2 can read the output of attention head B in layer 1 (because B's output was added to the stream). This enables complex multi-step computations: head B identifies the subject, head A uses that information to find the verb.

4. **Interpretability**: we can analyze what each head/MLP writes to the stream and trace information flow through the network. This is the foundation of mechanistic interpretability.

### The Math

For a Transformer with $L$ layers, each containing an attention sublayer and an MLP sublayer:

$$
\begin{aligned}
x_0 &= \text{embedding} + \text{positional\_encoding} \\[6pt]
\text{For } l &= 1 \text{ to } L: \\
x'_l &= x_{l-1} + \text{Attention}_l(x_{l-1}) \\
x_l &= x'_l + \text{MLP}_l(x'_l) \\[6pt]
\text{output} &= \text{LayerNorm}(x_L) \\
\text{logits} &= x_L\, W_\text{unembed}
\end{aligned}
$$

Expanding:

$$
x_L = x_0 + \sum_{l=1}^{L} \left[\text{Attention}_l(\ldots) + \text{MLP}_l(\ldots)\right]
$$

Each term in the sum is a "contribution" from that layer. We can measure the magnitude and direction of each contribution to understand what the layer is doing.

### Virtual Attention Heads

Because attention heads read from and write to the residual stream, they can form "virtual attention heads" through composition:

- **Q-composition**: head B's output is used as the query for head A (in a later layer)
- **K-composition**: head B's output is used as the key for head A
- **V-composition**: head B's output is part of the value that head A reads

This means the effective computation of a Transformer is richer than the sum of individual heads — heads collaborate across layers through the residual stream.

### Implications for Understanding Modern LLMs

The residual stream view explains several phenomena:
- **Why deeper models are better**: more layers = more opportunities for heads and MLPs to add useful information to the stream
- **Why pruning heads often works**: many heads write small contributions; removing them barely changes the stream
- **Why Transformers can do in-context learning**: the residual stream accumulates evidence from the prompt, and later layers use this accumulated evidence to generate appropriate responses

### Paper Reference

- Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)
- Olsson et al., "In-context Learning and Induction Heads" (2022)

---

## 11. Transformer Variants

### 11.1 BERT (Bidirectional Encoder Representations from Transformers)

**Architecture**: Encoder-only Transformer (12 layers for BERT-base, 24 for BERT-large)
**Training**: MLM + Next Sentence Prediction (NSP)
**Fine-tuning**: Add task-specific head on top of [CLS] or token representations

```
Input:  [CLS] The cat sat on the [MASK] . [SEP]
Output: [CLS_repr] [tok1_repr] ... [mask_repr] ... [SEP_repr]

For classification: feed [CLS_repr] through a linear classifier
For token tasks:    feed each token representation through a classifier
For MLM pretraining: predict "mat" at the [MASK] position
```

Key hyperparameters (BERT-base): d_model=768, h=12, L=12, d_ff=3072, vocab=30522 (WordPiece)

### 11.2 GPT (Generative Pre-trained Transformer)

**Architecture**: Decoder-only Transformer (no encoder, no cross-attention)
**Training**: Autoregressive language modeling
**Usage**: generation, in-context learning (GPT-3+)

```
Input:  The cat sat on the
Model:  Predicts next token at each position (using causal mask)
        P("cat"|"The"), P("sat"|"The cat"), ..., P("mat"|"The cat sat on the")

Generation: sample from P(next | context), append, repeat
```

The GPT decoder block is simpler than the original Transformer decoder — it has only:
1. Masked self-attention + Add & Norm
2. FFN + Add & Norm
(No cross-attention, because there is no encoder.)

GPT-3 key hyperparameters: d_model=12288, h=96, L=96, d_ff=49152, 175B parameters

### 11.3 T5 (Text-to-Text Transfer Transformer)

**Architecture**: Full encoder-decoder Transformer
**Training**: Span corruption (mask random contiguous spans of tokens)
**Key idea**: frame EVERY task as text-to-text

```
Classification: "classify: The movie was great"    -> "positive"
Translation:    "translate English to French: Hi"   -> "Bonjour"
Summarization:  "summarize: [long article]"         -> "[summary]"
QA:             "question: What is... context: ..." -> "42"
```

This unification means the same model, same loss function, same decoding procedure for all tasks.

### 11.4 Comparison Table

```
                BERT            GPT             T5
Architecture    Encoder-only    Decoder-only    Encoder-Decoder
Attention       Bidirectional   Causal (left)   Bidirectional (enc) + Causal (dec)
Pretraining     MLM + NSP       Autoregressive  Span Corruption
Adaptation      Fine-tuning     Prompting/ICL   Fine-tuning / Prompting
Generation      Unnatural       Natural         Natural
Understanding   Strong          Weaker*         Strong
Parameters      110M/340M       117M-175B       220M-11B
```

*GPT's "weaker" understanding is debatable — at scale, GPT-3+ shows strong understanding through in-context learning.

### Paper Reference

- Devlin et al., "BERT" (2019)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019) — GPT-2
- Brown et al., "Language Models are Few-Shot Learners" (2020) — GPT-3
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020) — T5

---

## 12. Vision Transformers

### 12.1 ViT (Vision Transformer)

**Intuition**: treat an image as a sequence of patches, just like NLP treats a sentence as a sequence of tokens.

**How it works:**

1. Split the image into non-overlapping patches of size $P \times P$ (e.g., $16 \times 16$)
2. Flatten each patch: a $P \times P \times C$ patch becomes a vector of dimension $P^2 \cdot C$
3. Linearly project each flattened patch to $d_\text{model}$ dimensions (the "patch embedding")
4. Prepend a learnable [CLS] token
5. Add learnable position embeddings
6. Feed through a standard Transformer encoder
7. Use the [CLS] token output for classification

For a $224 \times 224$ image with $16 \times 16$ patches:
- Number of patches: $(224/16) \times (224/16) = 14 \times 14 = 196$
- Sequence length: $196 + 1$ (CLS) $= 197$
- Each patch: $16 \times 16 \times 3 = 768$ dimensions -> project to $d_\text{model}$

**Why ViTs need more data than CNNs:**

CNNs have strong inductive biases:
- **Translation equivariance**: a feature detector works the same everywhere in the image (weight sharing)
- **Locality**: each neuron only sees a small receptive field (local connectivity)

These biases are excellent priors for image data. ViTs lack them — they must learn these properties from data. With limited data, the CNN's built-in priors give it an advantage. With massive data (JFT-300M), the ViT surpasses CNNs because it is not constrained by those biases.

### Code

```python
class PatchEmbedding(nn.Module):
    """Convert images to sequences of patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        # Equivalent to a Conv2d with kernel_size=patch_size and stride=patch_size
        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            (batch, num_patches, d_model)
        """
        # (batch, d_model, H/P, W/P)
        x = self.projection(x)
        # (batch, d_model, num_patches) -> (batch, num_patches, d_model)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Minimal Vision Transformer (ViT)."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embedding.num_patches

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # Patch embedding: (batch, num_patches, d_model)
        x = self.patch_embedding(x)

        # Prepend [CLS] token: (batch, num_patches + 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Classification: use [CLS] token output
        x = self.norm(x[:, 0])  # (batch, d_model)
        logits = self.classifier(x)  # (batch, num_classes)

        return logits
```

### 12.2 DeiT and Distillation

DeiT adds a distillation token alongside the CLS token. The CLS token is trained with the hard label (standard cross-entropy). The distillation token is trained to match the output of a CNN teacher. This injects CNN-like inductive biases without modifying the architecture.

### 12.3 Swin Transformer

Key innovations:
- **Windowed attention**: compute self-attention within local windows (e.g., $7 \times 7$ patches), reducing complexity from $O(n^2)$ to $O(n \cdot w^2)$ where $w$ is window size
- **Shifted windows**: in alternating layers, shift window boundaries by half the window size to allow cross-window information flow
- **Hierarchical structure**: progressively merge patches (like pooling), producing multi-scale feature maps for detection and segmentation

### Paper Reference

- Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020) — ViT
- Touvron et al., "Training data-efficient image transformers & distillation through attention" (2021) — DeiT
- Liu et al., "Swin Transformer" (2021)

---

## 13. Modern Efficiency Techniques

### 13.1 Flash Attention

**The Problem**: standard attention materializes the full $N \times N$ attention matrix. For $N = 8192$ (a moderate sequence length), this matrix has 67 million entries per head. Reading and writing this matrix from GPU HBM (high bandwidth memory) is the bottleneck — not the arithmetic.

**The Solution**: Flash Attention (Dao et al., 2022) tiles the computation into blocks that fit in SRAM (fast on-chip memory, ~20 MB on A100 vs 40-80 GB HBM). It never materializes the full attention matrix. Instead:

1. Load blocks of Q, K, V from HBM to SRAM
2. Compute attention for those blocks in SRAM
3. Use the "online softmax trick" to accumulate results across blocks without needing the full attention matrix
4. Write only the final output back to HBM

**The Online Softmax Trick**: normally, softmax requires two passes — one to compute the max (for numerical stability), one to compute the exponentials and normalize. The online version maintains a running max and running sum, updating them as new blocks arrive. This allows computing exact softmax in a single streaming pass.

**Results**: 2-4x wall-clock speedup, $O(N)$ memory instead of $O(N^2)$. This is NOT an approximation — it computes exact attention.

### 13.2 KV-Cache

**The Problem**: during autoregressive generation, at step $t$, the model computes attention over all $t$ tokens. Without caching, we recompute K and V projections for all previous tokens at every step. Total work: $O(n^2 \cdot d)$ for generating $n$ tokens.

**The Solution**: cache the K and V tensors for all previous tokens. At step $t$, only compute $K_t$ and $V_t$ for the new token, then concatenate with the cached values. The attention computation is then $Q_t$ ($1 \times d_k$) against all cached keys ($t \times d_k$), producing attention weights ($1 \times t$) and output ($1 \times d_v$). Total work: $O(n \cdot d)$ over all steps.

**Memory cost**: for each layer, store K and V tensors of shape (seq_len, $d_k$). For a model with $L$ layers and $h$ heads:

$$
\text{KV-cache memory} = 2 \cdot L \cdot h \cdot \text{seq\_len} \cdot d_k \cdot \text{bytes\_per\_param}
$$

Example (LLaMA 70B): $2 \times 80 \times 64 \times 4096 \times 128 \times 2$ bytes (fp16) $\approx 10.7$ GB for a single sequence of 4096 tokens.

This is why GQA (reducing the number of KV heads) is so important for inference.

### 13.3 Rotary Positional Embeddings (RoPE)

Covered in Section 4, but here are the key advantages for modern LLMs:

1. **No additional parameters**: positions are encoded through rotation, not learned embeddings
2. **Relative position**: the attention score between positions m and n depends only on m-n
3. **Decay with distance**: attention between distant tokens is naturally attenuated
4. **Length extrapolation**: better generalization to longer sequences than learned embeddings (though not perfect -- techniques like NTK-aware scaling further improve this)

### 13.4 Grouped Query Attention (GQA)

**Multi-Head Attention (MHA)**: each head has its own Q, K, V projections
- $H$ query heads, $H$ key heads, $H$ value heads
- KV-cache size proportional to $H$

**Multi-Query Attention (MQA)**: all heads share a single K, V projection
- $H$ query heads, 1 key head, 1 value head
- KV-cache reduced by factor $H$
- May reduce quality

**Grouped Query Attention (GQA)**: middle ground
- $H$ query heads, $G$ key-value heads ($G < H$, $G > 1$)
- Groups of $H/G$ query heads share one KV head
- KV-cache reduced by factor $H/G$
- Quality very close to MHA

Example (LLaMA 2 70B): $H=64$ query heads, $G=8$ KV heads. KV-cache is 8x smaller than MHA.

```python
class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA)."""
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads  # queries per KV head
        self.d_k = d_model // num_q_heads

        self.W_Q = nn.Linear(d_model, num_q_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(num_q_heads * self.d_k, d_model, bias=False)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.num_q_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Expand K, V to match Q's head count by repeating
        # (batch, num_kv_heads, seq_len, d_k) -> (batch, num_q_heads, seq_len, d_k)
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        # Standard scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)

        # Concatenate and project
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_O(context)
```

### 13.5 Mixture of Experts (MoE)

**Intuition**: instead of one large FFN, use multiple smaller "expert" FFNs and a routing mechanism that selects the top-k experts for each token. This allows scaling model capacity (total parameters) without proportionally scaling compute (only k experts run per token).

**The Math**:

$$
\begin{aligned}
\text{Router}(x) &= \text{softmax}(x\, W_\text{gate}) \\
\text{top-}k &= \text{TopK}(\text{Router}(x),\, k) \\
\text{output} &= \sum_i g_i \cdot \text{Expert}_i(x)
\end{aligned}
$$

**Load balancing**: a naive router might always select the same experts, leaving others unused. An auxiliary loss encourages uniform expert utilization:

$$
\mathcal{L}_\text{balance} = \alpha \cdot N_\text{experts} \cdot \sum_i f_i \cdot P_i
$$

where $f_i$ = fraction of tokens routed to expert $i$ and $P_i$ = average router probability for expert $i$.

**Key models using MoE**: Switch Transformer (top-1 routing, 1.6T params), Mixtral 8x7B (8 experts, top-2, 47B total params but ~13B active), GShard.

### Paper Reference

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- Fedus et al., "Switch Transformers" (2021)

---

## 14. Tokenization

### Intuition

Neural networks operate on numbers, not text. We need to convert text to token IDs. The choice of tokenization significantly impacts model performance:

- **Word-level**: "I love machine learning" -> ["I", "love", "machine", "learning"]. Problem: vocabulary is massive (100K+ words), rare words get poor representations, completely fails on unseen words.
- **Character-level**: "cat" -> ["c", "a", "t"]. Problem: sequences are very long (3-5x), hard for the model to learn word-level semantics from characters.
- **Subword**: the middle ground. "unhappiness" -> ["un", "happi", "ness"]. Vocabulary of 30-50K tokens captures common words as single tokens and rare words as compositions of frequent subwords.

### BPE (Byte Pair Encoding)

Algorithm:
1. Start with a vocabulary of individual characters
2. Count all pairs of adjacent tokens in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size is reached

Example:
```
Corpus: "the cat sat on the mat"
Initial: ['t', 'h', 'e', ' ', 'c', 'a', 't', ' ', 's', 'a', 't', ' ', 'o', 'n', ' ', 't', 'h', 'e', ' ', 'm', 'a', 't']

Step 1: Most frequent pair: ('t', 'h') -> merge to 'th'
Step 2: Most frequent pair: ('th', 'e') -> merge to 'the'
Step 3: Most frequent pair: ('a', 't') -> merge to 'at'
...
```

### WordPiece

Similar to BPE, but instead of merging the most frequent pair, merges the pair that maximizes the likelihood of the training data. Used by BERT. Prefixes subword tokens with "##":

```
"unhappiness" -> ["un", "##happi", "##ness"]
```

### SentencePiece

Language-agnostic tokenization that operates on raw text (no pre-tokenization into words). Treats the entire input as a sequence of unicode characters and learns a subword vocabulary. Used by T5, LLaMA.

```python
# Simple BPE implementation sketch
def train_bpe(corpus, num_merges):
    """Train BPE tokenizer."""
    # Start with character-level tokens
    vocab = set(corpus)
    tokens = list(corpus)

    merges = []
    for _ in range(num_merges):
        # Count all adjacent pairs
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair] = pairs.get(pair, 0) + 1

        if not pairs:
            break

        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        # Merge best pair into new token
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)

        # Replace all occurrences in tokens list
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return vocab, merges
```

### Paper Reference

- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016) — BPE
- Wu et al., "Google's Neural Machine Translation System" (2016) — WordPiece
- Kudo and Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing" (2018)

---

## 15. Scaling Laws

### Intuition

How does model performance improve as we increase model size, dataset size, and compute? Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") showed that these follow predictable power laws.

### Kaplan Scaling Laws (2020)

Test loss follows power laws with respect to:

$$
\begin{aligned}
L(N) &\sim N^{-0.076} & \text{(model parameters, fixed data and compute)} \\
L(D) &\sim D^{-0.095} & \text{(dataset size, fixed model and compute)} \\
L(C) &\sim C^{-0.050} & \text{(compute budget, optimal } N \text{ and } D\text{)}
\end{aligned}
$$

Key findings:
- Larger models are more sample-efficient (need less data per parameter)
- Performance is a smooth function of compute — no sharp phase transitions
- Architecture details (depth vs width) matter less than total parameter count

### Chinchilla Scaling Laws (2022)

Hoffman et al. found that Kaplan's recommendations led to undertrained models. The compute-optimal ratio:

$$
\text{Optimal tokens} \approx 20 \cdot N
$$

For a 70B parameter model: train on ~1.4 trillion tokens. For a 7B parameter model: train on ~140 billion tokens.

Chinchilla (70B params, 1.4T tokens) outperformed Gopher (280B params, 300B tokens) despite being 4x smaller, because Gopher was severely undertrained.

**Practical implication**: if you have a fixed compute budget, split it equally between model size and training data. Do not just make the model bigger.

### Post-Chinchilla Developments

LLaMA (Meta, 2023) trained even smaller models on even more data:
- LLaMA 7B trained on 1T tokens (7x the Chinchilla ratio)
- LLaMA 13B outperformed GPT-3 175B on many benchmarks

This suggests the optimal ratio depends on the goal: if you want the best model for a fixed inference budget (not just training budget), training smaller models longer is optimal.

### Paper Reference

- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022)
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)

---

## Summary of Key Equations

For quick reference, the core equations of the Transformer:

**1. Scaled Dot-Product Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

**2. Multi-Head Attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W_O$$

where $\text{head}_i = \text{Attention}(Q W_Q^i,\, K W_K^i,\, V W_V^i)$

**3. Feed-Forward Network:**

$$\text{FFN}(x) = \max(0,\, x W_1 + b_1)\, W_2 + b_2$$

**4. Encoder Block (Post-Norm):**

$$x = \text{LayerNorm}(x + \text{MultiHead}(x, x, x))$$
$$x = \text{LayerNorm}(x + \text{FFN}(x))$$

**5. Decoder Block (Post-Norm):**

$$x = \text{LayerNorm}(x + \text{MaskedMultiHead}(x, x, x))$$
$$x = \text{LayerNorm}(x + \text{MultiHead}(x, \text{enc\_out}, \text{enc\_out}))$$
$$x = \text{LayerNorm}(x + \text{FFN}(x))$$

**6. Sinusoidal Positional Encoding:**

$$PE(\text{pos},\, 2i) = \sin\!\left(\text{pos}\, / \, 10000^{2i/d_\text{model}}\right)$$
$$PE(\text{pos},\, 2i+1) = \cos\!\left(\text{pos}\, / \, 10000^{2i/d_\text{model}}\right)$$

**7. Noam Learning Rate Schedule:**

$$\text{lr} = d_\text{model}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)$$

**8. Label Smoothing:**

$$y_\text{smooth} = (1 - \epsilon) \cdot y_\text{onehot} + \frac{\epsilon}{V}$$

---

## Recommended Reading Order

1. Vaswani et al. (2017) — read this three times. First for the big picture, second for the math, third to understand every design decision.
2. The Illustrated Transformer (Jay Alammar) — for visual intuition.
3. The Annotated Transformer (Harvard NLP) — for a line-by-line code walkthrough.
4. Devlin et al. (2019) and Radford et al. (2018, 2019) — BERT and GPT.
5. Elhage et al. (2021) — the residual stream view. This will change how you think about Transformers.
6. Dao et al. (2022) — Flash Attention. Understanding the systems perspective is essential.

This is the architecture that changed everything. Know it cold.
