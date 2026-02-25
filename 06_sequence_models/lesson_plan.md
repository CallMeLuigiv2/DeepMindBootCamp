# Module 06: Sequence Models — Lesson Plan

## Week 10: From Recurrence to Attention

### Preamble

You have spent the last several weeks learning to see — CNNs taught you spatial pattern recognition. Now you learn to *listen*: to handle data where order is meaning. A sentence is not a bag of words. A stock price is not a set of numbers. A melody is not a chord.

This week is a history lesson disguised as a technical module. You will build the architectures that dominated NLP and time series for a decade, and in doing so, you will discover their limitations firsthand. By the end of Session 3, you will understand *exactly* why Vaswani et al. wrote "Attention Is All You Need" — because you will have felt the pain that motivated it.

RNNs are not dead. They remain excellent for small-scale sequence tasks, online learning, and resource-constrained environments. But their limitations are real, and understanding them is the only honest path to understanding Transformers.

---

## Session 1: Recurrent Neural Networks

**Duration**: 3 hours (1.5h theory + 1.5h coding)

### Learning Objectives

By the end of this session, you should be able to:

1. Explain why feedforward networks cannot handle variable-length sequential data natively.
2. Draw the RNN cell in both folded and unrolled views, labeling all weight matrices.
3. Derive backpropagation through time (BPTT) for a simple RNN.
4. Explain the vanishing/exploding gradient problem using the eigenvalue argument.
5. Implement a basic RNN cell from scratch in PyTorch.
6. Describe bidirectional and stacked RNN architectures.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 20 min | Why sequences need special treatment: the failure of feedforward |
| 2 | 25 min | The RNN cell: folded view, unrolled view, weight sharing |
| 3 | 20 min | BPTT: the chain rule through time |
| 4 | 25 min | Vanishing/exploding gradients: the eigenvalue argument |
| 5 | 15 min | Gradient clipping, bidirectional RNNs, stacked RNNs |
| 6 | 45 min | Coding: implement RNN cell from scratch, train on character prediction |
| 7 | 15 min | Coding: observe gradient norms across time steps |
| 8 | 15 min | Wrap-up, key takeaways, preview of LSTM |

### Block 1: Why Sequence Order Matters (20 min)

Start with a concrete failure case. Take the sentence "The dog bit the man" vs "The man bit the dog." Same words, different meaning — order is everything. A feedforward network that takes a fixed-size input cannot naturally handle:

- Variable-length sequences (sentences have different lengths)
- Position-dependent meaning (the role of a word depends on where it appears)
- Long-range dependencies ("The cat, which sat on the mat that was in the house that Jack built, *was* orange" — the verb must agree with "cat," not "built")

You could pad and feed fixed-length windows to a feedforward net, but this is a hack. The network has no *mechanism* for remembering what came before.

**Diagram exercise**: Draw a feedforward network processing one word at a time. Where does the memory of previous words live? (Nowhere — that is the problem.)

### Block 2: The RNN Cell (25 min)

The insight: give the network a *recurrent connection* — let it pass information from one time step to the next through a hidden state.

**Folded view**: Draw the single cell with an arrow looping back to itself. This is the compact notation.

**Unrolled view**: Unfold across T time steps. The SAME weights are shared at every step. This is critical — weight sharing means the network generalizes across positions, just as CNN weight sharing generalizes across spatial locations.

**Diagram exercise**: Draw the unrolled RNN for a 5-word sentence. Label:
- x_t: input at time t
- h_t: hidden state at time t
- W_xh: input-to-hidden weights
- W_hh: hidden-to-hidden weights
- W_hy: hidden-to-output weights
- b_h, b_y: biases

Write the equations on the board:
```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
y_t = W_hy * h_t + b_y
```

Emphasize: the hidden state h_t is a *compressed summary* of everything the network has seen so far. It is a fixed-size vector trying to encode an arbitrarily long history. This will become a problem.

### Block 3: Backpropagation Through Time (20 min)

BPTT is not a new algorithm. It is standard backpropagation applied to the unrolled computation graph. But unrolling reveals the cost: the computation graph for a sequence of length T is T layers deep.

Derive the gradient of the loss with respect to W_hh for a simple RNN. Show how it involves a product of T Jacobian matrices. This product is the source of all problems.

**Key insight to convey**: BPTT is computationally expensive (O(T) in both time and memory for the backward pass) and numerically unstable (the gradient product). Truncated BPTT is the practical compromise — backpropagate only K steps instead of T, trading gradient accuracy for stability.

**Diagram exercise**: On the unrolled RNN, trace the gradient path from loss at time T back to time step 1. Count the number of matrix multiplications.

### Block 4: The Vanishing/Exploding Gradient Problem (25 min)

This is the most important theoretical concept in this module, because it is the *reason* LSTMs exist and the *reason* Transformers were invented.

The gradient of h_T with respect to h_1 involves the product:

```
dh_T/dh_1 = prod_{t=2}^{T} diag(f'(z_t)) * W_hh
```

Each factor is a matrix. The behavior of this product is governed by the eigenvalues of W_hh:

- If the largest eigenvalue |lambda_max| < 1: the product shrinks exponentially. Gradients vanish. The network cannot learn long-range dependencies.
- If the largest eigenvalue |lambda_max| > 1: the product grows exponentially. Gradients explode. Training becomes numerically unstable.

There is a razor-thin band where training works, and it is very hard to stay in that band for long sequences.

**Gradient clipping**: A band-aid for exploding gradients. If the gradient norm exceeds a threshold, scale it down. This prevents numerical instability but does NOT solve vanishing gradients.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**Diagram exercise**: Plot (sketch) the gradient magnitude as a function of the time gap. For vanilla RNN with eigenvalues < 1, show the exponential decay. For eigenvalues > 1, show the exponential growth. Mark the "useful learning region" as a narrow band.

### Block 5: Bidirectional and Stacked RNNs (15 min)

**Bidirectional RNNs**: For tasks where you have access to the full sequence (classification, not generation), run two RNNs — one forward, one backward. Concatenate their hidden states. Now each position has context from both past and future.

**Stacked RNNs**: Stack multiple RNN layers. The hidden states of layer L become the inputs of layer L+1. Deeper networks can learn more abstract representations. Typically 2-4 layers; more than that is hard to train.

**Diagram exercise**: Draw a 2-layer bidirectional RNN for a 4-step sequence. Label all hidden states.

### Block 6-7: Coding Session (60 min)

1. Implement `RNNCell` using only `torch.Tensor` operations (no `nn.RNN`).
2. Build a character-level language model on a small text corpus.
3. Train it and observe:
   - Loss curves for short sequences (length 20) vs long sequences (length 200).
   - Gradient norms at different time steps — plot them.
   - The effect of gradient clipping.

### Block 8: Key Takeaways (15 min)

1. RNNs process sequences by maintaining a hidden state — a fixed-size summary of history.
2. BPTT is just backprop on the unrolled graph, but it creates deep gradient paths.
3. Vanishing gradients make it nearly impossible for vanilla RNNs to learn dependencies beyond ~20 time steps.
4. Gradient clipping helps with exploding gradients but not vanishing ones.
5. We need an architecture that can *selectively* remember and forget. That is the LSTM.

---

## Session 2: LSTM and GRU

**Duration**: 3 hours (1.5h theory + 1.5h coding)

### Learning Objectives

By the end of this session, you should be able to:

1. Draw the LSTM cell and explain the purpose of each gate.
2. Walk through one LSTM time step with concrete numbers.
3. Explain how the cell state acts as a "gradient highway."
4. Describe the GRU as a simplified LSTM and compare the two.
5. Implement both LSTM and GRU cells from scratch.
6. Know practical initialization tricks (forget gate bias).

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 10 min | Recap: the vanishing gradient problem (what we need to fix) |
| 2 | 30 min | The LSTM cell: four gates, cell state, detailed walkthrough |
| 3 | 20 min | Numerical example: one LSTM time step with real numbers |
| 4 | 15 min | The gradient highway: why LSTM solves vanishing gradients |
| 5 | 15 min | GRU: the simplified alternative |
| 6 | 10 min | LSTM vs GRU: practical guidance, peephole connections |
| 7 | 60 min | Coding: LSTM from scratch, character-level model, compare to RNN |
| 8 | 10 min | Practical tips: forget bias init, layer norm, dropout |
| 9 | 10 min | Wrap-up and preview of Seq2Seq |

### Block 2: The LSTM Cell in Detail (30 min)

The Long Short-Term Memory cell (Hochreiter and Schmidhuber, 1997) is one of the great architectural inventions in deep learning. It solves the vanishing gradient problem by introducing a *cell state* — a separate pathway through time that allows gradients to flow unimpeded.

The LSTM has FOUR components. Teach each one by its *purpose*:

**1. Forget Gate (f_t)**: "What should I throw away from the cell state?"
- Looks at h_{t-1} and x_t, outputs a value between 0 and 1 for each element of the cell state.
- 1 = keep everything. 0 = forget everything.
- This is the gate that allows the LSTM to *clear* irrelevant memories.

**2. Input Gate (i_t)**: "What new information should I store in the cell state?"
- Also looks at h_{t-1} and x_t.
- Works with the candidate cell state (g_t, sometimes called c_tilde) to determine what to add.
- The input gate *scales* the candidate — it is the bouncer deciding what gets in.

**3. Cell State Update (C_t)**: "Update the memory."
- C_t = f_t * C_{t-1} + i_t * g_t
- This is the key equation. The cell state is updated by *forgetting* some old content and *adding* some new content.
- Crucially, this is a LINEAR operation on C_{t-1} (multiply by f_t, add something). Gradients flow through this without vanishing.

**4. Output Gate (o_t)**: "What part of the cell state should I output as the hidden state?"
- Not everything in memory is relevant to the current output.
- The output gate filters the cell state through tanh and selects what to expose.
- h_t = o_t * tanh(C_t)

**Diagram exercise**: Draw the full LSTM cell. Use boxes for sigmoid and tanh activations. Use circles for pointwise operations (multiply, add). Trace the cell state pathway from left to right — notice how it is a "highway" with only pointwise operations, no matrix multiplications.

### Block 3: Numerical Example (20 min)

This is critical. Walk through one LSTM time step with actual numbers. Use a small cell state dimension (e.g., 2) so the numbers are tractable. See the notes for the full worked example.

The goal: the apprentice should be able to compute, by hand, the output of one LSTM step given concrete inputs.

### Block 4: The Gradient Highway (15 min)

Why does the LSTM solve vanishing gradients? Look at the gradient of C_T with respect to C_1:

```
dC_T/dC_1 = prod_{t=2}^{T} f_t
```

Each f_t is a diagonal matrix with entries in (0, 1). If the forget gate is close to 1, the gradient is close to 1 — it flows through time almost unimpeded. Compare this to the vanilla RNN where the gradient involves products of W_hh and tanh derivatives.

The cell state is a *gradient highway*. The gates learned to control what travels on this highway.

### Block 5: GRU (15 min)

The Gated Recurrent Unit (Cho et al., 2014) simplifies the LSTM:
- Merges the cell state and hidden state into one.
- Uses two gates instead of three: reset gate and update gate.
- Fewer parameters, often comparable performance.

**Diagram exercise**: Draw the GRU cell. Compare side-by-side with LSTM.

### Block 6: LSTM vs GRU and Practical Considerations (10 min)

- GRU: fewer parameters, faster to train, often works well on smaller datasets.
- LSTM: more expressive, sometimes better on tasks requiring fine-grained memory control.
- In practice, try both. The difference is often small.
- Peephole connections: let gates see the cell state directly (not just h_{t-1}). Rarely used in practice.
- **Critical initialization trick**: Initialize the forget gate bias to 1.0 (or even 2.0). This means the LSTM starts by remembering everything and learns what to forget. Without this, the LSTM may start by forgetting everything, which makes early training very difficult. (Jozefowicz et al., 2015)

### Block 7: Coding Session (60 min)

1. Implement `LSTMCell` from scratch (all four gates, explicit equations).
2. Implement `GRUCell` from scratch.
3. Train character-level language models with RNN, LSTM, and GRU.
4. Compare:
   - Training loss curves (LSTM and GRU should learn faster and better on longer sequences).
   - Generated text quality.
   - Gradient norms over time (LSTM should show much more stable gradients).
5. Verify your LSTM matches `nn.LSTM` output for the same inputs and weights.

### Block 8: Practical Tips (10 min)

- Forget gate bias initialization: set to 1.0 or higher.
- Layer normalization in LSTMs can help training stability.
- Dropout between LSTM layers (not within the recurrent connection).
- For very long sequences, consider truncated BPTT with overlapping segments.

### Block 9: Key Takeaways (10 min)

1. LSTM solves vanishing gradients by providing a cell state highway with only pointwise operations.
2. Each gate has a clear purpose: forget, input, output, plus a candidate state.
3. GRU is a simpler alternative that often works comparably.
4. Forget gate bias initialization to 1.0 is important in practice.
5. Both LSTM and GRU are still limited: the hidden state is a fixed-size bottleneck, and processing is strictly sequential (no parallelization across time steps). This motivates what comes next.

---

## Session 3: Sequence-to-Sequence and the Birth of Attention

**Duration**: 3 hours (1h theory + 1.5h coding + 0.5h reflection)

### Learning Objectives

By the end of this session, you should be able to:

1. Describe the encoder-decoder architecture for sequence-to-sequence tasks.
2. Identify and explain the information bottleneck problem.
3. Implement teacher forcing and explain its trade-offs.
4. Derive Bahdanau (additive) attention from first principles.
5. Explain attention as a soft lookup mechanism.
6. Articulate why attention was the key insight leading to Transformers.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 15 min | The sequence-to-sequence problem: machine translation as case study |
| 2 | 15 min | Encoder-decoder architecture and the information bottleneck |
| 3 | 10 min | Teacher forcing and scheduled sampling |
| 4 | 10 min | Beam search decoding |
| 5 | 25 min | Attention: Bahdanau (additive) and Luong (multiplicative) |
| 6 | 15 min | Attention as soft lookup, visualizing attention, the bridge to Transformers |
| 7 | 60 min | Coding: Seq2Seq without attention, then with attention |
| 8 | 20 min | Coding: attention visualization and beam search |
| 9 | 10 min | Reflection and module wrap-up |

### Block 1: The Seq2Seq Problem (15 min)

Machine translation is the canonical example: given a sentence in French, produce its translation in English. The input and output sequences have *different lengths*. You cannot use a simple RNN that produces one output per input.

Sutskever et al. (2014) proposed the encoder-decoder architecture: one RNN reads the input and compresses it into a fixed-size vector, another RNN generates the output from that vector.

This was a breakthrough. It worked remarkably well — for short sentences.

### Block 2: Encoder-Decoder and the Information Bottleneck (15 min)

**The encoder**: An LSTM reads the source sentence word by word. Its final hidden state is the "context vector" — a fixed-size summary of the entire source sentence.

**The decoder**: Another LSTM, initialized with the context vector, generates the target sentence word by word.

**The problem**: That fixed-size context vector must encode *everything* about the source sentence. For long sentences, this is an impossible compression. Quality degrades sharply beyond ~20 words.

**Diagram exercise**: Draw the encoder-decoder architecture. Draw the context vector as a narrow bottleneck between them. Annotate with "ALL information must flow through here."

Think about translating a 50-word sentence. The encoder must compress all lexical choices, grammatical structure, word order, and semantic meaning into a single vector of perhaps 512 dimensions. Something has to give.

### Block 3: Teacher Forcing (10 min)

During training, should the decoder use its own previous predictions as input, or the ground-truth previous word?

- **Autoregressive (free-running)**: Use own predictions. Errors compound — one bad prediction derails the rest.
- **Teacher forcing**: Use ground-truth previous word. Training is stable but creates a mismatch with inference (exposure bias).
- **Scheduled sampling**: Start with teacher forcing, gradually switch to own predictions. A practical compromise.

### Block 4: Beam Search Decoding (10 min)

At inference time, greedy decoding (always pick the highest-probability word) is suboptimal because locally good choices can lead to globally bad sequences.

Beam search maintains the top K candidates at each step, expanding each one and keeping the best K overall. With K=1, it reduces to greedy search. With K=5-10, quality improves significantly.

Trade-off: beam search is K times more expensive than greedy decoding.

### Block 5: Attention — The Key Insight (25 min)

This is the climax of the module. Everything has been building to this moment.

Bahdanau et al. (2015) asked: why should the decoder be limited to a single context vector? What if, at each decoding step, the decoder could *look back* at all encoder hidden states and decide which ones are relevant?

**The mechanism**:
1. The encoder produces hidden states h_1, h_2, ..., h_T (one per source word).
2. At each decoder step t, compute an *alignment score* between the decoder state s_t and each encoder state h_j.
3. Normalize the scores with softmax to get *attention weights* alpha_{t,j}.
4. Compute the *context vector* c_t as a weighted sum of encoder hidden states.
5. Use c_t (along with s_t) to predict the next word.

**Bahdanau (additive) attention**:
```
e_{t,j} = v^T * tanh(W_s * s_{t-1} + W_h * h_j)
alpha_{t,j} = softmax_j(e_{t,j})
c_t = sum_j alpha_{t,j} * h_j
```

**Luong (multiplicative) attention**:
```
e_{t,j} = s_t^T * W * h_j    (general)
e_{t,j} = s_t^T * h_j         (dot)
```

Multiplicative attention is simpler and faster. Both work well.

**Diagram exercise**: Draw the full attention mechanism as an ASCII diagram. Show the encoder states, the decoder state, the alignment scores, the softmax, and the weighted sum producing the context vector. This diagram should be burned into your memory.

### Block 6: Attention as Soft Lookup and the Bridge to Transformers (15 min)

Reframe attention as a differentiable dictionary lookup:
- **Query**: what the decoder is looking for (s_t)
- **Keys**: what each encoder position offers (h_j)
- **Values**: the actual content at each position (also h_j, or a projection of it)

The alignment score measures how well the query matches each key. The output is a weighted sum of values.

This query-key-value framing is EXACTLY the attention mechanism in the Transformer. The leap from Bahdanau attention to self-attention is: what if the queries, keys, and values all come from the *same* sequence?

**Why attention was revolutionary**:
1. It removed the information bottleneck — the decoder can access any encoder state directly.
2. It provided interpretability — attention weights show what the model is "looking at."
3. It created direct gradient paths from decoder to encoder (no need to flow through the bottleneck).
4. It revealed that the key operation is not recurrence but *attention* — which led to the question: do we need recurrence at all?

The answer to that question is the Transformer, which we study next week.

### Block 7-8: Coding Session (80 min)

1. Implement encoder-decoder for date format conversion (e.g., "January 5, 2023" to "2023-01-05").
2. Train WITHOUT attention. Observe performance on longer inputs.
3. Add Bahdanau attention. Observe the improvement.
4. Visualize attention weights as a heatmap — see which input positions the decoder attends to at each output step.
5. Implement beam search and compare output quality to greedy decoding.

### Block 9: Module Wrap-Up (10 min)

The arc of this module:

1. **RNNs** gave us sequential processing but suffered from vanishing gradients.
2. **LSTMs/GRUs** solved vanishing gradients with gating mechanisms but still processed sequentially and compressed everything into a fixed-size state.
3. **Seq2Seq** gave us variable-length-to-variable-length mapping but hit the information bottleneck.
4. **Attention** removed the bottleneck and revealed that the critical operation is computing relevance between positions.

Next week, we ask: if attention is all we need, can we throw away recurrence entirely? The answer is the Transformer.

### Key Takeaways for the Full Module

1. Sequential data requires architectures that respect order and handle variable length.
2. Vanilla RNNs fail on long sequences due to vanishing/exploding gradients.
3. LSTMs solve this with a gated cell state that provides a gradient highway.
4. Seq2Seq maps sequences to sequences via encoder-decoder but has an information bottleneck.
5. Attention removes the bottleneck by letting the decoder look at all encoder states.
6. The query-key-value formulation of attention is the direct precursor to the Transformer.
7. RNNs remain useful for streaming, time-series, and resource-constrained settings — but the Transformer has largely replaced them for NLP.

---

## Recommended Resources

- Andrej Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" (blog post, 2015)
- Christopher Olah, "Understanding LSTM Networks" (blog post, 2015) — the canonical visual explanation
- Bahdanau, Cho, Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
- Sutskever, Vinyals, Le, "Sequence to Sequence Learning with Neural Networks" (2014)
- Hochreiter and Schmidhuber, "Long Short-Term Memory" (1997)
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
- Luong, Pham, Manning, "Effective Approaches to Attention-based Neural Machine Translation" (2015)
