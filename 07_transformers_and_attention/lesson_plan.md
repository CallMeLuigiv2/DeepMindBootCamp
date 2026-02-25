# Module 7: Transformers and Attention — Lesson Plan

## Weeks 11-12

This is the central module of the entire course. The Transformer is not merely an architecture — it is the computational substrate of modern artificial intelligence. Every major language model, every frontier vision system, every multimodal reasoning engine is built on the ideas covered in these two weeks. You must understand this architecture at the level where you could re-derive it from first principles on a whiteboard.

By the end of this module, you will be able to:
- Derive scaled dot-product attention from scratch, explaining every design choice
- Implement a complete Transformer encoder-decoder from raw PyTorch tensor operations
- Train both BERT-style and GPT-style models and understand precisely why they differ
- Apply Transformers to vision tasks and understand the architectural adaptations required
- Reason about modern efficiency techniques (Flash Attention, KV-cache, RoPE, MoE) at a systems level

---

## Session 1: Attention Is All You Need

**Duration**: 3 hours
**Date**: Week 11, Day 1

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Explain the limitations of fixed-context seq2seq models that motivated attention
2. Derive the scaled dot-product attention formula and justify the scaling factor
3. Implement single-head and multi-head attention from raw matrix operations
4. Articulate the QKV intuition using the database analogy
5. Explain why multi-head attention is superior to single-head attention with larger dimension

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 30 min | Review: seq2seq attention, the information bottleneck problem |
| 2 | 45 min | Whiteboard: self-attention derivation, QKV decomposition |
| 3 | 30 min | Whiteboard: scaled dot-product attention — the variance argument |
| 4 | 45 min | Whiteboard: multi-head attention — representation subspaces |
| 5 | 30 min | Coding: implement attention from scratch in PyTorch |

### Core Concepts

**From Seq2Seq Attention to Self-Attention**

Begin with Bahdanau attention (2014): the decoder attends to all encoder hidden states at each step. This solved the bottleneck of compressing an entire sequence into a fixed vector. But Bahdanau attention is cross-attention — the query comes from the decoder, keys and values from the encoder.

Self-attention asks: what if we let each position in a sequence attend to all other positions in the same sequence? This is the fundamental insight of Vaswani et al. (2017).

**The Database Analogy for QKV**

- **Query (Q)**: "What am I looking for?" — the search query you type into a database
- **Key (K)**: "What do I contain?" — the index entries the database uses to match your query
- **Value (V)**: "What do I give if selected?" — the actual data the database returns

Every token broadcasts a key ("here is what I represent"), issues queries ("here is what I need"), and offers values ("here is what I contribute if attended to"). The attention mechanism is a soft lookup: instead of returning a single exact match, it returns a weighted combination of all values, weighted by how well each key matches the query.

**The Scaling Factor**

When $d_k$ is large, the dot products $Q K^\top$ grow in magnitude (variance proportional to $d_k$). Large dot products push the softmax into saturated regions where gradients are near zero. Dividing by $\sqrt{d_k}$ keeps the variance of the dot products at approximately 1, regardless of $d_k$. This is not optional — it is essential for stable training.

**Multi-Head Attention**

Rather than performing a single attention function with $d_\text{model}$-dimensional keys, queries, and values, project them $h$ times with different learned linear projections to $d_k$, $d_k$, and $d_v$ dimensions respectively. Each head can learn to attend to different types of relationships: syntactic structure, semantic similarity, positional patterns, coreference. The outputs are concatenated and projected back to $d_\text{model}$.

### Paper References

- Vaswani et al., "Attention Is All You Need" (2017) — the foundational paper
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014) — original attention mechanism
- Luong et al., "Effective Approaches to Attention-based Neural Machine Translation" (2015) — dot-product vs additive attention

### Whiteboard Derivation Exercises

1. Start from the Bahdanau attention formula. Show how self-attention emerges when Q, K, V all come from the same sequence.
2. Derive the variance of dot products as a function of $d_k$ (assuming independent random entries with mean 0 and variance 1). Show that $\text{Var}(q \cdot k) = d_k$. Therefore dividing by $\sqrt{d_k}$ normalizes variance to 1.
3. Write out the full multi-head attention formula. Show the parameter dimensions for $W_Q$, $W_K$, $W_V$, $W_O$ explicitly for $d_\text{model}=512$ and $h=8$.

### Coding Tasks

1. Implement `scaled_dot_product_attention(Q, K, V, mask=None)` using only `torch.matmul`, `torch.softmax`, and basic arithmetic.
2. Implement a `MultiHeadAttention` module from scratch (no `nn.MultiheadAttention`).
3. Create a small test: 4 tokens, $d_\text{model}$=8, 2 heads. Print attention weights. Verify shapes.

---

## Session 2: The Transformer Architecture

**Duration**: 3 hours
**Date**: Week 11, Day 2

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Implement sinusoidal positional encoding and explain the Fourier intuition
2. Build a complete Transformer encoder block with correct tensor shapes at every step
3. Build a complete Transformer decoder block including masked self-attention and cross-attention
4. Explain why Layer Normalization is used instead of Batch Normalization
5. Describe the residual stream interpretation of Transformer computation

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 40 min | Whiteboard: positional encoding — sinusoidal, learned, RoPE |
| 2 | 40 min | Whiteboard: encoder block — walk through shapes at every sublayer |
| 3 | 40 min | Whiteboard: decoder block — masked self-attention, cross-attention |
| 4 | 30 min | Discussion: LayerNorm vs BatchNorm, Pre-Norm vs Post-Norm |
| 5 | 30 min | Discussion: the residual stream view (Anthropic/Elhage interpretation) |

### Core Concepts

**Positional Encoding**

Transformers have no inherent notion of position — unlike RNNs, they process all tokens in parallel. We must inject position information. The original Transformer uses sinusoidal encodings:

$$
\begin{aligned}
PE(\text{pos},\, 2i) &= \sin\!\left(\text{pos}\, / \, 10000^{2i/d_\text{model}}\right) \\
PE(\text{pos},\, 2i+1) &= \cos\!\left(\text{pos}\, / \, 10000^{2i/d_\text{model}}\right)
\end{aligned}
$$

The Fourier intuition: each dimension of the encoding oscillates at a different frequency, from high-frequency (changing rapidly across positions) to low-frequency (changing slowly). This creates a unique "fingerprint" for each position. Moreover, for any fixed offset $k$, $PE(\text{pos}+k)$ can be represented as a linear function of $PE(\text{pos})$ -- the encoding captures relative positions.

Contrast with learned positional embeddings (BERT, GPT) which are simply trained vectors, and Rotary Position Embeddings (RoPE) which encode relative position through rotation matrices in the complex plane.

**The Encoder Block**

Input: (batch, seq_len, $d_\text{model}$)
1. Self-attention: (batch, seq_len, $d_\text{model}$) -> (batch, seq_len, $d_\text{model}$)
2. Residual connection + Layer Normalization
3. Position-wise Feed-Forward Network: $d_\text{model} \to d_{ff} \to d_\text{model}$ (typically $d_{ff} = 4 \cdot d_\text{model}$)
4. Residual connection + Layer Normalization
Output: (batch, seq_len, $d_\text{model}$)

The FFN applies the same two-layer network independently to each position. It is where the Transformer stores factual knowledge (the "key-value memory" interpretation).

**The Decoder Block**

Adds two complications over the encoder:
1. Masked self-attention: each position can only attend to earlier positions (causal mask)
2. Cross-attention: queries come from the decoder, keys and values come from the encoder output

**LayerNorm vs BatchNorm**

BatchNorm normalizes across the batch dimension — problematic for variable-length sequences and small batch sizes common in NLP. LayerNorm normalizes across the feature dimension for each individual example, making it independent of batch size and sequence length.

**The Residual Stream View**

From Elhage et al. (Anthropic, 2021): instead of thinking of a Transformer as a pipeline of layers that sequentially transform representations, think of it as a residual stream. Each layer reads from the stream, computes something, and adds its output back to the stream. The final prediction is a function of the accumulated stream. This view is powerful for mechanistic interpretability: attention heads and MLP layers are independent modules that communicate through the shared residual stream.

### Paper References

- Vaswani et al., "Attention Is All You Need" (2017) — Sections 3.2 (positional encoding), 3.1 (encoder-decoder)
- Ba et al., "Layer Normalization" (2016) — the normalization layer used in Transformers
- Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021) — the residual stream view
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) — RoPE

### Whiteboard Derivation Exercises

1. Compute the sinusoidal positional encoding matrix for 6 positions and $d_\text{model}=8$. Plot the values. Show that the encoding at position $p+k$ is a linear function of the encoding at position $p$.
2. Walk through a single encoder block for batch_size=2, seq_len=5, $d_\text{model}=512$, $d_{ff}=2048$, $h=8$. Write the tensor shape after every operation.
3. Draw the full encoder-decoder architecture. Trace a single forward pass through the entire model, showing where each attention mechanism draws its Q, K, V.

### Coding Tasks

1. Implement `PositionalEncoding` using the sinusoidal formula. Visualize the encoding as a heatmap.
2. Implement a full `TransformerEncoderBlock` from scratch.
3. Implement a full `TransformerDecoderBlock` from scratch (with both masked self-attention and cross-attention).

---

## Session 3: Training Transformers

**Duration**: 3 hours
**Date**: Week 11, Day 3

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Implement masked language modeling (BERT-style) and autoregressive language modeling (GPT-style)
2. Implement causal masking correctly, understanding why -inf before softmax works
3. Explain and implement learning rate warmup and why it is critical for Transformers
4. Describe label smoothing and its effect on model calibration
5. Articulate the "key-value memories" interpretation of FFN layers

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 40 min | Whiteboard: MLM vs autoregressive LM — objectives, masking strategies |
| 2 | 30 min | Whiteboard: causal masking implementation — the triangular mask |
| 3 | 40 min | Whiteboard: training recipes — warmup, label smoothing, dropout patterns |
| 4 | 30 min | Discussion: what do FFN layers learn? The key-value memory interpretation |
| 5 | 40 min | Coding: implement training loops for both MLM and autoregressive LM |

### Core Concepts

**Masked Language Modeling (BERT-style)**

Randomly mask 15% of tokens. Of those, 80% are replaced with [MASK], 10% with a random token, 10% kept unchanged. The model predicts the original token at each masked position. This is bidirectional — the model sees context from both left and right. This design choice means BERT cannot naturally generate text (it was not trained to do so).

**Autoregressive Language Modeling (GPT-style)**

Predict the next token given all previous tokens. $P(x_1, \ldots, x_n) = \prod P(x_t \mid x_1, \ldots, x_{t-1})$. The model only sees leftward context (enforced by the causal mask). This is naturally suited for generation: sample one token at a time, feed it back.

**Causal Masking**

Create an upper-triangular matrix of $-\infty$ values. When added to the attention scores (before softmax), positions that should not be attended to get attention weight $\exp(-\infty) = 0$. Implementation: `mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)`.

**Learning Rate Warmup**

Transformers are notoriously sensitive to the learning rate in early training. The Adam optimizer's second-moment estimates are inaccurate when few steps have been seen, leading to excessively large updates. Warmup linearly increases the learning rate from near-zero over the first several thousand steps, allowing the optimizer to build accurate statistics. The original Transformer uses: $\text{lr} = d_\text{model}^{-0.5} \cdot \min(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup\_steps}^{-1.5})$.

**Label Smoothing**

Instead of training with hard targets (one-hot), distribute a small amount of probability mass ($\epsilon$, typically 0.1) uniformly across all classes. This prevents the model from becoming overconfident and improves calibration. Cross-entropy with smoothed labels: $(1 - \epsilon) \cdot \log p_\text{correct} + \frac{\epsilon}{V} \cdot \sum \log p_i$.

**FFN as Key-Value Memories**

Geva et al. (2021) showed that FFN layers can be interpreted as key-value memories. The first linear layer's rows act as keys that match input patterns. The second linear layer's columns act as values that are retrieved. The ReLU (or GELU) activation acts as a sparse selection mechanism. This means FFN layers are where Transformers store factual knowledge.

### Paper References

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019) — MLM objective
- Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018) — GPT
- Vaswani et al., "Attention Is All You Need" (2017) — warmup schedule, label smoothing
- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (2021)

### Whiteboard Derivation Exercises

1. Write out the causal mask matrix for seq_len=6. Show how it interacts with the attention score matrix. Trace through the softmax to show that masked positions get zero weight.
2. Derive the warmup learning rate schedule from the formula. Plot lr vs step for $d_\text{model}=512$ and warmup_steps=4000. Identify the peak learning rate and when it occurs.
3. Compute the cross-entropy loss with and without label smoothing for a toy example (V=4, correct class=2, predicted distribution). Show how label smoothing changes the gradient signal.

### Coding Tasks

1. Implement a causal mask generator. Test it by running attention with and without the mask on a 5-token sequence — verify that token 3 cannot attend to tokens 4 and 5.
2. Implement the Noam learning rate scheduler ($\text{lr} = d_\text{model}^{-0.5} \cdot \min(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup\_steps}^{-1.5})$). Plot the schedule.
3. Implement a simple autoregressive training loop: given a character sequence, predict the next character at each position. Train on a small text and generate samples.

---

## Session 4: Transformer Variants for NLP

**Duration**: 3 hours
**Date**: Week 12, Day 1

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Compare and contrast BERT, GPT, and T5 architectures and training objectives
2. Explain in-context learning and why it emerges in large autoregressive models
3. Describe the scaling laws (Kaplan, Chinchilla) and their implications for training strategy
4. Implement BPE tokenization and explain why subword tokenization is necessary
5. Reason about the tradeoffs between encoder-only, decoder-only, and encoder-decoder designs

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 40 min | Whiteboard: BERT — architecture, MLM+NSP, fine-tuning paradigm |
| 2 | 40 min | Whiteboard: GPT — autoregressive, scaling, in-context learning |
| 3 | 30 min | Whiteboard: T5 — text-to-text framework, span corruption |
| 4 | 30 min | Discussion: scaling laws — Kaplan vs Chinchilla, compute-optimal training |
| 5 | 40 min | Whiteboard + Coding: tokenization — BPE, WordPiece, SentencePiece |

### Core Concepts

**BERT (Bidirectional Encoder Representations from Transformers)**

Encoder-only. Pre-trained with MLM + Next Sentence Prediction (NSP). Fine-tuned by adding a task-specific head on top of the [CLS] token representation (for classification) or on top of individual token representations (for token-level tasks like NER). Key insight: bidirectional context is essential for understanding tasks. Limitation: cannot generate text naturally.

**GPT (Generative Pre-trained Transformer)**

Decoder-only. Pre-trained with autoregressive language modeling. GPT-2 showed zero-shot task performance. GPT-3 demonstrated in-context learning — the model can perform tasks by conditioning on a few examples in the prompt, without any gradient updates. This was a paradigm shift: from "pre-train then fine-tune" to "pre-train then prompt."

**T5 (Text-to-Text Transfer Transformer)**

Encoder-decoder. Reframes every NLP task as text-to-text: classification becomes "classify: [input]" -> "positive", translation becomes "translate English to French: [input]" -> "[output]". Trained with span corruption (mask random spans, predict them). Unified framework for all tasks.

**Scaling Laws**

Kaplan et al. (2020): model performance (loss) follows power laws as a function of model size, dataset size, and compute. Larger models are more sample-efficient. Chinchilla (Hoffmann et al., 2022) revised this: the optimal strategy is to scale model size and training data equally. A 70B model trained on 1.4T tokens outperforms a 280B model trained on 300B tokens. Implication: many models were undertrained.

**Tokenization**

Word-level tokenization fails on rare words and morphological variations. Character-level is too fine-grained. Subword methods find the middle ground:
- **BPE (Byte Pair Encoding)**: iteratively merge the most frequent pair of tokens. Used by GPT.
- **WordPiece**: similar to BPE but selects merges that maximize likelihood of the training data. Used by BERT.
- **SentencePiece**: language-agnostic, treats the input as raw text (no pre-tokenization). Used by T5.

### Paper References

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019) — GPT-2
- Brown et al., "Language Models are Few-Shot Learners" (2020) — GPT-3
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020) — T5
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) — Chinchilla
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016) — BPE

### Whiteboard Derivation Exercises

1. Draw the architectures of BERT, GPT, and T5 side by side. For each, show: which attention masks are used, where the prediction head sits, and what the training objective is.
2. Walk through BPE tokenization on the sentence "the cat sat on the mat" starting from character-level. Perform 5 merge steps manually.
3. Given the Chinchilla scaling law, compute the optimal model size for a fixed compute budget of $10^{23}$ FLOPs. Compare with the Kaplan prediction.

### Coding Tasks

1. Load a pretrained BERT model using HuggingFace. Extract the [CLS] representation for a sentence. Show the attention pattern from one head.
2. Load GPT-2 and generate text with different temperature and top-k settings. Observe how generation quality changes.
3. Implement BPE tokenization from scratch on a small corpus. Show the vocabulary at each merge step.

---

## Session 5: Vision Transformers

**Duration**: 3 hours
**Date**: Week 12, Day 2

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Explain how ViT converts images into sequences of patch tokens
2. Implement a basic Vision Transformer from scratch
3. Describe the inductive bias differences between CNNs and ViTs
4. Explain how DeiT uses distillation to train ViTs with less data
5. Describe the Swin Transformer's hierarchical design and shifted window mechanism

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 40 min | Whiteboard: ViT — patches as tokens, class token, position embeddings |
| 2 | 30 min | Discussion: inductive bias — why ViTs need more data than CNNs |
| 3 | 40 min | Whiteboard: DeiT — knowledge distillation for efficient ViT training |
| 4 | 30 min | Whiteboard: Swin Transformer — hierarchical features, shifted windows |
| 5 | 40 min | Coding: implement a minimal ViT and train on CIFAR-10 |

### Core Concepts

**Vision Transformer (ViT)**

Dosovitskiy et al. (2020): split an image into fixed-size patches (e.g., $16 \times 16$), flatten each patch, linearly project to $d_\text{model}$ dimensions. These patch embeddings become the "tokens" of the Transformer. Prepend a learnable [CLS] token. Add learnable position embeddings. Feed through a standard Transformer encoder. Use the [CLS] token output for classification.

Key result: ViT trained on ImageNet alone underperforms ResNets. But ViT trained on JFT-300M (large-scale data) significantly outperforms them. ViTs lack the inductive biases of CNNs (translation invariance, locality) and must learn these from data.

**DeiT (Data-efficient Image Transformers)**

Touvron et al. (2021): train ViTs effectively on ImageNet alone using:
- Strong data augmentation (RandAugment, Mixup, CutMix, random erasing)
- Regularization (stochastic depth, label smoothing)
- Knowledge distillation from a CNN teacher (add a distillation token alongside the class token)

The distillation token learns to match the CNN teacher's output, effectively injecting CNN-like inductive biases into the ViT.

**Swin Transformer**

Liu et al. (2021): addresses two limitations of ViT:
1. Quadratic complexity of global self-attention: Swin computes attention within local windows (e.g., $7 \times 7$ patches), reducing complexity to linear in image size.
2. Single-scale features: Swin uses hierarchical patch merging (like pooling in CNNs) to produce multi-scale feature maps, making it suitable for dense prediction tasks (detection, segmentation).

Shifted windows: in alternating layers, windows are shifted by half the window size, allowing cross-window connections.

**Hybrid CNN-Transformer Architectures**

Use CNN for early stages (exploit locality bias) and Transformer for later stages (exploit global context). Examples: CoAtNet, LeViT. Alternatively, use CNN features as tokens for the Transformer.

### Paper References

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020) — ViT
- Touvron et al., "Training data-efficient image transformers & distillation through attention" (2021) — DeiT
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (2021)
- Dai et al., "CoAtNet: Marrying Convolution and Attention for All Data Sizes" (2021)

### Whiteboard Derivation Exercises

1. For an image of size $224 \times 224$ with patch size $16 \times 16$: compute the number of patches, the sequence length, and the total number of parameters in the patch embedding layer for $d_\text{model}=768$.
2. Compare the computational complexity of global self-attention (ViT) vs windowed self-attention (Swin) for an image of size $224 \times 224$. Express in terms of patch count and window size.
3. Draw the shifted window mechanism for a $4 \times 4$ grid of windows across two consecutive Swin Transformer layers. Show which patches can attend to which.

### Coding Tasks

1. Implement `PatchEmbedding`: takes a batch of images (B, C, H, W), splits into patches, projects to $d_\text{model}$.
2. Implement a minimal ViT: patch embedding + positional embedding + Transformer encoder + classification head.
3. Train the ViT on CIFAR-10 (small images, so use small patches like $4 \times 4$). Compare with a CNN baseline.

---

## Session 6: Modern Transformer Engineering

**Duration**: 3 hours
**Date**: Week 12, Day 3

### Learning Objectives

By the end of this session, the apprentice will be able to:
1. Explain why standard attention is memory-bound and how Flash Attention addresses this
2. Implement KV-cache for efficient autoregressive inference
3. Derive Rotary Positional Embeddings (RoPE) and explain why they are preferred in modern LLMs
4. Describe Grouped Query Attention and its efficiency tradeoffs
5. Explain Mixture of Experts (MoE) and why it decouples model capacity from compute cost

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 40 min | Whiteboard: Flash Attention — IO complexity, tiling, recomputation |
| 2 | 30 min | Whiteboard: KV-cache — the inference bottleneck, memory management |
| 3 | 40 min | Whiteboard: RoPE — rotation matrices, relative position encoding |
| 4 | 30 min | Whiteboard: Grouped Query Attention (GQA) and Multi-Query Attention (MQA) |
| 5 | 40 min | Whiteboard + Discussion: Mixture of Experts — routing, load balancing |

### Core Concepts

**Flash Attention**

Dao et al. (2022): standard attention materializes the full $N \times N$ attention matrix in HBM (GPU main memory), causing $O(N^2)$ memory reads/writes that dominate runtime. Flash Attention never materializes the full matrix. Instead, it tiles the computation into blocks that fit in SRAM (fast on-chip memory), computes attention within each block, and accumulates results using the online softmax trick. Result: 2-4x faster, $O(N)$ memory. This is not an approximation -- it computes exact attention.

**KV-Cache for Inference**

During autoregressive generation, each new token attends to all previous tokens. Without caching, we recompute the K and V projections for all previous tokens at every step -- $O(n^2)$ total work for $n$ tokens. The KV-cache stores computed K and V tensors, so each step only computes K, V for the new token and retrieves cached values for previous tokens. This reduces inference from $O(n^2)$ to $O(n)$ in the attention computation per step. The tradeoff: memory grows linearly with sequence length and batch size.

**Rotary Positional Embeddings (RoPE)**

Su et al. (2021): encode position by rotating the query and key vectors in 2D subspaces. For dimensions $(2i, 2i+1)$, apply a rotation by angle $\theta_i \cdot \text{position}$, where $\theta_i = 10000^{-2i/d}$. The inner product of rotated queries and keys depends only on relative position, not absolute position. RoPE has several advantages: it naturally decays with distance, generalizes better to unseen sequence lengths, and requires no additional parameters. Used in LLaMA, GPT-NeoX, and most modern LLMs.

**Grouped Query Attention (GQA)**

Ainslie et al. (2023): in standard multi-head attention, each head has its own Q, K, V projections. In Multi-Query Attention (MQA), all heads share a single K, V projection — dramatically reducing KV-cache memory but sometimes hurting quality. GQA is the middle ground: group heads and share K, V within each group. LLaMA 2 70B uses GQA with 8 KV heads for 64 query heads.

**Mixture of Experts (MoE)**

Shazeer et al. (2017), Fedus et al. (2021): replace the dense FFN with a set of expert FFNs. A router network selects the top-k experts for each token. Only the selected experts are activated, so compute cost scales with k, not the total number of experts. This allows scaling model capacity (total parameters) without proportionally increasing compute. Challenges: load balancing across experts, training instability, communication overhead in distributed settings. Used in Switch Transformer, Mixtral, and GPT-4 (rumored).

### Paper References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021)

### Whiteboard Derivation Exercises

1. Compute the memory footprint of the KV-cache for a model with $d_\text{model}=4096$, 32 layers, and a sequence of 2048 tokens. Express in GB. Then compute for GQA with 8 KV heads instead of 32.
2. Derive the RoPE rotation matrix for a single 2D subspace. Show that the inner product of rotated $q$ and $k$ depends only on the relative position ($m - n$).
3. For a Mixture of Experts layer with 8 experts and top-2 routing: compute the effective FLOPs compared to a dense FFN. Discuss the load balancing problem — what happens if the router always selects the same expert?

### Coding Tasks

1. Implement KV-cache for a decoder-only Transformer. Benchmark generation speed with and without cache for a 100-token sequence.
2. Implement RoPE from scratch. Apply it to a multi-head attention module and verify that attention scores depend on relative position.
3. Implement a simple top-k expert routing mechanism for a Mixture of Experts FFN layer.

---

## Module Assessment

### Formative Assessment (Throughout)

- In-session whiteboard derivations (all 6 sessions)
- Daily coding exercises submitted as Jupyter notebooks
- Verbal explanations: can the apprentice explain scaled dot-product attention, causal masking, and the residual stream view to a non-expert?

### Summative Assessment

1. **Assignment 1**: Attention from Scratch (Session 1-2 material)
2. **Assignment 2**: Transformer from Scratch (Session 2-3 material) — the flagship assignment
3. **Assignment 3**: Fine-tuning Pretrained Transformers (Session 4-6 material)

### Mastery Criteria

The apprentice has mastered this module when they can:
- Derive the Transformer architecture from first principles on a whiteboard with no references
- Implement a working Transformer from scratch in under 2 hours
- Train both encoder-only and decoder-only models and articulate the design tradeoffs
- Load a pretrained model, fine-tune it, and analyze its attention patterns
- Explain at least 3 modern efficiency techniques and why they matter at scale
- Read a new Transformer paper and understand it within 30 minutes

This is the standard. Nothing less.
