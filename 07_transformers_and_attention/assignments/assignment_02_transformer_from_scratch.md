# Assignment 2: Build a Transformer from Scratch

## Overview

This is the flagship assignment of the entire course. You will build a complete Transformer — every component, from positional encoding to the final softmax — using only basic PyTorch building blocks (`nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, tensor operations). No `nn.TransformerEncoder`, no `nn.TransformerDecoder`, no `nn.MultiheadAttention`.

When you finish this assignment, you will have written the most important architecture in modern AI with your own hands. There is no substitute for this experience.

**Estimated time**: 15-25 hours
**Prerequisites**: Assignment 1 (attention from scratch), Module 7 Sessions 1-3

---

## Part 1: Positional Encoding

### Task

Implement sinusoidal positional encoding as described in "Attention Is All You Need."

### Requirements

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Sinusoidal positional encoding.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pass

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) — token embeddings
        Returns:
            (batch, seq_len, d_model) — embeddings + positional encoding
        """
        pass
```

### Verification

1. **Shape test**: for d_model=512, max_len=100, verify the encoding matrix has shape (1, 100, 512) or equivalent.

2. **Uniqueness test**: compute pairwise cosine similarity between position encodings for positions 0-99. The diagonal should have similarity 1.0 (each position is identical to itself). Off-diagonal similarities should be less than 1.0.

3. **Visualization**: plot the positional encoding matrix as a heatmap for d_model=64 and max_len=50. You should see the characteristic pattern of alternating sine and cosine waves at different frequencies.

4. **Relative position test**: compute the dot product between PE(pos) and PE(pos+k) for various values of k. Show that the dot product depends primarily on k (the offset), not on the absolute position pos.

### Deliverable

- `PositionalEncoding` class
- Heatmap visualization
- Relative position analysis

---

## Part 2: Transformer Encoder Block

### Task

Implement a single Transformer encoder block.

### Requirements

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Single encoder block:
        1. Multi-Head Self-Attention + Residual + LayerNorm
        2. Feed-Forward Network + Residual + LayerNorm
        """
        pass

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            (batch, seq_len, d_model)
        """
        pass
```

You MUST use the `MultiHeadAttention` class you built in Assignment 1. If you need to modify it (e.g., to support masking), do so.

The feed-forward network must be:

$$\text{FFN}(x) = \text{Linear}_2(\text{Dropout}(\text{ReLU}(\text{Linear}_1(x))))$$

where $\text{Linear}_1: d_\text{model} \to d_{ff}$ and $\text{Linear}_2: d_{ff} \to d_\text{model}$.

### Verification

1. **Shape preservation**: input (4, 20, 512) should produce output (4, 20, 512).

2. **Residual test**: set all attention and FFN weights to zero (or very small values). The output should be close to the input (because of the residual connections). Specifically:
   ```python
   # With zeroed sublayer weights, output ~ LayerNorm(x + 0) = LayerNorm(x)
   # This verifies the residual path is connected correctly.
   ```

3. **Gradient flow test**: compute a scalar loss from the output, call `.backward()`, and verify that gradients are non-zero for all parameters.

4. **Parameter count**: for $d_\text{model}=512$, $d_{ff}=2048$, num_heads=8, verify the total parameter count matches the theoretical value:
   - Multi-head attention: $4 \times 512^2 = 1{,}048{,}576$
   - FFN: $512 \times 2048 + 2048 + 2048 \times 512 + 512 = 2{,}099{,}712$
   - LayerNorm (x2): $2 \times (512 + 512) = 2{,}048$
   - Total per block: $\approx 3{,}150{,}336$

### Deliverable

- `TransformerEncoderBlock` class with all verification tests passing

---

## Part 3: Transformer Decoder Block

### Task

Implement a single Transformer decoder block with three sub-layers.

### Requirements

```python
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Single decoder block:
        1. Masked Multi-Head Self-Attention + Residual + LayerNorm
        2. Multi-Head Cross-Attention + Residual + LayerNorm
        3. Feed-Forward Network + Residual + LayerNorm
        """
        pass

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input, (batch, tgt_len, d_model)
            encoder_output: Encoder output, (batch, src_len, d_model)
            src_mask: Mask for cross-attention (e.g., padding mask)
            tgt_mask: Causal mask for self-attention
        Returns:
            (batch, tgt_len, d_model)
        """
        pass
```

### Verification

1. **Shape test**: decoder input (4, 15, 512), encoder output (4, 20, 512). Output should be (4, 15, 512). The output length matches the decoder input length, not the encoder output length.

2. **Causal masking test**: run the decoder block with a causal mask. Verify that changing token at position 5 of the decoder input does NOT change the output at positions 0-4 (future tokens cannot influence past outputs).

   ```python
   # Test causal masking correctness:
   x1 = torch.randn(1, 10, 512)
   x2 = x1.clone()
   x2[0, 5, :] = torch.randn(512)  # modify token 5

   # With causal mask, outputs at positions 0-4 should be identical
   out1 = decoder_block(x1, enc_out, tgt_mask=causal_mask)
   out2 = decoder_block(x2, enc_out, tgt_mask=causal_mask)
   assert torch.allclose(out1[0, :5, :], out2[0, :5, :], atol=1e-6)
   ```

3. **Cross-attention test**: verify that the decoder block actually uses the encoder output. Change the encoder output and confirm the decoder output changes.

### Deliverable

- `TransformerDecoderBlock` class with all verification tests passing

---

## Part 4: Complete Encoder-Decoder Transformer

### Task

Assemble the full encoder-decoder Transformer.

### Requirements

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, max_len=5000,
                 dropout=0.1):
        """
        Complete Transformer:
        - Source and target embeddings (scaled by sqrt(d_model))
        - Positional encoding
        - N encoder blocks
        - N decoder blocks
        - Output linear projection to vocabulary
        """
        pass

    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        pass

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence given encoder output."""
        pass

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Full forward pass. Returns logits (batch, tgt_len, tgt_vocab_size)."""
        pass

    @staticmethod
    def generate_causal_mask(seq_len):
        """Generate upper-triangular causal mask."""
        pass
```

### Training Task: Sorting Numbers

Train your Transformer on a simple task to verify it works end-to-end.

**Task**: given a sequence of random integers, produce the sorted sequence.

```
Input:  [7, 3, 9, 1, 5]
Output: [1, 3, 5, 7, 9]
```

**Setup**:
- Vocabulary: integers 0-99 plus special tokens (PAD=0, SOS=101, EOS=102)
- Sequence length: 5-10 numbers
- Training data: generate on the fly (infinite data)
- Model: small Transformer (d_model=128, num_heads=4, num_layers=2, d_ff=256)
- Training: Adam optimizer with warmup, cross-entropy loss

**Expected results**:
- After ~5000 steps: the model should sort short sequences (length 5) with >80% accuracy
- After ~20000 steps: the model should sort most sequences correctly

### Deliverable

- Complete `Transformer` class
- Training code for the sorting task
- Training loss curve
- Accuracy over training (% of perfectly sorted sequences)
- 10 example predictions showing input, target, and model output

---

## Part 5: GPT-Style Decoder-Only Transformer

### Task

Implement a decoder-only Transformer (GPT-style) for autoregressive language modeling.

### Requirements

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4,
                 num_layers=4, d_ff=512, max_len=512, dropout=0.1):
        """
        Decoder-only Transformer:
        - Token embedding + positional encoding
        - N decoder blocks (self-attention only, with causal mask — no cross-attention)
        - Output projection to vocabulary
        """
        pass

    def forward(self, x):
        """
        Args:
            x: Token IDs, (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        pass

    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        """
        Autoregressive text generation.

        Args:
            start_tokens: (batch, prompt_len) — initial token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1 = more greedy, >1 = more random)

        Returns:
            generated: (batch, prompt_len + max_new_tokens)
        """
        pass
```

Note: the GPT decoder block is simpler than the encoder-decoder Transformer decoder block. It has only:
1. Masked self-attention + Residual + LayerNorm
2. FFN + Residual + LayerNorm

There is NO cross-attention because there is no encoder.

### Training Task: Character-Level Language Model

Train your GPT on a small text corpus at the character level.

**Recommended corpora** (pick one):
- Shakespeare's collected works (~1MB)
- A collection of Wikipedia articles (~5MB)
- The complete text of "Alice in Wonderland" (~150KB)

**Setup**:
- Vocabulary: all unique characters in the corpus (typically 50-100 characters)
- Context length: 128 characters
- Model: d_model=128, num_heads=4, num_layers=4, d_ff=256
- Training: Adam with learning rate warmup

**Training procedure**:
1. Create a character-to-index mapping
2. Chop the text into overlapping windows of length context_len + 1
3. Input: characters 0 to context_len-1; Target: characters 1 to context_len
4. Train with cross-entropy loss at each position (with causal masking)

**Expected results**:
- After ~1000 steps: output should be character-level gibberish but with some recognizable patterns
- After ~10000 steps: output should show word-like patterns and occasional real words
- After ~50000 steps: output should show coherent phrases and mimic the style of the training text

### Deliverable

- Complete `GPT` class
- Training code with loss curve
- Generated text samples at steps 1000, 5000, 10000, 50000 (or wherever you stop)
- Analysis: how does generation quality change with temperature? Show samples at temperature 0.5, 1.0, and 1.5

---

## Part 6: Analysis and Reflection

### Written Analysis (1-2 pages)

Answer the following questions based on your implementation experience:

1. **Architecture understanding**: explain in your own words why the Transformer uses residual connections. What happens if you remove them? (Try it and report the results.)

2. **Training dynamics**: what happened when you trained without learning rate warmup? (Try it.) How many warmup steps were needed for stable training?

3. **Attention patterns**: extract and visualize the attention weights from your trained sorting Transformer. Do different heads attend to different things? Does the pattern change across layers?

4. **Scaling intuition**: how does your sorting Transformer's accuracy change as you vary:
   - Number of layers (1, 2, 4, 6)
   - Number of heads (1, 2, 4, 8)
   - Model dimension (64, 128, 256, 512)
   Report at least 4 configurations and their sorting accuracy.

5. **Encoder-decoder vs decoder-only**: you implemented both architectures. In what situations would you use one vs the other? Could you train a decoder-only model to sort numbers? Try it and compare.

---

## Deliverables Summary

| Part | Deliverable | Weight |
|------|-------------|--------|
| 1 | Positional encoding + visualization | 10% |
| 2 | Encoder block + verification tests | 15% |
| 3 | Decoder block + causal masking verification | 15% |
| 4 | Full Transformer + sorting task training results | 25% |
| 5 | GPT + character-level LM training + text generation | 25% |
| 6 | Written analysis | 10% |

Submit as a Jupyter notebook (or multiple notebooks) with clear sections, inline outputs, training curves, and written analysis.

---

## Grading Criteria

**Passing (70%+)**:
- Parts 1-3 are correctly implemented and pass verification tests
- Part 4: the sorting Transformer trains (loss decreases) and achieves >50% accuracy on length-5 sequences
- Part 5: the GPT trains (loss decreases) and generates recognizable character patterns

**Distinction (85%+)**:
- All of the above, plus:
- Part 4: sorting accuracy >80% on length-5 sequences
- Part 5: generated text contains recognizable words and stylistic patterns from the training corpus
- Part 6: written analysis demonstrates deep understanding

**Exceptional (95%+)**:
- All of the above, plus:
- Part 4: sorting works on variable-length sequences up to length 10
- Part 5: generated text is coherent at the phrase level
- Part 6: includes ablation studies (removing components, varying hyperparameters) with thoughtful analysis
- Code is clean, well-documented, and could serve as a teaching reference

---

## Stretch Goals

1. **Beam search**: implement beam search decoding for the sorting Transformer. Compare greedy decoding vs beam search (beam width 3, 5, 10). Does beam search improve sorting accuracy?

2. **Different positional encodings**: implement learned positional embeddings alongside sinusoidal. Train both on the sorting task. Is there a difference? Now implement RoPE and compare all three.

3. **Pre-Norm vs Post-Norm**: the original Transformer uses Post-Norm ($\text{LayerNorm}(x + \text{SubLayer}(x))$). Implement Pre-Norm ($x + \text{SubLayer}(\text{LayerNorm}(x))$). Train both on the sorting task. Compare training stability (can you train without warmup using Pre-Norm?).

4. **Weight tying**: in the original Transformer paper, the source embedding, target embedding, and output projection share weights. Implement weight tying and compare parameter count and performance.

5. **Arithmetic Transformer**: instead of sorting, train your Transformer on addition (e.g., "123+456" -> "579"). This is surprisingly hard — can your model generalize to numbers longer than those seen in training?

6. **Gradient analysis**: at various points during training, compute the gradient norm for each layer. Plot gradient norms across layers. Does the Transformer exhibit vanishing or exploding gradients? How do residual connections affect this?

---

## Common Pitfalls

1. **Embedding scaling**: the original Transformer multiplies embeddings by $\sqrt{d_\text{model}}$. This is easily forgotten and affects training stability.

2. **Teacher forcing vs autoregressive**: during training, the decoder receives the ground-truth target sequence (shifted right). During inference, it receives its own previous predictions. Make sure your training loop uses teacher forcing correctly.

3. **Mask dimensions**: the causal mask should be applied to the attention scores, which have shape (batch, num_heads, seq_len, seq_len). Your mask must broadcast correctly to this shape.

4. **Start-of-sequence token**: the decoder input should be shifted right and prepended with a start-of-sequence (SOS) token. The decoder's input at position 0 is SOS, and it should predict the first target token. If you forget this shift, the model can trivially copy the input to the output.

5. **Evaluation vs training mode**: remember to call `model.eval()` and `torch.no_grad()` during inference. Dropout during generation will produce inconsistent results.

6. **Memory for long sequences**: if you train on long sequences and run out of GPU memory, reduce batch size or sequence length before reducing model size. Gradient accumulation can help maintain effective batch size.

This assignment is hard. It is supposed to be. When you finish, you will understand the Transformer in a way that reading papers alone cannot provide.
