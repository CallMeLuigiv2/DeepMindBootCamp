# Assignment 02: Sequence-to-Sequence with Attention

## Overview

In this assignment, you will build a complete sequence-to-sequence model with attention from the ground up. You will first experience the limitations of the vanilla encoder-decoder architecture — the information bottleneck — and then add the Bahdanau attention mechanism and observe how it transforms the model's capabilities. Finally, you will visualize the attention weights and implement beam search decoding.

This assignment is the bridge between RNNs and Transformers. By the time you finish it, you will understand attention not as an abstract concept but as a concrete mechanism you have built, trained, and visualized. You will know exactly what problem it solves and why it was the key insight that led to the Transformer.

**Estimated time**: 10-15 hours

**Prerequisites**: Assignment 01 (RNN/LSTM from scratch), understanding of the Seq2Seq architecture and attention mechanism from this module's notes.

---

## The Task: Date Format Conversion

Rather than machine translation (which requires large parallel corpora and significant compute), you will work on a simpler but structurally identical task: converting human-readable dates to machine-readable format.

```
Input:                          Output:
"January 5, 2023"        -->   "2023-01-05"
"5th of March, 1998"     -->   "1998-03-05"
"Mar 21, 2001"           -->   "2001-03-21"
"21 September 2015"      -->   "2015-09-21"
"Oct. 7, 1987"           -->   "1987-10-07"
```

This task is ideal because:
- It requires non-trivial reordering (year moves from end to beginning).
- The input format varies (the model must generalize across formats).
- It is small enough to train on a laptop in minutes.
- Attention patterns are clearly interpretable (you can see which input characters the decoder attends to when generating each output character).

### Dataset Generation

Generate a synthetic dataset of 50,000 date pairs. Include diverse input formats:

```python
import random
from datetime import datetime, timedelta

FORMATS = [
    "%B %d, %Y",          # January 05, 2023
    "%d %B %Y",            # 05 January 2023
    "%b %d, %Y",           # Jan 05, 2023
    "%d %b %Y",            # 05 Jan 2023
    "%B %d %Y",            # January 05 2023
    "%d/%m/%Y",            # 05/01/2023
    "%m/%d/%Y",            # 01/05/2023
]

def generate_date_pair():
    """Generate a random (human_date, machine_date) pair."""
    # Random date between 1950 and 2030
    start = datetime(1950, 1, 1)
    end = datetime(2030, 12, 31)
    delta = end - start
    random_date = start + timedelta(days=random.randint(0, delta.days))

    human_format = random.choice(FORMATS)
    human_date = random_date.strftime(human_format)
    machine_date = random_date.strftime("%Y-%m-%d")

    return human_date, machine_date

# Generate dataset
dataset = [generate_date_pair() for _ in range(50000)]
```

Split: 40,000 training, 5,000 validation, 5,000 test.

---

## Part 1: Encoder-Decoder WITHOUT Attention (25%)

### Task

Build a basic encoder-decoder model:

1. **Character-level tokenization**: Build vocabularies for input and output characters. Include special tokens: `<PAD>`, `<BOS>` (beginning of sequence), `<EOS>` (end of sequence).

2. **Encoder**: An LSTM that reads the input date character by character. The final hidden state and cell state are the context — the only information passed to the decoder.

3. **Decoder**: An LSTM initialized with the encoder's final states. At each step, it takes the previous output character (or ground truth during teacher forcing) and produces a probability distribution over output characters.

4. **Training**:
   - Use cross-entropy loss, ignoring padding tokens.
   - Use teacher forcing with a ratio of 0.5.
   - Train with Adam optimizer, learning rate 1e-3.
   - Use gradient clipping with max_norm=5.0.
   - Train for 30 epochs (or until convergence).

### Architecture Details

```
Encoder:
  - Embedding: vocab_size -> 64
  - LSTM: 64 -> 128, 1 layer

Decoder:
  - Embedding: vocab_size -> 64
  - LSTM: 64 -> 128, 1 layer
  - Linear: 128 -> vocab_size
```

### Requirements

1. Implement the `Encoder`, `Decoder`, and `Seq2Seq` classes.
2. Implement the training loop with teacher forcing.
3. Implement greedy decoding for inference.
4. Evaluate on the test set: report exact-match accuracy (percentage of dates converted correctly).
5. Report performance broken down by input format and input length.

### Expected Results

The model without attention should achieve reasonable but imperfect accuracy. It will likely struggle with:
- Longer input formats (more information to compress into the context vector).
- Formats where the year appears at the end (the decoder needs the year first, but it was encoded last and may be partially forgotten).

Record the exact-match accuracy. You will compare this to the attention model in Part 2.

---

## Part 2: Adding Bahdanau Attention (30%)

### Task

Add Bahdanau (additive) attention to your decoder. The decoder should now compute a context vector at each time step by attending to all encoder hidden states.

### Requirements

1. **BahdanauAttention module**:
   ```python
   class BahdanauAttention(nn.Module):
       def __init__(self, encoder_hidden_size, decoder_hidden_size, attention_size):
           # W_h: projects encoder states
           # W_s: projects decoder state
           # v: produces scalar score
           ...

       def forward(self, decoder_state, encoder_outputs):
           # decoder_state: (batch, decoder_hidden_size)
           # encoder_outputs: (batch, src_len, encoder_hidden_size)
           # Returns: context (batch, encoder_hidden_size), weights (batch, src_len)
           ...
   ```

2. **Modified decoder**: The LSTM input at each step should be the concatenation of the embedded input token and the attention context vector. The output projection should use the concatenation of the LSTM hidden state and the context vector.

3. **Return attention weights**: The decoder's forward method must return the attention weights at each step so you can visualize them later.

4. **Training**: Same hyperparameters as Part 1, same number of epochs.

5. **Evaluation**: Report exact-match accuracy on the test set. Compare to the model without attention.

### Architecture Details

```
Encoder: (same as Part 1)
  - Embedding: vocab_size -> 64
  - LSTM: 64 -> 128, 1 layer

Attention:
  - attention_size: 128

Decoder (modified):
  - Embedding: vocab_size -> 64
  - LSTM: (64 + 128) -> 128, 1 layer   # Input is embed + context
  - Linear: (128 + 128) -> vocab_size   # Output uses hidden + context
```

### Expected Results

The attention model should achieve notably higher accuracy, especially on longer inputs and formats where the year appears at the end. The attention mechanism allows the decoder to directly look at the part of the input it needs, bypassing the bottleneck.

---

## Part 3: Attention Visualization (20%)

### Task

Visualize the attention weights as heatmaps to understand what the model has learned.

### Requirements

1. **Generate attention heatmaps** for at least 10 test examples. For each:
   - Run the model in inference mode (greedy decoding).
   - Collect the attention weights at each decoding step.
   - Plot a heatmap with input characters on the x-axis and output characters on the y-axis.
   - The intensity at position (i, j) shows how much the decoder attended to input position j when generating output position i.

2. **Include examples from different input formats.** Show at least:
   - A case where the model correctly reorders year, month, day.
   - A case where the model handles a spelled-out month name (e.g., "September" -> "09").
   - A case where the model fails, if any.

3. **Annotate your visualizations.** For each heatmap, write 2-3 sentences explaining what the attention pattern reveals. For example:
   - "When generating the year digits '2023', the decoder attends strongly to the last four characters of the input, which contain '2023'. This shows the model has learned to find the year regardless of its position."
   - "When generating the month '09', the decoder attends to 'September', spreading attention across the first few characters. This suggests the model is reading the month name to determine the numeric code."

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(input_chars, output_chars, attention_weights):
    """
    input_chars: list of characters in the source
    output_chars: list of characters in the generated output
    attention_weights: numpy array of shape (output_len, input_len)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')

    ax.set_xticks(range(len(input_chars)))
    ax.set_xticklabels(input_chars, fontsize=10)
    ax.set_yticks(range(len(output_chars)))
    ax.set_yticklabels(output_chars, fontsize=10)

    ax.set_xlabel('Input (source date)')
    ax.set_ylabel('Output (target date)')
    ax.set_title('Attention Weights')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
```

### Expected Patterns

You should observe interpretable patterns:
- When generating year digits, the model attends to the part of the input containing the year.
- When generating month digits, the model attends to the month name or number.
- When generating day digits, the model attends to the day number.
- The attention pattern is NOT monotonic (not a simple diagonal) because the output format reorders the date components.

This interpretability is one of the great strengths of attention mechanisms and one of the reasons the research community was so excited about them.

---

## Part 4: Beam Search Decoding (15%)

### Task

Implement beam search and compare its output quality to greedy decoding.

### Requirements

1. **Implement beam search** with configurable beam width K.

   ```python
   def beam_search_decode(model, encoder_outputs, encoder_hidden,
                          beam_width=5, max_len=12):
       """
       Returns the highest-scoring sequence found by beam search.

       Each beam is a tuple: (log_probability, sequence, hidden_state, cell_state)
       At each step:
         1. Expand each beam with the top-K next tokens.
         2. From all expansions, keep the top-K beams overall.
         3. Stop when all beams have generated <EOS> or max_len is reached.
       """
       ...
   ```

2. **Evaluate with different beam widths**: K = 1 (greedy), 3, 5, 10.
   - Report exact-match accuracy for each beam width.
   - Report average inference time per example.

3. **Find examples where beam search helps.** Identify specific test cases where:
   - Greedy decoding produces an incorrect output.
   - Beam search (K=5) produces the correct output.
   - Explain why: what locally optimal choice did greedy decoding make that led it astray?

4. **Length normalization.** Raw beam search favors shorter sequences (shorter sequences accumulate less negative log-probability). Implement length normalization:
   ```
   score = log_probability / (length ^ alpha)
   ```
   where alpha is typically 0.6-0.7. Compare results with and without length normalization.

### Expected Results

For this task, beam search may provide a modest improvement over greedy decoding (since the output is short and highly constrained). The improvement will be more dramatic on tasks with longer, more varied outputs. The point is to understand the mechanism and implement it correctly.

---

## Part 5: Written Reflection (10%)

### Task

Write a 1-page reflection (approximately 400-600 words) titled:

**"How Attention Solves the Information Bottleneck"**

Address the following questions:

1. What is the information bottleneck in the vanilla encoder-decoder, and how did you observe its effects in your experiments?

2. How does the attention mechanism address this bottleneck? Be specific — refer to the equations and to what you observed in the attention weight visualizations.

3. What are the computational costs of attention? How does the attention computation scale with input length?

4. The attention mechanism creates a direct connection between the decoder and every encoder position. What does this mean for gradient flow during training? How does this relate to the vanishing gradient problem you studied in Assignment 01?

5. The query-key-value interpretation of attention is the foundation of the Transformer. Based on what you have learned, what would it mean to apply attention *within* a single sequence (self-attention) rather than between an encoder and decoder? What new capabilities might this enable?

This reflection is not busywork. It is preparation for the Transformer module. If you can articulate why attention matters, the Transformer will make immediate sense.

---

## Deliverables

Submit a Jupyter notebook (or organized Python scripts) containing:

1. **Data generation and preprocessing code.**
2. **Encoder-Decoder model WITHOUT attention** — implementation, training, evaluation.
3. **Bahdanau Attention module** — implementation with clear documentation.
4. **Encoder-Decoder model WITH attention** — implementation, training, evaluation.
5. **Comparison table**: accuracy with and without attention, broken down by input length/format.
6. **Attention visualizations**: at least 10 heatmaps with annotations.
7. **Beam search implementation** with comparison across beam widths.
8. **Written reflection** (1 page).

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Model correctness | 25% | Both models train and produce reasonable outputs |
| Attention implementation | 25% | Attention mechanism is correct, weights sum to 1, context vector is properly computed |
| Visualization quality | 20% | Heatmaps are clear, well-labeled, and accompanied by meaningful analysis |
| Beam search | 15% | Correct implementation, comparison across beam widths |
| Reflection | 15% | Thoughtful, specific, demonstrates genuine understanding |

---

## Stretch Goals

1. **Implement Luong (multiplicative) attention.** Implement both the "dot" and "general" variants. Compare attention weight patterns to Bahdanau attention. Are they similar? Does one converge faster?

2. **Implement a copy mechanism.** In date conversion, the model often needs to copy digits directly from input to output. Implement a simple copy mechanism (Pointer Network style): at each decoder step, compute a "copy probability" and allow the model to either generate from the vocabulary or copy a character from the input based on attention weights. Does this improve accuracy on digit-heavy dates?

3. **Try a harder task.** Replace dates with simple English-to-French translation pairs (you can find small parallel corpora online, or use a subset of the Tatoeba dataset). How does the model perform? Where does it fail? How does attention help differently on natural language compared to the structured date task?

4. **Implement scheduled sampling.** Instead of a fixed teacher forcing ratio, implement a schedule that decays from 1.0 to 0.0 over training. Compare training curves and final accuracy to fixed-ratio teacher forcing.

5. **Multi-head attention.** Instead of one set of attention weights, compute H independent attention heads (each with smaller dimensionality) and concatenate them. This foreshadows the multi-head attention in Transformers. Does it help on this task?

---

## Practical Tips

- **Start simple.** Get the model without attention working first. Debug it thoroughly before adding attention. Many bugs hide in the attention implementation, and having a known-working baseline makes them easier to find.

- **Check attention weight shapes.** The attention weights at each decoding step should be a probability distribution over source positions — they must sum to 1. Add an assertion for this during development.

```python
assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5), \
    f"Attention weights don't sum to 1: {weights.sum(dim=-1)}"
```

- **Pad sequences carefully.** Different input dates have different lengths. Use padding and make sure your attention mechanism masks padded positions (set their scores to -inf before softmax so they get zero weight).

```python
# Masking padded positions in attention
if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))
weights = torch.softmax(scores, dim=-1)
```

- **Save your best model.** Use validation loss to select the best checkpoint. The model may overfit after many epochs.

- **Use a consistent random seed** for reproducibility. Set seeds for `random`, `numpy`, and `torch`.

---

## The Bigger Picture

When you finish this assignment, take a moment to appreciate the arc of what you have built.

In Assignment 01, you built RNN and LSTM cells and saw that LSTMs can learn longer-range dependencies than vanilla RNNs by providing a gradient highway through the cell state.

In this assignment, you built an encoder-decoder that maps one sequence to another, and you saw that it fails when the input is too long for the fixed-size context vector to handle. You then added attention, which lets the decoder reach back and access any part of the input directly — removing the bottleneck and creating direct gradient paths.

Next week, you will see that the Transformer takes this further: it removes the recurrent backbone entirely and uses attention (specifically, self-attention) as the sole mechanism for mixing information between positions. Everything you have learned about attention in this assignment — the alignment scores, the softmax normalization, the weighted sum, the query-key-value interpretation — transfers directly.

The Transformer did not appear from thin air. It was the logical next step after the insights you have now gained firsthand. And that is exactly why we taught this module the way we did.
