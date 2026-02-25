# Assignment 3: Fine-Tuning Pretrained Transformers

## Overview

In Assignments 1 and 2, you built Transformers from scratch. Now you will work with production-scale pretrained models. This assignment bridges the gap between understanding the architecture and using it effectively in practice.

You will fine-tune BERT for text classification, use GPT-2 for text generation, analyze attention patterns from pretrained models, and confront the central question of modern NLP: when does fine-tuning a pretrained model beat training from scratch?

**Estimated time**: 10-15 hours
**Prerequisites**: Assignments 1-2, Module 7 Sessions 4-6, familiarity with HuggingFace transformers library

---

## Setup

### Required Libraries

```bash
pip install transformers datasets tokenizers evaluate accelerate
pip install torch  # if not already installed
pip install matplotlib seaborn
```

### Hardware

- GPU recommended (fine-tuning BERT on CPU is feasible but slow)
- Google Colab with T4 GPU is sufficient for all parts of this assignment
- Estimated GPU memory: 4-8 GB

---

## Part 1: Fine-Tune BERT for Sentiment Analysis

### Task

Fine-tune `bert-base-uncased` on a sentiment classification task.

### Dataset

Use the **SST-2** (Stanford Sentiment Treebank) dataset from the GLUE benchmark, or **IMDB** movie reviews. Both are available through HuggingFace Datasets:

```python
from datasets import load_dataset

# Option A: SST-2 (shorter texts, binary sentiment)
dataset = load_dataset("glue", "sst2")

# Option B: IMDB (longer reviews, binary sentiment)
dataset = load_dataset("imdb")
```

Pick one. SST-2 is faster to train on; IMDB has longer texts and is more challenging.

### Requirements

1. **Load and tokenize the data**:
   ```python
   from transformers import BertTokenizer

   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   # Tokenize all examples, truncate/pad to max_length=128 (SST-2) or 512 (IMDB)
   ```

2. **Load the pretrained model**:
   ```python
   from transformers import BertForSequenceClassification

   model = BertForSequenceClassification.from_pretrained(
       'bert-base-uncased', num_labels=2
   )
   ```

3. **Implement proper fine-tuning** with the following techniques:

   **a. Discriminative learning rates**: use a lower learning rate for the pretrained BERT layers and a higher rate for the new classification head. Rationale: the pretrained layers already contain useful features; large updates would destroy them.

   ```python
   # Example: 2e-5 for BERT layers, 1e-3 for the classification head
   optimizer = torch.optim.AdamW([
       {'params': model.bert.parameters(), 'lr': 2e-5},
       {'params': model.classifier.parameters(), 'lr': 1e-3}
   ], weight_decay=0.01)
   ```

   **b. Learning rate warmup**: use linear warmup for the first 10% of training steps, then linear decay to zero.

   ```python
   from transformers import get_linear_schedule_with_warmup

   num_training_steps = num_epochs * len(train_dataloader)
   num_warmup_steps = int(0.1 * num_training_steps)
   scheduler = get_linear_schedule_with_warmup(
       optimizer, num_warmup_steps, num_training_steps
   )
   ```

   **c. Gradient clipping**: clip gradients to max norm 1.0 to prevent instability.

4. **Training loop**: write an explicit training loop (do NOT use HuggingFace Trainer for this assignment — you should understand every step):
   ```python
   for epoch in range(num_epochs):
       model.train()
       for batch in train_dataloader:
           optimizer.zero_grad()
           outputs = model(**batch)
           loss = outputs.loss
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           optimizer.step()
           scheduler.step()
       # Evaluate on validation set after each epoch
   ```

5. **Hyperparameters**:
   - Epochs: 3-5 (BERT fine-tuning typically needs very few epochs)
   - Batch size: 16 or 32
   - Max sequence length: 128 (SST-2) or 256-512 (IMDB)

### Evaluation

Compute and report:
- Accuracy on the validation/test set
- F1 score (macro)
- Confusion matrix

**Expected results**:
- SST-2: accuracy > 90% (state-of-the-art is ~95%)
- IMDB: accuracy > 92% (state-of-the-art is ~96%)

If your accuracy is below 85%, something is wrong — check your learning rate, data preprocessing, or training loop.

### Deliverable

- Complete training code with proper fine-tuning techniques
- Training loss curve and validation accuracy curve
- Final evaluation metrics (accuracy, F1, confusion matrix)
- Analysis: how many epochs were needed? Did discriminative learning rates help?

---

## Part 2: Attention Pattern Analysis

### Task

Visualize and analyze what the pretrained BERT model attends to.

### Requirements

1. **Extract attention weights** from the pretrained (or fine-tuned) BERT model:
   ```python
   model.eval()
   outputs = model(**inputs, output_attentions=True)
   attentions = outputs.attentions  # tuple of (batch, num_heads, seq_len, seq_len) per layer
   ```

2. **Visualize attention patterns** for at least 3 sentences. For each sentence, create:
   - A heatmap of attention weights for each of the 12 layers
   - A heatmap for each of the 12 heads in a selected layer

3. **Analyze the following patterns** (provide evidence from your visualizations):

   **a. Positional patterns**: do any heads primarily attend to the previous token? The next token? The first token ([CLS])?

   **b. Separator attention**: do any heads attend strongly to [SEP] or [CLS] tokens? Why might this be useful?

   **c. Syntactic patterns**: for a sentence with clear syntactic structure (e.g., "The large brown dog that was sitting on the mat barked loudly"), do any heads show subject-verb attention? Adjective-noun attention?

   **d. Layer progression**: how do attention patterns change from layer 1 to layer 12? Are early layers more "local" (attending to nearby tokens) and later layers more "global"?

4. **Attention rollout**: implement attention rollout (Abnar and Zuidema, 2020) — a method to trace which input tokens most influence the final [CLS] representation by multiplying attention matrices across layers:
   ```python
   def attention_rollout(attentions):
       """
       Compute attention rollout across all layers.

       Args:
           attentions: list of (batch, num_heads, seq_len, seq_len) per layer

       Returns:
           rollout: (batch, seq_len) — how much each input token influences [CLS]
       """
       # Average attention across heads
       # Add identity matrix (residual connection)
       # Multiply across layers
       pass
   ```

### Deliverable

- Attention heatmaps for 3+ sentences across multiple layers and heads
- Written analysis of observed patterns (positional, separator, syntactic, layer progression)
- Attention rollout implementation and visualization

---

## Part 3: Training from Scratch vs Fine-Tuning

### Task

Train a small Transformer from scratch on the same sentiment classification task and compare with fine-tuned BERT.

### Requirements

1. **Build a small Transformer classifier from scratch** (using your code from Assignment 2):
   - Use a simple tokenizer (character-level or a small BPE vocabulary)
   - Encoder-only Transformer with a classification head on top of mean-pooled representations
   - d_model=128, num_heads=4, num_layers=2, d_ff=256
   - Train from random initialization on the same training data

2. **Train for a fair comparison**:
   - Use the same training data and evaluation data
   - Train for up to 50 epochs (from-scratch models need much more training)
   - Use the best learning rate schedule you can find (try several)

3. **Report results**:
   - Accuracy and F1 for your from-scratch model
   - Accuracy and F1 for fine-tuned BERT
   - Training time for each

4. **Vary the training data size** to understand when fine-tuning shines:
   - Train both models on 100, 500, 1000, 5000, and full training examples
   - Plot accuracy vs training data size for both models on the same graph

### Analysis Questions

Answer in writing:

1. How much training data does your from-scratch model need to match fine-tuned BERT? Can it ever match?

2. At what data size does fine-tuning start to significantly outperform training from scratch?

3. Why does fine-tuning work so well with so little data? What has BERT learned during pretraining that transfers to sentiment analysis?

4. If you had unlimited compute but only 500 labeled examples, would you fine-tune or train from scratch? What if you had 5 million labeled examples?

### Deliverable

- From-scratch Transformer classifier code and training results
- Accuracy vs training data size plot (both models)
- Written analysis answering the questions above

---

## Part 4: GPT-2 Text Generation

### Task

Use a pretrained GPT-2 model for text generation and explore sampling strategies.

### Requirements

1. **Load GPT-2**:
   ```python
   from transformers import GPT2LMHeadModel, GPT2Tokenizer

   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
   model = GPT2LMHeadModel.from_pretrained('gpt2')
   ```

2. **Implement generation with different sampling strategies**:

   Write your own generation loop (do NOT use `model.generate()` for this part — implement it yourself to understand the mechanics):

   **a. Greedy decoding**: always pick the most probable next token.
   ```python
   next_token = torch.argmax(logits[:, -1, :], dim=-1)
   ```

   **b. Temperature sampling**: divide logits by temperature before softmax.
   ```python
   probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
   next_token = torch.multinomial(probs, num_samples=1)
   ```

   **c. Top-k sampling**: zero out all but the top-k most probable tokens before sampling.
   ```python
   top_k_logits, top_k_indices = torch.topk(logits[:, -1, :], k=k)
   probs = F.softmax(top_k_logits, dim=-1)
   sampled_index = torch.multinomial(probs, num_samples=1)
   next_token = top_k_indices.gather(-1, sampled_index)
   ```

   **d. Top-p (nucleus) sampling**: include the smallest set of tokens whose cumulative probability exceeds p.
   ```python
   sorted_logits, sorted_indices = torch.sort(logits[:, -1, :], descending=True)
   sorted_probs = F.softmax(sorted_logits, dim=-1)
   cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
   # Zero out tokens beyond the nucleus
   sorted_logits[cumulative_probs > p] = float('-inf')
   probs = F.softmax(sorted_logits, dim=-1)
   sampled_index = torch.multinomial(probs, num_samples=1)
   next_token = sorted_indices.gather(-1, sampled_index)
   ```

3. **Generate text** with the following prompt: "The meaning of life is"

   Generate 100 tokens with each strategy. Use multiple settings:
   - Greedy
   - Temperature = 0.5, 0.7, 1.0, 1.5, 2.0
   - Top-k = 5, 10, 50, 100
   - Top-p = 0.8, 0.9, 0.95

4. **Analysis**: for each strategy, rate the generated text on:
   - Coherence (does it make sense?)
   - Diversity (is it repetitive or varied?)
   - Fluency (does it read like natural English?)

### Deliverable

- Custom generation loop implementation (no `model.generate()`)
- Generated text samples for each strategy/parameter combination
- Written analysis comparing strategies

---

## Part 5: Comparative Analysis

### Written Analysis (2-3 pages)

Synthesize your findings from all parts of this assignment:

1. **The power of pretraining**: based on your experiments, why is pretraining so effective? What does BERT learn during pretraining that makes it so data-efficient during fine-tuning? Reference the attention patterns you observed in Part 2.

2. **Encoder vs Decoder**: BERT (encoder) is excellent for understanding/classification tasks. GPT (decoder) is excellent for generation. Why? What architectural and training objective differences drive this specialization?

3. **The fine-tuning recipe**: based on your experience, what are the most important ingredients for successful fine-tuning? Rank the following in order of importance and justify:
   - Learning rate
   - Learning rate warmup
   - Discriminative learning rates
   - Number of epochs
   - Batch size
   - Weight decay

4. **Scaling intuition**: your from-scratch model had ~1M parameters. BERT-base has 110M. GPT-2 has 117M. Based on the scaling laws discussed in the notes, predict: if you trained a 10M parameter model from scratch on the full training set, how would it compare to BERT? What about 1B parameters?

5. **When NOT to fine-tune**: describe a scenario where fine-tuning a pretrained Transformer would be the wrong approach. What would you do instead?

---

## Deliverables Summary

| Part | Deliverable | Weight |
|------|-------------|--------|
| 1 | BERT fine-tuning: code, metrics, analysis | 25% |
| 2 | Attention analysis: visualizations + written analysis | 20% |
| 3 | From-scratch comparison: code, accuracy vs data plot, analysis | 20% |
| 4 | GPT-2 generation: sampling implementations, samples, analysis | 20% |
| 5 | Comparative written analysis | 15% |

Submit as a Jupyter notebook with clear sections, inline outputs, visualizations, and written analysis.

---

## Grading Criteria

**Passing (70%+)**:
- Part 1: BERT fine-tuned successfully, accuracy > 85%
- Part 2: at least basic attention heatmaps for 2+ sentences
- Part 3: from-scratch model trained, comparison table provided
- Part 4: at least 2 sampling strategies implemented

**Distinction (85%+)**:
- All of the above, plus:
- Part 1: accuracy > 90%, proper discriminative LR and warmup
- Part 2: attention analysis identifies at least 3 distinct patterns across heads/layers
- Part 3: accuracy vs data size plot with clear analysis
- Part 4: all 4 sampling strategies with thoughtful analysis
- Part 5: written analysis demonstrates genuine understanding

**Exceptional (95%+)**:
- All of the above, plus:
- Part 1: accuracy > 92%, ablation over learning rates and warmup
- Part 2: attention rollout implemented with insightful visualization
- Part 3: analysis includes concrete numbers for the data efficiency gap
- Part 4: generation analysis includes discussion of repetition penalty, sampling theory
- Part 5: analysis connects to scaling laws and references relevant papers

---

## Stretch Goals

1. **LoRA fine-tuning**: instead of fine-tuning all of BERT's parameters, implement or use LoRA (Low-Rank Adaptation). Compare:
   - Full fine-tuning: all 110M parameters updated
   - LoRA: only ~0.3M additional parameters trained (rank 4, applied to attention weights)
   - Report accuracy and training speed for both

   ```python
   # Using the peft library:
   from peft import get_peft_model, LoraConfig

   config = LoraConfig(r=4, lora_alpha=32, target_modules=["query", "value"])
   model = get_peft_model(model, config)
   ```

2. **BERT vs RoBERTa**: fine-tune both `bert-base-uncased` and `roberta-base` on the same task. RoBERTa was trained with better hyperparameters and more data. How much does this matter?

3. **Prompt engineering for GPT-2**: instead of fine-tuning GPT-2, try to get good sentiment classification by prompting:
   ```
   "Review: This movie was wonderful. Sentiment: positive
    Review: This movie was terrible. Sentiment: negative
    Review: [test review]. Sentiment:"
   ```
   How does few-shot prompting compare to fine-tuning BERT on 100 examples?

4. **Model distillation**: train a small Transformer (your code from Assignment 2) to mimic the fine-tuned BERT's predictions (knowledge distillation). Does the student model outperform the same model trained directly on labels?

5. **Layer-wise analysis**: freeze different numbers of BERT layers and fine-tune only the top layers. How many layers need to be fine-tuned to reach full performance? This tells you how much task-specific adaptation BERT needs.

---

## Common Pitfalls

1. **Tokenizer mismatch**: always use the tokenizer that matches the pretrained model. `bert-base-uncased` requires `BertTokenizer.from_pretrained('bert-base-uncased')`, not a generic tokenizer.

2. **Forgetting to call model.eval()**: during evaluation and generation, dropout and other training-specific behaviors must be disabled. Always call `model.eval()` and use `torch.no_grad()`.

3. **Learning rate too high for fine-tuning**: BERT fine-tuning works best with learning rates around 2e-5 to 5e-5. Using 1e-3 (common for training from scratch) will likely destroy the pretrained weights.

4. **Not using special tokens**: BERT expects [CLS] at the start and [SEP] at the end. The tokenizer handles this automatically if you use `tokenizer(text, return_tensors='pt')`, but be aware if you are manually constructing inputs.

5. **Overfitting on small datasets**: with only a few hundred fine-tuning examples, BERT can overfit quickly. Use early stopping based on validation performance, not training loss.

6. **GPT-2 padding**: GPT-2 does not have a padding token by default. If batching multiple sequences, set `tokenizer.pad_token = tokenizer.eos_token` and handle padding masks carefully.

7. **Generation repetition**: greedy decoding and low-temperature sampling often produce repetitive text ("the the the..."). This is a known issue. Top-k and top-p sampling mitigate it. Repetition penalty (dividing logits of already-generated tokens by a factor > 1) also helps.
