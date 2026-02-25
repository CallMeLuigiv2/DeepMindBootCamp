# Assignment 1: Paper Dissection — "Attention Is All You Need"

## Overview

This assignment teaches you to read a paper at the deepest possible level. You will take one
of the most important papers in modern machine learning — "Attention Is All You Need" (Vaswani
et al., 2017) — and dissect every component. By the end, you will understand the Transformer
architecture not just well enough to use it, but well enough to have invented it.

This is the standard that DeepMind expects: not "I read the paper," but "I could re-derive the
paper from the problem statement."

**Estimated time:** 15-20 hours over 1-2 weeks

**Paper:** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N.,
Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017.*
https://arxiv.org/abs/1706.03762

---

## Part 1: Three-Pass Reading (3-4 hours)

### Pass 1 — Triage (10 minutes)

Read only the title, abstract, section headings, all figures and tables (with captions), and
the conclusion. Then answer:

1. In one sentence, what is this paper about?
2. What type of paper is this? (new method / empirical study / theoretical / survey)
3. How many figures are there? What does each one show?
4. What is the main result claimed in the abstract?
5. What year was this published? How does that context matter?

Write your answers before moving to Pass 2.

### Pass 2 — Understanding (1-1.5 hours)

Read the full paper. Do not try to re-derive anything yet — focus on understanding the
argument flow. As you read, answer:

1. What is the main problem being addressed? Why do the authors argue that existing approaches
   (RNNs, convolution-based models) are insufficient?
2. What are the three key claims made in the introduction?
3. For each major component of the architecture (Figure 1), explain in one sentence what it
   does and why it is needed.
4. List every hyperparameter mentioned in the paper (model dimension, number of heads, etc.)
   and their values.
5. What baselines do the authors compare against? Are these baselines fair?
6. What are the main results on WMT English-to-German and English-to-French translation?
7. What does the ablation study (Table 3) reveal about which design choices matter?
8. What references do you want to follow up on? (List at least 3.)

### Pass 3 — Mastery (2+ hours)

Now go deep. Re-read the paper with the goal of being able to reconstruct every technical
detail from scratch:

1. Re-derive the scaled dot-product attention formula. Why scale by 1/sqrt(d_k)? What happens
   without scaling? (The paper gives a specific argument — find it and evaluate it.)
2. Explain multi-head attention. Why is it better than single-head attention with the same
   total dimension? What does each head learn?
3. Trace the exact data flow through a single encoder layer. What are the tensor shapes at
   every step?
4. Trace the exact data flow through a single decoder layer. How does it differ from the
   encoder?
5. Explain the positional encoding formula. Why sinusoidal functions? What property do they
   have that makes them suitable for representing position?
6. What is the label smoothing value they use? Why does label smoothing help?
7. What is the learning rate schedule? Write the exact formula. Why this particular schedule?

---

## Part 2: Equation-by-Equation Analysis (4-5 hours)

For every numbered equation in the paper, complete the following table. The paper contains
equations for scaled dot-product attention, multi-head attention, position-wise FFN, and
positional encoding. There are also unnumbered mathematical expressions throughout the text.

### For each equation:

**Template:**

```
Equation number: ___
Paper's notation:
    [copy the equation exactly as written]

Rewritten in my notation:
    [rewrite using notation you find clearer, if different]

What it computes (plain English):
    [explain in 2-3 sentences what this equation does]

Input shapes:
    [specify the tensor shapes going in]

Output shapes:
    [specify the tensor shapes coming out]

Key insight:
    [what is the purpose of this equation in the larger method?]

PyTorch equivalent:
    [write the 1-5 lines of PyTorch that implement this equation]

Questions or concerns:
    [anything unclear or potentially problematic about this equation]
```

Complete this for ALL of the following:
1. Equation 1: Scaled dot-product attention
2. Equation 2: Multi-head attention (with projection matrices)
3. Equation 3: Position-wise feed-forward network
4. Equation 4: Positional encoding (sine component)
5. Equation 5: Positional encoding (cosine component)
6. The softmax within the attention formula
7. The residual connection + layer normalization (described in text, Section 3.1)
8. The label smoothing formulation (Section 5.4)

---

## Part 3: Implementation Gap Analysis (2-3 hours)

Read the paper one more time, this time asking: "Could I implement this from what is written
here?"

### Identify at least 3 things the paper does NOT explain well enough to implement directly.

For each gap, provide:
1. **What is missing:** Describe the specific detail that is absent.
2. **Why it matters:** Explain how this ambiguity would affect your implementation.
3. **How to resolve it:** Where would you find the answer? (Appendix? Code? Follow-up paper?
   Experimentation?)

Examples of the kind of gaps to look for (do not use these exact ones unless they are genuine):
- Exact weight initialization scheme
- Dropout placement (where exactly in the architecture)
- How masking works in the decoder self-attention
- Tokenization and vocabulary construction details
- Whether residual connections are before or after layer normalization
- Gradient clipping details
- How teacher forcing works during training

---

## Part 4: Follow-Up Papers (3-4 hours)

Read two papers that directly built on the Transformer. Choose from:
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford et al., 2019)
- "An Image is Worth 16x16 Words" (Vision Transformer, Dosovitskiy et al., 2020)
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
- "Efficient Transformers: A Survey" (Tay et al., 2020)

For each follow-up paper:
1. Do a second-pass reading (1-1.5 hours per paper).
2. Answer:
   - What did they change about the original Transformer and why?
   - What problem with the original Transformer does their work address?
   - What did they keep unchanged, and does that seem appropriate?
   - How do their results compare to the original paper's results (on comparable tasks)?
3. Fill out a paper reading template for each.

---

## Part 5: Conference-Style Review (2-3 hours)

Write a 2-page critical review of "Attention Is All You Need" as if you were reviewing it
for a top ML conference. Follow the standard review format:

### 1. Summary (1 paragraph)
Demonstrate that you understand the paper. State the problem, method, and main results in
your own words. Do not copy the abstract.

### 2. Strengths (3-5 bullet points)
What makes this a good paper? Be specific. Reference particular sections, figures, or results.

### 3. Weaknesses (3-5 bullet points)
What are the limitations? Be specific and constructive. For each weakness, suggest how it
could be addressed. Remember: you are reviewing with hindsight, but evaluate based on what
was known in 2017.

### 4. Questions for the Authors (3-5 questions)
What would you ask the authors? These should be genuine questions that would help you
evaluate the paper.

### 5. Minor Issues
Any typos, unclear notation, or formatting issues you noticed.

### 6. Overall Assessment
Score (1-10) with a 2-3 sentence justification. Would you have accepted this paper?

**Important:** Write the review honestly. This paper changed the field, but no paper is
perfect. A thoughtful review that identifies genuine weaknesses demonstrates deeper
understanding than uncritical praise.

---

## Deliverables

Submit the following:

1. **Three-pass notes** — Your answers to all questions from Part 1 (3 passes).
2. **Equation analysis table** — Complete analysis for all 8+ equations from Part 2.
3. **Implementation gap analysis** — At least 3 gaps identified with resolution strategies
   from Part 3.
4. **Two paper reading templates** — Completed templates for the two follow-up papers from
   Part 4.
5. **Conference-style review** — 2-page review from Part 5.

All deliverables should be in markdown format. Place them in your module 11 work directory.

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Depth of understanding | 30% | Do your answers demonstrate genuine comprehension, not surface-level repetition? |
| Equation analysis quality | 25% | Are equations explained precisely? Are PyTorch equivalents correct? |
| Critical thinking | 20% | Do you identify genuine gaps, limitations, and non-obvious insights? |
| Review quality | 15% | Is the review specific, balanced, and professional? |
| Completeness | 10% | Are all deliverables present and thorough? |

### What "excellent" looks like:
- Your equation analysis includes subtle points like why the scaling factor prevents softmax
  saturation and how this relates to the variance of dot products in high dimensions.
- Your implementation gap analysis identifies details that would actually cause bugs, not
  just obvious omissions.
- Your review identifies weaknesses that even the authors might not have considered in 2017,
  while still acknowledging the paper's enormous contribution.
- Your follow-up paper analysis draws connections between the original Transformer and its
  descendants that go beyond "they added X."

### What "needs improvement" looks like:
- Equation analysis that restates the paper without adding understanding.
- Implementation gaps that are trivially obvious (e.g., "they didn't specify the batch size"
  — they did, in Section 5.1).
- A review that is entirely positive or entirely negative.
- Follow-up paper analysis that does not connect back to the original paper.

---

## A Note on This Assignment

"Attention Is All You Need" is not just a good paper to dissect — it is THE paper that defined
modern deep learning architectures. Every large language model, every vision transformer, every
multimodal system you will encounter in your career traces its lineage to this paper. The
Transformer is to deep learning what the transistor is to electronics.

Your goal is not to memorize the paper. Your goal is to understand it so deeply that you could
have written it yourself, given the same problem and the same prior work. That is the level of
understanding that makes original research possible.
