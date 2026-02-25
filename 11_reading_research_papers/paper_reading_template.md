# Paper Reading Template

> Fill out this template for every paper you read past the first pass. Be honest about what
> you do and do not understand. These templates become your personal knowledge base — the more
> precise you are now, the more useful they are later.
>
> Make a copy of this file for each paper. Name it: `YYYY_LastName_ShortTitle.md`
> Example: `2017_Vaswani_AttentionIsAllYouNeed.md`

---

## Citation

- **Title:**
- **Authors:**
- **Year:**
- **Venue:** (e.g., NeurIPS 2017, ICML 2020, arXiv preprint)
- **URL/DOI:**
- **Date I read this paper:**
- **Pass level completed:** (First / Second / Third)

---

## One-Sentence Summary

<!-- Summarize the entire paper in one sentence. This is harder than it sounds.
If you cannot do this, you do not yet understand the paper. -->



---

## Problem

<!-- What problem does this paper address? Why does it matter? Who would care about
a solution to this problem? Be specific — "improving NLP" is too vague. -->



---

## Key Insight

<!-- What is the single core idea that makes this work? One paragraph maximum.
This should be the thing that, if you forgot everything else about this paper,
would still capture its essence. Try to state it in a way that someone unfamiliar
with the paper could understand. -->



---

## Method

### High-Level Description

<!-- 2-3 sentences describing the approach at a high level. What goes in? What comes out?
What are the main computational steps? -->



### Key Equations

<!-- Write out the most important equations from the paper. For each one:
1. The equation itself
2. Define every symbol
3. Explain what it computes in plain English
4. Note the corresponding PyTorch operation if applicable -->

**Equation 1:**
```
[Write equation here]
```
- Symbols:
- Plain English:
- PyTorch:

**Equation 2:**
```
[Write equation here]
```
- Symbols:
- Plain English:
- PyTorch:

**Equation 3:**
```
[Write equation here]
```
- Symbols:
- Plain English:
- PyTorch:

### Architecture Diagram

<!-- Sketch or describe the architecture. If you drew it by hand, photograph it
and link it here. The act of drawing the architecture by hand, without looking
at the paper's figure, is one of the most effective ways to test your understanding. -->



### Loss Function

<!-- What loss function is used? What does it optimize? Why this particular loss? -->



### Key Design Choices

<!-- What are the most important design decisions? For each, note whether the authors
justify the choice and whether you find the justification convincing. -->

| Design Choice | Authors' Justification | Convincing? |
|---------------|----------------------|-------------|
| | | |
| | | |
| | | |

---

## Training Details

<!-- These details are critical for reproduction. Extract them carefully. -->

- **Dataset(s):**
- **Preprocessing / augmentation:**
- **Model size (parameters):**
- **Optimizer:**
- **Learning rate (initial):**
- **Learning rate schedule:**
- **Batch size:**
- **Training duration (epochs / steps):**
- **Compute used (GPUs, type, time):**
- **Weight initialization:**
- **Regularization (dropout, weight decay, etc.):**
- **Other key hyperparameters:**

---

## Results

### Main Findings

<!-- What are the headline results? Be specific with numbers. -->



### Comparison to Baselines

<!-- Fill in the comparison table. Use the paper's primary metric. -->

| Method | Metric 1 | Metric 2 | Notes |
|--------|----------|----------|-------|
| This paper | | | |
| Baseline 1 | | | |
| Baseline 2 | | | |
| Baseline 3 | | | |

### Ablation Results

<!-- What happens when you remove or change each component? This reveals what
actually matters in the method. -->

| Variant | Metric | Change from Full | What This Tells Us |
|---------|--------|-----------------|-------------------|
| Full model | | (baseline) | |
| Remove component A | | | |
| Remove component B | | | |
| Change component C | | | |

### Key Figures and Tables

<!-- Which figures and tables are most important? What do they show?
Reference them by number. -->



---

## Strengths

<!-- List at least 3 genuine strengths. Be specific. -->

1.

2.

3.

---

## Weaknesses

<!-- List at least 3 weaknesses or limitations. Be specific and constructive —
suggest how each could be addressed. -->

1.

2.

3.

---

## Questions

<!-- Things you do not understand, or things you would ask the authors.
Be specific. "I don't understand the method" is not useful.
"I don't understand why Equation 3 uses a stop-gradient on the target network" is useful. -->

1.

2.

3.

---

## Connections to Other Papers

<!-- How does this paper relate to others you have read? This section becomes
more valuable over time as your reading builds a network of connections. -->

- **Builds on:** <!-- Which prior works does this directly extend? -->

- **Competes with:** <!-- Which concurrent or subsequent works address the same problem differently? -->

- **Inspired:** <!-- What later works cite this as a key influence? -->

- **Related insight:** <!-- Does the key insight here connect to an insight in a different area? -->

- **Contradicts:** <!-- Does this paper's finding conflict with another paper's claims? -->

---

## Implementation Notes

<!-- If you were to implement this paper, what would you need to know beyond
what is in the method section? -->

### Details the paper specifies clearly:


### Details the paper does NOT specify (you would need to figure out):


### Potential numerical issues:


### Estimated implementation effort:

<!-- Simple (1 day) / Medium (1 week) / Hard (2+ weeks) / Research-level (1+ months) -->

---

## Rating: ___ / 10

<!-- Score the paper on a 1-10 scale and justify your rating in 2-3 sentences.
Consider all four dimensions: novelty, rigor, reproducibility, significance.

1-2: Fundamentally flawed or trivial
3-4: Below average, significant issues
5-6: Decent paper with notable strengths and weaknesses
7-8: Strong paper, clear contribution
9-10: Outstanding, field-changing work

Your ratings will calibrate over time. Do not be afraid to give low scores to
famous papers if you genuinely find weaknesses. Do not be afraid to give high
scores to obscure papers if the work is strong. -->

**Score:**

**Justification:**

---

## Personal Notes

<!-- Anything else you want to remember about this paper. Ideas it sparked,
connections to your own work, things you want to revisit. This section is
just for you. -->


