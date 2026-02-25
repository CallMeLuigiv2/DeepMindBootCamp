# Road to ML Expert: A DeepMind-Style Deep Learning Course

## Course Philosophy

This is not a "follow along and copy code" course. This is an **apprenticeship**. Every concept
is taught with the expectation that you will implement it from scratch, break it, fix it, and
then build something real with it. That's how DeepMind engineers are forged.

**Three Pillars:**
1. **Understand the math** — not memorize it, *understand* it. Every equation maps to code.
2. **Build from scratch first** — use frameworks second. You cannot debug what you don't understand.
3. **Read papers, implement papers** — the frontier of ML lives in papers, not tutorials.

---

## Who This Course Is For

You know Python. You've seen some ML. You can write a for loop and a class. But you haven't
yet hit the level where you can pick up a paper from NeurIPS, read it over coffee, and have a
working implementation by dinner. **That's where we're going.**

---

## Course Structure (16 Weeks)

| Week | Module | Focus |
|------|--------|-------|
| 1-2 | **01 - Classical ML Foundations** | The algorithms that still matter. Gradient descent deeply. |
| 3 | **02 - Math for Deep Learning** | Linear algebra, calculus, probability — the DL lens. |
| 4-5 | **03 - PyTorch Fundamentals** | Tensors, autograd, datasets, training loops from scratch. |
| 6-7 | **04 - Neural Networks Deep Dive** | MLPs, backprop by hand, initialization, regularization. |
| 8-9 | **05 - Convolutional Neural Networks** | Convolutions, classic architectures, transfer learning. |
| 10 | **06 - Sequence Models** | RNNs, LSTMs, GRUs, and why they still matter. |
| 11-12 | **07 - Transformers & Attention** | The architecture that changed everything. Build one. |
| 13 | **08 - Generative Models** | VAEs, GANs, Diffusion — learn to create, not just classify. |
| 14 | **09 - Advanced PyTorch** | Custom autograd, hooks, distributed training, TorchScript. |
| 15 | **10 - AI Performance Engineering** | Profiling, mixed precision, memory optimization, speed. |
| 15-16 | **11 - Reading Research Papers** | How to read, critique, and implement papers. |
| 16 | **12 - Capstone Project** | Your DeepMind-level project. End-to-end. |

---

## How Lessons Are Delivered

Each module contains:

- **`lesson_plan.md`** — Learning objectives, topic breakdown, timeline, teaching strategy.
- **`notes.md`** — Dense, reference-quality notes. Math is explained with intuition FIRST,
  formalism SECOND. Every equation has a "what this really means" block. Code examples
  are interleaved throughout.
- **`assignments/`** — Hands-on work. Each assignment has:
  - Clear objectives and deliverables
  - Starter hints (not starter code — you write it)
  - Evaluation criteria
  - Stretch goals for those who want more

---

## How Math Is Taught

Every mathematical concept follows this pattern:

1. **Intuition** — A plain-English explanation with a visual analogy
2. **The Equation** — Formal notation, broken down term by term
3. **Code Translation** — The same equation as NumPy/PyTorch code
4. **Why It Matters** — Where this shows up in real DL systems

Example:
> **Softmax** — Imagine you have a bag of scores. Softmax turns them into probabilities
> that sum to 1. Higher scores get exponentially more probability mass.
>
> `softmax(x_i) = exp(x_i) / sum(exp(x_j))`
>
> In PyTorch: `torch.softmax(logits, dim=-1)`
>
> This is the last layer of almost every classification network you'll ever build.

---

## Prerequisites

- Python proficiency (classes, decorators, generators, context managers)
- Basic linear algebra (vectors, matrices, dot products)
- Some exposure to ML concepts (what is training, what is a loss function)
- A GPU-capable machine or access to Google Colab / cloud GPUs

---

## Setup

See `00_course_overview/setup_guide.md` for environment setup instructions.

---

## Evaluation Philosophy

There are no grades. There is only: **can you build it?**

Every assignment ends with a working system. Every project produces something real.
The capstone is a paper-quality implementation that you could present at a reading group.

If your code runs, produces correct results, and you can explain every line — you pass.
If you copied it from Stack Overflow and can't explain the backward pass — you don't.

---

*Designed with the rigor of a DeepMind research internship and the patience of a good mentor.*
