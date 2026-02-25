# Assignment 2: Reproduce a Paper

## Overview

This is the flagship research assignment of the entire course. Reproducing a published paper
from scratch — without copying the authors' code — is the single best way to develop research
engineering skills. It is also how you prove to yourself (and future employers) that you can
go from reading to building.

At DeepMind, every new researcher is expected to reproduce a relevant paper during their first
weeks. Not because the reproduction itself is valuable, but because the process reveals whether
you can bridge the gap between a paper's description and working code. That gap is where all
the hard-won knowledge lives.

**Estimated time:** 40-60 hours over 2-4 weeks

---

## Paper Selection

Choose ONE paper from the list below. Each paper has been selected because: (a) the core method
is implementable by a single person, (b) the key experiments can be reproduced at small scale,
and (c) the paper teaches important lessons about the gap between description and implementation.

### Option A: ResNet (He et al., 2015)

**Paper:** "Deep Residual Learning for Image Recognition"
https://arxiv.org/abs/1512.03385

**What you will reproduce:**
- ResNet-18 and ResNet-34 on CIFAR-10
- The comparison between plain networks and residual networks at different depths
- The key finding: plain networks degrade with depth, residual networks do not

**Why this paper:**
Skip connections are one of the most important architectural innovations in deep learning.
Implementing ResNet teaches you about architecture design, the importance of initialization,
and how small design choices (projection shortcuts vs. identity shortcuts) affect performance.

**Scale:** Feasible on a single GPU in a few hours per experiment.

**Key challenges:**
- Getting the architecture details right (downsampling strategy, number of filters per stage)
- Matching the data augmentation pipeline
- Reproducing the training schedule (learning rate, weight decay, momentum)
- The paper trains on ImageNet; you will train on CIFAR-10, which requires architectural
  modifications not fully specified in the paper

### Option B: Batch Normalization (Ioffe & Szegedy, 2015)

**Paper:** "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
https://arxiv.org/abs/1502.03167

**What you will reproduce:**
- The acceleration effect: networks with batch norm train faster than identical networks without
- The key experiment: training curves with and without batch norm at various learning rates
- The finding that batch norm allows higher learning rates

**Why this paper:**
Batch normalization is used in nearly every modern architecture, yet the original paper's
explanation has been challenged. Implementing it from scratch teaches you about numerical
stability, the difference between training and evaluation behavior, and the importance of
running statistics.

**Scale:** Feasible on a single GPU. Experiments are fast (MNIST and CIFAR-10).

**Key challenges:**
- Implementing batch norm correctly (running mean/variance tracking, train vs eval mode)
- Numerical stability (epsilon, the order of operations)
- Fairly comparing networks with and without batch norm (controlling for parameter count)
- Understanding the difference between the paper's claimed mechanism and what actually happens

### Option C: DDPM (Ho et al., 2020)

**Paper:** "Denoising Diffusion Probabilistic Models"
https://arxiv.org/abs/2006.11239

**What you will reproduce:**
- The denoising diffusion process on MNIST or CIFAR-10
- The forward diffusion (adding noise) and reverse denoising (removing noise) processes
- Generate recognizable images from pure noise

**Why this paper:**
Diffusion models are the foundation of modern image generation (DALL-E 2, Stable Diffusion,
Midjourney). This paper made diffusion practical. Implementing it teaches you about noise
schedules, the connection between score matching and denoising, and how to train generative
models that are more stable than GANs.

**Scale:** MNIST is feasible on a single GPU in hours. CIFAR-10 requires more time but is
still manageable. Do not attempt high-resolution generation.

**Key challenges:**
- Understanding the mathematical framework (forward process, reverse process, variational bound)
- Implementing the noise schedule correctly (linear beta schedule)
- The U-Net architecture with time conditioning
- Sampling (the iterative denoising process) — getting this wrong produces garbage
- Training is slow relative to the other options; plan accordingly

### Option D: LoRA (Hu et al., 2021)

**Paper:** "LoRA: Low-Rank Adaptation of Large Language Models"
https://arxiv.org/abs/2106.09685

**What you will reproduce:**
- The LoRA mechanism: freezing pretrained weights and adding trainable low-rank matrices
- Fine-tuning a pretrained model (e.g., GPT-2 small or a BERT variant) on a downstream task
- The comparison between full fine-tuning and LoRA fine-tuning (parameter count vs. performance)

**Why this paper:**
LoRA is one of the most practically important papers of recent years. It makes fine-tuning
large models accessible. Implementing it teaches you about parameter-efficient methods, the
structure of pretrained models, and the surprising effectiveness of low-rank approximations.

**Scale:** GPT-2 small (124M parameters) can be fine-tuned with LoRA on a single GPU.
Use a standard NLP benchmark (e.g., GLUE tasks).

**Key challenges:**
- Understanding where to inject LoRA matrices (which layers, which projections)
- Implementing the low-rank decomposition correctly (A and B matrices, initialization)
- Freezing the right parameters while keeping LoRA parameters trainable
- Comparing fairly against full fine-tuning (same number of training steps, same data)
- The paper tests at scales you may not have access to; adapting to smaller scale requires
  careful choices

### Option E: Vision Transformer (Dosovitskiy et al., 2020)

**Paper:** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
https://arxiv.org/abs/2010.11929

**What you will reproduce:**
- The patch embedding mechanism (splitting images into patches, linear projection)
- The Vision Transformer (ViT) architecture applied to CIFAR-10
- Comparison against a CNN baseline (e.g., ResNet) of similar size

**Why this paper:**
ViT showed that the inductive biases of CNNs (locality, translation equivariance) are not
necessary when you have enough data and compute. Implementing it teaches you about the
flexibility of the transformer architecture and the role of inductive biases.

**Scale:** ViT-Tiny or ViT-Small on CIFAR-10 is feasible on a single GPU. Do not attempt
ViT-Large or ImageNet-scale experiments.

**Key challenges:**
- The patch embedding (reshaping images into sequences of patches)
- Positional embeddings for 2D data (the paper uses learned 1D positional embeddings)
- The [CLS] token mechanism for classification
- ViT underperforms CNNs on small datasets — the paper's strong results require pretraining
  on JFT-300M or ImageNet-21K. At CIFAR-10 scale, you will see weaker results; document this

### Proposing Your Own Paper

If you want to reproduce a different paper, submit a proposal including:
- The paper citation and link
- Why you want to reproduce this paper
- Which experiments you would reproduce
- An estimate of the compute required
- An assessment of whether the implementation is feasible in 40-60 hours

Your proposal must be approved before you begin.

---

## Requirements

### R1: Implement the Core Method from Scratch

You must implement the paper's core method yourself. This means:

- **Do NOT copy the authors' code.** You may look at it after your implementation is complete
  (for debugging), but your code must be written from your understanding of the paper.
- **Do NOT use high-level library implementations of the core method.** For example, if you
  are implementing ResNet, do not use `torchvision.models.resnet18`. You may use PyTorch's
  `nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d`, etc. — the building blocks.
- **You MAY use standard utilities:** data loaders, optimizers, learning rate schedulers,
  standard datasets, logging libraries.
- **You MAY use reference implementations for NON-CORE components.** For example, if you are
  implementing DDPM, you may use a standard U-Net implementation as long as you implement the
  diffusion process yourself. If you are implementing LoRA, you may load a pretrained model
  from HuggingFace.

The line is: YOU implement the paper's CONTRIBUTION. Everything else can be off-the-shelf.

### R2: Document Every Ambiguity

As you implement, maintain a running log of every decision where the paper was ambiguous:

```markdown
## Implementation Decision Log

### Decision 1: [Title]
- **What the paper says:** [Quote or paraphrase]
- **What is ambiguous:** [What exactly is unclear]
- **What I decided:** [Your implementation choice]
- **Why:** [Justification — from appendix, from follow-up paper, from experimentation, etc.]
- **Impact:** [How sensitive is the result to this choice?]
```

Aim for at least 10 documented decisions. If you have fewer than 10, you are not reading the
paper carefully enough.

### R3: Run the Key Experiments

You do not need to reproduce every experiment in the paper. Reproduce the KEY experiments —
the ones that demonstrate the paper's main claims. At minimum:

- Train the model to reasonable performance
- Compare against at least one baseline
- Run at least one ablation (remove or change one component)
- Report results with at least 2 different random seeds

If the paper trains on ImageNet but you only have one GPU, use CIFAR-10 or CIFAR-100. Scale
down the model if needed. The goal is to verify the MECHANISM, not match the exact numbers
at full scale.

### R4: Compare Your Results to the Paper's

Create a detailed comparison table:

```markdown
## Results Comparison

### [Experiment Name]

| Metric | Paper (original scale) | Paper (our scale, if reported) | Our Result | Gap |
|--------|----------------------|-------------------------------|------------|-----|
| | | | | |

### Analysis of Discrepancies

For each significant gap, hypothesize the cause:
1. [Gap description] — likely due to [hypothesis]
2. ...
```

Honest reporting of discrepancies is MORE valuable than matching numbers. If your results are
worse, investigate and document why. Do not cherry-pick seeds or hyperparameters to close
the gap artificially.

### R5: Write a Reproduction Report

Your report should follow this structure:

#### 1. Introduction (0.5 page)
- Which paper you reproduced and why
- Summary of the paper's key contribution

#### 2. Implementation Details (1-2 pages)
- Your implementation approach (architecture, training setup)
- Key decisions where the paper was ambiguous (summary of the decision log)
- Deviations from the paper and justifications

#### 3. Experiments (1-2 pages)
- Experimental setup (dataset, hyperparameters, compute)
- Results tables comparing to the paper
- Ablation study results

#### 4. Analysis (1 page)
- What matched the paper's results?
- What did not match? Why?
- What did you learn that you could not have learned from just reading the paper?

#### 5. Conclusion (0.5 page)
- Key takeaways from the reproduction process
- Recommendations for others who want to reproduce this paper

---

## Deliverables

1. **Code repository** with:
   - Clean, well-documented implementation of the core method
   - Training script with configurable hyperparameters
   - Evaluation script
   - `README.md` with setup instructions and how to reproduce your results
   - All configuration files needed to run your experiments

2. **Implementation decision log** (see R2) — at least 10 documented decisions.

3. **Experiment results** — saved model checkpoints (at least the best one), training logs
   (loss curves, metrics over time), and evaluation outputs.

4. **Reproduction report** (4-6 pages, markdown or PDF) — see R5 for structure.

5. **Training curves** — plots of loss and key metrics over training, for both your
   implementation and the baseline.

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Implementation quality | 30% | Is the code correct, clean, and well-structured? Does it faithfully implement the paper? |
| Decision documentation | 20% | Are ambiguities identified and resolved thoughtfully? Is the reasoning sound? |
| Experimental rigor | 20% | Are experiments fair, reproducible, and properly compared to the paper? |
| Report quality | 20% | Is the analysis insightful? Are discrepancies investigated honestly? |
| Code documentation | 10% | Could someone else understand and run your code? |

### What "excellent" looks like:
- Your implementation matches the paper's results within a small margin, or you provide a
  convincing explanation for the gap.
- Your decision log reveals non-obvious ambiguities that demonstrate deep reading.
- Your ablation study tests something interesting, not just the obvious components.
- Your report contains insights that would be useful to the next person reproducing this paper.
- Your code is clean enough that another apprentice could read it and learn from it.

### What "needs improvement" looks like:
- The model trains but produces significantly worse results with no investigation of why.
- The decision log contains fewer than 10 entries or only documents obvious choices.
- No ablation study, or ablation study only tests trivially obvious components.
- The report describes what you did but not what you learned.
- The code is a tangled notebook with no structure.

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Paper selected. Third-pass reading completed. Implementation plan written. Decision log started. |
| 2 | Core method implemented and tested on toy data. First training run on real data started. |
| 3 | Main experiments completed. Ablation study designed and running. Results comparison table drafted. |
| 4 | Report written. Code cleaned and documented. All deliverables assembled. |

Adjust this timeline based on your paper choice. DDPM and LoRA may require more time for
training. ResNet and Batch Normalization experiments are faster.

---

## Stretch Goals

If you complete the core requirements and want to push further:

1. **Reproduce on the original scale.** If you used CIFAR-10, try ImageNet (if you have the
   compute). If you used a small model, try the full-size version.

2. **Submit to the ML Reproducibility Challenge.** If your reproduction is high quality,
   consider submitting it to the annual ML Reproducibility Challenge
   (https://reproml.org/). This is a real venue for reproduction reports and looks
   excellent on a CV.

3. **Compare implementations.** After completing your implementation, read the authors' code
   (or a popular third-party implementation). Document every difference. Which implementation
   choices were you right about? Which were you wrong about? What did you learn?

4. **Benchmark your implementation.** How does your implementation compare in training speed
   (not just final accuracy) to a reference implementation? Profile it and identify
   bottlenecks.

5. **Write a blog post.** Write a public blog post about your reproduction experience. The ML
   community values reproduction reports, and writing publicly forces you to be precise.

---

## A Note on Failure

Your results will probably not exactly match the paper. This is normal. Papers often benefit
from unreported hyperparameter tuning, favorable random seeds, or computational resources you
do not have. A 2-3% gap on accuracy metrics is common and acceptable.

What matters is not whether you match the numbers, but whether you understand WHY your results
differ. A thoughtful analysis of a 5% gap teaches you more than a lucky exact match.

If your model does not train at all, that is a different problem. Come to office hours. But
do not be discouraged by imperfect results — that is the reality of research engineering.
The ability to diagnose and explain discrepancies is itself a valuable skill that this
assignment is designed to develop.
