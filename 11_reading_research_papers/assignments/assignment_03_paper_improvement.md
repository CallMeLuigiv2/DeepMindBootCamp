# Assignment 3: Paper Improvement — Your First Research Contribution

## Overview

This assignment is the bridge between student and researcher. You will take the paper you
reproduced in Assignment 2, identify a genuine weakness or limitation, propose an improvement,
implement it, test it experimentally, and write it up as a research note.

This is not a hypothetical exercise. You are doing original research. The improvement might be
small. It might not work. That is fine. What matters is that you follow the research process:
identify a gap, form a hypothesis, design an experiment, run it, and report the results
honestly.

At DeepMind, this is how projects start. Someone reads a paper, notices a limitation, and asks
"What if we tried X instead?" The ability to go from that question to a tested answer is the
core skill of a research engineer.

**Estimated time:** 30-40 hours over 2-3 weeks
**Prerequisite:** Assignment 2 must be substantially complete before beginning this assignment.

---

## Phase 1: Identify a Weakness (5-8 hours)

### Step 1: Systematic Weakness Analysis

Return to the paper you reproduced. Read it one more time, now with the explicit goal of
finding something to improve. Consider these categories of weaknesses:

**Architectural limitations:**
- Is there a component that seems unnecessarily complex?
- Is there a component that seems too simple for what it needs to do?
- Could a different building block work better?
- Are there architectural choices that seem arbitrary (unjustified in the paper)?

**Training limitations:**
- Does the training procedure have obvious inefficiencies?
- Could a different optimizer, schedule, or regularization technique help?
- Is the model sensitive to hyperparameters that could be made more robust?
- Are there stability issues during training?

**Data/efficiency limitations:**
- Does the method require excessive data or compute?
- Could the same performance be achieved with a simpler model?
- Is the method inefficient in a specific, measurable way?

**Generalization limitations:**
- Does the method fail on certain types of inputs?
- Is performance brittle to distribution shift?
- Are there domains or tasks where the method underperforms?

**Missing features:**
- Is there a natural extension the authors did not explore?
- Could the method be combined with another technique for better results?
- Is there a theoretical insight that suggests a specific modification?

### Step 2: Select Your Improvement

Your improvement must be:

1. **Specific.** Not "make the model better" but "replace the fixed positional encoding with
   a learned relative positional encoding."
2. **Testable.** You must be able to design an experiment that clearly shows whether your
   change helps, hurts, or makes no difference.
3. **Implementable.** You must be able to code it within the time budget. Do not propose
   something that requires training a model ten times larger than your reproduction.
4. **Justified.** You need a reason (intuitive, theoretical, or empirical) for why this
   change should help. "I wonder what happens if..." is the start. "I hypothesize that
   X because Y" is what you need.

### Step 3: Write Your Hypothesis

Before implementing anything, write:

```
HYPOTHESIS

Observation: [What weakness or limitation did you observe?]

Proposed change: [What specific modification will you make?]

Expected effect: [What do you predict will happen and why?]

How to test: [What experiment will confirm or refute your hypothesis?]

Success criterion: [How will you know if your modification worked?]
```

This is not busywork. Writing the hypothesis before coding prevents you from unconsciously
moving the goalposts when results come in.

---

## Phase 2: Implement the Modification (8-12 hours)

### Implementation Guidelines

1. **Start from your Assignment 2 codebase.** Do not rewrite everything. Your modification
   should be a clear, localized change.

2. **Keep the original method intact.** You need to run both the original and your modification
   in the same experimental setup. Use a configuration flag to switch between them.

3. **Minimize confounding variables.** Change ONE thing at a time. If your modification
   involves two changes (e.g., a new layer AND a new loss term), test each independently
   first, then test them together.

4. **Document your implementation clearly.**
   ```python
   # MODIFICATION: Replace fixed sinusoidal positional encoding with
   # learned relative positional encoding.
   #
   # Hypothesis: Learned relative positions will improve performance on
   # tasks requiring strong position sensitivity because [reason].
   #
   # See: Shaw et al., 2018, "Self-Attention with Relative Position
   # Representations" for theoretical motivation.
   ```

5. **Add assertions and tests.** Your modification should not silently break the model.
   Add shape checks, gradient flow checks, and sanity tests.

### Example Modifications (for inspiration, not prescription)

These examples illustrate the level of specificity expected. Your modification should be
similarly concrete.

**If you reproduced ResNet:**
- Replace ReLU with GELU or SiLU and measure the effect on training speed and final accuracy
- Add Squeeze-and-Excitation blocks to the residual branches
- Replace batch normalization with group normalization or layer normalization
- Test the effect of stochastic depth (randomly dropping residual blocks during training)
- Implement and test pre-activation residual blocks (He et al., 2016)

**If you reproduced Batch Normalization:**
- Compare batch norm against layer norm, group norm, and instance norm under the same
  experimental conditions
- Investigate the effect of batch size on batch norm performance (what happens at very
  small batch sizes?)
- Test whether the momentum parameter for running statistics significantly affects results
- Implement "ghost batch normalization" and test whether virtual batch sizes help

**If you reproduced DDPM:**
- Test a different noise schedule (cosine schedule from Nichol & Dhariwal, 2021)
- Implement classifier-free guidance and measure the quality/diversity tradeoff
- Test the effect of fewer diffusion steps at inference time (faster sampling)
- Replace the U-Net backbone with a simpler architecture and measure the quality loss

**If you reproduced LoRA:**
- Test different ranks (r) and find the Pareto frontier of parameters vs. performance
- Apply LoRA to different sets of weight matrices (attention only vs. all linear layers)
- Compare LoRA with other parameter-efficient methods (adapters, prefix tuning)
- Test whether LoRA initialization (zero for B, random for A) matters

**If you reproduced Vision Transformer:**
- Test different patch sizes and their effect on the accuracy/compute tradeoff
- Replace learned positional embeddings with 2D sinusoidal encodings
- Add convolutional stems (hybrid approach) and measure the effect on small-dataset performance
- Test the effect of the [CLS] token vs. global average pooling

---

## Phase 3: Run Experiments (8-12 hours)

### Experimental Design

Your experiments must answer one clear question: **Does your modification improve upon the
original method?**

At minimum, run the following:

#### Experiment 1: Head-to-Head Comparison

Train both the original method and your modification under identical conditions:
- Same dataset, same preprocessing
- Same optimizer, same learning rate schedule
- Same number of training steps
- Same random seeds (run at least 3 seeds for each)
- Same compute budget (if your modification is faster or slower, note this)

Report:
- Final performance (primary metric) for both methods, with mean and standard deviation
  across seeds
- Training curves (loss and primary metric over time) for both methods
- Wall-clock training time for both methods

#### Experiment 2: Ablation or Analysis

Depending on your modification, run one of:
- **Ablation:** If your modification has multiple components, test each independently.
- **Sensitivity analysis:** Test your modification with different hyperparameter values
  (e.g., if you added a new hyperparameter, sweep over it).
- **Failure analysis:** Find cases where your modification helps and cases where it hurts.
  Understanding the failure modes is as valuable as showing improvement.

#### Experiment 3: Diagnostic

Run at least one experiment that helps you understand WHY your modification helps (or doesn't):
- Visualize learned representations (t-SNE, PCA) before and after your change
- Measure gradient magnitudes through the modified components
- Analyze the loss landscape (if applicable)
- Compare convergence speed, not just final performance

### Recording Results

Maintain a lab notebook (markdown file) with:
- Date and time of each experiment
- Exact configuration used (hyperparameters, random seed, etc.)
- Results (metrics, training time, observations)
- Your interpretation

This is not optional. Research without records is not research.

---

## Phase 4: Write the Research Note (6-8 hours)

Write a short research note (approximately 4 pages, not counting references) in the style of
a workshop paper. This is the format used at venues like NeurIPS workshops, where researchers
present early-stage or focused results.

### Structure

#### Title
Choose a specific, informative title. Not "Improving ResNet" but "Stochastic Depth Improves
ResNet Training Efficiency on Small Datasets."

#### Abstract (150-200 words)
- One sentence: problem context
- One sentence: what you observed (the limitation)
- One sentence: what you proposed
- One sentence: key result
- One sentence: significance

#### 1. Introduction (0.5-1 page)

**Paragraph 1:** Context. What is the original paper about? Why does it matter?

**Paragraph 2:** Motivation. What limitation did you identify? Why does it matter? Be specific
about the problem.

**Paragraph 3:** Contribution. What did you do? State it clearly. "We propose [modification]
to address [limitation]. We show that [result]."

**Paragraph 4:** Summary of results. One paragraph with the key numbers.

#### 2. Background (0.5-1 page)

Briefly describe the original method. Assume the reader has read the original paper but needs
a refresher. Include only the details relevant to understanding your modification. This should
be shorter than a full method section because you are building on known work.

Include the key equations from the original paper that your modification changes.

#### 3. Method (0.5-1 page)

Describe your modification precisely. For each change:
- What exactly did you change?
- Why? (Theoretical or intuitive motivation)
- How? (Equations, pseudocode, or architectural diagrams)

If you changed an equation, show both the original and your version side by side.

Use notation consistent with the original paper so the reader can easily compare.

#### 4. Experiments (1-1.5 pages)

**Setup:** Dataset, model configuration, training details, compute used. Everything needed
to reproduce your experiments.

**Results:** Tables and figures comparing your modification to the original. Include:
- Head-to-head comparison table with error bars
- Training curves
- Ablation or sensitivity results
- Diagnostic analysis

**Discussion:** Interpret your results. Did the modification help? By how much? When does
it help and when does it not? Why?

Be honest. If the modification did not help, say so and analyze why. A well-analyzed negative
result is more valuable than a poorly understood positive result.

#### 5. Conclusion (0.25 page)

- What did you find?
- What does it mean?
- What would you try next?

#### References

Cite all papers you reference. Use a consistent citation format.

---

## Deliverables

1. **Hypothesis document** — Your written hypothesis from Phase 1 (before implementation).

2. **Code** — Your modified codebase with:
   - Clear separation between the original method and your modification
   - A configuration flag to switch between them
   - Scripts to reproduce all experiments
   - A README explaining how to run everything

3. **Lab notebook** — Your running log of experiments, decisions, and observations.

4. **Research note** — The 4-page write-up described in Phase 4.

5. **All experimental results** — Training logs, saved metrics, plots.

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Problem identification | 15% | Did you identify a genuine, non-trivial limitation? |
| Hypothesis quality | 10% | Is the hypothesis specific, testable, and well-motivated? |
| Implementation quality | 20% | Is the modification correctly implemented? Is the code clean? |
| Experimental rigor | 25% | Are experiments well-designed, fair, and reproducible? Are results properly reported with error bars? |
| Write-up quality | 20% | Is the research note clear, well-structured, and professionally written? |
| Intellectual honesty | 10% | Are negative results reported honestly? Are limitations discussed? |

### What "excellent" looks like:
- You identified a non-obvious limitation through careful analysis of the original paper
  and your reproduction results.
- Your hypothesis has clear theoretical or empirical motivation.
- Your experiments are controlled (same seeds, same compute), include ablations, and report
  error bars.
- Your write-up could be submitted to a workshop and would receive constructive reviews
  (even if it would not be accepted — that is a high bar for a first research note).
- You discuss what worked, what did not, and what you would try next.

### What "needs improvement" looks like:
- The "limitation" is trivially obvious (e.g., "the model is slow" with no specific analysis).
- The modification has no clear motivation ("I tried random thing X").
- Experiments use a single random seed with no error bars.
- The write-up lacks critical analysis — only reports positive results, ignores negative ones.
- The code is tangled and not reproducible.

---

## Stretch Goals

1. **Positive result.** If your modification genuinely improves the original method, consider
   expanding the experiments and write-up to a full short paper (6-8 pages). Discuss with
   your mentor whether it merits submission to a workshop.

2. **Multiple modifications.** If time allows, test 2-3 different modifications and compare
   them. Which helps most? Do they combine well?

3. **Cross-paper comparison.** If another apprentice reproduced a different paper, collaborate:
   try your modification idea on their paper and vice versa. Does the improvement generalize?

4. **Public release.** Clean up your code, write clear documentation, and release it on GitHub
   as a reproduction report with your extension. Tag it for the ML Reproducibility Challenge
   or Papers With Code.

5. **Presentation.** Prepare a 10-minute presentation of your research note and present it
   to the group. This is standard practice at research labs — every project gets a
   presentation.

---

## A Note on This Assignment

This is your first taste of original research. It will feel different from everything else
in this course. There is no solution manual. Your modification might not work. Your results
might be confusing. That ambiguity is the reality of research.

The purpose of this assignment is not to produce a breakthrough result. It is to teach you the
PROCESS: observe, hypothesize, implement, test, analyze, write. Once you have gone through this
cycle once, you know what research feels like. You know the difference between "I read about
method X" and "I tried to improve method X and here is what happened."

That experience — the hands-on knowledge of what it takes to go from an idea to an experimental
result — is what separates someone who can discuss papers from someone who can do research.

At DeepMind, this is the baseline expectation. Every engineer, every researcher, every intern
is expected to run this cycle. Most ideas do not work. That is normal. The skill is in knowing
how to learn from each cycle, good result or bad, and apply that learning to the next one.

This assignment is the beginning of that practice. Take it seriously.
