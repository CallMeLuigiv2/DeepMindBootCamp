# Module 12 -- Capstone Project: Evaluation Rubric

This document defines the detailed evaluation criteria for the capstone project. The rubric is designed to reward the qualities that matter in real ML research: technical depth, engineering discipline, scientific rigor, clear communication, and original thinking. Read this before you start your project, not after you finish it.

Total points: **100**

---

## Technical Depth (30 points)

This category evaluates the sophistication of your approach and the depth of your understanding. We are looking for evidence that you understand *why* your method works, not just *that* it works.

### 25-30 points: Exceptional

The approach is technically sophisticated and demonstrates deep understanding of the underlying methods. The apprentice goes beyond what was taught in the course -- perhaps combining techniques from multiple modules in a novel way, implementing a recent paper from scratch, or introducing a well-motivated modification to an existing method. Architecture and design choices are clearly justified with reference to the relevant theory or literature. The apprentice can explain every component of their system and why it is there.

**Example**: Implementing a class-conditional diffusion model with both classifier and classifier-free guidance, including a custom noise schedule, and providing a clear mathematical derivation of the training objective. The report connects the method to score matching and explains why the reparameterization trick is necessary for training.

### 20-24 points: Strong

Solid application of techniques from the course. Architecture choices are reasonable and well-justified. The apprentice demonstrates understanding of the core concepts and can explain their design decisions. The method is correct and appropriate for the problem.

**Example**: Implementing a standard Vision Transformer for image classification with proper positional encoding, multi-head attention, and layer normalization. The apprentice understands why each component is needed and can explain the computational cost of self-attention and how it scales with sequence length.

### 15-19 points: Adequate

The implementation is correct but the approach is straightforward, with limited evidence of deeper understanding. The apprentice uses techniques from the course appropriately but does not go beyond the basics. Design choices are not always justified.

**Example**: Fine-tuning a pretrained BERT model on a text classification task using standard hyperparameters from the original paper, without meaningful analysis of why those hyperparameters are appropriate or exploration of alternatives.

### 10-14 points: Below Expectations

Significant gaps in technical understanding. The approach has clear flaws (e.g., data leakage, inappropriate loss function, fundamental misunderstanding of the method). The apprentice cannot clearly explain why their approach should work.

**Example**: Training a model on the test set, using MSE loss for a classification task without justification, or implementing a Transformer without positional encoding and not noticing the problem.

### Below 10 points: Unacceptable

The approach is fundamentally flawed or demonstrates a lack of understanding of basic ML concepts.

---

## Implementation Quality (25 points)

This category evaluates your code as an engineering artifact. In research, bad code leads to bad science -- bugs masquerade as results, irreproducible experiments waste everyone's time, and messy code is impossible to debug or extend.

### 20-25 points: Exceptional

The codebase is clean, modular, and well-organized. Functions have clear docstrings and type hints. Configuration is managed through config files, not hardcoded values. The code is fully reproducible: given the same config and random seed, the same results are obtained. Key components have unit tests (at minimum, shape checks for the model and sanity checks for the data pipeline). PyTorch is used efficiently (proper use of `nn.Module`, `DataLoader`, mixed precision where appropriate, no unnecessary CPU-GPU transfers). The git history is clean and tells a coherent story.

**Example of good code**:
```python
class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers and a skip connection.

    If input and output channels differ, a 1x1 convolution is used
    to match dimensions for the skip connection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution. Default: 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out
```

**Example of what exceptional reproducibility looks like**:
```yaml
# configs/experiment_1.yaml
model:
  architecture: resnet18
  num_classes: 10
  dropout: 0.1

training:
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  batch_size: 128
  epochs: 100
  lr_schedule: cosine
  warmup_epochs: 5

data:
  dataset: cifar10
  augmentation: [random_crop, horizontal_flip]
  num_workers: 4

reproducibility:
  seed: 42
  deterministic: true
```

### 15-19 points: Good

Working code that is reasonably organized. Most functions are documented. The code can be reproduced with some effort (the apprentice may need to explain a few steps). No major engineering anti-patterns. The repository has a clear structure.

### 10-14 points: Acceptable

The code works and produces the reported results, but it is messy or hard to follow. Functions are long and do too many things. Magic numbers appear in the code. Reproduction requires significant effort or guesswork. No tests. The git history is a single large commit or a series of "fix" commits.

**Example of poor code**:
```python
def train(m, d, e=100, lr=0.001):
    o = torch.optim.Adam(m.parameters(), lr)
    for i in range(e):
        for x, y in d:
            x = x.cuda()
            y = y.cuda()
            p = m(x)
            l = F.cross_entropy(p, y)
            o.zero_grad()
            l.backward()
            o.step()
        print(l.item())  # only prints loss for last batch
```
This code has no docstring, single-letter variable names, no validation, prints only the last batch loss, no gradient clipping, no learning rate scheduling, no checkpointing, and no seed setting.

### Below 10 points: Poor

The code does not run, produces different results from those reported, is largely copied from tutorials without attribution, or is so disorganized that it cannot be reviewed.

---

## Experimental Rigor (20 points)

This category evaluates whether your experiments actually support your conclusions. Good experimental practice is the difference between science and anecdote.

### 16-20 points: Excellent

The experimental setup is thorough and scientifically sound:

- **Baselines**: At least two meaningful baselines are included. One should be a simple baseline (e.g., random, majority class, linear model) to establish a floor. One should be a reasonable non-trivial baseline (e.g., published result, standard architecture). Baselines are trained under the same conditions as the proposed method for fair comparison.

- **Ablation studies**: Systematic ablations that isolate the contribution of individual components. Each ablation changes exactly one thing and reports its effect. The ablations answer the question: "What parts of my system actually matter?"

- **Statistical validity**: Results are reported with confidence intervals or standard deviations over multiple runs (at least 3 seeds for key experiments). The apprentice acknowledges when differences are within noise.

- **Honest reporting**: Negative results are reported alongside positive ones. Failed experiments are documented with analysis of why they failed. There is no cherry-picking of results.

- **Proper evaluation**: The test set is used exactly once for final evaluation. All development decisions (hyperparameter tuning, model selection, early stopping) are made using the validation set only.

**Example of a good results table**:

| Method                      | CIFAR-10 Acc (%) | CIFAR-100 Acc (%) | Params (M) |
|-----------------------------|------------------:|-------------------:|------------:|
| Random baseline             |        10.0       |         1.0        |     --      |
| Linear classifier on pixels |        39.2       |        11.4        |    0.03     |
| ResNet-18 (our impl.)      |  93.1 +/- 0.2     |   75.8 +/- 0.3    |    11.2     |
| Our method                  |  **94.3 +/- 0.1** | **77.2 +/- 0.2**  |    11.4     |
| Our method (no augmentation)|  91.7 +/- 0.3     |   72.1 +/- 0.4    |    11.4     |
| Our method (no skip conn.)  |  92.4 +/- 0.2     |   74.5 +/- 0.3    |    10.8     |
| Our method (no warmup)      |  93.8 +/- 0.2     |   76.4 +/- 0.3    |    11.4     |

This table tells a clear story: the method improves over the baseline, augmentation is the most important factor, skip connections matter, and learning rate warmup has a small but consistent effect.

### 12-15 points: Good

Good baselines and evaluation metrics. Some ablation studies are included. Results are reported clearly but may lack confidence intervals or may not cover all important ablations. Evaluation is honest but could be more thorough.

### 8-11 points: Adequate

Basic experiments with limited comparison. Only one baseline, no ablations, or ablations that are superficial (e.g., only varying learning rate). Results are reported in a way that makes it hard to draw conclusions.

### 4-7 points: Below Expectations

No meaningful baselines. Results are cherry-picked or misleading. The test set was used for hyperparameter tuning. Claims are not supported by the evidence presented.

**Example of poor experimental practice**:
- Reporting the best result from a single run without noting the variance.
- Comparing your model with heavy augmentation against a baseline with no augmentation.
- Tuning hyperparameters on the test set and reporting test accuracy.
- Claiming "our method achieves state-of-the-art" when only compared against a trivial baseline.

### Below 4 points: Unacceptable

No experiments, fabricated results, or experiments that do not evaluate the claims made in the report.

---

## Writing and Presentation (15 points)

This category evaluates your ability to communicate technical work clearly. The best method in the world is worthless if nobody can understand what you did.

### 12-15 points: Excellent

**Report**:
- The writing is clear, concise, and well-structured.
- Every section of the required format is present and substantive.
- Figures and tables are well-designed with clear labels, captions, and legends. They are referenced in the text and contribute to the narrative.
- The related work section demonstrates genuine engagement with the literature, not just a list of summaries.
- Technical writing conventions are followed: equations are numbered and referenced, abbreviations are defined on first use, the paper can be understood by someone with general ML knowledge.

**Example of a good figure caption**: "Figure 3: Test accuracy on CIFAR-100 as a function of training epochs for ResNet-18 (blue), our method (orange), and our method without data augmentation (green dashed). Shaded regions show +/- 1 standard deviation over 5 runs. Our method converges faster and to a higher final accuracy, with data augmentation accounting for approximately 60% of the improvement."

**Example of a poor figure caption**: "Figure 3: Results."

**Presentation**:
- Engaging and well-paced. Stays within the time limit.
- Slides are visual, not walls of text.
- Includes a live demo that works.
- The apprentice can answer questions thoughtfully, saying "I don't know, but I would investigate it by..." when appropriate rather than guessing.

### 9-11 points: Good

The report covers all required sections and is generally clear. Some figures could be improved. The presentation is competent but may lack engagement or run over time. Questions are answered adequately.

### 6-8 points: Adequate

The report is present but missing sections, contains unclear explanations, or has poorly designed figures. The presentation is hard to follow or significantly over/under time. The apprentice struggles with questions.

### 3-5 points: Below Expectations

The report is incomplete or poorly written to the point that the work cannot be properly evaluated. The presentation is unprepared.

### Below 3 points: Unacceptable

No report, no presentation, or both are of such low quality that they do not communicate anything meaningful.

---

## Originality (10 points)

This category rewards independent thinking. You do not need to solve an open problem. But you should bring something of your own to the project -- a new perspective, an unexpected experiment, a creative application, or an insight that is not simply restating what a paper already said.

### 8-10 points: Genuine Creative Contribution

The project contains a clearly original element. This could be:
- A novel modification to an existing method with clear motivation and evaluation.
- An unexpected finding that emerges from careful experimentation, with thoughtful analysis.
- A creative application of a known technique to a new problem domain.
- An original analysis or visualization that reveals something not obvious from existing work.

**Example**: While implementing knowledge distillation, the apprentice discovers that distillation from a poorly calibrated teacher actually *hurts* performance on underrepresented classes, and designs an experiment to isolate and explain this effect. This is a genuine insight, even if the overall project is a standard distillation experiment.

### 5-7 points: Some Original Thinking

The project has elements of original thinking in the approach or analysis, but the core is a standard implementation. The apprentice makes thoughtful observations and designs targeted experiments to investigate them.

**Example**: Implementing a standard ViT vs. CNN comparison, but including a novel analysis of how the two architectures respond differently to adversarial perturbations, with a clear hypothesis and experiment.

### 3-4 points: Minor Variations

The project is a standard implementation with minor parameter variations. There is no meaningful original contribution, but the work is competently executed.

**Example**: Implementing GPT on Shakespeare and varying the model size, without deeper analysis of what the model learns or creative extensions to the approach.

### Below 3 points: No Original Thought

The project is a pure reproduction of a tutorial or existing implementation with no evidence of independent thinking. If the code closely follows a specific tutorial, this must be disclosed, and the apprentice must clearly describe what they added beyond the tutorial.

---

## Scoring Summary

| Category               | Points | Weight |
|------------------------|-------:|-------:|
| Technical Depth        |    30  |   30%  |
| Implementation Quality |    25  |   25%  |
| Experimental Rigor     |    20  |   20%  |
| Writing & Presentation |    15  |   15%  |
| Originality            |    10  |   10%  |
| **Total**              |**100** | **100%**|

---

## Grade Interpretation

### 90-100: DeepMind Internship Ready

Exceptional work across all categories. The project demonstrates the technical depth, engineering discipline, experimental rigor, communication skills, and independent thinking that would be expected of a strong ML research intern. The apprentice is ready to contribute to a research team.

This does not mean the project achieved record-breaking results. It means the *process* was exemplary: a well-defined problem, a thoughtful approach, a clean implementation, rigorous experiments, honest analysis, and clear communication. A well-executed project with modest results in this score range is worth far more than a sloppy project that happens to achieve a high accuracy number.

### 80-89: Strong ML Engineer

Very good work with strength across most categories. The project demonstrates solid technical understanding and good engineering practices. The apprentice is ready for ML engineering roles and could contribute meaningfully to research teams with some additional mentorship. Minor gaps in one or two areas (e.g., ablations could be more thorough, or the writing could be tighter) prevent a top score.

### 70-79: Solid Foundation

Good work that demonstrates competence in the core skills. The apprentice understands the fundamentals and can implement and evaluate ML systems. The project may lack depth in some areas -- perhaps the analysis is superficial, or the code could be better organized, or the experimental comparison is not entirely fair. The apprentice is ready for ML engineering roles and should continue building research skills.

### 60-69: Needs More Practice

The project shows effort but has significant gaps. Common issues at this level: the implementation works but is buggy or unreproducible, the experiments lack baselines or ablations, the report is incomplete or unclear, or the technical approach has notable flaws. The apprentice should review the weak areas and consider revisiting the relevant course modules before attempting more advanced work.

### Below 60: Revisit Core Modules

The project has fundamental problems that indicate insufficient mastery of the course material. This could mean the code does not work, the experimental methodology is flawed (e.g., evaluating on training data), the report is missing essential sections, or the technical approach reveals basic misunderstandings. The apprentice should revisit the core modules, particularly those relevant to their project domain, and attempt a simpler project before tackling a full capstone.

---

## Common Pitfalls to Avoid

These are the mistakes we see most often. Avoiding them will not guarantee a high score, but committing them will guarantee a lower one.

1. **Scope creep**: Starting too ambitiously and running out of time. A complete, well-analyzed small project scores higher than an incomplete ambitious one.

2. **No baselines**: You cannot claim your method is good without showing what it is better than. Even a random baseline has value.

3. **Evaluating on training data**: This is perhaps the most common and most damaging mistake. Always hold out a test set. Use it once.

4. **Ignoring failures**: If an experiment did not work, do not delete it. Document it. Explain what you tried and why you think it failed. A well-analyzed failure is a contribution.

5. **Copy-pasting code without understanding**: If you use code from a tutorial or library, cite it and explain what it does. If you cannot explain a line of code in your submission, you should not have included it.

6. **Reporting only the best run**: One good run out of ten is noise, not a result. Report means and standard deviations.

7. **Neglecting the report**: The code is not the deliverable. The report is. Many apprentices spend 95% of their time on code and 5% on the report. This ratio should be closer to 70/30.

8. **No figures**: A report without figures is almost always worse than one with figures. Training curves, architecture diagrams, sample outputs, attention maps, confusion matrices -- use them.

9. **Presenting someone else's results as your own**: This is academic dishonesty and will result in a failing grade. Always cite your sources. If you could not reproduce a baseline and are using published numbers, say so explicitly.

10. **Not starting the write-up early enough**: Begin your report outline in Week 1. Fill in sections as you go. Do not leave the entire report to the last two days.

---

## A Note on What Matters

The rubric assigns points, but points are a proxy for something harder to quantify: has this apprentice developed the judgment, skills, and habits needed to do good ML work independently?

A project that earns 85 points because the apprentice chose a challenging problem, fought through difficulties, documented what did not work, and delivered an honest, well-analyzed result is more impressive than a project that earns 90 points by executing a safe, well-trodden path with no risk.

Choose a project that teaches you something. Then do it properly.
