# Assignment 3: Systematic Regularization Experiments

## Overview

You will conduct a rigorous experimental study of regularization techniques. Starting with a
network that deliberately overfits, you will systematically apply each technique, measure its
effect, and analyze why some techniques work better than others.

This is how research in deep learning works: formulate a hypothesis, design a controlled
experiment, measure results, and explain what happened.

**Estimated time**: 10-14 hours.

---

## Experimental Setup

### Dataset

Use CIFAR-10, but deliberately create an overfitting scenario:

1. Take only **2,000 training samples** (200 per class).
2. Keep the full **10,000 test samples** for evaluation.
3. Create a validation set of **500 samples** from the training set (so 1,500 train, 500 val).

This small training set guarantees that an unregularized network will severely overfit.

### Base Model

Use the following architecture as your baseline (deliberately large for this dataset size):

```python
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)
```

This model has approximately 3.5 million parameters for 1,500 training samples.
It will overfit dramatically.

### Training Configuration (Baseline)

- Optimizer: Adam, lr=1e-3
- Batch size: 64
- Epochs: 200
- No regularization, no data augmentation

### Metrics to Track

For every experiment, record:
1. Training loss curve (every epoch)
2. Validation loss curve (every epoch)
3. Training accuracy curve (every epoch)
4. Validation accuracy curve (every epoch)
5. Final test accuracy
6. Best validation accuracy (and the epoch it occurred)
7. Gap between final training accuracy and test accuracy (overfitting measure)

---

## Experiments

### Experiment 0: Baseline (No Regularization)

Train the base model with no regularization. This is your control.

**Expected result**: Training accuracy near 100%, test accuracy around 35-45%.

Record all metrics. This establishes the overfitting gap that all other experiments aim to
reduce.

### Experiment 1: L2 Regularization (Weight Decay)

Train the base model with L2 regularization at the following strengths:

| Run | Weight decay (lambda) |
|-----|----------------------|
| 1a | $10^{-5}$ |
| 1b | $10^{-4}$ |
| 1c | $10^{-3}$ |
| 1d | $10^{-2}$ |
| 1e | $10^{-1}$ |

Use AdamW (NOT Adam with weight_decay parameter, as they differ — explain why in your
analysis).

**Questions to answer**:
- At what $\lambda$ does test accuracy peak?
- What happens to training accuracy as $\lambda$ increases?
- Plot weight magnitude distributions for each $\lambda$.

### Experiment 2: Dropout

Add dropout after each ReLU activation:

| Run | Dropout rate |
|-----|-------------|
| 2a | 0.1 |
| 2b | 0.2 |
| 2c | 0.3 |
| 2d | 0.5 |
| 2e | 0.7 |

**Questions to answer**:
- At what dropout rate does test accuracy peak?
- How does dropout affect training speed (epochs to reach a given training loss)?
- Verify that turning off dropout at test time matters: compare test accuracy with and
  without `model.eval()`.

### Experiment 3: Batch Normalization

Add BatchNorm before each ReLU activation:

```python
class BNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)
```

**Questions to answer**:
- Does BatchNorm reduce overfitting, or does it primarily affect training speed?
- Compare the loss landscape smoothness: plot training loss for different learning rates
  with and without BatchNorm.
- Does BatchNorm change the optimal learning rate?

### Experiment 4: Data Augmentation

Apply data augmentation of increasing aggressiveness:

| Run | Augmentation |
|-----|-------------|
| 4a | Random horizontal flip only |
| 4b | Flip + random crop (padding=4) |
| 4c | Flip + crop + color jitter (brightness=0.2, contrast=0.2) |
| 4d | Flip + crop + color jitter + random rotation (15 deg) |
| 4e | Flip + crop + color jitter + rotation + Cutout (16x16) |

Use `torchvision.transforms` for all augmentations. Implement Cutout yourself if it is
not available.

**Questions to answer**:
- Which individual augmentation has the biggest effect?
- Does increasing augmentation always help, or is there a point of diminishing returns?
- How does data augmentation change the effective training set size?

### Experiment 5: Early Stopping

Using the baseline (no regularization) model:

1. Train for 200 epochs and record validation loss every epoch.
2. Implement early stopping with patience values: 5, 10, 20, 50.
3. Report the epoch at which training stops and the test accuracy at that point.

**Questions to answer**:
- How much does early stopping improve test accuracy over training for the full 200 epochs?
- What patience value works best?
- Is early stopping equivalent to other forms of regularization, or does it do something
  different?

### Experiment 6: Combinations

Now combine techniques. Test at least the following combinations:

| Run | Combination |
|-----|------------|
| 6a | Best L2 + best dropout rate |
| 6b | Best L2 + BatchNorm |
| 6c | Best dropout + data augmentation (4c) |
| 6d | BatchNorm + data augmentation (4c) + L2 |
| 6e | All: BatchNorm + data augmentation (4c) + L2 + early stopping |
| 6f | Your best combination (justify your choices) |

**Questions to answer**:
- Do regularization techniques combine additively, or is there redundancy?
- Does the optimal strength of one technique change when combined with another?
- What is the best test accuracy you can achieve?

---

## Required Visualizations

Create publication-quality plots. This means: labeled axes, legends, consistent colors,
appropriate font sizes, and no default matplotlib styling. Use a clean style.

### Plot 1: Overfitting Gap

A bar chart showing (training accuracy - test accuracy) for: baseline, best L2, best dropout,
BatchNorm, best augmentation, best combination.

### Plot 2: Learning Curves Comparison

A 2x2 grid of subplots:
- Top left: Training loss for baseline, L2, dropout, BatchNorm
- Top right: Validation loss for the same
- Bottom left: Training accuracy for the same
- Bottom right: Validation accuracy for the same

### Plot 3: L2 Regularization Sweep

Test accuracy vs $\lambda$ (log scale x-axis). Mark the optimal $\lambda$.

### Plot 4: Dropout Rate Sweep

Test accuracy vs dropout rate. Mark the optimal rate.

### Plot 5: Data Augmentation Progression

Bar chart of test accuracy for each augmentation level (4a through 4e).

### Plot 6: The Final Comparison

A comprehensive bar chart showing test accuracy for ALL experiments. Sorted by test accuracy.
Include error bars if you run multiple seeds (recommended but not required).

---

## Written Analysis

Write a 2-page analysis (approximately 800-1200 words) addressing:

1. **Baseline characterization**: How severe is the overfitting? What does the loss curve
   look like?

2. **Individual technique comparison**: Rank the techniques by effectiveness. Explain why
   you think the ranking is what it is for this specific setup (small dataset, large MLP,
   CIFAR-10).

3. **Interaction effects**: When you combined techniques, what happened? Were there
   surprises? Did any combination perform worse than its components individually?

4. **The role of data augmentation**: Data augmentation was likely the most effective single
   technique. Why? Connect to the idea that augmentation increases the effective dataset
   size and encodes prior knowledge about the task.

5. **BatchNorm as regularizer**: Based on your experiments, does BatchNorm primarily
   regularize or primarily help optimization? Support your answer with your data.

6. **Practical recommendations**: If someone came to you with a new problem and limited
   data, what regularization strategy would you recommend, in what order? Justify with
   your experimental findings.

7. **Limitations**: What are the limitations of this experimental setup? How might the
   results differ with a different architecture (CNN vs MLP), different dataset, or
   different data scale?

---

## Deliverables

1. A Jupyter notebook containing:
   - All experiment code (well-organized with clear section headers).
   - All 6+ required plots (publication quality).
   - The written analysis.
   - A summary table of all experimental results.
2. The notebook must run end-to-end. Use random seeds for reproducibility:
   `torch.manual_seed(42)` and `np.random.seed(42)`.
3. All training logs should be saved (you may use CSV files or pandas DataFrames).

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Experimental rigor | 25% | All experiments run correctly. Metrics are tracked properly. Fair comparisons (same epochs, same base model, same data splits). |
| Visualizations | 20% | All required plots are present. Publication quality: labeled axes, legends, clean styling. Plots convey the key findings. |
| Written analysis | 25% | All 7 analysis questions are addressed. Analysis is supported by experimental evidence. Recommendations are justified. |
| Code quality | 15% | Clean, modular code. Experiment configurations are clearly defined. Easy to modify and rerun. |
| Depth of investigation | 15% | Goes beyond the minimum. Explores additional hypotheses. Tries additional combinations. Reports error bars or multiple seeds. |

---

## Stretch Goals

1. **Learning rate interaction**: For each regularization technique, find the optimal learning
   rate. Does regularization change the optimal LR? Plot the interaction.

2. **Spatial dropout**: Replace standard dropout with spatial dropout (for a CNN version).
   Compare the two.

3. **Mixup and CutMix**: Implement these advanced augmentation techniques. Compare to
   standard augmentation.

4. **Label smoothing**: Implement label smoothing (replace hard labels with soft labels:
   $y = (1 - \epsilon) \cdot y_{\text{hard}} + \epsilon / K$). Test $\epsilon \in \{0.01, 0.05, 0.1, 0.2\}$.

5. **Visualization of learned representations**: Use t-SNE or UMAP to visualize the
   penultimate layer representations for: baseline (overfit), best regularized model.
   Show that regularization produces better-structured representations.

6. **Scale experiment**: Repeat the comparison with 5,000 and 20,000 training samples.
   How does the relative benefit of regularization change with dataset size?

7. **CNN comparison**: Replace the MLP with a small CNN. Do the same regularization
   experiments. Does the ranking of techniques change?

---

## Tips

- **Reproducibility**: Set ALL random seeds (PyTorch, NumPy, CUDA). Use `torch.backends.cudnn.deterministic = True` if using GPU.
- **Training time**: 200 epochs on this small dataset should take 1-3 minutes per experiment
  on a modern GPU, or 5-15 minutes on CPU. Plan your time accordingly.
- **Tracking experiments**: Consider using a dictionary or dataclass to store experiment
  configs and results. This makes the final comparison table easy to generate.
- **Plotting**: Use `matplotlib.pyplot.style.use('seaborn-v0_8-whitegrid')` or a similar
  clean style. Set figure size to at least (10, 6) for readability.
- **Weight decay in Adam vs AdamW**: In Adam, weight decay is applied to the gradient
  (L2 regularization on the loss). In AdamW, weight decay is applied directly to the
  weights (decoupled weight decay). They are not equivalent because Adam's adaptive
  learning rate scales the gradient-based L2 term inconsistently. Always use AdamW for
  weight decay with adaptive optimizers.
