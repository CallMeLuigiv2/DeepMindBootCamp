# Assignment 2: Build the Landmark Architectures

## Overview

You will implement three historically significant CNN architectures from scratch, train them,
and analyze the results. This is not an exercise in copying code from the internet. You will
build each architecture by reading the original paper specifications, understanding every
design choice, counting every parameter, and measuring how each innovation affects performance.

By the end of this assignment, you will have a visceral understanding of why depth matters,
why skip connections were the most important CNN innovation, and how architectural choices
translate into measurable differences in training dynamics and accuracy.

**Estimated time:** 12-18 hours

---

## Part 1: LeNet-5 on MNIST

### Architecture Specification

Implement LeNet-5 following the original paper (LeCun et al., 1998), with minor modernizations:

```
Input: (1, 32, 32) — MNIST images are 28x28; pad to 32x32

Layer 1: Conv2d(1, 6, kernel_size=5)         -> (6, 28, 28)
         ReLU (original used sigmoid/tanh; use ReLU for comparison fairness)
         AvgPool2d(kernel_size=2, stride=2)   -> (6, 14, 14)

Layer 2: Conv2d(6, 16, kernel_size=5)        -> (16, 10, 10)
         ReLU
         AvgPool2d(kernel_size=2, stride=2)   -> (16, 5, 5)

Layer 3: Flatten                             -> (400,)
         Linear(400, 120)
         ReLU

Layer 4: Linear(120, 84)
         ReLU

Layer 5: Linear(84, 10)
```

### Requirements

1. **Implement the architecture as a PyTorch nn.Module.** Do not use torchvision models.

2. **Count parameters manually.** Fill in this table and verify against
   `sum(p.numel() for p in model.parameters())`:

   | Layer | Calculation | Parameters |
   |-------|------------|------------|
   | Conv1 | (5*5*1 + 1) * 6 | 156 |
   | Conv2 | (5*5*6 + 1) * 16 | 2,416 |
   | FC1 | (400 + 1) * 120 | ? |
   | FC2 | (120 + 1) * 84 | ? |
   | FC3 | (84 + 1) * 10 | ? |
   | **Total** | | **~61.7K** |

3. **Train on MNIST:**
   - Use `torchvision.datasets.MNIST`
   - Pad 28x28 images to 32x32 (use `transforms.Pad(2)`)
   - Training: SGD, lr=0.01, momentum=0.9, 10 epochs
   - Batch size: 64
   - Report final train/test accuracy

4. **Plot the training curve.** Loss and accuracy vs. epoch for both train and test.

5. **Expected performance:** >98% test accuracy. If you are below 97%, something is wrong.

---

## Part 2: VGG-11 on CIFAR-10

### Architecture Specification

Implement VGG-11 (configuration A from the paper), adapted for CIFAR-10's 32x32 input:

```
Input: (3, 32, 32)

Block 1: Conv2d(3, 64, 3, padding=1) -> BN -> ReLU -> MaxPool(2, 2)     -> (64, 16, 16)
Block 2: Conv2d(64, 128, 3, padding=1) -> BN -> ReLU -> MaxPool(2, 2)   -> (128, 8, 8)
Block 3: Conv2d(128, 256, 3, padding=1) -> BN -> ReLU
         Conv2d(256, 256, 3, padding=1) -> BN -> ReLU -> MaxPool(2, 2)   -> (256, 4, 4)
Block 4: Conv2d(256, 512, 3, padding=1) -> BN -> ReLU
         Conv2d(512, 512, 3, padding=1) -> BN -> ReLU -> MaxPool(2, 2)   -> (512, 2, 2)
Block 5: Conv2d(512, 512, 3, padding=1) -> BN -> ReLU
         Conv2d(512, 512, 3, padding=1) -> BN -> ReLU -> MaxPool(2, 2)   -> (512, 1, 1)

Classifier:
  Flatten                            -> (512,)
  Linear(512, 4096) -> ReLU -> Dropout(0.5)
  Linear(4096, 4096) -> ReLU -> Dropout(0.5)
  Linear(4096, 10)
```

**Note:** The original VGG did not use batch normalization. We add it here because (a) it
makes training much easier on CIFAR-10, and (b) it is standard modern practice. The original
VGG paper trained for weeks on ImageNet without BN.

**Alternative (lighter classifier):** The FC layers are enormous (>16M parameters just for the
first FC). For CIFAR-10, you may alternatively use:
```
  AdaptiveAvgPool2d(1)               -> (512,)
  Linear(512, 10)
```
Implement BOTH versions and compare parameter counts and accuracy.

### Requirements

1. **Implement the architecture as a PyTorch nn.Module.**

2. **Count parameters.** Fill in the table:

   | Layer | Calculation | Parameters |
   |-------|------------|------------|
   | Conv(3, 64, 3) | (3*3*3+1)*64 | 1,792 |
   | BN(64) | 2*64 | 128 |
   | Conv(64, 128, 3) | ? | ? |
   | ... | ... | ... |
   | FC(512, 4096) | ? | ? |
   | FC(4096, 4096) | ? | ? |
   | FC(4096, 10) | ? | ? |
   | **Total (with FC)** | | **~28M** |
   | **Total (with GAP)** | | **~9.2M** |

   **Question:** What percentage of parameters are in the FC layers? (This is why VGG was
   considered parameter-inefficient and why GAP was adopted.)

3. **Train on CIFAR-10:**
   - Use `torchvision.datasets.CIFAR10`
   - Data augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip
   - Normalize with CIFAR-10 mean/std: mean=[0.4914, 0.4822, 0.4465],
     std=[0.2470, 0.2435, 0.2616]
   - Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
   - Learning rate schedule: divide by 10 at epochs 80 and 120
   - Train for 150 epochs
   - Batch size: 128

4. **Plot training curves.** Train/test loss and accuracy vs. epoch.

5. **Expected performance:** ~90-92% test accuracy. If below 88%, check your implementation.

---

## Part 3: ResNet-18 on CIFAR-10

### Architecture Specification

Implement ResNet-18 adapted for CIFAR-10 (32x32 input). The key difference from ImageNet
ResNet-18: the first conv layer uses 3x3 kernel with stride 1 (not 7x7 stride 2), and
there is no max pooling at the beginning. The 32x32 input is too small for aggressive
early downsampling.

```
Input: (3, 32, 32)

Stem: Conv2d(3, 64, 3, stride=1, padding=1) -> BN -> ReLU      -> (64, 32, 32)

Layer 1: BasicBlock(64, 64)   x 2                                -> (64, 32, 32)
Layer 2: BasicBlock(64, 128, stride=2) + BasicBlock(128, 128)    -> (128, 16, 16)
Layer 3: BasicBlock(128, 256, stride=2) + BasicBlock(256, 256)   -> (256, 8, 8)
Layer 4: BasicBlock(256, 512, stride=2) + BasicBlock(512, 512)   -> (512, 4, 4)

AdaptiveAvgPool2d(1)                                              -> (512, 1, 1)
Flatten                                                           -> (512,)
Linear(512, 10)
```

**The BasicBlock (implement this carefully):**
```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection: if dimensions change, use 1x1 conv
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity       # <-- THE SKIP CONNECTION
        out = F.relu(out)
        return out
```

### Requirements

1. **Implement the full architecture as a PyTorch nn.Module.** Use the BasicBlock above
   (or your own equivalent).

2. **Count parameters:**

   | Component | Parameters |
   |-----------|-----------|
   | Stem conv + BN | ? |
   | Layer 1 (2 blocks) | ? |
   | Layer 2 (2 blocks + shortcut) | ? |
   | Layer 3 (2 blocks + shortcut) | ? |
   | Layer 4 (2 blocks + shortcut) | ? |
   | FC | ? |
   | **Total** | **~11.2M** |

3. **Train on CIFAR-10:** Use the same training setup as VGG (same augmentation, optimizer,
   LR schedule, epochs). This ensures a fair comparison.

4. **Plot training curves.** Train/test loss and accuracy vs. epoch.

5. **Expected performance:** ~93-95% test accuracy. ResNet should beat VGG.

---

## Part 4: Head-to-Head Comparison

### Task

Train all three architectures on CIFAR-10 with an identical training setup (LeNet needs to
be adapted for CIFAR-10: change input channels to 3 and output classes to 10).

**Identical training setup for fair comparison:**
- Data augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize
- Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
- LR schedule: divide by 10 at epochs 80 and 120
- 150 epochs, batch size 128

### Create a Comparison Table

| Metric | LeNet-5 | VGG-11 (GAP) | ResNet-18 |
|--------|---------|--------------|-----------|
| Total parameters | ~62K | ~9.2M | ~11.2M |
| Conv parameters | | | |
| FC parameters | | | |
| Test accuracy (best) | | | |
| Test accuracy (final) | | | |
| Training time (total) | | | |
| Time per epoch | | | |
| Final train loss | | | |
| Final test loss | | | |
| Epoch of best test acc | | | |

### Create Comparison Plots

1. **All three training curves on one plot** (test accuracy vs. epoch). Use different colors.
   This should clearly show ResNet converging faster and higher.

2. **All three loss curves on one plot** (test loss vs. epoch). Note which models overfit
   (train loss drops but test loss increases).

3. **Parameter efficiency plot:** Test accuracy vs. parameter count. Which model gives the
   best accuracy per parameter?

### Analysis Questions (Write at Least 500 Words Total)

1. **Why does ResNet outperform VGG despite similar parameter counts?**
   Discuss skip connections, gradient flow, effective depth, and optimization landscape.

2. **Why does LeNet underperform?** Is it purely about parameter count, or does architecture
   matter? Could you make a 62K-parameter model that performs much better than LeNet on
   CIFAR-10? How?

3. **Training dynamics.** At which epoch does each model "converge"? How do the LR drops
   affect each model? Does any model show signs of overfitting? Which model has the largest
   gap between train and test accuracy (indicating overfitting)?

4. **The skip connection effect.** Compare ResNet-18 test accuracy to a "PlainNet-18" — the
   same architecture but with skip connections removed (just comment out the `out += identity`
   line and the shortcut). How much does accuracy drop? Does PlainNet-18 even train
   successfully?

5. **Depth vs. width tradeoff.** ResNet-18 has more layers but fewer channels at each layer
   than VGG-11. Which matters more: depth or width? What does your experiment suggest?

---

## Part 5: The Skip Connection Experiment

This is the most important experiment in this assignment.

### Task

Create a "PlainNet-18" by modifying your ResNet-18 to remove all skip connections:

```python
def forward(self, x):
    # ResNet version:
    # identity = self.shortcut(x)
    # out = F.relu(self.bn1(self.conv1(x)))
    # out = self.bn2(self.conv2(out))
    # out += identity
    # out = F.relu(out)

    # PlainNet version (NO skip connection):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out = F.relu(out)
    return out
```

### Train PlainNet-18 with the same setup as ResNet-18.

### Analysis

1. **Does PlainNet-18 train at all?** Report final accuracy. It should work (18 layers is
   not crazy deep with BN), but likely worse than ResNet-18.

2. **Plot the training curves side by side:** ResNet-18 vs PlainNet-18.

3. **Try PlainNet-34 and ResNet-34.** (Add more blocks: 3, 4, 6, 3 blocks per layer instead
   of 2, 2, 2, 2.) With 34 layers, the degradation problem should be more visible.

4. **Gradient analysis.** After training, compute the gradient norms at different layers for
   both ResNet-18 and PlainNet-18. Do gradients vanish in the plain network?

   ```python
   def check_gradient_norms(model, dataloader):
       """Compute gradient norms at each layer."""
       model.train()
       inputs, targets = next(iter(dataloader))
       outputs = model(inputs)
       loss = F.cross_entropy(outputs, targets)
       loss.backward()

       for name, param in model.named_parameters():
           if param.grad is not None and 'weight' in name:
               grad_norm = param.grad.norm().item()
               print(f"{name:40s}  grad_norm: {grad_norm:.6f}")
   ```

---

## Deliverables

Submit a Jupyter notebook or Python scripts + PDF report containing:

1. **Three complete model implementations** (LeNet-5, VGG-11, ResNet-18) as PyTorch nn.Modules
2. **Parameter count tables** for all three models, computed manually AND verified via code
3. **Training code** with proper data loading, augmentation, and training loop
4. **Training curves** for all models (individual and comparison plots)
5. **Comparison table** filled in with actual experimental results
6. **Skip connection experiment** (PlainNet-18 vs ResNet-18) with gradient analysis
7. **Written analysis** (minimum 500 words) answering all analysis questions

### Grading Criteria

| Component | Weight | Criteria |
|-----------|--------|----------|
| Architecture correctness | 25% | All three models have correct shapes and parameter counts |
| Training pipeline | 15% | Proper augmentation, optimization, LR schedule |
| Training results | 15% | Reasonable accuracies (LeNet >75%, VGG >88%, ResNet >92% on CIFAR-10) |
| Comparison analysis | 20% | Complete table, clear plots, thoughtful written analysis |
| Skip connection experiment | 15% | PlainNet vs ResNet with gradient analysis |
| Code quality | 10% | Clean, well-organized, reproducible |

### Common Pitfalls

- **CIFAR-10 ResNet does NOT use 7x7 first conv.** The standard CIFAR-10 ResNet uses 3x3
  stride 1 with no initial max pooling. Using the ImageNet stem on 32x32 images is a common
  mistake that kills accuracy.
- **Forgetting bias=False when using BatchNorm.** The BN beta parameter replaces the conv bias.
  Having both wastes parameters and can cause subtle issues.
- **Not using model.eval() during validation.** BatchNorm behaves differently in train vs eval
  mode. Always call model.eval() before validation and model.train() before training.
- **Comparing models trained with different setups.** Use identical training hyperparameters
  for the comparison to be fair.

---

## Stretch Goals

1. **Implement an Inception module.** Build a small Inception-style network with 2-3 Inception
   modules. Train on CIFAR-10. Count parameters and compare efficiency to VGG and ResNet.

2. **Implement a bottleneck ResNet block.** Build ResNet-50 (using bottleneck blocks instead
   of basic blocks) adapted for CIFAR-10. Compare to ResNet-18.

3. **Implement squeeze-and-excitation.** Add SE blocks to your ResNet-18. Does it improve
   accuracy? How many parameters does it add?

4. **Learning rate finder.** Implement the learning rate range test (start with a very small
   LR, increase exponentially, plot loss vs LR). Find the optimal initial LR for each model.

5. **Weight initialization experiment.** Train ResNet-18 with (a) Kaiming initialization
   (default), (b) Xavier initialization, (c) zero initialization. How does initialization
   affect training speed and final accuracy? Does zero init work? (Answer: it should not
   for the conv layers, but the skip connections might save you. Investigate.)

6. **Implement PreAct-ResNet.** The "pre-activation" variant where the block order is
   BN -> ReLU -> Conv instead of Conv -> BN -> ReLU. Does it improve performance on
   CIFAR-10? (He et al., 2016, "Identity Mappings in Deep Residual Networks.")
