# Module 5: Convolutional Neural Networks — Comprehensive Reference Notes

## Table of Contents

1. [The Convolution Operation](#1-the-convolution-operation)
2. [Convolution Arithmetic](#2-convolution-arithmetic)
3. [Multi-Channel Convolution and Parameter Counting](#3-multi-channel-convolution-and-parameter-counting)
4. [Advanced Convolution Variants](#4-advanced-convolution-variants)
5. [Pooling Operations](#5-pooling-operations)
6. [Receptive Field Calculation](#6-receptive-field-calculation)
7. [Batch Normalization](#7-batch-normalization)
8. [Landmark Architectures](#8-landmark-architectures)
9. [ResNet and Skip Connections — Deep Dive](#9-resnet-and-skip-connections--deep-dive)
10. [Data Augmentation](#10-data-augmentation)
11. [Transfer Learning](#11-transfer-learning)
12. [Object Detection and Segmentation](#12-object-detection-and-segmentation)
13. [Debugging and Visualization](#13-debugging-and-visualization)
14. [Quick Reference Tables](#14-quick-reference-tables)

---

## 1. The Convolution Operation

### Intuition

A convolution slides a small template (kernel) across an image, computing at each position
how well the local patch matches the template. Think of it as a magnifying glass scanning
across the image, asking at each location: "does this spot contain the pattern I am looking
for?"

If the kernel contains an edge detector pattern, the output will be bright where there are
edges and dark where there are none. If the kernel looks for a corner, the output highlights
corners. The network LEARNS which patterns to look for — that is the power of convolution.

**Two critical priors baked into convolutions:**

1. **Spatial locality:** Each output neuron depends only on a small local patch of the input.
   A pixel in New York and a pixel in London should not directly interact — let that emerge
   through stacking layers.

2. **Translation equivariance:** The same kernel is used everywhere. A cat in the top-left is
   processed by the same weights as a cat in the bottom-right. Formally: if you shift the
   input, the output shifts by the same amount.

### The Math

**2D convolution (single channel, single kernel):**

Given input $I$ of size $H_{in} \times W_{in}$ and kernel $K$ of size $K_h \times K_w$:

$$\text{Output}(i, j) = \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} I(i \cdot s + m,\; j \cdot s + n) \cdot K(m, n) + b$$

where $s$ is the stride and $b$ is the bias term.

**Technical note:** Deep learning "convolution" is actually cross-correlation. True
mathematical convolution flips the kernel horizontally and vertically before applying it.
Since kernels are learned, this distinction is irrelevant in practice — the network can
learn the flipped version.

**Multi-channel convolution:**

For input with $C_{in}$ channels and $C_{out}$ output channels:

$$\text{Output}(c_{out}, i, j) = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} I(c,\; i \cdot s+m,\; j \cdot s+n) \cdot W(c_{out}, c, m, n) + b(c_{out})$$

The weight tensor $W$ has shape $(C_{out}, C_{in}, K_h, K_w)$.
Each of the $C_{out}$ filters is a 3D tensor of shape $(C_{in}, K_h, K_w)$.

### Code

```python
import torch
import torch.nn as nn

# Basic Conv2d layer
conv = nn.Conv2d(
    in_channels=3,      # e.g., RGB input
    out_channels=64,    # number of filters
    kernel_size=3,      # 3x3 kernel
    stride=1,           # slide one pixel at a time
    padding=1,          # same padding (output = input size)
    bias=True           # include bias term
)

# Check weight shape
print(conv.weight.shape)  # torch.Size([64, 3, 3, 3])
#                          (out_channels, in_channels, kernel_h, kernel_w)
print(conv.bias.shape)    # torch.Size([64])

# Forward pass
x = torch.randn(1, 3, 32, 32)  # (batch, channels, height, width)
out = conv(x)
print(out.shape)  # torch.Size([1, 64, 32, 32])
```

### Practical Tips

- Always use odd kernel sizes (3, 5, 7) so that "same" padding is exact (even kernels
  create asymmetric padding issues).
- 3x3 kernels are the default. Two 3x3 layers = one 5x5 receptive field but with fewer
  parameters and more nonlinearity. This is the VGG insight.
- The first layer of a network processing large images (224x224) often uses a larger kernel
  (7x7 with stride 2) to aggressively downsample. ResNet does this.
- For small images (32x32, CIFAR-10), do NOT use stride-2 or large kernels in the first
  layer — you would lose too much spatial information.

---

## 2. Convolution Arithmetic

### The Output Size Formula

**MEMORIZE THIS. You will use it constantly.**

$$\text{output\_size} = \left\lfloor \frac{\text{input\_size} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1$$

Equivalently, for those who prefer to think about it as: how many positions can the kernel
fit?

$$\text{output\_size} = \left\lfloor \frac{\overbrace{\text{input\_size} + 2 \times \text{padding} - \text{kernel\_size}}^{\text{effective input size}}}{\underbrace{\text{stride}}_{\text{step size}}} \right\rfloor + 1$$

### Padding Types

**Valid padding (padding=0):** No padding. Output shrinks.

$$\text{out} = \left\lfloor \frac{\text{in} - \text{kernel}}{\text{stride}} \right\rfloor + 1$$

**Same padding:** Output = Input (when stride=1).

$$\text{padding} = \frac{\text{kernel\_size} - 1}{2} \quad \text{(requires odd kernel\_size)}$$

**Full padding:** Padding = kernel_size - 1. Every possible overlap is computed. Output is
larger than input.

### Worked Examples

**Example 1: Basic convolution**
```
Input: 32x32, Kernel: 3x3, Stride: 1, Padding: 0
Output = (32 + 0 - 3) / 1 + 1 = 30
Output: 30x30
```

**Example 2: Same padding**
```
Input: 32x32, Kernel: 3x3, Stride: 1, Padding: 1
Output = (32 + 2 - 3) / 1 + 1 = 32
Output: 32x32  (same as input)
```

**Example 3: Stride 2 downsampling**
```
Input: 32x32, Kernel: 3x3, Stride: 2, Padding: 1
Output = (32 + 2 - 3) / 2 + 1 = 16
Output: 16x16  (halved)
```

**Example 4: ResNet first convolution**
```
Input: 224x224, Kernel: 7x7, Stride: 2, Padding: 3
Output = (224 + 6 - 7) / 2 + 1 = 112
Output: 112x112
```

**Example 5: Large kernel, no padding, stride 4 (AlexNet first conv)**
```
Input: 227x227, Kernel: 11x11, Stride: 4, Padding: 0
Output = (227 + 0 - 11) / 4 + 1 = 55
Output: 55x55
```

**Example 6: Odd case**
```
Input: 13x13, Kernel: 3x3, Stride: 2, Padding: 0
Output = (13 + 0 - 3) / 2 + 1 = 6
Output: 6x6
```

**Example 7: 1x1 convolution**
```
Input: 14x14, Kernel: 1x1, Stride: 1, Padding: 0
Output = (14 + 0 - 1) / 1 + 1 = 14
Output: 14x14  (spatial dimensions unchanged — 1x1 only changes channels)
```

**Example 8: Dilated convolution**
```
Effective kernel size = kernel + (kernel - 1) * (dilation - 1)
For kernel: 3x3, dilation: 2:
  Effective kernel = 3 + 2 * 1 = 5

Input: 32x32, Kernel: 3x3, Dilation: 2, Stride: 1, Padding: 2
Output = (32 + 4 - 5) / 1 + 1 = 32
Output: 32x32  (same padding with dilation)
```

### Quick Reference Table: Common Configurations

| Input | Kernel | Stride | Padding | Output | Notes |
|-------|--------|--------|---------|--------|-------|
| 32 | 3 | 1 | 0 | 30 | Valid conv |
| 32 | 3 | 1 | 1 | 32 | Same conv |
| 32 | 3 | 2 | 1 | 16 | Downsample 2x |
| 32 | 5 | 1 | 2 | 32 | Same conv, 5x5 kernel |
| 32 | 5 | 2 | 2 | 16 | Downsample 2x, 5x5 kernel |
| 224 | 7 | 2 | 3 | 112 | ResNet stem |
| 227 | 11 | 4 | 0 | 55 | AlexNet first conv |
| 56 | 3 | 1 | 1 | 56 | Standard ResNet block conv |
| 56 | 3 | 2 | 1 | 28 | ResNet downsample block |
| H | 1 | 1 | 0 | H | 1x1 conv (no spatial change) |
| H | 2 | 2 | 0 | H/2 | Common pooling |

### Transposed Convolution Output Size

For upsampling (transposed convolution):

$$\text{output\_size} = (\text{input\_size} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size} + \text{output\_padding}$$

**Example:** Upsample 2x:
```
Input: 16x16, Kernel: 4x4, Stride: 2, Padding: 1, Output_padding: 0
Output = (16 - 1) * 2 - 2 + 4 + 0 = 32
Output: 32x32  (doubled)
```

---

## 3. Multi-Channel Convolution and Parameter Counting

### Intuition

A single convolution filter is not a 2D matrix — it is a 3D tensor. If the input has $C_{in}$
channels, the filter has shape $(C_{in}, K_h, K_w)$. It processes ALL input channels
simultaneously to produce a SINGLE output channel (one feature map).

To get $C_{out}$ output channels, we need $C_{out}$ separate 3D filters. The full weight tensor has
shape $(C_{out}, C_{in}, K_h, K_w)$.

```
           +---------+
           | Filter 1 | --- (C_in, K_h, K_w) ---> 1 feature map
           +---------+
Input ---> | Filter 2 | --- (C_in, K_h, K_w) ---> 1 feature map    = C_out feature maps
(C_in,     | ...      |
 H, W)     | Filter N | --- (C_in, K_h, K_w) ---> 1 feature map
           +---------+
            N = C_out
```

### Parameter Count Formula

**THIS IS A DEEPMIND INTERVIEW QUESTION. Know it perfectly.**

$$\text{Parameters} = \underbrace{(K_h \times K_w \times C_{in}}_{\text{weights per filter}} + \underbrace{1}_{\text{bias}}) \times C_{out}$$

Without bias: $\text{Parameters} = K_h \times K_w \times C_{in} \times C_{out}$

### Worked Examples: Parameter Counting

**Example 1: First conv layer (RGB input)**
```
Conv2d(3, 64, 3):  (3 * 3 * 3 + 1) * 64 = 28 * 64 = 1,792 parameters
```

**Example 2: Deep conv layer**
```
Conv2d(256, 512, 3):  (3 * 3 * 256 + 1) * 512 = 2305 * 512 = 1,180,160 parameters
```

**Example 3: 1x1 convolution**
```
Conv2d(512, 128, 1):  (1 * 1 * 512 + 1) * 128 = 513 * 128 = 65,664 parameters
```

**Example 4: Fully connected layer (for comparison)**
```
Linear(512 * 7 * 7, 4096):  (512 * 7 * 7 + 1) * 4096 = 25089 * 4096 = 102,764,544 parameters
                              ^-- THIS IS WHY FC LAYERS ARE HUGE --^
```

**Example 5: Complete parameter count for a simple CNN**
```
Conv2d(3, 32, 3, padding=1):      (3*3*3 + 1) * 32    =      896
Conv2d(32, 64, 3, padding=1):     (3*3*32 + 1) * 64   =   18,496
Conv2d(64, 128, 3, padding=1):    (3*3*64 + 1) * 128  =   73,856
Global Avg Pool:                                             0
Linear(128, 10):                  (128 + 1) * 10       =    1,290
                                                        ---------
Total:                                                    94,538
```

Compare to a single FC layer on flattened 32x32x3 input:
```
Linear(3072, 128):  (3072 + 1) * 128 = 393,344  (4x more than the entire CNN)
```

### FLOPs (Floating Point Operations)

For a conv layer, the number of multiply-accumulate operations:

$$\text{FLOPs} = K_h \times K_w \times C_{in} \times C_{out} \times H_{out} \times W_{out}$$

This matters for understanding computational cost. Note that FLOPs depends on the spatial
output size, while parameter count does not.

### Code: Counting Parameters in PyTorch

```python
def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable

# Per-layer breakdown
def parameter_breakdown(model):
    """Print parameter count per layer."""
    for name, param in model.named_parameters():
        print(f"{name:40s} {param.numel():>10,}  {list(param.shape)}")
```

---

## 4. Advanced Convolution Variants

### 1x1 Convolutions

**Intuition:** A 1x1 convolution looks at a single spatial position but across ALL channels.
It is a fully-connected layer applied independently to each pixel, mixing information across
channels without any spatial mixing.

```
Input at position (i, j): a vector of length C_in
1x1 conv:                 a matrix multiply (C_out x C_in)
Output at position (i, j): a vector of length C_out
```

Formally: at each spatial position, the 1x1 conv computes $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ where $W \in \mathbb{R}^{C_{out} \times C_{in}}$.

**Why they are powerful:**
1. **Dimensionality reduction:** Reduce channels before an expensive 3x3 conv (Inception
   bottleneck, ResNet bottleneck).
2. **Dimensionality expansion:** Increase channels (inverted residual blocks in MobileNet).
3. **Cross-channel interaction:** Mix information from different feature maps.
4. **Added nonlinearity:** 1x1 conv + ReLU adds a nonlinear transformation with minimal
   computation.

**Example: Computational savings with 1x1 bottleneck**
```
Direct: Conv2d(256, 256, 3)  -> 3*3*256*256 = 589,824 params
Bottleneck: Conv2d(256, 64, 1) -> 1*1*256*64 =  16,384
            Conv2d(64, 64, 3)  -> 3*3*64*64  =  36,864
            Conv2d(64, 256, 1) -> 1*1*64*256 =  16,384
                                       Total =  69,632 params (8.5x reduction)
```

### Depthwise Separable Convolutions

**Intuition:** A standard convolution jointly learns spatial patterns AND cross-channel
interactions. A depthwise separable convolution factors this into two steps: (1) learn
spatial patterns per channel, (2) learn cross-channel interactions.

**Standard convolution:**
```
Input (C_in, H, W) -> Kernel (C_out, C_in, K, K) -> Output (C_out, H', W')
```
Parameters: $K^2 \cdot C_{in} \cdot C_{out}$

**Depthwise separable convolution:**
```
Step 1 — Depthwise: One K x K filter per input channel
  Input (C_in, H, W) -> Kernel (C_in, 1, K, K) -> Output (C_in, H', W')

Step 2 — Pointwise: 1x1 conv to mix channels
  Input (C_in, H', W') -> Kernel (C_out, C_in, 1, 1) -> Output (C_out, H', W')
```
Total parameters: $K^2 \cdot C_{in} + C_{in} \cdot C_{out}$

**Computational savings ratio:**

$$\frac{\text{Separable}}{\text{Standard}} = \frac{K^2 C_{in} + C_{in} C_{out}}{K^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}$$

For $K=3, C_{out}=256$: ratio $= 1/256 + 1/9 = 0.115$ (8.7x savings)

**Code:**
```python
# Standard convolution
conv_standard = nn.Conv2d(64, 128, 3, padding=1)
# Params: 3*3*64*128 + 128 = 73,856

# Depthwise separable convolution
conv_depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)  # groups=C_in
conv_pointwise = nn.Conv2d(64, 128, 1)
# Params: 3*3*64 + 64 + 64*128 + 128 = 576 + 64 + 8192 + 128 = 8,960 (8.2x fewer)
```

### Transposed Convolutions

**Intuition:** A transposed convolution goes the opposite direction — it upsamples. If a
regular convolution with stride 2 halves spatial dimensions, a transposed convolution with
stride 2 doubles them. Used in decoders (U-Net, GANs, autoencoders).

**How it works:** Insert zeros between input elements (stride - 1 zeros between each),
pad the result, then apply a regular convolution.

**Code:**
```python
# Upsample 2x
up = nn.ConvTranspose2d(
    in_channels=256,
    out_channels=128,
    kernel_size=4,
    stride=2,
    padding=1
)
x = torch.randn(1, 256, 16, 16)
out = up(x)
print(out.shape)  # torch.Size([1, 128, 32, 32])  -- doubled!
```

**Checkerboard artifacts:** Transposed convolutions can produce checkerboard patterns in the
output when the kernel size is not divisible by the stride. Common fix: use bilinear
upsampling followed by a regular convolution.

```python
# Alternative to transposed convolution (avoids checkerboard)
up_clean = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(256, 128, 3, padding=1)
)
```

### Dilated / Atrous Convolutions

**Intuition:** Spread the kernel elements apart with gaps. A 3x3 kernel with dilation 2 has
the same 9 parameters but covers a 5x5 area (with gaps). It increases the receptive field
without increasing parameters or reducing spatial resolution.

```
Standard 3x3:        Dilated 3x3 (dilation=2):

X X X                X . X . X
X X X                . . . . .
X X X                X . X . X
                     . . . . .
                     X . X . X
```

**Effective kernel size:** $K_{eff} = K + (K-1) \cdot (\text{dilation} - 1)$

For $K=3$, dilation=2: $K_{eff} = 3 + 2 \times 1 = 5$
For $K=3$, dilation=4: $K_{eff} = 3 + 2 \times 3 = 9$

**Code:**
```python
# Standard 3x3: receptive field = 3x3
conv_standard = nn.Conv2d(64, 64, 3, padding=1, dilation=1)

# Dilated 3x3: receptive field = 5x5 (same params)
conv_dilated = nn.Conv2d(64, 64, 3, padding=2, dilation=2)

# Both have 3*3*64*64 = 36,864 params, but different receptive fields!
```

**Used in:** DeepLab (semantic segmentation), WaveNet (audio generation).

---

## 5. Pooling Operations

### Max Pooling

**Intuition:** Take the maximum value in each local window. Keeps the strongest detected
feature. Provides some translation invariance — if the feature shifts by a pixel, the max
in the pooling window is likely the same.

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# Input: (B, C, 32, 32) -> Output: (B, C, 16, 16)
# Halves spatial dimensions, channels unchanged, ZERO parameters
```

### Average Pooling

**Intuition:** Take the mean of each window. Smoother than max pooling. Less commonly used
in intermediate layers.

```python
pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

### Global Average Pooling (GAP)

**Intuition:** Average across the entire spatial extent of each channel. Collapses (B, C, H, W)
to (B, C). Replaces the large FC layers that plagued early architectures (VGG).

```python
gap = nn.AdaptiveAvgPool2d(1)  # Output size 1x1 regardless of input size
x = torch.randn(1, 512, 7, 7)
out = gap(x)
print(out.shape)  # (1, 512, 1, 1) -- squeeze to (1, 512) before FC
out = out.view(out.size(0), -1)  # (1, 512)
```

**Why GAP is better than FC layers at the end:**
- Zero parameters (FC: 512*7*7*4096 = ~102M parameters)
- Forces each feature map to represent a concept (acts as regularizer)
- Makes the network input-size agnostic (can handle different spatial dimensions)

### Strided Convolution vs Pooling

**Strided convolution (stride=2):** The network LEARNS how to downsample. More flexible.
Used in modern architectures.

**Pooling:** Fixed operation, not learned. Provides exact translation invariance. Simpler.

**Practical guidance:** Both work. Modern architectures tend to prefer strided convolution
in the stem and pooling (or strided conv) elsewhere. The difference in accuracy is usually
small.

---

## 6. Receptive Field Calculation

### Intuition

The receptive field of a neuron is the region of the original input that can influence that
neuron's value. Deeper layers have larger receptive fields — they "see" more of the input.

A single 3x3 conv layer has a 3x3 receptive field. Stack two 3x3 layers, and the second
layer's neurons each depend on a 3x3 patch of the first layer's output, which in turn
depends on a 5x5 patch of the input. So the receptive field of the second layer is 5x5.

### The Formula

For a network with $L$ layers, where layer $l$ has kernel size $k_l$ and stride $s_l$:

$$RF_L = 1 + \sum_{l=1}^{L} (k_l - 1) \prod_{i=1}^{l-1} s_i$$

Or computed recursively:

$$\begin{aligned}
RF_0 &= 1 \quad \text{(a single pixel)} \\
RF_l &= RF_{l-1} + (k_l - 1) \cdot J_{l-1} \\
J_l &= J_{l-1} \cdot s_l \quad \text{(cumulative stride / "jump")}, \quad J_0 = 1
\end{aligned}$$

### Worked Example: Three Stacked 3x3 Convolutions

```
Layer 1: k=3, s=1
  J_1 = 1 * 1 = 1
  RF_1 = 1 + (3-1) * 1 = 3

Layer 2: k=3, s=1
  J_2 = 1 * 1 = 1
  RF_2 = 3 + (3-1) * 1 = 5

Layer 3: k=3, s=1
  J_3 = 1 * 1 = 1
  RF_3 = 5 + (3-1) * 1 = 7
```

**Result:** Three stacked 3x3 convolutions have a 7x7 receptive field.
Parameters: 3 * (3*3*C*C) = 27*C^2 (vs 49*C^2 for a single 7x7 conv).
More nonlinearities: 3 ReLUs vs 1.

### Worked Example: ResNet-18 Receptive Field (Simplified)

```
Conv 7x7, stride 2:    J=2,   RF = 1 + 6*1 = 7
MaxPool 3x3, stride 2: J=4,   RF = 7 + 2*2 = 11
Conv 3x3, stride 1:    J=4,   RF = 11 + 2*4 = 19
Conv 3x3, stride 1:    J=4,   RF = 19 + 2*4 = 27
Conv 3x3, stride 1:    J=4,   RF = 27 + 2*4 = 35
Conv 3x3, stride 1:    J=4,   RF = 35 + 2*4 = 43
Conv 3x3, stride 2:    J=8,   RF = 43 + 2*4 = 51
Conv 3x3, stride 1:    J=8,   RF = 51 + 2*8 = 67
...
```

After just a few blocks, the receptive field already covers a large portion of the 224x224
input. By the final layer, the theoretical receptive field typically exceeds the input size.

### Why This Matters

- If the receptive field is too small for the task (e.g., trying to classify whole images
  with a 3-layer network), the network literally cannot see enough context.
- Dilated convolutions increase the receptive field without adding parameters or reducing
  resolution — useful for tasks like segmentation that need both large context and fine
  spatial detail.
- The effective receptive field (accounting for gradient magnitudes) is typically much smaller
  than the theoretical receptive field — neurons attend most to the center of their
  receptive field.

---

## 7. Batch Normalization

### Intuition

Without batch normalization, the distribution of inputs to each layer changes every time the
preceding layers update. The network is trying to learn on shifting ground. Batch
normalization stabilizes the input to each layer by normalizing it to zero mean, unit
variance, then applying a learned affine transformation.

### The Math

During training, for a mini-batch $\mathcal{B}$ of size $m$, and for each channel $c$:

$$\begin{aligned}
\mu_c &= \frac{1}{m} \sum_{i=1}^{m} x_{i,c} & \text{(batch mean per channel)} \\
\sigma_c^2 &= \frac{1}{m} \sum_{i=1}^{m} (x_{i,c} - \mu_c)^2 & \text{(batch variance per channel)} \\
\hat{x}_{i,c} &= \frac{x_{i,c} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} & \text{(normalize)} \\
y_{i,c} &= \gamma_c \hat{x}_{i,c} + \beta_c & \text{(scale and shift)}
\end{aligned}$$

$\gamma$ and $\beta$ are learnable parameters (one pair per channel). $\epsilon$ is a small constant
(typically $10^{-5}$) for numerical stability.

During inference, use running averages of mean and variance accumulated during training.

### Parameters

For a conv layer with $C$ channels followed by batch norm:

BN parameters $= 2C$ ($\gamma$ and $\beta$).
BN also stores running mean and running variance ($2C$), but these are not optimized.

### Code

```python
# Standard pattern: Conv -> BN -> ReLU
layer = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1, bias=False),  # bias=False! BN has its own bias (beta)
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True)
)
```

**Important:** When using BN, set `bias=False` in the preceding Conv layer. The BN beta
parameter serves the same role as the conv bias, so having both is redundant.

### Placement Options

```
Option 1 (original):     Conv -> BN -> ReLU
Option 2 (alternative):  Conv -> ReLU -> BN
Option 3 (pre-activation ResNet): BN -> ReLU -> Conv
```

Option 1 is the most common and the default recommendation. Option 3 has theoretical
advantages for very deep ResNets (He et al., 2016, "Identity Mappings in Deep Residual
Networks").

### Practical Tips

- BN depends on batch statistics, so use a reasonable batch size (>=16). If batch size is
  very small, use Group Normalization or Layer Normalization instead.
- During evaluation, always call `model.eval()` to switch from batch statistics to running
  statistics.
- BN provides regularization (noisy batch statistics act like noise injection), so you can
  often reduce or eliminate dropout when using BN.
- If your model behaves very differently in train vs eval mode, check that BN running
  statistics are being updated properly.

---

## 8. Landmark Architectures

### Architecture Timeline

```
1998         2012         2014        2014        2015        2017       2019
LeNet-5      AlexNet      VGG         GoogLeNet   ResNet      DenseNet   EfficientNet
60K params   62M          138M        6.8M        25.6M       20-34M     5-66M
MNIST        ImageNet     ImageNet    ImageNet    ImageNet    ImageNet   ImageNet
             (15.3%)      (7.3%)      (6.7%)      (3.6%)      (~3.4%)    (~2.9%)
             ReLU,        3x3 only,   Inception   Skip        Dense      Compound
             GPU,         depth       module,     connections, connect.   scaling,
             Dropout      matters     1x1 conv    depth >>    feature    NAS
                                                              reuse
```

### LeNet-5 (LeCun et al., 1998)

```
Architecture (ASCII diagram):

Input 32x32x1
     |
[Conv 5x5, 6 filters] --> 28x28x6 --> [Sigmoid] --> [AvgPool 2x2] --> 14x14x6
     |
[Conv 5x5, 16 filters] --> 10x10x16 --> [Sigmoid] --> [AvgPool 2x2] --> 5x5x16
     |
[Flatten] --> 400
     |
[FC 400->120] --> [Sigmoid] --> [FC 120->84] --> [Sigmoid] --> [FC 84->10]
     |
Output (10 classes)
```

**Parameter count:**
```
Conv1: (5*5*1 + 1) * 6     =    156
Conv2: (5*5*6 + 1) * 16    =  2,416
FC1:   (400 + 1) * 120     = 48,120
FC2:   (120 + 1) * 84      = 10,164
FC3:   (84 + 1) * 10       =    850
                              ------
Total:                       61,706
```

### AlexNet (Krizhevsky et al., 2012)

```
Architecture:

Input 227x227x3
     |
[Conv 11x11, 96, stride 4] --> 55x55x96 --> [ReLU] --> [MaxPool 3x3, stride 2] --> 27x27x96
     |
[Conv 5x5, 256, pad 2] --> 27x27x256 --> [ReLU] --> [MaxPool 3x3, stride 2] --> 13x13x256
     |
[Conv 3x3, 384, pad 1] --> 13x13x384 --> [ReLU]
     |
[Conv 3x3, 384, pad 1] --> 13x13x384 --> [ReLU]
     |
[Conv 3x3, 256, pad 1] --> 13x13x256 --> [ReLU] --> [MaxPool 3x3, stride 2] --> 6x6x256
     |
[Flatten] --> 9216
     |
[FC 9216->4096] --> [ReLU] --> [Dropout 0.5]
     |
[FC 4096->4096] --> [ReLU] --> [Dropout 0.5]
     |
[FC 4096->1000]
```

**Parameter count:**
```
Conv1: (11*11*3 + 1) * 96       =    34,944
Conv2: (5*5*96 + 1) * 256       =   614,656
Conv3: (3*3*256 + 1) * 384      =   885,120
Conv4: (3*3*384 + 1) * 384      = 1,327,488
Conv5: (3*3*384 + 1) * 256      =   884,992
FC1:   (9216 + 1) * 4096        = 37,752,832
FC2:   (4096 + 1) * 4096        = 16,781,312
FC3:   (4096 + 1) * 1000        =  4,097,000
                                   ----------
Total:                            ~62,378,344

Conv layers: ~3.7M (6%)
FC layers:   ~58.6M (94%)  <-- THIS IS THE PROBLEM
```

### VGG-16 (Simonyan & Zisserman, 2014)

```
Architecture:

Input 224x224x3
     |
[Conv 3x3, 64] x 2 --> [MaxPool] --> 112x112x64
     |
[Conv 3x3, 128] x 2 --> [MaxPool] --> 56x56x128
     |
[Conv 3x3, 256] x 3 --> [MaxPool] --> 28x28x256
     |
[Conv 3x3, 512] x 3 --> [MaxPool] --> 14x14x512
     |
[Conv 3x3, 512] x 3 --> [MaxPool] --> 7x7x512
     |
[Flatten] --> 25088
     |
[FC 25088->4096] --> [ReLU] --> [Dropout]
     |
[FC 4096->4096] --> [ReLU] --> [Dropout]
     |
[FC 4096->1000]
```

**Key insight: Why 3x3?**
```
Two 3x3 convs:   5x5 receptive field,  2 * 3^2 * C^2 = 18 C^2 params
One 5x5 conv:    5x5 receptive field,  25 C^2 params

Three 3x3 convs: 7x7 receptive field,  3 * 9 * C^2 = 27 C^2 params
One 7x7 conv:    7x7 receptive field,  49 C^2 params

Smaller kernels, stacked deeper = same receptive field, fewer params, more nonlinearity.
```

### GoogLeNet / Inception Module

```
Inception Module:

                    Input
                 /   |    |   \
              1x1   1x1  1x1  MaxPool3x3
               |     |    |      |
               |   3x3  5x5   1x1
               |     |    |      |
                \    |    |    /
               Concatenate (depth)
                    |
                  Output

The 1x1 before 3x3 and 5x5 reduces channels (bottleneck), cutting computation.
```

**Example channel counts for one Inception module:**
```
Input: 256 channels

Branch 1: Conv1x1(256, 64)                              -> 64 channels
Branch 2: Conv1x1(256, 96) -> Conv3x3(96, 128)          -> 128 channels
Branch 3: Conv1x1(256, 16) -> Conv5x5(16, 32)           -> 32 channels
Branch 4: MaxPool3x3 -> Conv1x1(256, 32)                -> 32 channels
                                                            -----------
Concatenated output:                                     -> 256 channels

Without 1x1 bottlenecks, the 5x5 branch alone would be:
  Conv5x5(256, 32): 5*5*256*32 = 204,800 params
With bottleneck:
  Conv1x1(256, 16) + Conv5x5(16, 32): 4,096 + 12,800 = 16,896 params (12x reduction)
```

### ResNet (Detailed in Section 9)

### DenseNet

```
Dense Block:

Layer 0 output: x0
Layer 1 input: [x0]              output: x1
Layer 2 input: [x0, x1]          output: x2
Layer 3 input: [x0, x1, x2]     output: x3
Layer 4 input: [x0, x1, x2, x3] output: x4

Each layer receives ALL previous feature maps as input (concatenated).
```

**Growth rate (k):** Each layer produces k feature maps. After L layers in a dense block,
the number of channels is k0 + L*k (where k0 is the initial channel count).

**Transition layers** between dense blocks: BN -> 1x1 Conv (halve channels) -> AvgPool 2x2.

**Advantages:**
- Feature reuse: later layers can directly access early features
- Fewer parameters than ResNet for comparable accuracy
- Better gradient flow (every layer directly connected to the loss)

### EfficientNet

**Compound scaling insight:**

Standard scaling (pick one) vs. Compound scaling (all three):

$$\begin{aligned}
\text{depth:} \quad d &= \alpha^\phi \\
\text{width:} \quad w &= \beta^\phi \\
\text{resolution:} \quad r &= \gamma^\phi
\end{aligned}$$

with constraint: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (so FLOPs roughly double per increment of $\phi$).

Base architecture (B0) found via NAS, then scaled to B1-B7.

**Key building block: MBConv (Mobile Inverted Bottleneck):**
```
Input (C channels)
  -> Conv1x1 (expand to C*t channels, t=expansion ratio, typically 6)
  -> Depthwise Conv 3x3 or 5x5
  -> Squeeze-and-Excitation block
  -> Conv1x1 (project back to C channels)
  -> Add residual connection (if input and output shapes match)
```

---

## 9. ResNet and Skip Connections — Deep Dive

**This section is the most important in these notes. ResNet's skip connections are arguably
the single most impactful architectural innovation in deep learning. They appear in
transformers, U-Nets, diffusion models, and virtually every modern architecture.**

### The Problem: Degradation

Before ResNet, adding more layers to a CNN eventually HURT performance:

```
                 Training Error
20-layer:         6.0%
56-layer:         7.5%    <-- WORSE, not better

This is NOT overfitting (test error also gets worse, and training error increases too).
The deeper network is harder to OPTIMIZE.
```

The deeper network contains the shallower one as a subset (the extra layers could learn
identity). So in theory, the deeper network should be at least as good. The fact that it is
not reveals an optimization failure.

### The Solution: Residual Learning

Instead of learning the desired mapping $H(x)$ directly, learn the residual $F(x) = H(x) - x$.
Then the output is $F(x) + x$.

```
             +-----+
    x ------>| F(x) |------>(+)----> F(x) + x
    |        +-----+         ^
    |                        |
    +------------------------+   (skip / shortcut / identity connection)
```

**Why this helps:** If the optimal function is close to identity, the network only needs to
push $F(x)$ toward zero. Pushing weights toward zero is easy (weight decay does it naturally).
Learning a complete identity mapping from scratch is hard.

### The Residual Block in Detail

**Basic block (ResNet-18, ResNet-34):**
```
                x
                |
        Conv 3x3, BN, ReLU
                |
        Conv 3x3, BN
                |
           (+) <---- x (identity shortcut)
                |
              ReLU
                |
             output
```

**Bottleneck block (ResNet-50, 101, 152):**
```
                x
                |
        Conv 1x1, BN, ReLU     (reduce channels: 256 -> 64)
                |
        Conv 3x3, BN, ReLU     (spatial processing: 64 -> 64)
                |
        Conv 1x1, BN           (expand channels: 64 -> 256)
                |
           (+) <---- x (identity shortcut)
                |
              ReLU
                |
             output
```

**When dimensions change (stride > 1 or channel count changes):**
The shortcut uses a 1x1 convolution with appropriate stride to match dimensions:
```
           (+) <---- Conv1x1(stride=2), BN(x)     (projection shortcut)
```

### Why Skip Connections Work — Three Perspectives

**Perspective 1: Gradient Flow**

During backpropagation:

$$\text{output} = F(x) + x \quad \Rightarrow \quad \frac{\partial\, \text{output}}{\partial x} = \frac{\partial F(x)}{\partial x} + \underbrace{1}_{\text{key}}$$

The $+1$ term means that the gradient flowing through the skip connection is always at
least 1 (before scaling by subsequent layers). Even if $\frac{\partial F(x)}{\partial x}$ is very small (vanishing
gradient in the conv layers), the gradient through the shortcut is preserved.

In a network with N residual blocks, the gradient from the loss to the input passes through
N addition operations, each contributing a +1 term. This creates a "gradient highway" that
prevents vanishing gradients.

**Contrast with a plain network (no skip connections):**
```
d(layer_N output) / d(layer_1 input) = product of N Jacobians

If each Jacobian has eigenvalues < 1, the product goes to 0 exponentially.
With skip connections, the product includes +1 terms that prevent this collapse.
```

**Perspective 2: Ensemble Interpretation (Veit et al., 2016)**

A ResNet with $N$ blocks can be "unrolled" into a collection of $2^N$ paths:
```
3-block ResNet paths:

x -> F1 -> F2 -> F3 -> output     (all 3 blocks used)
x -> F1 -> F2 ------> output     (skip block 3)
x -> F1 ------> F3 -> output     (skip block 2)
x ------> F2 -> F3 -> output     (skip block 1)
x -> F1 -----------> output     (skip blocks 2,3)
x ------> F2 -------> output    (skip blocks 1,3)
x -----------> F3 -> output     (skip blocks 1,2)
x -----------------> output     (skip all — pure identity)
```

The network is implicitly an ensemble of paths of different depths. Experiments show that
most of the gradient flows through paths of moderate length, not the very longest path.
This means the effective depth is much less than the nominal depth.

**Perspective 3: Identity Mapping Ease**

A plain network layer must learn to pass information through:

$$y = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2 \quad \leftarrow \text{must configure } W_1, W_2 \text{ to approximate identity}$$

A residual block only needs to learn the deviation from identity:

$$y = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2 + x \quad \leftarrow \text{if } W_1, W_2 \text{ are small, } y \approx x$$

At initialization (small random weights), a residual block is approximately the identity
function. The network starts "doing nothing" and gradually learns useful transformations.
This is a much better starting point than a random transformation.

### ResNet Architecture Family

**ResNet-18:**
```
Input (3, 224, 224)
  |
[Conv 7x7, 64, stride 2] -> BN -> ReLU -> MaxPool 3x3, stride 2
  |                                        Output: (64, 56, 56)
  |
[BasicBlock(64, 64)] x 2                  Output: (64, 56, 56)
  |                                        Params: 2 * 2 * (3*3*64*64) = 147,456
  |
[BasicBlock(64, 128, stride=2)]           Output: (128, 28, 28)
[BasicBlock(128, 128)]                    Params: (3*3*64*128 + 3*3*128*128) +
  |                                                (3*3*128*128)*2 = ~500K
  |
[BasicBlock(128, 256, stride=2)]          Output: (256, 14, 14)
[BasicBlock(256, 256)]                    Params: ~2M
  |
[BasicBlock(256, 512, stride=2)]          Output: (512, 7, 7)
[BasicBlock(512, 512)]                    Params: ~8M
  |
[Global Average Pool]                     Output: (512,)
  |
[Linear(512, 1000)]                       Params: 513,000
  |
Output (1000 classes)

Total: ~11.7M parameters
```

### Code: ResNet Basic Block and ResNet-18

```python
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet basic residual block (two 3x3 convolutions)."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # used when dimensions change

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # match dimensions via 1x1 conv

        out += identity  # THE SKIP CONNECTION
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """ResNet bottleneck block (1x1 -> 3x3 -> 1x1)."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 implementation from scratch."""

    def __init__(self, num_classes=1000):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

---

## 10. Data Augmentation

### Intuition

Data augmentation generates new training examples by applying random transformations to
existing images. This is essentially free additional training data. It is one of the highest
impact, lowest effort techniques in computer vision.

The key principle: augmentations should produce images that could plausibly appear in the
real data distribution. Flipping a cat horizontally is realistic. Rotating it 180 degrees
is not (cats are rarely upside-down in photos, and an upside-down cat means something
different from a right-side-up one in some tasks).

### Standard Augmentations

```python
import torchvision.transforms as T

# Standard pipeline for ImageNet-scale training
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation pipeline (NO augmentation, just resize and normalize)
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Advanced Augmentations

**Cutout / Random Erasing:**
Randomly zero out a rectangular region of the image.
```python
T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
```
Forces the network to use the full image, not just the most discriminative region.

**Mixup:**
```python
def mixup(x1, y1, x2, y2, alpha=0.2):
    """Mix two samples and their labels."""
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2  # soft labels
    return x, y
```
Creates new training examples by linearly interpolating between pairs. Encourages the
network to behave linearly between training examples. Improves calibration.

**CutMix:**
```python
def cutmix(x1, y1, x2, y2, alpha=1.0):
    """Cut a patch from x2 and paste onto x1. Mix labels proportionally."""
    lam = np.random.beta(alpha, alpha)
    # Random bounding box
    W, H = x1.shape[-2:]
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1_new = x1.clone()
    x1_new[..., cx:cx+cut_w, cy:cy+cut_h] = x2[..., cx:cx+cut_w, cy:cy+cut_h]
    lam_actual = 1 - (cut_w * cut_h) / (W * H)
    y = lam_actual * y1 + (1 - lam_actual) * y2
    return x1_new, y
```
Combines Cutout (forcing use of full image) with Mixup (label smoothing).

### Test-Time Augmentation (TTA)

At test time, apply multiple augmentations to the same image and average predictions:
```python
def tta_predict(model, image, n_augments=10):
    """Test-time augmentation: average predictions over augmented versions."""
    model.eval()
    predictions = []
    for _ in range(n_augments):
        augmented = train_transform(image)  # random augmentation
        with torch.no_grad():
            pred = model(augmented.unsqueeze(0))
        predictions.append(pred)
    # Also include the non-augmented prediction
    clean = val_transform(image)
    with torch.no_grad():
        predictions.append(model(clean.unsqueeze(0)))
    return torch.stack(predictions).mean(dim=0)
```

Typically gives 1-2% accuracy improvement at the cost of N times more inference computation.
Common for competitions.

---

## 11. Transfer Learning

### Intuition

A CNN trained on ImageNet has learned to detect edges, textures, patterns, parts, and objects.
These features are broadly useful. Instead of learning them from scratch on your small dataset,
start with the ImageNet features and adapt them.

Transfer learning is not a hack — it is the most effective approach for the vast majority of
real-world vision tasks. Very few practitioners train from scratch.

### The Three Strategies

**Strategy 1: Feature Extraction (Frozen Backbone)**

```python
import torchvision.models as models

# Load pretrained ResNet-50
model = models.resnet50(weights='IMAGENET1K_V2')

# Freeze ALL backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Replace classification head
model.fc = nn.Linear(2048, num_classes)  # only this is trainable

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / Total: {total:,}")
# Trainable: 10,250 / Total: 25,567,306  (for 5 classes)
```

**When to use:** Small dataset (<1K images), task similar to ImageNet.
**Advantage:** Fast training, no overfitting risk.
**Disadvantage:** Cannot adapt features to your specific domain.

**Strategy 2: Fine-tuning Last Layers**

```python
# Freeze early layers
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

# Only layer4 and fc are trainable
# Use a smaller learning rate than training from scratch
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4  # smaller than the 1e-3 default for training from scratch
)
```

**When to use:** Medium dataset (1K-10K images), moderate domain shift.
**Advantage:** Adapts high-level features while preserving low-level ones.
**Disadvantage:** Risk of overfitting if too many layers are unfrozen.

**Strategy 3: Full Fine-tuning with Differential Learning Rates**

```python
# Different learning rates for different parts of the network
optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(),   'lr': 1e-6},
    {'params': model.bn1.parameters(),     'lr': 1e-6},
    {'params': model.layer1.parameters(),  'lr': 1e-5},
    {'params': model.layer2.parameters(),  'lr': 1e-5},
    {'params': model.layer3.parameters(),  'lr': 1e-4},
    {'params': model.layer4.parameters(),  'lr': 1e-4},
    {'params': model.fc.parameters(),      'lr': 1e-3},
])
```

**When to use:** Larger dataset (>5K images), significant domain shift (e.g., medical images).
**Advantage:** Full adaptation of all features.
**Disadvantage:** Slowest, highest overfitting risk.

### Practical Transfer Learning Recipe

```
Step 1: Freeze backbone, train only the new head for 5-10 epochs
        -> This gives the random head a chance to "warm up" without
           destroying pretrained features

Step 2: Unfreeze last block(s), fine-tune with small LR for 10-20 epochs
        -> Adapt high-level features to your task

Step 3: (Optional) Unfreeze everything, use differential LRs for 10+ epochs
        -> Full adaptation

Step 4: (Optional) Progressive resizing
        -> Start with 128x128, then 224x224, then 320x320
        -> Acts as regularization + curriculum
```

### Which Layers to Replace

For a pretrained model, you always replace the final classification layer:

```python
# ResNet: replace model.fc
model.fc = nn.Linear(model.fc.in_features, num_classes)

# VGG: replace model.classifier[-1]
model.classifier[-1] = nn.Linear(4096, num_classes)

# EfficientNet: replace model.classifier[-1]
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
```

For some tasks, you might also want to replace the final block or add layers:
```python
# More complex head for fine-grained classification
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes),
)
```

### Important: Input Normalization

**Always use the same normalization as the pretrained model's training.**

For ImageNet-pretrained models:
```python
# These are the ImageNet statistics
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
```

If your images have different statistics (e.g., medical images), you still use the ImageNet
normalization when fine-tuning. The pretrained features expect inputs in this range.

---

## 12. Object Detection and Segmentation

### Object Detection: R-CNN Family

**R-CNN (2014):**
```
Image -> Selective Search (~2000 regions) -> Warp each to 227x227
  -> CNN feature extraction (per region) -> SVM classifier + bbox regression
Speed: ~47 sec/image  (CNN runs 2000 times!)
```

**Fast R-CNN (2015):**
```
Image -> CNN (one pass on entire image) -> Feature map
  -> Project region proposals onto feature map -> ROI Pooling
  -> FC layers -> Classifier + bbox regression (per region)
Speed: ~0.3 sec/image  (CNN runs ONCE)
```

**Faster R-CNN (2015):**
```
Image -> CNN -> Feature map -> Region Proposal Network (RPN) -> Proposals
  -> ROI Pooling -> FC layers -> Classifier + bbox regression
Speed: ~0.2 sec/image  (RPN replaces slow selective search)
```

**Key concepts:**
- **Anchor boxes:** Predefined boxes at multiple scales/ratios at each position.
  RPN predicts "is there an object?" and refines box coordinates for each anchor.
- **ROI Pooling:** Extracts fixed-size features from variable-size regions of the feature map.
- **Non-Maximum Suppression (NMS):** Post-processing to remove duplicate detections.

### Object Detection: YOLO

```
Image -> Single CNN -> Grid of predictions
  Each grid cell predicts:
    - B bounding boxes (x, y, w, h, confidence)
    - C class probabilities
    - Total output: S x S x (B * 5 + C) tensor

For YOLO v1: S=7, B=2, C=20 -> 7x7x30 output tensor
```

**Key advantage:** Single forward pass. Real-time detection (45 FPS).
**Key disadvantage:** Struggles with small objects and objects close together.

### Semantic Segmentation: U-Net

```
Encoder (downsample)                Decoder (upsample)

Input   (1, H, W)
   |                                    |
[Conv, Conv] (64, H, W)   ------>  [Conv, Conv] (64, H, W) -> Output (C, H, W)
   |                                    ^
[Pool]  (64, H/2, W/2)                 |
   |                               [UpConv + Concat]
[Conv, Conv] (128, H/2, W/2) --->  [Conv, Conv] (128, H/2, W/2)
   |                                    ^
[Pool]  (128, H/4, W/4)               |
   |                               [UpConv + Concat]
[Conv, Conv] (256, H/4, W/4) --->  [Conv, Conv] (256, H/4, W/4)
   |                                    ^
[Pool]  (256, H/8, W/8)               |
   |                               [UpConv + Concat]
[Conv, Conv] (512, H/8, W/8) --->  [Conv, Conv] (512, H/8, W/8)
   |                                    ^
[Pool]  (512, H/16, W/16)             |
   |                               [UpConv + Concat]
[Conv, Conv] (1024, H/16, W/16)
          (bottleneck)

----> = skip connection (concatenation, not addition)
```

**Key design:**
- Symmetric encoder-decoder structure
- Skip connections carry fine spatial information from encoder to decoder
- Encoder captures "what" (semantics), skip connections provide "where" (localization)
- Output has C channels where C = number of classes (per-pixel classification)

---

## 13. Debugging and Visualization

### Grad-CAM Implementation

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which regions of the input
image are most important for a specific class prediction.

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    """Grad-CAM: Visual explanations from deep networks."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap."""
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Compute Grad-CAM
        # Global average pool the gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # ReLU (only positive influence)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='bilinear', align_corners=False)

        return cam.squeeze().numpy(), target_class


def visualize_gradcam(image, cam, title="Grad-CAM"):
    """Overlay Grad-CAM heatmap on the original image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Usage
model = models.resnet50(weights='IMAGENET1K_V2')
gradcam = GradCAM(model, model.layer4[-1])
cam, predicted_class = gradcam.generate(input_tensor)
visualize_gradcam(original_image, cam, f"Predicted: {class_names[predicted_class]}")
```

### Activation Visualization

```python
def visualize_activations(model, input_tensor, layers_to_visualize):
    """Visualize intermediate feature maps."""
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    hooks = []
    for name, layer in model.named_modules():
        if name in layers_to_visualize:
            hooks.append(layer.register_forward_hook(get_hook(name)))

    # Forward pass
    model.eval()
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Visualize
    for name, act in activations.items():
        n_channels = min(16, act.shape[1])  # show up to 16 channels
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < n_channels:
                ax.imshow(act[0, i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
        plt.suptitle(f"Activations: {name} (shape: {list(act.shape)})")
        plt.tight_layout()
        plt.show()
```

### Debugging Checklist

```
1. DATA PIPELINE (check FIRST — most bugs are here)
   [ ] Labels are correct (visualize a batch with labels)
   [ ] Normalization is correct (print mean/std of a batch)
   [ ] Augmentation is reasonable (visualize augmented samples)
   [ ] Image format is correct (CHW not HWC, RGB not BGR)
   [ ] Tensor dtype is float32 (not uint8)

2. MODEL ARCHITECTURE
   [ ] Print model summary and verify shapes match expectations
   [ ] Output layer has correct number of classes
   [ ] No dimension mismatches (run a dummy forward pass)

3. LOSS AND OPTIMIZATION
   [ ] Loss function matches the task (CrossEntropy for classification, not MSE)
   [ ] Learning rate is reasonable (start with 1e-3 from scratch, 1e-4 for fine-tuning)
   [ ] Loss decreases after first epoch (if not, likely a bug)
   [ ] Random baseline: initial loss should be $\approx -\ln(1/C)$ for $C$ classes with CrossEntropy

4. TRAINING DYNAMICS
   [ ] Training loss decreasing? If not: LR too high, or bug
   [ ] Validation loss decreasing? If not: overfitting, need regularization/augmentation
   [ ] Gradients flowing? Check for NaN/zero gradients
   [ ] Batch norm in eval mode during validation? (call model.eval())

5. ADVANCED DIAGNOSTICS
   [ ] Activation histograms: no dead neurons (all-zero feature maps)
   [ ] Grad-CAM: model attending to relevant image regions
   [ ] Confusion matrix: which classes are confused?
   [ ] Learning rate schedule: is it appropriate?
```

---

## 14. Quick Reference Tables

### Architecture Comparison

| Model | Year | Params | Top-5 Err | Key Innovation | Depth |
|-------|------|--------|-----------|----------------|-------|
| LeNet-5 | 1998 | 60K | N/A | CNN concept | 5 |
| AlexNet | 2012 | 62M | 15.3% | ReLU, GPU, Dropout | 8 |
| VGG-16 | 2014 | 138M | 7.3% | 3x3 everything | 16 |
| GoogLeNet | 2014 | 6.8M | 6.7% | Inception module | 22 |
| ResNet-50 | 2015 | 25.6M | 3.6% | Skip connections | 50 |
| ResNet-152 | 2015 | 60.2M | 3.6% | Skip connections | 152 |
| DenseNet-121 | 2017 | 8M | ~4.0% | Dense connections | 121 |
| EfficientNet-B0 | 2019 | 5.3M | ~4.5% | Compound scaling | ~18 |
| EfficientNet-B7 | 2019 | 66M | ~2.9% | Compound scaling | ~66 |

### Parameter Count Formulas

| Layer Type | Parameters |
|-----------|-----------|
| Conv2d($C_{in}$, $C_{out}$, $K$) | $(K^2 \cdot C_{in} + 1) \cdot C_{out}$ |
| Conv2d($C_{in}$, $C_{out}$, $K$, bias=False) | $K^2 \cdot C_{in} \cdot C_{out}$ |
| Linear(in, out) | $(\text{in} + 1) \cdot \text{out}$ |
| BatchNorm2d($C$) | $2C$ (learnable) + $2C$ (running stats, not optimized) |
| MaxPool2d / AvgPool2d | 0 |
| AdaptiveAvgPool2d | 0 |
| Dropout | 0 |
| ReLU | 0 |
| Depthwise Conv($C$, $K$) | $(K^2 + 1) \cdot C$ |

### Output Size Formulas

| Operation | Formula |
|----------|---------|
| Conv2d | $\lfloor(H + 2p - K) / s\rfloor + 1$ |
| ConvTranspose2d | $(H - 1) \cdot s - 2p + K + \text{out\_pad}$ |
| MaxPool2d / AvgPool2d | $\lfloor(H + 2p - K) / s\rfloor + 1$ |
| AdaptiveAvgPool2d($n$) | $n$ (any input size) |
| Dilated Conv | $\lfloor(H + 2p - K_{eff}) / s\rfloor + 1$, where $K_{eff} = K + (K-1)(d-1)$ |

### Transfer Learning Decision Matrix

| Dataset Size | Similar Domain | Different Domain |
|-------------|---------------|-----------------|
| Tiny (<500) | Frozen backbone | Frozen backbone + heavy augmentation |
| Small (<5K) | Fine-tune last block | Fine-tune last 2 blocks |
| Medium (<50K) | Fine-tune last 2+ blocks | Full fine-tune, differential LR |
| Large (>50K) | Full fine-tune or from scratch | Full fine-tune |

### Common PyTorch Patterns

```python
# Standard Conv-BN-ReLU block
def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

# Global Average Pooling -> Classification
def gap_classifier(in_features, num_classes, dropout=0.2):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.squeeze(x).view(b, c)
        scale = self.excitation(scale).view(b, c, 1, 1)
        return x * scale

# Learning rate warmup + cosine decay
def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Tensor Shape Conventions

```
PyTorch:    (Batch, Channels, Height, Width)    = NCHW  (default)
TensorFlow: (Batch, Height, Width, Channels)    = NHWC  (default)

PyTorch conv weight: (C_out, C_in, K_h, K_w)
PyTorch conv bias:   (C_out,)
```

### ImageNet Normalization Constants

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD  = [0.229, 0.224, 0.225]  # RGB

# To unnormalize for visualization:
def unnormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
```

---

## Summary: The Ten Commandments of CNNs

1. **Know the output size formula.** $\text{out} = \lfloor(\text{in} + 2 \cdot \text{pad} - \text{kernel}) / \text{stride}\rfloor + 1$.
   You will be asked this in interviews. You will need it when debugging.

2. **Know the parameter count formula.** $\text{params} = (K^2 \cdot C_{in} + 1) \cdot C_{out}$.
   Count parameters for every layer in every architecture you encounter.

3. **Skip connections are mandatory for deep networks.** If you are building anything deeper
   than ~10 layers, use residual connections. There is no downside.

4. **Use batch normalization.** Conv -> BN -> ReLU is the standard pattern. Set `bias=False`
   in the Conv when using BN.

5. **Use 3x3 convolutions.** Stack them for larger receptive fields. Two 3x3 = one 5x5
   receptive field with fewer parameters and more nonlinearity.

6. **End with global average pooling.** Not a massive FC layer. Zero parameters, better
   generalization, input-size agnostic.

7. **Use pretrained models.** For any real-world task with fewer than ~50K images, transfer
   learning will beat training from scratch. Start frozen, progressively unfreeze.

8. **Augment aggressively.** Random crop, horizontal flip, color jitter at minimum.
   CutMix/Mixup for state-of-the-art results. This is free accuracy.

9. **Visualize everything.** Your data, your augmentations, your activations, your Grad-CAMs.
   Most bugs are in the data pipeline, and visualization catches them fast.

10. **Depth + skip connections > width.** ResNet proved this. A deep, narrow network with
    skip connections will generally outperform a shallow, wide network.
