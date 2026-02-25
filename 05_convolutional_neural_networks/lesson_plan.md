# Module 5: Convolutional Neural Networks — Lesson Plan

## Weeks 8-9 | The Architecture That Gave Machines Eyes

### Module Philosophy

CNNs are a mature, deeply understood technology. That is precisely why you must learn them
thoroughly. Every modern vision system — even the attention-based ones — inherits ideas that
were first proven in convolutional networks. Skip connections, feature hierarchies, transfer
learning, multi-scale processing: these did not originate with transformers. They were forged
in the CNN era, and they remain essential.

You already understand MLPs and backprop. A CNN is an MLP with two structural priors baked
into its architecture: **spatial locality** (pixels near each other matter more than distant
ones) and **translation equivariance** (a cat is a cat regardless of where it appears in the
image). These two priors reduce parameters by orders of magnitude and give the network the
right inductive bias for visual data.

By the end of this module, you will be able to look at any convolutional architecture and
understand: why each design choice was made, how many parameters it has, what its receptive
field is, and how gradients flow through it. That is the level of understanding DeepMind
expects.

### Prerequisites

- Solid understanding of MLPs (forward pass, backprop, weight updates)
- Comfortable with PyTorch (nn.Module, autograd, training loops)
- Linear algebra fundamentals (matrix multiplication, tensor operations)
- Basic understanding of image representation (H x W x C tensors, pixel values)

---

## Session 1: The Convolution Operation

**Duration:** 3 hours
**Date:** Week 8, Day 1

### Learning Objectives

By the end of this session, you will be able to:
1. Explain exactly why fully-connected layers fail for image data
2. Implement 2D convolution from scratch and verify against PyTorch
3. Calculate output dimensions for any convolution configuration
4. Describe the roles of stride, padding, dilation, and kernel size
5. Explain 1x1 convolutions, depthwise separable convolutions, and transposed convolutions

### Time Allocation

| Block | Duration | Topic |
|-------|----------|-------|
| 1 | 30 min | The failure of fully-connected layers for images |
| 2 | 45 min | Convolution as template matching |
| 3 | 30 min | Stride, padding, and output size arithmetic |
| 4 | 30 min | Multi-channel convolution and feature maps |
| 5 | 25 min | Advanced convolution variants |
| 6 | 20 min | Coding exercises and verification |

### Block 1: Why Fully-Connected Fails (30 min)

**Core argument:** Take a modest 224x224x3 image. Flattened, that is 150,528 input neurons.
A single hidden layer of 1,000 neurons requires 150,528,000 parameters — just for one layer.
This is absurd. But the parameter explosion is not even the worst problem.

**Three failures of FC layers for images:**

1. **Parameter explosion.** A 224x224x3 input to a 4096-neuron FC layer requires ~617M
   parameters. VGG-16 uses 138M for the entire network (and most of those are in the FC
   layers at the end, which everyone agrees was a mistake).

2. **No spatial awareness.** An FC layer treats pixel (0,0) and pixel (223,223) identically
   to adjacent pixels. It has no concept of "nearby." Every spatial relationship must be
   learned from data, which requires exponentially more examples.

3. **No translation equivariance.** If the network learns to recognize a cat in the top-left,
   it must separately learn to recognize that same cat in the bottom-right. Every position
   requires independent learning.

**Exercise:** Calculate the parameter count for a three-layer MLP processing 224x224x3 images
with hidden layers of size [4096, 4096, 1000]. Compare to the total parameter count of
ResNet-50 (25.6M parameters). Discuss what this means.

### Block 2: Convolution as Template Matching (45 min)

**The core intuition:** A convolution slides a small template (the kernel/filter) across the
image, computing a similarity score at every position. Where the template matches the local
pattern, the output is high. Where it does not match, the output is low.

**Walk through on the whiteboard:**
- Start with a 5x5 input, 3x3 kernel
- Show the element-wise multiply and sum at each position
- The output is a 3x3 "feature map" or "activation map"
- Each value tells you "how much does this local patch match the template?"

**Key vocabulary:**
- **Kernel / Filter:** The small weight matrix that slides across the input
- **Feature map / Activation map:** The output of applying one filter
- **Receptive field:** The region of the input that affects a single output neuron

**The convolution formula (2D, single channel):**

```
Output(i, j) = sum over m, n of Input(i+m, j+n) * Kernel(m, n) + bias
```

More precisely, for a kernel of size K x K applied to an input of size H x W:

```
Output(i, j) = sum_{m=0}^{K-1} sum_{n=0}^{K-1} Input(i*s + m, j*s + n) * Kernel(m, n) + b
```

where s is the stride.

**Technical note:** What we call "convolution" in deep learning is technically
cross-correlation. True convolution flips the kernel. In practice this does not matter
because the kernel weights are learned, so the network can learn the flipped version.

**Drawing exercise:** Draw the convolution of a 6x6 input with a 3x3 kernel (stride 1, no
padding). Label every output element. Verify: output is 4x4.

### Block 3: Stride, Padding, and Output Size Arithmetic (30 min)

**The output size formula — memorize this:**

```
output_size = floor((input_size + 2 * padding - kernel_size) / stride) + 1
```

**Padding types:**
- **Valid (no padding):** No zeros added. Output shrinks. `padding = 0`.
- **Same padding:** Enough zeros added so output size equals input size (when stride=1).
  `padding = (kernel_size - 1) / 2` (requires odd kernel size).

**Why same padding matters:** Without it, every convolution layer shrinks the spatial
dimensions. After a few layers, your feature maps vanish. Same padding preserves spatial
dimensions, letting you control downsampling explicitly (via stride or pooling).

**Stride:** Skip positions when sliding the kernel. Stride 2 halves spatial dimensions.
Stride is an alternative to pooling for downsampling.

**Drill exercises:**
- Input: 32x32, kernel: 5x5, stride: 1, padding: 0. Output? **28x28**
- Input: 32x32, kernel: 3x3, stride: 1, padding: 1. Output? **32x32** (same padding)
- Input: 32x32, kernel: 3x3, stride: 2, padding: 1. Output? **16x16**
- Input: 224x224, kernel: 7x7, stride: 2, padding: 3. Output? **112x112** (ResNet first conv)
- Input: 13x13, kernel: 3x3, stride: 2, padding: 0. Output? **6x6**

### Block 4: Multi-Channel Convolution and Feature Maps (30 min)

**The real picture:** Images have 3 channels (RGB). A convolution filter is not 2D — it is
3D: (kernel_h, kernel_w, in_channels). The filter slides across the spatial dimensions but
extends through ALL input channels.

**Multiple output channels:** We apply multiple 3D filters to produce multiple feature maps.
If we apply 64 filters to an RGB input, each filter is 3x3x3 and produces one feature map.
The output has 64 channels.

**Parameter count for a conv layer:**

```
params = (kernel_h * kernel_w * in_channels + 1) * out_channels
                                            ^bias    ^one set per filter
```

**Example:** Conv layer with 3x3 kernel, 64 input channels, 128 output channels:
`(3 * 3 * 64 + 1) * 128 = 73,856 parameters`

**Drawing exercise:** Draw the tensor shapes flowing through:
Input (1, 3, 32, 32) -> Conv2d(3, 16, 3, padding=1) -> Output (1, 16, 32, 32)
Label the kernel tensor shape: (16, 3, 3, 3). Explain each dimension.

### Block 5: Advanced Convolution Variants (25 min)

**1x1 Convolutions (Network in Network):**
A 1x1 convolution seems pointless — it looks at a single pixel. But it operates across ALL
channels at that position. It is a per-pixel fully-connected layer across the channel
dimension. Uses:
- Change the number of channels (dimensionality reduction/expansion)
- Add nonlinearity (followed by ReLU) without changing spatial dimensions
- Cross-channel interaction without spatial mixing
- Used heavily in Inception, ResNet bottleneck blocks, and modern architectures

**Depthwise Separable Convolutions (MobileNet):**
Standard conv: one filter processes all input channels simultaneously.
Depthwise separable: split into two steps:
1. Depthwise conv: one filter PER input channel (no cross-channel mixing)
2. Pointwise conv: 1x1 conv to mix channels

Computational savings: If standard conv costs `K*K*C_in*C_out`, depthwise separable costs
`K*K*C_in + C_in*C_out`. For K=3, C_in=C_out=256: standard = 589,824, separable = 67,840.
That is an 8.7x reduction.

**Transposed Convolutions (Deconvolution):**
Used for upsampling. Instead of shrinking spatial dimensions, it expands them. Think of it as
"going backwards" through a convolution. Used in autoencoders, GANs, and segmentation
networks (U-Net decoder). Beware of checkerboard artifacts — sometimes resize + conv is
preferred.

**Dilated / Atrous Convolutions:**
Insert gaps between kernel elements. A 3x3 kernel with dilation 2 has the same number of
parameters as a standard 3x3 kernel but an effective receptive field of 5x5. Used to
increase receptive field without increasing parameters or reducing resolution. Critical in
semantic segmentation (DeepLab).

### Block 6: Coding Exercises (20 min)

**Exercise 1:** Implement 2D convolution from scratch in NumPy (single channel, single filter).
Verify against `torch.nn.functional.conv2d`.

**Exercise 2:** Create a `nn.Conv2d` layer in PyTorch and manually inspect the weight tensor
shape. Verify it matches your understanding of (out_channels, in_channels, kernel_h, kernel_w).

**Exercise 3:** Apply a known edge-detection kernel (Sobel) to an image using your
implementation. Visualize the result.

### Key Takeaways — Session 1

1. Convolution exploits two priors: locality and translation equivariance.
2. The output size formula is: `out = (in + 2*pad - kernel) / stride + 1`. Know it cold.
3. Parameters per conv layer: `(K*K*C_in + 1) * C_out`. This is tiny compared to FC layers.
4. 1x1 convolutions are fully-connected layers across channels — extremely powerful.
5. Depthwise separable convolutions give huge computational savings with minimal accuracy loss.

---

## Session 2: Pooling and Architecture Patterns

**Duration:** 2.5 hours
**Date:** Week 8, Day 2

### Learning Objectives

By the end of this session, you will be able to:
1. Explain the purpose of pooling and compare pooling strategies
2. Describe the classic CNN pattern and why it works
3. Explain how feature hierarchies emerge in deep CNNs
4. Visualize and interpret learned filters and activations

### Time Allocation

| Block | Duration | Topic |
|-------|----------|-------|
| 1 | 30 min | Pooling operations |
| 2 | 35 min | The classic CNN pattern |
| 3 | 35 min | Feature hierarchies |
| 4 | 30 min | Visualizing what CNNs learn |
| 5 | 20 min | Coding exercises |

### Block 1: Pooling Operations (30 min)

**Max Pooling:**
Take the maximum value in each local window. The most common choice. 2x2 max pool with
stride 2 halves spatial dimensions. It provides:
- Translation invariance (small shifts do not change the max)
- Downsampling (reduces computation for subsequent layers)
- Keeps the strongest activation (the most "detected" feature)

**Average Pooling:**
Take the mean value in each window. Smoother than max pooling. Less common in intermediate
layers but critical in one form:

**Global Average Pooling (GAP):**
Average across the entire spatial extent of each channel. Takes a (B, C, H, W) tensor to
(B, C, 1, 1). Replaces the large FC layers at the end of a network. Introduced in Network
in Network, used in every modern architecture. Advantages:
- No parameters (unlike FC layers)
- Acts as structural regularizer
- Forces each feature map to represent one concept

**Strided Convolution as Alternative to Pooling:**
Instead of conv(stride=1) + pool(stride=2), use conv(stride=2). Learns the downsampling
operation. Used in modern architectures (ResNet uses strided conv in some variants).
The "all convolutional net" paper showed this works well.

**The debate:** Pooling provides exactly the invariance we want (small translations do not
matter). But it also discards spatial information. For classification, this is fine. For
tasks requiring spatial precision (segmentation, detection), you need to be more careful.

**Exercise:** Take a 4x4 tensor. Apply 2x2 max pooling and 2x2 average pooling with stride 2.
Compare the results. When would you prefer one over the other?

### Block 2: The Classic CNN Pattern (35 min)

**The pattern that launched computer vision:**

```
[Input] -> [Conv -> ReLU -> Pool] x N -> [Flatten -> FC -> ReLU] x M -> [Output]
```

**Design principles:**
- Spatial dimensions decrease as you go deeper (pooling or strided conv)
- Channel count increases as you go deeper (more abstract features need more dimensions)
- Typical progression: 32 -> 64 -> 128 -> 256 -> 512 channels

**Why this works:**
- Early layers: large spatial dimensions, few channels. Detect simple, local features
  (edges, corners) — you do not need many channels because simple features are few.
- Deep layers: small spatial dimensions, many channels. Detect complex, abstract features
  (faces, wheels, text) — you need many channels because abstract features are diverse.
- The spatial shrinkage keeps computation manageable even as channels grow.

**Architecture drawing exercise:**
Draw the full tensor shape progression through a simple CNN:
```
Input:                (1,   3, 32, 32)
Conv2d(3, 32, 3, p=1) + ReLU:   (1,  32, 32, 32)
MaxPool2d(2):                    (1,  32, 16, 16)
Conv2d(32, 64, 3, p=1) + ReLU:  (1,  64, 16, 16)
MaxPool2d(2):                    (1,  64,  8,  8)
Conv2d(64, 128, 3, p=1) + ReLU: (1, 128,  8,  8)
MaxPool2d(2):                    (1, 128,  4,  4)
Flatten:                         (1, 2048)
Linear(2048, 256) + ReLU:       (1, 256)
Linear(256, 10):                 (1, 10)
```

Count the parameters at each layer. Note how the FC layer has more parameters than all
conv layers combined — this is the problem GAP solves.

### Block 3: Feature Hierarchies (35 min)

**The key insight of deep CNNs:**

```
Layer 1:  Edges, color gradients
Layer 2:  Corners, textures, simple shapes
Layer 3:  Parts of objects (eyes, wheels, windows)
Layer 4:  Whole objects or large object parts
Layer 5+: Scenes, object relationships, abstract concepts
```

This hierarchy emerges automatically from training. Nobody programs "detect edges in layer 1."
The network discovers that edge detection is useful for building texture detectors, which are
useful for building part detectors, and so on.

**Why this matters for transfer learning:** The early layers (edges, textures) are universal.
Every image dataset needs edge detectors. The later layers are task-specific. This is why
transfer learning works: keep the universal early layers, replace the task-specific later ones.

**Receptive field growth:**
Each layer's receptive field is larger than the previous. A single neuron in layer 5 might
"see" the entire input image, but it builds its understanding through this hierarchy.

Two stacked 3x3 convolutions have a 5x5 receptive field. Three stacked 3x3 convolutions
have a 7x7 receptive field. But with far fewer parameters than a single 7x7 convolution,
and more nonlinearities (more ReLUs in between). This is the VGG insight.

**Discussion:** Why does the hierarchy emerge? Could you train a network where layer 1 detects
faces and layer 5 detects edges? Why or why not? (Answer: The receptive field of layer 1 is
too small to see a face. The hierarchy is forced by the architecture.)

### Block 4: Visualizing What CNNs Learn (30 min)

**Methods for understanding CNN internals:**

1. **Filter visualization:** Plot the learned kernel weights directly. Works best for the
   first layer (filters look like oriented edges, color blobs). Deeper layers are harder to
   interpret directly.

2. **Activation visualization:** Feed an image through the network and plot intermediate
   feature maps. See which spatial regions activate for each filter.

3. **Maximally activating patches:** Find the image patches from the dataset that maximally
   activate each filter. Reveals what each filter is "looking for."

4. **Gradient-based visualization:** Compute gradients of the output with respect to the
   input. Highlights which pixels are most important for the prediction.

5. **Grad-CAM:** Gradient-weighted Class Activation Mapping. Uses the gradients flowing
   into the final convolutional layer to produce a coarse localization map highlighting
   important regions. We will implement this in Assignment 3.

**Exercise:** Load a pretrained VGG-16 from torchvision. Extract and visualize the first
convolutional layer's filters (they are 3x3x3 — visualize them as RGB images). What patterns
do you see?

### Block 5: Coding Exercises (20 min)

**Exercise 1:** Build the simple CNN from Block 2 in PyTorch. Train on CIFAR-10 for 10 epochs.
Report accuracy.

**Exercise 2:** Replace max pooling with strided convolution (stride 2). Compare accuracy and
training time.

**Exercise 3:** Replace the final FC layers with global average pooling. How does accuracy
change? How do parameter counts change?

### Key Takeaways — Session 2

1. Pooling provides downsampling and some translation invariance. GAP replaces FC layers.
2. The classic pattern: spatial dimensions shrink, channels grow.
3. Feature hierarchies (edges -> textures -> parts -> objects) emerge automatically.
4. Visualization reveals that CNNs learn interpretable features, especially in early layers.

---

## Session 3: Landmark Architectures

**Duration:** 3.5 hours
**Date:** Week 8, Day 3

### Learning Objectives

By the end of this session, you will be able to:
1. Describe the key innovation of each landmark CNN architecture
2. Draw the architecture diagrams for LeNet, AlexNet, VGG, Inception, and ResNet
3. Explain why ResNet's skip connections are the most important CNN innovation
4. Trace the historical progression of ideas and how each architecture built on its predecessors

### Time Allocation

| Block | Duration | Topic |
|-------|----------|-------|
| 1 | 25 min | LeNet-5: The Origin |
| 2 | 25 min | AlexNet: The Breakthrough |
| 3 | 25 min | VGGNet: Depth Matters |
| 4 | 30 min | GoogLeNet/Inception: Multi-Scale Processing |
| 5 | 45 min | ResNet: Skip Connections Change Everything |
| 6 | 30 min | DenseNet, EfficientNet, and Modern Designs |
| 7 | 20 min | Comparative analysis exercise |

### Block 1: LeNet-5 — The Origin (1998) (25 min)

**Key innovation:** Demonstrated that gradient-based learning applied to CNNs can recognize
visual patterns (handwritten digits) directly from pixels, without hand-engineered features.

**Architecture:**
```
Input (1, 32, 32)
  -> Conv2d(1, 6, 5)     -> (6, 28, 28)   -> Sigmoid -> AvgPool(2)  -> (6, 14, 14)
  -> Conv2d(6, 16, 5)    -> (16, 10, 10)  -> Sigmoid -> AvgPool(2)  -> (16, 5, 5)
  -> Flatten              -> (400,)
  -> Linear(400, 120)    -> Sigmoid
  -> Linear(120, 84)     -> Sigmoid
  -> Linear(84, 10)
```

**Parameter count:** ~60,000. Tiny by modern standards.

**Why it mattered:** Proved the CNN concept. Used at AT&T/Bell Labs for reading checks.
Showed that learned features beat hand-engineered features for vision. But limited by the
computational power of the late 1990s — could not scale to complex images.

**Historical context:** The "AI winter" following LeNet meant these ideas were largely ignored
for over a decade. The key ingredients missing were: (1) large datasets, (2) GPU computing,
(3) better training techniques (ReLU, dropout, batch norm).

### Block 2: AlexNet — The Breakthrough (2012) (25 min)

**Key innovation:** Won ImageNet 2012 by a massive margin (top-5 error: 15.3% vs 26.2% for
the runner-up). Proved that deep CNNs + GPUs + big data could solve real vision problems.
Launched the modern deep learning era.

**Architecture:**
```
Input (3, 227, 227)
  -> Conv2d(3, 96, 11, stride=4)   -> (96, 55, 55)  -> ReLU -> MaxPool(3, stride=2)
  -> Conv2d(96, 256, 5, pad=2)     -> (256, 27, 27)  -> ReLU -> MaxPool(3, stride=2)
  -> Conv2d(256, 384, 3, pad=1)    -> (384, 13, 13)  -> ReLU
  -> Conv2d(384, 384, 3, pad=1)    -> (384, 13, 13)  -> ReLU
  -> Conv2d(384, 256, 3, pad=1)    -> (256, 13, 13)  -> ReLU -> MaxPool(3, stride=2)
  -> Flatten -> FC(9216, 4096) -> ReLU -> Dropout(0.5)
  -> FC(4096, 4096) -> ReLU -> Dropout(0.5)
  -> FC(4096, 1000)
```

**Parameter count:** ~62M (most in the FC layers).

**Key contributions:**
- ReLU activation (instead of sigmoid/tanh) — faster training, reduced vanishing gradients
- Dropout for regularization — randomly zero out neurons during training
- GPU training — split across two GTX 580 GPUs (the original data parallelism)
- Data augmentation — random crops, horizontal flips, color jittering
- Local Response Normalization (LRN) — obsoleted by batch normalization

**The moment that mattered:** Yann LeCun said the reaction to AlexNet was like "a bomb went
off." Every major tech company immediately pivoted to deep learning.

### Block 3: VGGNet — Depth Matters (2014) (25 min)

**Key innovation:** Showed that network depth is critical for performance, and that you can
build very deep networks using only 3x3 convolutions.

**The 3x3 insight:** Two 3x3 convolutions have the same receptive field as one 5x5
convolution, but with fewer parameters (2 * 3*3 = 18 vs 25) and more nonlinearity (two ReLUs
instead of one). Three 3x3 convolutions equal one 7x7 (3 * 9 = 27 vs 49 parameters).

**VGG-16 Architecture:**
```
Input (3, 224, 224)
  -> [Conv3x3-64] x 2  -> MaxPool  -> (64, 112, 112)
  -> [Conv3x3-128] x 2 -> MaxPool  -> (128, 56, 56)
  -> [Conv3x3-256] x 3 -> MaxPool  -> (256, 28, 28)
  -> [Conv3x3-512] x 3 -> MaxPool  -> (512, 14, 14)
  -> [Conv3x3-512] x 3 -> MaxPool  -> (512, 7, 7)
  -> Flatten -> FC(25088, 4096) -> FC(4096, 4096) -> FC(4096, 1000)
```

**Parameter count:** ~138M. The three FC layers account for ~124M of those.

**Why it mattered:** Elegantly simple design. Proved depth helps. Became the go-to feature
extractor for transfer learning. But: too many parameters, too slow, and going deeper (VGG-19)
showed diminishing returns — or even degradation. Something was wrong.

**The degradation problem:** VGG showed that making networks deeper eventually HURTS
performance. Not because of overfitting (training error also increases) but because deeper
networks are harder to optimize. This set the stage for ResNet.

### Block 4: GoogLeNet / Inception — Multi-Scale Processing (2014) (30 min)

**Key innovation:** Process information at multiple scales simultaneously within each layer,
and use 1x1 convolutions for dimensionality reduction.

**The Inception module:**
```
        Input
       / | | \
     1x1  1x1  1x1  MaxPool3x3
      |    |    |       |
      |  3x3  5x5    1x1
       \  |   |     /
        Concat (depth)
```

Each branch processes the input at a different scale. The 1x1 convolutions before the 3x3
and 5x5 branches reduce channels, dramatically cutting computation.

**Parameter count:** ~6.8M. Vastly more efficient than VGG despite similar accuracy.

**Key ideas:**
- Multi-scale feature extraction within a single layer
- 1x1 convolutions as computational bottlenecks
- Auxiliary classifiers during training (inject gradients at intermediate layers)
- Global average pooling instead of FC layers at the end

**Later versions:**
- Inception v2/v3: Factorized convolutions (5x5 -> two 3x3; 3x3 -> 1x3 + 3x1)
- Inception v4: Combined with residual connections

**Drawing exercise:** Draw the Inception module. Label tensor shapes at each point, assuming
input has 256 channels and each branch outputs 64 channels (after reduction).

### Block 5: ResNet — Skip Connections Change Everything (2015) (45 min)

**This is the most important content in this entire module. Pay close attention.**

**The problem ResNet solved:** Before ResNet, making networks deeper eventually degraded
performance. A 56-layer network performed WORSE than a 20-layer network on both training and
test data. This is not overfitting — it is an optimization problem. Deeper networks are
harder to train because gradients must flow through more layers and can vanish or explode.

**The key insight:** If a deeper network should be at least as good as a shallower one (the
extra layers could just learn identity mappings), then the architecture should make it easy
to learn identity mappings.

**The residual block:**
```
    Input (x)
       |
    Conv -> BN -> ReLU
       |
    Conv -> BN
       |
    + x  <---- skip connection (identity shortcut)
       |
     ReLU
       |
    Output = F(x) + x
```

Instead of learning `H(x)` directly, the network learns the residual `F(x) = H(x) - x`.
If the optimal mapping is close to identity, the network just needs to push `F(x)` toward
zero, which is much easier than learning a complete identity mapping from scratch.

**Why skip connections work — three perspectives:**

1. **Gradient flow:** During backpropagation, gradients can flow directly through the skip
   connection, bypassing the conv layers. Even if the conv layers have vanishing gradients,
   the skip connection provides an unimpeded gradient highway. Mathematically:
   `dL/dx = dL/dout * (dF(x)/dx + 1)`. That `+1` means gradients never vanish completely.

2. **Ensemble interpretation:** A ResNet can be viewed as an ensemble of many paths of
   different lengths. With N residual blocks, there are 2^N possible paths from input to
   output (each block can be "skipped" or "used"). The network implicitly trains an
   exponential number of sub-networks.

3. **Identity mapping ease:** Without skip connections, learning the identity function
   requires the weights to be precisely configured. With skip connections, identity is the
   default — the network only needs to learn the deviation from identity.

**ResNet-18 Architecture:**
```
Input (3, 224, 224)
  -> Conv2d(3, 64, 7, stride=2, pad=3) -> BN -> ReLU -> MaxPool(3, stride=2)
     Output: (64, 56, 56)
  -> ResBlock(64, 64) x 2        -> (64, 56, 56)
  -> ResBlock(64, 128, stride=2) + ResBlock(128, 128)  -> (128, 28, 28)
  -> ResBlock(128, 256, stride=2) + ResBlock(256, 256) -> (256, 14, 14)
  -> ResBlock(256, 512, stride=2) + ResBlock(512, 512) -> (512, 7, 7)
  -> GlobalAvgPool -> (512,)
  -> Linear(512, 1000)
```

**Bottleneck blocks (ResNet-50 and deeper):**
```
    Input (x)                     (256 channels)
       |
    Conv 1x1 (reduce)            (64 channels)  -> BN -> ReLU
       |
    Conv 3x3                     (64 channels)  -> BN -> ReLU
       |
    Conv 1x1 (expand)            (256 channels) -> BN
       |
    + x
       |
     ReLU
```

The 1x1 convolutions reduce and then expand channels, creating a "bottleneck." This reduces
computation dramatically. ResNet-152 (60M params) is trainable thanks to this design.

**Parameter count:** ResNet-18: 11.7M. ResNet-50: 25.6M. ResNet-152: 60.2M.

**Why this mattered historically:** ResNet won ImageNet 2015 with 3.57% top-5 error (human
level is about 5%). It enabled training networks with 100+ layers. The skip connection idea
has been adopted in virtually every subsequent architecture (transformers included — they use
residual connections around every attention and FFN block).

**Crucial exercise:** Draw the gradient flow through a 3-block ResNet (with skip connections)
and a 3-block plain network (without). Show how gradients can flow through the skip
connections even when the conv layers have near-zero gradients.

### Block 6: DenseNet, EfficientNet, and Modern Designs (30 min)

**DenseNet (2017):**
Instead of adding the skip connection (`y = F(x) + x`), concatenate it (`y = [F(x), x]`).
Each layer receives feature maps from ALL preceding layers. Dense connectivity encourages
feature reuse and reduces parameters.
```
Layer 0 -> Layer 1 -> Layer 2 -> Layer 3
  |          |          |
  +--------->+--------->+--------->
  |                     |
  +-------------------->+--------->
  |                                |
  +------------------------------->
```

**EfficientNet (2019):**
Key idea: compound scaling. Previous architectures scaled one dimension (depth, width, or
resolution) independently. EfficientNet scales all three simultaneously with fixed ratios:
- depth: d = alpha^phi
- width: w = beta^phi
- resolution: r = gamma^phi

Where alpha * beta^2 * gamma^2 ~ 2 (so FLOPS roughly double for each increment of phi).

Base architecture (EfficientNet-B0) found via neural architecture search (NAS). Then scaled
up to B1-B7 using the compound scaling rule. B7 achieved state-of-the-art ImageNet accuracy
with fewer parameters than previous models.

**Key building block:** MBConv (Mobile inverted bottleneck convolution):
inverted residual block with depthwise separable convolutions and squeeze-and-excitation.

### Block 7: Comparative Analysis Exercise (20 min)

**Fill in this table:**

| Architecture | Year | Params | Top-5 Error | Key Innovation |
|-------------|------|--------|-------------|----------------|
| LeNet-5 | 1998 | 60K | N/A | CNN concept |
| AlexNet | 2012 | 62M | 15.3% | ReLU + GPU + Dropout |
| VGG-16 | 2014 | 138M | 7.3% | Depth with 3x3 convs |
| GoogLeNet | 2014 | 6.8M | 6.7% | Inception module, 1x1 conv |
| ResNet-152 | 2015 | 60M | 3.6% | Skip connections |
| DenseNet-264 | 2017 | 34M | ~3.4% | Dense connections |
| EfficientNet-B7| 2019 | 66M | ~2.9% | Compound scaling |

**Discussion question:** Notice how parameter count does not correlate linearly with accuracy.
GoogLeNet with 6.8M parameters beats VGG-16 with 138M. What does this tell you about
architecture design?

### Key Takeaways — Session 3

1. The history of CNNs is a story of learning how to train deeper networks.
2. ResNet's skip connections are the single most important CNN innovation — they solved the
   degradation problem and enabled arbitrary depth.
3. Efficiency comes from smart architecture design (1x1 convolutions, bottleneck blocks,
   depthwise separable convolutions), not just throwing parameters at the problem.
4. Every architecture after ResNet uses skip connections in some form.

---

## Session 4: Modern CNN Practices

**Duration:** 2.5 hours
**Date:** Week 9, Day 1

### Learning Objectives

By the end of this session, you will be able to:
1. Explain batch normalization and its placement relative to activation functions
2. Implement effective data augmentation pipelines
3. Apply transfer learning with appropriate strategies for different scenarios
4. Use squeeze-and-excitation blocks and understand channel attention

### Time Allocation

| Block | Duration | Topic |
|-------|----------|-------|
| 1 | 30 min | Batch normalization deep dive |
| 2 | 25 min | Residual connections and SE blocks |
| 3 | 35 min | Data augmentation strategies |
| 4 | 40 min | Transfer learning |
| 5 | 20 min | Coding exercises |

### Block 1: Batch Normalization Deep Dive (30 min)

**What it does:** Normalizes the input to each layer by subtracting the mean and dividing by
the standard deviation, computed over the mini-batch. Then applies a learned scale (gamma) and
shift (beta).

```
BN(x) = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
```

**Why it helps:**
- Reduces internal covariate shift (the distribution of inputs to each layer changes as
  previous layers update — BN stabilizes this)
- Allows higher learning rates (the normalization keeps activations in a stable range)
- Acts as a regularizer (batch statistics add noise, similar to dropout)
- Smooths the loss landscape (recent theory suggests this is the primary mechanism)

**Placement debate:** Conv -> BN -> ReLU (original) vs Conv -> ReLU -> BN. The original
paper puts BN before activation. Some practitioners prefer after. In ResNets, there is also
the "pre-activation" variant (BN -> ReLU -> Conv) which can improve performance in very deep
networks.

**At test time:** Use running averages of mean and variance (accumulated during training),
not batch statistics.

**Limitations:**
- Depends on batch size (small batches -> noisy statistics)
- Not suitable for sequence models (use Layer Norm instead)
- Breaks down with batch size 1 (use Instance Norm or Group Norm)

### Block 2: Residual Connections and Squeeze-and-Excitation Blocks (25 min)

**Residual connections beyond ResNet:**
Skip connections appear everywhere now:
- Transformer blocks: `output = LayerNorm(x + Attention(x))`
- U-Net: skip connections between encoder and decoder
- DenseNet: concatenation-based skip connections

**Squeeze-and-Excitation (SE) Blocks:**
A lightweight attention mechanism for channels. The idea: not all channels are equally
important for every input. Let the network learn to weight channels dynamically.

```
Input (B, C, H, W)
  -> Global Average Pool       -> (B, C, 1, 1)     [Squeeze]
  -> FC(C, C/r) -> ReLU        -> (B, C/r, 1, 1)   [Excitation]
  -> FC(C/r, C) -> Sigmoid     -> (B, C, 1, 1)
  -> Multiply with Input       -> (B, C, H, W)     [Scale]
```

The reduction ratio r (typically 16) keeps the parameter overhead small. SE blocks can be
inserted into any architecture (SE-ResNet, SE-Inception) for consistent improvement at
minimal cost.

### Block 3: Data Augmentation Strategies (35 min)

**Data augmentation is one of the most impactful techniques in computer vision.**

**Basic augmentations (always use these):**
- Random horizontal flip (for natural images, not text or medical images where left/right matters)
- Random crop (with padding) — train on random 32x32 crops of 40x40 padded images
- Random rotation (small angles, 10-15 degrees)
- Color jitter (random changes to brightness, contrast, saturation, hue)

**Advanced augmentations:**
- **Cutout / Random Erasing:** Randomly mask out rectangular regions. Forces the network to
  use the full image, not just the most discriminative part.
- **Mixup:** Linearly interpolate between pairs of training examples AND their labels.
  `x_new = lambda*x1 + (1-lambda)*x2`, `y_new = lambda*y1 + (1-lambda)*y2`.
  Encourages linear behavior between training examples.
- **CutMix:** Cut a patch from one image and paste it onto another. Labels are mixed
  proportionally to the area. Combines the benefits of Cutout and Mixup.

**Implementation in PyTorch:**
```python
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),
])
```

**Key principle:** Augmentation should reflect realistic variations. Do not flip medical
images vertically if that never happens in practice. Do not rotate satellite images 90
degrees unless orientation is truly irrelevant.

### Block 4: Transfer Learning (40 min)

**The most practically important technique in this entire module.**

**Why it works:** CNNs learn hierarchical features. Early layers (edges, textures) are
universal. Later layers are task-specific. By starting with a pretrained network and adapting
it to a new task, you leverage features learned from millions of images.

**Three strategies:**

**Strategy 1: Feature Extraction (Frozen Backbone)**
- Freeze all pretrained layers
- Replace the final classification head
- Train only the new head
- Use when: small dataset, similar domain to pretraining data
- Fast to train, low risk of overfitting

**Strategy 2: Fine-tuning Last Layers**
- Freeze early layers, unfreeze later layers + new head
- Train unfrozen layers with a small learning rate
- Use when: medium dataset, somewhat different domain
- Balances adaptation with feature preservation

**Strategy 3: Full Fine-tuning with Differential Learning Rates**
- Unfreeze everything
- Use much smaller learning rates for early layers, larger for later layers
- Typical: `[1e-5, 1e-4, 1e-3]` for [early, middle, new_head]
- Use when: large dataset or very different domain
- Most flexible, but highest risk of overfitting on small datasets

**Practical recipe:**
1. Always start with Strategy 1 (frozen backbone). Get a baseline.
2. If accuracy is insufficient, try Strategy 2. Unfreeze last 1-2 blocks.
3. If still insufficient, try Strategy 3 with differential LRs.
4. If working well, try progressive resizing: train on small images first, then larger.

**Which pretrained model to use:**
- ResNet-50: Good default. Well understood. Fast.
- EfficientNet-B0 to B4: Better accuracy-efficiency tradeoff.
- For small datasets: smaller models (ResNet-18, EfficientNet-B0) to avoid overfitting.
- For large datasets: larger models benefit from capacity.

### Block 5: Coding Exercises (20 min)

**Exercise 1:** Load a pretrained ResNet-50 from torchvision. Freeze the backbone. Replace the
final FC layer for a 5-class problem. Print the number of trainable vs frozen parameters.

**Exercise 2:** Implement a training pipeline with the data augmentation recipe from Block 3.
Train on a small subset of a dataset (100 images per class). Compare accuracy with and
without augmentation.

**Exercise 3:** Implement differential learning rates using PyTorch parameter groups:
```python
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-2},
])
```

### Key Takeaways — Session 4

1. Batch normalization stabilizes training and enables higher learning rates. Place it
   before the activation (or use pre-activation ResNet design).
2. SE blocks provide lightweight channel attention — easy win for any architecture.
3. Data augmentation is effectively free training data. Use it aggressively.
4. Transfer learning is the most practical technique here. Start frozen, progressively unfreeze.

---

## Session 5: Beyond Classification

**Duration:** 2.5 hours
**Date:** Week 9, Day 2

### Learning Objectives

By the end of this session, you will be able to:
1. Describe the key ideas behind object detection and segmentation architectures
2. Explain the R-CNN family evolution and the YOLO approach
3. Describe U-Net and Feature Pyramid Networks
4. Articulate how the field is transitioning from convolution to attention

### Time Allocation

| Block | Duration | Topic |
|-------|----------|-------|
| 1 | 35 min | Object detection fundamentals |
| 2 | 30 min | Semantic segmentation |
| 3 | 25 min | Feature Pyramid Networks |
| 4 | 30 min | The convolution to attention transition |
| 5 | 20 min | Discussion and exercises |

### Block 1: Object Detection (35 min)

**The task:** Not just "is there a cat?" but "where are the cats (and dogs and people) and
draw bounding boxes around each."

**Two-stage detectors (R-CNN family):**

1. **R-CNN (2014):** Extract ~2000 region proposals (selective search). Warp each to fixed
   size. Run each through a CNN independently. Classify each with an SVM. Extremely slow
   (47 seconds per image).

2. **Fast R-CNN (2015):** Run the CNN once on the whole image. Project region proposals onto
   the feature map. Use ROI pooling to extract fixed-size features per region. Much faster.

3. **Faster R-CNN (2015):** Replace selective search with a Region Proposal Network (RPN)
   that shares features with the detector. The RPN is a small CNN that slides over the
   feature map and proposes regions likely to contain objects. End-to-end trainable. ~5 FPS.

**One-stage detectors (YOLO family):**

**YOLO (You Only Look Once):** Divide the image into a grid. Each grid cell predicts bounding
boxes and class probabilities directly. No region proposal step. A single forward pass of one
network does everything. Much faster (real-time), slightly less accurate for small objects.

**Key tradeoff:** Two-stage = more accurate, slower. One-stage = faster, potentially less
accurate (but the gap has narrowed significantly with YOLOv5+).

**Anchor boxes:** Predefined boxes of various aspect ratios and scales at each spatial
location. The network predicts offsets from these anchors rather than absolute coordinates.
This makes the regression problem easier.

### Block 2: Semantic Segmentation (30 min)

**The task:** Classify every pixel. Not just "there is a cat" but "these exact pixels belong
to the cat."

**Fully Convolutional Network (FCN, 2015):**
- Replace all FC layers with conv layers
- Use transposed convolutions to upsample to original resolution
- Skip connections from early layers to preserve fine spatial detail
- The pioneer of dense prediction architectures

**U-Net (2015):**
```
Encoder (downsample)          Decoder (upsample)
Input  (1, 572, 572)          Output (2, 388, 388)
  Conv-Pool -> (64, 284)        <- UpConv + Concat + Conv (64)
  Conv-Pool -> (128, 140)       <- UpConv + Concat + Conv (128)
  Conv-Pool -> (256, 68)        <- UpConv + Concat + Conv (256)
  Conv-Pool -> (512, 32)        <- UpConv + Concat + Conv (512)
          Bottleneck (1024, 16)
```

**Key innovation:** Skip connections between encoder and decoder at matching resolutions.
The encoder captures "what" (semantics), the decoder needs "where" (spatial precision).
Skip connections give the decoder access to fine-grained spatial information from the encoder.

**Why U-Net is so influential:** Simple, effective, widely used in medical imaging,
satellite imagery, and any task requiring pixel-level prediction.

**DeepLab (2016-2018):**
Uses dilated convolutions to maintain spatial resolution while increasing receptive field.
Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale context. State-of-the-art
segmentation for years.

### Block 3: Feature Pyramid Networks (25 min)

**The problem:** Objects in images appear at vastly different scales. A small face and a large
face need different feature map resolutions.

**FPN Architecture:**
```
Bottom-up (encoder)       Top-down (decoder)       Lateral connections
C1 (high res, low semantic)
C2 ----------------------> P2 (high res + high semantic)
C3 ----------------------> P3
C4 ----------------------> P4
C5 (low res, high semantic) -> P5 (low res + high semantic)
```

**How it works:**
1. Bottom-up pathway: standard CNN feature extraction (e.g., ResNet)
2. Top-down pathway: upsample coarse features to higher resolution
3. Lateral connections: 1x1 convs from bottom-up to match channels, then add to top-down

**Result:** Multi-scale feature maps where every level has rich semantics. Small objects
detected at high-res levels, large objects at low-res levels. Used in Faster R-CNN + FPN
for state-of-the-art detection.

### Block 4: The Convolution to Attention Transition (30 min)

**The shift:** Starting around 2020, attention-based models began matching or exceeding CNNs
on vision tasks.

**Vision Transformer (ViT, 2020):**
- Split image into patches (e.g., 16x16)
- Flatten patches, linearly embed them
- Process with a standard transformer encoder
- No convolutions at all

**Key finding:** With enough data (JFT-300M), ViT matches or beats CNNs. With less data,
CNNs are still better (they have stronger inductive biases). The inductive biases of
convolutions (locality, translation equivariance) act as a form of regularization that helps
with limited data.

**Hybrid approaches:**
- **ConvNeXt (2022):** Modernized CNN inspired by transformer design choices (larger kernels,
  layer norm, fewer activation functions). Achieves ViT-level accuracy with pure convolution.
  Shows that it was not attention per se, but the training recipes and design choices.
- **CoAtNet:** Combines convolution (early layers) with attention (later layers).

**The lesson for practitioners:**
- For most practical tasks: pretrained CNNs with transfer learning remain the easiest path.
- For cutting-edge performance: Vision Transformers or hybrids, but they need more data.
- The architectural ideas from CNNs (skip connections, multi-scale processing, feature
  hierarchies) appear in every vision model, attention-based or not.
- Convolutions are not going away — they are being complemented.

### Block 5: Discussion and Exercises (20 min)

**Discussion:** Why did the convolution-to-attention transition happen? What do attention
mechanisms capture that convolutions miss? (Long-range dependencies without stacking many
layers. Global context from the first layer.)

**Exercise 1:** Draw the data flow through a Faster R-CNN pipeline. Label the RPN, ROI
pooling, and classification/regression heads.

**Exercise 2:** Draw the U-Net architecture. Show tensor shapes at each level.

### Key Takeaways — Session 5

1. Object detection extends classification with localization. R-CNN family and YOLO represent
   two fundamentally different approaches (two-stage vs one-stage).
2. Semantic segmentation requires encoder-decoder architectures with skip connections (U-Net).
3. FPN provides multi-scale features — critical for detecting objects at different sizes.
4. Attention is complementing, not replacing, convolution. The core CNN ideas persist.

---

## Session 6: Practical CNN Engineering

**Duration:** 2.5 hours
**Date:** Week 9, Day 3

### Learning Objectives

By the end of this session, you will be able to:
1. Design a CNN for a new task from scratch (or choose an existing one)
2. Debug CNN training by visualizing activations and gradients
3. Implement Grad-CAM for model interpretability
4. Make informed architecture choices for real-world problems

### Time Allocation

| Block | Duration | Topic |
|-------|----------|-------|
| 1 | 35 min | Designing a CNN for a new task |
| 2 | 25 min | Architecture search intuition |
| 3 | 30 min | When and how to use pretrained models |
| 4 | 35 min | Debugging CNNs |
| 5 | 25 min | Practical exercise: end-to-end pipeline |

### Block 1: Designing a CNN for a New Task (35 min)

**The decision framework:**

```
New vision task
    |
    v
Do you have >10K labeled images?
    |           |
   Yes          No --> Use pretrained model (transfer learning)
    |
    v
Is your task similar to ImageNet?
    |           |
   Yes          No
    |           |
    v           v
  Fine-tune   Train from scratch, but still consider
  pretrained   initializing from pretrained weights
```

**If designing from scratch, follow these principles:**

1. **Start simple.** A 5-layer CNN with standard design gets you 80% of the way.
2. **Use 3x3 convolutions** unless you have a specific reason for larger kernels.
3. **Double channels when you halve spatial dimensions.**
4. **Use batch normalization after every convolution.**
5. **Use ReLU (or its variants) everywhere.**
6. **Use skip connections if depth > 10 layers.** There is no reason not to.
7. **End with global average pooling, not a large FC layer.**
8. **Add dropout (0.2-0.5) before the final classification layer.**

**For specific domains:**
- **Medical imaging:** Usually small datasets. Transfer learning from ImageNet still works
  surprisingly well (features like edges and textures transfer). Use aggressive augmentation.
- **Satellite imagery:** May need larger input sizes. Multi-spectral data may require
  modifying the first conv layer (more input channels).
- **Video:** 3D convolutions (Conv3D) or 2D CNN + temporal modeling.
- **Small images (CIFAR-10 scale):** Skip the initial large-kernel downsampling used in
  ImageNet architectures. Start with 3x3 conv, stride 1, padding 1.

### Block 2: Architecture Search Intuition (25 min)

**Neural Architecture Search (NAS)** automates architecture design. Understanding the search
space gives you intuition for manual design.

**Key design choices NAS explores:**
- Kernel size at each layer (3x3, 5x5, 7x7)
- Number of channels at each layer
- Type of connection (residual add, dense concat, none)
- Type of normalization (batch norm, group norm, layer norm)
- Activation function (ReLU, SiLU/Swish, GELU)
- Squeeze-and-excitation: yes/no, reduction ratio

**What NAS has taught us:**
- Inverted residual blocks (expand then contract) work well for efficient models
- SiLU/Swish activation slightly outperforms ReLU in deeper models
- SE blocks are almost always worth the small overhead
- Compound scaling (EfficientNet) is more effective than scaling one dimension

**For practical work:** Do not run NAS. Use the architectures it has already found
(EfficientNet, RegNet). Focus your time on data quality, augmentation, and training recipes.

### Block 3: When and How to Use Pretrained Models (30 min)

**Decision matrix:**

| Your Data Size | Domain Similarity | Strategy |
|----------------|-------------------|----------|
| Small (<1K) | Similar | Feature extraction (frozen) |
| Small (<1K) | Different | Feature extraction + strong augmentation |
| Medium (1K-50K) | Similar | Fine-tune last blocks |
| Medium (1K-50K) | Different | Fine-tune all with differential LR |
| Large (>50K) | Similar | Full fine-tune or train from scratch |
| Large (>50K) | Different | Full fine-tune with pretrained init |

**Practical tips:**
- Always normalize inputs with the pretrained model's mean/std (usually ImageNet stats).
- When replacing the classification head, initialize it randomly (default PyTorch init is fine).
- Use a learning rate warmup for the first few epochs when fine-tuning.
- Monitor both training and validation loss. If training loss drops fast but validation loss
  does not, you are overfitting — freeze more layers or add more augmentation.

**Progressive resizing:**
1. Train on small images (e.g., 128x128) for speed
2. Increase resolution (e.g., 224x224) and fine-tune
3. Optionally increase again (e.g., 320x320)

This acts as a form of regularization and curriculum learning.

### Block 4: Debugging CNNs (35 min)

**When your CNN is not working, systematically check these:**

**1. Data pipeline issues (most common):**
- Are labels correct? Visually inspect a batch of images with their labels.
- Is normalization correct? Print mean and std of a batch after transforms.
- Is augmentation reasonable? Visualize augmented samples.
- Are images in the right format? (CHW vs HWC, RGB vs BGR, [0,1] vs [0,255])

**2. Architecture issues:**
- Are spatial dimensions collapsing too fast? Print shape after each layer.
- Is the network too shallow for the task? Try adding layers.
- Is the network too deep for the dataset? Try removing layers or adding more dropout.

**3. Training issues:**
- Is the learning rate appropriate? Start with 1e-3 for training from scratch, 1e-4 for
  fine-tuning. Use a learning rate finder.
- Is the loss decreasing? If not at all, there may be a bug. A randomly initialized
  network should quickly do better than chance.

**Visualization tools:**

**Activations:** Hook into intermediate layers and plot the feature maps. Dead feature maps
(all zeros) indicate dead ReLU neurons — try a smaller learning rate or use LeakyReLU.

```python
activations = {}
def hook_fn(module, input, output):
    activations[module] = output.detach()
model.layer3.register_forward_hook(hook_fn)
```

**Grad-CAM (Gradient-weighted Class Activation Mapping):**
1. Forward pass: get the prediction
2. Backward pass: compute gradients of the predicted class score w.r.t. the final conv layer
3. Global average pool the gradients over spatial dimensions -> channel weights
4. Weighted sum of the feature maps using these weights
5. Apply ReLU (we only care about positive influence)
6. Upsample to input size and overlay on the original image

Grad-CAM shows you WHERE the network is looking to make its decision. If it is looking at
the wrong region, your model has a problem (possibly a spurious correlation in the data).

### Block 5: Practical Exercise (25 min)

**End-to-end pipeline exercise:**

Given a new image classification task (food classification, 10 classes, 500 images per class):

1. Choose a pretrained model and justify your choice
2. Design the data augmentation pipeline
3. Choose a transfer learning strategy
4. Write the training loop with proper validation
5. Implement Grad-CAM for the final model
6. Debug a deliberately broken version (wrong normalization, labels shuffled)

This exercise synthesizes everything from the module.

### Key Takeaways — Session 6

1. For most real-world tasks, start with transfer learning. Train from scratch only with
   large datasets and specific requirements.
2. Architecture choice matters less than you think. Training recipe, data quality, and
   augmentation matter more.
3. Always visualize your data, your augmentations, and your model's activations. Most bugs
   are in the data pipeline.
4. Grad-CAM is an essential tool for understanding and trusting your model.

---

## Module Assessment Criteria

### What mastery looks like:

1. **Architecture literacy:** Given any CNN architecture diagram, you can count parameters,
   compute output sizes at each layer, and trace gradient flow.

2. **Historical understanding:** You can explain the progression LeNet -> AlexNet -> VGG ->
   Inception -> ResNet -> EfficientNet and articulate why each step was necessary.

3. **Transfer learning fluency:** Given a new task description and dataset size, you can
   immediately choose the right pretrained model, transfer strategy, and training recipe.

4. **Implementation skill:** You can implement convolution from scratch, build ResNet-18
   from scratch, and set up a complete transfer learning pipeline with augmentation and
   Grad-CAM.

5. **Debugging ability:** You can diagnose CNN training failures by inspecting data
   pipelines, visualizing activations, and analyzing learning curves.

---

## Recommended Resources

**Papers (read in this order):**
1. LeCun et al. (1998) — "Gradient-Based Learning Applied to Document Recognition" (LeNet)
2. Krizhevsky et al. (2012) — "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)
3. Simonyan & Zisserman (2014) — "Very Deep Convolutional Networks" (VGG)
4. Szegedy et al. (2015) — "Going Deeper with Convolutions" (Inception/GoogLeNet)
5. He et al. (2016) — "Deep Residual Learning for Image Recognition" (ResNet)
6. He et al. (2016) — "Identity Mappings in Deep Residual Networks" (pre-activation ResNet)
7. Tan & Le (2019) — "EfficientNet: Rethinking Model Scaling" (EfficientNet)
8. Liu et al. (2022) — "A ConvNet for the 2020s" (ConvNeXt)

**Textbooks:**
- Goodfellow, Bengio, Courville — Deep Learning, Chapter 9
- Prince — Understanding Deep Learning, Chapters 10-11

**Online:**
- CS231n (Stanford) CNN lectures and notes
- The Illustrated Transformer (for understanding the conv -> attention transition)
