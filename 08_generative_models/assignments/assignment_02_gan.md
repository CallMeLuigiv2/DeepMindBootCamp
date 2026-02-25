# Assignment 2: Generative Adversarial Networks

## Overview

In this assignment, you will build GANs of increasing sophistication. You will start with the simplest possible GAN on MNIST, experience the training instabilities firsthand, then implement architectural and objective improvements that address them. By the end, you will have trained a DCGAN and a Wasserstein GAN with gradient penalty, and you will understand viscerally why GAN training is both powerful and difficult.

**Estimated time:** 10-15 hours

**Prerequisites:** Module 8 Session 2 (GANs), Assignment 1 (VAE), PyTorch proficiency.

---

## Part 1: Basic GAN on MNIST

### 1.1 Implement the Generator

Build a simple MLP generator:

**Architecture:**
- Input: $z \sim \mathcal{N}(0, I)$, shape (batch_size, latent_dim) where latent_dim = 100
- Linear(100, 256) -> LeakyReLU(0.2)
- Linear(256, 512) -> LeakyReLU(0.2)
- Linear(512, 1024) -> LeakyReLU(0.2)
- Linear(1024, 784) -> Tanh()
- Output: (batch_size, 784), reshaped to (batch_size, 1, 28, 28)

**Note:** The output uses Tanh, so images are in [-1, 1]. You must normalize MNIST to [-1, 1] accordingly:
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

### 1.2 Implement the Discriminator

Build a simple MLP discriminator:

**Architecture:**
- Input: image (batch_size, 1, 28, 28), flattened to (batch_size, 784)
- Linear(784, 512) -> LeakyReLU(0.2) -> Dropout(0.3)
- Linear(512, 256) -> LeakyReLU(0.2) -> Dropout(0.3)
- Linear(256, 1) -> Sigmoid()
- Output: (batch_size, 1), a probability that the input is real

**Design notes:**
- LeakyReLU (not ReLU) in the discriminator is important. ReLU can cause dead neurons that never recover, crippling the discriminator.
- Dropout in the discriminator acts as a regularizer, preventing it from becoming too powerful too quickly.

### 1.3 Implement the Training Loop

```python
# Pseudocode — you fill in the implementation
for epoch in range(num_epochs):
    for real_images, _ in dataloader:

        # === Train Discriminator ===
        # 1. Forward pass on real images: d_real = D(real_images)
        # 2. Compute real loss: loss_real = BCE(d_real, ones)
        # 3. Generate fake images: z = randn(batch_size, latent_dim); fake = G(z)
        # 4. Forward pass on fake images: d_fake = D(fake.detach())
        #    NOTE: .detach() prevents gradients from flowing into G
        # 5. Compute fake loss: loss_fake = BCE(d_fake, zeros)
        # 6. Total D loss: loss_D = loss_real + loss_fake
        # 7. Backprop and update D

        # === Train Generator ===
        # 1. Generate fake images: z = randn(batch_size, latent_dim); fake = G(z)
        # 2. Forward pass through D: d_fake = D(fake)
        # 3. Generator loss: loss_G = BCE(d_fake, ones)
        #    (G wants D to think fakes are real)
        # 4. Backprop and update G
```

**Optimizer:** Adam with lr=2e-4, betas=(0.5, 0.999) for both G and D. These settings are from the DCGAN paper and work well in practice.

### 1.4 Logging and Monitoring

Log the following at every epoch:
- D loss on real images (should stay low)
- D loss on fake images (should start high, then decrease as G improves)
- G loss (should decrease if G is improving)
- $D(x_{\text{real}})$ — average discriminator output on real images (should stay near 1)
- $D(G(z))$ — average discriminator output on fake images (should increase toward 0.5 at equilibrium)

**Critical diagnostic:** If $D(G(z))$ drops to 0 and stays there, the discriminator has won — the generator receives no gradient signal. If $D(G(z))$ jumps to 1, the generator has found an exploit. Neither extreme is good.

### 1.5 Save Generated Samples

Every 5 epochs, generate a grid of 64 samples and save the image. After training, compile these into a progression showing how generation quality improves over time.

---

## Part 2: Experience the Instabilities

This part is deliberately about *failing*. Understanding GAN failure modes is as important as understanding successes.

### 2.1 Mode Collapse Experiment

Modify the training to make mode collapse more likely:
- Train the discriminator for only 1 step per generator step (this is standard, but combined with the next change...).
- Use a very high learning rate for G (e.g., 1e-3) and a low learning rate for D (e.g., 1e-5).

**Observation task:** Generate 1000 samples. Compute the distribution of predicted digit classes (use a pretrained MNIST classifier, or visually inspect). If the generator has mode-collapsed, most samples will be of 1-3 digit classes.

**Document:** Which modes (digit classes) survive? Does the generator oscillate between modes across epochs?

### 2.2 Discriminator Dominance

Now try the opposite: make the discriminator too strong.
- Train the discriminator for 5 steps per generator step.
- Use a high learning rate for D and a low learning rate for G.

**Observation task:** Plot $D(G(z))$ over training steps. It should rapidly approach 0 and stay there. The generator loss should plateau — $G$ has stopped learning.

### 2.3 Analysis

Write a paragraph for each failure mode explaining:
- What happened and why.
- What the loss curves and $D(G(z))$ values tell you.
- How you would diagnose this in a real project.

---

## Part 3: DCGAN on a Real Dataset

Now build a proper convolutional GAN following the DCGAN guidelines.

### 3.1 Choose a Dataset

Use one of:
- **CIFAR-10** (32x32 color images, 10 classes) — available via torchvision
- **CelebA** (178x218 face images, cropped and resized to 64x64) — available via torchvision or manual download

CelebA is recommended if you have the compute, as faces are the classic GAN benchmark.

### 3.2 Implement the DCGAN Generator

Follow the DCGAN architecture guidelines:

```python
class DCGANGenerator(nn.Module):
    """
    For 64x64 output images.
    Input: (batch_size, latent_dim, 1, 1)
    """
    def __init__(self, latent_dim=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # (latent_dim) -> (ngf*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # -> (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # -> (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # -> (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # -> (nc, 64, 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)
```

**For CIFAR-10 (32x32):** Remove one upsampling layer or start from a 2x2 feature map.

### 3.3 Implement the DCGAN Discriminator

```python
class DCGANDiscriminator(nn.Module):
    """
    For 64x64 input images.
    Output: probability (real or fake)
    """
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # (nc, 64, 64) -> (ndf, 32, 32)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*2, 16, 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*4, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*8, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (1, 1, 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)
```

### 3.4 Weight Initialization

DCGAN requires careful weight initialization:

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)
```

### 3.5 Training

- Optimizer: Adam with lr=2e-4, betas=(0.5, 0.999) for both networks.
- Train for at least 50 epochs on CIFAR-10, or 25 epochs on CelebA.
- Save sample grids every 5 epochs.

### 3.6 Training Progression

Create a figure showing generated image grids at epochs 1, 5, 10, 25, and 50 (or the final epoch). This progression is one of the most satisfying visualizations in deep learning — you can watch the network learn to generate increasingly realistic images.

---

## Part 4: Wasserstein GAN with Gradient Penalty

### 4.1 Modify the Discriminator (Critic)

For WGAN-GP, the discriminator (now called the "critic") has two changes:
1. **Remove the Sigmoid** from the output layer. The critic outputs an unbounded real number, not a probability.
2. **Remove BatchNorm** from the critic. BatchNorm interferes with the gradient penalty. Use LayerNorm or no normalization instead.

### 4.2 Implement the Gradient Penalty

```python
def gradient_penalty(critic, real_images, fake_images, device):
    """
    Compute the gradient penalty for WGAN-GP.
    Penalizes the critic for having gradient norm != 1 on interpolated samples.
    """
    batch_size = real_images.size(0)
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    # Interpolated samples
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    # Critic output on interpolated samples
    critic_interpolated = critic(interpolated)
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    # Flatten and compute norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    # Penalty: (||grad|| - 1)^2
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty
```

### 4.3 Implement the WGAN-GP Training Loop

```python
# Key differences from standard GAN:
# 1. Critic loss: critic(fake) - critic(real) + lambda_gp * gradient_penalty
# 2. Generator loss: -critic(fake)
# 3. Train critic for n_critic=5 steps per generator step
# 4. No log/sigmoid -- raw critic outputs
# 5. lambda_gp = 10 (standard value)

for epoch in range(num_epochs):
    for real_images, _ in dataloader:

        # === Train Critic (5 steps) ===
        for _ in range(n_critic):
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(z).detach()

            critic_real = critic(real_images).mean()
            critic_fake = critic(fake_images).mean()
            gp = gradient_penalty(critic, real_images, fake_images, device)

            loss_critic = critic_fake - critic_real + lambda_gp * gp

            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()

        # === Train Generator (1 step) ===
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(z)
        loss_gen = -critic(fake_images).mean()

        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()
```

### 4.4 Comparison: Standard GAN vs WGAN-GP

Train both a standard DCGAN and a WGAN-GP on the same dataset with the same architecture (modulo the BatchNorm/Sigmoid changes). Compare:

1. **Loss curves:** The WGAN critic loss should correlate with sample quality (lower critic loss = better samples). The standard GAN losses are less interpretable.
2. **Training stability:** Does the WGAN-GP oscillate less? Does it recover from bad states more gracefully?
3. **Sample quality:** Generate 64-image grids from both at the same epoch. Which looks better?
4. **Mode coverage:** Do both models generate all digit classes (MNIST) or diverse faces (CelebA)?

---

## Part 5: Quantitative Evaluation with FID

### 5.1 Understanding FID

The Frechet Inception Distance (FID) measures the distance between the distribution of generated images and real images in the feature space of a pretrained Inception-v3 network.

$$\text{FID} = \|\mu_{\text{real}} - \mu_{\text{gen}}\|^2 + \text{Tr}\left(\Sigma_{\text{real}} + \Sigma_{\text{gen}} - 2(\Sigma_{\text{real}} \Sigma_{\text{gen}})^{1/2}\right)$$

Lower FID = better (0 means the distributions are identical).

### 5.2 Compute FID

Use the `pytorch-fid` library or `torchmetrics`:

```bash
pip install pytorch-fid
```

```python
# Option 1: command-line tool
# Save 10,000 generated images to a directory, then:
# python -m pytorch_fid path/to/real_images path/to/generated_images

# Option 2: torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=2048)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)
print(f"FID: {fid.compute():.2f}")
```

### 5.3 FID Over Training

Compute FID at epochs 1, 10, 25, and 50. Plot FID vs. epoch. It should decrease over time (improving generation quality).

Compare FID between your standard GAN and WGAN-GP. Which achieves a lower FID?

---

## Deliverables

1. **Code:** Well-organized scripts or notebooks containing:
   - Basic GAN implementation (MLP on MNIST)
   - DCGAN implementation (convolutional on CIFAR-10 or CelebA)
   - WGAN-GP implementation with gradient penalty
   - FID computation
   - All training loops with proper logging

2. **Figures:**
   - Training progression: generated sample grids at multiple epochs
   - Loss curves: D loss, G loss, and D(G(z)) over training for each model variant
   - Mode collapse demonstration: samples from the deliberately broken training setup
   - Comparison grid: standard GAN vs. WGAN-GP samples at the same epoch
   - FID plot over training epochs

3. **Written analysis (1-2 pages):**
   - Describe the training dynamics you observed. When did training work well? When did it fail?
   - Explain mode collapse in your own words, with evidence from your experiments.
   - Compare the standard GAN loss and the Wasserstein loss. What practical difference did you observe?
   - What FID scores did you achieve? How do they compare to published benchmarks?

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Basic GAN implementation | 20% | Correct architecture, loss functions, and training loop |
| Instability experiments | 15% | Clear documentation of mode collapse and discriminator dominance |
| DCGAN implementation | 25% | Correct architecture following DCGAN guidelines, trained on real dataset |
| WGAN-GP implementation | 25% | Correct gradient penalty, critic training, and comparison with standard GAN |
| Quantitative evaluation | 15% | FID computation and meaningful comparison between models |

---

## Stretch Goals

### S1: Conditional GAN

Implement a conditional DCGAN that generates images of a specific class:
- Feed the class label to both G and D.
- For G: concatenate the one-hot label with z, or embed it and add to the feature maps.
- For D: concatenate the one-hot label with the image (as an extra channel), or embed and concatenate with features.
- Generate class-specific samples and verify they match the requested class.

### S2: Progressive Growing

Implement the progressive growing technique from Karras et al. (2018):
- Start training at 4x4 resolution.
- After the model stabilizes, add layers for 8x8.
- Continue doubling resolution up to 64x64 or 128x128.
- Use alpha-blending to smoothly transition between resolutions.

This is a significant implementation challenge but will give you deep understanding of how high-resolution GANs work.

### S3: Interpolation in Latent Space

Generate interpolation videos (or image sequences):
- Linear interpolation: z_1 -> z_2 over 60 frames.
- Spherical interpolation (slerp): interpolate on the hypersphere surface. This often gives smoother results because high-dimensional Gaussian samples concentrate on a thin shell.

```python
def slerp(z1, z2, alpha):
    """Spherical linear interpolation between z1 and z2."""
    omega = torch.acos((z1 * z2).sum() / (z1.norm() * z2.norm()))
    return (torch.sin((1 - alpha) * omega) * z1 + torch.sin(alpha * omega) * z2) / torch.sin(omega)
```

### S4: Track Additional Metrics

Implement or compute:
- **Inception Score (IS):** Measures both quality (confident class predictions) and diversity (uniform class distribution). Less reliable than FID but historically important.
- **Precision and Recall for Generative Models:** Separately measure quality (precision: are generated samples realistic?) and diversity (recall: does the generator cover all modes?).

---

*GANs are the most temperamental models you will train in this course. The instabilities are not bugs — they are fundamental to the adversarial framework. Experiencing them firsthand is the best teacher. Once you understand why GANs fail, you will appreciate both the engineering that makes them work and the motivation for diffusion models.*
