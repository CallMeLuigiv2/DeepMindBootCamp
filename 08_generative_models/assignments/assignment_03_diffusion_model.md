# Assignment 3: Building a Diffusion Model

## Overview

In this assignment, you will build a Denoising Diffusion Probabilistic Model (DDPM) from the ground up. You will implement the forward noising process, a U-Net denoising network, the training loop, and the iterative sampling procedure. You will train the model on MNIST, visualize the denoising process step by step, and experiment with the noise schedule and number of diffusion steps.

This is the most modern of the three generative model families. The training is simple (just predict noise with MSE), but the infrastructure around it — the noise schedule, the U-Net architecture, the sampling loop — requires careful implementation. By the end, you will understand exactly how Stable Diffusion and similar systems work at their core.

**Estimated time:** 12-16 hours

**Prerequisites:** Module 8 Session 3 (Diffusion Models), Assignments 1-2, familiarity with U-Net architecture.

---

## Part 1: The Forward Noising Process

### 1.1 Define the Noise Schedule

Implement a linear noise schedule and precompute all the quantities you will need:

```python
def linear_noise_schedule(T, beta_start=1e-4, beta_end=0.02):
    """
    Returns:
        betas:       (T,) - noise added at each step
        alphas:      (T,) - signal retained at each step (1 - beta)
        alpha_bars:  (T,) - cumulative signal retained from step 0 to step t
    """
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
```

Use T = 1000 as the default number of timesteps.

**Verification:** Plot $\bar{\alpha}_t$ vs. $t$. It should start near 1 (signal preserved) and decrease to near 0 (signal destroyed). For the linear schedule with the default parameters, $\bar{\alpha}_{1000}$ should be close to 0.

### 1.2 Implement the Forward Process

```python
def forward_diffusion(x_0, t, alpha_bars):
    """
    Add noise to x_0 at timestep t.

    Args:
        x_0: (B, C, H, W) - clean images
        t:   (B,) - timestep for each image in the batch
        alpha_bars: (T,) - precomputed cumulative products

    Returns:
        x_t:     (B, C, H, W) - noisy images
        epsilon: (B, C, H, W) - the noise that was added
    """
    # Look up alpha_bar for each sample's timestep
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])[:, None, None, None]
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bars[t])[:, None, None, None]

    # Sample noise
    epsilon = torch.randn_like(x_0)

    # Compute noisy image
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon

    return x_t, epsilon
```

### 1.3 Visualize the Forward Process

Take a single MNIST digit and show its progressive destruction:

1. Compute $x_t$ for $t = 0, 50, 100, 200, 400, 600, 800, 999$.
2. Display all 8 images in a row.
3. Label each with its timestep.

**What to observe:** The digit should be clearly visible at t=0, increasingly blurred and noisy through the middle timesteps, and indistinguishable from random noise by t=999.

Repeat for 3-4 different digits to build intuition.

---

## Part 2: The U-Net Denoising Network

### 2.1 Timestep Embedding

The network needs to know the current noise level. Implement sinusoidal timestep embeddings (the same idea as positional encoding in Transformers):

```python
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) - integer timesteps
        Returns:
            emb: (B, dim) - sinusoidal embedding
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)
```

### 2.2 Residual Block with Time Conditioning

Each block in the U-Net processes spatial features and incorporates the timestep:

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)
```

**Design notes:**
- GroupNorm is preferred over BatchNorm in diffusion models (it does not depend on batch statistics).
- SiLU (Swish) activation is standard in modern diffusion U-Nets.
- The timestep embedding is projected and added to the feature maps after the first convolution.

### 2.3 Build the U-Net

Implement a U-Net suitable for 28x28 grayscale images:

**Architecture outline:**

```
Encoder:
  ResBlock(1 -> 64)   at 28x28
  Downsample           to 14x14
  ResBlock(64 -> 128)  at 14x14
  Downsample           to 7x7

Bottleneck:
  ResBlock(128 -> 256) at 7x7
  (Optional: self-attention here)

Decoder:
  Upsample             to 14x14
  ResBlock(256+128 -> 128) at 14x14  (concatenate skip connection)
  Upsample             to 28x28
  ResBlock(128+64 -> 64)   at 28x28  (concatenate skip connection)

Output:
  GroupNorm -> SiLU -> Conv2d(64 -> 1)
```

**Key details:**
- Downsampling: use nn.Conv2d with stride 2, or nn.MaxPool2d followed by conv.
- Upsampling: use nn.ConvTranspose2d with stride 2, or nn.Upsample(scale_factor=2) followed by conv.
- Skip connections: concatenate encoder features with decoder features along the channel dimension. This is why the decoder ResBlocks have doubled input channels.
- The timestep embedding is passed to every ResBlock.

**Implementation task:** Code the full U-Net class. The forward method should take (x, t) where x is (B, 1, 28, 28) and t is (B,), and return predicted noise of shape (B, 1, 28, 28).

### 2.4 Verify the Architecture

Before training, verify:
1. The output shape matches the input shape: model(x, t).shape == x.shape.
2. The model has a reasonable number of parameters (aim for 1-5 million for MNIST).
3. A forward pass completes without errors for a random batch.

```python
model = UNet(...)
x = torch.randn(16, 1, 28, 28)
t = torch.randint(0, 1000, (16,))
out = model(x, t)
assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Part 3: The DDPM Training Loop

### 3.1 Implement Training

```python
def train_epoch(model, dataloader, optimizer, alpha_bars, T, device):
    model.train()
    total_loss = 0
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        batch_size = batch_x.shape[0]

        # 1. Sample random timesteps
        t = torch.randint(0, T, (batch_size,), device=device)

        # 2. Add noise using the forward process
        x_t, epsilon = forward_diffusion(batch_x, t, alpha_bars)

        # 3. Predict the noise
        epsilon_pred = model(x_t, t)

        # 4. Compute MSE loss
        loss = F.mse_loss(epsilon_pred, epsilon)

        # 5. Backpropagate and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

    return total_loss / len(dataloader.dataset)
```

### 3.2 Training Configuration

- **Optimizer:** Adam with lr=2e-4.
- **Batch size:** 128 (increase if your GPU has memory).
- **Epochs:** At least 50 (100 is better; diffusion models benefit from long training).
- **T:** 1000 timesteps.
- **Noise schedule:** Linear, beta_start=1e-4, beta_end=0.02.

### 3.3 Monitor Training

Log the MSE loss per epoch. Unlike GANs, the training loss for diffusion models is well-behaved: it should decrease steadily and eventually plateau.

There are no competing losses, no adversarial dynamics, no mode collapse. If the loss is not decreasing, check:
- Is the noise schedule correctly implemented?
- Are alpha_bars being indexed correctly with the timestep t?
- Is the timestep embedding reaching all layers of the U-Net?

---

## Part 4: The Sampling Loop

### 4.1 Implement DDPM Sampling

```python
@torch.no_grad()
def sample_ddpm(model, shape, T, betas, alphas, alpha_bars, device):
    """
    Generate samples by iterative denoising from pure noise.

    Args:
        model: trained noise prediction network
        shape: (B, C, H, W) - shape of samples to generate
        T: number of diffusion steps
        betas, alphas, alpha_bars: noise schedule quantities

    Returns:
        x_0: (B, C, H, W) - generated samples
    """
    # Start from pure noise
    x = torch.randn(shape, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict the noise
        epsilon_pred = model(x, t_batch)

        # Compute the mean of p(x_{t-1} | x_t)
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * epsilon_pred
        )

        # Add noise (except at the final step)
        if t > 0:
            sigma = torch.sqrt(beta_t)
            noise = torch.randn_like(x)
            x = mean + sigma * noise
        else:
            x = mean

    return x
```

### 4.2 Generate Samples

After training, generate a grid of 64 samples. Display them. Do they look like MNIST digits?

**Important:** Sampling with T=1000 steps is slow (1000 forward passes). Time it. On a typical GPU, expect 30-60 seconds for a batch of 64 images. This is the main limitation of diffusion models.

### 4.3 Collect the Denoising Trajectory

Modify the sampling loop to save intermediate states. Store $x_t$ at timesteps $t = T, 3T/4, T/2, T/4, T/8, T/16$, and $0$ (or a similar geometric progression).

For a single generated image, display this trajectory as a row of images:
- Far left: pure noise (t = T)
- Progressing through partially denoised states
- Far right: the final clean image (t = 0)

**Generate at least 5 such trajectories and display them.** This visualization is essential for understanding how diffusion works — you should see:
- At high t: random noise, no structure.
- At medium t: large-scale structure emerges (overall shape, rough form).
- At low t: fine details sharpen (edges, texture, specific features).

---

## Part 5: Experiments

### 5.1 Number of Diffusion Steps

Train three models with different values of T:
- T = 100
- T = 500
- T = 1000

Compare:
1. Training loss (does fewer steps make training harder or easier?)
2. Sample quality (generate 64 samples from each, compare visually)
3. Sampling speed (time the generation of 64 samples)

**Expected outcome:** T=1000 should give the best samples but slowest generation. T=100 will be much faster but may produce lower-quality samples. T=500 may be a good middle ground.

### 5.2 Noise Schedule Comparison

Implement a cosine noise schedule (Nichol & Dhariwal, 2021):

```python
def cosine_noise_schedule(T, s=0.008):
    """
    Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models."
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bars = f / f[0]
    # Clip to prevent singularities
    alpha_bars = torch.clamp(alpha_bars, min=1e-5, max=1.0 - 1e-5)
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    betas = torch.clamp(betas, min=1e-5, max=0.999)
    alphas = 1.0 - betas
    alpha_bars = alpha_bars[1:]  # remove step 0
    return betas.float(), alphas.float(), alpha_bars.float()
```

Train with both the linear and cosine schedules (same T, same architecture, same number of epochs). Compare:
1. Plot $\bar{\alpha}_t$ vs. $t$ for both schedules. The cosine schedule should be more gradual.
2. Compare sample quality. The cosine schedule often performs better because it avoids destroying too much information in the final steps.

### 5.3 Qualitative Analysis

Generate 100 samples from your best model. Manually inspect them:
- What fraction look like plausible MNIST digits?
- Are all 10 digit classes represented? (Diffusion models should NOT suffer from mode collapse.)
- Are there any artifacts (smearing, blurriness, incomplete digits)?

Compare your diffusion model samples with the VAE samples from Assignment 1 and the GAN samples from Assignment 2. Which model produces the sharpest samples? Which has the best diversity? Which was easiest to train?

---

## Part 6: Understanding the Noise Prediction

### 6.1 What Does the Network Actually Predict?

Take a training image $x_0$. Add noise at various timesteps to get $x_t$. Run the trained model to get $\epsilon_{\text{pred}}$. Compare:
- The actual noise $\epsilon$ (what was added)
- The predicted noise $\epsilon_{\text{pred}}$ (what the model thinks was added)
- The residual: $\epsilon - \epsilon_{\text{pred}}$ (prediction error)

Display all three for $t = 10, 100, 500, 900$. At which timesteps is the prediction most accurate? At which is it hardest?

**Expected insight:** The model is most accurate at intermediate timesteps. At very small t, the noise is tiny and hard to detect. At very large t, the signal is almost completely destroyed and there is little information to work with.

### 6.2 Predicted x_0

At any timestep, the model's noise prediction implies a prediction of $x_0$:

```python
x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)
```

Visualize $\hat{x}_0$ at different timesteps during sampling. Early in the reverse process (high $t$), $\hat{x}_0$ will be rough and noisy. As $t$ decreases, $\hat{x}_0$ will sharpen. This shows how the model refines its "guess" of the final image over the course of sampling.

---

## Deliverables

1. **Code:** Well-organized scripts or notebooks containing:
   - Noise schedule computation (both linear and cosine)
   - Forward diffusion process
   - U-Net architecture with timestep conditioning
   - DDPM training loop
   - DDPM sampling loop
   - All visualization code
   - Experiment code for varying T and noise schedules

2. **Figures:**
   - Forward process visualization (single image at multiple noise levels)
   - Generated samples (64-image grid from the trained model)
   - Denoising trajectories (at least 5 examples, showing progressive denoising from noise to image)
   - Comparison of T = 100, 500, 1000 (sample quality and timing)
   - Comparison of linear vs. cosine noise schedule (alpha_bar plots and samples)
   - Noise prediction analysis (actual vs. predicted noise at different timesteps)
   - Predicted x_0 at different stages of sampling

3. **Written analysis (1-2 pages):**
   - Describe the forward and reverse processes in your own words.
   - What did the denoising trajectories reveal about how diffusion models generate structure?
   - Compare the effect of T and noise schedule on sample quality and speed.
   - Compare your diffusion model results with your VAE (Assignment 1) and GAN (Assignment 2). Discuss sample quality, diversity, training ease, and speed.

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Forward process implementation | 10% | Correct noise schedule and forward diffusion |
| U-Net implementation | 25% | Correct architecture with timestep conditioning, skip connections |
| Training loop | 15% | Correct training objective, loss decreases, model converges |
| Sampling loop | 20% | Correct DDPM sampling, generates recognizable digits |
| Visualizations | 15% | Denoising trajectories, forward process, noise analysis |
| Experiments and analysis | 15% | T comparison, schedule comparison, thoughtful written analysis |

---

## Stretch Goals

### S1: DDIM Sampling

Implement the DDIM (Denoising Diffusion Implicit Models) sampling method (Song et al., 2020). DDIM is a deterministic variant that allows you to skip steps:

```python
@torch.no_grad()
def sample_ddim(model, shape, T, alpha_bars, device, num_steps=50):
    """
    DDIM sampling with fewer steps than DDPM.

    Args:
        num_steps: number of denoising steps (can be much less than T)
    """
    # Create a subsequence of timesteps
    step_size = T // num_steps
    timesteps = list(range(0, T, step_size))[::-1]  # reversed

    x = torch.randn(shape, device=device)

    for i in range(len(timesteps)):
        t = timesteps[i]
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        epsilon_pred = model(x, t_batch)

        alpha_bar_t = alpha_bars[t]
        alpha_bar_t_prev = alpha_bars[t_prev] if t_prev > 0 else torch.tensor(1.0)

        # Predict x_0
        x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        # DDIM update (deterministic, eta=0)
        x = (torch.sqrt(alpha_bar_t_prev) * x_0_pred +
             torch.sqrt(1 - alpha_bar_t_prev) * epsilon_pred)

    return x
```

Compare DDIM sampling at 50, 20, and 10 steps with DDPM at 1000 steps:


- How much faster is DDIM?
- How does sample quality degrade as you reduce steps?
- Is DDIM with 50 steps close in quality to DDPM with 1000 steps?

### S2: Classifier-Free Guidance

Implement classifier-free guidance for class-conditional generation on MNIST:

1. **Modify the U-Net** to accept an optional class label. Embed the class (0-9) into a vector and add it to the timestep embedding, or concatenate it.

2. **During training**, randomly drop the class label with probability 10% (replace with a "null" class token). This trains the model for both conditional and unconditional generation.

3. **During sampling**, compute two noise predictions at each step:
   - epsilon_uncond = model(x_t, t, null_class)
   - epsilon_cond = model(x_t, t, class_label)
   - epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)

   Use $w = 3.0$ as a starting point.

4. **Generate class-conditional samples** for each digit and compare with unconditional samples. The guided samples should be more clearly recognizable as the specified digit.

5. **Vary the guidance scale $w$** from 1.0 to 10.0. Show how higher guidance produces samples that are more clearly the target class but less diverse.

### S3: Train on Fashion-MNIST or CIFAR-10

Adapt your diffusion model for a harder dataset:

**Fashion-MNIST (28x28 grayscale):** A drop-in replacement for MNIST but harder (shirts, shoes, bags, etc.). Your existing architecture should work with minimal changes. Compare the sample quality with MNIST.

**CIFAR-10 (32x32 color):** Requires:
- Change input channels from 1 to 3.
- Increase model capacity (more channels in the U-Net).
- Train for significantly longer (200+ epochs).
- Use a larger batch size if possible.

This is a meaningful challenge. Published DDPM results on CIFAR-10 use models with ~35M parameters trained for 800K iterations. With a smaller model and less training, your results will be noisy but should show recognizable structure.

### S4: Noise Schedule Search

Implement a parameterized noise schedule and experiment:
- Try a quadratic schedule: betas increase quadratically.
- Try a sigmoid schedule: betas follow a sigmoid curve.
- For each, plot alpha_bar_t and compare sample quality.

The goal is to develop intuition for how the noise schedule affects generation. Does a schedule that preserves information longer (higher $\bar{\alpha}$ at intermediate $t$) produce better results?

---

*Diffusion models are now the backbone of the most capable generative systems in the world. By building one from scratch, you have gained hands-on understanding of the mechanism behind DALL-E, Stable Diffusion, and their successors. The training loop is simple; the magic is in the careful design of the noise schedule, the architecture, and the sampling procedure.*
