# Assignment 1: Variational Autoencoder from Scratch

## Overview

In this assignment, you will build a Variational Autoencoder (VAE) from the ground up. You will implement every component — the encoder that outputs distribution parameters, the reparameterization trick that enables gradient flow through sampling, and the decoder that reconstructs inputs. You will then train it to generate handwritten digits, explore the latent space, and experiment with the reconstruction-regularization tradeoff.

This is your first generative model. By the end, you will have a system that creates images that never existed in the training set.

**Estimated time:** 8-12 hours

**Prerequisites:** Module 8 Session 1 (VAEs), familiarity with PyTorch, understanding of the ELBO derivation.

---

## Part 1: Implement the VAE Architecture

### 1.1 The Encoder

Build an encoder network that takes a flattened MNIST image (784 dimensions) and outputs two vectors: `mu` and `log_var`, each of dimension `latent_dim`.

**Requirements:**
- Input: (batch_size, 784)
- Hidden layers: at least two fully connected layers with ReLU activations (e.g., 784 -> 512 -> 256)
- Two separate output heads branching from the last hidden layer:
  - `fc_mu`: Linear(256, latent_dim) — outputs the mean of $q(z|x)$
  - `fc_logvar`: Linear(256, latent_dim) — outputs the log-variance of $q(z|x)$

**Why log_var instead of sigma?**
- log_var is unconstrained (can be any real number), so the network can output it directly.
- $\sigma = \exp(0.5 \cdot \text{log\_var})$ is always positive.
- This avoids numerical issues with very small or negative standard deviations.

### 1.2 The Reparameterization Trick

Implement the reparameterization function:

```python
def reparameterize(self, mu, log_var):
    # Your implementation here
    # 1. Compute std from log_var
    # 2. Sample epsilon from N(0, I)
    # 3. Return z = mu + std * epsilon
    pass
```

**Verification:** After implementing, verify that:
- The output z has the same shape as mu and log_var.
- Calling reparameterize twice with the same mu and log_var gives different results (it is stochastic).
- Gradients flow through to mu and log_var (check with a simple backward pass).

### 1.3 The Decoder

Build a decoder network that maps from latent space back to image space.

**Requirements:**
- Input: (batch_size, latent_dim)
- Hidden layers: mirror the encoder (e.g., latent_dim -> 256 -> 512)
- Output layer: Linear(512, 784) followed by Sigmoid (since MNIST pixels are in [0, 1])
- Output: (batch_size, 784)

### 1.4 The Loss Function

Implement the VAE loss as the sum of two terms:

**Reconstruction loss:** Binary cross-entropy between the input and reconstruction, summed over all 784 pixels.

```python
recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
```

**KL divergence:** The closed-form KL between $\mathcal{N}(\mu, \sigma^2)$ and $\mathcal{N}(0, 1)$:

```python
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

**Total loss:**

```python
loss = recon_loss + kl_loss
```

Make sure you understand where each formula comes from. Refer to the ELBO derivation in the notes.

---

## Part 2: Train on MNIST

### 2.1 Data Setup

- Use `torchvision.datasets.MNIST` with standard train/test split.
- Normalize pixels to [0, 1] (they already are after ToTensor()).
- Use a batch size of 128.
- Use latent_dim = 2 initially (this enables 2D visualization).

### 2.2 Training

- Optimizer: Adam with learning rate 1e-3.
- Train for at least 30 epochs.
- Log the total loss, reconstruction loss, and KL loss separately at each epoch.

**What to monitor:**
- The reconstruction loss should decrease steadily.
- The KL loss will initially increase (the model starts using the latent space), then stabilize.
- If the KL loss is near 0 throughout training, you may have posterior collapse (the encoder is ignoring z). Try reducing the learning rate or using KL annealing (gradually increase the KL weight from 0 to 1 over the first few epochs).

### 2.3 Generate New Digits

After training, generate new images by:
1. Sampling $z \sim \mathcal{N}(0, I)$ — a random point in the 2D latent space.
2. Passing z through the decoder.
3. Reshaping the output to 28x28 and displaying it.

Generate a grid of at least 100 samples. Do they look like plausible digits?

---

## Part 3: Visualize the Latent Space

Since we are using latent_dim = 2, we can directly visualize the entire latent space.

### 3.1 Encode the Test Set

Pass all 10,000 test images through the encoder. For each image, record the mean vector mu (ignore log_var for visualization). Plot all 10,000 points in 2D, coloring each point by its digit class (0-9).

**What to look for:**
- Are the digit classes clustered? Are they separated?
- Is the overall distribution roughly Gaussian (centered around the origin)?
- Which digits overlap in latent space? Does this match your intuition about digit similarity?

### 3.2 Decode a Grid of Latent Points

Create a uniform grid of points covering the range [-3, 3] x [-3, 3] in the latent space (e.g., a 20x20 grid). Decode each point and display the resulting images in a grid.

**What to look for:**
- Smooth transitions between digit types as you move through the space.
- Every point in the grid should decode to *something* recognizable (this is the benefit of the KL regularization).
- The center of the space should produce the most "average" looking digits.

---

## Part 4: Latent Space Interpolation

### 4.1 Linear Interpolation

Choose two test images of different digit classes (e.g., a "3" and a "7"). Encode both to get $z_3$ and $z_7$ (use the $\mu$ values). Compute 10 evenly spaced points along the line from $z_3$ to $z_7$:

```python
alphas = torch.linspace(0, 1, 10)
z_interp = [(1 - a) * z_3 + a * z_7 for a in alphas]
```

Decode each interpolated z and display the sequence. You should see a smooth morphing from "3" to "7."

### 4.2 Random Walk

Starting from a random z, take small Gaussian steps in the latent space and decode at each step. Display the sequence. The decoded images should change smoothly.

---

## Part 5: Beta-VAE Experiments

The $\beta$-VAE modifies the loss function:

```python
loss = recon_loss + beta * kl_loss
```

### 5.1 Vary Beta

Train separate VAE models with $\beta = 0.1, 0.5, 1.0, 2.0, 5.0$, and $10.0$. For each:

1. Record the final reconstruction loss and KL loss.
2. Generate a grid of 100 samples.
3. Visualize the latent space (encode the test set and plot).

### 5.2 Analysis

Create a table or plot showing:
- $\beta$ vs. reconstruction quality (measured by reconstruction loss or visual inspection)
- $\beta$ vs. latent space organization (measured by KL divergence or visual inspection)

**Expected observations:**
- Low $\beta$ (0.1): sharp reconstructions, but the latent space may have gaps and be less organized.
- $\beta = 1.0$: the standard VAE tradeoff.
- High $\beta$ (5.0, 10.0): blurrier reconstructions, but the latent space is highly organized with clear separation between classes.

**Key insight to articulate:** The $\beta$ parameter controls the information bottleneck. Higher $\beta$ forces the encoder to compress more aggressively into a distribution close to $\mathcal{N}(0, I)$, losing reconstruction detail but gaining a more structured latent space.

---

## Part 6: Conditional VAE

Implement a Conditional VAE (CVAE) that conditions on the digit class.

### 6.1 Architecture Modifications

- **Encoder input:** Concatenate the one-hot encoded label (10 dimensions) with the image (784 dimensions), giving a 794-dimensional input.
- **Decoder input:** Concatenate the label (10 dimensions) with the latent code z, giving a (latent_dim + 10)-dimensional input.

### 6.2 Training

Train the CVAE on MNIST. The loss function is identical to the standard VAE.

### 6.3 Conditional Generation

Generate digits by specifying a class label:
1. Choose a label (e.g., "5").
2. One-hot encode it.
3. Sample $z \sim \mathcal{N}(0, I)$.
4. Concatenate z with the one-hot label.
5. Pass through the decoder.

Generate 10 samples for each digit class (0-9) and display as a 10x10 grid. Each row should show variations of the same digit.

### 6.4 Conditional Interpolation

Encode two images of the same digit class. Interpolate in latent space while keeping the class label fixed. You should see style variations (thickness, slant, etc.) while the digit identity remains constant.

---

## Deliverables

1. **Code:** A well-organized Python script or Jupyter notebook containing:
   - VAE class definition (encoder, reparameterize, decoder, forward)
   - Loss function implementation
   - Training loop
   - Generation, visualization, and interpolation code
   - Beta-VAE experiments
   - Conditional VAE implementation

2. **Figures:**
   - Reconstructed images vs. originals (side-by-side comparison)
   - Grid of generated samples from the standard VAE
   - 2D latent space plot (color-coded by digit class)
   - Decoded grid covering the latent space
   - Interpolation sequences between digit pairs
   - Beta-VAE comparison (samples and latent spaces for different beta values)
   - Conditional VAE generated samples (10x10 grid)

3. **Written analysis (1-2 pages):**
   - Explain the ELBO in your own words. What does each term encourage?
   - Describe what you observe in the latent space. Why are some digits closer together?
   - Analyze the beta-VAE results. What is the practical tradeoff?
   - Compare the standard VAE and conditional VAE outputs. What does conditioning buy you?

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Correct implementation | 30% | VAE architecture, reparameterization trick, and loss function are correct |
| Training quality | 20% | Model trains successfully, losses converge, generated samples are recognizable |
| Latent space analysis | 20% | Clear visualizations with thoughtful interpretation |
| Beta-VAE experiments | 15% | Multiple beta values tested with clear comparison |
| Conditional VAE | 15% | Working implementation with conditional generation |

---

## Stretch Goals

For those who want to go further:

### S1: Try CIFAR-10

Replace MNIST with CIFAR-10 (32x32 color images). You will need:
- A convolutional encoder and decoder (replace the MLP with Conv2d / ConvTranspose2d layers).
- Larger latent dimension (try 64 or 128).
- More training epochs and possibly learning rate scheduling.

CIFAR-10 is significantly harder than MNIST. Expect blurrier results — this is a known limitation of VAEs on complex data.

### S2: Implement VQ-VAE

Replace the continuous latent space with a discrete codebook (Van den Oord et al., 2017). Key changes:
- The encoder outputs a continuous vector.
- This vector is quantized to the nearest codebook entry.
- The straight-through estimator is used to pass gradients through the quantization step.
- The codebook is updated via exponential moving average.

This eliminates the blurriness problem and is the foundation of modern tokenized image generation.

### S3: Latent Space Arithmetic

With a well-trained VAE, try "concept arithmetic" in latent space:
- Encode several examples of "tilted 1" and "upright 1." Compute the average difference vector.
- Add this "tilt vector" to the encoding of an "upright 7." Decode. Does the 7 become tilted?

This is the same idea as word2vec arithmetic (king - man + woman = queen), but in image space.

---

*This assignment builds the foundation for all generative modeling. The VAE's ideas — latent spaces, reparameterization, variational inference — appear throughout modern deep learning. Master them here.*
