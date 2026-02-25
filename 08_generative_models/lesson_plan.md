# Module 8: Generative Models — Lesson Plan

## Week 13: Learning to Generate

**Prerequisites:** Modules 1-7 (foundations through Transformers). The apprentice should be comfortable with encoder-decoder architectures, latent representations, KL divergence, and gradient-based optimization.

**Module Philosophy:** Everything you have learned so far has been *discriminative* — mapping inputs to labels, predicting the next token, attending to relevant features. Now we cross a fundamental boundary. Generative models learn the data distribution itself. They don't just recognize a face; they can *create* one that never existed. This is where deep learning meets creativity, and where some of the most active frontier research lives today.

---

## Session 1: Variational Autoencoders (VAEs)

**Duration:** 3 hours

### Learning Objectives

By the end of this session, the apprentice will be able to:

1. Explain why vanilla autoencoders produce poor generative models and what "holes" in the latent space mean concretely.
2. Derive the Evidence Lower Bound (ELBO) from first principles, starting from the goal of maximizing log p(x).
3. Implement the reparameterization trick and explain precisely why sampling breaks the computation graph without it.
4. Train a VAE on MNIST and generate novel samples by decoding random latent vectors.
5. Articulate the reconstruction-vs-regularization tradeoff and how beta-VAE controls it.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 30 min | Autoencoder review and motivation for generation |
| 2 | 45 min | From autoencoders to variational inference |
| 3 | 45 min | The ELBO derivation — full walkthrough |
| 4 | 30 min | The reparameterization trick |
| 5 | 30 min | Implementation and coding exercise |

### Block 1: Autoencoder Review and the Generation Problem (30 min)

**Core idea:** An autoencoder learns to compress data into a bottleneck (the latent code $z$) and reconstruct it. The encoder maps $x \to z$, the decoder maps $z \to \hat{x}$, and we minimize reconstruction loss $\|x - \hat{x}\|^2$.

**The problem with vanilla autoencoders for generation:**

Draw the latent space of a trained autoencoder on MNIST. The encoder maps each training image to a specific point in $z$-space. But these points are scattered irregularly. If you sample a random $z$ that falls *between* encoded training points, the decoder produces garbage — it was never trained on that region.

Walk through a concrete example:
- Encode digit "3" -> z = [1.2, 0.8]
- Encode digit "8" -> z = [3.1, -0.5]
- Decode z = [2.0, 0.1] (midpoint) -> blurry nonsense

The latent space has "holes" — regions with no training signal. The autoencoder memorizes, it does not learn a smooth generative manifold.

**Key question to pose:** "How do we force the latent space to be smooth and complete, so that every point decodes to something meaningful?"

### Block 2: From Autoencoders to Variational Inference (45 min)

**The big idea:** Instead of encoding each input to a *point* $z$, encode it to a *distribution* over $z$. Specifically, the encoder outputs the parameters (mean $\mu$ and variance $\sigma^2$) of a Gaussian distribution $q(z|x) = \mathcal{N}(\mu, \sigma^2)$.

**Why distributions fix the problem:**
- Each input now "claims" a region of latent space, not just a point.
- We add a regularizer that pushes these distributions toward a standard normal $\mathcal{N}(0, I)$.
- This forces the distributions to overlap, filling in the gaps.
- Now every region of latent space has been "trained on" — sampling anywhere produces meaningful outputs.

**Variational inference intuition:**

We want to model $p(x)$ — the true distribution of images. We introduce a latent variable $z$ such that:

$$p(x) = \int p(x|z) \, p(z) \, dz$$

This integral is intractable (we'd need to integrate over all possible $z$). So we introduce an approximate posterior $q(z|x)$ that we can actually compute. The gap between our approximation and reality is measured by KL divergence.

**Key papers to reference:**
- Kingma & Welling, "Auto-Encoding Variational Bayes" (2013) — the original VAE paper
- Rezende et al., "Stochastic Backpropagation and Approximate Inference" (2014) — independent co-discovery

### Block 3: The ELBO Derivation (45 min)

This is the mathematical heart of the session. Go slowly. Every step must land.

**Starting point:** We want to maximize $\log p(x)$ for each training example $x$.

**Step 1:** Introduce the latent variable.

$$\log p(x) = \log \int p(x, z) \, dz$$

**Step 2:** This integral is intractable. Introduce $q(z|x)$, multiply and divide:

$$\log p(x) = \log \int \frac{p(x, z)}{q(z|x)} \cdot q(z|x) \, dz$$

**Step 3:** Apply Jensen's inequality (log of expectation $\geq$ expectation of log):

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x, z) - \log q(z|x)]$$

**Step 4:** Expand $p(x, z) = p(x|z) \cdot p(z)$:

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

This is the ELBO. Spend significant time on what each term means:
- $\mathbb{E}_{q(z|x)}[\log p(x|z)]$: "How well can the decoder reconstruct $x$ from $z$ samples drawn from the encoder?" This is the *reconstruction term*.
- $D_{KL}(q(z|x) \| p(z))$: "How far is the encoder's output distribution from the prior?" This is the *regularization term*.

**Derivation walkthrough on the board:**
1. Write $\log p(x) = \log p(x) \cdot \int q(z|x) \, dz$ (since the integral equals 1)
2. Bring $\log p(x)$ inside: $\int q(z|x) \log p(x) \, dz$
3. Use Bayes' rule: $\log p(x) = \log p(z|x) - \log p(z) + \log p(x)$ ... (alternative derivation path)
4. Show the gap: $\log p(x) = \text{ELBO} + D_{KL}(q(z|x) \| p(z|x))$, where the second term is always $\geq 0$
5. Therefore maximizing the ELBO is a valid proxy for maximizing $\log p(x)$

**The $\beta$-VAE extension:**

$$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) \| p(z))$$

When $\beta > 1$, we emphasize regularization — the latent space becomes more organized but reconstructions become blurrier. When $\beta < 1$, we get sharper reconstructions but a less structured latent space. This tradeoff is fundamental.

### Block 4: The Reparameterization Trick (30 min)

**The problem:** To compute the reconstruction term, we need to sample $z \sim q(z|x) = \mathcal{N}(\mu, \sigma^2)$. But sampling is not differentiable — you can't backpropagate through a random number generator.

**The solution:** Instead of sampling $z$ directly from $\mathcal{N}(\mu, \sigma^2)$, we:
1. Sample $\epsilon \sim \mathcal{N}(0, I)$ — this is independent of the model parameters
2. Compute $z = \mu + \sigma \cdot \epsilon$

Now $z$ is a deterministic function of $\mu$, $\sigma$, and $\epsilon$. The gradients flow through $\mu$ and $\sigma$ just fine. The randomness is "externalized" into $\epsilon$.

**Draw the computation graph both ways:**
- Without trick: $x \to \text{encoder} \to \mu, \sigma \to [\text{SAMPLE}] \to z \to \text{decoder} \to \hat{x}$ (gradient blocked at [SAMPLE])
- With trick: $x \to \text{encoder} \to \mu, \sigma$; $\epsilon \sim \mathcal{N}(0,I)$; $z = \mu + \sigma \cdot \epsilon \to \text{decoder} \to \hat{x}$ (gradient flows through $\mu$ and $\sigma$)

**Common mistake to flag:** Students often write $\sigma$ directly. In practice, the encoder outputs $\log(\sigma^2) = \text{log\_var}$, and we compute $\sigma = \exp(0.5 \cdot \text{log\_var})$. This ensures $\sigma$ is always positive and is more numerically stable.

### Block 5: Implementation and Coding Exercise (30 min)

Live-code a minimal VAE on MNIST:

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```

**Coding exercise:** Complete the loss function. Generate digits. Visualize the 2D latent space.

### Key Papers for This Session

1. Kingma & Welling (2013). "Auto-Encoding Variational Bayes." arXiv:1312.6114.
2. Rezende, Mohamed & Wierstra (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models." ICML.
3. Higgins et al. (2017). "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." ICLR.
4. van den Oord et al. (2017). "Neural Discrete Representation Learning." (VQ-VAE). NeurIPS.

---

## Session 2: Generative Adversarial Networks (GANs)

**Duration:** 3 hours

### Learning Objectives

By the end of this session, the apprentice will be able to:

1. Formulate the GAN objective as a minimax game and explain what Nash equilibrium means in this context.
2. Identify and explain the three major GAN training pathologies: mode collapse, vanishing gradients, and oscillation.
3. Derive the Wasserstein distance motivation and explain why it provides more useful gradients.
4. Implement a DCGAN with the standard architectural guidelines.
5. Evaluate generated samples qualitatively and understand the FID metric.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 30 min | The adversarial game — intuition and formulation |
| 2 | 40 min | The original GAN loss and its problems |
| 3 | 40 min | Wasserstein GAN and the gradient penalty |
| 4 | 30 min | Architectural evolution: DCGAN to StyleGAN |
| 5 | 40 min | Coding exercise and debugging GAN training |

### Block 1: The Adversarial Game (30 min)

**The core metaphor:** A counterfeiter (generator G) tries to produce fake banknotes. A detective (discriminator D) tries to distinguish real from fake. As the detective gets better, the counterfeiter must improve. As the counterfeiter improves, the detective must sharpen. This competition drives both to excellence.

**Formal setup:**
- $G$ takes random noise $z \sim p(z)$ and produces a fake sample $G(z)$
- $D$ takes a sample (real or fake) and outputs $D(x) \in [0, 1]$ — the probability the sample is real
- $G$ wants to fool $D$: make $D(G(z))$ close to 1
- $D$ wants to be correct: $D(x_{\text{real}})$ close to 1, $D(G(z))$ close to 0

**The minimax objective:**

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

**Nash equilibrium:** The game reaches equilibrium when $G$ produces samples indistinguishable from real data ($p_G = p_{\text{data}}$), and $D$ outputs $1/2$ for everything (it literally cannot tell). In practice, we never perfectly reach this equilibrium.

**Key paper:** Goodfellow et al. (2014). "Generative Adversarial Nets." NeurIPS.

### Block 2: The Original GAN Loss and Its Problems (40 min)

**Discriminator training:** Fix $G$, maximize:

$$\mathcal{L}_D = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

This is binary cross-entropy. The discriminator is just a binary classifier: real vs fake.

**Generator training:** Fix $D$, minimize:

$$\mathcal{L}_G = \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

Or equivalently (and more commonly used in practice), maximize:

$$\mathcal{L}_G = \mathbb{E}_{z \sim p(z)}[\log D(G(z))]$$

This alternative form has better gradient properties early in training.

**Problem 1 — Mode collapse:** The generator finds one output that fools the discriminator and produces only that. Example: on MNIST, G learns to produce perfect "7"s and nothing else. It has collapsed to a single mode of the data distribution.

Walk through the dynamics: D learns to reject "7"s -> G switches to "3"s -> D learns to reject "3"s -> G switches back to "7"s. The generator oscillates between modes instead of covering all of them.

**Problem 2 — Vanishing gradients:** If $D$ becomes too good ($D(G(z))$ approaches 0), then $\log(1 - D(G(z)))$ saturates and the gradient for $G$ vanishes. The generator receives no learning signal. This is the fundamental problem: a perfect discriminator is useless for training the generator.

**Problem 3 — Oscillation:** Neither network converges; they chase each other in parameter space. D gets better, then G adjusts, then D has to readjust. There is no guarantee of convergence with gradient descent on minimax objectives.

**Practical lesson:** GAN training is an art. You must carefully balance D and G training rates, monitor both losses, and use architectural tricks. This is why GANs were considered difficult for years.

### Block 3: Wasserstein GAN and the Gradient Penalty (40 min)

**The key insight (Arjovsky et al., 2017):** The Jensen-Shannon divergence (implicit in the original GAN loss) behaves poorly when the real and generated distributions have non-overlapping support (which is almost always the case in high dimensions). The gradient is either zero or undefined.

**Wasserstein distance (Earth Mover's distance):** Intuitively, it measures the minimum "work" to transform one distribution into another. Think of it as: if $p_{\text{data}}$ and $p_G$ are piles of dirt, how much dirt do you need to move and how far?

**Why it is better:** Wasserstein distance provides meaningful gradients even when distributions don't overlap. It decreases smoothly as G improves, giving a useful training signal throughout.

**The WGAN objective:**

$$\min_G \max_{D \in \text{1-Lipschitz}} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p(z)}[D(G(z))]$$

The discriminator (now called "critic") must be 1-Lipschitz: its gradient norm must be at most 1 everywhere.

**Enforcing Lipschitz constraint — gradient penalty (WGAN-GP):**

$$\mathcal{L}_D = \mathbb{E}_{z \sim p(z)}[D(G(z))] - \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] + \lambda \, \mathbb{E}_{\hat{x}}\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$$

where $\hat{x}$ is a random interpolation between a real and fake sample. This penalty forces the gradient norm to be close to 1, enforcing the Lipschitz constraint.

**Derivation walkthrough:** Show why the Lipschitz constraint is needed (Kantorovich-Rubinstein duality), why weight clipping (original WGAN) is problematic, and why the gradient penalty is preferred.

**Key papers:**
- Arjovsky, Chintala & Bottou (2017). "Wasserstein GAN." ICML.
- Gulrajani et al. (2017). "Improved Training of Wasserstein GANs." NeurIPS.

### Block 4: Architectural Evolution (30 min)

**DCGAN (Radford et al., 2016) — the first stable convolutional GAN:**

Architectural guidelines that became standard:
1. Replace pooling layers with strided convolutions (D) and transposed convolutions (G).
2. Use batch normalization in both G and D (except D's input layer and G's output layer).
3. Remove fully connected hidden layers.
4. Use ReLU activation in G (except output, which uses Tanh).
5. Use LeakyReLU activation in D.

**Conditional GAN (Mirza & Osindero, 2014):** Feed a class label to both G and D. G produces "a 7" instead of "some digit." D evaluates "is this a real 7?"

**Progressive Growing (Karras et al., 2018):** Start training at low resolution (4x4), then progressively add layers for higher resolutions (8x8, 16x16, ..., 1024x1024). This stabilizes training dramatically.

**StyleGAN (Karras et al., 2019) — overview only:**
- Mapping network: transforms z into an intermediate latent w
- Adaptive instance normalization (AdaIN) injects style at each resolution
- Noise injection adds stochastic variation (hair strands, pores)
- Style mixing enables incredible control over generated images
- Result: photorealistic face generation that stunned the field

**Key papers:**
- Radford, Metz & Chintala (2016). "Unsupervised Representation Learning with DCGANs." ICLR.
- Karras et al. (2018). "Progressive Growing of GANs." ICLR.
- Karras, Laine & Aila (2019). "A Style-Based Generator Architecture." CVPR.

### Block 5: Coding Exercise and Debugging (40 min)

Live-code a basic GAN on MNIST:

```python
# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))
```

**Debugging exercise:** Train this GAN, observe mode collapse, then implement the Wasserstein loss with gradient penalty and compare.

### Key Papers for This Session

1. Goodfellow et al. (2014). "Generative Adversarial Nets." NeurIPS.
2. Radford, Metz & Chintala (2016). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." ICLR.
3. Arjovsky, Chintala & Bottou (2017). "Wasserstein GAN." ICML.
4. Gulrajani et al. (2017). "Improved Training of Wasserstein GANs." NeurIPS.
5. Karras, Laine & Aila (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks." CVPR.

---

## Session 3: Diffusion Models

**Duration:** 3 hours

### Learning Objectives

By the end of this session, the apprentice will be able to:

1. Describe the forward and reverse diffusion processes mathematically and intuitively.
2. Derive the simplified DDPM training objective (predict the noise).
3. Explain the connection between diffusion models and score matching.
4. Implement a basic diffusion model training loop and sampling loop.
5. Articulate why diffusion models overtook GANs as the dominant generative paradigm.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 30 min | Intuition: destruction is easy, creation is hard |
| 2 | 50 min | The forward process and its closed-form solution |
| 3 | 40 min | The reverse process and DDPM training objective |
| 4 | 30 min | Sampling, guidance, and latent diffusion |
| 5 | 30 min | Coding exercise |

### Block 1: Intuition (30 min)

**The core idea:** Imagine dropping ink into a glass of clear water. Over time, the ink diffuses until the water is uniformly murky. This "forward process" is trivial — just add noise. Now imagine watching that video in reverse: the murky water gradually un-mixes until the ink droplet reappears. If we could learn this "reverse process," we could start from pure noise and generate structured data.

**Why this is a breakthrough framing:**
- The forward process is simple and has a closed-form solution (just add Gaussian noise).
- The reverse process is where the neural network comes in — it learns to gradually remove noise.
- Unlike GANs, there is no adversarial training. The objective is a straightforward MSE loss.
- Unlike VAEs, we don't need to worry about a single bottleneck. The generation happens over many small steps.

**Historical context:** Sohl-Dickstein et al. (2015) proposed the idea, but it was Ho et al. (2020) who made it practical with DDPM. Song et al. (2021) connected it to score matching and made it rigorous. By 2022, diffusion models had surpassed GANs on image generation benchmarks.

### Block 2: The Forward Process (50 min)

**Definition:** Given a data sample $x_0$, the forward process produces increasingly noisy versions $x_1, x_2, \ldots, x_T$ by adding Gaussian noise at each step:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t I)$$

where $\beta_t$ is a small positive constant (the noise schedule), typically between 0.0001 and 0.02.

**What this means:** At each step, we slightly shrink the signal (multiply by $\sqrt{1 - \beta_t}$) and add a small amount of noise (variance $\beta_t$). After $T$ steps (typically $T = 1000$), $x_T$ is indistinguishable from pure Gaussian noise.

**The closed-form shortcut:** Define $\alpha_t = 1 - \beta_t$, and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) I)$$

This means we can jump directly from $x_0$ to any $x_t$ without computing all intermediate steps:

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Derivation walkthrough:** Show this closed form by recursively applying the one-step formula. Use the fact that adding two independent Gaussians gives a Gaussian whose variances add.

**Spend time on the noise schedule:** Linear schedule (original DDPM), cosine schedule (Nichol & Dhariwal, 2021 — reduces the sudden destruction at the end), and why the schedule matters for sample quality.



### Block 3: The Reverse Process and Training Objective (40 min)

**The reverse process:** We want to learn $p_\theta(x_{t-1} | x_t)$ — given a noisy image, produce a slightly less noisy image. We parameterize this as a Gaussian:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

The neural network predicts the mean $\mu_\theta$. The variance $\sigma_t^2$ is typically fixed to $\beta_t$ or $\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$.

**The noise prediction reformulation (the key insight):** Instead of predicting the mean directly, we reparameterize $\mu_\theta$ in terms of the predicted noise:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

where $\epsilon_\theta$ is the neural network that predicts the noise that was added.

**The simplified training objective:** The full variational bound simplifies to:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right]$$

where:
- $t$ is uniformly sampled from $\{1, \ldots, T\}$
- $\epsilon \sim \mathcal{N}(0, I)$
- $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$

**In plain English:** Sample a training image. Pick a random timestep. Add the appropriate amount of noise. Ask the network to predict what noise was added. Penalize it with MSE.

**Connection to score matching:** The score of a distribution is the gradient of log probability: $s(x) = \nabla_x \log p(x)$. Score matching trains a network to estimate this gradient. It turns out that predicting the noise $\epsilon$ is equivalent to predicting the score (up to a scaling factor):

$$\epsilon_\theta(x_t, t) \propto -s_\theta(x_t, t)$$

This connection, established by Song et al., unifies the diffusion and score-matching perspectives.

**Key papers:**
- Ho, Jain & Abbeel (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.
- Song et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR.

### Block 4: Sampling, Guidance, and Latent Diffusion (30 min)

**The sampling algorithm (DDPM):**

```
Algorithm: DDPM Sampling
1. Sample x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a. Predict noise: epsilon_hat = epsilon_theta(x_t, t)
   b. Compute mean: mu = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_hat)
   c. Sample z ~ N(0, I) if t > 1, else z = 0
   d. x_{t-1} = mu + sigma_t * z
3. Return x_0
```

This requires T forward passes of the network (typically 1000). Each step removes a little noise.

**DDIM (Song et al., 2020) — faster sampling:** A deterministic variant that can skip steps. Instead of 1000 steps, use 50 or even 20. This makes diffusion models practical.

**Classifier-free guidance (Ho & Salimans, 2022):**

The idea: train the same model both conditionally and unconditionally (randomly drop the conditioning signal during training). At inference time, extrapolate away from the unconditional prediction:

$$\epsilon_{\text{guided}} = \epsilon_{\text{unconditional}} + w \cdot (\epsilon_{\text{conditional}} - \epsilon_{\text{unconditional}})$$

where $w > 1$ amplifies the effect of the conditioning. This dramatically improves sample quality and is used in essentially every modern diffusion system (DALL-E 2, Stable Diffusion, Imagen).

**The U-Net backbone:** The denoising network $\epsilon_\theta$ is typically a U-Net:
- Encoder path with downsampling (captures context)
- Decoder path with upsampling (produces full-resolution output)
- Skip connections (preserves spatial detail)
- Timestep embedding via sinusoidal positional encoding (tells the network which noise level to expect)
- Attention layers at lower resolutions (captures global structure)

**Latent diffusion (Rombach et al., 2022 — Stable Diffusion):**
- Running diffusion in pixel space at high resolution is extremely expensive.
- Solution: first encode the image into a lower-dimensional latent space using a pretrained autoencoder (VAE), then run diffusion in that latent space.
- This reduces computational cost dramatically while preserving quality.
- The decoder maps the denoised latent back to pixel space.

### Block 5: Coding Exercise (30 min)

**Exercise:** Implement the DDPM training loop and sampling loop for MNIST.

Key components to code:
1. The noise schedule (compute alpha_bar_t for all t)
2. The forward process (add noise to a batch at random timesteps)
3. The training step (predict noise, compute MSE loss)
4. The sampling loop (iterative denoising from pure noise)

```python
# Core training step
def train_step(model, x_0, optimizer):
    t = torch.randint(0, T, (x_0.shape[0],))      # random timesteps
    epsilon = torch.randn_like(x_0)                  # random noise
    x_t = sqrt_alpha_bar[t] * x_0 + sqrt_one_minus_alpha_bar[t] * epsilon
    epsilon_pred = model(x_t, t)                     # predict the noise
    loss = F.mse_loss(epsilon_pred, epsilon)          # simple MSE
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

**Coding exercise:** Complete the sampling loop. Visualize the denoising trajectory.

### Key Papers for This Session

1. Sohl-Dickstein et al. (2015). "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." ICML.
2. Ho, Jain & Abbeel (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.
3. Song et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR.
4. Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models." ICML.
5. Ho & Salimans (2022). "Classifier-Free Diffusion Guidance."
6. Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR.

---

## Week 13 Summary

| Concept | Key Idea | Core Math |
|---------|----------|-----------|
| VAE | Encode to a distribution, regularize toward prior | $\text{ELBO} = \mathbb{E}[\log p(x\|z)] - D_{KL}(q(z\|x) \| p(z))$ |
| GAN | Adversarial game between generator and discriminator | $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ |
| Diffusion | Learn to reverse a gradual noising process | $\mathcal{L} = \mathbb{E}[\| \epsilon - \epsilon_\theta(x_t, t) \|^2]$ |

**The progression:** VAEs (2013) gave us a principled but blurry generative model. GANs (2014) gave us sharp images but unstable training. Diffusion models (2020) gave us both quality and stability, at the cost of slow sampling. Understanding all three is essential — each reveals different aspects of the generative modeling problem, and ideas from all three continue to influence current research.

**Next module:** We move from generating images to generating sequences, connecting these generative principles with the Transformer architectures from Module 7.
