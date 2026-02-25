"""
Diffusion Model - Model Definitions

Contains:
1. Noise schedule functions (linear and cosine) [pre-written]
2. Sinusoidal time embedding [pre-written]
3. Residual block with time conditioning
4. U-Net denoising network [stubbed]
5. Forward diffusion process [pre-written]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ============================================================
# NOISE SCHEDULES (pre-written)
# ============================================================

def linear_noise_schedule(
    T: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear noise schedule.

    Returns:
        betas:      (T,) - Noise added at each step
        alphas:     (T,) - Signal retained at each step (1 - beta)
        alpha_bars: (T,) - Cumulative signal retained from step 0 to step t
    """
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def cosine_noise_schedule(
    T: int, s: float = 0.008
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cosine noise schedule (Nichol & Dhariwal, 2021).

    More gradual noise addition than linear schedule.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bars = f / f[0]
    alpha_bars = torch.clamp(alpha_bars, min=1e-5, max=1.0 - 1e-5)
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    betas = torch.clamp(betas, min=1e-5, max=0.999)
    alphas = 1.0 - betas
    alpha_bars = alpha_bars[1:]
    return betas.float(), alphas.float(), alpha_bars.float()


def get_noise_schedule(
    schedule_type: str, T: int, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get noise schedule by type name."""
    if schedule_type == "linear":
        return linear_noise_schedule(T, **kwargs)
    elif schedule_type == "cosine":
        return cosine_noise_schedule(T)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# ============================================================
# FORWARD DIFFUSION (pre-written)
# ============================================================

def forward_diffusion(
    x_0: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add noise to clean images at specified timesteps.

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    Args:
        x_0: (B, C, H, W) - Clean images
        t:   (B,) - Timestep for each image
        alpha_bars: (T,) - Precomputed cumulative products

    Returns:
        x_t:     (B, C, H, W) - Noisy images at timestep t
        epsilon: (B, C, H, W) - The noise that was added
    """
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bars[t])[:, None, None, None]

    epsilon = torch.randn_like(x_0)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * epsilon

    return x_t, epsilon


# ============================================================
# TIME EMBEDDING (pre-written)
# ============================================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding (same idea as Transformer positional encoding).

    Args:
        dim: Output embedding dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed integer timesteps.

        Args:
            t: (B,) - Integer timesteps

        Returns:
            emb: (B, dim) - Sinusoidal embedding
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device, dtype=torch.float) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ============================================================
# RESIDUAL BLOCK (pre-written)
# ============================================================

class ResBlock(nn.Module):
    """Residual block with time conditioning.

    Two conv layers with GroupNorm and SiLU activation.
    Time embedding is projected and added after the first conv.

    Args:
        in_ch: Input channels
        out_ch: Output channels
        time_dim: Dimension of time embedding
        num_groups: Number of groups for GroupNorm
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_ch, H, W) - Input features
            t_emb: (B, time_dim) - Time embedding

        Returns:
            (B, out_ch, H, W)
        """
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


# ============================================================
# U-NET (stubbed)
# ============================================================

class UNet(nn.Module):
    """U-Net denoising network for DDPM.

    Architecture for 28x28 MNIST:
        Encoder:
            ResBlock(1 -> 64)   at 28x28
            Downsample          to 14x14
            ResBlock(64 -> 128) at 14x14
            Downsample          to 7x7

        Bottleneck:
            ResBlock(128 -> 256) at 7x7

        Decoder:
            Upsample             to 14x14
            ResBlock(256+128 -> 128) at 14x14 (skip connection)
            Upsample             to 28x28
            ResBlock(128+64 -> 64)   at 28x28 (skip connection)

        Output:
            GroupNorm -> SiLU -> Conv2d(64 -> 1)

    Args:
        in_channels: Input image channels (1 for MNIST)
        base_channels: Base feature map count (64)
        channel_mults: Multipliers for each resolution level
        time_emb_dim: Dimension of time embedding
        num_groups: Groups for GroupNorm
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = None,
        time_emb_dim: int = 128,
        num_groups: int = 8,
    ):
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 4]

        # YOUR CODE HERE
        # 1. Time embedding MLP: SinusoidalEmbedding -> Linear -> SiLU -> Linear
        # 2. Encoder: initial conv, then for each level: ResBlock + Downsample
        # 3. Bottleneck: ResBlock at lowest resolution
        # 4. Decoder: for each level (reversed): Upsample + ResBlock (with skip connection)
        # 5. Output: GroupNorm -> SiLU -> Conv2d to in_channels
        raise NotImplementedError("Implement UNet __init__")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise from noisy image and timestep.

        Args:
            x: (B, C, H, W) - Noisy image
            t: (B,) - Integer timesteps

        Returns:
            (B, C, H, W) - Predicted noise (same shape as input)
        """
        # YOUR CODE HERE
        # 1. Compute time embedding
        # 2. Encoder pass: save skip connections at each level
        # 3. Bottleneck
        # 4. Decoder pass: concatenate skip connections, upsample
        # 5. Output projection
        raise NotImplementedError("Implement UNet forward")
