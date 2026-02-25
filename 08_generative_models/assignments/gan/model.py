"""
GAN - Model Definitions

Contains Generator and Discriminator architectures for:
1. Basic MLP GAN (MNIST)
2. DCGAN (convolutional, CIFAR-10 / CelebA)
3. WGAN-GP (Wasserstein critic with no sigmoid, no BatchNorm)
"""

import torch
import torch.nn as nn
from typing import Optional


class BasicGenerator(nn.Module):
    """MLP Generator for MNIST (28x28 grayscale).

    Architecture:
        Linear(latent_dim, 256) -> LeakyReLU(0.2)
        Linear(256, 512) -> LeakyReLU(0.2)
        Linear(512, 1024) -> LeakyReLU(0.2)
        Linear(1024, 784) -> Tanh

    Output is in [-1, 1] to match normalized MNIST.

    Args:
        latent_dim: Dimension of the input noise vector z
    """

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        # YOUR CODE HERE
        # Build the MLP generator
        raise NotImplementedError("Implement BasicGenerator __init__")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate fake images from noise.

        Args:
            z: (batch, latent_dim) - Random noise

        Returns:
            (batch, 1, 28, 28) - Generated images in [-1, 1]
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement BasicGenerator forward")


class BasicDiscriminator(nn.Module):
    """MLP Discriminator for MNIST.

    Architecture:
        Linear(784, 512) -> LeakyReLU(0.2) -> Dropout(0.3)
        Linear(512, 256) -> LeakyReLU(0.2) -> Dropout(0.3)
        Linear(256, 1) -> Sigmoid

    Args:
        None (fixed architecture for MNIST)
    """

    def __init__(self):
        super().__init__()
        # YOUR CODE HERE
        # Build the MLP discriminator
        raise NotImplementedError("Implement BasicDiscriminator __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify input as real or fake.

        Args:
            x: (batch, 1, 28, 28) - Input images

        Returns:
            (batch, 1) - Probability that input is real
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement BasicDiscriminator forward")


class DCGANGenerator(nn.Module):
    """DCGAN Generator for 64x64 images.

    Uses transposed convolutions to upsample from latent vector to image.

    Args:
        latent_dim: Dimension of the input noise vector
        ngf: Base number of generator feature maps
        nc: Number of output channels (3 for RGB, 1 for grayscale)
    """

    def __init__(self, latent_dim: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        # YOUR CODE HERE
        # Build the DCGAN generator following the architecture in the assignment:
        # (latent_dim, 1, 1) -> (ngf*8, 4, 4) -> (ngf*4, 8, 8) ->
        # (ngf*2, 16, 16) -> (ngf, 32, 32) -> (nc, 64, 64)
        # Use ConvTranspose2d, BatchNorm2d, ReLU, and Tanh
        raise NotImplementedError("Implement DCGANGenerator __init__")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise.

        Args:
            z: (batch, latent_dim, 1, 1) - Noise vector reshaped for conv

        Returns:
            (batch, nc, 64, 64) - Generated images in [-1, 1]
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement DCGANGenerator forward")


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator for 64x64 images.

    Uses strided convolutions to downsample from image to scalar output.

    Args:
        nc: Number of input channels
        ndf: Base number of discriminator feature maps
        use_sigmoid: Whether to apply sigmoid (False for WGAN-GP critic)
    """

    def __init__(self, nc: int = 3, ndf: int = 64, use_sigmoid: bool = True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        # YOUR CODE HERE
        # Build the DCGAN discriminator following the architecture in the assignment:
        # (nc, 64, 64) -> (ndf, 32, 32) -> (ndf*2, 16, 16) ->
        # (ndf*4, 8, 8) -> (ndf*8, 4, 4) -> (1, 1, 1)
        # Use Conv2d, BatchNorm2d (skip in first layer), LeakyReLU(0.2)
        # For WGAN-GP: no BatchNorm, no Sigmoid
        raise NotImplementedError("Implement DCGANDiscriminator __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify input as real or fake.

        Args:
            x: (batch, nc, 64, 64)

        Returns:
            (batch, 1) - Score (probability if sigmoid, raw value if WGAN)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement DCGANDiscriminator forward")


def gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute the gradient penalty for WGAN-GP.

    Penalizes the critic for having gradient norm != 1 on interpolated samples.

    Args:
        critic: The critic (discriminator) network
        real_images: (batch, C, H, W) - Real images
        fake_images: (batch, C, H, W) - Generated images
        device: torch device

    Returns:
        penalty: Scalar gradient penalty loss
    """
    # YOUR CODE HERE
    # 1. Sample random alpha for interpolation
    # 2. Create interpolated images
    # 3. Compute critic output on interpolated images
    # 4. Compute gradients w.r.t. interpolated images
    # 5. Compute gradient norm and penalty: mean((||grad|| - 1)^2)
    raise NotImplementedError("Implement gradient_penalty")
