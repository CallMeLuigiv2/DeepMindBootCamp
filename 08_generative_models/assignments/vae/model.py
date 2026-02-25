"""
Variational Autoencoder - Model Definitions

Architecture:
    Encoder: MLP mapping input -> (mu, log_var)
    Reparameterize: z = mu + std * epsilon
    Decoder: MLP mapping z -> reconstructed input

Loss:
    ELBO = Reconstruction (BCE) + beta * KL divergence
    KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class VAE(nn.Module):
    """Variational Autoencoder for MNIST.

    Architecture:
        Encoder: 784 -> 512 -> 256 -> (mu, log_var) of dim latent_dim
        Decoder: latent_dim -> 256 -> 512 -> 784

    Args:
        input_dim: Input dimension (784 for flattened MNIST)
        hidden_dims: List of hidden layer dimensions for encoder/decoder
        latent_dim: Dimension of the latent space
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = None,
        latent_dim: int = 2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.latent_dim = latent_dim

        # Encoder layers
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Mu and log_var heads (branch from last hidden layer)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder layers (mirror of encoder)
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: (batch, input_dim) - Flattened input images

        Returns:
            mu: (batch, latent_dim) - Mean of q(z|x)
            log_var: (batch, latent_dim) - Log-variance of q(z|x)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z using the reparameterization trick.

        z = mu + std * epsilon, where epsilon ~ N(0, I)

        Args:
            mu: (batch, latent_dim)
            log_var: (batch, latent_dim)

        Returns:
            z: (batch, latent_dim) - Sampled latent vector
        """
        # YOUR CODE HERE
        # 1. Compute std = exp(0.5 * log_var)
        # 2. Sample epsilon ~ N(0, I) with same shape as std
        # 3. Return z = mu + std * epsilon
        raise NotImplementedError("Implement reparameterize")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed input.

        Args:
            z: (batch, latent_dim)

        Returns:
            x_recon: (batch, input_dim) - Reconstructed input
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> reparameterize -> decode.

        Args:
            x: (batch, input_dim)

        Returns:
            x_recon: (batch, input_dim) - Reconstruction
            mu: (batch, latent_dim)
            log_var: (batch, latent_dim)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE ELBO loss.

    Loss = Reconstruction_loss + beta * KL_divergence

    Args:
        x_recon: (batch, input_dim) - Reconstructed input
        x: (batch, input_dim) - Original input
        mu: (batch, latent_dim) - Encoder mean
        log_var: (batch, latent_dim) - Encoder log-variance
        beta: Weight on the KL term (1.0 = standard VAE)

    Returns:
        total_loss: Scalar loss value
        recon_loss: Reconstruction component (BCE, summed over batch)
        kl_loss: KL divergence component (summed over batch)
    """
    # YOUR CODE HERE
    # 1. Reconstruction loss: BCE(x_recon, x, reduction='sum')
    # 2. KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    # 3. Total loss: recon_loss + beta * kl_loss
    raise NotImplementedError("Implement vae_loss")


class ConditionalVAE(nn.Module):
    """Conditional VAE that conditions on digit class labels.

    Encoder input: concatenate image (784) with one-hot label (10) = 794
    Decoder input: concatenate z (latent_dim) with one-hot label (10)

    Args:
        input_dim: Image dimension (784)
        hidden_dims: Hidden layer dimensions
        latent_dim: Latent space dimension
        num_classes: Number of classes (10 for MNIST digits)
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = None,
        latent_dim: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # YOUR CODE HERE
        # Build encoder: input is (input_dim + num_classes) -> hidden -> (mu, log_var)
        # Build decoder: input is (latent_dim + num_classes) -> hidden -> output (input_dim)
        raise NotImplementedError("Implement ConditionalVAE __init__")

    def encode(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input concatenated with one-hot label.

        Args:
            x: (batch, input_dim)
            labels: (batch,) - Integer class labels

        Returns:
            mu, log_var: (batch, latent_dim) each
        """
        # YOUR CODE HERE
        # 1. One-hot encode labels: (batch, num_classes)
        # 2. Concatenate with x: (batch, input_dim + num_classes)
        # 3. Pass through encoder
        raise NotImplementedError("Implement ConditionalVAE encode")

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Same reparameterization trick as standard VAE."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement ConditionalVAE reparameterize")

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Decode z concatenated with one-hot label.

        Args:
            z: (batch, latent_dim)
            labels: (batch,) - Integer class labels

        Returns:
            x_recon: (batch, input_dim)
        """
        # YOUR CODE HERE
        # 1. One-hot encode labels
        # 2. Concatenate with z
        # 3. Pass through decoder
        raise NotImplementedError("Implement ConditionalVAE decode")

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass with conditioning."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement ConditionalVAE forward")
