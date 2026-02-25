"""Models for hooks and debugging experiments.

Implements:
- DeepNetwork: configurable deep network for gradient flow experiments
- GradientReversalFunction / GradientReversalLayer: for domain adaptation
- DANN: Domain-Adversarial Neural Network architecture
- SimpleMNISTNet: basic network for pruning experiments
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ============================================================
# Part 1: Deep networks for gradient flow experiments
# ============================================================


class DeepNetwork(nn.Module):
    """Deep network for demonstrating gradient flow issues.

    Configurable depth, activation, and normalization to show
    vanishing/exploding gradients and how to fix them.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden layer width.
        num_layers: Number of hidden layers.
        num_classes: Number of output classes.
        activation: Activation function ('sigmoid', 'relu', 'tanh').
        use_batchnorm: Whether to add BatchNorm after each linear layer.
        use_residual: Whether to add residual (skip) connections.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        num_layers: int = 20,
        num_classes: int = 10,
        activation: str = "sigmoid",
        use_batchnorm: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual

        # YOUR CODE HERE
        # Build the network:
        # 1. Input projection: Linear(input_size, hidden_size)
        # 2. For each hidden layer:
        #    - Linear(hidden_size, hidden_size)
        #    - Optional: BatchNorm1d(hidden_size)
        #    - Activation (sigmoid/relu/tanh)
        # 3. Output layer: Linear(hidden_size, num_classes)
        # If use_residual, add skip connections around each hidden layer
        # Store layers in nn.ModuleList for proper hook registration
        raise NotImplementedError("Initialize DeepNetwork")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the deep network.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # YOUR CODE HERE
        # Pass through input projection, then each hidden layer (with optional
        # residual connections), then the output layer.
        raise NotImplementedError("Implement DeepNetwork.forward")


# ============================================================
# Part 3: Gradient Reversal Layer (for DANN)
# ============================================================


class GradientReversalFunction(Function):
    """Custom autograd Function for gradient reversal.

    Forward: identity (return input unchanged).
    Backward: negate the gradient and multiply by lambda_val.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """Identity in forward pass.

        Args:
            ctx: Context object.
            x: Input tensor.
            lambda_val: Scaling factor for gradient reversal.

        Returns:
            Input tensor unchanged.
        """
        # YOUR CODE HERE
        # 1. Save lambda_val for backward (use ctx.lambda_val = lambda_val)
        # 2. Return x unchanged (identity)
        raise NotImplementedError("Implement GradientReversalFunction.forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Negate and scale the gradient.

        Args:
            ctx: Context object.
            grad_output: Incoming gradient.

        Returns:
            Tuple of (negated gradient, None for lambda_val).
        """
        # YOUR CODE HERE
        # Return (-ctx.lambda_val * grad_output, None)
        raise NotImplementedError("Implement GradientReversalFunction.backward")


class GradientReversalLayer(nn.Module):
    """nn.Module wrapper for gradient reversal.

    Args:
        lambda_val: Scaling factor for gradient reversal.
    """

    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)


class DANN(nn.Module):
    """Domain-Adversarial Neural Network.

    Architecture:
    - Feature extractor: 2-3 linear layers with ReLU
    - Task classifier: 1 linear layer (predicts class label)
    - Domain classifier: gradient reversal + 1 linear layer (predicts domain)

    Args:
        input_size: Input feature dimension.
        feature_dim: Feature extractor output dimension.
        hidden_dim: Hidden layer size in feature extractor.
        num_classes: Number of task classes.
        lambda_val: Gradient reversal scaling factor.
    """

    def __init__(
        self,
        input_size: int = 784,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 5,
        lambda_val: float = 1.0,
    ):
        super().__init__()

        # YOUR CODE HERE
        # Build three components:
        # 1. feature_extractor: Sequential(Linear, ReLU, Linear, ReLU, Linear(hidden_dim, feature_dim), ReLU)
        # 2. task_classifier: Linear(feature_dim, num_classes)
        # 3. domain_classifier: Sequential(GradientReversalLayer(lambda_val), Linear(feature_dim, 2))
        raise NotImplementedError("Initialize DANN")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both task and domain predictions.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Tuple of (task_logits, domain_logits).
        """
        # YOUR CODE HERE
        # 1. features = self.feature_extractor(x)
        # 2. task_logits = self.task_classifier(features)
        # 3. domain_logits = self.domain_classifier(features)
        # 4. Return (task_logits, domain_logits)
        raise NotImplementedError("Implement DANN.forward")


# ============================================================
# Part 5: Simple network for pruning experiments
# ============================================================


class SimpleMNISTNet(nn.Module):
    """Simple MNIST classifier for pruning experiments.

    Args:
        input_size: Input dimension (784 for flattened MNIST).
        num_classes: Number of output classes.
    """

    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
