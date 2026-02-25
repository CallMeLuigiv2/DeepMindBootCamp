"""Custom autograd Functions and nn.Module wrappers.

Implements:
- ParameterizedSwish: x * sigmoid(beta * x) with learnable beta
- HardThresholdSTE: step function with straight-through gradient
- ClampedSTE: step function with clamped straight-through gradient
- AsymmetricMSEFunction: MSE loss with different penalties for over/under estimation
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ============================================================
# Part 1: Parameterized Swish / SiLU with learnable beta
# ============================================================

# Gradient derivation for f(x) = x * sigmoid(beta * x):
#
# Let s = sigmoid(beta * x), so f(x) = x * s
#
# df/dx = s + x * s' * beta
#       = s + x * beta * s * (1 - s)
#       = s * (1 + beta * x * (1 - s))
#
# df/dbeta = x * s' * x
#          = x^2 * s * (1 - s)


class ParameterizedSwish(Function):
    """Custom autograd Function for f(x) = x * sigmoid(beta * x).

    Supports learnable beta parameter with analytically derived gradients.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Compute x * sigmoid(beta * x).

        Args:
            ctx: Context object for saving tensors needed in backward.
            x: Input tensor of any shape.
            beta: Scalar parameter (learnable).

        Returns:
            Output tensor of same shape as x.
        """
        # YOUR CODE HERE
        # 1. Compute sigmoid(beta * x)
        # 2. Compute output = x * sigmoid(beta * x)
        # 3. Save tensors needed for backward using ctx.save_for_backward
        # 4. Return output
        raise NotImplementedError("Implement ParameterizedSwish.forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients for x and beta.

        Args:
            ctx: Context with saved tensors from forward.
            grad_output: Gradient of loss with respect to forward output.

        Returns:
            Tuple of (grad_x, grad_beta).
        """
        # YOUR CODE HERE
        # 1. Retrieve saved tensors from ctx
        # 2. Compute df/dx = sigmoid(beta*x) * (1 + beta * x * (1 - sigmoid(beta*x)))
        # 3. Compute df/dbeta = x^2 * sigmoid(beta*x) * (1 - sigmoid(beta*x))
        # 4. Multiply by grad_output and return (grad_x, grad_beta)
        #    Note: grad_beta should be summed over all elements since beta is scalar
        raise NotImplementedError("Implement ParameterizedSwish.backward")


class LearnableSwish(nn.Module):
    """nn.Module wrapper for ParameterizedSwish with a learnable beta parameter.

    Args:
        initial_beta: Initial value for the beta parameter.
    """

    def __init__(self, initial_beta: float = 1.0):
        super().__init__()
        # YOUR CODE HERE
        # Create a learnable parameter `beta` initialized to initial_beta
        raise NotImplementedError("Initialize LearnableSwish")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameterized swish activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            Activated tensor of same shape.
        """
        # YOUR CODE HERE
        # Call ParameterizedSwish.apply(x, self.beta)
        raise NotImplementedError("Implement LearnableSwish.forward")


# ============================================================
# Part 2: Straight-Through Estimators
# ============================================================


class HardThresholdSTE(Function):
    """Step function with straight-through estimator.

    Forward: output = 1 where input >= 0, else 0 (step function).
    Backward: gradient passes through unchanged (identity).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Apply hard threshold (step function).

        Args:
            ctx: Context object.
            x: Input tensor.

        Returns:
            Binary tensor (0s and 1s).
        """
        # YOUR CODE HERE
        # 1. Compute step function: output = (x >= 0).float()
        # 2. Save any tensors needed for backward
        # 3. Return output
        raise NotImplementedError("Implement HardThresholdSTE.forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass gradient through unchanged (straight-through estimator).

        Args:
            ctx: Context object.
            grad_output: Incoming gradient.

        Returns:
            Gradient passed through unchanged.
        """
        # YOUR CODE HERE
        # Return grad_output unchanged
        raise NotImplementedError("Implement HardThresholdSTE.backward")


class ClampedSTE(Function):
    """Step function with clamped straight-through estimator.

    Forward: output = 1 where input >= 0, else 0 (step function).
    Backward: gradient passes through only for inputs in [-1, 1], zero outside.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Apply hard threshold (step function).

        Args:
            ctx: Context object.
            x: Input tensor.

        Returns:
            Binary tensor (0s and 1s).
        """
        # YOUR CODE HERE
        # 1. Compute step function: output = (x >= 0).float()
        # 2. Save x for backward (needed to compute the clamp mask)
        # 3. Return output
        raise NotImplementedError("Implement ClampedSTE.forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass gradient through only for inputs in [-1, 1].

        Args:
            ctx: Context object.
            grad_output: Incoming gradient.

        Returns:
            Clamped gradient (zero outside [-1, 1] input range).
        """
        # YOUR CODE HERE
        # 1. Retrieve saved x
        # 2. Create mask: (x >= -1) & (x <= 1)
        # 3. Return grad_output * mask
        raise NotImplementedError("Implement ClampedSTE.backward")


class BinaryActivationNetwork(nn.Module):
    """Small network with binary (STE) activations for MNIST classification.

    Architecture: input(784) -> Linear(256) -> STE -> Linear(128) -> STE -> Linear(10)

    Args:
        input_size: Input feature dimension (784 for flattened MNIST).
        hidden_sizes: List of hidden layer sizes.
        num_classes: Number of output classes.
        ste_variant: Which STE to use ('hard_threshold' or 'clamped').
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] = None,
        num_classes: int = 10,
        ste_variant: str = "hard_threshold",
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.ste_variant = ste_variant

        # YOUR CODE HERE
        # Build the network layers:
        # - Linear layers from input_size -> hidden_sizes[0] -> ... -> num_classes
        # - Store as nn.ModuleList or individual attributes
        # Hint: you need len(hidden_sizes) + 1 linear layers total
        raise NotImplementedError("Initialize BinaryActivationNetwork")

    def _apply_ste(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selected STE variant."""
        if self.ste_variant == "hard_threshold":
            return HardThresholdSTE.apply(x)
        elif self.ste_variant == "clamped":
            return ClampedSTE.apply(x)
        else:
            raise ValueError(f"Unknown STE variant: {self.ste_variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with binary activations.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # YOUR CODE HERE
        # Apply linear layers with STE activations between them
        # The final layer should NOT have an activation (raw logits)
        raise NotImplementedError("Implement BinaryActivationNetwork.forward")


# ============================================================
# Part 3: Asymmetric MSE Loss
# ============================================================


class AsymmetricMSEFunction(Function):
    """Custom asymmetric MSE loss function.

    L = mean(alpha * max(0, y_pred - y_true)^2 + beta * max(0, y_true - y_pred)^2)

    Penalizes overestimation with weight alpha and underestimation with weight beta.
    """

    @staticmethod
    def forward(
        ctx,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        """Compute asymmetric MSE loss.

        Args:
            ctx: Context object.
            y_pred: Predicted values.
            y_true: True values.
            alpha: Weight for overestimation penalty.
            beta: Weight for underestimation penalty.

        Returns:
            Scalar loss value.
        """
        # YOUR CODE HERE
        # 1. Compute diff = y_pred - y_true
        # 2. Overestimation: where diff > 0, loss contribution = alpha * diff^2
        # 3. Underestimation: where diff < 0, loss contribution = beta * diff^2
        # 4. Take the mean
        # 5. Save tensors needed for backward
        # 6. Return scalar loss
        raise NotImplementedError("Implement AsymmetricMSEFunction.forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Compute gradient with respect to y_pred.

        Args:
            ctx: Context object.
            grad_output: Gradient from upstream.

        Returns:
            Tuple of (grad_y_pred, None, None, None) - only y_pred needs gradient.
        """
        # YOUR CODE HERE
        # 1. Retrieve saved tensors
        # 2. Compute gradient:
        #    Where overestimating (diff > 0): dL/dy_pred = 2 * alpha * diff / N
        #    Where underestimating (diff < 0): dL/dy_pred = 2 * beta * diff / N
        # 3. Multiply by grad_output
        # 4. Return (grad_y_pred, None, None, None)
        raise NotImplementedError("Implement AsymmetricMSEFunction.backward")


class AsymmetricMSELoss(nn.Module):
    """nn.Module wrapper for AsymmetricMSEFunction.

    Args:
        alpha: Weight for overestimation penalty.
        beta: Weight for underestimation penalty.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric MSE loss.

        Args:
            y_pred: Predicted values.
            y_true: True values (detached from computation graph).

        Returns:
            Scalar loss.
        """
        return AsymmetricMSEFunction.apply(y_pred, y_true.detach(), self.alpha, self.beta)


class SimpleRegressor(nn.Module):
    """Simple regression model for testing asymmetric loss.

    Args:
        input_dim: Number of input features.
        hidden_dim: Hidden layer size.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
