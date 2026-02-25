"""Transformer encoder for text classification.

Architecture specification:
- Embedding dimension: 256, Heads: 8, Layers: 6, FFN: 1024
- Vocabulary: 30K, Max sequence: 256
- Classification via mean pooling or [CLS] token

Supports: gradient checkpointing, Flash Attention, torch.compile
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TransformerClassifier(nn.Module):
    """Transformer encoder for text classification.

    Args:
        vocab_size: Vocabulary size.
        embedding_dim: Embedding and model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer layers.
        ffn_hidden_dim: FFN hidden dimension.
        max_seq_length: Maximum sequence length.
        num_classes: Number of output classes.
        dropout: Dropout rate.
        pooling: Pooling strategy ('mean' or 'cls').
        use_flash_attention: Use F.scaled_dot_product_attention.
        use_checkpointing: Use gradient checkpointing.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_hidden_dim: int = 1024,
        max_seq_length: int = 256,
        num_classes: int = 4,
        dropout: float = 0.1,
        pooling: str = "mean",
        use_flash_attention: bool = False,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        self.use_checkpointing = use_checkpointing

        # YOUR CODE HERE
        # Build the Transformer encoder:
        # 1. Token embedding: nn.Embedding(vocab_size, embedding_dim)
        # 2. Positional embedding: nn.Embedding(max_seq_length, embedding_dim) (learned)
        # 3. Dropout
        # 4. Transformer layers (nn.TransformerEncoderLayer or custom)
        #    - If use_flash_attention, use F.scaled_dot_product_attention internally
        # 5. Layer norm
        # 6. Classification head: Linear(embedding_dim, num_classes)
        raise NotImplementedError("Initialize TransformerClassifier")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token indices (B, seq_len).
            attention_mask: Mask for padding (B, seq_len), 1 for real tokens.

        Returns:
            Logits (B, num_classes).
        """
        # YOUR CODE HERE
        # 1. Embed tokens + positions
        # 2. Apply dropout
        # 3. Pass through Transformer layers (with optional checkpointing)
        # 4. Pool: mean over sequence or take [CLS] token
        # 5. Classify
        raise NotImplementedError("Implement TransformerClassifier.forward")

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def memory_budget(self, batch_size: int = 16) -> dict[str, float]:
        """Estimate memory budget in MB.

        Returns:
            Dictionary with parameter, gradient, optimizer, and activation memory estimates.
        """
        # YOUR CODE HERE
        # Compute memory estimates:
        # - Parameters: num_params * 4 bytes (FP32)
        # - Gradients: same as parameters
        # - Optimizer (AdamW): 2x parameter memory (momentum + variance)
        # - Activations: estimate based on batch_size, seq_len, hidden_dim, num_layers
        raise NotImplementedError("Implement memory_budget")
