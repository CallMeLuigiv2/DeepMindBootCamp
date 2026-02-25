"""
Transformer from Scratch - Model Definitions

Contains all components needed to build both an encoder-decoder Transformer
and a GPT-style decoder-only Transformer from basic PyTorch primitives.

Architecture reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (pre-written).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability applied after adding PE
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: (batch, seq_len, d_model) - Token embeddings

        Returns:
            (batch, seq_len, d_model) - Embeddings + positional encoding
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Each head operates on a d_k = d_model / num_heads dimensional subspace.

    Args:
        d_model: Total model dimension
        num_heads: Number of attention heads
        dropout: Dropout on attention weights
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # YOUR CODE HERE
        # Define four linear projections:
        # W_q: (d_model -> d_model) for queries
        # W_k: (d_model -> d_model) for keys
        # W_v: (d_model -> d_model) for values
        # W_o: (d_model -> d_model) for output projection
        # Also define dropout for attention weights
        raise NotImplementedError("Implement MultiHeadAttention __init__")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention.

        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  Optional mask, broadcastable to (batch, num_heads, seq_len_q, seq_len_k)

        Returns:
            output:  (batch, seq_len_q, d_model)
            weights: (batch, num_heads, seq_len_q, seq_len_k) - attention weights
        """
        # YOUR CODE HERE
        # 1. Project Q, K, V through linear layers
        # 2. Reshape to (batch, num_heads, seq_len, d_k)
        # 3. Compute scaled dot-product attention scores: QK^T / sqrt(d_k)
        # 4. Apply mask if provided (set masked positions to -inf)
        # 5. Apply softmax to get attention weights
        # 6. Apply dropout to attention weights
        # 7. Compute weighted sum of values
        # 8. Reshape back to (batch, seq_len_q, d_model)
        # 9. Apply output projection
        raise NotImplementedError("Implement MultiHeadAttention forward")


class FeedForward(nn.Module):
    """Position-wise feed-forward network (pre-written).

    FFN(x) = Linear_2(Dropout(ReLU(Linear_1(x))))

    Args:
        d_model: Input/output dimension
        d_ff: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block.

    Sub-layers:
        1. Multi-Head Self-Attention + Residual + LayerNorm
        2. Feed-Forward Network + Residual + LayerNorm

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: FFN hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # YOUR CODE HERE
        # Define:
        # - self.self_attn: MultiHeadAttention
        # - self.ffn: FeedForward
        # - self.norm1, self.norm2: nn.LayerNorm(d_model)
        # - self.dropout1, self.dropout2: nn.Dropout(dropout)
        raise NotImplementedError("Implement TransformerEncoderBlock __init__")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder block.

        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            (batch, seq_len, d_model)
        """
        # YOUR CODE HERE
        # Post-norm architecture:
        # 1. attn_out = self_attn(x, x, x, mask)
        # 2. x = norm1(x + dropout1(attn_out))
        # 3. ffn_out = ffn(x)
        # 4. x = norm2(x + dropout2(ffn_out))
        raise NotImplementedError("Implement TransformerEncoderBlock forward")


class TransformerDecoderBlock(nn.Module):
    """Single Transformer decoder block.

    Sub-layers:
        1. Masked Multi-Head Self-Attention + Residual + LayerNorm
        2. Multi-Head Cross-Attention + Residual + LayerNorm
        3. Feed-Forward Network + Residual + LayerNorm

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: FFN hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # YOUR CODE HERE
        # Define:
        # - self.self_attn: MultiHeadAttention (for masked self-attention)
        # - self.cross_attn: MultiHeadAttention (for cross-attention with encoder output)
        # - self.ffn: FeedForward
        # - self.norm1, self.norm2, self.norm3: nn.LayerNorm(d_model)
        # - self.dropout1, self.dropout2, self.dropout3: nn.Dropout(dropout)
        raise NotImplementedError("Implement TransformerDecoderBlock __init__")

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x: (batch, tgt_len, d_model) - Decoder input
            encoder_output: (batch, src_len, d_model) - Encoder output
            src_mask: Mask for cross-attention (e.g., source padding mask)
            tgt_mask: Causal mask for self-attention

        Returns:
            (batch, tgt_len, d_model)
        """
        # YOUR CODE HERE
        # 1. Masked self-attention: attn(x, x, x, tgt_mask) + residual + norm
        # 2. Cross-attention: attn(x, encoder_output, encoder_output, src_mask) + residual + norm
        # 3. FFN + residual + norm
        raise NotImplementedError("Implement TransformerDecoderBlock forward")


class Transformer(nn.Module):
    """Complete encoder-decoder Transformer.

    Components:
        - Source and target embeddings (scaled by sqrt(d_model))
        - Positional encoding
        - N encoder blocks
        - N decoder blocks
        - Output linear projection to target vocabulary

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of encoder/decoder blocks
        d_ff: FFN hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        # YOUR CODE HERE
        # Define:
        # - src_embedding, tgt_embedding: nn.Embedding
        # - positional_encoding: PositionalEncoding
        # - encoder_layers: nn.ModuleList of TransformerEncoderBlock
        # - decoder_layers: nn.ModuleList of TransformerDecoderBlock
        # - output_proj: nn.Linear(d_model, tgt_vocab_size)
        # - d_model (store for embedding scaling)
        raise NotImplementedError("Implement Transformer __init__")

    def encode(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode source sequence.

        Args:
            src: (batch, src_len) - Source token IDs
            src_mask: Optional source mask

        Returns:
            (batch, src_len, d_model) - Encoder output
        """
        # YOUR CODE HERE
        # 1. Embed and scale by sqrt(d_model)
        # 2. Add positional encoding
        # 3. Pass through all encoder layers
        raise NotImplementedError("Implement Transformer encode")

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target sequence given encoder output.

        Args:
            tgt: (batch, tgt_len) - Target token IDs
            encoder_output: (batch, src_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target causal mask

        Returns:
            (batch, tgt_len, d_model) - Decoder output
        """
        # YOUR CODE HERE
        # 1. Embed and scale by sqrt(d_model)
        # 2. Add positional encoding
        # 3. Pass through all decoder layers
        raise NotImplementedError("Implement Transformer decode")

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass.

        Args:
            src: (batch, src_len) - Source token IDs
            tgt: (batch, tgt_len) - Target token IDs
            src_mask: Source padding mask
            tgt_mask: Target causal mask

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        # YOUR CODE HERE
        # 1. Encode source
        # 2. Decode target with encoder output
        # 3. Project to vocabulary
        raise NotImplementedError("Implement Transformer forward")

    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
        """Generate upper-triangular causal mask.

        Returns:
            mask: (1, 1, seq_len, seq_len) - True for allowed positions, False for masked
        """
        # YOUR CODE HERE
        # Create a lower-triangular boolean mask where position i can attend to positions <= i
        raise NotImplementedError("Implement generate_causal_mask")


class GPT(nn.Module):
    """GPT-style decoder-only Transformer for autoregressive language modeling.

    Uses only masked self-attention (no cross-attention) with causal masking.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of decoder blocks
        d_ff: FFN hidden dimension
        max_len: Maximum sequence length (context window)
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # YOUR CODE HERE
        # Define:
        # - token_embedding: nn.Embedding(vocab_size, d_model)
        # - positional_encoding: PositionalEncoding(d_model, max_len, dropout)
        # - layers: nn.ModuleList of decoder blocks (self-attention + FFN only, no cross-attn)
        # - output_norm: nn.LayerNorm(d_model)
        # - output_proj: nn.Linear(d_model, vocab_size)
        # - d_model (for embedding scaling)
        raise NotImplementedError("Implement GPT __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking.

        Args:
            x: (batch, seq_len) - Token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # YOUR CODE HERE
        # 1. Embed tokens and scale by sqrt(d_model)
        # 2. Add positional encoding
        # 3. Generate causal mask
        # 4. Pass through all decoder layers (self-attention only)
        # 5. Apply final layer norm
        # 6. Project to vocabulary
        raise NotImplementedError("Implement GPT forward")

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Args:
            start_tokens: (batch, prompt_len) - Initial token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1 = greedy, >1 = random)

        Returns:
            generated: (batch, prompt_len + max_new_tokens)
        """
        # YOUR CODE HERE
        # 1. Start with the prompt tokens
        # 2. For each new token:
        #    a. Forward pass on current sequence (or truncated to max_len)
        #    b. Get logits for the last position
        #    c. Divide by temperature
        #    d. Sample from the resulting distribution
        #    e. Append to the sequence
        raise NotImplementedError("Implement GPT generate")
