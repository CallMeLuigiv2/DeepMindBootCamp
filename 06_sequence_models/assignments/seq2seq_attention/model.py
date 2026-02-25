"""
Seq2Seq with Attention - Model Definitions

Architecture:
    Encoder: LSTM that reads input character-by-character
    Decoder: LSTM that generates output character-by-character
    BahdanauAttention: Additive attention mechanism
    Seq2Seq: Wrapper combining encoder, decoder, and optional attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Encoder(nn.Module):
    """LSTM encoder that reads an input sequence and produces hidden states.

    Architecture:
        Embedding(vocab_size, embedding_dim)
        LSTM(embedding_dim, hidden_size, num_layers)

    Args:
        vocab_size: Size of the input vocabulary
        embedding_dim: Dimension of character embeddings
        hidden_size: Number of LSTM hidden units
        num_layers: Number of LSTM layers
        dropout: Dropout probability (applied between LSTM layers if num_layers > 1)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode the source sequence.

        Args:
            src: (batch, src_len) - Source token IDs

        Returns:
            outputs: (batch, src_len, hidden_size) - All encoder hidden states
            (hidden, cell): Final LSTM states, each (num_layers, batch, hidden_size)
        """
        # YOUR CODE HERE
        # 1. Embed the source tokens
        # 2. Pass through the LSTM
        # 3. Return all hidden states and the final (hidden, cell) state
        raise NotImplementedError("Implement the Encoder forward pass")


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism.

    Computes attention scores using a learned alignment model:
        e_ij = v^T * tanh(W_h * h_j + W_s * s_i)
        alpha_ij = softmax(e_ij)
        context_i = sum_j(alpha_ij * h_j)

    Args:
        encoder_hidden_size: Dimension of encoder hidden states
        decoder_hidden_size: Dimension of decoder hidden state
        attention_size: Dimension of the attention alignment layer
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_size: int,
    ):
        super().__init__()

        # YOUR CODE HERE
        # Define three linear layers:
        # W_h: projects encoder hidden states (encoder_hidden_size -> attention_size)
        # W_s: projects decoder hidden state (decoder_hidden_size -> attention_size)
        # v: produces scalar alignment score (attention_size -> 1)
        raise NotImplementedError("Implement BahdanauAttention __init__")

    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention context vector and weights.

        Args:
            decoder_state: (batch, decoder_hidden_size) - Current decoder hidden state
            encoder_outputs: (batch, src_len, encoder_hidden_size) - All encoder outputs
            mask: (batch, src_len) - Optional mask (1 for valid positions, 0 for padding)

        Returns:
            context: (batch, encoder_hidden_size) - Weighted sum of encoder outputs
            weights: (batch, src_len) - Attention weights (sum to 1 over src_len)
        """
        # YOUR CODE HERE
        # 1. Project encoder outputs with W_h: (batch, src_len, attention_size)
        # 2. Project decoder state with W_s: (batch, attention_size) -> unsqueeze for broadcast
        # 3. Compute alignment scores: v^T * tanh(W_h*h + W_s*s) -> (batch, src_len)
        # 4. Apply mask if provided (set masked positions to -inf before softmax)
        # 5. Compute attention weights with softmax over src_len
        # 6. Compute context as weighted sum of encoder outputs
        raise NotImplementedError("Implement BahdanauAttention forward pass")


class Decoder(nn.Module):
    """LSTM decoder with optional Bahdanau attention.

    Without attention:
        LSTM input: embedded token (embedding_dim)
        Output projection: hidden_state -> vocab_size

    With attention:
        LSTM input: [embedded token; context vector] (embedding_dim + encoder_hidden_size)
        Output projection: [hidden_state; context vector] -> vocab_size

    Args:
        vocab_size: Size of the output vocabulary
        embedding_dim: Dimension of character embeddings
        hidden_size: Number of LSTM hidden units
        encoder_hidden_size: Dimension of encoder hidden states (for attention)
        attention: Optional BahdanauAttention module
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        encoder_hidden_size: int = 128,
        attention: Optional[BahdanauAttention] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.attention = attention

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM input size depends on whether we use attention
        lstm_input_size = embedding_dim + encoder_hidden_size if attention else embedding_dim
        self.lstm = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection size depends on whether we use attention
        proj_input_size = hidden_size + encoder_hidden_size if attention else hidden_size
        self.output_proj = nn.Linear(proj_input_size, vocab_size)

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """Single decoder step.

        Args:
            input_token: (batch, 1) - Current input token IDs
            hidden: (h, c) each (num_layers, batch, hidden_size) - Previous LSTM state
            encoder_outputs: (batch, src_len, encoder_hidden_size) - Encoder outputs (for attention)
            mask: (batch, src_len) - Source padding mask

        Returns:
            logits: (batch, vocab_size) - Output logits for next token prediction
            hidden: Updated LSTM state
            attn_weights: (batch, src_len) or None - Attention weights if using attention
        """
        # YOUR CODE HERE
        # 1. Embed the input token: (batch, 1, embedding_dim)
        # 2. If using attention:
        #    a. Compute context and weights using current decoder hidden state (hidden[0][-1])
        #    b. Concatenate embedded token with context: (batch, 1, embedding_dim + enc_hidden)
        # 3. Pass through LSTM to get output and new hidden state
        # 4. Compute logits:
        #    - With attention: project [lstm_output; context]
        #    - Without attention: project lstm_output
        # 5. Return logits, new hidden state, and attention weights (or None)
        raise NotImplementedError("Implement Decoder forward_step")


class Seq2Seq(nn.Module):
    """Sequence-to-sequence model combining encoder and decoder.

    Supports teacher forcing during training and greedy/beam search during inference.

    Args:
        encoder: Encoder module
        decoder: Decoder module
        device: torch device
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with teacher forcing.

        Args:
            src: (batch, src_len) - Source token IDs
            trg: (batch, trg_len) - Target token IDs (used for teacher forcing)
            teacher_forcing_ratio: Probability of using ground truth as next input
            src_mask: (batch, src_len) - Source padding mask

        Returns:
            outputs: (batch, trg_len, vocab_size) - Predicted logits at each step
            attentions: (batch, trg_len, src_len) or None - Attention weights
        """
        # YOUR CODE HERE
        # 1. Encode the source sequence
        # 2. Initialize decoder hidden state with encoder final state
        # 3. For each target timestep:
        #    a. Run one decoder step
        #    b. Store the output logits
        #    c. Decide next input: ground truth (teacher forcing) or predicted token
        #    d. Store attention weights if available
        # 4. Return stacked outputs and attention weights
        raise NotImplementedError("Implement Seq2Seq forward pass")

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int = 12,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Greedy decoding for inference.

        Args:
            src: (batch, src_len) - Source token IDs
            max_len: Maximum output sequence length
            bos_token_id: Beginning-of-sequence token ID
            eos_token_id: End-of-sequence token ID
            src_mask: (batch, src_len) - Source padding mask

        Returns:
            predictions: (batch, max_len) - Predicted token IDs
            attentions: (batch, max_len, src_len) or None - Attention weights
        """
        # YOUR CODE HERE
        # 1. Encode the source sequence
        # 2. Start with BOS token as first decoder input
        # 3. Greedily select highest-probability token at each step
        # 4. Stop when EOS is generated or max_len is reached
        # 5. Return predicted token IDs and attention weights
        raise NotImplementedError("Implement greedy_decode")
