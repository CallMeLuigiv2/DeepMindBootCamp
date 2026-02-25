"""
Fine-Tune Pretrained Transformer - Model Definitions

Contains:
1. HuggingFace BERT wrapper for classification (with stub for custom head)
2. Small from-scratch Transformer classifier for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers import BertModel, BertForSequenceClassification


class BertClassifier(nn.Module):
    """BERT model with a custom classification head.

    Wraps HuggingFace's BertModel and adds a classification head on top
    of the [CLS] token representation.

    Args:
        model_name: Pretrained model name (e.g., 'bert-base-uncased')
        num_labels: Number of output classes
        dropout: Dropout probability for the classification head
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # YOUR CODE HERE
        # Define a classification head:
        # - Dropout layer
        # - Linear projection from hidden_size to num_labels
        # Consider whether you need additional layers (e.g., an intermediate dense layer)
        raise NotImplementedError("Implement BertClassifier __init__")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) - Tokenized input IDs
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
            token_type_ids: (batch, seq_len) - Segment IDs (optional)
            output_attentions: Whether to return attention weights

        Returns:
            dict with keys:
                'logits': (batch, num_labels) - Classification logits
                'attentions': Optional tuple of attention weights per layer
        """
        # YOUR CODE HERE
        # 1. Pass inputs through BERT (with output_attentions flag)
        # 2. Extract the [CLS] token representation (first token)
        # 3. Pass through your classification head
        # 4. Return logits (and optionally attentions)
        raise NotImplementedError("Implement BertClassifier forward")

    def get_parameter_groups(self, bert_lr: float, classifier_lr: float):
        """Create parameter groups with discriminative learning rates.

        Returns a list of dicts suitable for torch.optim.AdamW.
        Lower LR for pretrained BERT layers, higher LR for the new classification head.
        """
        # YOUR CODE HERE
        # Return [
        #   {'params': self.bert.parameters(), 'lr': bert_lr},
        #   {'params': self.classifier.parameters(), 'lr': classifier_lr},
        # ]
        raise NotImplementedError("Implement get_parameter_groups")


class SmallTransformerClassifier(nn.Module):
    """Small Transformer classifier trained from scratch for comparison.

    Uses a simple encoder-only Transformer with mean pooling and a classification head.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of Transformer encoder layers
        d_ff: FFN hidden dimension
        num_labels: Number of output classes
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        num_labels: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # YOUR CODE HERE
        # Define:
        # - Token embedding
        # - Positional encoding (or learned position embeddings)
        # - Stack of Transformer encoder layers (use nn.TransformerEncoderLayer for simplicity
        #   here since you already built it from scratch in Assignment 2)
        # - Classification head (Linear -> ReLU -> Dropout -> Linear)
        raise NotImplementedError("Implement SmallTransformerClassifier __init__")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) - Token IDs
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding

        Returns:
            dict with 'logits': (batch, num_labels)
        """
        # YOUR CODE HERE
        # 1. Embed tokens and add positional encoding
        # 2. Pass through Transformer encoder layers
        # 3. Mean-pool over non-padded positions
        # 4. Pass through classification head
        raise NotImplementedError("Implement SmallTransformerClassifier forward")
