"""
RLHF Utilities
===============

Fully implemented helper functions for RLHF training:
- Text generation and log-probability computation
- Synthetic preference functions (brevity, positive, helpfulness)
- Preference data generation
- KL divergence computation
- Response sampling and diversity metrics
"""

import os
import random
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter


# ---------------------------------------------------------------------------
# Text Generation
# ---------------------------------------------------------------------------

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text from a language model given a prompt.

    Uses nucleus (top-p) sampling with temperature control.

    Args:
        model: GPT-2 language model.
        tokenizer: Tokenizer for encoding/decoding.
        prompt: Input prompt string.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling threshold.
        device: Torch device.

    Returns:
        Generated text (response only, without the prompt).
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = outputs[0][prompt_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Log-Probability Computation
# ---------------------------------------------------------------------------

def compute_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-probabilities under a language model.

    This is the bridge between language modeling and RL: the log-probability
    of generated tokens under the policy is what REINFORCE and PPO optimize.

    Args:
        model: GPT-2 language model.
        input_ids: Token IDs of shape (batch_size, seq_len).
        attention_mask: Optional attention mask.

    Returns:
        log_probs: Per-token log-probabilities (batch_size, seq_len - 1).
                   Position i contains log P(token_{i+1} | tokens_{0..i}).
        total_log_prob: Sum of per-token log-probs (batch_size,).
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Shift: logits[i] predicts token[i+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Per-token log-probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    # Mask out padding if attention_mask is provided
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:]
        token_log_probs = token_log_probs * shift_mask

    total_log_prob = token_log_probs.sum(dim=-1)

    return token_log_probs, total_log_prob


# ---------------------------------------------------------------------------
# Synthetic Preference Functions
# ---------------------------------------------------------------------------

def synthetic_preference_brevity(response_1: str, response_2: str) -> int:
    """Prefer shorter responses (simulating preference for conciseness).

    Args:
        response_1: First response text.
        response_2: Second response text.

    Returns:
        1 if response_1 is preferred, 0 if response_2 is preferred.
    """
    len_1 = len(response_1.split())
    len_2 = len(response_2.split())
    if len_1 < len_2:
        return 1
    elif len_1 > len_2:
        return 0
    else:
        return random.choice([0, 1])


def synthetic_preference_positive(response_1: str, response_2: str) -> int:
    """Prefer responses with more positive sentiment words.

    Args:
        response_1: First response text.
        response_2: Second response text.

    Returns:
        1 if response_1 is preferred, 0 if response_2 is preferred.
    """
    positive_words = {
        "good", "great", "best", "better", "excellent", "wonderful",
        "amazing", "fantastic", "helpful", "positive", "success",
        "happy", "love", "enjoy", "beautiful", "perfect", "improve",
        "benefit", "important", "valuable", "effective", "progress",
        "achieve", "growth", "opportunity", "strength", "support",
    }

    def count_positive(text: str) -> int:
        words = text.lower().split()
        return sum(1 for w in words if w.strip(".,!?;:") in positive_words)

    score_1 = count_positive(response_1)
    score_2 = count_positive(response_2)

    if score_1 > score_2:
        return 1
    elif score_1 < score_2:
        return 0
    else:
        return random.choice([0, 1])


def synthetic_preference_helpfulness(response_1: str, response_2: str) -> int:
    """Prefer responses that contain helpful structural patterns.

    Looks for: numbered lists, bullet points, "Here are", "First,"
    and other indicators of structured, helpful responses.

    Args:
        response_1: First response text.
        response_2: Second response text.

    Returns:
        1 if response_1 is preferred, 0 if response_2 is preferred.
    """
    helpful_patterns = [
        r"\d+\.",           # numbered lists
        r"^-\s",            # bullet points
        r"here are",        # "here are..."
        r"first,",          # "first, ..."
        r"for example",     # examples
        r"step \d",         # steps
        r"important",       # importance markers
        r"recommend",       # recommendations
    ]

    def helpfulness_score(text: str) -> int:
        text_lower = text.lower()
        return sum(1 for p in helpful_patterns if re.search(p, text_lower))

    score_1 = helpfulness_score(response_1)
    score_2 = helpfulness_score(response_2)

    if score_1 > score_2:
        return 1
    elif score_1 < score_2:
        return 0
    else:
        return random.choice([0, 1])


# ---------------------------------------------------------------------------
# Preference Data Generation
# ---------------------------------------------------------------------------

def generate_preference_pairs(
    model,
    tokenizer,
    prompt: str,
    preference_fn,
    n_pairs: int = 20,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, str]]:
    """Generate preference pairs for a single prompt.

    Generates pairs of responses and uses the preference function to
    determine which is preferred.

    Args:
        model: Language model for generation.
        tokenizer: Tokenizer.
        prompt: Input prompt.
        preference_fn: Function(response_1, response_2) -> 1 or 0.
        n_pairs: Number of pairs to generate.
        max_new_tokens: Max tokens per response.
        temperature: Sampling temperature.
        device: Torch device.

    Returns:
        List of dicts with keys "prompt", "chosen", "rejected".
    """
    pairs = []
    for _ in range(n_pairs):
        response_1 = generate_text(model, tokenizer, prompt, max_new_tokens, temperature, device=device)
        response_2 = generate_text(model, tokenizer, prompt, max_new_tokens, temperature, device=device)

        if not response_1.strip() or not response_2.strip():
            continue

        preference = preference_fn(response_1, response_2)
        if preference == 1:
            pairs.append({"prompt": prompt, "chosen": response_1, "rejected": response_2})
        else:
            pairs.append({"prompt": prompt, "chosen": response_2, "rejected": response_1})

    return pairs


# ---------------------------------------------------------------------------
# KL Divergence Computation
# ---------------------------------------------------------------------------

def compute_kl_divergence(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-sequence KL divergence between policy and reference.

    KL(pi || pi_ref) = E_pi[log(pi/pi_ref)] = mean over tokens of (log_pi - log_ref)

    Args:
        policy_log_probs: Per-token log-probs under policy (batch, seq_len).
        ref_log_probs: Per-token log-probs under reference (batch, seq_len).
        mask: Optional token mask (batch, seq_len).

    Returns:
        KL divergence per sequence (batch_size,).
    """
    kl_per_token = policy_log_probs - ref_log_probs  # (batch, seq_len)

    if mask is not None:
        kl_per_token = kl_per_token * mask
        # Mean over non-padding tokens
        token_counts = mask.sum(dim=-1).clamp(min=1)
        kl = kl_per_token.sum(dim=-1) / token_counts
    else:
        kl = kl_per_token.mean(dim=-1)

    return kl


# ---------------------------------------------------------------------------
# Perplexity and Diversity Metrics
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device = torch.device("cpu"),
    max_length: int = 512,
) -> float:
    """Compute average perplexity of texts under a language model.

    Lower perplexity means the text is more "expected" by the model.
    High perplexity after RLHF may indicate reward hacking.

    Args:
        model: Language model for perplexity computation.
        tokenizer: Tokenizer.
        texts: List of text strings.
        device: Torch device.
        max_length: Maximum sequence length.

    Returns:
        Average perplexity across all texts.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding, labels=encoding["input_ids"])
            nll = outputs.loss.item() * encoding["input_ids"].shape[1]
            total_nll += nll
            total_tokens += encoding["input_ids"].shape[1]

    avg_nll = total_nll / max(total_tokens, 1)
    perplexity = np.exp(avg_nll)
    return perplexity


def compute_distinct_ngrams(texts: List[str], n: int = 2) -> float:
    """Compute distinct n-gram ratio (diversity metric).

    Ratio of unique n-grams to total n-grams across all texts.
    Higher values indicate more diverse outputs.

    Args:
        texts: List of text strings.
        n: N-gram size.

    Returns:
        Distinct n-gram ratio (0 to 1).
    """
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    return unique_ngrams / total_ngrams


# ---------------------------------------------------------------------------
# Response Sampling
# ---------------------------------------------------------------------------

def sample_responses(
    model,
    tokenizer,
    prompts: List[str],
    n_per_prompt: int = 3,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[str]]:
    """Generate multiple responses for each prompt.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        prompts: List of prompt strings.
        n_per_prompt: Number of responses per prompt.
        max_new_tokens: Maximum new tokens per response.
        temperature: Sampling temperature.
        device: Torch device.

    Returns:
        Dictionary mapping prompts to lists of generated responses.
    """
    results = {}
    for prompt in prompts:
        responses = []
        for _ in range(n_per_prompt):
            response = generate_text(model, tokenizer, prompt, max_new_tokens, temperature, device=device)
            responses.append(response)
        results[prompt] = responses
    return results


# ---------------------------------------------------------------------------
# Seed and Device
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
