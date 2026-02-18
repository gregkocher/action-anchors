"""Attention suppression analysis via KL divergence.

For each sentence in a transcript, mask attention to that sentence's tokens
across all layers/heads, then measure how the output logit distribution
changes (KL divergence). Produces a sentence × sentence suppression matrix.

Adapted from thought-anchors attention_analysis/attn_supp_funcs.py.
Simplified: no pkld caching, no sparse logit compression — we compute KL
directly from the full logit vectors.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .attention import extract_attention_and_logits
from .model_loader import N_LAYERS, N_HEADS, clear_gpu_memory


def _kl_divergence_logits(
    baseline_logits: np.ndarray,
    suppressed_logits: np.ndarray,
    temperature: float = 0.6,
) -> float:
    """Compute KL(P || Q) from raw logit vectors at a single token position.

    P = softmax(baseline_logits / T), Q = softmax(suppressed_logits / T).
    """
    b = torch.from_numpy(baseline_logits.astype(np.float32))
    s = torch.from_numpy(suppressed_logits.astype(np.float32))

    log_p = F.log_softmax(b / temperature, dim=0)
    log_q = F.log_softmax(s / temperature, dim=0)

    p = torch.exp(log_p)
    kl_terms = p * (log_p - log_q)
    kl_terms = torch.where(p == 0, torch.tensor(0.0), kl_terms)

    kl = torch.sum(kl_terms).item()
    return max(kl, 0.0)


def compute_suppression_matrix(
    model,
    tokenizer,
    text: str,
    sentence_boundaries: List[Tuple[int, int]],
    temperature: float = 0.6,
    take_log: bool = True,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """Compute the sentence-to-sentence KL suppression matrix.

    For each sentence *s*, mask attention to its tokens across all layers/heads,
    run a forward pass, and measure KL divergence at every token position.
    Then aggregate by sentence to get a ``(n_sentences, n_sentences)`` matrix.

    ``matrix[receiver, suppressed]`` = mean log-KL at tokens in *receiver*
    when attention to *suppressed* is masked.

    Args:
        model: HuggingFace model (eager attention).
        tokenizer: Corresponding tokenizer.
        text: Full prompt+response text.
        sentence_boundaries: ``[(start_tok, end_tok), ...]`` per sentence.
        temperature: Softmax temperature for KL computation.
        take_log: Apply log to KL values (as in thought-anchors).
        verbose: Print progress.

    Returns:
        Suppression matrix of shape ``(n_sentences, n_sentences)``, or None
        if an error occurs.
    """
    n_sent = len(sentence_boundaries)
    all_layers_heads = {i: list(range(N_HEADS)) for i in range(N_LAYERS)}

    # Baseline forward pass (no masking)
    if verbose:
        print("Running baseline forward pass...")
    try:
        baseline = extract_attention_and_logits(
            model, tokenizer, text, return_logits=True, attn_layers=[]
        )
    except Exception as e:
        print(f"Baseline forward pass failed: {e}")
        return None

    baseline_logits = baseline["logits"]  # (1, seq_len, vocab)
    n_tokens = baseline_logits.shape[1]

    suppression_matrix = np.full((n_sent, n_sent), np.nan, dtype=np.float32)

    for sent_idx in tqdm(range(n_sent), desc="Suppression analysis", disable=not verbose):
        token_range = list(sentence_boundaries[sent_idx])

        try:
            suppressed = extract_attention_and_logits(
                model,
                tokenizer,
                text,
                return_logits=True,
                attn_layers=[],
                token_range_to_mask=token_range,
                mask_layers=all_layers_heads,
            )
        except Exception as e:
            print(f"Suppression failed for sentence {sent_idx}: {e}")
            return None

        supp_logits = suppressed["logits"]

        # Compute per-token KL, then aggregate by sentence
        kl_per_token = np.zeros(n_tokens, dtype=np.float32)
        for t in range(n_tokens):
            kl = _kl_divergence_logits(
                baseline_logits[0, t], supp_logits[0, t], temperature
            )
            kl_per_token[t] = np.log(kl + 1e-9) if take_log else kl

        # Aggregate into sentence-level
        for recv_idx, (rs, re_) in enumerate(sentence_boundaries):
            rs = min(rs, n_tokens)
            re_ = min(re_, n_tokens)
            if rs < re_:
                suppression_matrix[recv_idx, sent_idx] = np.nanmean(kl_per_token[rs:re_])

        clear_gpu_memory()

    return suppression_matrix
