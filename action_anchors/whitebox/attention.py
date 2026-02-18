"""Core attention extraction and analysis functions.

Merges functionality from thought-anchors:
- pytorch_models/analysis.py  (extract_attention_and_logits, analyze_text)
- attention_analysis/attn_funcs.py (sentence boundaries, averaging, vertical scores)

Adapted for action-anchors: no pkld caching, no short-name model config,
direct HuggingFace model path usage.
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer

from .model_loader import clear_gpu_memory, load_model, N_LAYERS, N_HEADS


# ---------------------------------------------------------------------------
# Forward pass — extract attention weights (and optionally logits)
# ---------------------------------------------------------------------------

def extract_attention_and_logits(
    model,
    tokenizer,
    text: str,
    return_logits: bool = False,
    attn_layers: Optional[List[int]] = None,
    token_range_to_mask: Optional[List[int]] = None,
    mask_layers: Optional[Dict[int, List[int]]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a forward pass to extract attention weights and (optionally) logits.

    Args:
        model: HuggingFace CausalLM loaded with ``attn_implementation="eager"``.
        tokenizer: Corresponding tokenizer.
        text: Full input text (prompt + response).
        return_logits: Whether to also return output logits.
        attn_layers: If given, only keep these layer indices.
        token_range_to_mask: Token range to mask via hooks (for suppression).
        mask_layers: Layer→heads mapping for attention masking.
        verbose: Print debug info.

    Returns:
        Dict with keys: text, tokens, token_texts, input_length,
        attention_weights (dict layer→tensor), and optionally logits.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    if verbose:
        print(f"Encoded to {input_ids.shape[1]} tokens")

    logits = None
    hooks_applied = False
    attention_weights: Dict[int, torch.Tensor] = {}

    try:
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*does not support `output_attentions=True`.*",
                )

                if token_range_to_mask and mask_layers:
                    from .hooks import (
                        apply_qwen_attn_mask_hooks,
                    )

                    apply_qwen_attn_mask_hooks(
                        model, token_range_to_mask, layer_2_heads_suppress=mask_layers
                    )
                    hooks_applied = True

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=(attn_layers is None or len(attn_layers) > 0),
                    return_dict=True,
                    use_cache=False,
                    output_hidden_states=False,
                )

            # Collect attention weights
            if token_range_to_mask is None:
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    for layer_idx, attn in enumerate(outputs.attentions):
                        if attn_layers is not None and layer_idx not in attn_layers:
                            continue
                        attention_weights[layer_idx] = attn.detach().cpu()

            if return_logits and hasattr(outputs, "logits"):
                logits = outputs.logits.detach().cpu().numpy()

    except Exception as e:
        print(f"WARNING: Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if hooks_applied:
            from .hooks import remove_qwen_attn_mask_hooks
            remove_qwen_attn_mask_hooks(model)

    clear_gpu_memory()

    all_tokens = input_ids[0].tolist()
    token_texts = tokenizer.convert_ids_to_tokens(all_tokens)

    result: Dict[str, Any] = {
        "text": text,
        "tokens": all_tokens,
        "token_texts": token_texts,
        "input_length": len(all_tokens),
        "attention_weights": attention_weights,
    }
    if logits is not None:
        result["logits"] = logits

    clear_gpu_memory()
    return result


def analyze_text(
    text: str,
    model=None,
    tokenizer=None,
    model_name: str = "Qwen/Qwen3-8B",
    return_logits: bool = False,
    attn_layers: Optional[List[int]] = None,
    verbose: bool = False,
    token_range_to_mask: Optional[List[int]] = None,
    layers_to_mask: Optional[Dict[int, List[int]]] = None,
    float32: bool = False,
    device_map: str = "auto",
) -> Dict[str, Any]:
    """High-level: load model (if needed), run forward pass, return results.

    If *model* and *tokenizer* are provided they are reused (recommended to
    avoid loading the model repeatedly).
    """
    owns_model = model is None
    if owns_model:
        model, tokenizer = load_model(model_name, float32=float32, device_map=device_map)

    if verbose:
        print(f"Analyzing text: {text[:100]}...")

    result = extract_attention_and_logits(
        model,
        tokenizer,
        text,
        return_logits=return_logits,
        attn_layers=attn_layers,
        verbose=verbose,
        token_range_to_mask=token_range_to_mask,
        mask_layers=layers_to_mask,
    )

    if owns_model:
        del model
        clear_gpu_memory()

    return result


# ---------------------------------------------------------------------------
# Sentence ↔ token boundary mapping
# ---------------------------------------------------------------------------

def get_sentence_token_boundaries(
    text: str,
    sentences: List[str],
    tokenizer: AutoTokenizer,
) -> List[Tuple[int, int]]:
    """Map sentence strings to (start_token, end_token) positions in *text*.

    Args:
        text: Full text that was tokenized.
        sentences: Ordered list of sentence strings (substrings of *text*).
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        List of ``(start, end)`` token-index tuples, one per sentence.
    """
    if not sentences:
        return []

    def _normalize(s: str) -> str:
        return re.sub(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]", " ", s)

    text_norm = _normalize(text)
    search_start = 0
    char_positions: List[Tuple[int, int]] = []

    for sentence in sentences:
        sent_norm = _normalize(sentence)
        pos = text_norm.find(sent_norm, search_start)
        if pos == -1:
            pos = text_norm.find(sent_norm.strip(), search_start)
            if pos == -1:
                raise ValueError(f"Sentence not found in text: {sentence!r}")
            end = pos + len(sent_norm.strip())
        else:
            end = pos + len(sent_norm)
        char_positions.append((pos, end))
        search_start = end

    token_boundaries: List[Tuple[int, int]] = []
    for char_start, char_end in char_positions:
        tok_start = len(tokenizer.encode(text[:char_start], add_special_tokens=False)) if char_start > 0 else 0
        tok_end = len(tokenizer.encode(text[:char_end], add_special_tokens=False))
        token_boundaries.append((tok_start, tok_end))

    return token_boundaries


# ---------------------------------------------------------------------------
# Sentence-averaged attention matrix
# ---------------------------------------------------------------------------

def compute_averaged_matrix(
    matrix: np.ndarray,
    sentence_boundaries: List[Tuple[int, int]],
) -> np.ndarray:
    """Average a raw token-level attention matrix into a sentence-level matrix.

    ``result[i, j]`` = mean attention from tokens in sentence *i*
    to tokens in sentence *j*.
    """
    n = len(sentence_boundaries)
    result = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        rs, re_ = sentence_boundaries[i]
        rs = min(rs, matrix.shape[0] - 1)
        re_ = min(re_, matrix.shape[0] - 1)
        if rs >= re_:
            continue
        for j in range(n):
            cs, ce = sentence_boundaries[j]
            cs = min(cs, matrix.shape[1] - 1)
            ce = min(ce, matrix.shape[1] - 1)
            if cs >= ce:
                continue
            region = matrix[rs:re_, cs:ce]
            if region.size > 0:
                result[i, j] = np.mean(region)

    return result


def get_avg_attention_matrix(
    result: Dict[str, Any],
    layer: int,
    head: int,
    sentence_boundaries: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Extract and optionally sentence-average the attention matrix.

    Args:
        result: Output of ``extract_attention_and_logits`` or ``analyze_text``.
        layer: Layer index.
        head: Head index.
        sentence_boundaries: If provided, average over sentence regions.

    Returns:
        Attention matrix (token-level or sentence-level).
    """
    if layer not in result["attention_weights"]:
        raise KeyError(f"Layer {layer} not in extracted attention weights")
    matrix = result["attention_weights"][layer][0, head].numpy().astype(np.float32)
    if sentence_boundaries is not None:
        matrix = compute_averaged_matrix(matrix, sentence_boundaries)
    return matrix


# ---------------------------------------------------------------------------
# Vertical attention scores
# ---------------------------------------------------------------------------

def get_vertical_scores(
    avg_mat: np.ndarray,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> np.ndarray:
    """Compute vertical attention scores from a sentence-averaged matrix.

    For each sentence position *i*, the vertical score is the mean attention
    that later sentences (beyond ``proximity_ignore``) pay to sentence *i*.

    Args:
        avg_mat: Sentence-averaged attention matrix (n × n, lower-triangular).
        proximity_ignore: Ignore this many nearby sentences.
        control_depth: If True, rank-normalize each row first.

    Returns:
        Array of length n with vertical scores.
    """
    avg_mat = avg_mat.copy()
    n = avg_mat.shape[0]

    # Zero out upper triangle
    avg_mat[np.triu_indices_from(avg_mat, k=1)] = np.nan
    # Zero out the proximity band
    avg_mat[np.triu_indices_from(avg_mat, k=-proximity_ignore + 1)] = np.nan

    if control_depth:
        per_row = np.sum(~np.isnan(avg_mat), axis=1)
        per_row[per_row == 0] = 1  # avoid division by zero
        avg_mat = stats.rankdata(avg_mat, axis=1, nan_policy="omit") / per_row[:, None]

    vert_scores = []
    for i in range(n):
        col = avg_mat[i + proximity_ignore:, i]
        if len(col) == 0:
            vert_scores.append(np.nan)
        else:
            vert_scores.append(np.nanmean(col))

    return np.array(vert_scores)


# ---------------------------------------------------------------------------
# Multi-head vertical scores & kurtosis (for receiver head identification)
# ---------------------------------------------------------------------------

def get_all_heads_vert_scores(
    result: Dict[str, Any],
    sentence_boundaries: List[Tuple[int, int]],
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> np.ndarray:
    """Compute vertical scores for every (layer, head) pair.

    Args:
        result: Output of ``extract_attention_and_logits`` (must have all layers).
        sentence_boundaries: Token boundaries for each sentence.
        proximity_ignore: Proximity ignore for vertical scores.
        control_depth: Rank-normalize rows before scoring.

    Returns:
        Array of shape ``(n_layers, n_heads, n_sentences)``.
    """
    n_layers = max(result["attention_weights"].keys()) + 1
    sample_attn = next(iter(result["attention_weights"].values()))
    n_heads = sample_attn.shape[1]
    n_sentences = len(sentence_boundaries)

    scores = np.full((n_layers, n_heads, n_sentences), np.nan, dtype=np.float32)

    for layer_idx, attn_tensor in result["attention_weights"].items():
        for head_idx in range(n_heads):
            matrix = attn_tensor[0, head_idx].numpy().astype(np.float32)
            avg_mat = compute_averaged_matrix(matrix, sentence_boundaries)
            vs = get_vertical_scores(avg_mat, proximity_ignore, control_depth)
            scores[layer_idx, head_idx, :len(vs)] = vs

    return scores


def compute_kurtosis(layer_head_vert_scores: np.ndarray) -> np.ndarray:
    """Compute kurtosis over the sentence axis for each (layer, head).

    Args:
        layer_head_vert_scores: Shape ``(n_layers, n_heads, n_sentences)``.

    Returns:
        Array of shape ``(n_layers, n_heads)`` with Fisher kurtosis values.
    """
    return stats.kurtosis(
        layer_head_vert_scores, axis=2, fisher=True, bias=True, nan_policy="omit"
    )


def get_top_k_receiver_heads(
    kurtosis_matrix: np.ndarray,
    top_k: int = 20,
) -> np.ndarray:
    """Identify the top-k attention heads by mean kurtosis.

    Args:
        kurtosis_matrix: Shape ``(n_layers, n_heads)`` or
            ``(n_examples, n_layers, n_heads)`` (will be averaged over axis 0).
        top_k: Number of heads to return.

    Returns:
        Array of shape ``(top_k, 2)`` with ``[layer, head]`` pairs,
        sorted by descending kurtosis.
    """
    if kurtosis_matrix.ndim == 3:
        kurtosis_matrix = np.nanmean(kurtosis_matrix, axis=0)

    flat = kurtosis_matrix.flatten()
    valid_mask = ~np.isnan(flat)
    valid_indices = np.where(valid_mask)[0]
    valid_values = flat[valid_indices]

    k = min(top_k, len(valid_values))
    top_in_valid = np.argpartition(valid_values, -k)[-k:]
    top_in_valid = top_in_valid[np.argsort(-valid_values[top_in_valid])]
    top_flat = valid_indices[top_in_valid]

    coords = np.array(np.unravel_index(top_flat, kurtosis_matrix.shape)).T
    return coords.astype(int)
