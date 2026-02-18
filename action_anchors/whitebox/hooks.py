"""Attention masking hooks for Qwen-style models.

Used for the suppression analysis: mask attention to specific token ranges
and observe how output logits change (KL divergence).

Vendored from thought-anchors pytorch_models/hooks.py.
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple
from types import MethodType

import torch
import torch.nn as nn

from .rope_utils import apply_rotary_pos_emb, repeat_kv


# Store original methods for restoration
_original_forward_methods: Dict[str, object] = {}


def apply_qwen_attn_mask_hooks(
    model,
    token_range,
    layer_2_heads_suppress=None,
) -> None:
    """Patch Qwen attention modules to mask out a token range.

    Args:
        model: HuggingFace model.
        token_range: ``[start, end]`` or list of ``[start, end]`` pairs.
        layer_2_heads_suppress: Optional dict ``{layer_idx: [head_indices]}``.
    """
    global _original_forward_methods
    _original_forward_methods = {}

    if token_range is None:
        return

    assert isinstance(token_range, list)
    if isinstance(token_range[0], int):
        token_range = [token_range]

    # Find rotary embedding module
    rotary_emb_module = None
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_emb_module = model.model.rotary_emb

    target_modules = []
    for name, module in model.named_modules():
        if name.startswith("model.layers") and name.endswith("self_attn"):
            try:
                layer_idx = int(name.split(".")[2])
                if layer_2_heads_suppress is None or layer_idx in layer_2_heads_suppress:
                    if all(
                        hasattr(module, a)
                        for a in ("config", "q_proj", "k_proj", "v_proj", "o_proj")
                    ):
                        target_modules.append((name, module, layer_idx))
            except (IndexError, ValueError):
                pass

    if not target_modules:
        warnings.warn("No Qwen attention modules found for patching.")
        return

    def _create_masked_forward(original_fwd, layer_idx, rotary_ref, heads_mask=None):
        def masked_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position=None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            bsz, q_len, _ = hidden_states.size()
            config = self.config
            device = hidden_states.device

            num_heads = config.num_attention_heads
            head_dim = config.hidden_size // num_heads
            num_kv_heads = config.num_key_value_heads
            num_kv_groups = num_heads // num_kv_heads
            hidden_size = config.hidden_size

            query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            if position_ids is None:
                position_ids = torch.arange(q_len, dtype=torch.long, device=device).unsqueeze(0)
            else:
                position_ids = position_ids.to(device)

            if rotary_ref is not None and callable(rotary_ref):
                try:
                    cos, sin = rotary_ref(value_states.to(device), position_ids=position_ids)
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos.to(device), sin.to(device), position_ids
                    )
                except Exception:
                    pass

            kv_seq_len = q_len
            key_states = repeat_kv(key_states, num_kv_groups)
            value_states = repeat_kv(value_states, num_kv_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            # Apply custom mask for suppression
            for tr in token_range:
                assert isinstance(tr, list)
                eff_start = min(tr[0], kv_seq_len)
                eff_end = min(tr[1], kv_seq_len)
                if eff_start < eff_end:
                    mask_val = torch.finfo(attn_weights.dtype).min
                    if heads_mask is None:
                        attn_weights[..., eff_start:eff_end] = mask_val
                    else:
                        attn_weights[:, heads_mask, :, eff_start:eff_end] = mask_val

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                expected = (bsz, 1, q_len, kv_seq_len)
                if attention_mask.shape != expected:
                    if attention_mask.ndim == 2:
                        attention_mask = attention_mask[:, None, None, :]
                    elif attention_mask.shape[2] == 1 and q_len > 1:
                        attention_mask = attention_mask.expand(*expected)
                    else:
                        attention_mask = None

                if attention_mask is not None:
                    if attention_mask.dtype == torch.bool:
                        attention_mask = torch.where(
                            attention_mask, 0.0, torch.finfo(attn_weights.dtype).min
                        ).to(attn_weights.dtype)
                    else:
                        attention_mask = attention_mask.to(attn_weights.dtype)
                    attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, hidden_size)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights

        return masked_forward

    for name, attn_module, layer_idx in target_modules:
        _original_forward_methods[name] = attn_module.forward
        heads_mask = (
            layer_2_heads_suppress[layer_idx] if layer_2_heads_suppress is not None else None
        )
        attn_module.forward = MethodType(
            _create_masked_forward(attn_module.forward, layer_idx, rotary_emb_module, heads_mask),
            attn_module,
        )


def remove_qwen_attn_mask_hooks(model) -> None:
    """Restore original forward methods after suppression analysis."""
    global _original_forward_methods
    if not _original_forward_methods:
        return

    for name, module in model.named_modules():
        if name in _original_forward_methods:
            module.forward = _original_forward_methods[name]

    _original_forward_methods = {}
