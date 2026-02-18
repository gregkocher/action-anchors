"""Model loading for white-box attention analysis.

Loads Qwen3-8B via HuggingFace Transformers with eager attention
(required for output_attentions=True). Adapted from thought-anchors
pytorch_models/model_loader.py.
"""

import warnings
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Qwen3-8B architecture constants
N_LAYERS = 36
N_HEADS = 32


def print_gpu_memory(prefix: str = "") -> None:
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} - GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print(f"{prefix} - Running on CPU")


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(
    model_name: str = "Qwen/Qwen3-8B",
    float32: bool = False,
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with eager attention for weight extraction.

    Args:
        model_name: HuggingFace model path.
        float32: Use float32 precision (more accurate attention but 2x memory).
        device_map: Device mapping strategy.

    Returns:
        (model, tokenizer) tuple.
    """
    warnings.filterwarnings(
        "ignore",
        message="Sliding Window Attention is enabled but not implemented",
    )
    warnings.filterwarnings(
        "ignore", message="Setting `pad_token_id` to `eos_token_id`"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": device_map,
        "attn_implementation": "eager",  # required for output_attentions
        "force_download": False,
        "dtype": torch.float32 if float32 else torch.float16,
    }

    print_gpu_memory("Before model loading")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    print_gpu_memory("After model loading")

    return model, tokenizer
