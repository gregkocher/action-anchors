#!/usr/bin/env python3
"""Residual stream probing experiment for tool-use prediction.

Generates rollouts with vLLM, extracts residual stream activations at sentence
boundaries using HuggingFace, trains per-(layer, position) logistic regression
probes, and produces analysis plots.

Usage:
    uv run python run_probe_experiment.py --n-examples 100
    uv run python run_probe_experiment.py --skip-generate
    uv run python run_probe_experiment.py --skip-generate --skip-extract
    uv run python run_probe_experiment.py --n-examples 500
"""

import argparse
import gc
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from action_anchors.agent.prompt_builder import PromptBuilder
from action_anchors.agent.tool_parser import parse_generation
from action_anchors.resampling.sentence_splitter import split_cot_into_sentences
from action_anchors.tasks.gsm8k_calculator import (
    CALCULATOR_TOOL,
    GSM8KCalculatorTask,
    create_gsm8k_subset,
)
from action_anchors.whitebox.attention import get_sentence_token_boundaries
from action_anchors.whitebox.model_loader import N_LAYERS, clear_gpu_memory


# ---- Output paths -----------------------------------------------------------

OUTPUT_DIR = Path("outputs")
TRANSCRIPTS_FILE = OUTPUT_DIR / "probe_transcripts_gsm8k.json"
ACTIVATIONS_DIR = OUTPUT_DIR / "probe_activations"
RESULTS_FILE = OUTPUT_DIR / "probe_results_gsm8k.json"
WEIGHTS_FILE = OUTPUT_DIR / "probe_weights.npz"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ---- Utility ----------------------------------------------------------------

def position_sort_key(pos: str) -> tuple[int, int]:
    """Sort key so positions are ordered: think_start, sent_00..N, think_end."""
    if pos == "think_start":
        return (0, 0)
    elif pos.startswith("sent_"):
        return (1, int(pos.split("_")[1]))
    elif pos == "think_end":
        return (2, 0)
    return (3, 0)


def reconstruct_full_text(
    prompt_builder: PromptBuilder,
    system_prompt: str,
    tools: list[dict],
    question: str,
    raw_output: str,
) -> str:
    """Reconstruct full prompt + response text for the forward pass."""
    prompt = prompt_builder.build_initial_prompt(system_prompt, tools, question)
    return prompt + raw_output


# =============================================================================
# Phase 1: Generate rollouts with vLLM
# =============================================================================

def phase1_generate(n_examples: int) -> list[dict]:
    """Generate one rollout per question with vLLM and save transcripts."""
    from vllm import LLM, SamplingParams

    with open("action_anchors/config.yaml") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]

    # Create gsm8k_subset.json if missing
    data_path = Path("action_anchors/data/gsm8k_subset.json")
    if not data_path.exists():
        print("Creating GSM8K subset...")
        create_gsm8k_subset(
            n=max(n_examples, config["tasks"]["gsm8k"]["n_problems"]),
            output_path=str(data_path),
        )

    # Load task and examples
    task = GSM8KCalculatorTask(config)
    examples = task.get_examples()[:n_examples]
    print(f"Using {len(examples)} questions")

    # Build prompts
    builder = PromptBuilder(model_name)
    system_prompt = task.get_system_prompt()
    tools = task.get_tools()

    prompts = []
    for ex in examples:
        prompt = builder.build_initial_prompt(system_prompt, tools, ex.question)
        prompts.append(prompt)

    # Generate with vLLM (offline mode)
    print(f"Loading vLLM model: {model_name}...")
    llm = LLM(
        model=model_name,
        max_model_len=config["model"]["max_model_len"],
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
    )

    params = SamplingParams(
        n=1,
        max_tokens=config["collection"]["max_new_tokens"],
        temperature=config["collection"]["temperature"],
        top_p=config["collection"]["top_p"],
        stop=["<|im_end|>"],
    )

    print(f"Generating {len(prompts)} rollouts...")
    all_outputs = llm.generate(prompts, params)

    # Parse and build transcripts
    transcripts = []
    n_skipped_parse = 0
    n_skipped_empty = 0

    for ex, output in zip(examples, all_outputs):
        raw_output = output.outputs[0].text
        parsed = parse_generation(raw_output)

        if parsed.parse_error:
            n_skipped_parse += 1
            continue
        if not parsed.thinking.strip():
            n_skipped_empty += 1
            continue

        split = split_cot_into_sentences(parsed.thinking)
        has_tool_call = len(parsed.tool_calls) > 0

        transcript = {
            "example_id": ex.id,
            "question": ex.question,
            "raw_output": raw_output,
            "thinking": parsed.thinking,
            "n_sentences": len(split.sentences),
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "raw": tc.raw_text}
                for tc in parsed.tool_calls
            ],
            "final_answer": parsed.final_answer,
            "ground_truth": ex.ground_truth,
            "metadata": ex.metadata,
            "has_tool_call": has_tool_call,
        }
        transcripts.append(transcript)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRANSCRIPTS_FILE, "w") as f:
        json.dump(transcripts, f, indent=2)

    n_tool = sum(1 for t in transcripts if t["has_tool_call"])
    n_no_tool = len(transcripts) - n_tool
    print(f"\nPhase 1 complete:")
    print(f"  Saved {len(transcripts)} transcripts to {TRANSCRIPTS_FILE}")
    print(f"  {n_tool} with tool call, {n_no_tool} without")
    if n_skipped_parse:
        print(f"  Skipped {n_skipped_parse} (parse error)")
    if n_skipped_empty:
        print(f"  Skipped {n_skipped_empty} (empty thinking)")

    # Clean up vLLM
    del llm
    gc.collect()
    clear_gpu_memory()

    return transcripts


# =============================================================================
# Phase 2: Extract activations with HuggingFace
# =============================================================================

def phase2_extract(max_sentence_positions: int = 50) -> dict:
    """Extract residual stream activations at sentence boundaries and think tokens."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open("action_anchors/config.yaml") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]

    # Load transcripts
    with open(TRANSCRIPTS_FILE) as f:
        transcripts = json.load(f)
    print(f"Loaded {len(transcripts)} transcripts from {TRANSCRIPTS_FILE}")

    # Load HuggingFace model (fp16, no eager attention needed)
    print(f"Loading HuggingFace model: {model_name} (fp16)...")
    warnings.filterwarnings(
        "ignore",
        message="Sliding Window Attention is enabled but not implemented",
    )
    warnings.filterwarnings(
        "ignore",
        message="Setting `pad_token_id` to `eos_token_id`",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Build prompt builder for text reconstruction
    builder = PromptBuilder(model_name)
    system_prompt = config["tasks"]["gsm8k"]["system_prompt"].strip()
    tools = [CALCULATOR_TOOL]

    # Accumulate per-position activations
    # per_position[pos_name] = {"activations": [...], "labels": [...], "example_ids": [...]}
    per_position: dict[str, dict] = {}
    n_skipped = 0

    for idx, transcript in enumerate(tqdm(transcripts, desc="Extracting activations")):
        example_id = transcript["example_id"]
        question = transcript["question"]
        raw_output = transcript["raw_output"]
        thinking = transcript["thinking"]
        has_tool_call = transcript["has_tool_call"]

        # Reconstruct full text
        full_text = reconstruct_full_text(
            builder, system_prompt, tools, question, raw_output
        )

        # Split thinking into sentences
        split = split_cot_into_sentences(thinking)
        sentences = split.sentences

        if not sentences:
            print(f"  Skipping {example_id}: no sentences after split")
            n_skipped += 1
            continue

        # Get sentence token boundaries
        try:
            sentence_boundaries = get_sentence_token_boundaries(
                full_text, sentences, tokenizer
            )
        except ValueError as e:
            print(f"  Skipping {example_id}: boundary mapping failed: {e}")
            n_skipped += 1
            continue

        # Find <think> and </think> character positions
        think_start_char = full_text.find("<think>")
        think_end_char = full_text.find("</think>")

        if think_start_char == -1:
            print(f"  Skipping {example_id}: <think> not found")
            n_skipped += 1
            continue

        # Map character positions to token positions
        think_start_tok = len(
            tokenizer.encode(full_text[:think_start_char], add_special_tokens=False)
        )
        think_end_tok = None
        if think_end_char != -1:
            think_end_tok = len(
                tokenizer.encode(full_text[:think_end_char], add_special_tokens=False)
            )

        # Tokenize full text for the forward pass
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]

        if seq_len > config["model"]["max_model_len"]:
            print(f"  Skipping {example_id}: sequence too long ({seq_len} tokens)")
            n_skipped += 1
            del inputs
            clear_gpu_memory()
            continue

        # Forward pass with hidden states
        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        except Exception as e:
            print(f"  Skipping {example_id}: forward pass error: {e}")
            n_skipped += 1
            del inputs
            clear_gpu_memory()
            continue

        # hidden_states: tuple of 37 tensors (embed + 36 layers), each (1, seq_len, 4096)
        hidden_states = outputs.hidden_states

        # Collect target positions
        positions_to_extract: dict[str, int] = {}

        # Position: think_start (at the <think> token)
        if 0 <= think_start_tok < seq_len:
            positions_to_extract["think_start"] = think_start_tok

        # Positions: sent_00, sent_01, ..., sent_N (last token of each sentence)
        n_sent = min(len(sentence_boundaries), max_sentence_positions)
        for k in range(n_sent):
            tok_idx = sentence_boundaries[k][1] - 1  # last token of sentence k
            if 0 <= tok_idx < seq_len:
                positions_to_extract[f"sent_{k:02d}"] = tok_idx

        # Position: think_end (at the </think> token)
        if think_end_tok is not None and 0 <= think_end_tok < seq_len:
            positions_to_extract["think_end"] = think_end_tok

        # Extract activations at each position
        for pos_name, tok_idx in positions_to_extract.items():
            # Stack all 36 transformer layer outputs (skip embedding layer at index 0)
            activation = torch.stack(
                [hs[0, tok_idx, :] for hs in hidden_states[1:]],  # layers 1..36
                dim=0,
            )  # shape: (36, 4096)
            activation_np = activation.cpu().float().numpy()

            if pos_name not in per_position:
                per_position[pos_name] = {
                    "activations": [],
                    "labels": [],
                    "example_ids": [],
                }
            per_position[pos_name]["activations"].append(activation_np)
            per_position[pos_name]["labels"].append(int(has_tool_call))
            per_position[pos_name]["example_ids"].append(example_id)

        # Free memory
        del outputs, hidden_states, inputs
        clear_gpu_memory()

    # Save per-position npz files
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    for pos_name in sorted(per_position.keys(), key=position_sort_key):
        data = per_position[pos_name]
        activations = np.stack(data["activations"])  # (n, 36, 4096)
        labels = np.array(data["labels"], dtype=np.int32)
        example_ids = np.array(data["example_ids"])

        filename = f"pos_{pos_name}.npz"
        filepath = ACTIVATIONS_DIR / filename
        np.savez(filepath, activations=activations, labels=labels, example_ids=example_ids)

        manifest[pos_name] = {
            "file": filename,
            "n_examples": len(data["labels"]),
            "n_positive": int(labels.sum()),
            "n_negative": int((1 - labels).sum()),
        }
        print(
            f"  {pos_name}: {len(data['labels'])} examples "
            f"({int(labels.sum())} pos, {int((1 - labels).sum())} neg)"
        )

    with open(ACTIVATIONS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nPhase 2 complete:")
    print(f"  Saved activations for {len(manifest)} positions to {ACTIVATIONS_DIR}")
    if n_skipped:
        print(f"  Skipped {n_skipped} examples")

    # Clean up
    del model
    gc.collect()
    clear_gpu_memory()

    return manifest


# =============================================================================
# Phase 3: Train probes
# =============================================================================

def phase3_train_probes(test_size: float = 0.2) -> tuple[list[dict], dict]:
    """Train per-(layer, position) logistic regression probes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    with open(ACTIVATIONS_DIR / "manifest.json") as f:
        manifest = json.load(f)

    all_results: list[dict] = []
    all_weights: dict[tuple[str, int], np.ndarray] = {}  # (pos, layer) -> (4096,)

    positions_sorted = sorted(manifest.keys(), key=position_sort_key)

    for pos_name in positions_sorted:
        info = manifest[pos_name]
        filepath = ACTIVATIONS_DIR / info["file"]
        data = np.load(filepath)
        activations = data["activations"]  # (n, 36, 4096)
        labels = data["labels"]  # (n,)

        n = len(labels)
        n_pos = int(labels.sum())
        n_neg = n - n_pos

        if n < 20:
            print(f"  Skipping {pos_name}: too few examples ({n})")
            continue
        if n_pos == 0 or n_neg == 0:
            print(f"  Skipping {pos_name}: single class ({n_pos} pos, {n_neg} neg)")
            continue

        # Train/test split (stratified, same split for all 36 layers at this position)
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=42
        )

        best_test_auc = -1.0
        best_layer = -1

        for layer in range(N_LAYERS):
            X = activations[:, layer, :]  # (n, 4096)
            y = labels

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train logistic regression probe
            clf = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
            clf.fit(X_train_scaled, y_train)

            # Training AUC
            y_train_prob = clf.predict_proba(X_train_scaled)[:, 1]
            try:
                auc_train = roc_auc_score(y_train, y_train_prob)
            except ValueError:
                auc_train = float("nan")

            # Test AUC
            y_test_prob = clf.predict_proba(X_test_scaled)[:, 1]
            try:
                auc_test = roc_auc_score(y_test, y_test_prob)
            except ValueError:
                auc_test = float("nan")

            result = {
                "position": pos_name,
                "layer": layer,
                "auc_roc_train": auc_train,
                "auc_roc_test": auc_test,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_pos_train": int(y_train.sum()),
                "n_pos_test": int(y_test.sum()),
                "pos_frac": n_pos / n,
            }
            all_results.append(result)

            # Save probe weight vector (in standardized feature space)
            all_weights[(pos_name, layer)] = clf.coef_[0].copy()

            if not np.isnan(auc_test) and auc_test > best_test_auc:
                best_test_auc = auc_test
                best_layer = layer

        print(
            f"  {pos_name}: n={n} ({n_pos}+/{n_neg}-), "
            f"train={len(train_idx)}, test={len(test_idx)}, "
            f"best test AUC={best_test_auc:.3f} @ layer {best_layer}"
        )

    # Save results JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save probe weight vectors as npz
    if all_weights:
        weight_keys = sorted(all_weights.keys(), key=lambda k: (position_sort_key(k[0]), k[1]))
        weight_positions = np.array([k[0] for k in weight_keys])
        weight_layers = np.array([k[1] for k in weight_keys], dtype=np.int32)
        weight_vectors = np.stack([all_weights[k] for k in weight_keys])  # (n_probes, 4096)

        np.savez(
            WEIGHTS_FILE,
            positions=weight_positions,
            layers=weight_layers,
            weights=weight_vectors,
        )
        print(f"\nPhase 3 complete:")
        print(f"  Trained {len(all_results)} probes across {len(set(r['position'] for r in all_results))} positions")
        print(f"  Results saved to {RESULTS_FILE}")
        print(f"  Probe weights saved to {WEIGHTS_FILE} ({len(weight_keys)} vectors)")
    else:
        print("\nPhase 3 complete: no probes trained (insufficient data)")

    # Print summary table of best layer per position
    print("\n  Position         | Best Layer | Test AUC | N examples")
    print("  -----------------|------------|----------|----------")
    seen_positions = set()
    for r in all_results:
        pos = r["position"]
        if pos in seen_positions:
            continue
        # Find best layer for this position
        pos_results = [x for x in all_results if x["position"] == pos]
        best = max(pos_results, key=lambda x: x["auc_roc_test"] if not np.isnan(x["auc_roc_test"]) else -1)
        n_total = best["n_train"] + best["n_test"]
        print(f"  {pos:<17s} | {best['layer']:>10d} | {best['auc_roc_test']:>8.3f} | {n_total:>10d}")
        seen_positions.add(pos)

    return all_results, all_weights


# =============================================================================
# Phase 4: Plots and analysis
# =============================================================================

def phase4_plots():
    """Generate all plots and analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load results
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)

    if not all_results:
        print("No probe results to plot.")
        return

    # Load probe weights
    weights_data = np.load(WEIGHTS_FILE, allow_pickle=True)
    weight_positions = weights_data["positions"]
    weight_layers = weights_data["layers"]
    weight_vectors = weights_data["weights"]  # (n_probes, 4096)

    # Create output directories
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "cosine_sim" / "by_layer").mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "cosine_sim" / "by_position").mkdir(parents=True, exist_ok=True)

    # 4a: AUC-ROC line plots
    print("  Generating AUC-ROC plots...")
    _plot_auc_curves(all_results, plt)

    # 4b: Cosine similarity heatmaps
    print("  Generating cosine similarity heatmaps...")
    _plot_cosine_similarity(weight_positions, weight_layers, weight_vectors, plt, sns)

    # 4c: t-SNE and UMAP embedding plots
    print("  Generating embedding plots...")
    _plot_embeddings(weight_positions, weight_layers, weight_vectors, plt)

    # 4d: Probe weight norm heatmap
    print("  Generating probe weight norm heatmap...")
    _plot_weight_norms(weight_positions, weight_layers, weight_vectors, plt, sns)

    print(f"\nPhase 4 complete: all plots saved to {PLOTS_DIR}")


def _plot_auc_curves(results: list[dict], plt):
    """Plot AUC-ROC vs layer, one line per position (train and test)."""
    positions = sorted(set(r["position"] for r in results), key=position_sort_key)

    # Use a colormap that works for many lines
    n_pos = len(positions)
    if n_pos <= 10:
        cmap = plt.cm.tab10
    elif n_pos <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis

    for metric, title_prefix, filename in [
        ("auc_roc_train", "Training", "probe_auc_train.png"),
        ("auc_roc_test", "Test", "probe_auc_test.png"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, pos in enumerate(positions):
            pos_results = sorted(
                [r for r in results if r["position"] == pos],
                key=lambda r: r["layer"],
            )
            layers = [r["layer"] for r in pos_results]
            aucs = [r[metric] for r in pos_results]
            color = cmap(i / max(n_pos - 1, 1))
            ax.plot(layers, aucs, color=color, alpha=0.7, linewidth=1.5, label=pos)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("AUC-ROC", fontsize=12)
        ax.set_title(f"{title_prefix} AUC-ROC by Layer and Position", fontsize=14)
        ax.set_xlim(0, N_LAYERS - 1)
        ax.set_ylim(0.0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")

        if n_pos <= 20:
            ax.legend(fontsize=7, ncol=2, loc="upper left")
        else:
            # Colorbar for many positions
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(0, n_pos - 1)
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, label="Position index")
            key_ticks = [0, n_pos // 4, n_pos // 2, 3 * n_pos // 4, n_pos - 1]
            key_ticks = sorted(set(key_ticks))
            cbar.set_ticks(key_ticks)
            cbar.set_ticklabels([positions[t] for t in key_ticks])

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close()
        print(f"    Saved {filename}")


def _plot_cosine_similarity(positions_arr, layers_arr, weights, plt, sns):
    """Plot cosine similarity heatmaps by layer and by position."""
    # L2-normalize all weight vectors to unit norm
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = weights / norms

    # Build lookup: (position_str, layer_int) -> index
    lookup: dict[tuple[str, int], int] = {}
    for i in range(len(positions_arr)):
        lookup[(str(positions_arr[i]), int(layers_arr[i]))] = i

    unique_positions = sorted(set(str(p) for p in positions_arr), key=position_sort_key)
    unique_layers = sorted(set(int(l) for l in layers_arr))

    # ---- By layer: P x P heatmaps ----
    for layer in tqdm(unique_layers, desc="    Cosine sim by layer"):
        pos_indices = []
        pos_labels = []
        for pos in unique_positions:
            if (pos, layer) in lookup:
                pos_indices.append(lookup[(pos, layer)])
                pos_labels.append(pos)

        if len(pos_indices) < 2:
            continue

        vecs = normalized[pos_indices]
        sim_matrix = vecs @ vecs.T

        fig_size = max(8, len(pos_labels) * 0.3)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
        sns.heatmap(
            sim_matrix,
            xticklabels=pos_labels,
            yticklabels=pos_labels,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            ax=ax,
            square=True,
        )
        ax.set_title(f"Probe Cosine Similarity — Layer {layer}", fontsize=12)
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / "cosine_sim" / "by_layer" / f"layer_{layer:02d}.png",
            dpi=150,
        )
        plt.close()

    # ---- By position: 36 x 36 heatmaps ----
    for pos in tqdm(unique_positions, desc="    Cosine sim by position"):
        layer_indices = []
        layer_labels = []
        for layer in unique_layers:
            if (pos, layer) in lookup:
                layer_indices.append(lookup[(pos, layer)])
                layer_labels.append(str(layer))

        if len(layer_indices) < 2:
            continue

        vecs = normalized[layer_indices]
        sim_matrix = vecs @ vecs.T

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            sim_matrix,
            xticklabels=layer_labels,
            yticklabels=layer_labels,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            ax=ax,
            square=True,
        )
        ax.set_title(f"Probe Cosine Similarity — Position {pos}", fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        plt.xticks(rotation=0, fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / "cosine_sim" / "by_position" / f"pos_{pos}.png",
            dpi=150,
        )
        plt.close()

    print(f"    Saved {len(unique_layers)} by-layer and {len(unique_positions)} by-position heatmaps")


def _plot_embeddings(positions_arr, layers_arr, weights, plt):
    """Plot t-SNE and UMAP embeddings of probe weight vectors (unnormalized)."""
    from sklearn.manifold import TSNE

    n_probes = len(weights)
    if n_probes < 5:
        print("    Skipping embeddings: too few probes")
        return

    unique_positions = sorted(set(str(p) for p in positions_arr), key=position_sort_key)
    pos_to_idx = {p: i for i, p in enumerate(unique_positions)}

    layer_values = np.array([int(l) for l in layers_arr])
    pos_indices = np.array([pos_to_idx[str(p)] for p in positions_arr])

    # ---- t-SNE ----
    print("    Computing t-SNE...")
    perplexity = min(30, n_probes - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    coords_tsne = tsne.fit_transform(weights.astype(np.float64))

    _save_embedding_plot(
        coords_tsne,
        layer_values,
        pos_indices,
        unique_positions,
        "t-SNE",
        PLOTS_DIR / "probe_embedding_tsne.png",
        plt,
    )

    # ---- UMAP ----
    try:
        import umap

        print("    Computing UMAP...")
        n_neighbors = min(15, n_probes - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        coords_umap = reducer.fit_transform(weights.astype(np.float64))

        _save_embedding_plot(
            coords_umap,
            layer_values,
            pos_indices,
            unique_positions,
            "UMAP",
            PLOTS_DIR / "probe_embedding_umap.png",
            plt,
        )
    except ImportError:
        print("    Skipping UMAP: umap-learn not installed")


def _save_embedding_plot(
    coords, layer_values, pos_indices, unique_positions, method_name, output_path, plt
):
    """Save a 2-subplot embedding plot colored by layer and by position."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left subplot: colored by layer
    sc1 = ax1.scatter(
        coords[:, 0], coords[:, 1],
        c=layer_values, cmap="viridis", s=20, alpha=0.7,
    )
    plt.colorbar(sc1, ax=ax1, label="Layer")
    ax1.set_title(f"{method_name} — Colored by Layer", fontsize=12)
    ax1.set_xlabel(f"{method_name} dim 1")
    ax1.set_ylabel(f"{method_name} dim 2")

    # Right subplot: colored by position
    n_pos = len(unique_positions)
    cmap_pos = plt.cm.tab20 if n_pos <= 20 else plt.cm.nipy_spectral
    sc2 = ax2.scatter(
        coords[:, 0], coords[:, 1],
        c=pos_indices, cmap=cmap_pos, s=20, alpha=0.7,
    )
    if n_pos <= 20:
        handles = []
        for i, pos in enumerate(unique_positions):
            color = cmap_pos(i / max(n_pos - 1, 1))
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o", color="w",
                    markerfacecolor=color, markersize=6, label=pos,
                )
            )
        ax2.legend(handles=handles, fontsize=6, ncol=2, loc="upper left")
    else:
        cbar = plt.colorbar(sc2, ax=ax2, label="Position index")
        key_ticks = sorted(set([0, n_pos // 4, n_pos // 2, 3 * n_pos // 4, n_pos - 1]))
        cbar.set_ticks(key_ticks)
        cbar.set_ticklabels([unique_positions[t] for t in key_ticks])

    ax2.set_title(f"{method_name} — Colored by Position", fontsize=12)
    ax2.set_xlabel(f"{method_name} dim 1")
    ax2.set_ylabel(f"{method_name} dim 2")

    plt.suptitle(f"Probe Weight Vectors — {method_name} Embedding", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    Saved {output_path.name}")


def _plot_weight_norms(positions_arr, layers_arr, weights, plt, sns):
    """Plot probe weight L2 norm as a heatmap (layer x position)."""
    unique_positions = sorted(set(str(p) for p in positions_arr), key=position_sort_key)
    unique_layers = sorted(set(int(l) for l in layers_arr))

    pos_to_col = {p: i for i, p in enumerate(unique_positions)}
    layer_to_row = {l: i for i, l in enumerate(unique_layers)}

    norm_matrix = np.full((len(unique_layers), len(unique_positions)), np.nan)
    for i in range(len(positions_arr)):
        pos = str(positions_arr[i])
        lay = int(layers_arr[i])
        norm_val = np.linalg.norm(weights[i])
        norm_matrix[layer_to_row[lay], pos_to_col[pos]] = norm_val

    fig_width = max(10, len(unique_positions) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 10))
    sns.heatmap(
        norm_matrix,
        xticklabels=unique_positions,
        yticklabels=[str(l) for l in unique_layers],
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("Probe Weight L2 Norm (layer x position)", fontsize=14)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "probe_weight_norm.png", dpi=150)
    plt.close()
    print(f"    Saved probe_weight_norm.png")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Residual stream probing for tool-use prediction"
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=100,
        help="Number of questions to use (default: 100)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip Phase 1: reuse existing probe_transcripts_gsm8k.json",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip Phase 2: reuse existing probe_activations/*.npz",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test fraction (default: 0.2)",
    )
    parser.add_argument(
        "--max-sentence-positions",
        type=int,
        default=50,
        help="Max number of sentence positions to collect (default: 50)",
    )
    args = parser.parse_args()

    # Phase 1: Generate rollouts
    if not args.skip_generate:
        print("\n" + "=" * 60)
        print("Phase 1: Generating rollouts with vLLM")
        print("=" * 60)
        phase1_generate(args.n_examples)
    else:
        print("\nPhase 1 skipped (--skip-generate)")
        if not TRANSCRIPTS_FILE.exists():
            print(f"ERROR: {TRANSCRIPTS_FILE} not found. Run without --skip-generate first.")
            return

    # Phase 2: Extract activations
    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("Phase 2: Extracting activations with HuggingFace")
        print("=" * 60)
        phase2_extract(args.max_sentence_positions)
    else:
        print("\nPhase 2 skipped (--skip-extract)")
        if not (ACTIVATIONS_DIR / "manifest.json").exists():
            print(f"ERROR: {ACTIVATIONS_DIR / 'manifest.json'} not found. Run without --skip-extract first.")
            return

    # Phase 3: Train probes
    print("\n" + "=" * 60)
    print("Phase 3: Training probes")
    print("=" * 60)
    phase3_train_probes(args.test_size)

    # Phase 4: Plots and analysis
    print("\n" + "=" * 60)
    print("Phase 4: Generating plots and analysis")
    print("=" * 60)
    phase4_plots()

    print("\n" + "=" * 60)
    print("Done! All outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
