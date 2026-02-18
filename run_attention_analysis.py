#!/usr/bin/env python3
"""White-box attention analysis for action-anchors transcripts.

Loads saved transcripts, extracts attention weights via HuggingFace
(eager attention, one example at a time), and generates 7 visualizations
adapted from thought-anchors.

Usage:
    uv run python run_attention_analysis.py --task gsm8k --n-examples 5
    uv run python run_attention_analysis.py --task gsm8k --n-examples 3 --skip-suppression
    uv run python run_attention_analysis.py --task gsm8k --layers 0.5 --heads 0
    uv run python run_attention_analysis.py --task gsm8k --layers 0.0,0.5,0.8 --heads 0,1,2,3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from action_anchors.agent.prompt_builder import PromptBuilder
from action_anchors.resampling.sentence_splitter import split_cot_into_sentences
from action_anchors.tasks.gsm8k_calculator import CALCULATOR_TOOL
from action_anchors.tasks.factual_recall_search import SEARCH_TOOL
from action_anchors.whitebox.attention import (
    analyze_text,
    compute_averaged_matrix,
    compute_kurtosis,
    get_all_heads_vert_scores,
    get_avg_attention_matrix,
    get_sentence_token_boundaries,
    get_top_k_receiver_heads,
    get_vertical_scores,
)
from action_anchors.whitebox.model_loader import (
    N_HEADS,
    N_LAYERS,
    clear_gpu_memory,
    load_model,
)
from action_anchors.whitebox.plotting import (
    plot_attention_grid,
    plot_kurtosis_stats,
    plot_single_attention_heatmap,
    plot_split_half_reliability,
    plot_suppression_heatmap,
    plot_taxonomy_comparison,
    plot_vertical_scores_layer,
    plot_vertical_scores_top_k,
)
from action_anchors.whitebox.suppression import compute_suppression_matrix


# ---- Task-specific config ------------------------------------------------

TASK_CONFIG = {
    "gsm8k": {
        "transcript_file": "action_anchors/outputs/transcripts_gsm8k.json",
        "tools": [CALCULATOR_TOOL],
        "system_prompt_key": "gsm8k",
    },
    "factual_recall": {
        "transcript_file": "action_anchors/outputs/transcripts_factual_recall.json",
        "tools": [SEARCH_TOOL],
        "system_prompt_key": "factual_recall",
    },
}


# ---- Helpers --------------------------------------------------------------

def load_transcripts(task_name: str, n_examples: int) -> list[dict]:
    """Load transcript JSON and return the first *n_examples* entries."""
    fp = Path(TASK_CONFIG[task_name]["transcript_file"])
    if not fp.exists():
        print(f"ERROR: Transcript file not found: {fp}")
        print("Run calibration first: uv run python run_calibration.py")
        sys.exit(1)

    with open(fp) as f:
        data = json.load(f)

    selected = data[:n_examples]
    print(f"Loaded {len(selected)}/{len(data)} transcripts from {fp}")
    return selected


def reconstruct_full_text(
    prompt_builder: PromptBuilder,
    system_prompt: str,
    tools: list[dict],
    question: str,
    raw_output: str,
) -> str:
    """Reconstruct the full prompt+response text for attention analysis.

    The model sees: [system + tools + user] + assistant response.
    We reproduce this so the tokenization matches what the model processed.
    """
    prompt = prompt_builder.build_initial_prompt(system_prompt, tools, question)
    # prompt ends with '<|im_start|>assistant\n'
    # raw_output starts with '<think>\n...'
    return prompt + raw_output


# ---- Main pipeline --------------------------------------------------------

def _fraction_to_layer(fraction: float) -> int:
    """Convert a [0.0, 1.0] fraction to a layer index (0-based).

    0.0 → layer 0, 0.5 → mid-layer, 1.0 → last layer.
    """
    layer = int(round(fraction * (N_LAYERS - 1)))
    return max(0, min(layer, N_LAYERS - 1))


def run_analysis(
    task_name: str,
    n_examples: int,
    skip_suppression: bool,
    output_dir: Path,
    top_k: int = 20,
    proximity_ignore: int = 4,
    layer_fracs: list[float] | None = None,
    heads: list[int] | None = None,
):
    """Run the full attention analysis pipeline."""

    # None means "all"
    if layer_fracs is None:
        layer_fracs = [i / (N_LAYERS - 1) for i in range(N_LAYERS)]
    if heads is None:
        heads = list(range(N_HEADS))

    # Load config
    with open("action_anchors/config.yaml") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    task_cfg = TASK_CONFIG[task_name]
    system_prompt = config["tasks"][task_cfg["system_prompt_key"]]["system_prompt"].strip()
    tools = task_cfg["tools"]

    # Load transcripts
    transcripts = load_transcripts(task_name, n_examples)
    if not transcripts:
        return

    # Build prompt builder (uses tokenizer only, lightweight)
    prompt_builder = PromptBuilder(model_name)

    # Load model via HuggingFace (eager attention)
    print(f"\nLoading {model_name} with eager attention...")
    model, tokenizer = load_model(model_name, float32=False, device_map="auto")

    # Resolve layers from fractions
    plot_layers = [_fraction_to_layer(f) for f in layer_fracs]
    plot_heads = heads

    # Build all (layer, head) combos for per-example plots
    layer_head_combos = [(l, h) for l in plot_layers for h in plot_heads]
    print(f"Per-example plots: {len(layer_head_combos)} (layer, head) combos:")
    for l, h in layer_head_combos:
        frac = l / (N_LAYERS - 1) if N_LAYERS > 1 else 0
        print(f"  L{l} (frac={frac:.2f}) H{h}")

    # Storage for cross-example aggregation
    all_kurtosis: list[np.ndarray] = []
    all_vert_scores: list[np.ndarray] = []
    all_sentences: list[list[str]] = []
    # Per (layer, head) combo: list of avg matrices and titles
    all_avg_matrices: dict[tuple[int, int], list[np.ndarray]] = {c: [] for c in layer_head_combos}
    all_titles: list[str] = []

    task_output = output_dir / task_name
    task_output.mkdir(parents=True, exist_ok=True)

    # Create subfolders for each (layer, head) combo
    head_outputs: dict[tuple[int, int], Path] = {}
    for l, h in layer_head_combos:
        sub = task_output / f"L{l}_H{h}"
        sub.mkdir(parents=True, exist_ok=True)
        head_outputs[(l, h)] = sub

    for idx, transcript in enumerate(transcripts):
        example_id = transcript["example_id"]
        thinking = transcript["thinking"]
        question = transcript["question"]
        raw_output = transcript["raw_output"]

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(transcripts)}] {example_id}")
        print(f"{'='*60}")

        # Split CoT into sentences (reuse existing action-anchors splitter)
        split = split_cot_into_sentences(thinking)
        sentences = split.sentences
        if len(sentences) < 3:
            print(f"  Skipping: only {len(sentences)} sentences")
            continue

        print(f"  {len(sentences)} sentences in CoT")
        all_sentences.append(sentences)

        # Reconstruct full text
        full_text = reconstruct_full_text(
            prompt_builder, system_prompt, tools, question, raw_output
        )
        print(f"  Full text: {len(full_text)} chars")

        # ---- Forward pass: extract all attention weights ----
        print("  Extracting attention weights...")
        result = analyze_text(
            full_text,
            model=model,
            tokenizer=tokenizer,
            return_logits=False,
            attn_layers=None,  # all layers
            verbose=False,
        )
        print(f"  {result['input_length']} tokens, {len(result['attention_weights'])} layers extracted")

        # Map sentences to token boundaries within the full text
        # We need to find where the thinking text starts in the full text
        # The thinking text is inside the raw_output after <think>\n
        think_start = full_text.find(thinking)
        if think_start == -1:
            print("  WARNING: Could not locate thinking text in full text, skipping")
            continue

        # Get token boundaries for sentences within the thinking portion
        # We pass the full text and sentences — the function finds them by substring
        try:
            sentence_boundaries = get_sentence_token_boundaries(
                full_text, sentences, tokenizer
            )
        except ValueError as e:
            print(f"  WARNING: Sentence boundary mapping failed: {e}")
            continue

        print(f"  Sentence boundaries mapped ({len(sentence_boundaries)} sentences)")

        # ---- Compute sentence-averaged attention + vertical scores ----
        print("  Computing vertical scores for all heads...")
        vert_scores = get_all_heads_vert_scores(
            result, sentence_boundaries, proximity_ignore=proximity_ignore
        )
        all_vert_scores.append(vert_scores)

        # Kurtosis for this example
        kurt = compute_kurtosis(vert_scores)
        all_kurtosis.append(kurt)

        all_titles.append(f"{example_id}")

        # ---- Per (layer, head) combo plots ----
        for (pl, ph) in layer_head_combos:
            ho = head_outputs[(pl, ph)]

            # Plot 1: Single attention heatmap (head-specific → subfolder)
            if pl in result["attention_weights"]:
                avg_mat = get_avg_attention_matrix(result, pl, ph, sentence_boundaries)
                all_avg_matrices[(pl, ph)].append(avg_mat)

                plot_single_attention_heatmap(
                    avg_mat,
                    title=f"{example_id} — L{pl} H{ph}",
                    output_path=ho / f"heatmap_{example_id}.png",
                )

            # Plot 3: Vertical scores for one layer (head-specific → subfolder)
            if pl < vert_scores.shape[0]:
                plot_vertical_scores_layer(
                    vert_scores[pl],
                    layer=pl,
                    highlight_head=ph,
                    title=f"{example_id} — Layer {pl}",
                    output_path=ho / f"vert_scores_{example_id}.png",
                )

        # ---- Plot 4: Suppression KL heatmap (if not skipped) ----
        if not skip_suppression:
            print("  Computing suppression matrix (this is slow)...")
            supp_mat = compute_suppression_matrix(
                model, tokenizer, full_text, sentence_boundaries, verbose=True
            )
            if supp_mat is not None:
                plot_suppression_heatmap(
                    supp_mat,
                    title=f"{example_id} — KL Suppression",
                    output_path=task_output / f"suppression_{example_id}.png",
                )

        # Free attention weights for this example
        del result
        clear_gpu_memory()

    # ---- Cross-example aggregation plots ----
    print(f"\n{'='*60}")
    print("Generating cross-example plots...")
    print(f"{'='*60}")

    if not all_kurtosis:
        print("No examples processed successfully. Exiting.")
        return

    # ---- Plot 5: Kurtosis scatter + histogram ----
    print("  Kurtosis stats...")
    plot_kurtosis_stats(all_kurtosis, output_dir=task_output)

    # ---- Identify top-k receiver heads ----
    stacked_kurt = np.stack(all_kurtosis, axis=0)  # (n_examples, n_layers, n_heads)
    receiver_heads = get_top_k_receiver_heads(stacked_kurt, top_k=min(top_k, 20))
    print(f"  Top receiver heads: {receiver_heads[:5].tolist()} ...")

    # ---- Plot 2: Grid of attention matrices (head-specific → subfolder) ----
    for (pl, ph) in layer_head_combos:
        matrices = all_avg_matrices[(pl, ph)]
        if len(matrices) >= 2:
            plot_attention_grid(
                matrices,
                all_titles,
                suptitle=f"Attention matrices — L{pl} H{ph}",
                n_cols=min(4, len(matrices)),
                output_path=head_outputs[(pl, ph)] / "attention_grid.png",
            )

    # ---- Plot 3 (aggregate): Top-k receiver heads overlay ----
    if len(all_vert_scores) > 0 and len(receiver_heads) > 0:
        # Average vert scores across examples for top-k heads
        # Use the first example as representative
        first_vs = all_vert_scores[0]
        topk_vs = []
        topk_coords = []
        for layer, head in receiver_heads[:min(10, len(receiver_heads))]:
            if layer < first_vs.shape[0] and head < first_vs.shape[1]:
                topk_vs.append(first_vs[layer, head])
                topk_coords.append([layer, head])
        if topk_vs:
            plot_vertical_scores_top_k(
                np.stack(topk_vs, axis=0),
                np.array(topk_coords),
                title=f"Top receiver heads — {all_titles[0]}",
                output_path=task_output / "vert_scores_topk.png",
            )

    # ---- Plot 6: Taxonomy comparison ----
    if len(all_sentences) >= 1 and len(all_vert_scores) >= 1:
        print("  Taxonomy comparison...")
        plot_taxonomy_comparison(
            all_sentences,
            all_vert_scores,
            receiver_heads,
            plot_type="box",
            output_path=task_output / "taxonomy_comparison.png",
        )

    # ---- Plot 7: Split-half reliability ----
    if len(all_kurtosis) >= 4:
        print("  Split-half reliability...")
        plot_split_half_reliability(
            all_kurtosis,
            output_path=task_output / "split_half_reliability.png",
        )
    else:
        print(f"  Split-half reliability: skipped (need >= 4 examples, have {len(all_kurtosis)})")

    # Clean up model
    del model
    clear_gpu_memory()

    print(f"\nAll plots saved to: {task_output}")
    print("Done!")


# ---- CLI ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="White-box attention analysis for action-anchors transcripts"
    )
    parser.add_argument(
        "--task",
        choices=["gsm8k", "factual_recall"],
        default="gsm8k",
        help="Task to analyze (default: gsm8k)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=5,
        help="Number of transcripts to analyze (default: 5)",
    )
    parser.add_argument(
        "--skip-suppression",
        action="store_true",
        help="Skip suppression KL analysis (slow: O(n_sentences) forward passes per example)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="action_anchors/outputs/attention_plots",
        help="Directory for output plots",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top receiver heads to identify (default: 20)",
    )
    parser.add_argument(
        "--proximity-ignore",
        type=int,
        default=4,
        help="Nearby sentences to ignore for vertical scores (default: 4)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer fractions in [0.0, 1.0] for per-example plots. "
             "0.0 = first layer, 0.5 = middle, 0.8 = ~80%% depth, 1.0 = last layer. "
             "Example: --layers 0.0,0.5,0.8  "
             "If omitted, plots ALL layers (0 through {}).".format(N_LAYERS - 1),
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=None,
        help="Comma-separated attention head indices for per-example plots. "
             "Example: --heads 0,1,2,3  "
             "If omitted, plots ALL heads (0 through {}).".format(N_HEADS - 1),
    )

    args = parser.parse_args()

    # Parse comma-separated values, or None → all
    if args.layers is not None:
        layer_fracs: list[float] | None = [float(x.strip()) for x in args.layers.split(",")]
    else:
        layer_fracs = None  # signals "all layers"

    if args.heads is not None:
        head_indices: list[int] | None = [int(x.strip()) for x in args.heads.split(",")]
    else:
        head_indices = None  # signals "all heads"

    run_analysis(
        task_name=args.task,
        n_examples=args.n_examples,
        skip_suppression=args.skip_suppression,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        proximity_ignore=args.proximity_ignore,
        layer_fracs=layer_fracs,
        heads=head_indices,
    )


if __name__ == "__main__":
    main()
