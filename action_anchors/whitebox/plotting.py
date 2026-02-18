"""All 7 thought-anchors-style visualizations, adapted for action-anchors.

1. Single attention heatmap
2. Grid of attention matrices
3. Vertical attention score line plots
4. Suppression KL heatmap
5. Kurtosis scatter + histogram
6. Taxonomy comparison (receiver head scores by sentence category)
7. Split-half reliability
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white_to_blues(N: int = 256):
    """Custom white-to-blue colormap (matching thought-anchors)."""
    blues = plt.cm.Blues(np.linspace(0, 1, N))
    white_blue = mcolors.LinearSegmentedColormap.from_list(
        "wb", [(1, 1, 1), (0, 0, 1)]
    )(np.linspace(0, 1, N))
    w = np.linspace(1, 0, N)[:, None]
    blended = w * white_blue + (1 - w) * blues
    return mcolors.LinearSegmentedColormap.from_list("WhiteToBlues", blended)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Sentence taxonomy heuristics for tool-calling CoTs
# ---------------------------------------------------------------------------

_TAXONOMY_PATTERNS = {
    "tool_deliberation": re.compile(
        r"\b(calculator|search|tool|use\s+it|web\s+search|let\s+me\s+input"
        r"|let\s+me\s+use|should\s+I\s+use|maybe\s+I\s+should\s+use)\b",
        re.IGNORECASE,
    ),
    "computation": re.compile(
        r"(?:\d+\s*[\+\-\*/]\s*\d+|=\s*\d+|\d+\s*\*\s*\d+|\d+\s*\+\s*\d+)",
    ),
    "verification": re.compile(
        r"\b(check|verify|wait|let\s+me\s+check|hmm|double.check|re.?check|"
        r"let\s+me\s+re|is\s+that\s+right|that'?s\s+correct)\b",
        re.IGNORECASE,
    ),
    "conclusion": re.compile(
        r"\b(so\s+the\s+answer|therefore|thus|in\s+conclusion|the\s+answer\s+is|"
        r"final\s+answer|so\s+i\s+think|yep|result)\b",
        re.IGNORECASE,
    ),
    "reasoning": re.compile(
        r"\b(because|since|means|implies|if\s+we|which\s+means|so\s+that|"
        r"this\s+gives|we\s+know|first|next|then)\b",
        re.IGNORECASE,
    ),
}

_TAXONOMY_COLORS = {
    "tool_deliberation": "tab:red",
    "computation": "tab:green",
    "verification": "tab:purple",
    "conclusion": "tab:blue",
    "reasoning": "tab:orange",
    "other": "tab:gray",
}

_TAXONOMY_LABELS = {
    "tool_deliberation": "Tool\nDeliberation",
    "computation": "Computation",
    "verification": "Verification",
    "conclusion": "Conclusion",
    "reasoning": "Reasoning",
    "other": "Other",
}


def classify_sentence(sentence: str) -> str:
    """Classify a CoT sentence into a taxonomy category via keyword heuristics."""
    for cat in ("tool_deliberation", "verification", "conclusion", "computation"):
        if _TAXONOMY_PATTERNS[cat].search(sentence):
            return cat
    if _TAXONOMY_PATTERNS["reasoning"].search(sentence):
        return "reasoning"
    return "other"


# ---------------------------------------------------------------------------
# 1. Single attention heatmap
# ---------------------------------------------------------------------------

def plot_single_attention_heatmap(
    avg_matrix: np.ndarray,
    title: str = "Sentence-Averaged Attention",
    output_path: Optional[Path] = None,
    vmax_quantile: float = 0.95,
) -> None:
    """Plot a single sentence-averaged attention matrix as a heatmap."""
    cmap = _white_to_blues()
    # Remove first/last rows (prompt/output framing) if large enough
    mat = avg_matrix
    if mat.shape[0] > 4:
        mat = mat[1:-1, 1:-1]

    tril = np.tril(mat, k=-1)
    vmax = np.nanquantile(tril[tril > 0], vmax_quantile) if np.any(tril > 0) else 1.0

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, vmin=0, vmax=vmax, cmap=cmap)
    plt.colorbar(label="Attention weight", shrink=0.8)
    plt.xlabel("Source sentence", fontsize=11)
    plt.ylabel("Target sentence", fontsize=11)
    plt.title(title, fontsize=12)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_path:
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Grid of attention matrices
# ---------------------------------------------------------------------------

def plot_attention_grid(
    matrices: List[np.ndarray],
    titles: List[str],
    suptitle: str = "Attention matrices across transcripts",
    n_cols: int = 4,
    output_path: Optional[Path] = None,
) -> None:
    """Plot an N x M grid of sentence-averaged attention matrices."""
    n = len(matrices)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    cmap = _white_to_blues()

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axs = axs[np.newaxis, :] if n_cols > 1 else np.array([[axs]])
    if n_cols == 1:
        axs = axs[:, np.newaxis]

    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axs[r, c]
        if idx >= n:
            ax.axis("off")
            continue
        mat = matrices[idx]
        if mat.shape[0] > 4:
            mat = mat[1:-1, 1:-1]
        tril = np.tril(mat, k=-1)
        vmax = np.nanquantile(tril[tril > 0], 0.95) if np.any(tril > 0) else 1.0
        ax.imshow(mat, vmin=0, vmax=vmax, cmap=cmap)
        ax.set_title(titles[idx], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if r == n_rows - 1:
            ax.set_xlabel("Sentence", fontsize=8)
        if c == 0:
            ax.set_ylabel("Sentence", fontsize=8)

    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()

    if output_path:
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Vertical attention score line plots
# ---------------------------------------------------------------------------

def plot_vertical_scores_layer(
    vert_scores_all_heads: np.ndarray,
    layer: int,
    highlight_head: Optional[int] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Plot vertical attention scores for all heads in a single layer.

    Args:
        vert_scores_all_heads: Shape ``(n_heads, n_sentences)``.
        layer: Layer index (for labeling).
        highlight_head: Optional head to highlight in navy.
    """
    n_heads = vert_scores_all_heads.shape[0]
    plt.figure(figsize=(8, 3))
    plt.rcParams["font.size"] = 11

    for h in range(n_heads):
        vs = vert_scores_all_heads[h]
        if highlight_head is not None and h == highlight_head:
            plt.plot(vs, label=f"Head {h}", color="navy", zorder=100, linewidth=1.5)
        else:
            plt.plot(vs, label=f"Head {h}", linewidth=0.8, alpha=0.6)

    plt.title(title or f"Layer {layer}: vertical attention scores", fontsize=12)
    plt.xlabel("Sentence position", fontsize=11)
    plt.ylabel("Vertical attention score", fontsize=11)

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, -3))
    plt.gca().yaxis.set_major_formatter(fmt)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_path:
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")
    plt.close()


def plot_vertical_scores_top_k(
    vert_scores_topk: np.ndarray,
    coords: np.ndarray,
    title: str = "Top-k receiver heads",
    output_path: Optional[Path] = None,
) -> None:
    """Overlay vertical scores for the top-k receiver heads.

    Args:
        vert_scores_topk: Shape ``(k, n_sentences)``.
        coords: Shape ``(k, 2)`` with ``[layer, head]`` pairs.
    """
    plt.figure(figsize=(8, 3))
    plt.rcParams["font.size"] = 14

    for i in range(len(coords)):
        layer, head = coords[i]
        vs = vert_scores_topk[i]
        plt.plot(vs, label=f"L{layer}H{head}", linewidth=1)

    plt.title(title, fontsize=13)
    plt.xlabel("Sentence position", fontsize=14)
    plt.ylabel("Receiver head score", fontsize=14)

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, -3))
    plt.gca().yaxis.set_major_formatter(fmt)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_path:
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Suppression KL heatmap
# ---------------------------------------------------------------------------

def plot_suppression_heatmap(
    suppression_matrix: np.ndarray,
    title: str = "KL Suppression Matrix",
    output_path: Optional[Path] = None,
) -> None:
    """Sentence-to-sentence KL suppression heatmap (white-to-red)."""
    cmap = mcolors.LinearSegmentedColormap.from_list("wr", [(1, 1, 1), (0.8, 0, 0)])

    mat = suppression_matrix.copy()
    # Replace NaN for display
    vmin = np.nanmin(mat)
    vmax = np.nanquantile(mat, 0.95) if not np.all(np.isnan(mat)) else 0

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(label="log KL divergence", shrink=0.8)
    plt.xlabel("Suppressed sentence", fontsize=11)
    plt.ylabel("Receiver sentence", fontsize=11)
    plt.title(title, fontsize=12)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_path:
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Kurtosis scatter + histogram
# ---------------------------------------------------------------------------

def plot_kurtosis_stats(
    kurtosis_per_example: List[np.ndarray],
    output_dir: Optional[Path] = None,
) -> None:
    """Scatter plot of kurtosis per head by layer + histogram.

    Args:
        kurtosis_per_example: List of ``(n_layers, n_heads)`` kurtosis arrays,
            one per example.
    """
    # Average kurtosis across examples
    stacked = np.stack(kurtosis_per_example, axis=0)
    mean_kurt = np.nanmean(stacked, axis=0)  # (n_layers, n_heads)
    n_layers, n_heads = mean_kurt.shape

    # ---- Scatter: kurtosis by layer ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    layers_flat = []
    kurt_flat = []
    for layer in range(n_layers):
        for head in range(n_heads):
            val = mean_kurt[layer, head]
            if not np.isnan(val):
                layers_flat.append(layer)
                kurt_flat.append(val)

    ax1.scatter(layers_flat, kurt_flat, alpha=0.4, s=15, color="dodgerblue")
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Mean kurtosis", fontsize=11)
    ax1.set_title("Kurtosis by layer", fontsize=12)
    ax1.spines[["top", "right"]].set_visible(False)

    # ---- Histogram ----
    valid = np.array(kurt_flat)
    ax2.hist(valid, bins=40, color="dodgerblue", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Mean kurtosis", fontsize=11)
    ax2.set_ylabel("Count (heads)", fontsize=11)
    ax2.set_title("Distribution of kurtosis across heads", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if output_dir:
        fp = _ensure_dir(output_dir) / "kurtosis_stats.png"
        plt.savefig(fp, dpi=300)
        print(f"Saved: {fp}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Taxonomy comparison (receiver head scores by sentence category)
# ---------------------------------------------------------------------------

def plot_taxonomy_comparison(
    sentences_all: List[List[str]],
    vert_scores_all: List[np.ndarray],
    top_k_coords: np.ndarray,
    plot_type: str = "box",
    output_path: Optional[Path] = None,
) -> None:
    """Box/bar plots comparing receiver head scores across sentence categories.

    Args:
        sentences_all: List of sentence lists, one per example.
        vert_scores_all: List of ``(n_layers, n_heads, n_sentences)`` arrays.
        top_k_coords: ``(k, 2)`` receiver head coordinates.
        plot_type: ``"box"`` or ``"bar"``.
    """
    tags_order = ["tool_deliberation", "computation", "verification", "reasoning", "conclusion", "other"]

    # Build per-sentence receiver scores and categories
    records: List[Dict] = []
    for ex_idx, (sentences, all_vs) in enumerate(zip(sentences_all, vert_scores_all)):
        # Average receiver head vertical scores
        rec_scores = []
        for layer, head in top_k_coords:
            if layer < all_vs.shape[0] and head < all_vs.shape[1]:
                rec_scores.append(all_vs[layer, head, :len(sentences)])
        if not rec_scores:
            continue
        mean_rec = np.nanmean(np.stack(rec_scores, axis=0), axis=0)

        for s_idx, sent in enumerate(sentences):
            if s_idx >= len(mean_rec):
                break
            cat = classify_sentence(sent)
            records.append({
                "example": ex_idx,
                "sentence_idx": s_idx,
                "category": cat,
                "receiver_score": mean_rec[s_idx],
            })

    if not records:
        print("No data for taxonomy plot.")
        return

    import pandas as pd

    df = pd.DataFrame(records)
    df = df[df["category"].isin(tags_order)]

    labels = [_TAXONOMY_LABELS.get(t, t) for t in tags_order]
    colors = [_TAXONOMY_COLORS.get(t, "gray") for t in tags_order]

    plt.figure(figsize=(7, 3))
    plt.rcParams["font.size"] = 11

    df["label"] = df["category"].map(_TAXONOMY_LABELS)

    if plot_type == "box":
        sns.boxplot(
            data=df,
            x="label",
            y="receiver_score",
            order=labels,
            palette={_TAXONOMY_LABELS[t]: _TAXONOMY_COLORS[t] for t in tags_order},
            width=0.6,
        )
    else:
        means = []
        ses = []
        for tag in tags_order:
            vals = df[df["category"] == tag]["receiver_score"].dropna().values
            means.append(np.mean(vals) if len(vals) else 0)
            ses.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        plt.bar(
            labels, means, yerr=np.array(ses) * 1.96,
            color=colors, edgecolor="black", linewidth=0.5, capsize=5, alpha=0.8,
        )

    plt.xlabel("")
    plt.ylabel("Mean receiver-head score", fontsize=11)
    plt.title("Receiver-head scores by sentence category", fontsize=12)
    plt.xticks(fontsize=9)

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, -3))
    plt.gca().yaxis.set_major_formatter(fmt)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_path:
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 7. Split-half reliability
# ---------------------------------------------------------------------------

def plot_split_half_reliability(
    kurtosis_per_example: List[np.ndarray],
    output_path: Optional[Path] = None,
) -> None:
    """Scatter: kurtosis from even-indexed examples vs odd-indexed examples.

    Tests whether receiver head identification is stable across subsets.

    Args:
        kurtosis_per_example: List of ``(n_layers, n_heads)`` kurtosis arrays.
    """
    if len(kurtosis_per_example) < 4:
        print(f"Split-half reliability needs >= 4 examples, got {len(kurtosis_per_example)}. Skipping.")
        return

    even = np.stack(kurtosis_per_example[::2], axis=0)
    odd = np.stack(kurtosis_per_example[1::2], axis=0)

    mean_even = np.nanmean(even, axis=0)
    mean_odd = np.nanmean(odd, axis=0)

    # Flatten, exclude NaN / layer 0
    mean_even[0, :] = np.nan
    mean_odd[0, :] = np.nan

    e_flat = mean_even.flatten()
    o_flat = mean_odd.flatten()
    valid = ~(np.isnan(e_flat) | np.isnan(o_flat))
    e_v = e_flat[valid]
    o_v = o_flat[valid]

    if len(e_v) < 3:
        print("Not enough valid heads for reliability plot.")
        return

    r, p = stats.pearsonr(e_v, o_v)

    plt.figure(figsize=(4, 3.5))
    plt.rcParams["font.size"] = 11
    plt.scatter(e_v, o_v, alpha=0.25, s=20, color="dodgerblue")

    # Line of best fit
    z = np.polyfit(e_v, o_v, 1)
    x_line = np.array([e_v.min(), e_v.max()])
    plt.plot(x_line, z[0] * x_line + z[1], "k--", alpha=0.7, linewidth=1.5)

    plt.text(
        0.6, 0.7, f"r = {r:.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.xlabel("Even-indexed examples kurtosis", fontsize=10)
    plt.ylabel("Odd-indexed examples kurtosis", fontsize=10)
    plt.title("Split-half reliability", fontsize=12)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.axis("square")
    plt.tight_layout()

    if output_path:
        _ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")
    plt.close()

    print(f"Split-half reliability: r={r:.3f}, p={p:.4f}")
