#!/usr/bin/env python3
"""Render aggregate-only multi-comment results, not questionnaire accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    """Show retained author information, readout location and frozen support separately."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--support-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    table = pd.read_csv(a.analysis_dir / "comparison.csv").set_index("method")
    support = pd.read_csv(a.support_dir / "support_by_volume.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.3), gridspec_kw={"width_ratios": [1.2, 1.4, 0.9]}
    )
    fig.patch.set_facecolor("#fbfaf6")
    for ax in axes:
        ax.set_facecolor("#fbfaf6")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    names = [
        ("GRAIN unit mean", "grain_unit_mean"),
        ("Word TF-IDF", "tfidf_word"),
        ("Char TF-IDF", "tfidf_char"),
        ("BGE pooled", "bge_large_independent_mean"),
        ("E5 pooled", "e5_large_independent_mean"),
        ("Qwen block 10 content", "qwen_block_10_history_mean"),
    ]
    for i, (label, key) in enumerate(names):
        row = table.loc[key + "__chronological"]
        axes[0].errorbar(
            row.author_rank_auc,
            i,
            xerr=[
                [row.author_rank_auc - row.ci_low],
                [row.ci_high - row.author_rank_auc],
            ],
            fmt="o",
            color="#126b72",
            capsize=3,
        )
    axes[0].set_yticks(range(len(names)), [label for label, _ in names])
    axes[0].set_xlim(0.5, 1)
    axes[0].set_xlabel("Same-author rank (95% node bootstrap CI)")
    axes[0].set_title(
        "A. Equal raw-comment evidence\n96 comparison authors", loc="left"
    )
    sites = [
        "input_embedding",
        "block_00",
        "block_05",
        "block_10",
        "block_16",
        "block_21",
        "block_27",
        "final_norm",
    ]
    for mode, label, color in (
        ("history_mean", "Original body tokens", "#126b72"),
        ("decision_behavior", "Behavior decision", "#c05a28"),
        ("decision_disposition", "Disposition decision", "#687378"),
    ):
        rows = table.loc[[f"qwen_{s}_{mode}__chronological" for s in sites]]
        axes[1].plot(
            range(8), rows.author_rank_auc, "o-", color=color, label=label, markersize=4
        )
        axes[1].fill_between(
            range(8), rows.ci_low, rows.ci_high, color=color, alpha=0.10
        )
    axes[1].axhline(0.5, color="#888888", linestyle=":")
    axes[1].set_xticks(
        range(8), ["Input", "0", "5", "10", "16", "21", "27", "Norm"], rotation=35
    )
    axes[1].set_ylim(0.45, 1)
    axes[1].set_xlabel("Frozen Qwen capture site")
    axes[1].set_title("B. Information retention != task judgment", loc="left")
    axes[1].legend(fontsize=8, loc="lower right")
    s = support[support.units >= 12]
    axes[2].plot(s.units, s.ready / s.authors, "o-", color="#c05a28")
    for r in s.itertuples():
        axes[2].annotate(
            f"{r.ready}/{r.authors}",
            (r.units, r.ready / r.authors),
            xytext=(-8, 22) if r.units == 32 else (0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axes[2].set_xticks(s.units)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Observed comments per score")
    axes[2].set_ylabel("Inside frozen GRAIN reference envelope")
    axes[2].set_title("C. Observation support matters", loc="left")
    fig.suptitle(
        "GRAIN vector-mechanism research: independent multi-comment discovery",
        x=0.04,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.04,
        0.03,
        "No personality labels used. These are author-correspondence diagnostics, not Big Five/MBTI prediction.\nCalibration (48) and mechanism holdout (48) remain unopened. Frozen GRAIN support threshold was not relaxed.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.13, 0.99, 0.9), w_pad=2)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
