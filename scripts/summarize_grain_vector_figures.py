#!/usr/bin/env python3
"""Draw aggregate, non-identifying layer/readout/causal comparisons from saved CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """Keep different estimands in separate panels; never imply personality accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--binding-dir", type=Path, required=True)
    parser.add_argument("--patch-analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    natural = pd.read_csv(args.baseline_dir / "comparison.csv").set_index("method")
    binding = pd.read_csv(args.binding_dir / "layer_readability.csv")
    binding = binding[binding.task.eq("narrator_positive")].set_index("method")
    span = pd.read_csv(args.patch_analysis_dir / "span_summary.csv")
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
    names = ["Input", "0", "5", "10", "16", "21", "27", "Norm"]
    x = np.arange(len(sites))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    ax = axes[0]
    n = natural.loc[[f"qwen_{k}__history_mean" for k in sites]]
    ax.plot(x, n.author_rank_auc, "o-", color="#255c80", label="Qwen history mean")
    ax.fill_between(x, n.ci_low, n.ci_high, color="#255c80", alpha=0.15)
    ax.axhline(
        natural.loc["e5_large", "author_rank_auc"], ls="--", color="#ad5b28", label="E5"
    )
    ax.axhline(0.5, color="#888888", lw=0.8)
    ax.set(
        title="A. Independent natural comments",
        ylabel="Author-match concordance",
        ylim=(0.45, 0.85),
    )
    ax.legend(fontsize=8)
    ax = axes[1]
    for view, label, color in (
        ("content_mean", "Content mean", "#255c80"),
        ("evidence_end", "Pre-question endpoint", "#ad5b28"),
    ):
        ax.plot(
            x,
            [binding.loc[f"qwen_{k}__{view}", "heldout_auc"] for k in sites],
            "o-",
            label=label,
            color=color,
        )
    ax.axhline(0.5, color="#888888", lw=0.8)
    ax.set(
        title="B. Synthetic binding: fixed probes",
        ylabel="Held-out relational-fact AUC",
        ylim=(0.4, 1.04),
    )
    ax.legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks(x, names)
        ax.set_xlabel("Residual-stream observation site")
    ax = axes[2]
    for i, site in enumerate(("block_10", "block_16")):
        selected = span[
            (span.site == site)
            & (span.order == "identity_first")
            & (span.task == "narrator_positive")
            & (span.span == "all_evidence")
            & (span.contrast == "full_minus_token_norm_random")
        ].iloc[0]
        ax.errorbar(
            i,
            selected["mean"],
            yerr=[
                [selected["mean"] - selected.ci_low],
                [selected.ci_high - selected["mean"]],
            ],
            fmt="o",
            capsize=5,
            color="#255c80",
        )
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks([0, 1], ["Block 10", "Block 16"])
    ax.set(
        title="C. Selective evidence-span replacement",
        ylabel="Donor-oriented logit effect minus random",
        xlim=(-0.5, 1.5),
    )
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(
        "Readable information and causal-use sites need not coincide", fontsize=14
    )
    fig.supxlabel(
        "A: 24 authors, node-bootstrap CI. B/C: 4 held-out scenario templates; exploratory, not personality validity.",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
