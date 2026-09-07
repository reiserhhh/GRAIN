#!/usr/bin/env python3
"""S3: learn cached comment sensitivity, separating position, content and context."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_functional_basis import author_folds
from scripts.analyze_grain_multiquery_sensitivity import (
    cluster_r_interval,
    corr,
    load_panel,
    response_rows,
)
from scripts.run_grain_real_author_path import BASE, sha, write

FAMILIES = ("position_length", "comment_content", "comment_context")


def feature_blocks(vectors, lengths, family):
    """Preserve observed structure, then append content and contextual interactions."""
    vectors, lengths = np.asarray(vectors, float), np.asarray(lengths, float)
    if len(vectors) != len(lengths) or np.any(lengths < 1):
        raise ValueError("Positive matched comment lengths required")
    positions = np.arange(len(vectors)) / max(len(vectors) - 1, 1)
    blocks = [np.column_stack([np.log1p(lengths), positions, positions**2])]
    if family != "position_length":
        blocks.append(vectors)
    if family == "comment_context":
        context = vectors.mean(0, keepdims=True)
        blocks.extend([np.repeat(context, len(vectors), axis=0), vectors * context])
    return blocks


def fit_scale(blocks, rows):
    """Fit reference scales on training comments only, retaining each block's weight."""
    return [(b[rows].mean(0), np.maximum(b[rows].std(0), 1e-6)) for b in blocks]


def scale_blocks(blocks, rows, scales):
    """Apply fixed training scales to held-author comments without batch adaptation."""
    return np.column_stack(
        [
            (b[rows] - center) / scale / np.sqrt(b.shape[1])
            for b, (center, scale) in zip(blocks, scales, strict=True)
        ]
    )


def panel_arrays(panel):
    """Return per-token block16 sensitivities, without reading raw text or labels."""
    manifest, records = load_panel(panel)
    encoded = panel / "private_encoder.npz"
    if sha(encoded) != manifest["encoder_sha256"]:
        raise ValueError("Frozen per-comment vectors changed")
    with np.load(encoded, allow_pickle=False) as z:
        vectors = z["comment_vectors"]
    metadata, blocks = [], {family: [] for family in FAMILIES}
    for material, packet, _ in records:
        h = vectors[packet["material_row"]]
        for query, q in packet["queries"].items():
            lengths = np.asarray(q["body_token_counts"])
            values = np.asarray(q["comment_attribution"])
            for j, (length, value) in enumerate(zip(lengths, values, strict=True)):
                metadata.append(
                    {
                        "author_id": material["author_id"],
                        "view": material["view"],
                        "query": query,
                        "comment": j,
                        "tokens": int(length),
                        "target": float(value / length),
                        "attribution": float(value),
                        "e1": j in material["e1"],
                        "e2": j in material["e2"],
                    }
                )
            for family in FAMILIES:
                blocks[family].append(feature_blocks(h, lengths, family))
    blocks = {
        family: [np.vstack([view[j] for view in views]) for j in range(len(views[0]))]
        for family, views in blocks.items()
    }
    responses, _, _ = response_rows(records)
    return pd.DataFrame(metadata), blocks, responses


def main():
    """Nested author-CV of three finite sensitivity models; no new observer calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private sensitivity data must stay in ignored results")
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S3_COMMENT_SENSITIVITY_V1",
            "script_sha256": sha(__file__),
            "panel_sha256": sha(args.panel / "manifest.json"),
            "families": FAMILIES,
            "alphas": [1, 10, 100],
            "target": "signed block16 gradient-times-state per body token",
            "selection": "five outer / three inner author folds; per-token MSE only",
            "natural_responses_used_for_selection": False,
            "psychological_labels_read": False,
        },
    )
    start = time.monotonic()
    meta, blocks, responses = panel_arrays(args.panel)
    all_groups = meta.author_id.to_numpy()
    outer = author_folds(all_groups, 5, "s3_outer")
    predictions, selection = [], []
    with threadpool_limits(limits=2):
        for query in sorted(meta["query"].unique()):
            selected = np.flatnonzero(meta["query"].to_numpy() == query)
            target = meta.target.to_numpy()
            for family in FAMILIES:
                for fold in range(5):
                    train = selected[outer[selected] != fold]
                    test = selected[outer[selected] == fold]
                    inner = author_folds(all_groups[train], 3, f"s3_inner_{fold}")
                    errors = {alpha: [] for alpha in (1.0, 10.0, 100.0)}
                    for f in range(3):
                        t, v = train[inner != f], train[inner == f]
                        scale = fit_scale(blocks[family], t)
                        xt, xv = [
                            scale_blocks(blocks[family], ix, scale) for ix in (t, v)
                        ]
                        for alpha, losses in errors.items():
                            model = Ridge(alpha=alpha).fit(xt, target[t])
                            losses.append(
                                float(np.mean((model.predict(xv) - target[v]) ** 2))
                            )
                    alpha = min(errors, key=lambda a: (np.mean(errors[a]), a))
                    scale = fit_scale(blocks[family], train)
                    xt, xv = [
                        scale_blocks(blocks[family], ix, scale) for ix in (train, test)
                    ]
                    model = Ridge(alpha=alpha).fit(xt, target[train])
                    part = meta.iloc[test].copy()
                    part["prediction"] = model.predict(xv)
                    part["baseline"] = target[train].mean()
                    part["family"], part["fold"] = family, fold
                    predictions.append(part)
                    selection.append(
                        {"query": query, "family": family, "fold": fold, "alpha": alpha}
                    )
                print(f"S3 {query}/{family} complete", flush=True)
    frame = pd.concat(predictions, ignore_index=True)
    metrics, surrogate = [], []
    for (query, family), part in frame.groupby(["query", "family"]):
        within = [
            corr(g.target, g.prediction) for _, g in part.groupby(["author_id", "view"])
        ]
        within = [v for v in within if v is not None]
        mse = np.mean((part.prediction - part.target) ** 2)
        baseline = np.mean((part.baseline - part.target) ** 2)
        metrics.append(
            {
                "query": query,
                "family": family,
                "authors": part.author_id.nunique(),
                "comments": len(part),
                **cluster_r_interval(part.target, part.prediction, part.author_id),
                "mean_within_view_r": float(np.mean(within)) if within else None,
                "mse_ratio_to_mean": float(mse / baseline),
            }
        )
        for edit in ("e1", "e2"):
            selected = part[part[edit]].copy()
            selected["pred_sum"] = -selected.prediction * selected.tokens
            selected["target_sum"] = -selected.target * selected.tokens
            sums = selected.groupby(["author_id", "view"])[
                ["pred_sum", "target_sum"]
            ].sum()
            actual = responses[(responses["query"] == query) & (responses.edit == edit)]
            actual = actual.set_index(["author_id", "view"])
            joined = sums.join(actual[["natural"]], validate="one_to_one")
            surrogate.append(
                {
                    "query": query,
                    "family": family,
                    "edit": edit,
                    "surrogate_fidelity_r": corr(joined.target_sum, joined.pred_sum),
                    "natural_response_r": corr(joined.natural, joined.pred_sum),
                    "natural_mse_ratio_to_zero": float(
                        np.mean((joined.pred_sum - joined.natural) ** 2)
                        / np.mean(joined.natural**2)
                    ),
                }
            )
    frame.to_csv(out / "private_predictions.csv", index=False)
    pd.DataFrame(selection).to_csv(out / "selected_models.csv", index=False)
    table, response_table = pd.DataFrame(metrics), pd.DataFrame(surrogate)
    table.to_csv(out / "metrics.csv", index=False)
    response_table.to_csv(out / "removal_surrogates.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "neural_forwards": 0,
            "labels_read": False,
            "seconds": time.monotonic() - start,
            "metrics": metrics,
            "surrogates": surrogate,
        },
    )
    report = "# S3 comment sensitivity from fixed E5 features\n\n"
    report += "Per-token block16 attribution is a local surrogate, not a deletion derivative.\n"
    report += "Raw text and psychological labels were not read. Natural effects were never used to fit/select.\n\n"
    report += table.to_markdown(index=False, floatfmt=".4f") + "\n\n"
    report += response_table.to_markdown(index=False, floatfmt=".4f") + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(response_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
