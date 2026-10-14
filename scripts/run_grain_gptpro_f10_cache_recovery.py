#!/usr/bin/env python3
"""Reconnect frozen text contrasts and native interventions without target mixing."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    """Rebuild the source direct operator and keep native targets in separate rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = (
        "action_arrays.npz",
        "pairs.csv",
        "pair_order.json",
        "native_effects.jsonl",
        "native_status.json",
    )
    manifest = {
        name: hashlib.sha256((args.cache / name).read_bytes()).hexdigest()
        for name in names
    }
    if (
        json.loads((args.cache / "native_status.json").read_text())["status"]
        != "COMPLETE"
    ):
        raise ValueError("incomplete native cache")
    pairs = list(csv.DictReader((args.cache / "pairs.csv").open()))
    order = json.loads((args.cache / "pair_order.json").read_text())
    if order != [r["pair"] for r in pairs]:
        raise ValueError("pair ordering mismatch")
    with np.load(args.cache / "action_arrays.npz") as z:
        dx, ds, source, direct = z["dx"], z["ds"], z["source"], z["direct"]
    np.testing.assert_array_equal(source, [r["split"] == "source" for r in pairs])
    if source.sum() != 36 or (~source).sum() != 72:
        raise ValueError("frozen role counts")
    scale = dx[source].std(axis=0)
    scale[scale < 1e-12] = 1
    x = dx[source] / scale
    coef = np.linalg.lstsq(
        np.vstack([x, np.eye(4)]), np.vstack([ds[source], np.zeros((4, 2))]), rcond=None
    )[0]
    rebuilt = (dx / scale) @ coef
    np.testing.assert_allclose(rebuilt, direct, atol=1e-10, rtol=1e-8)
    queries = ("extraversion", "neuroticism")
    grid = np.array([0, 0.25, 0.5, 0.75, 1.0])
    selected = {}
    choices = []
    for j, query in enumerate(queries):
        loss = np.array(
            [np.mean((c * rebuilt[source, j] - ds[source, j]) ** 2) for c in grid]
        )
        best = float(grid[np.flatnonzero(loss <= loss.min() + 1e-14)[0]])
        selected[query] = best
        choices.extend(
            {
                "query": query,
                "c": float(c),
                "source_mse": float(l),
                "selected": bool(c == best),
            }
            for c, l in zip(grid, loss, strict=True)
        )
    native = [
        json.loads(line)
        for line in (args.cache / "native_effects.jsonl").read_text().splitlines()
    ]
    lookup = {(r["pair"], r["query"], r["kind"]): r for r in native}
    if len(lookup) != 576 or any(r["split"] == "source" for r in native):
        raise ValueError("unexpected native coverage")
    rows = []
    for i, pair in enumerate(pairs):
        if source[i]:
            continue
        for j, query in enumerate(queries):
            compiled = lookup[(pair["pair"], query, "compiled")]
            donor = lookup[(pair["pair"], query, "donor")]
            np.testing.assert_allclose(compiled["contrast"], ds[i, j], atol=1e-12)
            for target, value, methods in (
                (
                    "full_text_contrast",
                    ds[i, j],
                    {
                        "zero": 0.0,
                        "direct": rebuilt[i, j],
                        "source_shrunk_direct": selected[query] * rebuilt[i, j],
                    },
                ),
                (
                    "donor_native_action",
                    donor["effect"],
                    {"zero": 0.0, "compiled_native": compiled["effect"]},
                ),
                (
                    "full_text_contrast_native",
                    ds[i, j],
                    {
                        "zero": 0.0,
                        "compiled_native": compiled["effect"],
                        "donor_native": donor["effect"],
                    },
                ),
            ):
                for method, prediction in methods.items():
                    rows.append(
                        {
                            "pair": pair["pair"],
                            "family": pair["family"],
                            "situation": pair["situation"],
                            "seed": pair["seed"],
                            "split": pair["split"],
                            "query": query,
                            "target": target,
                            "method": method,
                            "prediction": float(prediction),
                            "truth": float(value),
                            "squared_error": float((prediction - value) ** 2),
                        }
                    )
    summaries = []
    for key in dict.fromkeys((r["query"], r["target"], r["method"]) for r in rows):
        subset = [r for r in rows if (r["query"], r["target"], r["method"]) == key]
        p = np.array([r["prediction"] for r in subset])
        y = np.array([r["truth"] for r in subset])
        summaries.append(
            {
                "query": key[0],
                "target": key[1],
                "method": key[2],
                "n": len(subset),
                "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                "pearson": None
                if p.std() == 0 or y.std() == 0
                else float(np.corrcoef(p, y)[0, 1]),
            }
        )
    for filename, data in (
        ("source_shrinkage.csv", choices),
        ("paired_predictions.csv", rows),
        ("metrics.csv", summaries),
    ):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    result = {
        "status": "REAL_CACHE_RECONNECTED_NATIVE_SOURCE_GAP_EXPLICIT",
        "source_pairs": 36,
        "application_pairs": 72,
        "native_effects": len(native),
        "max_direct_rebuild_difference": float(np.max(np.abs(rebuilt - direct))),
        "source_selected_full_contrast_c": selected,
        "source_selection": "in-sample frozen-source MSE,not cross-validation",
        "native_shrinkage": None,
        "native_shrinkage_reason": "no source native interventions;source full-text contrasts are a different target",
        "native_J_metric": None,
        "native_J_metric_reason": "cached head differences are not gradients;no common four-edit chart established",
        "input_hashes": manifest,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "input_hashes"}), flush=True
    )


if __name__ == "__main__":
    main()
