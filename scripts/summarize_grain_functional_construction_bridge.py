#!/usr/bin/env python3
"""Expand completed response accounting into signed, matched-query mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.construction_bridge import (
    bootstrap_quality,
    candidate_predictions,
    covariance_matrix,
    response_account,
    two_order_allocation,
)


def digest(path):
    """Verify frozen local inputs without exposing their content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component_covariance(components, left, right):
    """Sixteen signed cells whose sum is the covariance of two total responses."""
    values = list(components.values())
    x = np.stack([a[..., left] for a in values], axis=-1)
    y = np.stack([a[..., right] for a in values], axis=-1)
    return covariance_matrix(x, y)


def run(source):
    """Summarize one completed packet; no new predictions, model search or labels."""
    manifest = json.loads((source / "manifest.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    if summary["status"] != "COMPLETE":
        raise ValueError("Complete numerical packet required")
    for name, expected in (
        ("private_inputs.npz", manifest["input_sha256"]),
        ("split_manifest.json", manifest["split_sha256"]),
        *summary["output_hashes"].items(),
    ):
        if digest(source / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    split = json.loads((source / "split_manifest.json").read_text())
    held = ~np.asarray(split["reference_mask"], bool)
    with np.load(source / "private_inputs.npz", allow_pickle=False) as z:
        g, d, y, m = (z[k] for k in ("gradients", "displacement", "natural", "patch"))
    with np.load(source / "private_reference.npz", allow_pickle=False) as z:
        rg, scale, mean = (z[k] for k in ("gradient", "scale", "cell_mean"))
    models, crossed = candidate_predictions(rg, d, mean, scale)
    a = models["private_evidence_private_reader"][held]
    l = np.einsum("avqd,aveqd->aveq", g, d)[held] / scale
    y, m = y[held] / scale, m[held] / scale
    components = response_account(y, m, l, a)
    names = list(components)
    queries = manifest["queries"]
    draws = manifest["bootstrap_draws"]
    rng = np.random.default_rng(manifest["bootstrap_seed"])
    take = rng.integers(len(y), size=(draws, len(y)))
    observed = covariance_matrix(y)
    rows, contrast, pairs = [], [], []
    max_error = 0.0
    for q, r in itertools.combinations(range(len(queries)), 2):
        matrix = component_covariance(components, q, r)
        max_error = max(max_error, float(abs(matrix.sum() - observed[q, r])))
        samples = np.asarray(
            [
                component_covariance({k: v[t] for k, v in components.items()}, q, r)
                for t in take
            ]
        )
        low, high = np.quantile(samples, [0.025, 0.975], axis=0)
        for i, j in itertools.product(range(4), repeat=2):
            rows.append(
                {
                    "left": queries[q],
                    "right": queries[r],
                    "left_component": names[i],
                    "right_component": names[j],
                    "covariance": matrix[i, j],
                    "lower": low[i, j],
                    "upper": high[i, j],
                }
            )
        pair = {"left": queries[q], "right": queries[r]}
        for name, value in (
            ("natural", y),
            ("patch", m),
            ("transported", a),
            ("outside_site", y - m),
        ):
            cov = covariance_matrix(value)
            pair[name + "_covariance"] = cov[q, r]
            pair[name + "_correlation"] = cov[q, r] / np.sqrt(cov[q, q] * cov[r, r])
        pairs.append(pair)
        allocated = two_order_allocation(crossed[held], q, r)
        target = y[..., q] - y[..., r]
        for name in ("difference", "readout", "evidence"):
            contrast.append(
                {
                    "left": queries[q],
                    "right": queries[r],
                    "component": name,
                    **bootstrap_quality(target, allocated[name], draws),
                }
            )
    out = source / "interpretation"
    out.mkdir(exist_ok=False)
    tables = {
        "signed_component_covariance.csv": rows,
        "natural_query_contrasts.csv": contrast,
        "matched_query_relations.csv": pairs,
    }
    for name, table in tables.items():
        pd.DataFrame(table).to_csv(out / name, index=False)
    result = {
        "status": "COMPLETE",
        "scope": "post-result signed accounting expansion; no model selection",
        "evaluation_authors": int(held.sum()),
        "units": "frozen_reference_natural_sd",
        "centering": "within_edit_view",
        "bootstrap_draws": draws,
        "max_sixteen_term_error": max_error,
        "new_neural_calls": 0,
        "label_joins": 0,
        "parent_manifest_sha256": digest(source / "manifest.json"),
        "script_sha256": digest(Path(__file__)),
        "output_hashes": {name: digest(out / name) for name in tables},
        "pair_relations": pairs,
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/functional_construction_bridge_v1",
    )
    run(parser.parse_args().source)
