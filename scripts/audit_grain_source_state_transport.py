#!/usr/bin/env python3
"""Replay and independently account for NEXT52 without new fits or model calls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.source_state_transport import (
    ARMS,
    body_directions,
)
from scripts.run_grain_source_state_transport import (
    BASE,
    ENDPOINT,
    SOURCE,
    arrays,
    frozen,
    read,
    reconstruct,
    ref,
    selected,
    sha,
    verify,
    verify_hashes,
    write,
)


def audit(out):
    """Check exact packets, source-only assets, units, metrics and response geometry."""
    m = verify(out)
    mf, pf, rf = (
        frozen(out, name) for name in ("models", "predictions", "replay_predictions")
    )
    summary = read(out / "analysis/summary.json")
    verify_hashes(out / "analysis", summary["files"])
    replay_summary = read(out / "replay_analysis/summary.json")
    verify_hashes(out / "replay_analysis", replay_summary["files"])
    if set(pf["files"]) != set(rf["files"]) or len(pf["files"]) != 24:
        raise ValueError("Exact 24 prediction packets required")
    for name in pf["files"]:
        a, b = (
            arrays(out / "predictions" / name),
            arrays(out / "replay_predictions" / name),
        )
        if set(a) != set(b):
            raise ValueError("Prediction field mismatch")
        for field in a:
            np.testing.assert_array_equal(a[field], b[field])
    for name in summary["files"]:
        if sha(out / "analysis" / name) != sha(out / "replay_analysis" / name):
            raise ValueError("Replay table differs: " + name)
    if summary["retained"] != replay_summary["retained"]:
        raise ValueError("Replay decision mismatch")
    metric_error = 0.0
    for raw, table, groups in (
        ("private_actions.csv", "query_summary.csv", ["cohort", "arm", "query_name"]),
        ("private_contrasts.csv", "contrast_summary.csv", ["cohort", "arm"]),
    ):
        frame = pd.read_csv(out / "analysis" / raw)
        tab = pd.read_csv(out / "analysis" / table).set_index(groups)
        for key, g in frame.groupby(groups):
            y, p = g.actual.to_numpy(), g.predicted.to_numpy()
            got = tab.loc[key]
            metric_error = max(
                metric_error,
                abs(np.corrcoef(y, p)[0, 1] - got.action_r),
                abs(np.dot(y - p, y - p) / np.dot(y, y) - got.action_mse_over_zero),
            )
    jobs = read(out / "private_jobs.json")
    fixed_error, unit_error, rows = 0.0, 0.0, []
    for j in jobs:
        members = mf["sources"][ref(j)]
        if (j["cohort"] == "source4" and j["index"] in members) or len(members) != (
            3 if j["cohort"] == "source4" else 4
        ):
            raise ValueError("Source exclusion invalid")
        fixed = arrays(out / "private_fixed" / (j["key"] + ".npz"))
        old_r = selected(ENDPOINT / "predictions" / (j["key"] + ".npz"), ("readers",))[
            "readers"
        ]
        old_q = selected(
            BASE / "logic_source_query_selection_v1/predictions" / (j["key"] + ".npz"),
            ("anchors",),
        )["anchors"][:, 0]
        fixed_error = max(
            fixed_error,
            float(np.max(abs(fixed["r"] - old_r))),
            float(np.max(abs(fixed["q"] - old_q))),
        )
        models = {
            arm: arrays(out / "models" / f"{ref(j)}_{arm}.npz") for arm in ARMS[1:]
        }
        for model in models.values():
            np.testing.assert_array_equal(model["members"], members)
            assert int(model["rank"]) == 32 and float(model["alpha"]) == 0.01
            np.testing.assert_allclose(
                model["basis"].T @ model["basis"], np.eye(32), atol=1e-12
            )
        for q, k in zip(
            ("neuroticism", "confirmation_seeking"), j["query_keys"], strict=True
        ):
            data = arrays(out / "private_inputs" / (k + ".npz"))
            packet = arrays(out / "predictions" / (k + ".npz"))
            native = selected(SOURCE / "private_capture" / (k + ".npz"), ("d",))[
                "d"
            ].astype(float)
            d = body_directions(data)
            h = data["state"][data["positions"]].astype(float)
            g = data["grain"]
            hc, gc = h - h.mean(0), g - g.mean(0)
            for f in range(4):
                raw = gc[:, f, None] * hc
                expected = (raw * np.sqrt(np.mean(h * h) / np.mean(raw * raw))).astype(
                    np.float32
                )
                unit_error = max(
                    unit_error, float(np.max(abs(d[f, data["positions"]] - expected)))
                )
            for ai, arm in enumerate(ARMS):
                _, dd = reconstruct(data, packet, models, ai)
                for f in range(4):
                    y, p = native[f].ravel(), dd[f].ravel()
                    den = float(np.dot(y, y))
                    orig = d[f].ravel()
                    projection = d[f] @ models["pointwise"]["basis"]
                    rows.append(
                        {
                            "author_key": j["key"],
                            "cohort": j["cohort"],
                            "query_name": q,
                            "arm": arm,
                            "factor": f"whole:F{f + 1:02d}",
                            "native_norm_over_input": float(
                                np.linalg.norm(y) / np.linalg.norm(orig)
                            ),
                            "prediction_norm_over_native": float(
                                np.linalg.norm(p) / np.linalg.norm(y)
                            ),
                            "tangent_cosine": float(
                                np.dot(y, p) / np.linalg.norm(y) / np.linalg.norm(p)
                            ),
                            "tangent_projected_gain": float(np.dot(y, p) / den),
                            "input_direction_PCA_energy": float(
                                np.sum(projection**2) / np.dot(orig, orig)
                            ),
                        }
                    )
    gf = pd.DataFrame(rows)
    gf.to_csv(out / "analysis/private_gain_geometry.csv", index=False)
    columns = [
        "native_norm_over_input",
        "prediction_norm_over_native",
        "tangent_cosine",
        "tangent_projected_gain",
        "input_direction_PCA_energy",
    ]
    gf.groupby(["cohort", "arm", "query_name"])[columns].mean().reset_index().to_csv(
        out / "analysis/gain_geometry_summary.csv", index=False
    )
    if metric_error > 1e-11 or fixed_error != 0 or unit_error != 0:
        raise ValueError("Independent replay audit failed")
    budget = m["prepare_seconds"] + mf["seconds"] + pf["seconds"]
    if budget > m["max_seconds"] or mf["fits"] != 10:
        raise ValueError("Bounded construction budget violated")
    result = {
        "status": "PASS",
        "exact_prediction_packets": 24,
        "exact_primary_tables": len(summary["files"]),
        "fixed_anchor_reader_max_error": fixed_error,
        "original_BODY_units_max_error": unit_error,
        "independent_metric_max_error": float(metric_error),
        "primary_compute_seconds": budget,
        "cached_replay_seconds": rf["seconds"],
        "native_calls": 0,
        "human_labels_read": False,
        "fit_denial_probes": mf["denial_probes"],
        "prediction_denial_probes": pf["denial_probes"],
        "extra_denials": mf["extra_denials"]
        + pf["extra_denials"]
        + rf["extra_denials"],
        "primary_commit": m["commit"],
        "audit_code_sha256": sha(Path(__file__)),
        "post_outcome_diagnostic": "No tuning: derivative norm, angle and source PCA coverage",
        "diagnostic_files": {
            n: sha(out / "analysis" / n)
            for n in ("private_gain_geometry.csv", "gain_geometry_summary.csv")
        },
    }
    write(out / "replay_audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        audit(BASE / "logic_source_state_transport_v1")
