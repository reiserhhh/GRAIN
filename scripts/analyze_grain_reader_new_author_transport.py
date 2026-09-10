#!/usr/bin/env python3
"""Evaluate frozen ensemble fields and GRAIN actions on eight new reader authors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.native_material_reader import direction
from grain_research.vector_mechanisms.query_residual_input import coordinate_support
from grain_research.vector_mechanisms.reader_ensemble import (
    ensemble_field,
    unpack_models,
)
from scripts.analyze_grain_position_reader import action_metrics
from scripts.run_grain_reader_new_author_transport import (
    ARMS,
    QUERIES,
    packet_arrays,
    verify_prediction_freeze,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write


def cluster_interval(frame, repeats=2000, seed=3701):
    """Resample authors, retaining all four actions; report descriptive uncertainty."""
    y = frame.pivot(index="index", columns="factor", values="actual").to_numpy()
    p = frame.pivot(index="index", columns="factor", values="predicted").to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(y), size=(repeats, len(y)))
    yy, pp = y[indices].reshape(repeats, -1), p[indices].reshape(repeats, -1)
    yy, pp = yy - yy.mean(1, keepdims=True), pp - pp.mean(1, keepdims=True)
    denom = np.sqrt(np.sum(yy**2, axis=1) * np.sum(pp**2, axis=1))
    vals = np.divide(
        np.sum(yy * pp, axis=1), denom, out=np.full(repeats, np.nan), where=denom > 0
    )
    return {
        "action_r_cluster_low": float(np.nanquantile(vals, 0.025)),
        "action_r_cluster_high": float(np.nanquantile(vals, 0.975)),
        "valid_bootstraps": int(np.isfinite(vals).sum()),
    }


def analyze(out, name):
    """Separate field error, projected action agreement and author-relative reading."""
    freeze = verify_prediction_freeze(out)
    state = read(out / "run_status.json")
    if (
        state["status"] != "COMPLETE"
        or state["calls"] != 54
        or state["backwards"] != 24
    ):
        raise ValueError("Complete predeclared 54-forward/24-backward packet required")
    if not freeze["before_new_gradient_access"]:
        raise ValueError("Predictions were not frozen before evaluation")
    dest = out / name
    dest.mkdir(exist_ok=False)
    models, unions = {}, {}
    for q in QUERIES:
        for arm in ARMS:
            with np.load(
                out / "private_models" / f"{q}__{arm}.npz", allow_pickle=False
            ) as z:
                packed = dict(z)
            models[q, arm] = packed
            v = np.concatenate([m["basis"] for m in unpack_models(packed)])
            _, s, vt = np.linalg.svd(v, full_matrices=False)
            unions[q, arm] = vt[s > 1e-10 * s[0]]
    with np.load(out / "private_predictions.npz", allow_pickle=False) as z:
        predictions = dict(z)
    actions, fields, support, fd, prefixes, packet_hashes = [], [], [], [], [], {}
    for a in read(out / "private_design.json"):
        i = a["index"]
        with np.load(out / "private_inputs" / f"a{i:02d}.npz", allow_pickle=False) as z:
            h, g = z["hidden"], z["grain"]
        directions = np.stack([direction(h, g, k)[0] for k in range(4)])
        for q in QUERIES:
            path = out / "private_packets" / f"a{i:02d}_{q}.json"
            p, arrays = packet_arrays(path)
            for file in (path, path.with_suffix(".npz")):
                packet_hashes[str(file.relative_to(out))] = sha(file)
            j = arrays["body_gradient"].astype(float)
            if j.shape != h.shape:
                raise ValueError("Held body gradient layout changed")
            y = np.einsum("kth,th->k", directions, j)
            energy = float(np.sum(j**2))
            prefixes.append(
                {
                    "query": q,
                    "index": i,
                    "prefix_max": p["prefix_max"],
                    "prefix_rms": p["prefix_rms"],
                    "score_replay_error": p["score_replay_error"],
                }
            )
            if p["finite_difference"] is not None:
                fd.append({"query": q, **p["finite_difference"]})
            for arm in ARMS:
                packed = models[q, arm]
                coef = predictions[f"{q}__a{i}__{arm}"]
                pred = ensemble_field(coef, packed)
                py = np.einsum("kth,th->k", directions, pred)
                v = unions[q, arm]
                oracle = (j @ v.T) @ v
                missing, excess = oracle - j, pred - oracle
                total = float(np.sum((pred - j) ** 2))
                check = total - float(np.sum(missing**2) + np.sum(excess**2))
                if abs(check) > 1e-9 * max(energy, 1e-15):
                    raise ValueError("Union-subspace orthogonal error identity failed")
                meta = {"query": q, "arm": arm, "index": i}
                fields.append(
                    meta
                    | {
                        "tokens": len(j),
                        "union_rank": len(v),
                        "field_actual_energy": energy,
                        "field_error_energy": total,
                        "field_error_ratio": total / energy,
                        "field_union_missing_ratio": float(np.sum(missing**2) / energy),
                        "field_union_reading_ratio": float(np.sum(excess**2) / energy),
                        "field_identity_error": check / energy,
                    }
                )
                for k in range(4):
                    actions.append(
                        meta
                        | {
                            "factor": f"whole:F{k + 1:02d}",
                            "actual": y[k],
                            "predicted": py[k],
                        }
                    )
                offset = 0
                for s, m in enumerate(unpack_models(packed)):
                    rank = len(m["basis"])
                    target = len(j) * j @ m["basis"].T
                    denom = float(np.sum(target**2))
                    support.append(
                        meta
                        | {
                            "component": s,
                            **coordinate_support(h, m),
                            "component_coefficient_error_ratio": float(
                                np.sum((coef[:, offset : offset + rank] - target) ** 2)
                                / denom
                            )
                            if denom
                            else float("nan"),
                        }
                    )
                    offset += rank
    a, f, s = pd.DataFrame(actions), pd.DataFrame(fields), pd.DataFrame(support)
    if (
        len(a) != 384
        or len(f) != 96
        or len(s) != 384
        or a.duplicated(["query", "arm", "index", "factor"]).any()
    ):
        raise ValueError("Incomplete eight-author comparison")
    a.to_csv(dest / "private_actions.csv", index=False)
    f.to_csv(dest / "private_fields.csv", index=False)
    s.to_csv(dest / "private_support.csv", index=False)
    pd.DataFrame(prefixes).to_csv(dest / "private_prefix_checks.csv", index=False)
    pd.DataFrame(fd).to_csv(dest / "finite_difference.csv", index=False)
    rows, factors, authors, supports = [], [], [], []
    for (q, arm), group in a.groupby(["query", "arm"], sort=True):
        meta = {"query": q, "arm": arm}
        fg = f[(f["query"] == q) & (f.arm == arm)]
        sg = s[(s["query"] == q) & (s.arm == arm)]
        centered = group[["actual", "predicted"]] - group.groupby("factor")[
            ["actual", "predicted"]
        ].transform("mean")
        rows.append(
            meta
            | {
                "authors": 8,
                **action_metrics(group.actual, group.predicted),
                **cluster_interval(group),
                "factor_centered_action_r": action_metrics(
                    centered.actual, centered.predicted
                )["action_r"],
                "field_error_mean": float(fg.field_error_ratio.mean()),
                "field_error_pooled": float(
                    fg.field_error_energy.sum() / fg.field_actual_energy.sum()
                ),
                "field_union_missing_mean": float(fg.field_union_missing_ratio.mean()),
                "field_union_reading_mean": float(fg.field_union_reading_ratio.mean()),
                "component_coefficient_error_mean": float(
                    sg.component_coefficient_error_ratio.mean()
                ),
            }
        )
        for factor, gg in group.groupby("factor", sort=True):
            factors.append(
                meta | {"factor": factor, **action_metrics(gg.actual, gg.predicted)}
            )
        for i, gg in group.groupby("index", sort=True):
            authors.append(
                meta | {"index": i, **action_metrics(gg.actual, gg.predicted)}
            )
        supports.append(
            meta
            | {
                "coordinate_abs_max": float(sg.held_coordinate_abs_max.max()),
                "outside_fraction_mean": float(
                    sg.held_coordinate_outside_fraction.mean()
                ),
                "outside_fraction_max": float(
                    sg.held_coordinate_outside_fraction.max()
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(dest / "reader_comparison.csv", index=False)
    pd.DataFrame(factors).to_csv(dest / "factor_actions.csv", index=False)
    pd.DataFrame(authors).to_csv(dest / "private_author_actions.csv", index=False)
    pd.DataFrame(supports).to_csv(dest / "coordinate_support.csv", index=False)
    summary = {
        "authors": 8,
        "queries": 3,
        "arms": 4,
        "unique_native_actions": 96,
        "paired_prediction_actions": 384,
        "new_fits": 0,
        "human_labels_read": False,
        "native_calls": state["calls"],
        "native_backwards": state["backwards"],
        "native_seconds": state["seconds"],
        "score_replay_error_max": max(p["score_replay_error"] for p in prefixes),
        "prefix_max": max(p["prefix_max"] for p in prefixes),
        "finite_delta_error_max": max(p["finite_delta_error"] for p in fd),
        "field_identity_error_max": float(f.field_identity_error.abs().max()),
        "runtime": freeze["runtime"],
        "files": {p.name: sha(p) for p in sorted(dest.glob("*.csv"))},
    }
    write(
        dest / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "native_packet_hashes": packet_hashes,
            "bootstrap_seed": 3701,
            "bootstrap_repeats": 2000,
            "new_neural_calls": 0,
            "new_fits": 0,
        },
    )
    write(dest / "summary.json", summary)
    print(
        comparison[
            [
                "query",
                "arm",
                "action_r",
                "action_mse_over_zero",
                "factor_centered_action_r",
            ]
        ].to_string(index=False)
    )
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="logic_reader_new_author_transport_v1")
    p.add_argument("--analysis", default="analysis")
    args = p.parse_args()
    with threadpool_limits(limits=4):
        analyze(BASE / args.name, args.analysis)
