#!/usr/bin/env python3
"""NEXT16: cached-state kernel action construction, then separate source linkage."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import association
from grain_research.vector_mechanisms.factor_path_routes import paired_delta
from grain_research.vector_mechanisms.kernel_action import (
    ARMS,
    contrast_features,
    fit_action,
    predict_action,
    training_geometry,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_teacher_targets import interval, verify_digest

FIELD = BASE / "centered_adjoint_field_v1"
PARITY = BASE / "configuration_parity_v1_r2"
STATES = BASE / "source_anchor_diagnosis_v1/private_packets"
PLAN = ROOT / "docs/GRAIN_LOGIC_KERNEL_ACTION_PLAN_V1.md"
KEYS = ["author_id", "query"]


def project(out):
    """Project every cached body token once; no tokenizer, network, Y or neural call."""
    if (out / "input_freeze.json").exists():
        print("Projection already frozen; use a new directory to replay")
        return
    started = time.monotonic()
    design_path = FIELD / "private_design.json"
    field, parity = read(FIELD / "manifest.json"), read(PARITY / "manifest.json")
    verify_digest(design_path, field["design_sha256"])
    verify_digest(design_path, parity["design_sha256"])
    design = read(design_path)
    endpoint_hashes = read(BASE / "odd_action_construction_v1/manifest.json")["inputs"]
    if len(design) != 48:
        raise ValueError("Expected complete48-author design")
    if not out.exists():
        out.mkdir(parents=True)
        (out / "private_projected").mkdir()
        write(
            out / "manifest.json",
            {
                "plan_sha256": sha(PLAN),
                "design_sha256": sha(design_path),
                "script_sha256": sha(__file__),
                "core_sha256": sha(
                    ROOT / "grain_research/vector_mechanisms/kernel_action.py"
                ),
                "source_field_manifest_sha256": sha(FIELD / "manifest.json"),
                "directions": 32,
                "seed": 20260910,
                "max_seconds": 600,
                "new_neural_calls": 0,
                "human_labels_read": False,
            },
        )
    manifest = read(out / "manifest.json")
    for path, key in [
        (PLAN, "plan_sha256"),
        (design_path, "design_sha256"),
        (Path(__file__), "script_sha256"),
    ]:
        verify_digest(path, manifest[key])
    records, packet_hashes, source_hashes, matrices = [], {}, {}, {}
    completed = 0
    for qi, query in enumerate(sorted({r["query"] for r in design})):
        entries = sorted(
            [r for r in design if r["query"] == query], key=lambda r: r["author_id"]
        )
        if len(entries) != 24:
            raise ValueError("Expected24 authors/query")
        rng = np.random.default_rng(20260910 + qi)
        fold_order = rng.permutation(24)
        folds = {entries[int(j)]["author_id"]: i % 4 for i, j in enumerate(fold_order)}
        # Independent random directions, unchanged by any data or target value.
        feature_rng = np.random.default_rng(20260910 + qi)
        w = feature_rng.normal(size=(1536, 32)) / np.sqrt(1536)
        eta = feature_rng.normal(size=32)
        matrices[query + "__W"], matrices[query + "__eta"] = w, eta
        for row in entries:
            index = row["index"]
            stem = f"author_{index:03d}"
            state_path = STATES / (stem + ".npz")
            target_path = PARITY / "private_packets" / (stem + ".json")
            cell_path = (
                BASE
                / "configuration_query_factorial_v1/private_packets"
                / (stem + ".json")
            )
            for path, hashes in [
                (state_path, field["sources"]),
                (cell_path, field["sources"]),
                (target_path, endpoint_hashes),
            ]:
                expected = hashes[str(path.relative_to(ROOT))]
                verify_digest(path, expected)
                source_hashes[str(path.relative_to(ROOT))] = expected
            target, cells = read(target_path), read(cell_path)
            if (target["author_id"], target["query"]) != (
                row["author_id"],
                query,
            ) or cells["author_id"] != row["author_id"]:
                raise ValueError("Source identities changed")
            dst = out / "private_projected" / (stem + ".npz")
            meta = dst.with_suffix(".json")
            if dst.exists():
                verify_digest(dst, read(meta)["sha256"])
            else:
                if time.monotonic() - started > manifest["max_seconds"]:
                    write(
                        out / "run_status.json",
                        {
                            "status": "PAUSED_RESOURCE_BOUNDARY",
                            "authors": completed,
                            "seconds": time.monotonic() - started,
                        },
                    )
                    return
                values = {}
                with np.load(state_path, allow_pickle=False) as cache:
                    for v in row["views"]:
                        i = v["view"]
                        h = cache[f"view_{i}"]
                        groups = [np.asarray(g, int) for g in v["groups"]]
                        body = np.concatenate(groups)
                        if (
                            len(set(body)) != len(body)
                            or not np.array_equal(body, np.sort(body))
                            or np.asarray(v["p_mask"], bool)[body].any()
                        ):
                            raise ValueError("Ordered disjoint body groups required")
                        z = h[body].astype(float) @ w
                        c = np.empty_like(z)
                        offset = 0
                        for g in groups:
                            c[offset : offset + len(g)] = z[
                                offset : offset + len(g)
                            ].mean(0)
                            offset += len(g)
                        d = z - c
                        pos = -1 + 2 * body / max(len(h) - 1, 1)
                        values.update({f"c_{i}": c, f"d_{i}": d, f"position_{i}": pos})
                np.savez_compressed(dst, **values)
                write(
                    meta,
                    {
                        "sha256": sha(dst),
                        "source_state_sha256": source_hashes[
                            str(state_path.relative_to(ROOT))
                        ],
                    },
                )
            packet_hashes[str(dst.relative_to(out))] = sha(dst)
            for v, cell in zip(target["views"], cells["views"]):
                if (
                    v["view"] != cell["view"]
                    or max(
                        abs(v["f_positive"] - cell["f10"]),
                        abs(v["f_origin"] - cell["f00"]),
                    )
                    > 1e-12
                ):
                    raise ValueError(
                        "Positive action changed relative to NEXT15 four cells"
                    )
                if (
                    abs(v["plus"] - v["odd"] - v["even"]) > 1e-12
                    or abs(v["minus"] + v["odd"] - v["even"]) > 1e-12
                ):
                    raise ValueError("Cached parity identity failed")
                records.append(
                    {
                        "author_id": str(row["author_id"]),
                        "query": query,
                        "index": index,
                        "view": v["view"],
                        "fold": folds[row["author_id"]],
                        "odd": v["odd"],
                        "even": v["even"],
                        "plus": v["plus"],
                        "minus": v["minus"],
                    }
                )
            completed += 1
            if completed % 12 == 0:
                print(
                    {
                        "stage": "PROJECT",
                        "authors": completed,
                        "seconds": round(time.monotonic() - started, 2),
                    },
                    flush=True,
                )
    np.savez_compressed(out / "private_directions.npz", **matrices)
    pd.DataFrame(records).to_csv(out / "private_targets.csv", index=False)
    write(
        out / "input_freeze.json",
        {
            "packet_sha256": packet_hashes,
            "sources": source_hashes,
            "target_sha256": sha(out / "private_targets.csv"),
            "directions_sha256": sha(out / "private_directions.npz"),
            "human_labels_read": False,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PROJECTED",
            "authors": completed,
            "seconds": time.monotonic() - started,
        },
    )


def construct(out):
    """Author-OOF construction uses only endpoint targets and training-fold geometry."""
    if (out / "prediction_freeze.json").exists():
        raise ValueError("Preserve frozen predictions")
    started = time.monotonic()
    freeze = read(out / "input_freeze.json")
    verify_digest(out / "private_targets.csv", freeze["target_sha256"])
    verify_digest(out / "private_directions.npz", freeze["directions_sha256"])
    data = pd.read_csv(
        out / "private_targets.csv", dtype={"author_id": str}
    ).sort_values(KEYS + ["view"])
    if (
        data.duplicated(KEYS + ["view"]).any()
        or data.groupby(KEYS).size().ne(3).any()
        or data.groupby(KEYS).fold.nunique().ne(1).any()
    ):
        raise ValueError("Complete author-disjoint views required")
    packets = {}
    for p, digest in freeze["packet_sha256"].items():
        verify_digest(out / p, digest)
        with np.load(out / p, allow_pickle=False) as a:
            packets[int(Path(p).stem.split("_")[-1])] = {k: a[k] for k in a.files}
    directions = np.load(out / "private_directions.npz", allow_pickle=False)
    predictions, models, diagnostics = [], [], []
    for query, group in data.groupby("query", sort=True):
        rows = list(group.to_dict("records"))
        for fold in range(4):
            train = np.array([r["fold"] != fold for r in rows])
            if (
                len({r["author_id"] for r in rows if r["fold"] != fold}) != 18
                or len({r["author_id"] for r in rows if r["fold"] == fold}) != 6
            ):
                raise ValueError("Expected18 training and6 test authors")

            def packet(r):
                a = packets[r["index"]]
                i = r["view"]
                return a[f"c_{i}"], a[f"d_{i}"], a[f"position_{i}"]

            means = [packet(r)[0].mean(0) for r in rows]
            variances = [(packet(r)[1] ** 2).mean(0) for r in rows]
            center, scale = training_geometry(means, variances, train)
            for arm in ARMS:
                phi = [
                    contrast_features(
                        *packet(r), directions[query + "__eta"], center, scale, arm
                    )
                    for r in rows
                ]
                odd, even = (np.array([v[k] for v in phi]) for k in (0, 1))
                outputs = {}
                for effect, x in [("odd", odd), ("even", even)]:
                    target = np.array([r[effect] for r in rows])
                    model = fit_action(x[train], target[train])
                    outputs[effect] = predict_action(x[~train], model)
                    models.append(
                        {
                            "query": query,
                            "fold": fold,
                            "arm": arm,
                            "effect": effect,
                            "center": center.tolist(),
                            "scale": scale.tolist(),
                            "feature_scale": model["scale"].tolist(),
                            "coef": model["coef"].tolist(),
                            "penalty": model["penalty"],
                        }
                    )
                for k, r in enumerate([r for r in rows if r["fold"] == fold]):
                    predictions.append(
                        {
                            **r,
                            "arm": arm,
                            "pred_odd": outputs["odd"][k],
                            "pred_even": outputs["even"][k],
                            "pred_plus": outputs["odd"][k] + outputs["even"][k],
                            "pred_minus": -outputs["odd"][k] + outputs["even"][k],
                        }
                    )
                # One non-fitted check per fold/arm; all scoring uses the same formula.
                c, d, pos = packet(rows[0])
                po, pe = contrast_features(
                    c, -d, pos, directions[query + "__eta"], center, scale, arm
                )
                zo, ze = contrast_features(
                    c,
                    np.zeros_like(d),
                    pos,
                    directions[query + "__eta"],
                    center,
                    scale,
                    arm,
                )
                diagnostics.append(
                    {
                        "query": query,
                        "fold": fold,
                        "arm": arm,
                        "parity_error": float(
                            max(np.max(abs(po + odd[0])), np.max(abs(pe - even[0])))
                        ),
                        "zero_error": float(max(np.max(abs(zo)), np.max(abs(ze)))),
                    }
                )
    directions.close()
    predictions = pd.DataFrame(predictions)
    if (
        len(predictions) != 576
        or len(models) != 64
        or max(v["parity_error"] + v["zero_error"] for v in diagnostics) > 1e-12
    ):
        raise ValueError("Prediction coverage/parity/zero check failed")
    predictions.to_csv(out / "private_predictions.csv", index=False)
    write(out / "private_models.json", {"models": models})
    pd.DataFrame(diagnostics).to_csv(out / "functional_checks.csv", index=False)
    write(
        out / "prediction_freeze.json",
        {
            "human_labels_read": False,
            "predictions_sha256": sha(out / "private_predictions.csv"),
            "models_sha256": sha(out / "private_models.json"),
            "input_freeze_sha256": sha(out / "input_freeze.json"),
            "model_fits": len(models),
            "new_neural_calls": 0,
            "seconds": time.monotonic() - started,
            "script_sha256": sha(__file__),
            "core_sha256": sha(
                ROOT / "grain_research/vector_mechanisms/kernel_action.py"
            ),
        },
    )
    print(
        {
            "status": "PREDICTIONS_FROZEN",
            "fits": len(models),
            "seconds": round(time.monotonic() - started, 2),
        },
        flush=True,
    )


def analyze(out):
    """Measure frozen action fidelity before separately linking exposed source Y."""
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed analysis")
    freeze = read(out / "prediction_freeze.json")
    verify_digest(out / "private_predictions.csv", freeze["predictions_sha256"])
    verify_digest(out / "private_models.json", freeze["models_sha256"])
    data = pd.read_csv(out / "private_predictions.csv", dtype={"author_id": str})
    values = [
        "odd",
        "even",
        "plus",
        "minus",
        "pred_odd",
        "pred_even",
        "pred_plus",
        "pred_minus",
    ]
    authors = data.groupby(KEYS + ["arm", "fold"])[values].mean().reset_index()
    rows, comparisons = [], []
    for unit, frame in [("author", authors), ("material", data)]:
        for (q, arm), g in frame.groupby(["query", "arm"], sort=True):
            for effect in ["odd", "even", "plus"]:
                y, p = g[effect].to_numpy(), g["pred_" + effect].to_numpy()
                mse = float(np.mean((y - p) ** 2))
                rows.append(
                    {
                        "query": q,
                        "arm": arm,
                        "unit": unit,
                        "effect": effect,
                        "n": len(y),
                        "r": association(p, y)["pearson_r"],
                        "mse": mse,
                        "mse_over_zero": mse / float(np.mean(y * y)),
                        "prediction_sd": float(p.std()),
                        "target_sd": float(y.std()),
                    }
                )
    contrasts = [
        ("fourier_bag", "taylor3_bag"),
        ("fourier_position", "taylor3_position"),
        ("fourier_position", "fourier_bag"),
    ]
    for qi, (query, group) in enumerate(authors.groupby("query", sort=True)):
        ix = np.random.default_rng(20260910 + qi).integers(0, 24, (1999, 24))
        for candidate, reference in contrasts:
            a = group[group.arm == candidate].set_index("author_id").sort_index()
            b = group[group.arm == reference].set_index("author_id").loc[a.index]
            for effect in ["odd", "even", "plus"]:
                y, p, r = (
                    v.to_numpy()
                    for v in (a[effect], a["pred_" + effect], b["pred_" + effect])
                )
                gain = (y - r) ** 2 - (y - p) ** 2
                comparisons.append(
                    {
                        "query": query,
                        "candidate": candidate,
                        "reference": reference,
                        "effect": effect,
                        **paired_delta(y, p, r, draws=1999),
                        "mse_gain": float(gain.mean()),
                        **{
                            "mse_gain_" + k: v
                            for k, v in interval(gain[ix].mean(1)).items()
                        },
                    }
                )
    pd.DataFrame(rows).to_csv(out / "action_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    path = BASE / "all_factor_source_m29_v1/private_external_links.csv"
    verify_digest(
        path, read(BASE / "logic_reading_levels_v1/manifest.json")["criterion_sha256"]
    )
    y = pd.read_csv(path, usecols=KEYS + ["y_true"], dtype={"author_id": str})
    linked = authors.merge(y, on=KEYS, how="left", validate="many_to_one")
    if not np.isfinite(linked.y_true).all():
        raise ValueError("Incomplete exposed-source criteria")
    linked.to_csv(out / "private_criterion_crosswalk.csv", index=False)
    links = []
    for (query, arm), g in linked.groupby(["query", "arm"], sort=True):
        for effect in ["odd", "even", "plus"]:
            links.append(
                {
                    "query": query,
                    "arm": arm,
                    "effect": effect,
                    "n": len(g),
                    "actual_criterion_r": association(g[effect], g.y_true)["pearson_r"],
                    "predicted_criterion_r": association(g["pred_" + effect], g.y_true)[
                        "pearson_r"
                    ],
                }
            )
    pd.DataFrame(links).to_csv(out / "criterion_links.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "authors": 48,
            "arms": len(ARMS),
            "new_neural_calls": 0,
            "criterion_sha256": sha(path),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "human_criterion_fits": 0,
            "analysis_script_sha256": sha(__file__),
        },
    )
    print(pd.DataFrame(rows).query("unit=='author'").to_string(index=False))


def main():
    """Private resumable projection, frozen OOF construction, then analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("project", "construct", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private ignored destination required")
    with threadpool_limits(limits=4):
        {"project": project, "construct": construct, "analyze": analyze}[args.phase](
            out
        )


if __name__ == "__main__":
    main()
