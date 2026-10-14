#!/usr/bin/env python3
"""Compare raw-native and learned text readers on an identical available cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.native_construction import (
    ALPHAS,
    ARMS,
    fit_selected,
    fixed_splits,
    paired_interval,
)
from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.analyze_grain_author_discovery import read_native
from scripts.analyze_grain_direct_source_scores import direct_prediction
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_local_construction import correlation_row

INVENTORY = BASE / "all_plan_closure_v1/native_construction_inventory_r1"
REFERENCE = BASE / "all_plan_closure_v1/local_bilinear_r2/pandora"
QUEUE = BASE / "d1_direct_snapshot_47"
PLAN = ROOT / "docs/GRAIN_NATIVE_CONSTRUCTION_COMPARISON_PLAN_V1.md"
CONTRASTS = (
    ("semantic_native", "semantic"),
    ("additive_native", "additive"),
    ("native_calibrated", "native_raw"),
    ("semantic", "native_raw"),
    ("additive", "semantic"),
)


def prepare(out):
    """Freeze outcome-independent membership and verify existing material hashes."""
    out.mkdir(parents=True, exist_ok=False)
    material = json.loads((DEFAULT / "material_manifest.json").read_text())
    inputs = {}
    for name, sha in material["files"].items():
        p = DEFAULT / name
        if digest(p) != sha:
            raise ValueError("Changed original construction material")
        inputs[str(p)] = sha
    reference = json.loads((REFERENCE / "material.json").read_text())
    for name, sha in reference["files"].items():
        p = REFERENCE / name
        if digest(p) != sha:
            raise ValueError("Changed frozen representation blocks")
        inputs[str(p)] = sha
    inv = json.loads((INVENTORY / "manifest.json").read_text())
    if inv["material_manifest"] != digest(DEFAULT / "material_manifest.json"):
        raise ValueError("Native inventory used different construction material")
    records, native_hashes, queue_manifest, status = read_native(QUEUE)
    if (
        native_hashes != inv["native_hashes"]
        or inv["queue_manifest"] != digest(QUEUE / "manifest.json")
        or inv["queue_identity"] != queue_manifest["identity"]
    ):
        raise ValueError("Native snapshot changed after token-identity verification")
    source = json.loads((DEFAULT / "private_material.json").read_text())
    authors = json.loads((REFERENCE / "private_authors.json").read_text())
    ids = [b["author_id"] for b in source["bundles"]]
    hashes = [
        hashlib.sha256(
            "\n\n".join(c["body"] for v in b["views"] for c in v).encode()
        ).hexdigest()
        for b in source["bundles"]
    ]
    if (
        len(set(ids)) != len(ids)
        or ids != authors["ids"]
        or hashes != authors["document_hashes"]
    ):
        raise ValueError("Author or original-document mismatch")
    if any(b["source_group"] != "big5_only" for b in source["bundles"]):
        raise ValueError("Only Big5-only construction roles are permitted")
    available = np.asarray([i for i, uid in enumerate(ids) if uid in records])
    chosen_ids = [ids[i] for i in available]
    native, diagnostics = direct_prediction(records, chosen_ids, "big5", "raw")
    original_folds = np.asarray(
        [b["official_folds"]["big5"] for b in source["bundles"]]
    )
    folds = original_folds[available]
    split = fixed_splits(folds)
    if (
        len(ids) != 624
        or len(available) != 213
        or [len(split[k]) for k in ("train", "valid", "held")] != [126, 44, 43]
    ):
        raise ValueError("Availability population changed; freeze a different protocol")
    coverage = pd.read_csv(INVENTORY / "private_coverage.csv")
    np.testing.assert_array_equal(coverage.author_id.to_numpy(str), ids)
    np.testing.assert_array_equal(np.flatnonzero(coverage.native_available), available)
    np.testing.assert_array_equal(coverage.fold, original_folds)
    views = np.zeros((len(available), 3, 5))
    for u, uid in enumerate(chosen_ids):
        for j, trait in enumerate(TRAITS):
            for row in records[uid]["readouts"]:
                if row["arm"] == "raw" and row["task"] == trait.lower():
                    views[u, row["view"], j] = row["conditional_score"]
    np.testing.assert_allclose(views.mean(1), native, atol=1e-14)
    for p in (
        DEFAULT / "material_manifest.json",
        REFERENCE / "material.json",
        INVENTORY / "manifest.json",
        INVENTORY / "summary.json",
        INVENTORY / "private_coverage.csv",
        INVENTORY / "private_native_readouts.csv",
    ):
        inputs[str(p)] = digest(p)
    contract = {
        "protocol": "GRAIN_NATIVE_CONSTRUCTION_AVAILABLE_V1",
        "arms": ARMS,
        "alphas": ALPHAS,
        "contrasts": CONTRASTS,
        "counts": {k: len(v) for k, v in split.items()},
        "available": len(available),
        "unavailable": len(ids) - len(available),
        "membership": "all token-verified cached Big5-only authors; no outcome filter",
        "split": "official0 evaluation, official1 source validation, official234 training",
        "evaluation_role": "previously inspected development fold, not fresh lockbox",
        "same_material_scope": "same36 original comments; encoder context/tokenizers differ",
        "new_neural_calls": 0,
        "completed_native_shards": len(status["completed"]),
        "inputs": inputs,
        "code": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "grain_research/native_construction.py",
                PLAN,
            )
        },
    }
    write(out / "contract.json", contract)
    # Labels are loaded only after the membership, arms, folds and contrasts freeze.
    with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["author_ids"], ids)
        np.testing.assert_array_equal(z["folds"], original_folds)
        y = z["y"][available]
    if y.shape != (213, 5) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Expected five finite 0..1 Big5 targets")
    with np.load(REFERENCE / "private_blocks.npz", allow_pickle=False) as z:
        blocks = {k: z[k][available] for k in ("semantic", "grain", "affinity")}
    blocks["native"] = native
    np.savez_compressed(
        out / "private_material.npz",
        **blocks,
        y=y,
        views=views,
        original_indices=available,
        folds=folds,
        **split,
    )
    write(
        out / "private_authors.json",
        {
            "ids": chosen_ids,
            "original_indices": available.tolist(),
            "document_hashes": [hashes[i] for i in available],
        },
    )
    pd.DataFrame(diagnostics).to_csv(
        out / "private_native_diagnostics.csv", index=False
    )
    return blocks, y, views, split, contract


def run(out):
    """Fit the fixed family, freeze its scores, and report paired held outcomes."""
    start = time.monotonic()
    blocks, y, views, split, contract = prepare(out)
    held = split["held"]
    prediction, trial_rows, choice_rows, checks = (
        {"native_raw": blocks["native"][held]},
        [],
        [],
        [],
    )
    intercepts = {"native_raw": np.zeros(5)}
    weight_maps = {"native_raw": {"native": np.eye(5)}}
    for arm in ARMS[1:]:
        obj = fit_selected(blocks, y, split, arm)
        feature, head = obj["features"], obj["head"]
        prediction[arm] = head.predict(feature.transform(blocks, held))
        intercept, weights = feature.affine(head, blocks)
        intercepts[arm], weight_maps[arm] = intercept, weights
        direct = intercept + sum(blocks[k][held] @ w for k, w in weights.items())
        single = np.vstack(
            [head.predict(feature.transform(blocks, [i]))[0] for i in held]
        )
        np.testing.assert_allclose(direct, prediction[arm], atol=1e-11)
        np.testing.assert_allclose(single, prediction[arm], atol=1e-11)
        np.testing.assert_array_equal(feature.fit_indices_, split["source"])
        joblib.dump(obj, out / f"private_model_{arm}.joblib")
        np.savez_compressed(
            out / f"private_affine_{arm}.npz", intercept=intercept, **weights
        )
        trial_rows.extend(obj["trials"])
        choice_rows.append(obj["chosen"])
        checks.append(
            {
                "arm": arm,
                "columns": len(feature.scaler_.mean_),
                "affine_error": float(abs(direct - prediction[arm]).max()),
                "single_batch_error": float(abs(single - prediction[arm]).max()),
            }
        )
        print(json.dumps({"stage": "source_fit", **obj["chosen"]}), flush=True)
    # Every candidate prediction is immutable on disk before criterion comparison.
    np.savez_compressed(
        out / "private_predictions.npz", **prediction, held_indices=held
    )
    pd.DataFrame(trial_rows).to_csv(out / "source_tuning.csv", index=False)
    rows, norms, components = [], [], []
    for arm in ARMS:
        p = prediction[arm]
        weights = weight_maps[arm]
        source_scores = intercepts[arm] + sum(
            blocks[k][split["source"]] @ w for k, w in weights.items()
        )
        for j, trait in enumerate(TRAITS):
            error = p[:, j] - y[held, j]
            row = {
                "arm": arm,
                "trait": trait,
                **correlation_row(y[held, j], p[:, j]),
                "mae": float(abs(error).mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "prediction_sd": float(p[:, j].std(ddof=1)),
            }
            rows.append(row)
            norms.append(
                {
                    "arm": arm,
                    "trait": trait,
                    "n_source": len(split["source"]),
                    "score_mean": float(source_scores[:, j].mean()),
                    "score_sd": float(source_scores[:, j].std(ddof=1)),
                    "role": "source score reference; fitted arms have in-source shrinkage, not latent-trait norms",
                }
            )
            yc = y[held, j] - y[held, j].mean()
            denominator = y[held, j].std(ddof=1) * p[:, j].std(ddof=1)
            terms = []
            for key, w in weights.items():
                part = (blocks[key][held] @ w)[:, j]
                covariance = float(yc @ (part - part.mean()) / (len(held) - 1))
                value = covariance / denominator
                components.append(
                    {
                        "arm": arm,
                        "trait": trait,
                        "block": key,
                        "cov_y_component": covariance,
                        "signed_r_contribution": value,
                    }
                )
                terms.append(value)
            np.testing.assert_allclose(sum(terms), row["r"], atol=1e-11)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics_by_trait.csv", index=False)
    means = metrics.groupby("arm", sort=False)[["r", "rho", "mae", "rmse"]].mean()
    means.to_csv(out / "metrics_by_arm.csv")
    pd.DataFrame(norms).to_csv(out / "source_norms.csv", index=False)
    pd.DataFrame(components).to_csv(out / "signed_block_contributions.csv", index=False)
    sensitivity = [
        {
            "trait": trait,
            "authors": len(views),
            "mean_within_author_view_sd": float(
                views[:, :, j].std(axis=1, ddof=1).mean()
            ),
            "source_only": False,
            "scope": "descriptive variation across 3 distinct12-comment views, not temporal reliability",
        }
        for j, trait in enumerate(TRAITS)
    ]
    pd.DataFrame(sensitivity).to_csv(out / "native_view_sensitivity.csv", index=False)
    deltas = [
        {
            "arm": a,
            "reference": b,
            **paired_interval(y[held], prediction[a], prediction[b]),
        }
        for a, b in CONTRASTS
    ]
    pd.DataFrame(deltas).to_csv(out / "paired_deltas.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "protocol": contract["protocol"],
        "counts": contract["counts"],
        "arms": ARMS,
        "mean_metrics": means.reset_index().to_dict("records"),
        "selections": choice_rows,
        "deltas": deltas,
        "numeric_checks": checks,
        "wall_seconds": time.monotonic() - start,
        "neural_calls": 0,
        "labels_used_for_fitting": "170 Big5-only source authors, never43 held",
        "new_final_heads": 5,
        "source_validation_fits": 20,
        "not_closed": [
            "remaining411nativePandora",
            "nativeEssays",
            "fullCV",
            "condition/path production objects",
            "independent interpretation",
        ],
        "files": {
            p.name: digest(p)
            for p in out.iterdir()
            if p.is_file() and not p.name.startswith("._")
        },
    }
    write(out / "summary.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("status", "counts", "mean_metrics", "deltas", "wall_seconds")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "all_plan_closure_v1/native_construction_r1",
    )
    with threadpool_limits(limits=2):
        run(parser.parse_args().output)
