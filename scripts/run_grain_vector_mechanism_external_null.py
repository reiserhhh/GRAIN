#!/usr/bin/env python3
"""Bounded 20/199 author-label permutation null with exact nested OOF reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.external_heads import task_score
from grain_research.vector_mechanisms.external_null import (
    nested_predict_cached,
    prepare_nested_folds,
)
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_features import bundle_arms
from scripts.run_grain_vector_mechanism_external_links import dense_features
from scripts.summarize_grain_vector_external_links import load_oof


def main():
    """Reuse only unsupervised fold transforms; refit all supervised choices."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--vector-dir", type=Path)
    p.add_argument("--feature-cache-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--disk-fold-cache",
        action="store_true",
        help="Lossless one-outer-fold RAM plans; unchanged nested estimator",
    )
    a = p.parse_args()
    if not 60 <= a.max_seconds <= 28800:
        raise ValueError("Invalid null execution budget")
    start = time.monotonic()
    verified = load_oof(a.parent_run)
    m = verified["manifest"]
    if (
        m.get("thread_control", "retain") != "retain"
        or m["permutation_seed"] is not None
    ):
        raise ValueError(
            "Initial unconditional null needs a retained, unpermuted source parent"
        )
    source = json.loads(a.views.read_text())
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Changed common material")
    bundles = [b for b in source if b["source_group"] == m["source_group"]]
    if m.get("exclude_reference"):
        bundles = exclude_reference(
            bundles,
            pd.read_csv(
                a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}
            ),
        )
    if material_hash(bundles) != m["source_material_sha256"]:
        raise ValueError("Different source author set")
    ids = np.asarray([b["author_id"] for b in bundles])
    old = {u: i for i, u in enumerate(verified["ids"])}
    order = [old[u] for u in ids]
    y, expected = verified["y"][order], verified["p"][order]
    folds = np.asarray([b["official_folds"][m["task"]] for b in bundles])
    if not np.array_equal(folds, verified["folds"][order]):
        raise ValueError("Different official folds")
    feature_kind = "word_char" if m["feature"] == "word_char" else "dense"
    if feature_kind == "word_char":
        x = np.asarray(["\n\n".join(bundle_arms(b)[m["arm"]]) for b in bundles])
    else:
        opts = SimpleNamespace(
            feature=m["feature"],
            arm=m["arm"],
            preflight_dir=a.preflight_dir,
            vector_dir=a.vector_dir,
            feature_cache_dir=a.feature_cache_dir,
        )
        x = dense_features(bundles, source, common, opts)
    input_sha = (
        material_hash(x.tolist())
        if feature_kind == "word_char"
        else hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    )
    if (
        m.get("feature_input_sha256") is not None
        and input_sha != m["feature_input_sha256"]
    ):
        raise ValueError("Null input features differ from the frozen source parent")
    fields = {
        "protocol": "VM05_AUTHOR_PERMUTATION_199_V1",
        "parent_identity": m["identity"],
        "source_material_sha256": m["source_material_sha256"],
        "task": m["task"],
        "arm": m["arm"],
        "feature": m["feature"],
        "feature_input_sha256": input_sha,
        "input_commitment_in_parent": "feature_input_sha256" in m,
        "legacy_input_verification": "canonical material/operator plus exact observed OOF replay",
        "pca_applied": feature_kind == "dense" and m["pca"],
        "exclude_reference": m.get("exclude_reference", False),
        "draws": 199,
        "seed_sequence": "731+j, j=0..198",
        "first_gate": 20,
        "cached_steps": "label-free inner/outer vocabulary, scaler, PCA only",
        "refitted_steps": "every candidate head and inner selection at every permutation",
        "new_source_label_values_read": False,
        "bridge_labels_read": False,
        "observed_oof_sha256": hashlib.sha256(
            (a.parent_run / "private_oof_predictions.csv").read_bytes()
        ).hexdigest(),
        "scoring_code": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "grain_research/vector_mechanisms/external_null.py",
                "grain_research/vector_mechanisms/external_heads.py",
                "grain_research/vector_mechanisms/external_controls.py",
                "grain_research/vector_mechanisms/external_material.py",
                "scripts/run_grain_vector_mechanism_external_links.py",
                "scripts/run_grain_vector_mechanism_external_features.py",
                "suica_core/v7_psychometric.py",
                "scripts/run_grain_vector_mechanism_external_null.py",
            )
        },
    }
    fields["identity"] = material_hash(fields)
    if a.disk_fold_cache:
        fields.pop("identity")
        fields["fold_cache_storage"] = "lossless_disk_one_outer_v1"
        fields["disk_cache_code_sha256"] = hashlib.sha256(
            (
                ROOT / "grain_research/vector_mechanisms/external_null_disk.py"
            ).read_bytes()
        ).hexdigest()
        fields["identity"] = material_hash(fields)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "manifest.json"
    if mp.exists():
        if not a.resume or json.loads(mp.read_text()) != fields:
            raise ValueError("Existing null requires explicit identical resume")
    elif any(out.iterdir()):
        raise ValueError("Use a new null output directory")
    else:
        write_json(mp, fields)
        archive_source(ROOT, out / "source_snapshot.zip")
    (out / "private_cache").mkdir(exist_ok=True)
    records_path = out / "permutation_records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    sanity_path = out / "sanity20.json"
    if sanity_path.exists() and json.loads(sanity_path.read_text()).get(
        "operational_positive_null_alert"
    ):
        raise ValueError("Positive null alert requires review, not an automatic resume")
    for j, r in enumerate(records):
        path = out / "private_cache" / f"draw_{j:03d}.npz"
        if (
            r["draw"] != j
            or hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]
        ):
            raise ValueError("Invalid partial null checkpoint")
        with np.load(path, allow_pickle=False) as z:
            ix = np.random.default_rng(731 + j).permutation(len(y))
            if not np.array_equal(ix, z["permutation"]) or not np.isclose(
                task_score(y[ix], z["prediction"], m["task"]), r["score"], atol=1e-12
            ):
                raise ValueError("Null checkpoint score/permutation does not replay")
    write_json(
        out / "run_status.json",
        {"status": "PREPARING_FOLDS", "completed": len(records), "target": 199},
    )
    with threadpool_limits(limits=4):
        if a.disk_fold_cache:
            from grain_research.vector_mechanisms.external_null_disk import (
                prepare_disk_folds,
            )

            plans = prepare_disk_folds(
                out / "private_fold_cache",
                x,
                ids,
                folds,
                feature_kind,
                m["task"],
                parent_identity=fields["identity"],
                pca=m["pca"],
            )
        else:
            plans = prepare_nested_folds(
                x, ids, folds, feature_kind, m["task"], pca=m["pca"]
            )

        def validate_cache(stage):
            """Seal cache integrity at every return boundary, not every draw."""
            if a.disk_fold_cache:
                plans.validate_files()
                write_json(
                    out / "fold_cache_audit.json",
                    {
                        "stage": stage,
                        "identity": plans.manifest["identity"],
                        "arrays": len(plans.manifest["arrays"]),
                        "bytes": plans.manifest["bytes"],
                        "sha256_verified": True,
                        "labels_used": False,
                    },
                )

        observed, parameters = nested_predict_cached(plans, y, m["task"])
        max_error = float(np.max(abs(observed - expected)))
        if max_error > 1e-10:
            raise ValueError(
                "Cached transform implementation differs from canonical source OOF"
            )
        write_json(
            out / "observed_replay.json",
            {"max_absolute_error": max_error, "parameters": parameters},
        )
        for j in range(len(records), 199):
            if time.monotonic() - start > a.max_seconds:
                validate_cache("PAUSED_BUDGET")
                write_json(
                    out / "run_status.json",
                    {
                        "status": "PAUSED_BUDGET",
                        "completed": len(records),
                        "target": 199,
                    },
                )
                return
            ix = np.random.default_rng(731 + j).permutation(len(y))
            prediction, selected = nested_predict_cached(plans, y[ix], m["task"])
            path = out / "private_cache" / f"draw_{j:03d}.npz"
            if path.exists():
                raise ValueError("Unindexed partial draw exists; preserve and inspect")
            np.savez_compressed(path, prediction=prediction, permutation=ix)
            records.append(
                {
                    "draw": j,
                    "seed": 731 + j,
                    "score": task_score(y[ix], prediction, m["task"]),
                    "parameters": selected,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            write_json(records_path, records)
            status = {
                "status": "RUNNING",
                "completed": len(records),
                "target": 199,
                "seconds": time.monotonic() - start,
                "bridge_labels_read": False,
            }
            write_json(out / "run_status.json", status)
            if j == 19:
                first_mean = float(np.mean([r["score"] for r in records]))
                alert = first_mean > (0.10 if m["task"] == "big5" else 0.55)
                write_json(
                    out / "sanity20.json",
                    {
                        "completed": 20,
                        "mean": first_mean,
                        "operational_positive_null_alert": alert,
                        "not_a_personality_success_gate": True,
                    },
                )
                if alert:
                    validate_cache("SANITY_REVIEW_REQUIRED")
                    write_json(
                        out / "run_status.json",
                        status | {"status": "SANITY_REVIEW_REQUIRED"},
                    )
                    return
            if j % 20 == 19:
                print(json.dumps(status), flush=True)
        validate_cache("COMPLETE")
    values = np.asarray([r["score"] for r in records])
    score = task_score(y, observed, m["task"])
    summary = {
        "status": "COMPLETE",
        "authors": len(y),
        "draws": 199,
        "observed": score,
        "null_mean": float(values.mean()),
        "null_sd": float(values.std(ddof=1)),
        "null_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "p_upper": float((1 + np.sum(values >= score)) / 200),
        "min_resolvable_p": 0.005,
        "observed_replay_error": max_error,
        "seconds": time.monotonic() - start,
        "parent_identity": fields["identity"],
        "scope": "Unconditional author-label association; not conditional/cross-thread causality",
        "bridge_labels_read": False,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "run_status.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
