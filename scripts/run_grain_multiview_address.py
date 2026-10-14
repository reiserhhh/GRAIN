#!/usr/bin/env python3
"""Replay frozen full multiview GRAIN, original addresses and disjoint authors."""

from __future__ import annotations

import argparse
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
from grain_research.evidence_contracts import digest
from grain_research.multiview_address import (
    align_units,
    author_rank_auc,
    sentence_coverage,
    token_index,
)
from grain_research.semantic_link import validate_addresses, validate_reference_exposure
from scripts.run_grain_native_reference import E1, GEO, align
from suica_core.v7_geometry import GeometryBundle, score_geometry_bundle
from suica_core.v7_multiview import evaluate_cross_view, transform_feature_block
from suica_core.v7_observations import (
    ObservationSpec,
    build_observations,
    canonicalize_comments,
    prepare_source_panel,
)
from suica_core.v7_operator_controls import source_disjoint_partition
from suica_core.v7_psychometric import author_features_from_embeddings

BASE = ROOT / "results/grain_vector_mechanisms"
PREDECESSOR = BASE / "predecessor_integration_v1"
PLAN = ROOT / "docs/GRAIN_MULTIVIEW_ADDRESS_REPLAY_PLAN_V1.md"
VIEWS = (
    "whole",
    "native",
    "sentence_pack_160",
    "fixed_128_cross",
    "fixed_128_within_comment",
)


def write(path, data):
    """Write finite JSON artifacts; NaN must not silently become a result."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def verify_inventory(directory):
    """Check every file in a frozen original inventory before deserialization."""
    inventory = json.loads((directory / "artifact_inventory.json").read_text())
    for item in inventory["files"]:
        path = directory / item["path"]
        if not path.exists():
            path = ROOT / item["path"]
        if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("Changed historical artifact: " + str(path))
    return {
        "files": len(inventory["files"]),
        "inventory_sha256": digest(directory / "artifact_inventory.json"),
    }


def evaluate(runtime, blocks):
    """Use the exact frozen consensus, direct and one-draw broken maps."""
    return evaluate_cross_view(
        blocks,
        model=runtime["final_model"],
        direct_models=runtime["direct_models"],
        permuted_models=runtime["permuted_models"],
    )


def replay_fixture(runtime, current_ids):
    """Reproduce the published 48-author fixture and check original240 exclusion."""
    fixture = json.loads((E1 / "replay_fixture.json").read_text())
    blocks = {}
    for view, info in fixture["feature_files"].items():
        frame = pd.read_csv(ROOT / info["path"], dtype={"user_id": str})
        if len(set(frame.user_id) & set(current_ids)):
            raise ValueError("Current material overlaps E1")
        if frame.user_id.nunique() != 240:
            raise ValueError("Unexpected original cohort")
        blocks[view] = transform_feature_block(
            frame,
            scaler=runtime["scalers"][view],
            user_ids=fixture["confirmation_user_ids"],
        )
    replay = (
        evaluate(runtime, blocks)
        .sort_values(["source_view", "target_view"])
        .reset_index(drop=True)
    )
    with np.load(ROOT / fixture["expected_file"], allow_pickle=False) as z:
        expected = (
            pd.DataFrame({k: z[k] for k in z.files})
            .sort_values(["source_view", "target_view"])
            .reset_index(drop=True)
        )
    errors = {}
    for key in ("consensus_global_r2", "direct_global_r2", "permuted_direct_global_r2"):
        np.testing.assert_allclose(replay[key], expected[key], **fixture["comparison"])
        errors[key] = float(np.max(np.abs(replay[key] - expected[key])))
    return {
        "original_authors": 240,
        "fixture_authors": 48,
        "overlap": 0,
        "max_errors": errors,
    }


def load_material():
    """Load authorized text/fold/address data without loading personality arrays."""
    material = PREDECESSOR / "material"
    manifest = json.loads((material / "material_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(material / name) != sha:
            raise ValueError("Changed material: " + name)
    data = json.loads((material / "private_material.json").read_text())
    comments = validate_addresses(data)
    ids = [b["author_id"] for b in data["bundles"]]
    folds = np.array([b["official_folds"]["big5"] for b in data["bundles"]])
    if (
        len(ids) != 624
        or len(data["sentences"]) != 29075
        or any(len(c) != 36 for c in comments)
    ):
        raise ValueError("Frozen material size changed")
    exposure = pd.read_csv(BASE / "vm05_preflight_r1/budget_by_author.csv")
    validate_reference_exposure(exposure, ids)
    raw = {}
    for a, cs in enumerate(comments):
        for j, c in enumerate(cs):
            if not np.isfinite(c["timestamp"]) or str(c["author_id"]) != str(ids[a]):
                raise ValueError("Invalid raw provenance")
            raw[a * 36 + j] = {
                "author": str(ids[a]),
                "body": c["body"],
                "created_utc": c["timestamp"],
                "subreddit": c["subreddit"],
                "comment_id": c["comment_id"],
            }
    sentences = [
        {**s, "source_row": s["author_index"] * 36 + s["comment_index"]}
        for s in data["sentences"]
    ]
    return ids, folds, raw, sentences, manifest


def history_summary():
    """Extract corrected historical evidence, preserving its exploratory status."""
    e4 = ROOT / "results/v7_operator_family/e4_v72_full_20260715_r3"
    w4 = ROOT / "results/v7_multiview_spectrum/w4b_effective_rank_corrected_20260715"
    ep, au = (
        pd.read_csv(e4 / "registered_endpoints.csv"),
        pd.read_csv(e4 / "source_disjoint_alignment.csv"),
    )
    return {
        "E4_inventory": verify_inventory(e4),
        "E4_endpoints": len(ep),
        "E4_supported": int((ep.status == "FWER_SUPPORTED").sum()),
        "E4_disjoint_auc_min": float(au.alignment_auc.min()),
        "E4_disjoint_auc_max": float(au.alignment_auc.max()),
        "E4_alignment_tier": "POST_HOC_EXPLORATORY_CORRECTED_20260717_NOT_NEW_PREREGISTRATION",
        "W4_inventory": verify_inventory(w4),
        "W4_decision": json.loads((w4 / "decision.json").read_text()),
        "W4_effective_rank": pd.read_csv(w4 / "effective_rank_profiles.csv").to_dict(
            "records"
        ),
        "G1_inventory": verify_inventory(GEO.parent),
        "G1_decision": json.loads((GEO.parent / "decision.json").read_text()),
    }


def run(args):
    """Execute the bounded no-fit, no-neural, no-label replay on a fresh output."""
    start = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    contract = {
        "plan_sha256": digest(PLAN),
        "script_sha256": digest(Path(__file__)),
        "sidecar_sha256": digest(ROOT / "grain_research/multiview_address.py"),
        "seed": 916,
        "bootstrap": 500,
        "fit": False,
        "labels_loaded": False,
        "neural_calls": 0,
    }
    write(out / "contract.json", contract)
    inventory = verify_inventory(E1)
    ids, folds, raw, sentences, material = load_material()
    runtime = joblib.load(E1 / "artifacts/multiview_runtime.joblib")
    if (
        tuple(runtime["view_names"]) != VIEWS
        or runtime["selected_rank"] != 23
        or runtime["selected_alpha"] != 10
    ):
        raise ValueError("Frozen model changed")
    fixture = replay_fixture(runtime, ids)
    write(
        out / "historical_evidence.json",
        {"E1_inventory": inventory, "E1_fixture": fixture, **history_summary()},
    )
    config = json.loads((E1 / "run_manifest.json").read_text())["config"]
    specs = {
        s["name"]: ObservationSpec(
            **s,
            min_tokens=config["min_tokens_per_unit"],
            max_units_per_user=config["max_units_per_user"],
            max_source_comments_per_user=config["max_source_comments_per_user"],
            max_source_tokens_per_user=config["max_source_tokens_per_user"],
        )
        for s in config["operators"]
    }
    comments = canonicalize_comments(
        pd.DataFrame(raw.values()), min_tokens=config["min_tokens_per_unit"]
    )
    comments["split"] = "current_replay"
    full = prepare_source_panel(comments, specs["native"])
    left, right = source_disjoint_partition(full, seed=916)
    if set(left.source_row) & set(right.source_row) or set(left.source_row) | set(
        right.source_row
    ) != set(full.source_row):
        raise ValueError("Partition source support mismatch")
    pd.concat(
        [
            f.assign(panel=name)
            for name, f in (("full", full), ("A", left), ("B", right))
        ]
    ).drop(columns="text").to_csv(out / "private_source_panels.csv", index=False)
    index = token_index(raw)
    feature_arrays, allblocks, summaries, coverage_rows, reconstruction = (
        {},
        {},
        [],
        [],
        [],
    )
    for panel_name, panel in (("full", full), ("A", left), ("B", right)):
        allblocks[panel_name] = {}
        for view in VIEWS:
            if time.monotonic() - start > 1200:
                raise TimeoutError("Frozen 20-minute bound; preserve partial artifacts")
            units = build_observations(panel, specs[view])
            segments = align_units(units, index, specs[view])
            segments.to_csv(
                out / f"private_segments_{panel_name}_{view}.csv", index=False
            )
            units.to_csv(out / f"private_units_{panel_name}_{view}.csv", index=False)
            h = runtime["representation"].transform(units.text.tolist())
            frame = align(author_features_from_embeddings(units, h), ids)
            values = frame[runtime["feature_names"]].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite author features; imputation forbidden")
            x = transform_feature_block(
                frame, scaler=runtime["scalers"][view], user_ids=ids
            )
            s = runtime["final_model"].encoders[view].predict(x)
            j = s @ runtime["final_model"].decoders[view]
            r = x - j
            reconstruction.append(
                {
                    "panel": panel_name,
                    "view": view,
                    "identity_max_error": float(np.max(np.abs(x - j - r))),
                    "raw_energy": float(np.mean(x * x)),
                    "shared_reconstruction_energy": float(np.mean(j * j)),
                    "remainder_energy": float(np.mean(r * r)),
                    "twice_cross_energy": float(2 * np.mean(j * r)),
                }
            )
            for part, array in (("raw", x), ("shared", s), ("remainder", r)):
                feature_arrays[f"{panel_name}__{view}__{part}"] = array
            allblocks[panel_name][view] = x
            frame.to_csv(out / f"private_features_{panel_name}_{view}.csv", index=False)
            summary = {
                "panel": panel_name,
                "view": view,
                "authors": len(frame),
                "units": len(units),
                "source_segments": len(segments),
                "format_changed_segments": int(segments.format_changed.sum()),
                "median_units": float(frame.n_units.median()),
                "mean_retained_tokens": float(frame.n_tokens.mean()),
            }
            summaries.append(summary)
            if panel_name == "full":
                coverage_rows.append(
                    sentence_coverage(sentences, segments, index).assign(view=view)
                )
                if view == "native":
                    geometry = GeometryBundle.from_dict(json.loads(GEO.read_text()))
                    if (
                        digest(E1 / "author_features_native.csv")
                        != geometry.reference_population["cohort_commitment"]
                    ):
                        raise ValueError("Geometry reference changed")
                    scores = score_geometry_bundle(
                        geometry,
                        frame[geometry.feature_names].to_numpy(float),
                        unit_counts=frame.n_units.to_numpy(),
                    )
                    old = (
                        pd.read_csv(
                            PREDECESSOR / "native_grain_r1/private_support.csv",
                            dtype={"user_id": str},
                        )
                        .set_index("user_id")
                        .loc[ids]
                    )
                    np.testing.assert_array_equal(scores["status"], old.native_status)
                    np.testing.assert_allclose(
                        scores["reference_radius"],
                        old.native_radius,
                        rtol=1e-10,
                        atol=1e-10,
                    )
                    summary["geometry_ready"] = int(
                        (old.native_status == "GEOMETRY_PROFILE_READY").sum()
                    )
                    summary["radius_max_error"] = float(
                        np.max(
                            np.abs(
                                np.asarray(scores["reference_radius"])
                                - old.native_radius
                            )
                        )
                    )
            print(json.dumps(summary), flush=True)
    np.savez_compressed(
        out / "private_configurations.npz",
        author_ids=np.array(ids),
        folds=folds,
        **feature_arrays,
    )
    pd.DataFrame(reconstruction).to_csv(
        out / "reconstruction_accounting.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(out / "observation_summary.csv", index=False)
    all_cross = []
    for cohort, mask in (
        ("all624", np.ones(len(ids), bool)),
        ("prior_held_fold0", folds == 0),
    ):
        all_cross.append(
            evaluate(
                runtime, {k: v[mask] for k, v in allblocks["full"].items()}
            ).assign(cohort=cohort)
        )
    pd.concat(all_cross).to_csv(out / "cross_view.csv", index=False)
    ranking = []
    for view in VIEWS:
        for part in ("raw", "shared", "remainder"):
            ranking.append(
                {
                    "view": view,
                    "part": part,
                    **author_rank_auc(
                        feature_arrays[f"A__{view}__{part}"],
                        feature_arrays[f"B__{view}__{part}"],
                    ),
                }
            )
    pd.DataFrame(ranking).to_csv(out / "source_disjoint_rank.csv", index=False)
    coverage = pd.concat(coverage_rows, ignore_index=True)
    coverage.to_csv(out / "private_sentence_coverage.csv", index=False)
    annotation = pd.read_csv(
        BASE
        / "all_plan_closure_v1/semantic_link_r2/private_annotation_coordinate_join.csv"
    )
    if len(annotation) != 80 or annotation.audit_id.duplicated().any():
        raise ValueError("Changed original annotation identities")
    joined = annotation[
        ["audit_id", "sentence_index", "simpa_category", "scope"]
    ].merge(coverage, on="sentence_index", how="left", validate="many_to_many")
    if len(joined) != 400 or joined.retention.isna().any():
        raise ValueError("Annotation crosswalk missing")
    joined.to_csv(out / "private_annotation_coverage.csv", index=False)
    cover_summary = []
    for cohort, f in (
        ("all_sentences", coverage),
        ("old80_source_annotations", joined),
    ):
        for view, g in f.groupby("view"):
            cover_summary.append(
                {
                    "cohort": cohort,
                    "view": view,
                    "sentences": len(g),
                    "fully_retained": int(g.full_retention.sum()),
                    "one_unit_complete": int(g.one_unit_complete.sum()),
                    "mean_token_retention": float(g.retention.mean()),
                    "fully_retained_but_fragmented": int(
                        (g.full_retention & ~g.one_unit_complete).sum()
                    ),
                    "no_tokens_retained": int((g.retained_tokens == 0).sum()),
                    "partial_source_tokens": int(g.partial_source_token.sum()),
                }
            )
    pd.DataFrame(cover_summary).to_csv(out / "coverage_summary.csv", index=False)
    files = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    summary = {
        "status": "COMPLETE",
        "authors": len(ids),
        "sentences": len(sentences),
        "elapsed_seconds": time.monotonic() - start,
        "material_files": material["files"],
        "E1_fixture": fixture,
        "source_disjoint_overlap": 0,
        "feature_reconstruction_max_error": max(
            x["identity_max_error"] for x in reconstruction
        ),
        "files": files,
    }
    write(out / "summary.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"files", "material_files"}}
        ),
        flush=True,
    )


def main():
    """CLI for fresh bounded replay; failure preserves artifacts and logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "all_plan_closure_v1/multiview_address_r1"
    )
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
