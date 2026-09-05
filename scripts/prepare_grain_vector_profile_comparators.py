#!/usr/bin/env python3
"""Prepare same-evidence frozen geometry channels without psychological labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import frozen_masked_features
from grain_research.vector_mechanisms.external_profiles import (
    landmark_information_audit,
    profile_features,
    ready_cohorts,
)
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from suica_core.v7_geometry import GeometryBundle


def main():
    """Seal profile support and geometry differences before source head comparisons."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--feature-cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=1800)
    a = p.parse_args()
    if not 60 <= a.max_seconds <= 3600:
        raise ValueError("Require bounded profile preparation")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(q for q in out.iterdir() if not q.name.startswith("._")):
        raise ValueError("Use a new output directory")
    source = json.loads(a.views.read_text())
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    pre = json.loads((a.preflight_dir / "manifest.json").read_text())
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Common material changed")
    ids = np.asarray([b["author_id"] for b in source])
    if len(set(ids)) != len(ids) or set(ids) != set(common["author_ids"]):
        raise ValueError("Duplicate or missing common author")
    config_path = ROOT / "configs/v8_pandora_external_connection.json"
    config = json.loads(config_path.read_text())["score"]
    geometry_path = ROOT / config["geometry_bundle"]
    geometry_sha = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    if geometry_sha != pre["geometry_sha256"]:
        raise ValueError("Geometry differs from the frozen preflight")
    geometry = GeometryBundle.from_dict(json.loads(geometry_path.read_text()))
    if list(geometry.feature_names) != pre["feature_names"]:
        raise ValueError("Frozen intermediate column order changed")
    npz_path = a.preflight_dir / "grain_frozen_features.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        old_ids = z["author_ids"].tolist()
        if len(set(old_ids)) != len(old_ids) or set(ids) - set(old_ids):
            raise ValueError("Preflight feature authors are incomplete")
        lookup = {uid: i for i, uid in enumerate(old_ids)}
        index = [lookup[uid] for uid in ids]
        raw = z["all36_raw_features"][index]
        expected_profile = z["all36_landmark_profile"][index]
        expected_status = z["all36_status"][index]
    fields = {
        "protocol": "VM05_FROZEN_GEOMETRY_CHANNELS_V1",
        "material_sha256": common["material_sha256"],
        "preflight_identity": pre["identity"],
        "preflight_npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "geometry_sha256": geometry_sha,
        "runtime_sha256": pre["runtime_sha256"],
        "legacy_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "relative_tolerance": config["relative_tolerance"],
        "arms": ["raw", "cue_mask", "placebo_mask"],
        "source_code": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "grain_research/vector_mechanisms/external_profiles.py",
                "grain_research/vector_mechanisms/external_controls.py",
                "grain_research/vector_mechanisms/external_material.py",
                "scripts/prepare_grain_vector_profile_comparators.py",
                "scripts/run_grain_vector_mechanism_external_features.py",
                "suica_core/v7_psychometric.py",
                "suica_core/v7_geometry.py",
                "suica_core/v8_bridge.py",
                "suica_core/v8_external_connection.py",
            )
        },
        "authors": len(source),
        "labels_read": False,
        "bridge_labels_read": False,
        "outside_support_values": "diagnostic_only_not_formal_scores",
        "claim_scope": "Frozen V7/V8 geometry components, not the complete GRAIN 4W theory",
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    archive_source(ROOT, out / "source_snapshot.zip")
    arrays, diagnostics, status, rows = {"author_ids": ids}, {}, {}, []
    started = time.monotonic()
    with threadpool_limits(limits=4):
        for arm in fields["arms"]:
            x = (
                raw
                if arm == "raw"
                else frozen_masked_features(
                    ROOT, a.preflight_dir, source, arm, a.feature_cache_dir
                )
            )
            values, diagnostic = profile_features(
                geometry,
                x,
                np.full(len(source), 36),
                relative_tolerance=config["relative_tolerance"],
            )
            if arm == "raw" and (
                not np.array_equal(values["sorted_distances"], expected_profile)
                or not np.array_equal(values["status"], expected_status)
            ):
                raise ValueError("Existing preflight raw geometry failed exact replay")
            diagnostics[arm] = diagnostic
            status[arm] = values["status"]
            arrays.update({f"{arm}_{key}": value for key, value in values.items()})
            for i, b in enumerate(source):
                rows.append(
                    {
                        "author_id": b["author_id"],
                        "source_group": b["source_group"],
                        "arm": arm,
                        "status": values["status"][i],
                        "reference_radius": values["reference_radius"][i],
                    }
                )
            progress = {
                "status": "RUNNING",
                "completed_arms": list(diagnostics),
                "seconds": time.monotonic() - started,
                "labels_read": False,
            }
            write_json(out / "run_status.json", progress)
            print(json.dumps(progress), flush=True)
            if time.monotonic() - started > a.max_seconds:
                write_json(
                    out / "run_status.json", progress | {"status": "PARTIAL_BUDGET"}
                )
                raise RuntimeError("Profile preparation exceeded arm-boundary budget")
    cohort = ready_cohorts(status)
    arrays.update(cohort)
    groups = np.asarray([b["source_group"] for b in source])
    support = pd.DataFrame(rows)
    support.to_csv(out / "private_support_by_author.csv", index=False)
    support.groupby(["source_group", "arm", "status"]).size().reset_index(
        name="authors"
    ).to_csv(out / "support_summary.csv", index=False)
    np.savez_compressed(out / "private_profile_features.npz", **arrays)
    geometry_audit = landmark_information_audit(
        arrays["raw_whitened"], np.asarray(geometry.reference_landmarks)
    )
    write_json(out / "geometry_information_audit.json", geometry_audit)
    write_json(out / "canonical_diagnostics.json", diagnostics)
    summary = {
        "status": "COMPLETE",
        "parent_identity": fields["identity"],
        "authors": len(source),
        "seconds": time.monotonic() - started,
        "cohorts": {
            name: {
                group: int(np.sum(mask & (groups == group)))
                for group in sorted(set(groups))
            }
            for name, mask in cohort.items()
        },
        "raw_preflight_replay_exact": True,
        "labels_read": False,
        "bridge_labels_read": False,
        "outputs_sha256": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in (
                "private_profile_features.npz",
                "private_support_by_author.csv",
                "support_summary.csv",
                "geometry_information_audit.json",
                "canonical_diagnostics.json",
            )
        },
    }
    write_json(out / "summary.json", summary)
    write_json(out / "run_status.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
