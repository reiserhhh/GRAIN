#!/usr/bin/env python3
"""Seal same-ready populations and nested membership before comparator fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_registry import (
    TASKS,
    membership_feasibility,
    registry_arrays,
)
from scripts.run_grain_vector_mechanism_controls import write_json

COHORTS = (
    "raw_ready",
    "all_arms_ready",
    "raw_ready_reference_free",
    "all_arms_ready_reference_free",
)


def digest(path):
    """Stream file commitments without keeping a second copy in memory."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    """Prepare only text-derived support, IDs and folds; never load target values."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "profile-dir", "preflight-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=600)
    a = p.parse_args()
    if not 60 <= a.max_seconds <= 1800:
        raise ValueError("Require bounded label-free preparation")
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private output must remain under results")
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Fresh registry output required")
    start = time.monotonic()
    source = json.loads(a.views.read_text())
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    pm = json.loads((a.profile_dir / "manifest.json").read_text())
    ps = json.loads((a.profile_dir / "summary.json").read_text())
    if (
        material_hash(source) != common["material_sha256"]
        or pm["material_sha256"] != common["material_sha256"]
        or material_hash({k: v for k, v in pm.items() if k != "identity"})
        != pm["identity"]
        or ps["status"] != "COMPLETE"
        or ps["parent_identity"] != pm["identity"]
        or pm["labels_read"]
    ):
        raise ValueError("Changed or incomplete profile source")
    for name, sha in ps["outputs_sha256"].items():
        path = (a.profile_dir / name).resolve()
        if not path.is_relative_to(a.profile_dir.resolve()) or digest(path) != sha:
            raise ValueError("Changed profile artifact")
    with np.load(
        a.profile_dir / "private_profile_features.npz", allow_pickle=False
    ) as z:
        profiles = {k: z[k] for k in z.files}
    budget_path = a.preflight_dir / "budget_by_author.csv"
    budget = pd.read_csv(budget_path, dtype={"author_id": str})
    clean_ids = {b["author_id"] for b in exclude_reference(source, budget)}
    reference = {b["author_id"] for b in source} - clean_ids
    arrays = registry_arrays(source, profiles, reference)
    if set(arrays["author_ids"]) != set(common["author_ids"]):
        raise ValueError("Common author set mismatch")
    fields = {
        "protocol": "VM05_SAME_READY_REGISTRY_V1",
        "material_sha256": common["material_sha256"],
        "profile_identity": pm["identity"],
        "profile_outputs_sha256": ps["outputs_sha256"],
        "budget_sha256": digest(budget_path),
        "cohorts": list(COHORTS),
        "tasks": list(TASKS),
        "source_code": {
            name: digest(ROOT / name)
            for name in (
                "scripts/prepare_grain_same_ready_registry.py",
                "grain_research/vector_mechanisms/ready_registry.py",
                "grain_research/vector_mechanisms/external_controls.py",
                "grain_research/vector_mechanisms/external_profiles.py",
            )
        },
        "labels_read": False,
        "bridge_outcomes_read": False,
        "support_rule_changed": False,
        "max_seconds": a.max_seconds,
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    archive_source(ROOT, out / "source_snapshot.zip")
    coverage, records = [], []
    try:
        for cohort in COHORTS:
            for group in ("big5_only", "mbti_only", "bridge"):
                eligible = arrays[cohort] & (arrays["source_group"] == group)
                coverage.append(
                    {
                        "cohort": cohort,
                        "source_group": group,
                        "authors": int(eligible.sum()),
                        "reference_members": int(
                            (eligible & arrays["reference_member"]).sum()
                        ),
                        "author_commitment": material_hash(
                            arrays["author_ids"][eligible].tolist()
                        ),
                    }
                )
            for task in TASKS:
                records.extend(membership_feasibility(source, arrays, cohort, task))
                if time.monotonic() - start > a.max_seconds:
                    raise TimeoutError("Registry membership budget exhausted")
        np.savez_compressed(out / "private_ready_registry.npz", **arrays)
        pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
        frame = pd.DataFrame(records)
        frame.to_csv(out / "private_membership_feasibility.csv", index=False)
        support = frame.groupby(["cohort", "task"]).agg(
            selection_points=("scope", "size"),
            minimum_train_count=("train_retained", "min"),
            all_min20_supported=("dense_min20_supported", "all"),
        )
        support.to_csv(out / "training_support.csv")
        names = (
            "private_ready_registry.npz",
            "coverage.csv",
            "private_membership_feasibility.csv",
            "training_support.csv",
        )
        summary = {
            "status": "COMPLETE",
            "parent_identity": fields["identity"],
            "authors": len(source),
            "source_label_values_read": False,
            "bridge_outcomes_read": False,
            "model_refits": 0,
            "neural_forwards": 0,
            "membership_records": len(records),
            "under_min20_records": int((~frame.dense_min20_supported).sum()),
            "class_support_not_checked_without_labels": True,
            "outputs_sha256": {name: digest(out / name) for name in names},
            "coverage": [
                {k: v for k, v in r.items() if k != "author_commitment"}
                for r in coverage
            ],
            "seconds": time.monotonic() - start,
        }
        write_json(out / "summary.json", summary)
        write_json(out / "run_status.json", summary)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "outputs_sha256"}),
            flush=True,
        )
    except Exception as error:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_STOP",
                "exception_type": type(error).__name__,
                "membership_records_completed": len(records),
                "labels_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
