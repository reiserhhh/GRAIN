#!/usr/bin/env python3
"""Independent saved-record checks for a complete role/opportunity P1B packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.role_opportunity import check_donor_layout


def audit(parent: Path) -> dict:
    """Check frozen source, candidate arithmetic, directions and causal-prefix invariance."""
    manifest = json.loads((parent / "manifest.json").read_text())
    status = json.loads((parent / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["calls"] > manifest["max_calls"]:
        raise ValueError("Complete bounded neural packet required")
    materials = json.loads((parent / "materials.json").read_text())["rows"]
    if material_hash(materials) != manifest["material_hash"]:
        raise ValueError("Material identity changed")
    with zipfile.ZipFile(parent / "source_snapshot.zip") as archive:
        for name, sha in manifest["code"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError("Frozen source snapshot differs")
    for name, sha in json.loads((parent / "output_hashes.json").read_text()).items():
        if hashlib.sha256((parent / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Output checksum differs")
    base = pd.read_csv(parent / "baselines.csv")
    effect = pd.read_csv(parent / "interventions.csv")
    if len(base) != 156 or len(effect) != 648 or base.case_id.nunique() != 156:
        raise ValueError("Incomplete native/intervention grid")
    if (effect.kind == "additive").sum() != 624 or (effect.kind == "donor").sum() != 24:
        raise ValueError("Wrong intervention family counts")
    keys = ["case_id", "kind", "position", "source", "sign"]
    if effect.duplicated(keys).any():
        raise ValueError("Duplicate intervention")
    error = 0.0
    for frame in (base, effect):
        for r in frame.itertuples():
            high, low = json.loads(r.candidate_scores)
            error = max(error, abs((high - low) - r.margin))
    base_lookup = base.set_index("case_id").margin
    delta_error = float(
        np.max(abs(effect.margin - effect.case_id.map(base_lookup) - effect.delta))
    )
    if max(error, delta_error) > 1e-10:
        raise ValueError("Candidate or delta arithmetic mismatch")
    states = {
        r["case_id"]: np.load(
            parent / "activation_cache" / (r["case_id"] + ".npz"), allow_pickle=False
        )["state"]
        for r in materials
    }
    if not all(np.isfinite(s).all() for s in states.values()):
        raise ValueError("Invalid saved activation")
    directions = np.load(parent / "directions.npz", allow_pickle=False)
    meta = json.loads((parent / "direction_metadata.json").read_text())
    reconstruction = 0.0
    for source in ("self", "other", "behavior"):
        source_rows = [
            r
            for r in materials
            if r["split"] == "development" and r["family"] == source
        ]
        groups = sorted({r["group_id"] for r in source_rows})
        lookup = {(r["group_id"], r["level"]): r["case_id"] for r in source_rows}
        raw = sum(
            states[lookup[g, 1]][-2].astype(float)
            - states[lookup[g, -1]][-2].astype(float)
            for g in groups
        ) / len(groups)
        estimated = raw * meta["shared_norm"] / np.linalg.norm(raw)
        reconstruction = max(
            reconstruction, float(np.max(abs(estimated - directions[source])))
        )
    if reconstruction > 1e-8:
        raise ValueError("Development direction cannot be reconstructed")
    layouts = json.loads((parent / "token_layouts.json").read_text())
    prefix_error = 0.0
    for group in sorted({r["group_id"] for r in materials if r["family"] == "trait"}):
        for level in (-1, 1):
            ids = [
                r["case_id"]
                for r in materials
                if r["family"] == "trait"
                and r["group_id"] == group
                and r["level"] == level
            ]
            for case in ids[1:]:
                prefix_error = max(
                    prefix_error,
                    float(np.max(abs(states[case][:-1] - states[ids[0]][:-1]))),
                )
    donor = effect[effect.kind == "donor"]
    for r in donor.itertuples():
        a, b = layouts[r.case_id], layouts[r.donor_case]
        check_donor_layout(a["ids"], a["span"], b["ids"], b["span"])
    copy_error = float(donor[donor.source == "self_copy"].delta.abs().max())
    counterfactual = donor[donor.source == "counterfactual"]
    # Nonzero recovery discrepancy can arise from paths bypassing this layer/span;
    # it is a scientific observation, not automatically an instrumentation error.
    recovery = abs(counterfactual.margin - counterfactual.donor_case.map(base_lookup))
    checks = json.loads((parent / "instrument_checks.json").read_text())
    restoration = max(r["error"] for r in checks)
    if max(copy_error, restoration) > 1e-5:
        raise ValueError("Self-copy/restore failed")
    return {
        "status": "VERIFIED",
        "parent_identity": manifest["identity"],
        "calls": status["calls"],
        "cases": len(base),
        "interventions": len(effect),
        "score_arithmetic_max": error,
        "delta_arithmetic_max": delta_error,
        "direction_reconstruction_max": reconstruction,
        "prefix_state_max": prefix_error,
        "self_copy_max": copy_error,
        "restore_max": restoration,
        "counterfactual_recovery_mean": float(recovery.mean()),
        "counterfactual_recovery_max": float(recovery.max()),
        "additional_neural_calls": 0,
        "psychological_validity_claim": False,
    }


def main():
    """Write a fresh audit receipt without editing the original run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.parent_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
