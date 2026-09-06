#!/usr/bin/env python3
"""Independently replay the fixed-site response-depth scalar evidence."""

import argparse
import json
import sys
import zipfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from scripts.audit_grain_meps_material import verify_outputs
from scripts.run_grain_vector_mechanism_controls import write_json

EXPECTED_SITES = (
    "block_04",
    "block_10",
    "block_16",
    "block_22",
    "block_27",
    "final_norm",
)


def interval(values):
    """Independent conditional-bootstrap arithmetic in the frozen case order."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(731)
    draws = rng.integers(0, len(x), size=(1000, len(x)))
    low, high = np.quantile(np.mean(x[draws], axis=1), [0.025, 0.975])
    return {
        "mean": float(x.mean()),
        "lower": float(low),
        "upper": float(high),
        "positive_count": int(np.count_nonzero(x > 0)),
        "participants": len(x),
    }


def compare(found, expected):
    """Require the whole claimed numeric object, not merely its headline mean."""
    if found.keys() != expected.keys():
        raise ValueError("Claimed summary keys differ from recomputation")
    for name, value in expected.items():
        if isinstance(value, dict):
            compare(found[name], value)
        elif value is None:
            if found[name] is not None:
                raise ValueError("Undefined ratio reported as a number")
        elif not np.isclose(found[name], value, rtol=0, atol=1e-12):
            raise ValueError("Depth scalar/interval failed independent replay")


def audit(run_dir, parent_dir):
    """Verify checksums, paired coverage, controls and aggregate math without text."""
    result = verify_outputs(run_dir)
    parent = verify_outputs(parent_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (
        result["status"] != "COMPLETE"
        or parent["status"] != "COMPLETE"
        or result["parent_identity"] != manifest["identity"]
        or result["rc1_identity"] != parent["parent_identity"]
        or manifest["parent_identity"] != parent["parent_identity"]
        or manifest["identity"]
        != material_hash({k: v for k, v in manifest.items() if k != "identity"})
        or tuple(manifest["sites"]) != EXPECTED_SITES
    ):
        raise ValueError("Run or parent identity is incomplete/altered")
    material = json.loads((run_dir / "private_material_manifest.json").read_text())
    old_material = json.loads(
        (parent_dir / "private_material_manifest.json").read_text()
    )
    if (
        material != old_material
        or material_hash(material["cases"]) != material["identity"]
    ):
        raise ValueError("Changed response or donor assignment")
    codes = [c["participant_code"] for c in material["cases"]]
    metrics = pd.read_csv(run_dir / "private_depth_metrics.csv")
    base = pd.read_csv(run_dir / "private_baseline_metrics.csv").set_index(
        "participant_code"
    )
    old = pd.read_csv(parent_dir / "private_response_metrics.csv").set_index(
        "participant_code"
    )
    keys = set(zip(metrics.participant_code, metrics.site))
    if (
        len(codes) != len(set(codes))
        or len(metrics) != 6 * len(codes)
        or keys != {(c, s) for c in codes for s in EXPECTED_SITES}
        or metrics.duplicated(["participant_code", "site"]).any()
        or base.index.has_duplicates
        or set(base.index) != set(codes)
        or result["participants"] != len(codes)
        or result["forwards"] != 12 + 14 * len(codes)
    ):
        raise ValueError("Layer or response coverage mismatch")
    if (
        not np.isfinite(
            metrics.drop(columns=["participant_code", "site"]).to_numpy()
        ).all()
        or not np.isfinite(base.to_numpy()).all()
    ):
        raise ValueError("Nonfinite evidence")
    base = base.loc[codes]
    expected_lengths = [c["response_tokens"] for c in material["cases"]]
    if not np.array_equal(base.response_tokens, expected_lengths):
        raise ValueError("Token count drift")
    for field in ("actual_tail_logp", "matched_tail_logp"):
        if not np.allclose(base[field], old.loc[codes, field], rtol=0, atol=1e-5):
            raise ValueError("RC1 baseline failed replay")
    if not base.parent_replay_max_error.between(0, 1e-5).all():
        raise ValueError("Parent replay tolerance failed")
    if (
        not metrics.self_patch_max_error.eq(0).all()
        or not metrics.patch_first_token_change.eq(0).all()
    ):
        raise ValueError("Self-patch or prefix causality control failed")
    terminal = metrics.loc[metrics.site.isin(["block_27", "final_norm"])]
    if not terminal.patch_actual_tail_max_error.between(0, 1e-5).all():
        raise ValueError("Private positive control failed")
    numerical = json.loads((run_dir / "technical_numerical_gate.json").read_text())
    readout = json.loads((run_dir / "technical_readout_gate.json").read_text())
    if len(numerical) != 2 or len(readout) != 2:
        raise ValueError("Incomplete synthetic controls")
    if any(
        not x["passed"]
        or x["self_patch_max_logp_error"] != 0
        or x["compact_fixed_mean_logp_difference"] > 0.02
        or x["minimum_site_cosine"] < 0.999
        for x in numerical
    ):
        raise ValueError("Synthetic numerical gate failed")
    if any(
        not x["passed"] or x["tail_max_error"] > 1e-5 or x["first_token_change"] != 0
        for x in readout
    ):
        raise ValueError("Synthetic final readout gate failed")
    effects, rows = {}, []
    for site in EXPECTED_SITES:
        part = (
            metrics.loc[metrics.site == site].set_index("participant_code").loc[codes]
        )
        for field in ("actual_tail_logp", "matched_tail_logp"):
            if not np.allclose(part[field], base[field], rtol=0, atol=1e-12):
                raise ValueError("Per-layer baseline differs from paired base")
        delta = part.patch_tail_logp - part.matched_tail_logp
        gap = part.actual_tail_logp - part.matched_tail_logp
        effects[site] = {
            "patch_minus_matched": interval(delta),
            "actual_minus_matched": interval(gap),
            "actual_minus_patch": interval(gap - delta),
            "ratio_of_group_means": float(delta.mean() / gap.mean())
            if abs(gap.mean()) > 1e-12
            else None,
            "maximum_self_patch_error": float(part.self_patch_max_error.max()),
            "maximum_tail_replay_error": float(part.patch_actual_tail_max_error.max()),
            "maximum_first_token_change": float(part.patch_first_token_change.max()),
        }
        rows.append(
            {
                "site": site,
                **interval(delta),
                "residual_gap": float((gap - delta).mean()),
                "ratio_of_group_means": effects[site]["ratio_of_group_means"],
            }
        )
    pivot = metrics.pivot(
        index="participant_code", columns="site", values="patch_tail_logp"
    )
    adjacent = {
        f"{b}_minus_{a}": interval(pivot[b] - pivot[a])
        for a, b in pairwise(EXPECTED_SITES)
    }
    compare(result["effects"], effects)
    compare(result["adjacent_layer_contrasts"], adjacent)
    return {
        "status": "COMPLETE_SCALAR_AND_MATERIAL_AUDIT",
        "parent_identity": manifest["identity"],
        "rc1_identity": parent["parent_identity"],
        "participants": len(codes),
        "layer_rows": len(metrics),
        "forwards": result["forwards"],
        "effects": effects,
        "adjacent_layer_contrasts": adjacent,
        "parent_replay_max_error": float(base.parent_replay_max_error.max()),
        "new_model_forwards": 0,
        "raw_text_reopened": False,
        "labels_read": False,
        "scope": "scalar/material audit; does not independently reconstruct unpersisted activations",
    }, rows


def main():
    """Write a new audit artifact without modifying the sealed experiment."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "parent-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(x for x in a.output_dir.iterdir() if not x.name.startswith("._")):
        raise ValueError("Fresh audit directory required")
    result, rows = audit(a.run_dir, a.parent_dir)
    pd.DataFrame(rows).to_csv(a.output_dir / "layer_effects.csv", index=False)
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        z.write(Path(__file__), Path(__file__).relative_to(ROOT))
    write_json(a.output_dir / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
