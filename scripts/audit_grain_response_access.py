#!/usr/bin/env python3
"""Independently audit RC4 scalar identities, intervention coverage and contrasts."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from scripts.audit_grain_meps_material import verify_outputs
from scripts.audit_grain_response_depth import compare, interval
from scripts.run_grain_vector_mechanism_controls import write_json

CUTS = (-1, 10, 16, 22, 27)
SITES = (10, 16, 22)
CONDITIONS = (
    {f"{r}_{k}": (k, False, r) for k in CUTS for r in ("actual", "matched")}
    | {
        f"patch_{k}_{label}": (27 if label == "full" else k, False, "matched")
        for k in SITES
        for label in ("full", "cut")
    }
    | {"mask_sham": (10, True, "matched"), "self_sham": (16, False, "matched")}
)


def frame_audit(frame, codes, checks):
    """Verify complete per-person ledgers; never trust headline effect signs alone."""
    if (
        len(codes) != len(set(codes))
        or len(frame) != 18 * len(codes)
        or frame.duplicated(["participant_code", "condition"]).any()
        or set(zip(frame.participant_code, frame.condition))
        != {(c, s) for c in codes for s in CONDITIONS}
    ):
        raise ValueError("Access condition/participant coverage mismatch")
    numeric = frame.drop(columns=["participant_code", "condition"])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite access evidence")
    check_map = {c["participant_code"]: c for c in checks}
    if len(checks) != len(codes) or set(check_map) != set(codes):
        raise ValueError("Missing instrument checks")
    for code in codes:
        part = frame[frame.participant_code == code].set_index("condition")
        if part.response_tokens.nunique() != 1:
            raise ValueError("Changed body token count")
        for name, (cut, sham, role) in CONDITIONS.items():
            row = part.loc[name]
            if (
                row.cut != cut
                or row.sham != sham
                or row.layers != 27 - cut
                or row.edges_per_layer
                != (row.prefix_tokens * row.response_tokens if cut != 27 else 0)
            ):
                raise ValueError("Attention edge or layer coverage mismatch")
            if (
                row.prefix_tokens != part.loc[f"{role}_27", "prefix_tokens"]
                or row.first_logp != part.loc[f"{role}_27", "first_logp"]
            ):
                raise ValueError("Prefix or first-token causality mismatch")
            expected_mean = (
                row.first_logp + (row.response_tokens - 1) * row.tail_logp
            ) / row.response_tokens
            if not np.isclose(row.mean_logp, expected_mean, rtol=0, atol=2e-5):
                raise ValueError("Tail/all-token score arithmetic mismatch")
        if (
            abs(
                part.loc["actual_-1", "tail_logp"] - part.loc["matched_-1", "tail_logp"]
            )
            > 1e-5
        ):
            raise ValueError("First-layer independence identity failed")
        for site in SITES:
            if (
                abs(
                    part.loc[f"patch_{site}_cut", "tail_logp"]
                    - part.loc[f"actual_{site}", "tail_logp"]
                )
                > 1e-5
            ):
                raise ValueError("Patch plus cut identity failed")
        for arm, base in (("mask_sham", "matched_27"), ("self_sham", "matched_16")):
            if not np.array_equal(
                part.loc[arm, ["mean_logp", "tail_logp", "first_logp"]],
                part.loc[base, ["mean_logp", "tail_logp", "first_logp"]],
            ):
                raise ValueError("Mask/self sham identity failed")
        record = check_map[code]
        for key in ("cut0_tail_error", "parent_replay_error", "depth_replay_error"):
            if key in record and not 0 <= record[key] <= 1e-5:
                raise ValueError("Recorded replay error outside tolerance")
        if (
            any(
                record[k] != 0
                for k in ("mask_sham_error", "self_sham_error", "first_token_error")
            )
            or set(record["patch_cut_errors"]) != {str(k) for k in SITES}
            or any(not 0 <= v <= 1e-5 for v in record["patch_cut_errors"].values())
        ):
            raise ValueError("Recorded access identity failed")
    p = frame.pivot(index="participant_code", columns="condition", values="tail_logp")
    native = p.actual_27 - p.matched_27
    effects, patches = {}, {}
    for cut in CUTS:
        retained = p[f"actual_{cut}"] - p[f"matched_{cut}"]
        effects[str(cut)] = {
            "retained_context_difference": interval(retained),
            "differential_access_effect": interval(native - retained),
            "actual_access_effect": interval(p.actual_27 - p[f"actual_{cut}"]),
            "matched_access_effect": interval(p.matched_27 - p[f"matched_{cut}"]),
        }
    for site in SITES:
        full = p[f"patch_{site}_full"] - p.matched_27
        cut = p[f"patch_{site}_cut"] - p[f"matched_{site}"]
        patches[str(site)] = {
            "full_patch_effect": interval(full),
            "cut_patch_effect": interval(cut),
            "full_minus_cut_patch_effect": interval(full - cut),
        }
    return {"effects": effects, "patch_effects": patches}


def audit(run, parent, depth):
    """Replay sealed parent links and numerical evidence without private raw text."""
    result, rc1, rc2 = (verify_outputs(p) for p in (run, parent, depth))
    m = json.loads((run / "manifest.json").read_text())
    if (
        result["status"] != "COMPLETE"
        or result["participants"] != 44
        or result["forwards"] != 834
        or result["parent_identity"] != m["identity"]
        or m["identity"]
        != material_hash({k: v for k, v in m.items() if k != "identity"})
        or tuple(m["cuts"]) != CUTS
        or tuple(m["sites"]) != SITES
    ):
        raise ValueError("Incomplete/changed RC4 identity or coverage")
    if (
        rc1["status"] != "COMPLETE"
        or rc2["status"] != "COMPLETE"
        or any(x["rc1_identity"] != rc1["parent_identity"] for x in (m, result, rc2))
        or any(x["rc2_identity"] != rc2["parent_identity"] for x in (m, result))
    ):
        raise ValueError("Different RC1/RC2 origin")
    material = json.loads((run / "private_material_manifest.json").read_text())
    if (
        material != json.loads((parent / "private_material_manifest.json").read_text())
        or material_hash(material["cases"]) != material["identity"]
    ):
        raise ValueError("Changed material/assignment")
    codes = [x["participant_code"] for x in material["cases"]]
    f = pd.read_csv(run / "private_access_metrics.csv")
    checks = json.loads((run / "private_access_checks.json").read_text())
    findings = frame_audit(f, codes, checks)
    lengths = {x["participant_code"]: x["response_tokens"] for x in material["cases"]}
    if not f.response_tokens.eq(f.participant_code.map(lengths)).all():
        raise ValueError("Frozen response length drift")
    old = pd.read_csv(parent / "private_response_metrics.csv").set_index(
        "participant_code"
    )
    old_depth = pd.read_csv(depth / "private_depth_metrics.csv").set_index(
        ["participant_code", "site"]
    )
    native_error, depth_error = 0, 0
    for code in codes:
        p = f[f.participant_code == code].set_index("condition")
        native_error = max(
            native_error,
            *(
                abs(p.loc[f"{arm}_27", k] - old.loc[code, f"{arm}_{k}"])
                for arm in ("actual", "matched")
                for k in ("mean_logp", "tail_logp")
            ),
        )
        depth_error = max(
            depth_error,
            *(
                abs(
                    p.loc[f"patch_{site}_full", "tail_logp"]
                    - old_depth.loc[(code, f"block_{site:02}"), "patch_tail_logp"]
                )
                for site in SITES
            ),
        )
    if max(native_error, depth_error) > 1e-5:
        raise ValueError("Native/RC2 patch replay failed")
    numerical = json.loads((run / "technical_numerical_gate.json").read_text())
    if len(numerical) != 2 or any(
        not x["passed"]
        or x["self_patch_max_logp_error"] != 0
        or x["compact_fixed_mean_logp_difference"] > 0.02
        or x["minimum_site_cosine"] < 0.999
        for x in numerical
    ):
        raise ValueError("Nonprivate numerical gate failed")
    tech = pd.read_csv(run / "technical_access_metrics.csv").rename(
        columns={"language": "participant_code"}
    )
    tc = [
        {"participant_code": x["language"], **x}
        for x in json.loads((run / "technical_access_gate.json").read_text())
    ]
    frame_audit(tech, ["en", "ja"], tc)
    for name in findings:
        compare(result[name], findings[name])
    return {
        "status": "COMPLETE_ACCESS_SCALAR_AND_MATERIAL_AUDIT",
        "parent_identity": m["identity"],
        "participants": 44,
        "private_rows": len(f),
        "technical_rows": len(tech),
        "forwards": 834,
        "native_replay_max_error": native_error,
        "depth_replay_max_error": depth_error,
        **findings,
        "new_model_forwards": 0,
        "raw_text_reopened": False,
        "labels_read": False,
        "scope": "scalar/material/recorded-hook audit; not independent neural or attention-tensor replay",
    }


def main():
    """Write a fresh independent audit packet, not changes to the original evidence."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "parent-dir", "depth-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(x for x in a.output_dir.iterdir() if not x.name.startswith("._")):
        raise ValueError("Fresh audit directory required")
    result = audit(a.run_dir, a.parent_dir, a.depth_dir)
    write_json(a.output_dir / "summary.json", result)
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for path in (
            Path(__file__),
            ROOT / "scripts/audit_grain_response_depth.py",
            ROOT / "tests/test_grain_response_access_audit.py",
        ):
            z.write(path, path.relative_to(ROOT))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "participants",
                    "private_rows",
                    "native_replay_max_error",
                    "depth_replay_max_error",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
