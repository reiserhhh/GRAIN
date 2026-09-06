#!/usr/bin/env python3
"""Audit saved RC5 scalar identities and person-weighted results independently."""

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


def audit_frame(frame, checks):
    """Recompute scalar algebra and aggregation without importing the runner math."""
    c = pd.DataFrame(checks).set_index("participant_code")
    if (
        c.index.has_duplicates
        or set(frame.participant_code) != set(c.index)
        or frame.duplicated(["participant_code", "response_token_position"]).any()
    ):
        raise ValueError("Missing/duplicate person or position")
    numeric = frame.drop(columns="participant_code")
    if (
        not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or not np.isfinite(c.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Nonfinite readout record")
    for code in c.index:
        f = frame[frame.participant_code == code]
        count = int(c.loc[code, "response_tokens"])
        if not np.array_equal(
            np.sort(f.response_token_position), np.arange(2, count + 1)
        ):
            raise ValueError("Tail position coverage or off-by-one mismatch")
        if (
            c.loc[code, "head_batches"] != 2
            or c.loc[code, "head_execution_rows"] != count
            or c.loc[code, "vocabulary_reduction_batches"] != (count - 1 + 7) // 8
        ):
            raise ValueError("Unrecorded head-only batches")
    if (
        not c.algebra_max_error.between(0, 1e-9).all()
        or not c.head_replay_max_error.between(0, 1e-5).all()
        or not c.mean_predictor_state_cosine.between(-1, 1).all()
    ):
        raise ValueError("Recorded head/algebra replay failed")
    if (
        "parent_replay_max_error" in c
        and not c.parent_replay_max_error.between(0, 1e-5).all()
    ):
        raise ValueError("Recorded RC1 replay failed")
    f = frame
    identities = (
        f.delta_logp - (f.actual_logp - f.matched_logp),
        f.delta_logp - (f.target_logit_shift - f.log_partition_shift),
        f.delta_logp - (f.matched_gradient_dot_delta - f.kl_matched_to_actual),
        f.delta_logp - (f.actual_gradient_dot_delta + f.kl_actual_to_matched),
    )
    error = max(float(abs(x).max()) for x in identities)
    if error > 1e-9:
        raise ValueError("Exact softmax scalar identity failed")
    if (
        min(
            f.kl_matched_to_actual.min(),
            f.kl_actual_to_matched.min(),
            f.matched_directional_curvature.min(),
            f.state_displacement_norm.min(),
        )
        < -1e-9
        or not f.state_cosine.between(-1, 1).all()
    ):
        raise ValueError("Readout positivity or geometry bounds failed")
    if ((f.actual_gradient_dot_delta - f.delta_logp) > 1e-9).any() or (
        (f.delta_logp - f.matched_gradient_dot_delta) > 1e-9
    ).any():
        raise ValueError("Endpoint concavity bound failed")
    cols = [k for k in f if k not in ("participant_code", "response_token_position")]
    persons = f.groupby("participant_code", sort=True)[cols].mean()
    persons["mean_predictor_state_cosine"] = c.mean_predictor_state_cosine
    effects = {k: interval(persons[k]) for k in persons}
    return persons, effects, error


def audit(run, parent):
    """Verify full source links and saved scalar evidence, without raw text access."""
    result, rc1 = verify_outputs(run), verify_outputs(parent)
    m = json.loads((run / "manifest.json").read_text())
    if (
        result["status"] != "COMPLETE"
        or rc1["status"] != "COMPLETE"
        or result["participants"] != 44
        or result["forwards"] != 98
        or result["parent_identity"] != m["identity"]
        or m["identity"]
        != material_hash({k: v for k, v in m.items() if k != "identity"})
        or any(x["rc1_identity"] != rc1["parent_identity"] for x in (m, result))
    ):
        raise ValueError("Incomplete or changed readout identity")
    old_manifest = json.loads((parent / "manifest.json").read_text())
    if (
        m["model_files"] != old_manifest["model_files"]
        or m["template_sha256"] != old_manifest["template_sha256"]
        or m["head_execution"] != "native_reply_length"
        or m["vocabulary_reduction_rows"] != 8
        or m["readout_reduction_dtype"] != "float64"
    ):
        raise ValueError("Readout instrument identity changed")
    material = json.loads((run / "private_material_manifest.json").read_text())
    if (
        material != json.loads((parent / "private_material_manifest.json").read_text())
        or material_hash(material["cases"]) != material["identity"]
    ):
        raise ValueError("Material/assignment drift")
    frame = pd.read_csv(run / "private_readout_metrics.csv")
    checks = json.loads((run / "private_readout_checks.json").read_text())
    persons, effects, error = audit_frame(frame, checks)
    expected = {x["participant_code"]: x["response_tokens"] for x in material["cases"]}
    if (
        {x["participant_code"]: x["response_tokens"] for x in checks} != expected
        or len(frame) != result["tail_token_rows"]
        or len(persons) != 44
    ):
        raise ValueError("Changed private person/token coverage")
    saved = pd.read_csv(run / "private_person_readout_means.csv").set_index(
        "participant_code"
    )
    if (
        set(saved.index) != set(persons.index)
        or list(saved.columns) != list(persons.columns)
        or not np.allclose(saved.loc[persons.index], persons, rtol=0, atol=1e-12)
    ):
        raise ValueError("Person-weighted means differ from token records")
    old = pd.read_csv(parent / "private_response_metrics.csv").set_index(
        "participant_code"
    )
    replay = max(
        float(
            abs(
                persons[role + "_logp"] - old.loc[persons.index, role + "_tail_logp"]
            ).max()
        )
        for role in ("actual", "matched")
    )
    if replay > 1e-5:
        raise ValueError("RC1 native tail scores failed independent replay")
    compare(result["effects"], effects)
    numerical = json.loads((run / "technical_numerical_gate.json").read_text())
    if len(numerical) != 2 or any(
        not x["passed"]
        or x["self_patch_max_logp_error"] != 0
        or x["compact_fixed_mean_logp_difference"] > 0.02
        or x["minimum_site_cosine"] < 0.999
        for x in numerical
    ):
        raise ValueError("Numerical preflight failed")
    tf = pd.read_csv(run / "technical_readout_metrics.csv").rename(
        columns={"language": "participant_code"}
    )
    tc = [
        {k if k != "language" else "participant_code": v for k, v in x.items()}
        for x in json.loads((run / "technical_readout_checks.json").read_text())
    ]
    if {x["participant_code"] for x in tc} != {"en", "ja"}:
        raise ValueError("Technical language coverage mismatch")
    audit_frame(tf, tc)
    if result["head_only_batches"] != sum(x["head_batches"] for x in [*tc, *checks]):
        raise ValueError("Head-only compute accounting mismatch")
    return {
        "status": "COMPLETE_READOUT_SCALAR_AND_MATERIAL_AUDIT",
        "parent_identity": m["identity"],
        "participants": 44,
        "tail_token_rows": len(frame),
        "forwards": 98,
        "head_only_batches": result["head_only_batches"],
        "algebra_replay_max_error": error,
        "rc1_tail_replay_max_error": replay,
        "effects": effects,
        "new_model_forwards": 0,
        "raw_text_reopened": False,
        "labels_read": False,
        "scope": "scalar identity/aggregation audit, not independent neural or full-vocabulary replay",
    }


def main():
    """Save an independent audit without overwriting the frozen result packet."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "parent-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(x for x in a.output_dir.iterdir() if not x.name.startswith("._")):
        raise ValueError("Fresh audit directory required")
    result = audit(a.run_dir, a.parent_dir)
    write_json(a.output_dir / "summary.json", result)
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for path in (
            Path(__file__),
            ROOT / "scripts/audit_grain_response_depth.py",
            ROOT / "tests/test_grain_response_readout_audit.py",
        ):
            z.write(path, path.relative_to(ROOT))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "participants",
                    "tail_token_rows",
                    "algebra_replay_max_error",
                    "rc1_tail_replay_max_error",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
