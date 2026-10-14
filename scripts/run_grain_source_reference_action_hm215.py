#!/usr/bin/env python3
"""Compare baseline-matched reference transport with new-evidence action drift."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_source_reference_action as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_source_reference_action_hm215.json"
DEFAULT = ROOT / "results/grain_math_response_v1/source_reference_action_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SOURCE_REFERENCE_ACTION_PLAN_HM215.md",
    "grain_research/construction/math_source_reference_action.py",
    "scripts/run_grain_source_reference_action_hm215.py",
    "scripts/score_grain_source_reference_cold.py",
    "tests/test_grain_source_reference_action.py",
    "grain_research/construction/math_diagnostic_action_bridge.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "scripts/run_grain_observed_family_hm170.py",
)
KEYS = (
    "source_action",
    "destination_action",
    "transport_action",
    "calibration_action",
    "response_defect",
    "defect_identity",
)


def select(path, keys):
    """Read only registered numerical arrays, never native outputs/current labels."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def banks(path):
    """Load the two historical SOURCE instruments; all32 profiles remain supported."""
    with np.load(path, allow_pickle=False) as z:
        return [
            {k[3:]: z[k] for k in z.files if k.startswith(f"b{b}_")} for b in (0, 1)
        ]


def prepare(out):
    """Seal SAME evidence and SOURCE rules before any new reference comparison."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent, ("computed_seal.json", "verification_seal.json", "cold_seal.json"))
    points = select(parent / "observations.npz", ("edited_points",))["edited_points"]
    source = select(parent / "analysis.npz", ("source_actions",))["source_actions"]
    bs = banks(ROOT / cfg["banks"])
    if points.shape != (24, 2, 2, 2, 1024) or source.shape != (2, 24, 2, 3, 5):
        raise ValueError("Changed SAME-event protocol")
    if float(bs[0]["variance"]) != float(bs[1]["variance"]):
        raise ValueError("Reference change has a different observation channel")
    inputs = [
        parent / p
        for p in (
            "observations.npz",
            "analysis.npz",
            "manifest.json",
            "cold_example.json",
        )
    ] + [ROOT / cfg["banks"]]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "native_or_application_labels_read": False,
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", points=points, old_source_actions=source)
    R.write(out / "cold_example.json", R.read(parent / "cold_example.json"))
    R.seal(out, "prepared_seal.json", ("observations.npz", "cold_example.json"))
    print("PREPARED HM215: same48edits,all32profiles,bothreferences", flush=True)


def rms(a):
    """Return the RMS in the declared action units, not a psychological accuracy."""
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def compute(out):
    """Compute all fixed reference transports without choosing a favorable rule."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    bs, start = banks(ROOT / cfg["banks"]), time.monotonic()
    a = {k: np.empty((2, 3, 24, 2, 5)) for k in KEYS}
    a["conditional_cross_ratio"] = np.empty((2, 3, 24, 2, 5, 16))
    a["profile_relative_eta"] = np.empty((2, 3, 24, 2, 32))
    control_error = telescope_error = inherited_error = 0.0
    for u, v in np.ndindex(24, 2):
        logs = [M.logs(b, z["points"][u, v], v) for b in bs]
        for direction, (source, destination) in enumerate(cfg["directions"]):
            c = M.certificate(logs[source], logs[destination])
            for k, values in a.items():
                values[direction, :, u, v] = c[k]
            control_error = max(control_error, c["control_profile_error"])
            telescope_error = max(telescope_error, c["telescope_error"])
            inherited_error = max(
                inherited_error,
                float(
                    np.max(
                        abs(c["source_action"] - z["old_source_actions"][source, u, v])
                    )
                ),
            )
    rows, individual = [], []
    for direction, law, q in np.ndindex(2, 3, 5):
        source, destination = cfg["directions"][direction]
        gap = (a["destination_action"] - a["source_action"])[
            direction, law, :, :, q
        ].mean(1)
        cal = a["calibration_action"][direction, law, :, :, q].mean(1)
        defect = a["response_defect"][direction, law, :, :, q].mean(1)
        kappa = a["conditional_cross_ratio"][direction, law, :, :, q]
        row = {
            "source": source,
            "destination": destination,
            "law": cfg["laws"][law],
            "target": cfg["targets"][q],
            "bank_action_gap_RMS": rms(gap),
            "calibration_action_RMS": rms(cal),
            "response_defect_RMS": rms(defect),
            "conditional_cross_ratio_RMS": rms(kappa),
            "signed_cross_account": float(2 * np.mean(cal * defect)),
            "gap_mean_square": float(np.mean(gap**2)),
            "calibration_mean_square": float(np.mean(cal**2)),
            "response_mean_square": float(np.mean(defect**2)),
            "defect_over_gap_RMS": rms(defect) / rms(gap) if rms(gap) else None,
            "defect_mean": float(defect.mean()),
            "defect_positive_authors": int(np.sum(defect > 0)),
            "defect_negative_authors": int(np.sum(defect < 0)),
            "descriptive_not_independent_replications": True,
        }
        rows.append(row)
        for u, v in np.ndindex(24, 2):
            individual.append(
                {
                    "source": source,
                    "destination": destination,
                    "law": cfg["laws"][law],
                    "target": cfg["targets"][q],
                    "author_slot": u,
                    "view": v,
                    **{k: float(a[k][direction, law, u, v, q]) for k in KEYS},
                }
            )
    identity_error = float(np.max(abs(a["response_defect"] - a["defect_identity"])))
    if (
        max(control_error, telescope_error, inherited_error, identity_error)
        > cfg["comparison_tolerance"]
    ):
        R.write(out / "compute_failure.json", {"no_retry": True})
        raise ValueError("Reference construction identity failed; preserve outcome")
    summary = {
        "packet": "HM215",
        "primary": [r for r in rows if r["law"] == cfg["laws"][0]],
        "control_profile_max_error": control_error,
        "telescope_max_error": telescope_error,
        "inherited_source_action_max_error": inherited_error,
        "defect_identity_max_error": identity_error,
        "source_authors": [203, 183],
        "panel_authors": 24,
        "observed_edits": 48,
        "all_laws_directions_retained": True,
        "original_scope_complete": False,
        "new_capability": "fullprofile_control_matched_source_reference_transport_and_conditional_evidence_drift_certificate",
        "new_native_fit_API_private_training_calls": 0,
        "elapsed_seconds": time.monotonic() - start,
    }
    if summary["elapsed_seconds"] > cfg["max_stage_seconds"]:
        R.write(out / "compute_failure.json", {"finite_cap": True, "no_retry": True})
        raise RuntimeError("Finite cap reached; no retry")
    R.archive(out / "analysis.npz", **a)
    R.table(out / "reference_accounts.csv", rows)
    R.table(out / "individual_actions.csv", individual)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "analysis.npz",
            "reference_accounts.csv",
            "individual_actions.csv",
            "summary.json",
        ),
    )
    print("COMPUTED HM215", json.dumps(summary), flush=True)


def verify(out):
    """Replay literal positive likelihood ratios and every signed aggregate account."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    z = R.prepared(out)
    a = select(out / "analysis.npz", KEYS + ("conditional_cross_ratio",))
    bs, start = banks(ROOT / cfg["banks"]), time.monotonic()
    error, values = 0.0, 0
    for u, v in np.ndindex(24, 2):
        laws = [M.literal_laws(b, z["points"][u, v], v) for b in bs]
        for direction, (source, destination) in enumerate(cfg["directions"]):
            lit = M.literal_certificate(laws[source], laws[destination])
            for k, value in lit.items():
                error = max(error, float(np.max(abs(a[k][direction, :, u, v] - value))))
                values += value.size
    summary = R.read(out / "summary.json")
    for row in summary["primary"]:
        total = (
            row["calibration_mean_square"]
            + row["response_mean_square"]
            + row["signed_cross_account"]
        )
        error = max(error, abs(total - row["gap_mean_square"]))
        values += 1
    if error > cfg["comparison_tolerance"] or not np.isfinite(error):
        R.write(
            out / "verification_failure.json", {"max_error": error, "no_retry": True}
        )
        raise ValueError("Reference certificate failed literal audit")
    receipt = {
        "PASS": True,
        "max_error": error,
        "literal_values": values,
        "law_direction_edit_cases": 288,
        "same_agent_not_blind": True,
        "elapsed_seconds": time.monotonic() - start,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM215", receipt, flush=True)


def cold(out):
    """Execute both reference directions from SOURCE banks and addressed vectors only."""
    cfg = R.check(out, ("computed_seal.json", "verification_seal.json"))["config"]
    a = select(out / "analysis.npz", KEYS + ("conditional_cross_ratio",))
    receipts, error = [], 0.0
    for source in (0, 1):
        command = [
            sys.executable,
            str(ROOT / "scripts/score_grain_source_reference_cold.py"),
            "--banks",
            str(ROOT / cfg["banks"]),
            "--input",
            str(out / "cold_example.json"),
            "--source",
            str(source),
        ]
        r = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
        value = json.loads(r.stdout)
        for k in a:
            error = max(
                error, float(np.max(abs(np.array(value[k]) - a[k][source, :, 0, 0])))
            )
        receipts.append({"command": command, "output": value})
    if error > cfg["comparison_tolerance"]:
        raise ValueError("Cold certificate differs; do not retry")
    R.write(out / "cold_output.json", receipts)
    R.write(
        out / "cold_verification.json",
        {
            "PASS": True,
            "max_error": error,
            "inputs": "SOURCEbanks+addressedvectorrecord ONLY;no native/currentlabels/model",
        },
    )
    R.seal(out, "cold_seal.json", ("cold_output.json", "cold_verification.json"))
    print("COLD HM215 both directions", error, flush=True)


def main():
    """Run one explicit write-once stage; check never repeats scientific computation."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compute", "verify", "cold", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "check":
        R.check(
            args.out, ("computed_seal.json", "verification_seal.json", "cold_seal.json")
        )
        print("ALL HM215 SCIENCE/OUTPUT SEALS PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
