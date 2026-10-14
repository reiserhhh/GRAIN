#!/usr/bin/env python3
"""Bridge frozen SOURCE score actions to SAME cached framework interventions."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_anchor_carrier_contrasts as C
from grain_research.construction import math_diagnostic_action_bridge as M
from grain_research.construction import math_same_event_multitarget as N
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_diagnostic_action_bridge_hm214.json"
DEFAULT = ROOT / "results/grain_math_response_v1/diagnostic_action_bridge_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_DIAGNOSTIC_ACTION_BRIDGE_PLAN_HM214.md",
    "grain_research/construction/math_diagnostic_action_bridge.py",
    "scripts/run_grain_diagnostic_action_bridge_hm214.py",
    "scripts/score_grain_diagnostic_action_cold.py",
    "tests/test_grain_diagnostic_action_bridge.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "grain_research/construction/math_same_event_multitarget.py",
    "grain_research/construction/math_anchor_carrier_contrasts.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def arrays(path):
    """Load only declared numeric archives, without pickle or raw material."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def banks(path):
    """Recover the two pre-existing source instruments, not new fitted models."""
    z = arrays(path)
    return [{k[3:]: v for k, v in z.items() if k.startswith(f"b{b}_")} for b in (0, 1)]


def record(z, author, view):
    """Build an addressed vector-only edit; no current-author labels or responses."""
    uid = str(z["author_ids"][author])
    return {
        "frame_key": f"{uid}/view{view}/remove_selected_minus_remove_control",
        "windows": [
            [
                {
                    "event_id": f"{uid}:{v}:{t}",
                    "vector": z["panel_vectors"][author, v, t].tolist(),
                    "condition": int(z["panel_conditions"][author, v, t]),
                }
                for t in range(12)
            ]
            for v in (0, 1)
        ],
        "edit": {
            "view": view,
            "selected": [f"{uid}:{view}:{t}" for t in z["selected"][author, view]],
            "control": [f"{uid}:{view}:{t}" for t in z["control"][author, view]],
        },
    }


def prepare(out):
    """Freeze SAME source/event interfaces before new score-response contacts."""
    cfg = R.read(ROOT / CONFIG)
    sp, mp = (ROOT / cfg[k] for k in ("source_parent", "mechanism_parent"))
    R.check(sp, ("built_seal.json", "analyzed_seal.json", "verified_seal.json"))
    R.check(mp, ("computed_seal.json", "verification_seal.json"))
    z, a = arrays(sp / "observations.npz"), arrays(mp / "observations.npz")
    if any(not np.array_equal(z[k], a[k]) for k in ("source_means", "source_indices")):
        raise ValueError("Different SOURCE event family")
    if not np.array_equal(z["codes"], a["labels"] @ 2 ** np.arange(5)):
        raise ValueError("Different SOURCE target assignments")
    if set(a["source_indices"]) & set(a["panel_indices"]):
        raise ValueError("SOURCE/panel overlap")
    points = np.empty((24, 2, 2, 2, 1024))
    for u, v in np.ndindex(24, 2):
        points[u, v], _ = M.edit_points(record(z, u, v), 1024)
    dx = np.stack([points[:, v, 0, v] - points[:, v, 1, v] for v in (0, 1)], 1)
    error = float(np.max(abs(dx - a["vector_actions"])))
    if error > cfg["comparison_tolerance"]:
        raise ValueError("Native/source edits do not address the SAME vectors")
    files = [
        sp / "observations.npz",
        sp / "source_banks.npz",
        sp / "analysis.npz",
        sp / "manifest.json",
        sp / "verified_seal.json",
        mp / "observations.npz",
        mp / "manifest.json",
        mp / "verification_seal.json",
    ]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        **z,
        edited_points=points,
        response=a["response"],
        reference_fields=a["fields"],
        panel_indices=a["panel_indices"],
    )
    R.write(out / "cold_example.json", record(z, 0, 0))
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "source_banks": np.bincount(z["folds"]).tolist(),
            "panel_authors": 24,
            "edit_records": 48,
            "same_vector_action_error": error,
            "companion_full": True,
            "no_current_labels_or_native_answers_to_scorer": True,
            "new_neural_API_private_train_calls": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_example.json", "input_quality.json"),
    )
    print("PREPARED HM214", error, flush=True)


def number(value):
    """Undefined directions stay absent rather than becoming successful zeroes."""
    return float(value) if np.isfinite(value) else None


def cosine(a, b):
    """Keep centered action geometry distinct from prediction accuracy."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    x, y = x - x.mean(), y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return number(np.sum(x * y) / den) if den else None


def compute(out):
    """Evaluate one frozen source operation; compare all registered native modes."""
    cfg = R.check(out)["config"]
    z, start = R.prepared(out), time.monotonic()
    bs = banks(ROOT / cfg["source_parent"] / "source_banks.npz")
    source = np.empty((2, 24, 2, 3, 5))
    midpoint = np.empty((2, 24, 2, 5))
    for b, u, v in np.ndindex(2, 24, 2):
        r = M.cold(bs[b], record(z, u, v))
        source[b, u, v] = r["finite_actions"]
        midpoint[b, u, v] = r["single_midpoint_action"]
    contacts = np.empty((2, 3, 12, 3, 4, 3, 5, 5))
    rows, primary = [], []
    for b, law, channel, account, unit, view in np.ndindex(2, 3, 12, 3, 4, 3):
        s = source[b, :, view, law] if view < 2 else source[b, :, :, law].mean(1)
        r = (
            z["response"][:, view, channel, account, :, unit]
            if view < 2
            else z["response"][:, :, channel, account, :, unit].mean(1)
        )
        cov = N.covariances(s, r)
        contacts[b, law, channel, account, unit, view] = cov
        if view == 2:
            for q, reader in np.ndindex(5, 5):
                row = {
                    "source_bank": b,
                    "law": M.LAWS[law],
                    "channel": C.CHANNELS[channel],
                    "account": ("raw", "common", "specific")[account],
                    "unit": cfg["units"][unit],
                    "source_target": M.R.TARGETS[q],
                    "reader_target": M.R.TARGETS[reader],
                    "covariance": float(cov[q, reader]),
                }
                rows.append(row)
                if (
                    law == 0
                    and channel == 6
                    and account == 2
                    and unit == 2
                    and q == reader
                ):
                    primary.append(
                        row
                        | {
                            "action_cosine": cosine(s[:, q], r[:, reader]),
                            "descriptive_not_population_p": True,
                        }
                    )
    remainder = source[:, :, :, 0] - midpoint
    scale = np.sqrt(np.mean(source[:, :, :, 0] ** 2, axis=(1, 2)))
    nonlinear = np.sqrt(np.mean(remainder**2, axis=(1, 2)))
    summary = {
        "packet": "HM214",
        "primary": primary,
        "single_midpoint_relative_remainder_RMS": [
            [number(v) for v in row] for row in nonlinear / scale
        ],
        "cross_bank_single_action_cosine": [
            cosine(source[0, :, :, 0, q].mean(1), source[1, :, :, 0, q].mean(1))
            for q in range(5)
        ],
        "paired_vs_single_action_RMS": np.sqrt(
            np.mean((source[:, :, :, 1] - source[:, :, :, 0]) ** 2, axis=(1, 2))
        ).tolist(),
        "paired_vs_product_action_RMS": np.sqrt(
            np.mean((source[:, :, :, 1] - source[:, :, :, 2]) ** 2, axis=(1, 2))
        ).tolist(),
        "single_action_RMS": scale.tolist(),
        "new_capability": "cold_source_defined_finite_event_edit_actions_and_same_event_native_framework_contacts",
        "construction_inputs": "frozen_source_bank_and_addressed_vectors_ONLY",
        "native_companion_absent": True,
        "historical_exploratory_not_lockbox_or_accuracy": True,
        "original_scope_complete": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    if summary["elapsed_seconds"] > cfg["max_stage_seconds"]:
        R.write(out / "compute_failure.json", {"finite_cap": True, "no_retry": True})
        raise RuntimeError("Finite cap reached; preserve failure")
    R.archive(
        out / "analysis.npz",
        source_actions=source,
        midpoint_actions=midpoint,
        remainders=remainder,
        contacts=contacts,
    )
    R.table(out / "all_contacts.csv", rows)
    R.table(out / "primary.csv", primary)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        ("analysis.npz", "all_contacts.csv", "primary.csv", "summary.json"),
    )
    print("COMPUTED HM214", summary, flush=True)


def verify(out):
    """Check finite source actions using direct distances and inherited pair scores."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    z, a, start = R.prepared(out), arrays(out / "analysis.npz"), time.monotonic()
    bs = banks(ROOT / cfg["source_parent"] / "source_banks.npz")
    old = arrays(ROOT / cfg["source_parent"] / "analysis.npz")
    error, values = 0.0, 0
    for b, u, v in np.ndindex(2, 24, 2):
        literal = M.literal(bs[b], z["edited_points"][u, v], v)
        change = literal[0] - literal[1]
        error = max(error, float(np.max(abs(change - a["source_actions"][b, u, v]))))
        ss, cc = (1, 2) if v == 0 else (5, 6)
        prior = old[f"panel{b}_log_odds"].reshape(24, 9, 2, 5)
        error = max(error, float(np.max(abs(prior[u, ss] - prior[u, cc] - change[1:]))))
        values += change.size + 10
    for b, q in np.ndindex(2, 5):
        x = a["source_actions"][b, :, :, 0, q].mean(1)
        y = z["response"][:, :, 6, 2, q, 2].mean(1)
        value = sum(
            float(xx - x.mean()) * float(yy - y.mean())
            for xx, yy in zip(x, y, strict=True)
        ) / (len(x) - 1)
        error = max(error, abs(value - a["contacts"][b, 0, 6, 2, 2, 2, q, q]))
        values += 1
    if error > cfg["comparison_tolerance"] or not np.isfinite(error):
        R.write(
            out / "verification_failure.json", {"max_error": error, "no_retry": True}
        )
        raise ValueError("Source-action verification failed")
    receipt = {
        "PASS": True,
        "max_error": error,
        "literal_action_and_inherited_values": values,
        "same_agent_not_blind": True,
        "elapsed_seconds": time.monotonic() - start,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM214", receipt, flush=True)


def main():
    """Run one explicit immutable stage or a numeric-only cold read."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "compute", "verify", "cold", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage in ("check", "cold"):
        cfg = R.check(args.out, ("computed_seal.json", "verification_seal.json"))[
            "config"
        ]
        if args.stage == "cold":
            command = [
                sys.executable,
                str(ROOT / "scripts/score_grain_diagnostic_action_cold.py"),
                "--bank",
                str(ROOT / cfg["source_parent"] / "source_banks.npz"),
                "--input",
                str(args.out / "cold_example.json"),
            ]
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=30
            )
            result = json.loads(completed.stdout)
            expected = arrays(args.out / "analysis.npz")["source_actions"][0, 0, 0]
            error = float(np.max(abs(expected - result["finite_actions"])))
            if error > cfg["comparison_tolerance"]:
                raise ValueError("Cold process differs from frozen action")
            R.write(args.out / "cold_output.json", result)
            R.write(
                args.out / "cold_verification.json",
                {
                    "PASS": True,
                    "max_error": error,
                    "command": command,
                    "inputs": "source_banks.npz+cold_example.json ONLY;no native outputs/model/API",
                },
            )
            R.seal(
                args.out,
                "cold_seal.json",
                ("cold_output.json", "cold_verification.json"),
            )
            print("COLD HM214", result, "error", error)
        else:
            print("ALL SEALS PASS HM214")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
