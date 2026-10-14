#!/usr/bin/env python3
"""One CPU-only same-event SOURCE/behavior action, preserving author association."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_event_behavior as M
from grain_research.construction import math_trait_anchor as A
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_event_behavior_hm262.json"
DEFAULT = ROOT / "results/grain_math_response_v1/event_behavior_correspondence_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVENT_BEHAVIOR_CORRESPONDENCE_PLAN_HM262.md",
    "grain_research/construction/math_event_behavior.py",
    "scripts/run_grain_event_behavior_hm262.py",
    "tests/test_grain_event_behavior.py",
    "grain_research/construction/math_trait_anchor.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
)


def components(law):
    """Keep the full source field and conditional choice/within without refitting."""
    A.verify_law(law)
    w = np.asarray(law["probabilities"]) - np.asarray(law["prior"])
    r = np.asarray(law["reference_mass"])
    mu = np.array([r[i : i + 4] @ w[i : i + 4] / r[i : i + 4].sum() for i in (0, 4)])
    choice = np.repeat(mu, 4, axis=0)
    return {"full": w, "choice": choice, "within": w - choice}


def prepare(out):
    """Bind only completed numerical observations and declared historical source laws."""
    if out.exists():
        raise FileExistsError("Do not replace a previous result or renew its budget")
    cfg = R.read(ROOT / CONFIG)
    parent, address, meaning = [
        ROOT / cfg[k] for k in ("behavior_parent", "address_parent", "meaning_parent")
    ]
    R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    S.parent_seal(
        address,
        "prepared_seal.json",
        ("source_observations.npz", "private_bindings.json"),
    )
    S.parent_seal(
        meaning, "compiled_seal.json", ("source_law.json", "permuted_law.json")
    )
    expression = (
        "[.[] | {unit_id,author,context_window,source_field_own_foreign_difference}]"
    )
    ledger = json.loads(
        subprocess.check_output(
            ["jq", expression, str(address / "private_bindings.json")], text=True
        )
    )
    cases = R.read(parent / "cases.json")
    if len(ledger) != 48 or len(cases) != 48:
        raise ValueError("All48 identified uses required; no partial rescue")
    for i, (row, case) in enumerate(zip(ledger, cases, strict=True)):
        if (row["unit_id"], row["author"], row["context_window"]) != (
            i,
            i % 24,
            i // 24,
        ):
            raise ValueError("Registered author/direction identities changed")
        if any(row[k] != case[k] for k in ("unit_id", "author", "context_window")):
            raise ValueError("Behavior and SOURCE actions are not the same unit")
    z = S.selected(parent / "source_observations.npz", ("fields", "choice"))
    old = S.selected(address / "source_observations.npz", ("fields", "choice"))
    if any(not np.array_equal(z[k], old[k]) for k in z):
        raise ValueError("Inherited author-level association changed")
    laws = [R.read(meaning / n) for n in ("source_law.json", "permuted_law.json")]
    if any(
        laws[0][k] != laws[1][k]
        for k in ("reference_mass", "coordinate_cutoffs", "traits", "prior")
    ):
        raise ValueError("Historical null must change labels, not evidence geometry")
    fields = [components(law) for law in laws]
    delta = np.array([r["source_field_own_foreign_difference"] for r in ledger])
    changes, positions, lifts = {}, {}, {}
    for name, field in zip(("actual", "historical_null"), fields, strict=True):
        changes[name], positions[name], lifts[name] = {}, {}, {}
        for part, w in field.items():
            changes[name][part], dc = M.support_transfer(
                delta, fields[0]["full"], w, "difference"
            )
            positions[name][part], pc = M.support_transfer(
                z["fields"], fields[0]["full"], w, "mean"
            )
            lifts[name][part] = {
                "delta_matching_pairs": dc,
                "position_matching_pairs": pc,
            }
    if not np.allclose(changes["actual"]["full"], delta, rtol=0, atol=1e-12):
        raise ValueError("Recorded actual SOURCE difference could not be reconstructed")
    for part in ("full", "choice"):
        if not np.allclose(
            positions["actual"][part],
            z["fields" if part == "full" else "choice"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("Source author field provenance mismatch")
    out.mkdir(parents=True, exist_ok=False)
    inputs = [
        parent / n
        for n in (
            "manifest.json",
            "analyzed_seal.json",
            "verification_seal.json",
            "cold_numeric.json",
            "source_observations.npz",
            "cases.json",
        )
    ]
    inputs += [
        address / n
        for n in (
            "prepared_seal.json",
            "private_bindings.json",
            "source_observations.npz",
        )
    ]
    inputs += [
        meaning / n
        for n in ("compiled_seal.json", "source_law.json", "permuted_law.json")
    ]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.write(out / "numeric_ledger.json", ledger)
    R.write(out / "finite_lifts.json", lifts)
    R.write(
        out / "source_actions.json",
        {
            "changes": {
                n: {k: v.tolist() for k, v in p.items()} for n, p in changes.items()
            },
            "positions": {
                n: {k: v.tolist() for k, v in p.items()} for n, p in positions.items()
            },
        },
    )
    R.write(out / "behavior.json", R.read(parent / "cold_numeric.json")["accounts"])
    R.seal(
        out,
        "prepared_seal.json",
        (
            "numeric_ledger.json",
            "finite_lifts.json",
            "source_actions.json",
            "behavior.json",
        ),
    )
    print("PREPARED_IDENTICAL_EVENT_ACTION", flush=True)


def run(out):
    """Retain both estimands and all target/condition/mean-product accounts once."""
    cfg = R.check(out)["config"]
    if (out / "summary.json").exists():
        raise FileExistsError("No scientific rerun or post-outcome selection")
    began = time.monotonic()
    data, behavior = R.read(out / "source_actions.json"), R.read(out / "behavior.json")
    records, matrices, intervals, zero, errors = [], {}, {}, [], []
    for law in data["changes"]:
        for part, values in data["changes"][law].items():
            x = np.array(values).reshape(2, 24, 5)
            s = np.array(data["positions"][law][part]).reshape(2, 24, 5)
            for scale, accounts in behavior.items():
                for account in ("balanced_key", "common_support", "equal_item_key"):
                    y = np.array(accounts[account]).reshape(2, 24, 5)
                    key = f"{law}/{part}/{scale}/{account}"
                    intervals[key] = M.paired_intervals(
                        x, y, cfg["bootstrap_seed"], cfg["bootstrap_draws"]
                    ).tolist()
                    for direction in ("pooled", 0, 1):
                        xx, yy, ss = (
                            (x.reshape(48, 5), y.reshape(48, 5), s.mean(0))
                            if direction == "pooled"
                            else (x[direction], y[direction], s[direction])
                        )
                        c = M.contacts(xx, yy)
                        position = M.contacts(
                            ss, y.mean(0) if direction == "pooled" else yy
                        )["covariance"]
                        matrices[f"{key}/{direction}"] = {
                            **{k: v.tolist() for k, v in c.items()},
                            "position_association": position.tolist(),
                        }
                        errors.append(
                            float(
                                np.max(abs(c["action"] - c["covariance"] - c["means"]))
                            )
                        )
                        errors.append(
                            float(
                                np.max(
                                    abs(M.contacts(-xx, -yy)["action"] - c["action"])
                                )
                            )
                        )
                        for target in range(5):
                            records.append(
                                {
                                    "law": law,
                                    "part": part,
                                    "scale": scale,
                                    "account": account,
                                    "direction": direction,
                                    "target": target,
                                    **{
                                        k: float(v[target, target])
                                        for k, v in c.items()
                                    },
                                    "position_association": float(
                                        position[target, target]
                                    ),
                                }
                            )
    x = np.array(data["changes"]["actual"]["full"])
    y = np.array(behavior["sum"]["balanced_key"])
    mask = np.max(abs(x), axis=1) <= 1e-12
    for unit in np.flatnonzero(mask):
        zero.append(
            {
                "unit_id": int(unit),
                "author": int(unit % 24),
                "direction": int(unit // 24),
                "source_delta": x[unit].tolist(),
                "behavior_B_delta": y[unit].tolist(),
            }
        )
    primary = "actual/full/sum/balanced_key"
    null = "historical_null/full/sum/balanced_key"
    summary = {
        "authors": 24,
        "dependent_uses": 48,
        "primary_D": np.diag(matrices[primary + "/pooled"]["action"]).tolist(),
        "primary_marginal_intervals": intervals[primary],
        "old_A_kept": np.diag(
            matrices[primary + "/pooled"]["position_association"]
        ).tolist(),
        "historical_null_D": np.diag(matrices[null + "/pooled"]["action"]).tolist(),
        "source_zero_uses": int(mask.sum()),
        "source_zero_behavior_RMS": np.sqrt(np.mean(y[mask] ** 2, axis=0)).tolist()
        if mask.any()
        else None,
        "algebra_error": max(errors),
        "cpu_scientific_seconds": time.monotonic() - began,
        "new_model_API_private_training_calls": 0,
        "not_accuracy_or_human_causal_trait": True,
        "one_historical_null_not_permutation_pvalue": True,
        "original_scope_complete": False,
    }
    if (
        summary["cpu_scientific_seconds"] > cfg["max_cpu_seconds"]
        or max(errors) > 1e-10
    ):
        raise RuntimeError("Finite cap or identity failed; no rescue")
    R.write(out / "contacts.json", matrices)
    R.write(
        out / "intervals.json",
        {"marginal_source_conditional_not_familywise": True, "values": intervals},
    )
    R.write(out / "source_zero_actions.json", zero)
    R.table(out / "contacts.csv", records)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "contacts.json",
            "intervals.json",
            "source_zero_actions.json",
            "contacts.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary), flush=True)


def verify(out):
    """Replay every signed contact by scalar arithmetic, without a model or refit."""
    R.check(out, ("analyzed_seal.json",))
    data, behavior, saved = [
        R.read(out / n)
        for n in ("source_actions.json", "behavior.json", "contacts.json")
    ]
    error, checks = 0.0, 0
    for key, expected in saved.items():
        law, part, scale, account, direction = key.split("/")
        x, s = [
            np.asarray(data[k][law][part]).reshape(2, 24, 5)
            for k in ("changes", "positions")
        ]
        y = np.asarray(behavior[scale][account]).reshape(2, 24, 5)
        x, y, s, yp = (
            (x.reshape(48, 5), y.reshape(48, 5), s.mean(0), y.mean(0))
            if direction == "pooled"
            else (
                x[int(direction)],
                y[int(direction)],
                s[int(direction)],
                y[int(direction)],
            )
        )
        for p in range(5):
            for q in range(5):
                mx, my = float(x[:, p].mean()), float(y[:, q].mean())
                ms, mp = float(s[:, p].mean()), float(yp[:, q].mean())
                literal = {
                    "action": sum(
                        float(a) * float(b)
                        for a, b in zip(x[:, p], y[:, q], strict=True)
                    )
                    / len(x),
                    "covariance": sum(
                        (float(a) - mx) * (float(b) - my)
                        for a, b in zip(x[:, p], y[:, q], strict=True)
                    )
                    / len(x),
                    "means": mx * my,
                    "position_association": sum(
                        (float(a) - ms) * (float(b) - mp)
                        for a, b in zip(s[:, p], yp[:, q], strict=True)
                    )
                    / len(s),
                }
                for name, value in literal.items():
                    error = max(error, abs(value - expected[name][p][q]))
                    checks += 1
    if error > 1e-10:
        raise ArithmeticError("Signed contact replay failed; no precision grid")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "literal_contacts": checks,
            "max_error": error,
            "same_agent_not_blind": True,
            "new_model_calls": 0,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED_IDENTICAL_EVENT_ACTION", flush=True)


def main():
    """Expose one bounded numeric experiment, never a native model continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("analyzed_seal.json", "verification_seal.json"))
        print("ALL_SEALS_PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
