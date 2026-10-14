#!/usr/bin/env python3
"""HM130: same-evidence reader-order triage, zero new inference or calibration."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_reading as J
from grain_research.construction import math_native_readout_location as N
from grain_research.construction import math_observer_order as M
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_joint_reading_hm106 as F
from scripts import run_grain_native_readout_location_hm108 as L

FIRST = ROOT / "results/grain_math_response_v1/joint_reading_law_r1"
SECOND = ROOT / "results/grain_math_response_v1/native_readout_location_r1"
CODE = (
    "docs/GRAIN_OBSERVER_ORDER_TRIAGE_PLAN_HM130.md",
    "grain_research/construction/math_observer_order.py",
    "scripts/run_grain_observer_order_hm130.py",
    "tests/test_grain_math_observer_order.py",
)
TOLERANCE = 1e-10


def table(path):
    """Read a public numeric table without reading or inferring text evidence."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def inherited():
    """Authenticate inherited code/captures and join complete baseline supports."""
    fm, fs = F.check(FIRST)
    nm = L.check(SECOND)
    H.verify_hashes(FIRST, H.load(FIRST / "output_seal.json"))
    H.verify_hashes(SECOND, H.load(SECOND / "output_seal.json"))
    targets = H.load(fs / "00/reader.json")["targets"]
    if targets != nm["spec"]["targets"] or len(targets) != 6:
        raise ValueError("Target order changed across inherited observations")
    with np.load(FIRST / "01/laws.npz", allow_pickle=False) as saved:
        labels = np.array(saved["labels"])
        coefficients = np.array(saved["coefficients"][:, 0])
    if labels.shape != (72, 4) or coefficients.shape != (72, 4, 6):
        raise ValueError("All 72 full corner supports are required")
    if fm["spec"]["weight_modes"][0] != "native":
        raise ValueError("HM106 mode order changed")
    coverage = table(SECOND / "coverage.csv")
    by_coordinate = {
        tuple(int(row[k]) for k in ("bank", "bridge", "carrier", "action")): row
        for row in coverage
    }
    if len(by_coordinate) != 72 or len(coverage) != 72:
        raise ValueError("Incomplete or duplicate native coordinate support")
    cases = []
    with np.load(SECOND / "logits.npz", allow_pickle=False) as saved:
        for index, label in enumerate(labels):
            key = tuple(int(x) for x in label)
            row = by_coordinate[key]
            if row["baseline"] != "True":
                raise ValueError("Do not impute missing native baseline")
            group = int(row["group"])
            native = np.stack(
                [
                    N.readout(saved[f"g{group}__baseline__q{t}__logits"])
                    for t in range(6)
                ],
                axis=1,
            )
            cases.append((group, key, J.reconstruct(coefficients[index]), native))
    return targets, cases


def summarize(records, pairs):
    """Count every signed/tied comparison; relative residual is descriptive only."""
    relative = [
        r["relative_residual"] for r in records if r["relative_residual"] is not None
    ]
    return {
        "cases": len(records),
        "event_pairs": len(pairs),
        "event_order": dict(Counter(r["classification"] for r in pairs)),
        "own_A_high_low": dict(Counter(r["own_A_order"] for r in records)),
        "fixed_margin_author_contrast": dict(
            Counter(r["fixed_margin_order"] for r in records)
        ),
        "positive_affine_supports": sum(
            r["positive_affine_on_support"] for r in records
        ),
        "positive_slope_supports": sum(
            r["slope"] is not None and r["slope"] > 0 for r in records
        ),
        "undefined_relative_residual": len(records) - len(relative),
        "relative_residual_median": float(np.median(relative)) if relative else None,
    }


def compute(out):
    """Save one exhaustive finite diagnostic without changing any inherited score."""
    if out.exists():
        raise FileExistsError("Preserve prior diagnostics and failures")
    started = time.monotonic()
    targets, cases = inherited()
    out.mkdir(parents=True)
    input_files = (
        FIRST / "00/manifest.json",
        FIRST / "01/laws.npz",
        FIRST / "01/characters.csv",
        FIRST / "output_seal.json",
        SECOND / "manifest.json",
        SECOND / "logits.npz",
        SECOND / "coverage.csv",
        SECOND / "characters.csv",
        SECOND / "output_seal.json",
    )
    H.write(
        out / "manifest.json",
        {
            "packet": "HM130",
            "classification": "EXPLORATORY_INSTRUMENT_TRIAGE_NOT_NEW_MECHANISM",
            "code": {name: H.sha(ROOT / name) for name in CODE},
            "inputs": {str(p.relative_to(ROOT)): H.sha(p) for p in input_files},
            "targets": targets,
            "tolerance": TOLERANCE,
            "new_forwards": 0,
            "new_predictor_fits": 0,
            "new_private_reads": 0,
            "preliminary_counts_seen": True,
            "program_difference": "F local constructed query/covector;N after PersonA target suffix",
            "prior_readonly_preview_error": "Metadata preview indexed a queries dict by a slice;KeyError after numeric counts;no files/model calls/results rewritten",
        },
    )
    records, pairs, laws = [], [], []
    for group, coordinates, first, second in cases:
        context = dict(
            zip(("bank", "bridge", "carrier", "action"), coordinates, strict=True)
        )
        for t, u in itertools.product(range(6), range(2)):
            f, n = first[:, t], second[:, t, u]
            base = {
                **context,
                "group": group,
                "target": targets[t],
                "unit": N.UNITS[u],
                "diagonal": t == context["action"],
            }
            fc, nc = M.fixed_margin_contrast(f), M.fixed_margin_contrast(n)
            records.append(
                {
                    **base,
                    **M.affine_certificate(f, n, TOLERANCE),
                    "first_own_A_contrast": float(f[0] - f[1]),
                    "second_own_A_contrast": float(n[0] - n[1]),
                    "own_A_order": M.contrast_class(
                        f[0] - f[1], n[0] - n[1], TOLERANCE
                    ),
                    "first_fixed_margin_contrast": fc,
                    "second_fixed_margin_contrast": nc,
                    "fixed_margin_order": M.contrast_class(fc, nc, TOLERANCE),
                    **{f"first_{i}": float(x) for i, x in enumerate(f)},
                    **{f"second_{i}": float(x) for i, x in enumerate(n)},
                }
            )
            pairs.extend({**base, **p} for p in M.pair_orders(f, n, TOLERANCE))
            for name, p in F.LAWS.items():
                laws.append(
                    {
                        **base,
                        "law": name,
                        "first": M.evidence_mean(f, p),
                        "second": M.evidence_mean(n, p),
                    }
                )
    H.table(out / "supports.csv", records)
    H.table(out / "event_pairs.csv", pairs)
    H.table(out / "author_laws.csv", laws)
    banks = {}
    for bank, unit in itertools.product(range(3), N.UNITS):
        selected = [r for r in records if r["bank"] == bank and r["unit"] == unit]
        chosen_pairs = [r for r in pairs if r["bank"] == bank and r["unit"] == unit]
        banks[f"bank{bank}__{unit}"] = {
            "all_action_target_cells": summarize(selected, chosen_pairs),
            "action_target_diagonal": summarize(
                [r for r in selected if r["diagonal"]],
                [r for r in chosen_pairs if r["diagonal"]],
            ),
        }
    summary = {
        "packet": "HM130",
        "classification": "INSTRUMENT_TRIAGE_NOT_THEORY_BREAKTHROUGH",
        "state": "COMPUTED",
        "groups": 72,
        "historical_corners": 288,
        "matched_native_baseline_fields": 432,
        "supports_including_two_native_units": len(records),
        "event_pairs": len(pairs),
        "author_law_observations": len(laws),
        "new_forwards": 0,
        "new_predictor_fits": 0,
        "api_private_training_calls": 0,
        "seconds": time.monotonic() - started,
        "banks": banks,
        "decision": "Do not calibrate mismatch away;connect different observation stages by declared causal/conditional operators,not a fidelity leaderboard",
        "original_scope_complete": False,
    }
    H.write(out / "summary.json", summary)
    outputs = ("supports.csv", "event_pairs.csv", "author_laws.csv", "summary.json")
    H.write(out / "output_seal.json", {p: H.sha(out / p) for p in outputs})
    print(
        "HM130 COMPUTED",
        {k: summary[k] for k in ("groups", "event_pairs", "new_forwards", "seconds")},
    )


def verify(out):
    """Recount all orders by explicit signs and compare original character tables."""
    manifest = H.load(out / "manifest.json")
    H.verify_hashes(ROOT, manifest["code"])
    H.verify_hashes(ROOT, manifest["inputs"])
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    targets, cases = inherited()
    old_f = {
        tuple(int(r[k]) for k in ("bank", "bridge", "carrier", "action"))
        + (r["character"], r["target"]): float(r["coefficient"])
        for r in table(FIRST / "01/characters.csv")
        if r["weight_mode"] == "native"
    }
    old_n = {
        (int(r["group"]), r["target"], r["unit"], r["character"]): float(r["native"])
        for r in table(SECOND / "characters.csv")
        if r["arm"] == "baseline"
    }
    saved = table(out / "supports.csv")
    saved_laws = table(out / "author_laws.csv")
    saved_pairs = table(out / "event_pairs.csv")
    errors = {
        "first_character": 0.0,
        "native_character": 0.0,
        "direct_author": 0.0,
        "saved_corner": 0.0,
        "fixed_margin": 0.0,
    }
    lookup = {(int(r["group"]), r["target"], r["unit"]): r for r in saved}
    law_lookup = {
        (int(r["group"]), r["target"], r["unit"], r["law"]): r for r in saved_laws
    }
    pair_lookup = {
        (int(r["group"]), r["target"], r["unit"], int(r["left"]), int(r["right"])): r
        for r in saved_pairs
    }
    checked_pairs = 0
    for group, coords, f, n in cases:
        for t, u in itertools.product(range(6), range(2)):
            target, unit = targets[t], N.UNITS[u]
            row = lookup[group, target, unit]
            for side, x in (("first", f[:, t]), ("second", n[:, t, u])):
                a = np.array(
                    [sum(J.BASIS[j, i] * x[i] for i in range(4)) / 4 for j in range(4)]
                )
                for j, char in enumerate(J.CHARACTERS):
                    reference = (
                        old_f[coords + (char, target)]
                        if side == "first"
                        else old_n[group, target, unit, char]
                    )
                    key = "first_character" if side == "first" else "native_character"
                    errors[key] = max(errors[key], abs(a[j] - reference))
                for i in range(4):
                    errors["saved_corner"] = max(
                        errors["saved_corner"], abs(x[i] - float(row[f"{side}_{i}"]))
                    )
                errors["fixed_margin"] = max(
                    errors["fixed_margin"],
                    abs(a[3] - float(row[f"{side}_fixed_margin_contrast"])),
                )
                for name, p in F.LAWS.items():
                    direct = sum(p[i] * x[i] for i in range(4))
                    moment = sum(
                        sum(J.BASIS[j, i] * p[i] for i in range(4)) * a[j]
                        for j in range(4)
                    )
                    persisted = float(law_lookup[group, target, unit, name][side])
                    errors["direct_author"] = max(
                        errors["direct_author"],
                        abs(direct - moment),
                        abs(direct - persisted),
                    )
            for i, j in itertools.combinations(range(4), 2):
                a, b = f[i, t] - f[j, t], n[i, t, u] - n[j, t, u]
                if abs(a) <= TOLERANCE and abs(b) <= TOLERANCE:
                    order = "both_tied"
                elif abs(a) <= TOLERANCE or abs(b) <= TOLERANCE:
                    order = "one_sided_tie"
                else:
                    order = "concordant" if (a > 0) == (b > 0) else "reversed"
                if pair_lookup[group, target, unit, i, j]["classification"] != order:
                    raise ValueError("Saved event order differs from independent signs")
                checked_pairs += 1
    if len(saved) != 864 or len(saved_pairs) != 5184 or len(saved_laws) != 3456:
        raise ValueError("The complete finite battery was not retained")
    if max(errors.values()) > TOLERANCE:
        raise ValueError(errors)
    summary = H.load(out / "summary.json")
    for bank, unit in itertools.product(range(3), N.UNITS):
        for diagonal, name in (
            (False, "all_action_target_cells"),
            (True, "action_target_diagonal"),
        ):
            eligible = [
                r
                for r in saved_pairs
                if int(r["bank"]) == bank
                and r["unit"] == unit
                and (not diagonal or r["diagonal"] == "True")
            ]
            actual = dict(Counter(r["classification"] for r in eligible))
            if summary["banks"][f"bank{bank}__{unit}"][name]["event_order"] != actual:
                raise ValueError("Saved summary and independent recount differ")
    verification = {
        "state": "VERIFIED_FINITE_TRIAGE_NOT_THEORY_CLOSURE",
        "actual_matched_fields": 432,
        "checked_event_pairs": checked_pairs,
        "errors": errors,
        "same_agent_alternate_formula_not_blind_review": True,
        "new_forwards": 0,
        "original_scope_complete": False,
    }
    H.write(out / "verification.json", verification)
    print("HM130 VERIFIED", verification)


def main():
    """Run a single sealed CPU diagnostic and its separate formula verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/observer_order_triage_r1",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if not args.verify_only:
        compute(out)
    verify(out)


if __name__ == "__main__":
    main()
