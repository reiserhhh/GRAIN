#!/usr/bin/env python3
"""Result-after HM74 mechanism interpretation; no inference, fitting or selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/grain_math_response_v1/causal_ownership_sectors_r1"
CHARACTERS = ("constant", "g", "y", "yg", "x", "xg", "xy", "xyg")
CORNERS = np.asarray(list(itertools.product((-1, 1), repeat=3)), float)
SIGNS = np.asarray(
    [
        np.prod(CORNERS[:, indexes], axis=1)
        for indexes in ((), (2,), (1,), (1, 2), (0,), (0, 2), (0, 1), (0, 1, 2))
    ]
)


def digest(path):
    """Hash a numerical source or companion file without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    """Read bounded structured metadata, not private text or model weights."""
    return json.loads(path.read_text())


def coefficient_table(values):
    """Calculate all eight finite coefficients without fitting a regression."""
    a = np.asarray(values, float)
    if a.shape[0] != 8 or not np.isfinite(a).all():
        raise ValueError("Complete finite cube required")
    return np.asarray([np.tensordot(s, a, axes=(0, 0)) / 8 for s in SIGNS])


def actions(values):
    """Literal low/high four-cell contrasts and their equal-x average."""
    a = np.asarray(values, float)
    low = a[3] - a[1] - a[2] + a[0]
    high = a[7] - a[5] - a[6] + a[4]
    return np.asarray([(low + high) / 2, low, high])


def category(group):
    """Keep content-matched, focal and neutral observations distinct."""
    if group["companion"] == "neutral":
        return "neutral"
    return "matched" if group["query"] == group["companion"] else "focal"


def rms(value):
    """Root mean squared finite magnitude, not psychological signal strength."""
    return float(np.sqrt(np.mean(np.square(value))))


def scalar_account(native, observed):
    """Describe signed and magnitude changes without unstable casewise ratios."""
    a, b = np.asarray(native, float), np.asarray(observed, float)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Paired finite observations required")
    denom = rms(a)
    ratio = rms(b) / denom if denom > 0 else None
    return {
        "n": len(a),
        "native_mean": float(a.mean()),
        "observed_mean": float(b.mean()),
        "change_mean": float((b - a).mean()),
        "native_rms": denom,
        "observed_rms": rms(b),
        "change_rms": rms(b - a),
        "remaining_rms_ratio": ratio,
        "rms_reduction_fraction": None if ratio is None else 1 - ratio,
        "native_positive_count": int(np.sum(a > 0)),
        "observed_positive_count": int(np.sum(b > 0)),
        "absolute_decrease_count": int(np.sum(abs(b) < abs(a))),
        "negative_change_count": int(np.sum(b - a < 0)),
    }


def check_profiles(record, value):
    """Replay both probability vectors, lower marginals and full score contrasts."""
    j = CHARACTERS.index(record["character"])
    plus, minus = (
        np.asarray(json.loads(record[k]), float)
        for k in ("plus_probability", "minus_probability")
    )
    expected = [(1 + sign * 0.5 * SIGNS[j]) / 8 for sign in (1, -1)]
    if any(
        not np.array_equal(a, b) for a, b in zip((plus, minus), expected, strict=True)
    ):
        raise ValueError("Undeclared author profile")
    if min(plus.min(), minus.min()) <= 0 or plus.sum() != 1 or minus.sum() != 1:
        raise ValueError("Invalid probability population")
    orders = (1,) if j == 3 else (1, 2)
    for order in orders:
        for indexes in itertools.combinations(range(3), order):
            for cell in itertools.product((-1, 1), repeat=order):
                mask = np.all(CORNERS[:, indexes] == cell, axis=1)
                if abs(plus[mask].sum() - minus[mask].sum()) > 1e-12:
                    raise ValueError("Claimed matched marginals differ")
    outputs = [np.tensordot(p, value, axes=(0, 0)) for p in (plus, minus)]
    target = ("plus_score", "minus_score", "difference")
    actual = (*outputs, outputs[0] - outputs[1])
    error = max(
        float(np.max(abs(a - np.asarray(json.loads(record[k])))))
        for a, k in zip(actual, target, strict=True)
    )
    if error > 1e-11:
        raise ValueError("Author composition score mismatch")
    return error


def write_csv(path, rows):
    """Persist complete finite accounts without choosing the best target."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interpret(source, destination):
    """Inspect every acquired table, arm, context and output sector once."""
    if destination.exists():
        raise FileExistsError("Preserve prior interpretation")
    verification = read(source / "verification.json")
    if verification["status"] != "VERIFIED_FINITE_HM74_NOT_GLOBAL":
        raise ValueError("Primary numerical verification required")
    for name, sha in read(source / "03/output_seal.json").items():
        if digest(source / name) != sha:
            raise ValueError("Primary frozen analysis drift")
    cubes, order = read(source / "00/cubes.json"), read(source / "03/tables.json")
    arms, stages = order["arm_order"], order["stage_order"]
    with np.load(source / "03/tables.npz") as archive:
        tables = {k: archive[k] for k in archive.files}
    if len(cubes) != 66 or len(tables) != 198 or len(arms) != 8:
        raise ValueError("Complete declared scope required")
    lookup = {(g["focal"], g["companion"], g["query"]): g["cube"] for g in cubes}
    accounts, spectra, pair_accounts, joint = [], [], [], []
    for stage in stages:
        for cat in ("matched", "focal", "neutral"):
            groups = [g for g in cubes if category(g) == cat]
            native = [tables[f"cube_{g['cube']:03d}_{stage}"][0] for g in groups]
            for index, arm in enumerate(arms):
                observed = [
                    tables[f"cube_{g['cube']:03d}_{stage}"][index] for g in groups
                ]
                if stage != "centered_logits":
                    for k, context in enumerate(("marginal", "x_low", "x_high")):
                        accounts.append(
                            {
                                "stage": stage,
                                "category": cat,
                                "arm": arm,
                                "context": context,
                                **scalar_account(
                                    [actions(v)[k] for v in native],
                                    [actions(v)[k] for v in observed],
                                ),
                            }
                        )
                for j, character in enumerate(CHARACTERS):
                    spectra.append(
                        {
                            "stage": stage,
                            "category": cat,
                            "arm": arm,
                            "character": character,
                            **scalar_account(
                                [coefficient_table(v)[j] for v in native],
                                [coefficient_table(v)[j] for v in observed],
                            ),
                        }
                    )
            interactions = [
                tables[f"cube_{g['cube']:03d}_{stage}"][0]
                - tables[f"cube_{g['cube']:03d}_{stage}"][1]
                - tables[f"cube_{g['cube']:03d}_{stage}"][2]
                + tables[f"cube_{g['cube']:03d}_{stage}"][3]
                for g in groups
            ]
            losses = [
                v - tables[f"cube_{g['cube']:03d}_{stage}"][3]
                for v, g in zip(native, groups, strict=True)
            ]
            joint.append(
                {
                    "stage": stage,
                    "category": cat,
                    "n": len(groups),
                    "interaction_rms": rms(interactions),
                    "mean_cube_interaction_rms": float(
                        np.mean([rms(v) for v in interactions])
                    ),
                    "maximum_cube_interaction_rms": max(rms(v) for v in interactions),
                    "joint_loss_rms": rms(losses),
                    "interaction_to_joint_loss_rms": rms(interactions) / rms(losses)
                    if rms(losses)
                    else None,
                }
            )
        if stage == "centered_logits":
            continue
        matched = [g for g in cubes if category(g) == "matched"]
        for trait in ("all_six", *dict.fromkeys(g["companion"] for g in matched)):
            groups = [
                g for g in matched if trait == "all_six" or g["companion"] == trait
            ]
            for index, arm in enumerate(arms):
                for k, context in enumerate(("marginal", "x_low", "x_high")):
                    targets, others, original, orig_other = [], [], [], []
                    for g in groups:
                        a = tables[f"cube_{g['cube']:03d}_{stage}"]
                        b = tables[
                            f"cube_{lookup[g['focal'], g['companion'], g['focal']]:03d}_{stage}"
                        ]
                        targets.append(actions(a[index])[k])
                        others.append(actions(b[index])[k])
                        original.append(actions(a[0])[k])
                        orig_other.append(actions(b[0])[k])
                    pair_accounts.append(
                        {
                            "stage": stage,
                            "companion": trait,
                            "arm": arm,
                            "context": context,
                            "matched_mean": float(np.mean(targets)),
                            "focal_mean": float(np.mean(others)),
                            **scalar_account(
                                np.asarray(original) - orig_other,
                                np.asarray(targets) - others,
                            ),
                        }
                    )
    primary = read(source / "03/summary.json")
    max_error = 0.0
    for row in primary["aggregate"]:
        companion = next(
            v
            for v in accounts
            if all(v[k] == row[k] for k in ("stage", "arm", "context", "category"))
        )
        pairs = (
            (row["n"], companion["n"]),
            (row["action_change_mean"], companion["change_mean"]),
            (row["action_change_rms"], companion["change_rms"]),
            (row["negative_change_count"], companion["negative_change_count"]),
        )
        max_error = max(max_error, *(abs(a - b) for a, b in pairs))
    if max_error > 1e-11 or len(primary["aggregate"]) != len(accounts):
        raise ValueError("Primary summary does not match the full tables")
    max_profile_error, profile_count = 0.0, 0
    with (source / "03/composition_consequences.csv").open() as stream:
        for row in csv.DictReader(stream):
            value = tables[f"cube_{int(row['cube']):03d}_{row['stage']}"][
                arms.index(row["arm"])
            ]
            max_profile_error = max(max_profile_error, check_profiles(row, value))
            profile_count += 1
    if profile_count != 3168:
        raise ValueError("Missing full composition accounting")
    destination.mkdir()
    outputs = {
        "aggregate_actions.csv": accounts,
        "all_sector_consequences.csv": spectra,
        "paired_target_accounts.csv": pair_accounts,
        "joint_operation_accounts.csv": joint,
    }
    for name, rows in outputs.items():
        write_csv(destination / name, rows)
    report = {
        "packet": "HM74_result_after_interpretation",
        "source": str(source),
        "actual_new_model_calls": 0,
        "fits_or_model_selection": 0,
        "primary_summary_max_error": max_error,
        "all_profile_rows_checked": profile_count,
        "profile_vector_marginal_score_max_error": max_profile_error,
        "outputs": {k: len(v) for k, v in outputs.items()},
        "n_is_finite_cubes_not_independent_humans": True,
        "remaining_ratio_is_effect_ratio_not_personality_information": True,
        "input_seals": {
            str(p.relative_to(source)): digest(p)
            for p in (
                source / "verification.json",
                source / "03/output_seal.json",
                source / "03/tables.npz",
                source / "00/cubes.json",
            )
        },
        "companion_code_sha256": digest(Path(__file__)),
    }
    (destination / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "output_seal.json").write_text(
        json.dumps(
            {p.name: digest(p) for p in destination.iterdir() if p.is_file()}, indent=2
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))


def main():
    """Replay completed HM74 science; never start a model or overwrite a result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    args = parser.parse_args()
    interpret(args.source, args.source / "04_interpretation")


if __name__ == "__main__":
    main()
