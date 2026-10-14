#!/usr/bin/env python3
"""Alternate all-score/cut-assignment/four-cell verification with no neural calls."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_cut import OUT, check, write
from scripts.verify_grain_math_evaluative_routing import native_score


def csv_rows(path):
    """Read all actual rows, not a completion claim inferred from a filename."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify(out):
    """Reconstruct384scores,336assignments and48interactions independently."""
    m, status = check(out), read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 384
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
        or status["model_seconds"] > 600
    ):
        raise ValueError("Require384 bounded actual calls")
    for ledger in ("03/artifact_hashes.json", "01/source_barrier.json"):
        hashes = read_json(out / ledger)
        hashes = hashes["files"] if ledger.startswith("01/") else hashes
        for name, sha in hashes.items():
            if digest(out / name) != sha:
                raise ValueError("Changed actual output " + name)
    barrier = read_json(out / "01/source_barrier.json")
    if barrier["source_calls"] != 192 or barrier["actual_calls_at_seal"] != 192:
        raise ValueError("Source192 must precede application192")
    design = read_json(out / "00/design.json")
    rows = {r["unit"]: r for r in read_json(out / "00/receivers.json")}
    captures, scores, masses, errors = {}, {}, {}, []
    for e in design:
        meta, arrays = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        if any(
            meta[k] != e[k]
            for k in ("unit", "pair", "role", "kind", "arm", "receiver", "donor")
        ):
            raise ValueError("Wrong actual call identity")
        y, mass = native_score(meta["native"])
        errors.extend(
            (
                abs(y - meta["native"]["conditional_score"]),
                abs(mass - meta["native"]["candidate_mass"]),
            )
        )
        scores[e["unit"]], masses[e["unit"]] = y, mass
        if e["kind"] == "capture":
            if arrays["full"].shape != (len(rows[e["receiver"]]["input_ids"]), 1536):
                raise ValueError("Not a full fixed-observer state")
            captures[e["receiver"]] = (e, arrays["full"], y)
    expected, by_pair, groups, content_groups = (
        {},
        defaultdict(dict),
        defaultdict(list),
        defaultdict(list),
    )
    max_pre_difference, max_random_relative_error = 0.0, 0.0
    for e in design:
        a, b, proof = rows[e["receiver"]], rows[e["donor"]], e["partition"]
        changed = [
            i
            for i in range(len(a["input_ids"]))
            if a["input_ids"][i] != b["input_ids"][i]
        ]
        if changed != proof["names"] or changed != proof["changed_input_positions"]:
            raise ValueError("Name exchange is not the only actual input difference")
        if proof["prefix"] + proof["body"] + proof["suffix"] != list(
            range(proof["length"])
        ):
            raise ValueError("Incomplete full cut")
        if e["direction"] != e["pair"] % 2:
            raise ValueError("Random-control direction contract drift")
        if e["kind"] == "capture":
            continue
        meta, arrays = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        current, donor = captures[e["receiver"]][1], captures[e["donor"]][1]
        arm = e["arm"]
        if arm == "names_properties":
            p = sorted(proof["names"] + proof["properties"])
        elif arm in ("full", "noop"):
            p = proof["body"] + proof["suffix"]
        else:
            p = proof["names"] if arm == "random_names" else proof[arm]
        if list(arrays["patch_positions"]) != p:
            raise ValueError("Actual patch positions differ")
        if arm == "noop":
            values = current[p]
        elif arm == "random_names":
            v = np.random.default_rng(m["spec"]["random_seed"] + e["pair"] // 2).normal(
                size=current[p].shape
            )
            norms = np.sqrt(
                np.sum((donor[p].astype(float) - current[p].astype(float)) ** 2, axis=1)
            )
            v = v * (norms / np.sqrt(np.sum(v * v, axis=1)))[:, None]
            values = (
                current[p].astype(float) + (1 if e["direction"] == 0 else -1) * v
            ).astype(current.dtype)
            actual_norm = np.sqrt(
                np.sum((values.astype(float) - current[p].astype(float)) ** 2, axis=1)
            )
            relative = np.max(abs(actual_norm - norms) / np.maximum(norms, 1e-20))
            max_random_relative_error = max(max_random_relative_error, float(relative))
        else:
            values = donor[p]
        if not np.array_equal(values, arrays["patch_values"]):
            raise ValueError("Replacement values do not match actual donor/control")
        expected_full = arrays["before"].copy()
        expected_full[p] = values
        if not np.array_equal(expected_full, arrays["full"]):
            raise ValueError("Changed unselected position or wrong selected assignment")
        max_pre_difference = max(
            max_pre_difference,
            float(np.max(abs(arrays["before"].astype(float) - current.astype(float)))),
        )
        y, native, target = (
            scores[e["unit"]],
            captures[e["receiver"]][2],
            captures[e["donor"]][2],
        )
        full, change = target - native, y - native
        expected[e["unit"]] = {
            "native": native,
            "donor": target,
            "patched": y,
            "candidate_mass": masses[e["unit"]],
            "full_change": full,
            "patch_change": change,
            "remaining": target - y,
        }
        groups[e["role"], arm].append((full, change))
        content_groups[e["role"], e["content"], arm].append((full, change))
        by_pair[e["pair"]][arm] = y
    effects = csv_rows(out / "03/effects.csv")
    if len(effects) != 336 or len(expected) != 336 or len(by_pair) != 48:
        raise ValueError("Incomplete intervention rows")
    for row in effects:
        for key, value in expected[int(row["unit"])].items():
            errors.append(abs(float(row[key]) - value))
    cell_expected, energy_groups = {}, defaultdict(list)
    for e, a, native in captures.values():
        values = by_pair[e["pair"]]
        donor = captures[e["donor"]][2]
        b, q = values["body"] - native, values["suffix"] - native
        interaction = math.fsum(
            (values["full"], native, -values["body"], -values["suffix"])
        )
        joint = values["full"] - native
        prefix = e["partition"]["prefix"]
        prefix_diff = float(
            np.max(
                abs(
                    a[prefix].astype(float)
                    - captures[e["donor"]][1][prefix].astype(float)
                )
            )
        )
        cell_expected[e["pair"]] = {
            "y00": native,
            "y10": values["body"],
            "y01": values["suffix"],
            "y11": values["full"],
            "body_action": b,
            "suffix_action": q,
            "interaction": interaction,
            "joint_action": joint,
            "reconstruction_error": math.fsum((joint, -b, -q, -interaction)),
            "fresh_donor": donor,
            "full_replay_error": values["full"] - donor,
            "noop_error": values["noop"] - native,
            "common_prefix_max_difference": prefix_diff,
        }
        energy_groups[e["role"]].append((b, q, interaction, joint))
    cells = csv_rows(out / "03/four_cells.csv")
    if len(cells) != 48:
        raise ValueError("Missing four cells")
    for row in cells:
        for key, value in cell_expected[int(row["pair"])].items():
            errors.append(abs(float(row[key]) - value))
    for row in csv_rows(out / "03/signed_energy.csv"):
        x = energy_groups[row["role"]]
        n = len(x)
        terms = {
            "body_square": math.fsum(v[0] ** 2 for v in x) / n,
            "suffix_square": math.fsum(v[1] ** 2 for v in x) / n,
            "interaction_square": math.fsum(v[2] ** 2 for v in x) / n,
            "body_suffix_cross": math.fsum(2 * v[0] * v[1] for v in x) / n,
            "body_interaction_cross": math.fsum(2 * v[0] * v[2] for v in x) / n,
            "suffix_interaction_cross": math.fsum(2 * v[1] * v[2] for v in x) / n,
        }
        terms.update(
            joint_square=math.fsum(v[3] ** 2 for v in x) / n,
            reconstructed_square=math.fsum(terms.values()),
        )
        for key, value in terms.items():
            errors.append(abs(float(row[key]) - value))
    for filename, lookup in (
        ("arm_summary.csv", groups),
        ("content_summary.csv", content_groups),
    ):
        actual = csv_rows(out / "03" / filename)
        if len(actual) != len(lookup):
            raise ValueError("Missing grouped conditions")
        for row in actual:
            key = (
                (row["role"], row["arm"])
                if filename.startswith("arm")
                else (row["role"], row["content"], row["arm"])
            )
            values, n = lookup[key], len(lookup[key])
            denom = math.fsum(f * f for f, _ in values)
            expected_group = {
                "rows": n,
                "patch_rms": math.sqrt(math.fsum(d * d for _, d in values) / n),
                "full_rms": math.sqrt(denom / n),
                "signed_projection": math.fsum(f * d for f, d in values) / denom
                if denom
                else None,
                "remaining_rmse": math.sqrt(
                    math.fsum((f - d) ** 2 for f, d in values) / n
                ),
                "direction_count": sum(f * d > 0 for f, d in values),
                "max_abs_change": max(abs(d) for _, d in values),
            }
            for name, value in expected_group.items():
                if value is None:
                    if row[name] != "":
                        raise ValueError("Invented zero-denominator ratio")
                else:
                    errors.append(abs(float(row[name]) - value))
    replays = csv_rows(out / "03/native_replay.csv")
    if len(replays) != 48:
        raise ValueError("Missing historical replay")
    historical_max = 0.0
    for row in replays:
        e, _, y = next(v for v in captures.values() if v[0]["pair"] == int(row["pair"]))
        old, _ = native_score(
            read_json(
                ROOT / m["historical_root"] / f"01/units/unit_{e['receiver']:04d}.json"
            )["native"]
        )
        errors.extend(
            (
                abs(float(row["native"]) - y),
                abs(float(row["historical"]) - old),
                abs(float(row["difference"]) - (y - old)),
            )
        )
        historical_max = max(historical_max, abs(y - old))
    full_error = max(abs(r["full_replay_error"]) for r in cell_expected.values())
    noop_error = max(abs(r["noop_error"]) for r in cell_expected.values())
    if (
        max(errors) > 1e-10
        or max_random_relative_error > 1e-5
        or max(full_error, noop_error) > m["spec"]["score_replay_tolerance"]
    ):
        raise ValueError(
            "Actual numerical assignment/replay/identity verification failed"
        )
    summary = read_json(out / "03/summary.json")
    if summary["effects"] != 336 or summary["four_cells"] != 48:
        raise ValueError("False summary scope")
    for key, value in (
        ("full_replay_max_error", full_error),
        ("noop_max_error", noop_error),
        ("native_historical_max_error", historical_max),
    ):
        if abs(summary[key] - value) > 1e-10:
            raise ValueError("Wrong summary value")
    result = {
        "status": "VERIFIED_FINITE_HM50_NOT_GLOBAL",
        "actual_forwards": 384,
        "source_calls_sealed_before_application": 192,
        "full_captures": 48,
        "patches_checked_all_token_positions": 336,
        "four_cells_checked": 48,
        "maximum_alternate_error": max(errors),
        "full_replay_max_error": full_error,
        "noop_max_error": noop_error,
        "native_historical_max_error": historical_max,
        "max_repeated_prepatch_state_difference": max_pre_difference,
        "max_random_norm_relative_error": max_random_relative_error,
        "same_agent_alternate_arithmetic_not_neural_replication": True,
        "verifier_new_model_calls": 0,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result, flush=True)


def main():
    """Verify the current frozen finite packet without changing scientific inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
