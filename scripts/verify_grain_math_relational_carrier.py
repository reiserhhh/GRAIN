#!/usr/bin/env python3
"""Alternate arithmetic for all HM49 source vectors, patches and scored effects."""

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
from scripts.run_grain_math_relational_carrier import OUT, check, write
from scripts.verify_grain_math_evaluative_routing import native_score


def csv_rows(path):
    """Load every row without guessing whether a filename proves completion."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def indices(row, which):
    """Derive semantic indices independently from explicit A/B captured layout."""
    high_a = row["behaviour"] == 0
    use_a = high_a if which == "high" else not high_a
    n = row["property_a_count"]
    return list(range(n)) if use_a else list(range(n, len(row["property_positions"])))


def verify(out):
    """Rebuild all mathematical quantities without fresh model forwards or fitting."""
    manifest, status = check(out), read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 384
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
        or status["model_seconds"] > 600
    ):
        raise ValueError("Require complete, bounded actual384 collection")
    for ledger in ("03/artifact_hashes.json", "01/source_barrier.json"):
        hashes = read_json(out / ledger)
        hashes = hashes["files"] if ledger.startswith("01/") else hashes
        for name, sha in hashes.items():
            if digest(out / name) != sha:
                raise ValueError("Changed frozen actual output " + name)
    if read_json(out / "01/source_barrier.json")["source_calls"] != 192:
        raise ValueError("Require all192 source calls sealed before application")
    parent = ROOT / manifest["spec"]["parent"]
    rows = {r["unit"]: r for r in read_json(out / "00/receivers.json")}
    data = {
        unit: load_arrays(parent / f"01/units/unit_{unit:04d}.json") for unit in rows
    }
    _, assets = load_arrays(out / "00/source_vectors.json")
    errors = []
    source_terms = defaultdict(list)
    for entry in read_json(out / "00/design.json"):
        if (
            entry["role"] != "source"
            or entry["direction"] != 0
            or entry["arm"] != "high"
        ):
            continue
        a, b = rows[entry["receiver"]], rows[entry["donor"]]
        terms = []
        for which in ("high", "low"):
            i, j = indices(a, which), indices(b, which)
            difference = sum(
                (
                    data[b["unit"]][1]["property"][q].astype(float)
                    - data[a["unit"]][1]["property"][p].astype(float)
                )
                / len(i)
                for p, q in zip(i, j, strict=True)
            )
            terms.append(difference)
        source_terms[entry["content"]].append(np.stack(terms))
    if len(source_terms) != 6 or any(len(v) != 2 for v in source_terms.values()):
        raise ValueError("Missing source averaging strata")
    components = np.stack(
        [sum(source_terms[t]) / 2 for t in manifest["spec"]["targets"]]
    )
    rebuilt = sum(components) / 6
    errors.extend(
        (
            float(np.max(abs(rebuilt - assets["shared"]))),
            float(np.max(abs(components - assets["per_content"]))),
            float(
                np.max(
                    abs(
                        np.linalg.norm(assets["random"], axis=1)
                        - np.linalg.norm(rebuilt, axis=1)
                    )
                )
            ),
        )
    )
    expected, carrier, groups, content_groups = (
        {},
        {},
        defaultdict(list),
        defaultdict(list),
    )
    for entry in read_json(out / "00/design.json"):
        a, b = rows[entry["receiver"]], rows[entry["donor"]]
        before_meta, before = data[a["unit"]]
        donor_meta, donor = data[b["unit"]]
        if any(
            a[k] != b[k]
            for k in ("role", "content", "query", "person", "body_endpoint")
        ) or len(a["input_ids"]) != len(b["input_ids"]):
            raise ValueError("Changed pair identity")
        if (
            a["input_ids"][a["body_endpoint"] + 1 :]
            != b["input_ids"][b["body_endpoint"] + 1 :]
        ):
            raise ValueError("Changed actual question suffix")
        for which in ("high", "low"):
            i, j = indices(a, which), indices(b, which)
            p, q = (
                [a["property_positions"][k] for k in i],
                [b["property_positions"][k] for k in j],
            )
            if p != q or [a["input_ids"][k] for k in p] != [
                b["input_ids"][k] for k in q
            ]:
                raise ValueError("Changed exact property tokens/positions")
        meta, after = load_arrays(out / f"01/units/unit_{entry['unit']:04d}.json")
        if any(meta[k] != entry[k] for k in ("unit", "pair", "role", "arm")):
            raise ValueError("Wrong intervention identity")
        arm, p, values = entry["arm"], [], []
        if arm in ("body_endpoint", "query_end"):
            key = "endpoint" if arm == "body_endpoint" else "query_end"
            p = [a["body_endpoint"] if key == "endpoint" else len(a["input_ids"]) - 1]
            values = donor[key][None]
        else:
            for which in (arm,) if arm in ("high", "low") else ("high", "low"):
                i, j = indices(a, which), indices(b, which)
                p.extend(a["property_positions"][k] for k in i)
                if arm in ("source_mean", "random"):
                    vector = assets["shared" if arm == "source_mean" else "random"][
                        int(which == "low")
                    ]
                    values.append(
                        (
                            before["property"][i].astype(float)
                            + (1 - 2 * a["behaviour"]) * vector
                        ).astype(before["property"].dtype)
                    )
                else:
                    values.append(
                        before["property"][i] if arm == "noop" else donor["property"][j]
                    )
            values = np.concatenate(values)
        if p != meta["positions"] or not np.array_equal(after["patch_positions"], p):
            raise ValueError("Wrong actual patch position")
        errors.append(
            float(
                np.max(abs(after["patch_values"].astype(float) - values.astype(float)))
            )
        )
        # Check all captured block16 positions, both replaced and unselected.
        for key in ("property", "endpoint", "query_end"):
            addresses = (
                a["property_positions"]
                if key == "property"
                else [
                    a["body_endpoint"] if key == "endpoint" else len(a["input_ids"]) - 1
                ]
            )
            predicted = before[key].copy().reshape(len(addresses), -1)
            for location, position in enumerate(addresses):
                if position in p:
                    predicted[location] = values[p.index(position)]
            errors.append(
                float(
                    np.max(
                        abs(
                            after[key].reshape(predicted.shape).astype(float)
                            - predicted.astype(float)
                        )
                    )
                )
            )
            d = after[key].astype(float) - before[key].astype(float)
            carrier[entry["unit"], key] = (
                float(np.sqrt(np.sum(d * d))),
                float(np.max(abs(d))),
            )
        y, mass = native_score(meta["native"])
        native, _ = native_score(before_meta["native"])
        target, _ = native_score(donor_meta["native"])
        errors.extend(
            (
                abs(y - meta["native"]["conditional_score"]),
                abs(mass - meta["native"]["candidate_mass"]),
            )
        )
        full, change = target - native, y - native
        expected[entry["unit"]] = {
            "native": native,
            "donor": target,
            "patched": y,
            "candidate_mass": mass,
            "full_change": full,
            "patch_change": change,
            "remaining": target - y,
            "ratio": change / full if full else None,
            "same_direction": change * full > 0,
        }
        groups[entry["role"], arm].append((full, change))
        content_groups[entry["role"], entry["content"], arm].append((full, change))
    effects = csv_rows(out / "03/effects.csv")
    if len(effects) != 384 or len(expected) != 384:
        raise ValueError("Incomplete effect ledger")
    for row in effects:
        for key, value in expected[int(row["unit"])].items():
            if key == "same_direction":
                if (row[key] == "True") != value:
                    raise ValueError("Wrong signed direction")
            elif value is None:
                if row[key] != "":
                    raise ValueError("False zero-denominator ratio")
            else:
                errors.append(abs(float(row[key]) - value))
    carrier_rows = csv_rows(out / "03/carrier_checks.csv")
    if len(carrier_rows) != 1152:
        raise ValueError("Incomplete actual carrier norm ledger")
    for row in carrier_rows:
        norm, maximum = carrier[int(row["unit"]), row["carrier"]]
        errors.extend(
            (
                abs(norm - float(row["difference_norm"])),
                abs(maximum - float(row["max_abs_difference"])),
            )
        )
    summaries = {}
    for filename, grouped in (
        ("arm_summary.csv", groups),
        ("content_summary.csv", content_groups),
    ):
        table = csv_rows(out / "03" / filename)
        if len(table) != len(grouped):
            raise ValueError("Incomplete summary scope")
        for row in table:
            key = (
                (row["role"], row["arm"])
                if filename == "arm_summary.csv"
                else (row["role"], row["content"], row["arm"])
            )
            pairs = grouped[key]
            n = len(pairs)
            f2, c2 = (
                math.fsum(f * f for f, _ in pairs),
                math.fsum(c * c for _, c in pairs),
            )
            result = {
                "rows": n,
                "patch_rms": math.sqrt(c2 / n),
                "full_rms": math.sqrt(f2 / n),
                "signed_projection": math.fsum(f * c for f, c in pairs) / f2
                if f2
                else None,
                "remaining_rmse": math.sqrt(
                    math.fsum((f - c) ** 2 for f, c in pairs) / n
                ),
                "direction_count": sum(f * c > 0 for f, c in pairs),
                "max_abs_change": max(abs(c) for _, c in pairs),
            }
            for field, value in result.items():
                if value is not None:
                    errors.append(abs(float(row[field]) - value))
            if filename == "arm_summary.csv":
                summaries["/".join(key)] = result
    summary = read_json(out / "03/summary.json")
    for key, result in summaries.items():
        for field, value in result.items():
            if value is not None:
                errors.append(abs(summary["groups"][key][field] - value))
    noop = max(
        abs(c)
        for (role, arm), pairs in groups.items()
        if arm == "noop"
        for _, c in pairs
    )
    errors.append(abs(noop - summary["no_op_max_error"]))
    # Arithmetic tolerance is not an effect-size promotion threshold.
    if max(errors) > 1e-10 or noop > 1e-6:
        raise ValueError("Arithmetic or no-op mismatch " + str(max(errors)))
    result = {
        "status": "VERIFIED_FINITE_HM49_NOT_GLOBAL",
        "actual_forwards": 384,
        "source_calls_before_application": 192,
        "receivers": 48,
        "effect_rows": 384,
        "carrier_norm_rows_recomputed": 1152,
        "maximum_alternate_error": max(errors),
        "no_op_max_error": noop,
        "new_model_calls": 0,
        "same_agent_alternate_arithmetic_not_neural_replication": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result, flush=True)
    return result


def main():
    """Verify a completed finite packet, never acquire or repair its raw results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
