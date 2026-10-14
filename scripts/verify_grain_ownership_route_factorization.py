#!/usr/bin/env python3
"""HM75 alternate arithmetic and actual assignment audit, zero model calls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_causal_ownership_sectors import array_hash
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_causal_ownership_sectors import unit_path as old_unit
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_ownership_route_factorization import (
    OUT,
    STAGES,
    TABLE_ARMS,
    check,
    payload_path,
    unit_path,
)
from scripts.verify_grain_causal_ownership_sectors import (
    alternate_coefficients,
    direct_action,
    literal_signs,
)


def scalar_values(meta):
    """Recompute conditional score/odds with Python sums, not the runner decoder."""
    z = meta["native"]["candidate_logits"]
    w = [math.exp(v - max(z)) for v in z]
    norm = math.fsum(w)
    return {
        "score": math.fsum(i * v / 9 for i, v in enumerate(w)) / norm,
        "log_odds": math.log(math.fsum(w[5:])) - math.log(math.fsum(w[:5])),
        "centered_logits": np.array(z) - math.fsum(z) / len(z),
    }


def verify(out):
    """Audit all actual units, fixed payloads and every reported numeric account."""
    m = check(out)
    status = read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 3234
            for k in ("attempted_forward_calls", "forward_calls", "completed_units")
        )
    ):
        raise ValueError("Full actual3234 terminal acquisitions required")
    for seal in ("raw_seal.json", "03/output_seal.json"):
        for name, sha in read_json(out / seal).items():
            if digest(out / name) != sha:
                raise ValueError("Frozen capture/report drift: " + name)
    rows, cubes, design = (
        read_json(out / n) for n in ("00/rows.json", "00/cubes.json", "00/design.json")
    )
    parent = ROOT / m["spec"]["parent"]
    reuse = {
        (v["receiver"], v["arm"]): v["unit"]
        for v in read_json(out / "00/reuse_map.json")
    }
    _, tables = load_arrays(out / "03/tables.json")
    readings = {}
    max_payload, max_report, checks = 0.0, 0.0, 0
    noop_drift = []
    for cube in cubes:
        native = {
            r: load_arrays(old_unit(parent, r))[1]["full"] for r in cube["receivers"]
        }
        suffixes = np.stack(
            [native[r][rows[r]["partition"]["suffix"]] for r in cube["receivers"]]
        )
        c = alternate_coefficients(suffixes, literal_signs())
        cache = {
            a: load_arrays(payload_path(out, cube["cube"], a))[1]
            for a in (*m["spec"]["arms"], "noop")
        }
        chosen = [d for d in design if d["cube"] == cube["cube"]]
        for d in chosen:
            row = rows[d["receiver"]]
            a = native[d["receiver"]]
            b = native[cube["receivers"][d["corner"] ^ 1]]
            split = row["partition"]["suffix"][0]
            arm = d["arm"]
            expected = a.copy()
            if arm not in ("suffix_even", "noop"):
                mean = (
                    a[:split].astype(np.float64) + b[:split].astype(np.float64)
                ) * 0.5
                if arm.startswith("random_body"):
                    direction = np.random.default_rng(
                        m["spec"]["random_seed"] + 8 * cube["cube"] + d["corner"] // 2
                    ).normal(size=mean.shape)
                    direction /= np.sqrt(
                        np.sum(direction * direction, axis=-1, keepdims=True)
                    )
                    delta = a[:split].astype(float) - mean
                    magnitude = np.sqrt(np.sum(delta * delta, axis=-1, keepdims=True))
                    expected[:split] = (
                        a[:split].astype(float)
                        - row["coordinates"][2] * magnitude * direction
                    ).astype(np.float32)
                else:
                    expected[:split] = mean.astype(np.float32)
            if arm in ("suffix_even", "both_even"):
                expected[split:] = (
                    (a[split:].astype(float) + b[split:].astype(float)) * 0.5
                ).astype(np.float32)
            elif arm.endswith("suffix_remove_both"):
                wanted = (
                    suffixes[d["corner"]].astype(float)
                    - float(literal_signs()[3, d["corner"]]) * c[3]
                    - float(literal_signs()[7, d["corner"]]) * c[7]
                )
                expected[split:] = wanted.astype(np.float32)
            bundle = cache[arm]
            p, v = bundle[f"positions_{d['corner']}"], bundle[f"values_{d['corner']}"]
            meta, actual = load_arrays(unit_path(out, d["unit"]))
            if any(meta[k] != d[k] for k in d) or not np.array_equal(
                actual["positions"], p
            ):
                raise ValueError("Wrong actual unit/positions")
            err = float(np.max(abs(v.astype(float) - expected[p].astype(float))))
            max_payload = max(max_payload, err)
            if (
                err > 2e-5
                or not np.array_equal(actual["actual"], v)
                or meta["before_sha256"] != array_hash(a)
                or meta["parent_sha256"] != array_hash(a)
                or meta["unpatched_before_sha256"] != meta["unpatched_after_sha256"]
                or meta["assigned_max_error"] != 0
                or meta["native_cut_max_error"] != 0
            ):
                raise ValueError(
                    "Actual payload/native replay/unchanged positions fail"
                )
            checks += v.size
            values = scalar_values(meta)
            if abs(values["score"] - meta["native"]["conditional_score"]) > 1e-12:
                raise ValueError("Stored conditional score differs")
            if arm == "noop":
                old = scalar_values(read_json(old_unit(parent, d["receiver"])))
                noop_drift.append(
                    max(
                        float(np.max(abs(np.asarray(values[k]) - old[k])))
                        for k in STAGES
                    )
                )
            else:
                readings[d["receiver"], arm] = values
        for r in cube["receivers"]:
            readings[r, "native"] = scalar_values(
                read_json(old_unit(parent, reuse[r, "native"]))
            )
            readings[r, "query_remove_both"] = scalar_values(
                read_json(old_unit(parent, reuse[r, "remove_both"]))
            )
        for i in (0, 2, 4, 6):
            v = cache["both_even"][f"values_{i}"]
            w = cache["both_even"][f"values_{i ^ 1}"]
            if not np.array_equal(v, w):
                raise ValueError("both-even full cut not g-invariant")
    coefficients = {}
    for cube in cubes:
        for stage in STAGES:
            t = np.asarray(
                [[readings[r, a][stage] for r in cube["receivers"]] for a in TABLE_ARMS]
            )
            err = float(np.max(abs(t - tables[f"cube_{cube['cube']:03d}_{stage}"])))
            max_report = max(max_report, err)
            coefficients[cube["cube"], stage] = np.asarray(
                [alternate_coefficients(v, literal_signs()) for v in t]
            )
            checks += t.size
    with (out / "03/coefficients.csv").open() as stream:
        for row in csv.DictReader(stream):
            names = ("constant", "g", "y", "yg", "x", "xg", "xy", "xyg")
            wanted = coefficients[int(row["cube"]), row["stage"]][
                TABLE_ARMS.index(row["arm"]), names.index(row["character"])
            ]
            max_report = max(
                max_report,
                float(np.max(abs(np.asarray(json.loads(row["value"])) - wanted))),
            )
            checks += np.size(wanted)
    cubes_by_id = {c["cube"]: c for c in cubes}
    with (out / "03/actions.csv").open() as stream:
        for row in csv.DictReader(stream):
            cube = cubes_by_id[int(row["cube"])]
            v = np.asarray(
                [readings[r, row["arm"]][row["stage"]] for r in cube["receivers"]]
            )
            wanted = direct_action(v)[
                ("average", "x_low", "x_high").index(row["context"])
            ]
            max_report = max(
                max_report,
                float(np.max(abs(wanted - np.asarray(json.loads(row["value"]))))),
            )
            checks += np.size(wanted)
    with (out / "03/composition_consequences.csv").open() as stream:
        for row in csv.DictReader(stream):
            cube = cubes_by_id[int(row["cube"])]
            j = 3 if row["character"] == "yg" else 7
            v = np.asarray(
                [readings[r, row["arm"]][row["stage"]] for r in cube["receivers"]],
                np.longdouble,
            )
            plus, minus = (
                (1 + 0.5 * literal_signs()[j]) / 8,
                (1 - 0.5 * literal_signs()[j]) / 8,
            )
            for name, w in (("plus_probability", plus), ("minus_probability", minus)):
                max_report = max(
                    max_report,
                    float(np.max(abs(w - np.asarray(json.loads(row[name]))))),
                )
            p = sum(plus[i] * v[i] for i in range(8))
            q = sum(minus[i] * v[i] for i in range(8))
            for name, w in (
                ("plus_score", p),
                ("minus_score", q),
                ("difference", p - q),
            ):
                max_report = max(
                    max_report,
                    float(np.max(abs(w - np.asarray(json.loads(row[name]))))),
                )
            allowed = 3 if j == 3 else 7
            if np.max(abs(literal_signs()[:allowed] @ (plus - minus))) > 1e-15:
                raise ValueError("Author profiles lose matched lower marginals")
            checks += 16 + 3 * np.size(p)
    with (out / "03/route_interactions.csv").open() as stream:
        for row in csv.DictReader(stream):
            cube = cubes_by_id[int(row["cube"])]
            receiver = cube["receivers"][int(row["corner"])]
            v = [
                np.asarray(readings[receiver, a][row["stage"]])
                for a in ("native", "body_even", "suffix_even", "both_even")
            ]
            wanted = (v[0] - v[1]) - (v[2] - v[3])
            max_report = max(
                max_report,
                float(np.max(abs(wanted - np.asarray(json.loads(row["value"]))))),
            )
            checks += np.size(wanted)
    with (out / "03/route_summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            selected = [
                c
                for c in cubes
                if (
                    "neutral"
                    if c["companion"] == "neutral"
                    else "matched"
                    if c["query"] == c["companion"]
                    else "other"
                )
                == row["query_role"]
            ]
            j = 3 if row["character"] == "yg" else 7
            values = np.asarray(
                [
                    coefficients[c["cube"], row["stage"]][
                        TABLE_ARMS.index(row["arm"]), j
                    ]
                    for c in selected
                ],
                np.longdouble,
            )
            wanted = (
                float(np.sum(values) / values.size),
                float(np.sqrt(np.sum(values * values) / values.size)),
            )
            for key, value in zip(
                ("coefficient_mean", "coefficient_RMS"), wanted, strict=True
            ):
                max_report = max(max_report, abs(value - float(row[key])))
            if (
                int(row["cubes"]) != len(selected)
                or int(row["components"]) != values.size
            ):
                raise ValueError("Cube/component count mixed")
    both = max(
        float(np.max(abs(coefficients[c["cube"], s][4, [1, 3, 5, 7]])))
        for c in cubes
        for s in STAGES
    )
    summary = read_json(out / "03/summary.json")
    if (
        max_report > 1e-10
        or both > 1e-6
        or max(noop_drift) > 1e-6
        or abs(both - summary["both_even_odd_coefficients_max_error"]) > 1e-10
    ):
        raise ValueError("Report, full-cut or noop control disagreement")
    receipt = {
        "status": "VERIFIED_FINITE_HM75_NOT_GLOBAL",
        "actual_forwards": 3234,
        "alternate_scalar_checks": checks,
        "maximum_payload_float32_replay_error": max_payload,
        "maximum_report_error": max_report,
        "both_even_odd_max_error": both,
        "noop_output_max_error": max(noop_drift),
        "zero_new_model_calls": True,
        "same_agent_not_third_party": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    verify(p.parse_args().out)
