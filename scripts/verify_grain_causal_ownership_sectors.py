#!/usr/bin/env python3
"""HM74 alternate arithmetic replay, with no additional model calls."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_causal_ownership_sectors import array_hash
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_causal_ownership_sectors import (
    OUT,
    check,
    payload_path,
    unit_path,
)
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_evaluative_routing import load_arrays


def literal_signs():
    """Build characters independently from the named-variable definition."""
    corners = list(itertools.product((-1, 1), repeat=3))
    characters = ((), (2,), (1,), (1, 2), (0,), (0, 2), (0, 1), (0, 1, 2))
    return np.asarray(
        [[math.prod(z[k] for k in indices) for z in corners] for indices in characters],
        np.longdouble,
    )


def alternate_coefficients(values, signs):
    """Sum eight signed terms in extended precision, not the construction matrix product."""
    a = np.asarray(values, np.longdouble)
    expand = (8,) + (1,) * (a.ndim - 1)
    return np.asarray(
        [np.sum(s.reshape(expand) * a, axis=0, dtype=np.longdouble) / 8 for s in signs],
        float,
    )


def direct_action(values):
    """Use literal corner indices, independently of coefficient-derived actions."""
    a = np.asarray(values, np.longdouble)
    low, high = a[3] - a[1] - a[2] + a[0], a[7] - a[5] - a[6] + a[4]
    return np.asarray([(low + high) / 2, low, high], float)


def verify(out):
    """Replay all captures, assignments, coefficients and reported consequences."""
    m = check(out)
    status = read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 4224
            for k in ("attempted_forward_calls", "forward_calls", "completed_units")
        )
    ):
        raise ValueError("Full terminal actual acquisitions required")
    for seal in ("raw_seal.json", "03/output_seal.json"):
        for name, sha in read_json(out / seal).items():
            if digest(out / name) != sha:
                raise ValueError("Frozen capture/output drift: " + name)
    rows, cubes, design = (
        read_json(out / n) for n in ("00/rows.json", "00/cubes.json", "00/design.json")
    )
    _, tables = load_arrays(out / "03/tables.json")
    signs = literal_signs()
    max_payload_error, max_scalar_error, checks = 0.0, 0.0, 0
    sources, scalar = {}, {}
    for entry in design:
        meta = read_json(unit_path(out, entry["unit"]))
        if any(meta[k] != entry[k] for k in entry):
            raise ValueError("Wrong acquisition identity")
        z = meta["native"]["candidate_logits"]
        shifted = [math.exp(t - max(z)) for t in z]
        p = [v / math.fsum(shifted) for v in shifted]
        score = math.fsum(v * i / 9 for i, v in enumerate(p))
        odds = math.log(math.fsum(shifted[5:])) - math.log(math.fsum(shifted[:5]))
        scalar[entry["cube"], entry["arm"], entry["corner"]] = {
            "score": score,
            "log_odds": odds,
            "centered_logits": np.asarray(z) - math.fsum(z) / 10,
        }
        err = abs(score - meta["native"]["conditional_score"])
        max_scalar_error = max(max_scalar_error, err)
        if err > 1e-12:
            raise ValueError("Native candidate score disagreement")
        if entry["kind"] == "capture":
            _, arr = load_arrays(unit_path(out, entry["unit"]))
            full = arr["full"]
            if array_hash(full) != meta["full_sha256"]:
                raise ValueError("Native state identity drift")
            sources[entry["receiver"]] = (full, meta["full_sha256"])
        checks += 12
    expected = {}
    for group in cubes:
        cube = np.stack(
            [sources[r][0][rows[r]["partition"]["suffix"]] for r in group["receivers"]]
        )
        c = alternate_coefficients(cube, signs)
        for arm in m["spec"]["arms"]:
            pmeta, parr = load_arrays(payload_path(out, group["cube"], arm))
            actual = parr["values"]
            if array_hash(actual) != pmeta["values_sha256"]:
                raise ValueError("Sealed payload identity drift")
            if arm == "noop":
                wanted = cube
            elif arm == "exact_owner":
                wanted = cube[[1, 0, 3, 2, 5, 4, 7, 6]]
            else:
                indices = {
                    "remove_yg": (3,),
                    "remove_xyg": (7,),
                    "remove_both": (3, 7),
                    "remove_xy": (6,),
                    "random_both": (3, 7),
                }[arm]
                changes = np.asarray(
                    [sum(float(signs[j, i]) * c[j] for j in indices) for i in range(8)]
                )
                if arm == "random_both":
                    random = np.random.default_rng(
                        m["spec"]["random_seed"] + group["cube"]
                    ).normal(size=cube.shape[1:])
                    norms = np.sqrt(np.sum(random**2, axis=1, keepdims=True))
                    random /= norms
                    changes = (
                        np.asarray(signs[3], float)[:, None, None]
                        * np.sqrt(np.sum(changes**2, axis=-1, keepdims=True))
                        * random[None]
                    )
                wanted = (cube.astype(float) - changes).astype(np.float32)
            err = float(np.max(abs(actual.astype(float) - wanted.astype(float))))
            max_payload_error = max(max_payload_error, err)
            if err > 2e-5:
                raise ValueError("Counterfactual assignment formula disagreement")
            expected[group["cube"], arm] = actual
            checks += actual.size
        for arm_index, arm in enumerate(("native", *m["spec"]["arms"])):
            for stage in ("score", "log_odds", "centered_logits"):
                observed = np.asarray(
                    [scalar[group["cube"], arm, i][stage] for i in range(8)]
                )
                table = tables[f"cube_{group['cube']:03d}_{stage}"][arm_index]
                err = float(np.max(abs(observed - table)))
                max_scalar_error = max(max_scalar_error, err)
                if err > 1e-11:
                    raise ValueError("Published table disagreement")
                checks += table.size
    for entry in design:
        if entry["kind"] != "patch":
            continue
        meta, arr = load_arrays(unit_path(out, entry["unit"]))
        native, sha = sources[entry["receiver"]]
        suffix = rows[entry["receiver"]]["partition"]["suffix"]
        if (
            meta["before_sha256"] != sha
            or meta["unchanged_before_sha256"] != array_hash(native[: suffix[0]])
            or meta["unchanged_before_sha256"] != meta["unchanged_after_sha256"]
        ):
            raise ValueError("Undeclared input/context change")
        assigned = expected[entry["cube"], entry["arm"]][entry["corner"]]
        if (
            not np.array_equal(arr["actual_suffix"], assigned)
            or meta["actual_payload_max_error"] != 0
            or meta["native_before_max_error"] != 0
        ):
            raise ValueError("Actual assigned states differ from sealed payload")
        checks += assigned.size
    for name in (
        "coefficients.csv",
        "actions.csv",
        "composition_consequences.csv",
        "joint_ablation.csv",
        "query_pair_contrasts.csv",
    ):
        with (out / "03" / name).open() as stream:
            records = list(csv.DictReader(stream))
        for record in records:
            if name == "query_pair_contrasts.csv":
                groups = {
                    (g["focal"], g["companion"], g["query"]): g["cube"] for g in cubes
                }
                focal, companion, stage = (
                    record["focal"],
                    record["companion"],
                    record["stage"],
                )
                index = ("native", *m["spec"]["arms"]).index(record["arm"])
                k = ("marginal", "x_low", "x_high").index(record["context"])
                matched = tables[
                    f"cube_{groups[focal, companion, companion]:03d}_{stage}"
                ]
                other = tables[f"cube_{groups[focal, companion, focal]:03d}_{stage}"]
                a, b = direct_action(matched[index])[k], direct_action(other[index])[k]
                na, nb = direct_action(matched[0])[k], direct_action(other[0])[k]
                pairs = (
                    (a, float(record["matched_action"])),
                    (b, float(record["focal_action"])),
                    (a - b, float(record["matched_minus_focal"])),
                    (a - b - na + nb, float(record["selectivity_change"])),
                )
                for actual, published in pairs:
                    err = float(np.max(abs(actual - published)))
                    max_scalar_error = max(max_scalar_error, err)
                    if err > 1e-10:
                        raise ValueError("Paired-query contrast disagreement")
                    checks += np.size(actual)
                continue
            cube, stage, arm = (
                int(record["cube"]),
                record["stage"],
                record.get("arm", "native"),
            )
            table = tables[f"cube_{cube:03d}_{stage}"]
            value = table[("native", *m["spec"]["arms"]).index(arm)]
            if name == "coefficients.csv":
                j = ("constant", "g", "y", "yg", "x", "xg", "xy", "xyg").index(
                    record["character"]
                )
                pairs = (
                    (
                        alternate_coefficients(value, signs)[j],
                        json.loads(record["value"]),
                    ),
                    (
                        alternate_coefficients(table[0], signs)[j],
                        json.loads(record["native_value"]),
                    ),
                )
            elif name == "actions.csv":
                k = ("marginal", "x_low", "x_high").index(record["context"])
                a, b = direct_action(value)[k], direct_action(table[0])[k]
                pairs = (
                    (a, json.loads(record["patched"])),
                    (b, json.loads(record["native"])),
                    (a - b, json.loads(record["change"])),
                    (b - a, json.loads(record["loss"])),
                )
            elif name == "composition_consequences.csv":
                j = 3 if record["character"] == "yg" else 7
                probabilities = [
                    (1 + s * m["spec"]["author_profile_amplitude"] * signs[j]) / 8
                    for s in (1, -1)
                ]
                sums = [
                    np.tensordot(
                        p.astype(np.longdouble),
                        value.astype(np.longdouble),
                        axes=(0, 0),
                    )
                    for p in probabilities
                ]
                pairs = (
                    (sums[0], json.loads(record["plus_score"])),
                    (sums[1], json.loads(record["minus_score"])),
                    (sums[0] - sums[1], json.loads(record["difference"])),
                )
            else:
                actual = table[0] - table[1] - table[2] + table[3]
                pairs = (
                    (actual, json.loads(record["interaction"])),
                    (np.sqrt(np.mean(actual**2)), float(record["rms"])),
                )
            for a, b in pairs:
                err = float(
                    np.max(
                        abs(np.asarray(a, np.longdouble) - np.asarray(b, np.longdouble))
                    )
                )
                max_scalar_error = max(max_scalar_error, err)
                if err > 1e-10:
                    raise ValueError("Alternate report replay disagreement: " + name)
                checks += np.size(a)
    summary = read_json(out / "03/summary.json")
    if summary["noop_max_error"] != 0:
        raise ValueError("No-op control changed readout")
    result = {
        "status": "VERIFIED_FINITE_HM74_NOT_GLOBAL",
        "actual_forwards": status["forward_calls"],
        "alternate_scalar_checks": checks,
        "maximum_payload_float32_replay_error": max_payload_error,
        "maximum_score_report_error": max_scalar_error,
        "zero_new_model_calls": True,
        "same_agent_not_third_party": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    """Verify an existing complete root, never generate replacement observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.out)


if __name__ == "__main__":
    main()
