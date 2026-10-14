#!/usr/bin/env python3
"""HM80 alternate finite sums, S3 actions, equal-count inputs and author effects."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_role_count_sectors_hm80 import OUT, check, raw_tables, seal_check


def verify(out):
    """Recalculate all finite observation sectors without their primary helpers."""
    manifest = check(out)
    seal_check(out, out / "03/seal.json")
    tables, raw_checks = raw_tables(out)
    summary = read_json(out / "03/summary.json")
    rows = read_json(out / "00/rows.json")
    points = np.asarray(list(itertools.product((-1, 1), repeat=5)), dtype=int)
    masks = list(itertools.product((0, 1), repeat=5))
    permutations = list(itertools.permutations((2, 3, 4)))
    lookup = {tuple(p): k for k, p in enumerate(points)}
    characters = np.array(
        [
            [np.prod([point[k] for k, b in enumerate(mask) if b]) for mask in masks]
            for point in points
        ],
        dtype=float,
    )
    epsilon = manifest["spec"]["profile_epsilon"]
    alternate, error, comparisons = {}, 0.0, 0

    def compare(a, b):
        nonlocal error, comparisons
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Mismatched/nonfinite alternate comparison")
        error = max(error, float(np.max(np.abs(a - b))))
        comparisons += a.size

    proofs = read_json(out / "00/count_proofs.json")
    if len(proofs) != 2880 or len({tuple(p["key"]) for p in proofs}) != 2880:
        raise ValueError("Count-input proof scope wrong")
    checked_units = set()
    for proof in proofs:
        first = Counter(rows[proof["units"][0]]["input_ids"])
        if sorted(first.items()) != [tuple(a) for a in proof["token_bag"]]:
            raise ValueError("Frozen count bag differs")
        for unit in proof["units"]:
            row = rows[unit]
            key = [
                row["family"],
                row["focal"],
                row["companion"],
                *row["coordinates"][:2],
                row["person_a_events"],
                row["query_role"],
            ]
            if key != proof["key"] or Counter(row["input_ids"]) != first:
                raise ValueError("Equal-count actual inputs not equivalent")
            if row["body"].count("Person A ") != row["person_a_events"]:
                raise ValueError("Actual actor count not declared count")
            if row["coordinates"] != points[row["corner"]].tolist():
                raise ValueError("Actual design coordinate mismatch")
            checked_units.add(unit)
    if checked_units != set(range(5760)):
        raise ValueError("Count proofs omit actual readings")
    for edit in read_json(out / "00/edit_proofs.json"):
        a, b = [rows[k] for k in edit["units"]]
        changed = [
            k
            for k, (u, v) in enumerate(zip(a["input_ids"], b["input_ids"], strict=True))
            if u != v
        ]
        if changed != [edit["position"]] or changed[0] not in a["partition"]["body"]:
            raise ValueError("Actual one-token edit mismatch")
        wanted = a["coordinates"].copy()
        wanted[edit["bit"]] *= -1
        if wanted != b["coordinates"]:
            raise ValueError("Wrong coordinate flipped")

    with np.load(out / "03/tables_and_sectors.npz", allow_pickle=False) as saved:
        for context, (key, table) in enumerate(sorted(tables.items())):
            if list(key) != summary["context_keys"][context]:
                raise ValueError("Wrong context attribution")
            for stage in summary["stages"]:
                a = table[stage]
                c = np.array(
                    [
                        sum(characters[k, j] * a[k] for k in range(32)) / 32
                        for j in range(32)
                    ]
                )
                p = np.zeros_like(a)
                for order in permutations:
                    for k, point in enumerate(points):
                        index = lookup[tuple(point[:2]) + tuple(point[list(order)])]
                        p[k] += a[index] / 6
                r = a - p
                edge = c[[30, 29, 27]]
                common = np.broadcast_to(edge.mean(0), edge.shape)
                expected = {
                    "values": a,
                    "coefficients": c,
                    "count": p,
                    "role": r,
                    "pair_coefficients": edge,
                    "pair_count": common,
                    "pair_role": edge - common,
                }
                alternate[(context, stage)] = expected
                for name, value in expected.items():
                    compare(value, saved[f"{name}_{context:02d}_{stage}"])
                compare((characters @ c.reshape(32, -1)).reshape(a.shape), a)
                compare(np.asarray(np.mean((p - p.mean(0)) * r)), np.asarray(0.0))
                if stage == "centered_logits":
                    z = a - a.max(-1, keepdims=True)
                    probability = np.exp(z)
                    probability /= probability.sum(-1, keepdims=True)
                    compare(probability @ np.linspace(0, 1, 10), table["score"][..., 0])
                    compare(
                        np.log(
                            probability[..., 5:].sum(-1) / probability[..., :5].sum(-1)
                        ),
                        table["log_odds"][..., 0],
                    )
                for k, l in itertools.combinations(range(3), 2):
                    edge_sign = characters[:, [30, 29, 27]]
                    delta = (edge_sign[:, k] - edge_sign[:, l]) / 2
                    plus, minus = (1 + epsilon * delta) / 32, (1 - epsilon * delta) / 32
                    compare(
                        (plus - minus) @ characters[:, np.array(masks).sum(1) <= 3],
                        np.zeros(26),
                    )
                    for x, y, n in itertools.product((-1, 1), (-1, 1), range(4)):
                        select = (points[:, 0] == x) & (points[:, 1] == y)
                        select &= (points[:, 2:] > 0).sum(1) == n
                        compare(
                            np.asarray((plus - minus)[select].sum()), np.asarray(0.0)
                        )
                    compare(
                        plus @ a.reshape(32, -1) - minus @ a.reshape(32, -1),
                        (epsilon * (edge[k] - edge[l])).reshape(-1),
                    )

    query_roles = manifest["spec"]["query_roles"]

    def entry(row):
        context, stage = int(row["context"]), row["stage"]
        key = summary["context_keys"][context]
        if (
            int(row["family"]) != key[0]
            or row["focal"] != key[1]
            or row["companion"] != key[2]
        ):
            raise ValueError("CSV context attribution wrong")
        q = 0 if stage == "body_endpoint" else query_roles.index(row["query_role"])
        expected = alternate[(context, stage)]
        if stage != "body_endpoint":
            targets = manifest["spec"]["targets"]
            start = targets.index(key[2])
            off = next(
                targets[(start + k) % 6]
                for k in range(1, 6)
                if targets[(start + k) % 6] not in key[1:3]
            )
            if row["query"] != (key[1], key[2], off)[q]:
                raise ValueError("CSV query target mislabeled")
        return expected, q

    derived_accounts = []
    for name in ("coefficients", "count_accounts", "pair_roles", "author_compositions"):
        with (out / f"03/{name}.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                e, q = entry(row)
                if name == "coefficients":
                    mask = (
                        tuple(int(v in row["term"]) for v in "xyghi")
                        if row["term"] != "1"
                        else (0,) * 5
                    )
                    value = e["coefficients"][masks.index(mask), q]
                    compare(np.asarray(float(row["rms"])), np.sqrt(np.mean(value**2)))
                    if row["scalar"]:
                        compare(np.asarray(float(row["scalar"])), value[0])
                elif name == "pair_roles":
                    k = ("gh", "gi", "hi").index(row["edge"])
                    c, r = e["pair_coefficients"][k, q], e["pair_role"][k, q]
                    for field, value in (
                        ("coefficient_rms", np.sqrt(np.mean(c * c))),
                        (
                            "count_mean_rms",
                            np.sqrt(np.mean(e["pair_count"][k, q] ** 2)),
                        ),
                        ("role_rms", np.sqrt(np.mean(r * r))),
                    ):
                        compare(np.asarray(float(row[field])), value)
                    if row["coefficient_scalar"]:
                        compare(np.asarray(float(row["coefficient_scalar"])), c[0])
                        compare(np.asarray(float(row["role_scalar"])), r[0])
                elif name == "author_compositions":
                    k, l = (
                        ("gh", "gi", "hi").index(row[field])
                        for field in ("first", "second")
                    )
                    value = epsilon * (
                        e["pair_coefficients"][k, q] - e["pair_coefficients"][l, q]
                    )
                    compare(
                        np.asarray(float(row["difference_rms"])),
                        np.sqrt(np.mean(value**2)),
                    )
                    if row["difference_scalar"]:
                        compare(np.asarray(float(row["difference_scalar"])), value[0])
                else:
                    a, p, r = e["values"][:, q], e["count"][:, q], e["role"][:, q]
                    fields = {
                        "total_variance": np.mean((a - a.mean(0)) ** 2),
                        "count_variance": np.mean((p - p.mean(0)) ** 2),
                        "role_variance": np.mean(r * r),
                        "cross_inner_mean": np.mean((p - p.mean(0)) * r),
                        "pair_count_scaled_mean_square": np.mean(
                            (8 * e["pair_count"][:, q]) ** 2
                        ),
                        "pair_role_scaled_mean_square": np.mean(
                            (8 * e["pair_role"][:, q]) ** 2
                        ),
                    }
                    for field, value in fields.items():
                        compare(np.asarray(float(row[field])), value)
                    compare(
                        fields["total_variance"],
                        fields["count_variance"] + fields["role_variance"],
                    )
                    derived_accounts.append(
                        dict(
                            family=int(row["family"]),
                            stage=row["stage"],
                            query_role=row["query_role"],
                            **fields,
                        )
                    )
    for row in summary["summary_by_group"]:
        selected = [
            a
            for a in derived_accounts
            if a["family"] == row["family"]
            and a["stage"] == row["stage"]
            and (
                row["query_role"] == "all_queries"
                or a["query_role"] == row["query_role"]
            )
        ]
        if len(selected) != row["contexts_queries"]:
            raise ValueError("Summary scope mismatch")
        for field in (
            "total_variance",
            "count_variance",
            "role_variance",
            "cross_inner_mean",
            "pair_count_scaled_mean_square",
            "pair_role_scaled_mean_square",
        ):
            compare(
                np.asarray(row[field]),
                np.asarray(np.mean([r[field] for r in selected])),
            )
        compare(
            np.asarray(row["pair_count_scaled_rms"]),
            np.sqrt(row["pair_count_scaled_mean_square"]),
        )
        compare(
            np.asarray(row["pair_role_scaled_rms"]),
            np.sqrt(row["pair_role_scaled_mean_square"]),
        )
        if row["role_variation_fraction"] is not None:
            compare(
                np.asarray(row["role_variation_fraction"]),
                np.asarray(row["role_variance"] / row["total_variance"]),
            )
    status = read_json(out / "run_status.json")
    if (
        any(
            status[k] != 5760
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
        or status["in_flight"] is not None
    ):
        raise ValueError("Actual call count incomplete")
    if error > manifest["spec"]["arithmetic_tolerance"]:
        raise ValueError(f"Alternate arithmetic disagreement:{error}")
    result = dict(
        status="VERIFIED_FINITE_HM80_NOT_GLOBAL",
        actual_new_forwards=5760,
        old_neural_reuses=0,
        actual_scalar_comparisons=comparisons,
        maximum_alternate_error=error,
        verified_count_bag_classes=2880,
        verified_actual_input_edits=8640,
        **raw_checks,
        same_agent_alternate_not_third_party=True,
        original_scope_complete=False,
    )
    write(out / "verification.json", result)
    print(result, flush=True)


def main():
    """Verify frozen input/analysis data without new model reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    verify(parser.parse_args().out)


if __name__ == "__main__":
    main()
