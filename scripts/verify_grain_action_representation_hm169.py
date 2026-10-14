#!/usr/bin/env python3
"""Read-only direct-sum and direct-distance audit of HM169; no new science runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_action_representation_hm169 as H


def direct_center(x):
    """Compute Euclidean distances with explicit row pairs and the H matrix."""
    distance = np.array(
        [[float(np.linalg.norm(left - right)) for right in x] for left in x]
    )
    center = np.eye(len(x)) - np.ones((len(x), len(x))) / len(x)
    return center @ distance @ center


def verify(out):
    """Audit every cold, source ledger, distance statistic and profile permutation."""
    manifest = H.check(out)
    for seal in ("source_program_seal.json", "cold_seal.json", "analysis_seal.json"):
        H.require_seal(out, seal)
    pairs = H.A.read(out / "pairs.json")
    with np.load(out / "source_actions.npz", allow_pickle=False) as saved:
        ys, groups = saved["responses"], saved["groups"]
    with np.load(out / "application_actions.npz", allow_pickle=False) as saved:
        ya = saved["responses"]
    with np.load(out / "paired_actions.npz", allow_pickle=False) as saved:
        native, cold = saved["native"], saved["cold"]
    with np.load(out / "source_group_contributions.npz", allow_pickle=False) as saved:
        ledger, weights = saved["contributions"], saved["source_weights"]
        group_keys = saved["source_groups"]
    with np.load(out / "permutation_accounts.npz", allow_pickle=False) as saved:
        permutations = {
            role: saved[role + "_indices"] for role in manifest["config"]["roles"]
        }
        nulls = {role: saved[role + "_null"] for role in manifest["config"]["roles"]}
    distance_rows = list(csv.DictReader((out / "distance_dependence.csv").open()))
    all_pairs = pairs["source"] + pairs["application"]
    all_y = H.M.order_fields(
        np.concatenate(
            [H.paired(ys, pairs["source"]), H.paired(ya, pairs["application"])]
        )
    )
    role_names = np.array([p["role"] for p in all_pairs])
    all_groups = np.array([p["situation"] + "|" + p["style"] for p in all_pairs])
    errors = {
        "cold_direct_source_sum": 0.0,
        "ledger_direct_group_sum": 0.0,
        "ledger_components": 0.0,
        "distance_statistics": 0.0,
        "permutation_statistics": 0.0,
        "permutation_indices": 0.0,
        "native_pairs": 0.0,
    }
    checks = {
        "cold_values": 0,
        "ledger_values": 0,
        "dependence_accounts": 0,
        "null_accounts": 0,
    }
    errors["native_pairs"] = float(np.max(abs(native - all_y[len(pairs["source"]) :])))
    for vi, view in enumerate(manifest["config"]["views"]):
        with np.load(out / view / "source_features.npz", allow_pickle=False) as saved:
            xs = saved["features"]
        with np.load(
            out / view / "application_features.npz", allow_pickle=False
        ) as saved:
            xa = saved["features"]
        with np.load(out / view / "cold_readings.npz", allow_pickle=False) as saved:
            whole_cold = saved["readings"]
        origin, base = xs.mean(0), ys.mean(0)
        xc = xs - origin
        expected = []
        for point in xa:
            w = np.array([np.dot(source, point - origin) / len(xs) for source in xc])
            totals = [
                sum(
                    (w[i] * (ys[i] - base) for i in range(len(xs))), np.zeros_like(base)
                )
            ]
            within, between = np.zeros_like(base), np.zeros_like(base)
            for key in group_keys:
                ids = np.flatnonzero(groups == key)
                xg, yg = xs[ids].mean(0), ys[ids].mean(0)
                within += sum(
                    (
                        (ys[i] - yg) * np.dot(xs[i] - xg, point - origin) / len(xs)
                        for i in ids
                    ),
                    np.zeros_like(base),
                )
                between += (
                    len(ids)
                    / len(xs)
                    * (yg - base)
                    * np.dot(xg - origin, point - origin)
                )
            expected.append(np.stack(totals + [within, between]))
        expected = np.array(expected)
        errors["cold_direct_source_sum"] = max(
            errors["cold_direct_source_sum"], float(np.max(abs(expected - whole_cold)))
        )
        np.testing.assert_allclose(expected, whole_cold, rtol=1e-12, atol=1e-12)
        checks["cold_values"] += expected.size
        dx = H.paired(xa, pairs["application"])
        for i, displacement in enumerate(dx):
            individual = np.array(
                [np.dot(source, displacement) / len(xs) for source in xc]
            )
            np.testing.assert_allclose(weights[i, vi], individual, atol=1e-12)
            for j, key in enumerate(group_keys):
                ids = np.flatnonzero(groups == key)
                value = sum(
                    (individual[e] * (ys[e] - base) for e in ids), np.zeros_like(base)
                )
                value = np.array(
                    [
                        (value[..., 0] + value[..., 1]) / 2,
                        (value[..., 0] - value[..., 1]) / 2,
                    ]
                ).transpose(1, 2, 0)
                errors["ledger_direct_group_sum"] = max(
                    errors["ledger_direct_group_sum"],
                    float(np.max(abs(value - ledger[i, vi, 0, j]))),
                )
                np.testing.assert_allclose(
                    value, ledger[i, vi, 0, j], rtol=1e-12, atol=1e-12
                )
                checks["ledger_values"] += value.size
        errors["ledger_components"] = max(
            errors["ledger_components"],
            float(np.max(abs(ledger[:, vi, 0] - ledger[:, vi, 1] - ledger[:, vi, 2]))),
            float(np.max(abs(ledger[:, vi].sum(2) - cold[:, vi]))),
        )
        all_x = np.concatenate([H.paired(xs, pairs["source"]), dx])
        for row in distance_rows:
            if row["view"] != view:
                continue
            ids = np.flatnonzero(role_names == row["role"])
            channel = H.R.CHANNELS.index(row["channel"])
            field = ("even", "odd").index(row["order_field"])
            x, y, g = all_x[ids], all_y[ids, channel, :, field], all_groups[ids]
            a, b = direct_center(x), direct_center(y)
            pooled = float(np.trace(a @ b) / len(ids) ** 2)
            blocked = 0.0
            block_matrices = []
            for key in np.unique(g):
                local = np.flatnonzero(g == key)
                aa, bb = direct_center(x[local]), direct_center(y[local])
                blocked += len(local) / len(ids) * np.trace(aa @ bb) / len(local) ** 2
                block_matrices.append((local, aa, bb))
            expected_stats = [pooled, float(blocked)]
            for mode, value in zip(("pooled", "blocked"), expected_stats, strict=True):
                saved_value = float(row[mode + "_distance_covariance_squared"])
                errors["distance_statistics"] = max(
                    errors["distance_statistics"], abs(value - saved_value)
                )
                np.testing.assert_allclose(value, saved_value, atol=1e-12, rtol=1e-12)
                checks["dependence_accounts"] += 1
            null = nulls[row["role"]][vi, channel, field]
            for pi, p in enumerate(permutations[row["role"]]):
                np.testing.assert_array_equal(g[p], g)
                np.testing.assert_array_equal(np.sort(p), np.arange(len(ids)))
                one = [float(np.trace(a @ b[np.ix_(p, p)]) / len(ids) ** 2), 0.0]
                for local, aa, bb in block_matrices:
                    lookup = {int(value): k for k, value in enumerate(local)}
                    mapped = np.array([lookup[int(value)] for value in p[local]])
                    one[1] += (
                        len(local)
                        / len(ids)
                        * np.trace(aa @ bb[np.ix_(mapped, mapped)])
                        / len(local) ** 2
                    )
                errors["permutation_statistics"] = max(
                    errors["permutation_statistics"],
                    float(np.max(abs(np.array(one) - null[pi]))),
                )
                np.testing.assert_allclose(one, null[pi], atol=1e-12, rtol=1e-12)
                checks["null_accounts"] += 2
    if len(distance_rows) != 36 or max(errors.values()) > 1e-12:
        raise AssertionError(
            "Registered scientific accounts did not pass alternate audit"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "errors": errors,
                "checks": checks,
                "new_forwards": 0,
                "same_agent_not_blind": True,
            }
        ),
        flush=True,
    )


def main():
    """Verify an existing immutable package without replacing any artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=H.DEFAULT)
    args = parser.parse_args()
    verify(args.out.resolve())


if __name__ == "__main__":
    main()
