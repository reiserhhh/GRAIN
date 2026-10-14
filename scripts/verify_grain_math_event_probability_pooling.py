#!/usr/bin/env python3
"""Independently bind HM57 tables to raw JSON and alternate categorical formulas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_math_response_v1/event_probability_pooling_r1"


def read(path):
    """Read original JSON independently of the analysis module."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Compute an independent file digest for all frozen source links."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(out, name):
    """Read one complete result table."""
    with (out / f"03/{name}.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def softmax(a):
    """Use scipy logsumexp instead of the runner's max-shift normalization."""
    return np.exp(a - logsumexp(a, axis=-1, keepdims=True))


def compare(actual, expected, maximum):
    """Check every expected field, exact identity/nulls and finite numeric values."""
    for k, v in expected.items():
        if k not in actual:
            raise ValueError("Missing field: " + k)
        got = actual[k]
        if v is None:
            if got not in (None, ""):
                raise ValueError("Noncollision was given a lower bound")
        elif isinstance(v, (bool, str)):
            if str(got) != str(v):
                raise ValueError(f"Identity mismatch {k}: {got} != {v}")
        else:
            value = float(got)
            if not np.isfinite(value):
                raise ValueError("Nonfinite verified field")
            maximum = max(maximum, abs(value - float(v)))
    return maximum


def verify(out):
    """Reconstruct all references, paired collisions and summaries from raw logits."""
    m = read(out / "00/manifest.json")
    spec, v = m["spec"], np.asarray(m["values"])
    for name, expected in m["code"].items():
        if sha(ROOT / name) != expected or sha(out / "00/code" / name) != expected:
            raise ValueError("Frozen HM57 code drift")
    for name, expected in m["assets"].items():
        if sha(out / name) != expected:
            raise ValueError("Frozen input drift")
    for name, expected in m["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Source provenance drift")
    for name, expected in read(out / "03/artifact_hashes.json").items():
        if sha(out / name) != expected:
            raise ValueError("Result artifact drift")
    design = read(out / "00/rows.json")
    if design != read(ROOT / spec["source"] / "00/rows.json"):
        raise ValueError("Source design snapshot changed")
    paths, snapshot = (
        read(out / "00/raw_paths.json"),
        np.load(out / "00/raw_logits.npy", allow_pickle=False),
    )
    if len(design) != 1296 or snapshot.shape != (1296, 10) or len(paths) != 1296:
        raise ValueError("Incomplete full raw roster")
    atoms, accounts, native, maximum = {}, {}, {}, 0.0
    for r, path, frozen in zip(design, paths, snapshot):
        original = read(ROOT / path)
        a = np.asarray(original["native"]["candidate_logits"], float)
        if not np.array_equal(a, frozen):
            raise ValueError("Raw logit snapshot differs")
        if r["reused"]:
            if (
                original["unit"] != r["parent_unit"]
                or original["target"] != r["query"]
                or r["query"] != r["content"]
            ):
                raise ValueError("Reused query identity mismatch")
        elif any(
            original[k] != r[k]
            for k in (
                "unit",
                "parent_unit",
                "role",
                "content",
                "query",
                "family",
                "polarity",
                "order",
                "regime",
            )
        ):
            raise ValueError("Fresh query identity mismatch")
        n = softmax(a)
        maximum = max(
            maximum,
            abs(float(n @ v) - original["native"]["conditional_score"]),
            float(np.max(np.abs(n - original["native"]["conditional_probabilities"]))),
        )
        native[r["unit"]] = n
        if r["role"] == "atom":
            key = r["content"], r["family"], r["polarity"], r["query"]
            if key in atoms:
                raise ValueError("Repeated atom key")
            atoms[key] = a
        else:
            group = accounts.setdefault(r["parent_unit"], {})
            if r["query"] in group:
                raise ValueError("Repeated account/query")
            group[r["query"]] = r
    if len(atoms) != 144 or len(accounts) != 192:
        raise ValueError("Wrong atom/account scope")
    scalar_index = {
        (int(r["parent_unit"]), r["query"], r["basis"]): r
        for r in rows(out, "scalar_decomposition")
    }
    probability_index = {
        (int(r["parent_unit"]), r["query"], r["basis"], int(r["digit"])): r
        for r in rows(out, "probability_decomposition")
    }
    if len(scalar_index) != 2304 or len(probability_index) != 23040:
        raise ValueError("Duplicate/missing scalar or probability row")
    expected_scalars, vectors = [], {}
    for unit, group in accounts.items():
        if set(group) != set(m["targets"]):
            raise ValueError("Missing target crossing")
        r = group[m["targets"][0]]
        if any(
            any(
                doc[k] != r[k]
                for k in (
                    "content",
                    "family",
                    "polarity",
                    "order",
                    "regime",
                    "body",
                    "history",
                )
            )
            for doc in group.values()
        ):
            raise ValueError("Query profile mixed different accounts")
        base = {
            k: r[k]
            for k in ("parent_unit", "content", "family", "polarity", "order", "regime")
        }
        n_all = np.array([native[group[q]["unit"]] for q in m["targets"]])
        state = {"parent_unit": unit, "native": n_all}
        for basis in spec["bases"]:
            # Independent replay of the four-regime new/alias design, not active_records.
            history, seen, event_ids = r["history"], {}, set()
            for h in history:
                if h["kind"] not in {"new", "alias"} or h["record_id"] in seen:
                    raise ValueError("Unexpected finite history operation")
                if h["kind"] == "new":
                    if h["event_id"] in event_ids:
                        raise ValueError("Duplicated new occasion")
                    event_ids.add(h["event_id"])
                elif (
                    h["parent_id"] not in seen
                    or h["event_id"] != seen[h["parent_id"]]["event_id"]
                ):
                    raise ValueError("Invalid alias identity")
                seen[h["record_id"]] = h
            selected = [
                h
                for h in history
                if h["actor"] == "A" and (basis == "mention" or h["kind"] == "new")
            ]
            counts = Counter((h["family"], h["polarity"]) for h in selected)
            keys, w = (
                sorted(counts),
                np.array([counts[k] / len(selected) for k in sorted(counts)]),
            )
            b_all, g_all = [], []
            for j, q in enumerate(m["targets"]):
                a = np.array([atoms[r["content"], f, p, q] for f, p in keys])
                p, n = softmax(a), n_all[j]
                b = np.sum(w[:, None] * p, axis=0)
                unnormalized = np.prod(p ** w[:, None], axis=0)
                g = unnormalized / np.sum(unnormalized)
                ns, bs, gs = (float(z @ v) for z in (n, b, g))
                pool, residual, total = gs - bs, ns - gs, ns - bs
                forward = float(
                    np.sum(w * np.sum(p * (np.log(p) - np.log(n)), axis=1))
                    - np.sum(w * np.sum(p * (np.log(p) - np.log(b)), axis=1))
                )
                reverse = float(
                    np.sum(w * np.sum(n * (np.log(n) - np.log(p)), axis=1))
                    - np.sum(w * np.sum(g * (np.log(g) - np.log(p)), axis=1))
                )
                ends = p @ v
                weight = w[keys.index((r["family"], r["polarity"]))]
                dx = (
                    atoms[r["content"], r["family"], r["polarity"], q]
                    - atoms[r["content"], 1 - r["family"], 1 - r["polarity"], q]
                )
                mean, variance = float(g @ dx), float(g @ (dx - float(g @ dx)) ** 2)
                first = g * (dx - mean)
                second = g * ((dx - mean) ** 2 - variance)
                expected = base | {
                    "query": q,
                    "basis": basis,
                    "count": len(selected),
                    "focal_weight": weight,
                    "native": ns,
                    "mixture": bs,
                    "geometric": gs,
                    "pooling_change": pool,
                    "fixed_reference_residual": residual,
                    "total_change": total,
                    "score_cross": 2 * pool * residual,
                    "score_identity_error": abs(total - pool - residual),
                    "score_energy_error": abs(
                        total**2 - pool**2 - residual**2 - 2 * pool * residual
                    ),
                    "native_mixture_tv": float(np.sum(np.abs(n - b)) / 2),
                    "native_geometric_tv": float(np.sum(np.abs(n - g)) / 2),
                    "pooling_tv": float(np.sum(np.abs(g - b)) / 2),
                    "probability_cross": float(2 * np.sum((g - b) * (n - g))),
                    "probability_energy_error": abs(
                        float(
                            np.sum((n - b) ** 2)
                            - np.sum((g - b) ** 2)
                            - np.sum((n - g) ** 2)
                            - 2 * np.sum((g - b) * (n - g))
                        )
                    ),
                    "mixture_kl_identity_error": abs(
                        forward - float(np.sum(b * (np.log(b) - np.log(n))))
                    ),
                    "geometric_kl_identity_error": abs(
                        reverse - float(np.sum(n * (np.log(n) - np.log(g))))
                    ),
                    "mixture_objective_gap": forward,
                    "geometric_objective_gap": reverse,
                    "geometric_outside_endpoints": bool(
                        gs < min(ends) - spec["arithmetic_tolerance"]
                        or gs > max(ends) + spec["arithmetic_tolerance"]
                    ),
                    "native_outside_endpoints": bool(
                        ns < min(ends) - spec["arithmetic_tolerance"]
                        or ns > max(ends) + spec["arithmetic_tolerance"]
                    ),
                    "e_path_first": float(v @ first),
                    "e_path_second": float(v @ second),
                    "e_path_identity_error": 0.0,
                }
                maximum = compare(scalar_index[unit, q, basis], expected, maximum)
                expected_scalars.append(expected)
                for digit in range(10):
                    expected_p = base | {
                        "query": q,
                        "basis": basis,
                        "digit": digit,
                        "native": n[digit],
                        "mixture": b[digit],
                        "geometric": g[digit],
                        "pooling_change": g[digit] - b[digit],
                        "fixed_reference_residual": n[digit] - g[digit],
                        "total_change": n[digit] - b[digit],
                        "cross": 2 * (g[digit] - b[digit]) * (n[digit] - g[digit]),
                    }
                    maximum = compare(
                        probability_index[unit, q, basis, digit], expected_p, maximum
                    )
                b_all.append(b)
                g_all.append(g)
            state[basis] = {"mixture": np.array(b_all), "geometric": np.array(g_all)}
        vectors[r["content"], r["family"], r["polarity"], r["order"], r["regime"]] = (
            state
        )
    pairs = []
    for c, f, p, order in itertools.product(m["targets"], (0, 1), (0, 1), (0, 1)):
        for name, hi, lo in spec["contrasts"]:
            pairs.append(
                (
                    {
                        "content": c,
                        "family": f,
                        "polarity": p,
                        "order": order,
                        "contrast": name,
                    },
                    vectors[c, f, p, order, hi],
                    vectors[c, f, p, order, lo],
                )
            )
    for c, f, p, regime in itertools.product(
        m["targets"], (0, 1), (0, 1), spec["regimes"]
    ):
        pairs.append(
            (
                {
                    "content": c,
                    "family": f,
                    "polarity": p,
                    "order": -1,
                    "contrast": "order_reversed_" + regime,
                },
                vectors[c, f, p, 1, regime],
                vectors[c, f, p, 0, regime],
            )
        )
    collision_index = {
        (r["contrast"], int(r["hi"]), int(r["lo"]), r["basis"], r["pool"]): r
        for r in rows(out, "collisions")
    }
    change_index = {
        (r["contrast"], int(r["hi"]), int(r["lo"]), r["basis"], r["query"]): r
        for r in rows(out, "paired_changes")
    }
    if len(pairs) != 240 or len(collision_index) != 960 or len(change_index) != 2880:
        raise ValueError("Missing complete paired roster")
    expected_collisions = []
    for base, hi, lo in pairs:
        base |= {"hi": hi["parent_unit"], "lo": lo["parent_unit"]}
        delta, dp = (hi["native"] - lo["native"]) @ v, hi["native"] - lo["native"]
        for basis, pool in itertools.product(spec["bases"], ("mixture", "geometric")):
            diff = float(np.max(np.abs(hi[basis][pool] - lo[basis][pool])))
            same = diff <= spec["summary_equality_tolerance"]
            expected = base | {
                "basis": basis,
                "pool": pool,
                "summary_max_difference": diff,
                "same_summary": same,
                "native_profile_change_norm": float(np.sqrt(np.sum(delta**2))),
                "native_probability_tv_mean": float(
                    np.mean(np.sum(np.abs(dp), axis=1) / 2)
                ),
                "minimum_common_profile_rms": float(np.sqrt(np.sum(delta**2)) / 2)
                if same
                else None,
                "minimum_common_coordinate_rms": float(np.sqrt(np.mean(delta**2)) / 2)
                if same
                else None,
                "minimum_common_probability_coordinate_rms": float(
                    np.sqrt(np.mean(dp**2)) / 2
                )
                if same
                else None,
            }
            maximum = compare(
                collision_index[base["contrast"], base["hi"], base["lo"], basis, pool],
                expected,
                maximum,
            )
            expected_collisions.append(expected)
        for basis, qj in itertools.product(spec["bases"], range(6)):
            b, g = ((hi[basis][k] - lo[basis][k]) @ v for k in ("mixture", "geometric"))
            expected = base | {
                "basis": basis,
                "query": m["targets"][qj],
                "native_change": delta[qj],
                "mixture_change": b[qj],
                "geometric_change": g[qj],
                "pooling_change": g[qj] - b[qj],
                "fixed_reference_residual": delta[qj] - g[qj],
                "total_change": delta[qj] - b[qj],
                "cross": 2 * (g[qj] - b[qj]) * (delta[qj] - g[qj]),
            }
            maximum = compare(
                change_index[
                    base["contrast"], base["hi"], base["lo"], basis, m["targets"][qj]
                ],
                expected,
                maximum,
            )
    grouped = defaultdict(list)
    for r in expected_scalars:
        grouped[r["basis"], r["regime"], r["query"]].append(r)
        grouped[r["basis"], r["regime"], "ALL"].append(r)
    expected_summary = {}
    for key, group in grouped.items():
        e = {
            "basis": key[0],
            "regime": key[1],
            "query": key[2],
            "scalar_rows": len(group),
        }
        for name, field in (
            ("native_minus_mixture_rms", "total_change"),
            ("geometric_minus_mixture_rms", "pooling_change"),
            ("native_minus_geometric_rms", "fixed_reference_residual"),
        ):
            e[name] = float(np.sqrt(sum(r[field] ** 2 for r in group) / len(group)))
        for name, field in (
            ("signed_cross_mean", "score_cross"),
            ("native_mixture_tv_mean", "native_mixture_tv"),
            ("native_geometric_tv_mean", "native_geometric_tv"),
        ):
            e[name] = sum(r[field] for r in group) / len(group)
        e["geometric_outside_scalar_endpoints"] = sum(
            r["geometric_outside_endpoints"] for r in group
        )
        e["native_outside_scalar_endpoints"] = sum(
            r["native_outside_endpoints"] for r in group
        )
        expected_summary[key] = e
    got = rows(out, "pooling_summary")
    if len(got) != 56:
        raise ValueError("Missing all query/regime summaries")
    for r in got:
        maximum = compare(
            r, expected_summary[r["basis"], r["regime"], r["query"]], maximum
        )
    collision_group = defaultdict(list)
    for r in expected_collisions:
        collision_group[r["contrast"], r["basis"], r["pool"]].append(r)
    expected_cs = {}
    for key, group in collision_group.items():
        same = [r for r in group if r["same_summary"]]
        e = {
            "contrast": key[0],
            "basis": key[1],
            "pool": key[2],
            "paired_cases": len(group),
            "same_summary_count": len(same),
            "same_summary_native_changes": sum(
                r["native_profile_change_norm"] > spec["arithmetic_tolerance"]
                for r in same
            ),
            "maximum_summary_difference": max(
                r["summary_max_difference"] for r in group
            ),
            "native_profile_change_rms": float(
                np.sqrt(
                    sum(r["native_profile_change_norm"] ** 2 for r in group)
                    / len(group)
                )
            ),
            "minimum_common_profile_rms": float(
                np.sqrt(
                    sum(r["minimum_common_profile_rms"] ** 2 for r in same) / len(same)
                )
            )
            if same
            else None,
            "native_probability_tv_mean": sum(
                r["native_probability_tv_mean"] for r in group
            )
            / len(group),
        }
        expected_cs[key] = e
    got = rows(out, "collision_summary")
    if len(got) != 28:
        raise ValueError("Missing collision condition summaries")
    for r in got:
        maximum = compare(r, expected_cs[r["contrast"], r["basis"], r["pool"]], maximum)
    summary = read(out / "03/summary.json")
    if len(summary["pooling_results"]) != 56 or len(summary["collision_results"]) != 28:
        raise ValueError("Incomplete report summary")
    for r in summary["pooling_results"]:
        maximum = compare(
            r, expected_summary[r["basis"], r["regime"], r["query"]], maximum
        )
    for r in summary["collision_results"]:
        maximum = compare(r, expected_cs[r["contrast"], r["basis"], r["pool"]], maximum)
    maximum = compare(
        summary,
        {
            "source_readouts": 1296,
            "accounts": 192,
            "scalar_decompositions": 2304,
            "probability_coordinates": 23040,
            "paired_change_coordinates": 2880,
            "collision_certificates": 960,
            "new_neural_calls": 0,
            "fit_calls": 0,
            "API_calls": 0,
            "original_scope_complete": False,
        },
        maximum,
    )
    status = read(out / "run_status.json")
    if status["status"] != "ANALYZED" or any(
        status[k] != 0 for k in ("new_neural_calls", "fit_calls", "API_calls")
    ):
        raise ValueError("Packet not terminal zero-call")
    if maximum > spec["arithmetic_tolerance"]:
        raise ValueError(f"Alternate arithmetic error {maximum}")
    result = {
        "status": "VERIFIED_FINITE_HM57_NOT_GLOBAL",
        "raw_readouts": 1296,
        "scalar_decompositions": 2304,
        "probability_coordinates": 23040,
        "paired_changes": 2880,
        "collision_certificates": 960,
        "summary_rows": 84,
        "maximum_alternate_error": maximum,
        "new_neural_calls": 0,
        "fit_calls": 0,
        "API_calls": 0,
        "independent_method_replication": False,
        "original_scope_complete": False,
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


def main():
    """Verify one terminal run, without changing any scientific result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
