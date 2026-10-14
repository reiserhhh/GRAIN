#!/usr/bin/env python3
"""Rebuild HM90 with dense boundary solves and direct sums, no neural calls."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_cold_evidence import rotate
from scripts.run_grain_hierarchical_query_hm88 import load, sha, verify, write
from scripts.run_grain_query_evidence_hm86 import unit

OUT = ROOT / "results/grain_math_response_v1/channel_boundary_transport_r1"


def table(path):
    """Read every primary numerical row without importing its producer."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def probability(scores):
    """Normalize explicit exponential weights without primary softmax code."""
    p = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def solve(t, labels, direct=False):
    """Use dense linear equations instead of the primary forward recursion."""
    n = len(labels)
    b = np.zeros((n, 3))
    boundary, interior = np.flatnonzero(labels >= 0), np.flatnonzero(labels < 0)
    b[boundary, labels[boundary]] = 1
    rhs = t[np.ix_(interior, boundary)] @ b[boundary]
    if direct:
        rhs[:, 2] = 1 - rhs[:, :2].sum(1)
        b[interior] = rhs
    else:
        rhs[:, 2] += 1 - t[interior].sum(1)
        b[interior] = np.linalg.solve(
            np.eye(len(interior)) - t[np.ix_(interior, interior)], rhs
        )
    return b


def run():
    """Check all new vector fields, graph decompositions and author consequences."""
    manifest = load(OUT / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(OUT, manifest["inputs"])
    verify(OUT, load(OUT / "01/raw_seal.json"))
    verify(OUT, load(OUT / "02/output_seal.json"))
    bundle, rows = load(OUT / "00/bundle.json"), load(OUT / "00/rows.json")
    with np.load(OUT / "02/projections.npz", allow_pickle=False) as file:
        original, original_raw = file["fields"].copy(), file["raw_fields"].copy()
    fields, raw_fields = np.zeros_like(original), np.zeros_like(original_raw)
    accounts = {
        (int(r["unit"]), r["readout"], r["kind"]): r
        for r in table(OUT / "02/conditioning_accounts.csv")
    }
    graphs = {int(r["unit"]): r for r in table(OUT / "02/graph_collapse_accounts.csv")}
    roles = {
        (int(r["unit"]), r["kind"], int(r["event"])): r
        for r in table(OUT / "02/token_role_channels.csv")
    }
    points = list(itertools.product((-1, 1), repeat=4))
    errors = {
        key: 0.0
        for key in ("fields", "raw_fields", "conditioning", "graph", "role", "author")
    }
    checks = 0

    def compare(expected, actual, kind):
        nonlocal checks
        expected, actual = np.asarray(expected), np.asarray(actual)
        errors[kind] = max(errors[kind], float(np.max(np.abs(expected - actual))))
        checks += expected.size

    with threadpool_limits(limits=1):
        for row in rows:
            raw = unit(OUT, row["unit"])
            pos, labels = row["body_indices"], np.array(row["root_labels"])
            k, v, q = (
                raw["k"][pos].astype(float),
                raw["v"][pos].astype(float),
                raw["q_body"].astype(float),
            )
            n, heads = len(pos), q.shape[1]
            scores = np.stack(
                [q[:, h] @ k[:, h // (heads // k.shape[1])].T for h in range(heads)]
            )
            scores[:, np.triu_indices(n, 1)[0], np.triu_indices(n, 1)[1]] = -np.inf
            full = probability(scores)
            strict = full.copy()
            strict[:, np.arange(n), np.arange(n)] = 0
            t = np.divide(
                strict,
                strict.sum(2)[:, :, None],
                out=np.zeros_like(strict),
                where=strict.sum(2)[:, :, None] > 0,
            )
            mean_w = full.mean(0)
            mean_w[np.arange(n), np.arange(n)] = 0
            scalar = np.divide(
                mean_w,
                mean_w.sum(1)[:, None],
                out=np.zeros_like(mean_w),
                where=mean_w.sum(1)[:, None] > 0,
            )
            laws = {
                kind: np.array([solve(head, labels, kind == "direct") for head in t])
                for kind in ("direct", "harmonic")
            }
            gates = {
                "cold": np.ones((heads, n)),
                "scalar_direct": np.broadcast_to(
                    solve(scalar, labels, True)[:, 0], (heads, n)
                ),
                "scalar_harmonic": np.broadcast_to(
                    solve(scalar, labels)[:, 0], (heads, n)
                ),
            }
            for kind, law in laws.items():
                gates[kind + "_paired"] = law[:, :, 0]
                gates[kind + "_average_law"] = np.broadcast_to(
                    law[:, :, 0].mean(0), (heads, n)
                )
                for shift in range(1, heads):
                    gates[f"{kind}_shift_{shift:02d}"] = np.roll(
                        law[:, :, 0], shift, axis=0
                    )
                for event, span in enumerate(row["action_groups"]):
                    alignment = (law[:, span, 0] - law[:, span, 1]).mean(1)
                    record = roles[row["unit"], kind, event]
                    compare(
                        [
                            alignment.mean() * row["coordinates"][event + 2],
                            alignment.std(),
                            law[:, span, 2].mean(),
                        ],
                        [
                            float(record[key])
                            for key in (
                                "mean_actor_alignment",
                                "head_allocation_std",
                                "unresolved",
                            )
                        ],
                        "role",
                    )
            averaged_t, averaged_h = t.mean(0), laws["harmonic"].mean(0)
            scalar_h = solve(averaged_t, labels)
            delta = averaged_h - scalar_h
            covariance = (
                sum(
                    (head - averaged_t) @ (law - averaged_h)
                    for head, law in zip(t, laws["harmonic"], strict=True)
                )
                / heads
            )
            identity = (delta - averaged_t @ delta - covariance)[labels < 0]
            if np.max(np.abs(identity)) > 1e-10:
                raise ValueError("Solving covariance identity failed")
            graph = graphs[row["unit"]]
            compare(
                [
                    np.sqrt(np.mean(delta**2)),
                    np.sqrt(np.mean(covariance**2)),
                    np.sqrt(np.mean((scalar - averaged_t) ** 2)),
                ],
                [
                    float(graph[key])
                    for key in (
                        "solving_collapse_rms",
                        "solving_covariance_rms",
                        "normalizing_collapse_rms",
                    )
                ],
                "graph",
            )
            fi, oi, ti, ci = (
                manifest["spec"]["frames"].index(row["frame"]),
                row["order"],
                bundle["targets"].index(row["target"]),
                points.index(tuple(row["coordinates"])),
            )
            for qi, target in enumerate(bundle["targets"]):
                seed = rotate(
                    bundle["raw_queries"][qi],
                    pos[-1] + bundle["gaps"][qi],
                    bundle["frequencies"],
                    bundle["scaling"],
                )
                kernel = probability(
                    np.array(
                        [
                            k[:, h // (heads // k.shape[1])] @ seed[h]
                            for h in range(heads)
                        ]
                    )
                )
                values = np.array(
                    [
                        np.sum(
                            v[:, h // (heads // v.shape[1])]
                            * np.array(bundle["directions"])[qi, h],
                            axis=1,
                        )
                        for h in range(heads)
                    ]
                )
                readings = {}
                for ai, arm in enumerate(manifest["arms"]):
                    weighted = kernel * gates[arm]
                    mass = weighted.sum(1)
                    numerator = np.sum(weighted * values, axis=1)
                    readings[arm] = (mass, numerator)
                    fields[fi, oi, ti, ci, ai, qi] = np.sum(numerator / mass)
                    raw_fields[fi, oi, ti, ci, ai, qi] = numerator.sum()
                for kind in laws:
                    mp, np_ = readings[kind + "_paired"]
                    mm, nm = readings[kind + "_average_law"]
                    nt, mt = (np_ - nm) / mp, -(mp - mm) * (nm / mm) / mp
                    orbit = [
                        readings[kind + "_paired"],
                        *(readings[f"{kind}_shift_{i:02d}"] for i in range(1, heads)),
                    ]
                    remainder = np.mean([np.sum(b / a) for a, b in orbit]) - np.sum(
                        nm / mm
                    )
                    record = accounts[row["unit"], target, kind]
                    expected = [
                        nt.sum(),
                        mt.sum(),
                        np.sum(np_ / mp - nm / mm),
                        np_.sum(),
                        nm.sum(),
                        mp.mean(),
                        mm.mean(),
                        remainder,
                    ]
                    keys = (
                        "numerator_term",
                        "normalization_term",
                        "difference",
                        "paired_raw_action",
                        "mean_raw_action",
                        "paired_mean_mass",
                        "average_mean_mass",
                        "orbit_conditional_remainder",
                    )
                    compare(
                        expected, [float(record[key]) for key in keys], "conditioning"
                    )
                    if (
                        max(
                            np.max(np.abs(np.mean([a for a, b in orbit], 0) - mm)),
                            np.max(np.abs(np.mean([b for a, b in orbit], 0) - nm)),
                        )
                        > 1e-10
                    ):
                        raise ValueError("Cyclic numerator/mass identity failed")
        compare(fields, original, "fields")
        compare(raw_fields, original_raw, "raw_fields")
    point_array = np.array(points)
    z = (
        point_array[:, 0] * point_array[:, 2] + point_array[:, 1] * point_array[:, 3]
    ) / 2
    difference = 2 * manifest["spec"]["epsilon"] * z / 16
    for row in table(OUT / "02/author_law_contrasts.csv"):
        index = (
            manifest["spec"]["frames"].index(row["frame"]),
            int(row["order"]),
            bundle["targets"].index(row["target"]),
        )
        ai, qi = (
            manifest["arms"].index(row["arm"]),
            bundle["targets"].index(row["readout"]),
        )
        compare(
            [
                difference @ fields[index][:, ai, qi],
                difference @ raw_fields[index][:, ai, qi],
            ],
            [
                float(row["author_conditional_contrast"]),
                float(row["author_raw_contrast"]),
            ],
            "author",
        )
    if max(errors.values()) > 1e-9:
        raise ValueError(errors)
    result = {
        "status": "VERIFIED_FINITE_HM90_NOT_GLOBAL",
        "errors": errors,
        "checks": checks,
        "new_model_calls": 0,
        "same_agent_alternate_formula_not_third_party": True,
        "code_sha256": sha(Path(__file__)),
        "original_scope_complete": False,
    }
    write(OUT / "verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    run()
