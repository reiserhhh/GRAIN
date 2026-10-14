#!/usr/bin/env python3
"""HM30 alternate rational maps, explicit balanced projections and full score replay."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import check
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_position_support import rational_inverse, score

PARENT = ROOT / "results/grain_math_response_v1/crossed_wording_context_r1"
OUT = ROOT / "results/grain_math_response_v1/crossed_wording_context_verification_r1"


def table(path):
    """Read exactly one declared rectangular table, not resource-fork globs."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def alternate_parts(x):
    """Apply explicit product-space averaging matrices instead of primary means."""
    nc, nw = x.shape[:2]
    jc, jp = np.ones((nc, nc)) / nc, np.ones((nw, nw)) / nw
    context_average = np.einsum("ab,bp...->ap...", jc, x)
    wording_average = np.einsum("ab,cb...->ca...", jp, x)
    both = np.einsum("ab,bp...->ap...", jc, wording_average)
    c, p = wording_average - both, context_average - both
    residual = x - wording_average - context_average + both
    centered = x - both
    square = lambda v: float(np.sum(v * v) / (nc * nw))
    total = square(centered)
    ss = {"context": square(c), "wording": square(p), "interaction": square(residual)}
    mean_energy, all_energy = square(both), square(x)
    summary = {
        **{name + "_energy": v for name, v in ss.items()},
        "total_centered_energy": total,
        "mean_energy": mean_energy,
        "mean_square_energy": all_energy,
        "cancellation_ratio": mean_energy / all_energy if all_energy else None,
        "reconstruction_error": float(np.max(abs(x - both - c - p - residual))),
        "energy_identity_error": abs(total - sum(ss.values())),
        **{name + "_share": v / total if total else None for name, v in ss.items()},
    }
    return {
        "mean": both[0, 0],
        "context": c[:, 0],
        "wording": p[0],
        "interaction": residual,
    }, summary


def verify(parent, out):
    """Reconstruct every field, native endpoint, action and reported decomposition."""
    if out.exists():
        raise FileExistsError("preserve existing verifier output")
    m = check(parent)
    spec = m["spec"]
    ledger = read_json(parent / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(parent / name) != sha:
            raise ValueError("frozen output changed " + name)
    design, docs = (
        read_json(parent / "00/design.json"),
        read_json(parent / "00/documents.json"),
    )
    if len(docs) != 32 or len({d["text"] for d in docs}) != 32:
        raise ValueError("incomplete or duplicate material crossing")
    parent_texts = {
        d["text"] for d in read_json(ROOT / spec["parent"] / "00/documents.json")
    }
    if any(d["text"] in parent_texts for d in docs):
        raise ValueError("new material overlaps parent")
    expected_cells = set(itertools.product(range(8), range(4), spec["queries"]))
    if {
        (r["context"], r["wording"], r["query"]) for r in design
    } != expected_cells or len(design) != 96:
        raise ValueError("crossed reference design incomplete")
    calls = read_json(parent / "run_status.json")
    if (calls["forward_calls"], calls["backward_calls"], calls["in_flight"]) != (
        480,
        96,
        None,
    ):
        raise ValueError("actual call ledger incomplete")
    exact_matrix, exact_inverse = rational_inverse()
    matrix, inverse = np.array(exact_matrix, float), np.array(exact_inverse, float)
    source = read_json(parent / "01/source_manifest.json")
    if len(source) != 15:
        raise ValueError("15 frozen source fields required")
    errors = {
        k: 0.0
        for k in (
            "native",
            "field",
            "action",
            "metric",
            "array",
            "decomposition",
            "covariance",
            "hook",
        )
    }

    def compare(kind, actual, reported):
        if actual is None:
            if reported not in (None, ""):
                raise ValueError("zero-energy statistic was fabricated")
        else:
            errors[kind] = max(
                errors[kind],
                float(
                    np.max(abs(np.asarray(actual, float) - np.asarray(reported, float)))
                ),
            )

    baselines = {}
    for r in design:
        unit = r["unit"]
        p = parent / "01/baselines" / f"unit_{unit:03d}.json"
        meta = read_json(p)
        expected_role = (
            "source"
            if r["context"] < 4
            else "validation"
            if r["context"] < 6
            else "application"
        )
        if (
            r["role"] != expected_role
            or meta["address"] != r
            or digest(p.with_suffix(".npz")) != meta["array_sha256"]
        ):
            raise ValueError("reference role, address or array hash mismatch")
        if len(r["spans"]["A_property"]["positions"]) != 4:
            raise ValueError("property tokens changed")
        compare(
            "native",
            score(meta["native"], m["alphabet"]["values"]),
            meta["native"]["conditional_score"],
        )
        if r["role"] == "source":
            if meta["source_sha256"] is not None or any(
                datetime.fromisoformat(meta["acquired_at"])
                >= datetime.fromisoformat(e["frozen_at"])
                for e in source.values()
            ):
                raise ValueError("source acquisition did not precede fit")
        elif meta["source_sha256"] != digest(parent / "01/source_manifest.json") or any(
            datetime.fromisoformat(meta["acquired_at"])
            <= datetime.fromisoformat(e["frozen_at"])
            for e in source.values()
        ):
            raise ValueError("held acquisition did not follow all source fits")
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as data:
            baselines[unit] = {
                "meta": meta,
                "h": data["h"].copy(),
                "g": data["gradient"].astype(float),
            }
    fits = {}
    for q in spec["queries"]:
        for w in (-1, 0, 1, 2, 3):
            key = q + ("_global" if w == -1 else f"_wording{w}")
            entry = source[key]
            selected = [
                r
                for r in design
                if r["role"] == "source"
                and r["query"] == q
                and (w == -1 or r["wording"] == w)
            ]
            if entry["source_units"] != [r["unit"] for r in selected] or len(
                selected
            ) != (16 if w == -1 else 4):
                raise ValueError("source field inclusion changed")
            for name, sha in {
                entry["file"]: entry["sha256"],
                **entry["source_hashes"],
            }.items():
                if digest(parent / name) != sha:
                    raise ValueError("source field dependency changed")
            mean_g = sum(baselines[r["unit"]]["g"] for r in selected) / len(selected)
            fits[key] = inverse.T @ mean_g
            with np.load(parent / entry["file"], allow_pickle=False) as data:
                compare("field", fits[key], data["covector"])
                if int(data["rank"]) != 4:
                    raise ValueError("rank4 source missing")
    with np.load(parent / "00/directions.npz", allow_pickle=False) as data:
        bank = {k: data[k].copy() for k in data.files}
    actions = []
    reported = {(int(r["unit"]), r["arm"]): r for r in table(parent / "02/actions.csv")}
    endpoint_table = {
        (int(r["unit"]), r["arm"], r["sign"]): r
        for r in table(parent / "02/endpoints.csv")
    }
    endpoints = 0
    for r in design:
        unit, q = r["unit"], r["query"]
        base = baselines[unit]
        b = bank[q + "_b"] / np.linalg.norm(bank[q + "_b"])
        for arm in spec["arms"]:
            target = [1.0, 0, 0, 0] if arm == "known_mean" else [0.0, -1, 0, 1]
            direction = np.outer(inverse @ target, b)
            delta = (
                direction
                / np.linalg.norm(direction)
                * spec["radius_relative_span_norm"]
                * np.linalg.norm(base["h"])
            )
            plus = (base["h"] + delta.astype(np.float32)).astype(np.float32)
            minus = (base["h"] - delta.astype(np.float32)).astype(np.float32)
            half = (plus.astype(float) - minus.astype(float)) / 2
            moments = sum(np.outer(matrix[:, j], half[j]) for j in range(4))
            scores = {}
            for sign in ("plus", "minus"):
                p = parent / "01/patches" / f"unit_{unit:03d}_{arm}_{sign}.json"
                end = read_json(p)
                scores[sign] = score(end["native"], m["alphabet"]["values"])
                compare("native", scores[sign], end["native"]["conditional_score"])
                compare(
                    "native",
                    scores[sign],
                    endpoint_table[unit, arm, sign]["conditional_score"],
                )
                if (
                    end["address"] != {"unit": unit, "arm": arm, "sign": sign}
                    or end["source_sha256"] != base["meta"]["source_sha256"]
                    or end["baseline_sha256"]
                    != digest(parent / "01/baselines" / f"unit_{unit:03d}.json")
                ):
                    raise ValueError("endpoint exact reference/source address changed")
                if datetime.fromisoformat(end["acquired_at"]) <= datetime.fromisoformat(
                    base["meta"]["acquired_at"]
                ):
                    raise ValueError("endpoint precedes baseline")
                errors["hook"] = max(errors["hook"], *end["audit"].values())
                endpoints += 1
            base_score = score(base["meta"]["native"], m["alphabet"]["values"])
            values = {
                "odd_action": (scores["plus"] - scores["minus"]) / 2,
                "even_change": (scores["plus"] + scores["minus"]) / 2 - base_score,
                "plus_change": scores["plus"] - base_score,
                "minus_change": scores["minus"] - base_score,
                "physical_delta_norm": float(np.sqrt(np.sum(half * half))),
                "global_source": float(np.sum(fits[q + "_global"] * moments)),
                "wording_source": float(
                    np.sum(fits[q + f"_wording{r['wording']}"] * moments)
                ),
                "local_derivative": float(np.sum(base["g"] * half)),
            }
            row = {k: r[k] for k in ("unit", "query", "context", "wording", "role")} | {
                "arm": arm,
                **values,
            }
            actions.append(row)
            for key, value in values.items():
                compare("action", value, reported[unit, arm][key])
    methods = ("global_source", "wording_source", "local_derivative")
    for row in table(parent / "02/metrics.csv"):
        items = [
            r
            for r in actions
            if (r["role"], r["query"], r["arm"])
            == (row["role"], row["query"], row["arm"])
        ]
        y, d = (
            np.array([r["odd_action"] for r in items]),
            np.array([r["physical_delta_norm"] for r in items]),
        )
        rms = lambda v: float(np.sqrt(sum(float(a) ** 2 for a in v) / len(v)))
        values = {
            "cells": len(items),
            "contexts": len({r["context"] for r in items}),
            "wordings": 4,
            "native_action_rms": rms(y),
            "native_unit_action_rms": rms(y / d),
            "even_change_rms": rms([r["even_change"] for r in items]),
        }
        for method in methods:
            p = np.array([r[method] for r in items])
            values.update(
                {
                    method + "_rmse": rms(p - y),
                    method + "_unit_rmse": rms((p - y) / d),
                    method + "_prediction_rms": rms(p),
                    method + "_signs": int(sum(p * y > 0)),
                }
            )
        for key, value in values.items():
            compare("metric", value, row[key])
    expected_arrays = {}
    field_rows, action_rows, cov_rows = (
        table(parent / "02/field_decomposition.csv"),
        table(parent / "02/action_decomposition.csv"),
        table(parent / "02/paired_query_covariance.csv"),
    )
    for role in ("source", "validation", "application"):
        contexts = sorted({r["context"] for r in design if r["role"] == role})
        for q in spec["queries"]:
            lookup = {
                (r["context"], r["wording"]): r
                for r in design
                if r["role"] == role and r["query"] == q
            }
            g = np.array(
                [
                    [baselines[lookup[c, w]["unit"]]["g"] for w in range(4)]
                    for c in contexts
                ]
            )
            lam = np.array(
                [[inverse.T @ g[c, w] for w in range(4)] for c in range(len(contexts))]
            )
            for space, x in (("gradient", g), ("covector", lam)):
                key = role + "_" + q + "_" + space
                parts, stats = alternate_parts(x)
                expected_arrays[key] = x
                expected_arrays.update({key + "_" + k: v for k, v in parts.items()})
                row = next(
                    r
                    for r in field_rows
                    if (r["role"], r["query"], r["space"]) == (role, q, space)
                )
                for name, value in stats.items():
                    compare("decomposition", value, row[name])
            for arm in spec["arms"]:
                lookup = {
                    (r["context"], r["wording"]): r
                    for r in actions
                    if (r["role"], r["query"], r["arm"]) == (role, q, arm)
                }
                for units in ("raw", "unit_displacement"):
                    x = np.array(
                        [
                            [
                                lookup[c, w]["odd_action"]
                                / (
                                    1
                                    if units == "raw"
                                    else lookup[c, w]["physical_delta_norm"]
                                )
                                for w in range(4)
                            ]
                            for c in contexts
                        ]
                    )
                    expected_arrays[role + "_" + q + "_" + arm + "_" + units] = x
                    _, stats = alternate_parts(x)
                    row = next(
                        r
                        for r in action_rows
                        if (r["role"], r["query"], r["arm"], r["units"])
                        == (role, q, arm, units)
                    )
                    for name, value in stats.items():
                        compare("decomposition", value, row[name])
        for q, z in itertools.combinations(spec["queries"], 2):
            for arm in spec["arms"]:
                for units in ("raw", "unit_displacement"):
                    x = expected_arrays[role + "_" + q + "_" + arm + "_" + units]
                    y = expected_arrays[role + "_" + z + "_" + arm + "_" + units]
                    a, _ = alternate_parts(x)
                    b, _ = alternate_parts(y)
                    values = {
                        name: float(np.mean(a[name] * b[name]))
                        for name in ("context", "wording", "interaction")
                    }
                    values["native"] = float(
                        np.mean((x - np.mean(x)) * (y - np.mean(y)))
                    )
                    values["identity_error"] = abs(
                        values["native"]
                        - sum(values[n] for n in ("context", "wording", "interaction"))
                    )
                    row = next(
                        r
                        for r in cov_rows
                        if (r["role"], r["left"], r["right"], r["arm"], r["units"])
                        == (role, q, z, arm, units)
                    )
                    for name, value in values.items():
                        compare("covariance", value, row[name])
    with np.load(parent / "02/crossed_arrays.npz", allow_pickle=False) as data:
        if set(data.files) != set(expected_arrays):
            raise ValueError("crossed array addresses differ")
        for name, value in expected_arrays.items():
            compare("array", value, data[name])
    counts = {
        "references": len(design),
        "actions": len(actions),
        "endpoints": endpoints,
        "metrics": len(table(parent / "02/metrics.csv")),
        "source_fields": len(fits),
        "field_decompositions": len(field_rows),
        "action_decompositions": len(action_rows),
        "paired_covariances": len(cov_rows),
    }
    if counts != {
        "references": 96,
        "actions": 192,
        "endpoints": 384,
        "metrics": 18,
        "source_fields": 15,
        "field_decompositions": 18,
        "action_decompositions": 36,
        "paired_covariances": 36,
    }:
        raise ValueError("finite numerical coverage incomplete")
    if errors["hook"] != 0 or max(errors.values()) > 1e-10:
        raise ValueError("alternate numerical reconstruction differs " + repr(errors))
    write_json(
        out / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM30_NOT_GLOBAL",
            "counts": counts,
            "maximum_errors": errors,
            "artifact_hashes_checked": len(ledger),
            "inherited_files_checked": len(m["files"]),
            "checked_at": datetime.now().astimezone().isoformat(),
            "scope": "same-agent alternate formulas, no independent human validation or new model calls",
        },
    )
    print(read_json(out / "summary.json"), flush=True)


def main():
    """Replay the finite packet without model inference or modifying primary outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.parent, args.out)


if __name__ == "__main__":
    main()
