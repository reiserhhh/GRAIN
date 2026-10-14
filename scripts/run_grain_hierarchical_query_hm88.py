#!/usr/bin/env python3
"""Run frozen cached HM88 measure interventions, with no neural dependency."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_cold_evidence as C
from grain_research.construction import math_hierarchical_query as M

SPEC = ROOT / "configs/grain_hierarchical_query_measure_hm88.json"
CODE = (
    "configs/grain_hierarchical_query_measure_hm88.json",
    "docs/GRAIN_HIERARCHICAL_QUERY_MEASURE_PLAN_HM88.md",
    "grain_research/construction/math_cold_evidence.py",
    "grain_research/construction/math_hierarchical_query.py",
    "scripts/run_grain_hierarchical_query_hm88.py",
    "tests/test_grain_math_hierarchical_query.py",
)


def load(path):
    """Read explicit public/numerical experiment metadata."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Authenticate exactly named immutable artifacts."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    """Refuse to overwrite a generated checkpoint or historical result."""
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def table(path, rows):
    """Save every fixed condition, including opposite-signed actions."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify(base, hashes):
    """Check source ancestry without loading weights or model code."""
    for name, expected in hashes.items():
        if sha(base / name) != expected:
            raise ValueError("Frozen input drift: " + name)


def unit(source, index):
    """Read a sealed early-prefix cache and verify its stop trace."""
    path = source / f"01/units/unit_{index:04d}.json"
    meta = load(path)
    if sha(path.with_suffix(".npz")) != meta["array_sha256"]:
        raise ValueError("Captured arrays changed")
    if meta["trace"] != {"completed_blocks": list(range(17)), "forbidden_calls": []}:
        raise ValueError("Unexpected neural trace")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def run():
    """Discriminate shared/target and event/within operations on all cached bodies."""
    spec = load(SPEC)
    source, out = ROOT / spec["source"], ROOT / spec["output"]
    if out.exists():
        raise FileExistsError("Preserve completed or failed HM88")
    if load(source / "verification.json")["cold_arithmetic_and_tables"] != "PASS":
        raise ValueError("HM87 numeric predecessor missing")
    ancestry = load(source / "00/manifest.json")
    verify(ROOT, ancestry["code"] | ancestry["source"])
    verify(source, ancestry["inputs"])
    verify(source, load(source / "01/raw_seal.json"))
    verify(source, load(source / "04_verified_completion/output_seal.json"))
    out.mkdir()
    write(
        out / "manifest.json",
        {
            "spec": spec,
            "code": {name: sha(ROOT / name) for name in CODE},
            "source_raw_seal": sha(source / "01/raw_seal.json"),
            "source_completion_seal": sha(
                source / "04_verified_completion/output_seal.json"
            ),
            "HM87_failed_prefix_gate_retained": load(source / "verification.json")[
                "frozen_prefix_gate"
            ],
            "native_queries_are_observed_not_independently_constructed": True,
            "retrospective_cache_analysis": True,
        },
    )
    begin = time.monotonic()
    bundle, rows = load(source / "00/bundle.json"), load(source / "00/rows.json")
    bodies = [r for r in rows if r["stage"] == "body" and r["order"] == 0]
    if len(bodies) != spec["expected_bodies"]:
        raise ValueError("Incomplete body battery")
    native = {
        (r["body_unit"], r["query_index"]): r for r in rows if r["stage"] == "native"
    }
    directions = np.asarray(bundle["directions"])
    fields = np.zeros((6, 16, 6, 6, 6))  # context/corner/query/arm/direction
    errors = {
        "native": 0.0,
        "normalization": 0.0,
        "event_constraint": 0.0,
        "within_constraint": 0.0,
        "kl_chain": 0.0,
        "independent_projection": 0.0,
    }
    records, checks = [], 0
    for body in bodies:
        raw = unit(source, body["unit"])
        ix = sorted(i for g in body["groups"] for i in g)
        ids = np.asarray(
            [next(j for j, g in enumerate(body["groups"]) if i in g) for i in ix]
        )
        k, v = raw["k"][ix].astype(float), raw["v"][ix].astype(float)
        ratio = spec["expected_heads"] // k.shape[1]
        values = np.repeat(v, ratio, axis=1)
        bases, fulls = [], []
        for qi in range(spec["expected_queries"]):
            qr = unit(source, native[body["unit"], qi]["unit"])["raw_query"]
            pos = ix[-1] + bundle["gaps"][qi]
            q0 = C.rotate(
                bundle["raw_queries"][qi], pos, bundle["frequencies"], bundle["scaling"]
            )
            q1 = C.rotate(qr, pos, bundle["frequencies"], bundle["scaling"])
            bases.append(np.einsum("lhd,hd->hl", np.repeat(k, ratio, axis=1), q0))
            fulls.append(np.einsum("lhd,hd->hl", np.repeat(k, ratio, axis=1), q1))
        bases, fulls = np.asarray(bases), np.asarray(fulls)
        shared, specific = M.shared_specific(fulls - bases)
        for qi in range(6):
            f = M.factorize(bases[qi], fulls[qi] - bases[qi], ids)
            logs = np.stack(
                [
                    f["cold"],
                    M.normalize(bases[qi] + shared),
                    M.normalize(bases[qi] + specific[qi]),
                    f["native"],
                    f["event_only"],
                    f["within_only"],
                ]
            )
            weights = np.exp(logs)
            errors["normalization"] = max(
                errors["normalization"], float(np.max(np.abs(weights.sum(-1) - 1)))
            )
            errors["native"] = max(
                errors["native"],
                float(np.max(np.abs(logs[3] - M.normalize(fulls[qi])))),
            )
            errors["kl_chain"] = max(
                errors["kl_chain"],
                float(np.max(np.abs(f["total_kl"] - f["between_kl"] - f["within_kl"]))),
            )
            for g in range(3):
                selected = ids == g
                errors["event_constraint"] = max(
                    errors["event_constraint"],
                    float(
                        np.max(
                            np.abs(weights[5][:, selected].sum(-1) - f["mass0"][:, g])
                        )
                    ),
                )
                a, b = weights[4][:, selected], weights[0][:, selected]
                errors["within_constraint"] = max(
                    errors["within_constraint"],
                    float(
                        np.max(np.abs(a / a.sum(-1)[:, None] - b / b.sum(-1)[:, None]))
                    ),
                )
            vectors = np.einsum("ahl,lhd->ahd", weights, values)
            projection = np.einsum("ahd,thd->at", vectors, directions)
            alternative = np.zeros_like(projection)
            for g in range(3):
                selected = ids == g
                for h in range(12):
                    value_scores = values[selected, h] @ directions[:, h].T
                    alternative += weights[:, h, selected] @ value_scores
            errors["independent_projection"] = max(
                errors["independent_projection"],
                float(np.max(np.abs(projection - alternative))),
            )
            fields[body["context"], body["corner"], qi] = projection
            for h in range(12):
                records.append(
                    {
                        "context": body["context"],
                        "corner": body["corner"],
                        "query": bundle["targets"][qi],
                        "head": h,
                        "between_kl": float(f["between_kl"][h]),
                        "within_kl": float(f["within_kl"][h]),
                        "total_kl": float(f["total_kl"][h]),
                        "naive_mass_drift": float(f["naive_mass_drift"][h]),
                        **{
                            f"cold_event_{g}": float(f["mass0"][h, g]) for g in range(3)
                        },
                        **{
                            f"native_event_{g}": float(f["mass1"][h, g])
                            for g in range(3)
                        },
                    }
                )
            checks += weights.size + projection.size
    points = np.asarray(list(itertools.product((-1, 1), repeat=4)))
    results, components = [], []
    for ti, target in enumerate(bundle["targets"]):
        ci = (ti - 1) % 6
        scores = fields[ci, :, :, :, ti]
        own = scores[(points[:, 1] == 1) & (points[:, 2] == 1)].mean(0) - scores[
            (points[:, 1] == -1) & (points[:, 2] == 1)
        ].mean(0)
        foreign = scores[(points[:, 1] == 1) & (points[:, 2] == -1)].mean(0) - scores[
            (points[:, 1] == -1) & (points[:, 2] == -1)
        ].mean(0)
        selected = own[ti] - np.delete(own, ti, axis=0).mean(0)
        yg = points[:, 1] * points[:, 2]
        for ai, arm in enumerate(spec["arms"]):
            contrast = 2 * spec["epsilon"] * np.mean(yg * scores[:, ti, ai])
            keep = points[:, 3] == -points[:, 2]
            restricted = 2 * spec["epsilon"] * np.mean((yg * scores[:, ti, ai])[keep])
            results.append(
                {
                    "target": target,
                    "arm": arm,
                    "own": float(own[ti, ai]),
                    "foreign": float(foreign[ti, ai]),
                    "selectivity": float(selected[ai]),
                    "author_yg_contrast": float(contrast),
                    "equal_name_count_yg_contrast": float(restricted),
                }
            )
        for metric, values in (
            ("selectivity", selected),
            ("own", own[ti]),
            ("foreign", foreign[ti]),
        ):
            for name, a, b in (("shared_specific", 1, 2), ("event_within", 4, 5)):
                first, second, interaction = M.crossed_components(
                    values[0], values[a], values[b], values[3]
                )
                components.append(
                    {
                        "target": target,
                        "metric": metric,
                        "split": name,
                        "first": float(first),
                        "second": float(second),
                        "interaction": float(interaction),
                        "total_change": float(values[3] - values[0]),
                    }
                )
    np.savez_compressed(out / "projections.npz", projections=fields)
    table(out / "probability_accounts.csv", records)
    table(out / "target_author_actions.csv", results)
    table(out / "finite_components.csv", components)
    status = (
        "VERIFIED_FINITE_HM88"
        if max(errors.values()) <= spec["tolerance"]
        else "FAILED_NUMERICAL_GATE_HM88"
    )
    write(
        out / "summary.json",
        {
            "status": status,
            "cached_bodies": len(bodies),
            "cached_queries": len(native),
            "new_neural_calls": 0,
            "fits": 0,
            "errors": errors,
            "checks": checks,
            "seconds": time.monotonic() - begin,
            "native_queries_still_required": True,
            "retrospective_no_independent_replication": True,
            "prefix_gate_from_HM87": "FAIL_preserved",
            "original_scope_complete": False,
            "targets": results,
            "components": components,
            "mean_between_kl": float(np.mean([r["between_kl"] for r in records])),
            "mean_within_kl": float(np.mean([r["within_kl"] for r in records])),
            "naive_mass_drift_max": max(r["naive_mass_drift"] for r in records),
        },
    )
    write(
        out / "output_seal.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in sorted(out.iterdir())
            if p.is_file() and not p.name.startswith("._")
        },
    )
    if status.startswith("FAILED"):
        raise ValueError("Retained failed fixed numerical gate")
    print(
        json.dumps(
            {"status": status, "errors": errors, "seconds": time.monotonic() - begin}
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
