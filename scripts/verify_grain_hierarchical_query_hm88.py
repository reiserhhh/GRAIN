#!/usr/bin/env python3
"""Independently reconstruct cached hierarchical weights and finite action tables."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_hierarchical_query_hm88 import (
    SPEC,
    load,
    sha,
    unit,
    verify,
    write,
)


def probability(logits):
    """Direct max-shifted exponential normalization, independent of factorizer."""
    value = np.exp(logits - logits.max(-1, keepdims=True))
    return value / value.sum(-1, keepdims=True)


def query(raw, position, bundle):
    """Explicit two-coordinate rotary formula with the frozen float32 phases."""
    raw = np.asarray(raw, float)
    angle = np.float32(position) * np.asarray(bundle["frequencies"], np.float32)
    lo, hi = np.split(raw, 2, axis=1)
    cosine, sine = np.cos(angle), np.sin(angle)
    return (
        np.concatenate((lo * cosine - hi * sine, hi * cosine + lo * sine), axis=1)
        * bundle["scaling"]
    )


def main():
    """Replay all six interventions from sealed vectors, then all scalar tables."""
    spec = load(SPEC)
    out, source = ROOT / spec["output"], ROOT / spec["source"]
    manifest = load(out / "manifest.json")
    verify(ROOT, manifest["code"])
    verify(out, load(out / "output_seal.json"))
    bundle, rows = load(source / "00/bundle.json"), load(source / "00/rows.json")
    verify(source, load(source / "01/raw_seal.json"))
    native = {
        (r["body_unit"], r["query_index"]): r for r in rows if r["stage"] == "native"
    }
    directions = np.asarray(bundle["directions"])
    with np.load(out / "projections.npz", allow_pickle=False) as archive:
        frozen = archive["projections"].copy()
    rebuilt = np.zeros_like(frozen)
    checks, max_error = 0, 0.0
    for row in (r for r in rows if r["stage"] == "body" and r["order"] == 0):
        ix = sorted(itertools.chain.from_iterable(row["groups"]))
        ids = np.asarray(
            [next(g for g, group in enumerate(row["groups"]) if i in group) for i in ix]
        )
        raw = unit(source, row["unit"])
        k, v = raw["k"][ix].astype(float), raw["v"][ix].astype(float)
        base, full = [], []
        for qi in range(6):
            original = unit(source, native[row["unit"], qi]["unit"])["raw_query"]
            position = ix[-1] + bundle["gaps"][qi]
            q0, q1 = (
                query(bundle["raw_queries"][qi], position, bundle),
                query(original, position, bundle),
            )
            base.append(np.stack([k[:, h // 6] @ q0[h] for h in range(12)]))
            full.append(np.stack([k[:, h // 6] @ q1[h] for h in range(12)]))
        base, full = np.asarray(base), np.asarray(full)
        change = full - base
        change -= change.sum(-1, keepdims=True) / change.shape[-1]
        shared = change.sum(0) / 6
        specific = change - shared
        for qi in range(6):
            cold, actual = probability(base[qi]), probability(full[qi])
            between, within = np.zeros_like(cold), np.zeros_like(cold)
            for g in range(3):
                ix_g = ids == g
                p0, p1 = cold[:, ix_g], actual[:, ix_g]
                m0, m1 = p0.sum(1)[:, None], p1.sum(1)[:, None]
                between[:, ix_g] = p0 * m1 / m0
                within[:, ix_g] = p1 * m0 / m1
            all_weights = (
                cold,
                probability(base[qi] + shared),
                probability(base[qi] + specific[qi]),
                actual,
                between,
                within,
            )
            scores = np.zeros((6, 6))
            for ai, weights in enumerate(all_weights):
                for h in range(12):
                    scores[ai] += weights[h] @ (v[:, h // 6] @ directions[:, h].T)
            rebuilt[row["context"], row["corner"], qi] = scores
            max_error = max(
                max_error,
                float(
                    np.max(np.abs(scores - frozen[row["context"], row["corner"], qi]))
                ),
            )
            checks += scores.size
    with np.load(
        source / "04_verified_completion/projections.npz", allow_pickle=False
    ) as old:
        predecessor_error = max(
            float(np.max(np.abs(frozen[:, :, :, 0] - old["cold"][:, 0, :, :, 0]))),
            float(np.max(np.abs(frozen[:, :, :, 3] - old["native"][:, 0, :, :, 0]))),
        )
    points = np.asarray(list(itertools.product((-1, 1), repeat=4)))
    with (out / "target_author_actions.csv").open() as stream:
        target_rows = list(csv.DictReader(stream))
    table_error = 0.0
    selections = {}
    for row in target_rows:
        ti, ai = bundle["targets"].index(row["target"]), spec["arms"].index(row["arm"])
        score = rebuilt[(ti - 1) % 6, :, :, ai, ti]
        high, low, own_mask = points[:, 1] == 1, points[:, 1] == -1, points[:, 2] == 1
        own = score[high & own_mask].mean(0) - score[low & own_mask].mean(0)
        foreign = score[high & ~own_mask].mean(0) - score[low & ~own_mask].mean(0)
        selectivity = own[ti] - (own.sum() - own[ti]) / 5
        yg, keep = points[:, 1] * points[:, 2], points[:, 3] == -points[:, 2]
        contrast = 2 * spec["epsilon"] * (yg * score[:, ti]).mean()
        restricted = 2 * spec["epsilon"] * (yg * score[:, ti])[keep].mean()
        computed = (own[ti], foreign[ti], selectivity, contrast, restricted)
        names = (
            "own",
            "foreign",
            "selectivity",
            "author_yg_contrast",
            "equal_name_count_yg_contrast",
        )
        table_error = max(
            table_error,
            *(
                abs(float(row[n]) - value)
                for n, value in zip(names, computed, strict=True)
            ),
        )
        selections[row["target"], row["arm"]] = dict(zip(names, computed, strict=True))
        checks += len(computed)
    with (out / "finite_components.csv").open() as stream:
        components = list(csv.DictReader(stream))
    component_error = 0.0
    for row in components:
        target, metric = row["target"], row["metric"]
        a, b = (
            ("shared_only", "specific_only")
            if row["split"] == "shared_specific"
            else ("event_only", "within_only")
        )
        cold, first, second, actual = (
            selections[target, arm][metric] for arm in ("cold", a, b, "native")
        )
        expected = (
            (first - cold + actual - second) / 2,
            (second - cold + actual - first) / 2,
            actual - first - second + cold,
            actual - cold,
        )
        component_error = max(
            component_error,
            *(
                abs(float(row[n]) - value)
                for n, value in zip(
                    ("first", "second", "interaction", "total_change"),
                    expected,
                    strict=True,
                )
            ),
        )
        checks += len(expected)
    if (
        max(max_error, predecessor_error, table_error, component_error)
        > spec["tolerance"]
    ):
        raise ValueError("Frozen finite verification failed; preserve all results")
    result = {
        "status": "VERIFIED_FINITE_HM88_NOT_GLOBAL",
        "projection_error": max_error,
        "predecessor_replay_error": predecessor_error,
        "table_error": table_error,
        "component_error": component_error,
        "checks": checks,
        "same_agent_not_independent_reviewer": True,
        "new_neural_calls": 0,
        "HM87_prefix_gate": "FAIL_preserved",
        "original_scope_complete": False,
        "verifier_sha256": sha(Path(__file__)),
    }
    write(out / "verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
