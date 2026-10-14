#!/usr/bin/env python3
"""Independent normal-equation replay and paired native action summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_operator import metrics
from scripts.run_grain_conditional_operator import inputs, table
from scripts.run_grain_persona_generation_trace import read_jsonl, sha, write


def paired_interval(rows, pred_a, pred_b, target):
    """Bootstrap error differences by family/task with all seeds/axes together."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["family"], r["situation"]), []).append(i)
    groups = [np.array(v) for v in groups.values()]
    a, b, y = np.array(pred_a), np.array(pred_b), np.array(target)
    rng = np.random.default_rng(20260915)
    values = []
    for _ in range(2000):
        ix = np.concatenate(
            [groups[j] for j in rng.integers(len(groups), size=len(groups))]
        )
        values.append(
            np.sqrt(np.mean((a[ix] - y[ix]) ** 2))
            - np.sqrt(np.mean((b[ix] - y[ix]) ** 2))
        )
    return np.quantile(values, [0.025, 0.975]).tolist()


def run(out):
    """Recompute the source estimator with an augmented least-squares algorithm."""
    for name, expected in json.loads((out / "manifest.json").read_text()).items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Changed source: {name}")
    if json.loads((out / "native_status.json").read_text())["status"] != "COMPLETE":
        raise ValueError("Native run incomplete")
    _, pairs = inputs()
    with np.load(out / "action_arrays.npz") as z:
        dx, dh, ds, source = z["dx"], z["dh"], z["ds"], z["source"]
        predicted, direct = z["predicted"], z["direct"]
    expected_source = np.array([p["split"] == "source" for p in pairs])
    np.testing.assert_array_equal(source, expected_source)
    assert source.sum() == 36 and (~source).sum() == 72
    expected_dx = np.array([np.subtract(p["high"]["g"], p["low"]["g"]) for p in pairs])
    np.testing.assert_allclose(dx, expected_dx, atol=0, rtol=0)
    max_score_error = 0.0
    for i, pair in enumerate(pairs):
        blocks = []
        for qi, query in enumerate(("extraversion", "neuroticism")):
            heads, suffixes = [], []
            for pole in ("high", "low"):
                row = pair[pole]
                digest = hashlib.sha256(
                    f"{row['text_id']}|{query}".encode()
                ).hexdigest()
                with np.load(out / "states" / f"{digest}.npz") as state:
                    heads.append(state["head"].astype(float))
                    suffixes.append(state["suffix"])
                    max_score_error = max(
                        max_score_error, abs(float(state["score"]) - row["scores"][qi])
                    )
            np.testing.assert_array_equal(suffixes[0], suffixes[1])
            blocks.append((heads[0] - heads[1]).ravel())
        np.testing.assert_allclose(np.concatenate(blocks), dh[i], atol=0, rtol=0)
    assert max_score_error < 1e-6
    scale = np.std(dx[source], axis=0)
    scale[scale < 1e-12] = 1
    x = dx[source] / scale
    augmented = np.vstack([x, np.eye(4)])
    w = np.linalg.lstsq(
        augmented, np.vstack([dh[source], np.zeros((4, dh.shape[1]))]), rcond=None
    )[0]
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    rebuilt = dx / scale @ ((u[:, :2] * s[:2]) @ vt[:2])
    pred_error = float(np.max(np.abs(rebuilt - predicted)))
    np.testing.assert_allclose(rebuilt, predicted, atol=1e-9, rtol=1e-9)
    b = np.linalg.lstsq(
        augmented, np.vstack([ds[source], np.zeros((4, 2))]), rcond=None
    )[0]
    np.testing.assert_allclose(dx / scale @ b, direct, atol=1e-10, rtol=1e-10)
    records = read_jsonl(out / "native_effects.jsonl")
    assert len(records) == len({r["key"] for r in records}) == 576
    lookup = {(r["pair"], r["query"], r["kind"]): r for r in records}
    self_error = max(abs(r["effect"]) for r in records if r["kind"] == "self")
    assert self_error < 1e-6
    for r in records:
        assert r["split"] != "source"
        assert abs(r["score"] - r["low"] - r["effect"]) < 1e-12
        assert abs(r["high"] - r["low"] - r["contrast"]) < 1e-12
        assert 0 <= r["mass"] <= 1 and 0 <= r["score"] <= 1
    summaries = []
    cis = {}
    for query in ("extraversion", "neuroticism"):
        for axis in ("all", "extraversion", "neuroticism"):
            for split in ("application", "new_task", "new_wording", "new_both"):
                rr = [
                    r
                    for r in records
                    if r["query"] == query
                    and r["kind"] == "compiled"
                    and (axis == "all" or r["axis"] == axis)
                    and (split == "application" or r["split"] == split)
                ]
                pred = {
                    "compiled": [r["effect"] for r in rr],
                    "direct": [r["direct"] for r in rr],
                    "zero": [0.0 for r in rr],
                }
                for kind in ("donor", "random"):
                    pred[kind] = [
                        lookup[(r["pair"], query, kind)]["effect"] for r in rr
                    ]
                full = [r["contrast"] for r in rr]
                for target_name, target in (
                    ("full_contrast", full),
                    ("donor_action", pred["donor"]),
                ):
                    for name, values in pred.items():
                        summaries.append(
                            {
                                "query": query,
                                "axis": axis,
                                "split": split,
                                "target": target_name,
                                "method": name,
                                **metrics(values, target),
                            }
                        )
                if split == "application":
                    cis[f"{query}|{axis}"] = {
                        "compiled_minus_direct_rmse_full_ci": paired_interval(
                            rr, pred["compiled"], pred["direct"], full
                        ),
                        "compiled_minus_random_rmse_donor_ci": paired_interval(
                            rr, pred["compiled"], pred["random"], pred["donor"]
                        ),
                    }
    table(out / "metrics.csv", summaries)
    write(out / "paired_intervals.json", cis)
    audit = {
        "status": "PASS",
        "pairs": len(pairs),
        "native_records": len(records),
        "operator_replay_error": pred_error,
        "self_replay_error": self_error,
        "cached_reader_replay_error": max_score_error,
        "audit_type": "same-agent independent algebra; not third-party replication",
    }
    remainder = read_jsonl(out / "remainder_effects.jsonl")
    if remainder:
        assert len(remainder) == len({r["key"] for r in remainder}) == 144
        errors = []
        for r in remainder:
            a = lookup[(r["pair"], r["query"], "compiled")]
            d = lookup[(r["pair"], r["query"], "donor")]
            f0, fa, fr, fd = a["low"], a["score"], r["residual_score"], d["score"]
            errors.extend(
                [
                    abs(r["interaction"] - (fd - fa - fr + f0)),
                    abs(r["symmetric_compiled"] - ((fa - f0) + (fd - fr)) / 2),
                    abs(r["symmetric_remainder"] - ((fr - f0) + (fd - fa)) / 2),
                    abs(r["residual"] - (fr - f0)),
                ]
            )
        audit["remainder_records"] = len(remainder)
        audit["four_corner_replay_error"] = max(errors)
        assert max(errors) < 1e-12
    write(out / "audit.json", audit)
    print(json.dumps(audit, indent=2))
    for r in summaries:
        if (
            r["split"] == "application"
            and r["axis"] == "all"
            and r["target"] == "full_contrast"
        ):
            print(json.dumps(r))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    run(ap.parse_args().output)
