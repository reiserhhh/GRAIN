#!/usr/bin/env python3
"""Alternate scalar/account checks for HM94; do not repeat model fitting."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_grain_opportunity_current_hm94 as R


def parts(joint):
    """Compute complete-graph projection by closed potential, not least squares."""
    a = (joint - joint.T) / 2
    potential = a.sum(1) / len(a)
    g = potential[:, None] - potential[None]
    return np.stack(((joint + joint.T) / 2, g, a - g))


def verify(out):
    """Verify frozen seals, every author account, toy laws and saved summaries."""
    cfg = R.sealed(out)
    for stage in ("01", "02", "03"):
        R.previous(out, stage)
    if (out / "verification.json").exists():
        raise ValueError("Preserve previous verification")
    split = R.read(out / "01/partition.json")
    meta = R.read(out / "02/private_metadata.json")
    if set(split["source"]) & set(split["application"]):
        raise AssertionError("Author overlap")
    if [r["opaque_author"] for r in meta] != split["application"]:
        raise AssertionError("Cold author identity changed")
    with np.load(out / "01/source_model.npz", allow_pickle=False) as z:
        model = {k: z[k] for k in z.files}
    with np.load(out / "02/private_author_accounts.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    total = model["source_counts"].sum(0)
    nz, nr, ns, _ = total.shape
    alpha = cfg["pseudocount"]
    source_support = (model["source_counts"].sum((-2, -1)) > 0).sum(0)
    if not np.array_equal(source_support, model["source_support"]):
        raise AssertionError("Source support mismatch")
    error, scalars, comments = 0.0, 0, 0
    # Independent cell loops check all source reference arms and fallback rules.
    embedding_support = (model["source_counts"].sum((2, 3, 4)) > 0).sum(0)
    for arm in cfg["arms"]:
        for z, r in itertools.product(range(nz), range(nr)):
            if arm == "pooled":
                counts = total.sum((0, 1))
            elif (
                arm == "position"
                or (
                    arm == "joint"
                    and source_support[z, r] < cfg["minimum_reference_authors"]
                )
                or (
                    arm == "embedding"
                    and embedding_support[z] < cfg["minimum_reference_authors"]
                )
            ):
                counts = total[:, r].sum(0)
            elif arm == "embedding":
                counts = total[z].sum(0)
            else:
                counts = total[z, r]
            expected = (counts + alpha) / (counts.sum() + alpha * ns**2)
            error = max(
                error, float(np.max(abs(expected - model["reference_" + arm][z, r])))
            )
            scalars += expected.size
    all_cycles = []
    for row in meta:
        key = row["key"]
        x, condition, view = (
            arrays[key + s] for s in ("_states", "_conditions", "_views")
        )
        comments += len(x)
        rebuilt = np.zeros((3, nz, nr, ns, ns))
        for comment in range(len(x)):
            for r in range(3):
                rebuilt[
                    view[comment],
                    condition[comment],
                    r,
                    x[comment, r],
                    x[comment, r + 1],
                ] += 1
        if not np.array_equal(rebuilt, arrays[key + "_counts"]):
            raise AssertionError("Comment/state/opportunity binding failed")
        cycles = []
        for v in range(3):
            c = rebuilt[v]
            q = c.sum((-2, -1)) / c.sum()
            error = max(error, float(np.max(abs(q - arrays[key + "_q"][v]))))
            vcycle = []
            for arm_index, arm in enumerate(cfg["arms"]):
                raw, expected = c.sum((0, 1)) / c.sum(), np.zeros((ns, ns))
                for z, r in itertools.product(range(nz), range(nr)):
                    expected += q[z, r] * model["reference_" + arm][z, r]
                laws = np.stack((raw, expected, raw - expected))
                projected = np.array([parts(j) for j in laws])
                error = max(
                    error,
                    float(np.max(abs(laws - arrays[key + "_laws"][v, arm_index]))),
                    float(
                        np.max(abs(projected - arrays[key + "_parts"][v, arm_index]))
                    ),
                )
                for term in projected:
                    error = max(error, float(np.max(abs(term[2].sum(1)))))
                vcycle.append(projected[:, 2])
                scalars += laws.size + projected.size
            cycles.append(vcycle)
        all_cycles.append(cycles)
    all_cycles = np.array(all_cycles)
    table = pd.read_csv(out / "03/cycle_variance_accounts.csv")
    for index, arm in enumerate(cfg["arms"]):
        c = all_cycles[:, :, index] - all_cycles[:, :, index].mean(0)
        e = np.mean(np.sum(c**2, (-2, -1)), (0, 1))
        cross = 2 * np.mean(np.sum(c[:, :, 1] * c[:, :, 2], (-2, -1)))
        actual = table.set_index("arm").loc[arm]
        expected = np.r_[e, cross]
        error = max(
            error,
            float(
                np.max(
                    abs(
                        expected
                        - actual[
                            [
                                "raw_variance",
                                "opportunity_variance",
                                "response_variance",
                                "cross_term",
                            ]
                        ].to_numpy(float)
                    )
                )
            ),
        )
    with np.load(out / "03/toy_accounts.npz", allow_pickle=False) as toy:
        forward = toy["forward_opportunity_only"]
        reverse = toy["reverse_opportunity_only"]
        unordered = toy["unordered_opportunity_only"]
        if not (
            np.linalg.norm(parts(forward[0])[2]) > 0.1 and np.max(abs(forward[2])) == 0
        ):
            raise AssertionError("Opportunity-only counterexample absent")
        error = max(
            error,
            float(np.max(abs(forward[0].T - reverse[0]))),
            float(np.max(abs(parts(unordered[0])[2]))),
        )
        for name in toy.files:
            a = toy[name]
            for j in a[:2]:
                if j.min() < 0 or not np.isclose(j.sum(), 1):
                    raise AssertionError("Toy probability law invalid")
            error = max(error, float(np.max(abs(a[0] - a[1] - a[2]))))
    source_summary, apply_summary, summary = (
        R.read(out / p / "summary.json") for p in ("01", "02", "03")
    )
    if (
        comments != apply_summary["application_comments"]
        or comments + source_summary["source_comments"] != 9165
    ):
        raise AssertionError("Comment totals changed")
    if error > 1e-10:
        raise AssertionError(f"Alternate arithmetic failed: {error}")
    result = {
        "status": "VERIFIED_FINITE_HM94_NOT_GLOBAL",
        "maximum_alternate_error": error,
        "scalar_checks": scalars,
        "source_authors": len(split["source"]),
        "application_authors": len(meta),
        "source_comments": source_summary["source_comments"],
        "application_comments": comments,
        "new_calls": 0,
        "new_predictors": 0,
        "same_agent_alternate_arithmetic": True,
        "human_personality_accuracy_measured": False,
        "original_scope_complete": False,
        "evaluation_status": summary["status"],
    }
    R.write(out / "verification.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=R.OUT)
    verify(parser.parse_args().out)
