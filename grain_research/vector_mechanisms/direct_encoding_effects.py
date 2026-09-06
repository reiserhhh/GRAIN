#!/usr/bin/env python3
"""Semantic-aligned candidate-logit interactions; no fitted calibration or labels."""

import numpy as np

from .direct_semantics import panel
from .direct_traits import BIG5


def aligned_logits(row):
    """Remove the softmax gauge after sorting by the declared trait-value coding."""
    z = np.asarray(row["candidate_logits"], dtype=float)
    v = np.asarray(row["alphabet"]["values"], dtype=float)
    size = 10 if row["task"] in BIG5 else 2
    if (
        z.shape != (size,)
        or v.shape != z.shape
        or not np.isfinite(z).all()
        or not np.isfinite(v).all()
        or not np.allclose(np.sort(v), np.linspace(0, 1, size), atol=1e-14, rtol=0)
    ):
        raise ValueError("Require complete finite distinct semantic answer coding")
    ordered = z[np.argsort(v)]
    return ordered - np.mean(ordered)


def analyze(rows):
    """Return all task, case and pair contrasts on the frozen90-case grid."""
    if len(rows) != 90 or any(
        any(r.get(k) != v for k, v in c.items())
        for r, c in zip(rows, panel(), strict=True)
    ):
        raise ValueError("Complete unchanged90-case semantic grid required")
    states = {
        (r["task"], r["variant"], r["pair"], r["polarity"]): aligned_logits(r)
        for r in rows
    }
    task_rows, case_rows, pair_rows = [], [], []
    for task in dict.fromkeys(r["task"] for r in rows):
        bias = states[task, "counter", -1, -1] - states[task, "primary", -1, -1]
        residuals, shifts = [], {}
        for pair in range(2):
            for pol in (0, 1):
                shift = (
                    states[task, "counter", pair, pol]
                    - states[task, "primary", pair, pol]
                )
                r = shift - bias
                shifts[pair, pol] = shift
                residuals.append(r)
                item = {
                    "task": task,
                    "pair": pair,
                    "polarity": pol,
                    "semantic_dimension": len(r),
                    "no_evidence_relative_rms": float(np.sqrt(np.mean(r**2))),
                    "no_evidence_relative_max_abs": float(np.max(np.abs(r))),
                    "semantic_shift": shift.tolist(),
                    "no_evidence_relative_vector": r.tolist(),
                }
                if task not in BIG5:
                    item.update(
                        primary_log_odds=float(
                            np.diff(states[task, "primary", pair, pol])[0]
                        ),
                        counter_log_odds=float(
                            np.diff(states[task, "counter", pair, pol])[0]
                        ),
                        log_odds_bias=float(bias[1] - bias[0]),
                        log_odds_residual=float(r[1] - r[0]),
                    )
                case_rows.append(item)
        mixed = []
        for pair in range(2):
            j = shifts[pair, 1] - shifts[pair, 0]
            mixed.append(j)
            record = {
                "task": task,
                "pair": pair,
                "semantic_dimension": len(j),
                "mixed_rms": float(np.sqrt(np.mean(j**2))),
                "mixed_max_abs": float(np.max(np.abs(j))),
                "mixed_vector": j.tolist(),
            }
            if task not in BIG5:
                record["mixed_log_odds"] = float(j[1] - j[0])
            pair_rows.append(record)
        task_rows.append(
            {
                "task": task,
                "semantic_dimension": len(bias),
                "no_evidence_bias_rms": float(np.sqrt(np.mean(bias**2))),
                "anchored_residual_rms": float(
                    np.sqrt(np.mean(np.asarray(residuals) ** 2))
                ),
                "mixed_pair_rms": float(np.sqrt(np.mean(np.asarray(mixed) ** 2))),
                "mixed_pair_max_abs": float(np.max(np.abs(mixed))),
                "no_evidence_bias": bias.tolist(),
            }
        )
    return task_rows, case_rows, pair_rows
