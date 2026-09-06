#!/usr/bin/env python3
"""Fixed cached Qwen readouts; task changes are not history counterfactuals."""

from __future__ import annotations

import numpy as np

ARMS = ("raw", "cue_mask", "placebo_mask")
SITES = ("block_04", "block_10", "block_16", "final_norm")
TASKS = ("behavior", "disposition")
KINDS = (
    "body_token",
    "body_equal_view",
    "body_variance_parts",
    "decision_behavior",
    "decision_disposition",
    "decision_task_delta",
)


def readout_table(meta):
    """Validate all18 paired observations, including actual body-token weights."""
    if meta["causal_prefix_max"] != 0 or meta["replay_max"] != 0:
        raise ValueError("Cached Qwen prefix or fixed-shape replay failed")
    expected = {(a, v, t) for a in ARMS for v in range(3) for t in TASKS}
    table = {}
    for row in meta["readouts"]:
        key = row["arm"], row["view"], row["task"]
        n, prompt = row["body_tokens"], row["prompt_tokens"]
        if (
            key not in expected
            or key in table
            or isinstance(row["view"], bool)
            or not isinstance(row["view"], int)
            or isinstance(n, bool)
            or not isinstance(n, int)
            or n < 1
            or isinstance(prompt, bool)
            or not isinstance(prompt, int)
            or prompt <= n
            or not np.isfinite(row["margin"])
        ):
            raise ValueError("Invalid paired readout metadata")
        table[key] = row
    if set(table) != expected:
        raise ValueError("Missing fixed task/view observation")
    for a in ARMS:
        for v in range(3):
            if (
                table[a, v, TASKS[0]]["body_tokens"]
                != table[a, v, TASKS[1]]["body_tokens"]
            ):
                raise ValueError("Pre-question body token count changed across tasks")
    return table


def variance_parts(means, stds, weights):
    """Use centered total-variance algebra, without treating views as new authors."""
    m, s, w = (np.asarray(x, dtype=np.float64) for x in (means, stds, weights))
    if (
        m.ndim != 2
        or m.shape[0] != 3
        or not m.shape[1]
        or s.shape != m.shape
        or w.shape != (3,)
        or not all(np.isfinite(x).all() for x in (m, s, w))
        or np.any(s < 0)
        or np.any(w <= 0)
    ):
        raise ValueError("Invalid three-view moments or weights")
    w = w / w.sum()
    mean = w @ m
    within = w @ (s * s)
    between = w @ ((m - mean) ** 2)
    return mean, within, between


def _matrix(features, arm, site, suffix, dimension=None):
    """Reject changed axes, missing/invalid numerical values and mixed dimensions."""
    x = np.asarray(features[f"{arm}_{site}_{suffix}"], dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[0] != 3
        or not x.shape[1]
        or not np.isfinite(x).all()
        or dimension is not None
        and x.shape[1] != dimension
    ):
        raise ValueError("Cached readout is not a finite3-view aligned matrix")
    return x


def channel(features, meta, arm, site, kind):
    """Calculate one predeclared vector, never a fitted or label-selected direction."""
    if arm not in ARMS or site not in SITES or kind not in KINDS:
        raise ValueError("Unsupported frozen readout; history delta is not cached")
    table = readout_table(meta)
    m = _matrix(features, arm, site, "body_mean")
    s = _matrix(features, arm, site, "body_std", m.shape[1])
    if kind.startswith("body_"):
        w = (
            [table[arm, v, "behavior"]["body_tokens"] for v in range(3)]
            if kind == "body_token"
            else np.ones(3)
        )
        mean, within, between = variance_parts(m, s, w)
        if kind == "body_variance_parts":
            return np.concatenate([mean, np.sqrt(within), np.sqrt(between)])
        return np.concatenate([mean, np.sqrt(within + between)])
    behavior = _matrix(features, arm, site, "decision_behavior", m.shape[1])
    disposition = _matrix(features, arm, site, "decision_disposition", m.shape[1])
    values = {
        "decision_behavior": behavior,
        "decision_disposition": disposition,
        "decision_task_delta": disposition - behavior,
    }[kind]
    return np.concatenate([values.mean(0), values.std(0, ddof=0)])


def margin_features(meta, arm):
    """Return four raw-logit diagnostics; these are not calibrated trait scores."""
    if arm not in ARMS:
        raise ValueError("Unsupported text arm")
    table = readout_table(meta)
    values = np.asarray([[table[arm, v, t]["margin"] for v in range(3)] for t in TASKS])
    return np.column_stack([values.mean(1), values.std(1, ddof=0)]).ravel()


def author_channels(features, meta):
    """Preserve all75 named vectors; do not duplicate site-independent margins."""
    result = {
        f"{arm}__{site}__{kind}": channel(features, meta, arm, site, kind)
        for arm in ARMS
        for site in SITES
        for kind in KINDS
    }
    result.update({f"{arm}__logit_margins": margin_features(meta, arm) for arm in ARMS})
    return result
