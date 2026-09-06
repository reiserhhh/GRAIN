#!/usr/bin/env python3
"""Frozen role-separated pooling and node-level MEPS correspondence statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["participant_code", "view_kind", "condition"]
PRIMARY = [
    ("meps_q1", "meps_q2"),
    ("meps_q1", "meps_q3"),
    ("meps_q2", "meps_q3"),
    ("meps_q1", "free_chat"),
    ("meps_q2", "free_chat"),
    ("meps_q3", "free_chat"),
    ("direct_all", "free_chat"),
]


def normalize(x):
    """Normalize finite nonzero vectors in float64; never impute missing vectors."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.all(np.isfinite(x)) or np.any(norm <= 1e-12):
        raise ValueError("Nonfinite or zero representation")
    return x / norm


def aggregate_windows(frame, vectors):
    """Pool only declared speakers, retaining missing AI views and turn boundaries."""
    if len(frame) != len(vectors) or (frame.content_tokens <= 0).any():
        raise ValueError("Mismatched windows or nonpositive content weights")
    vectors = normalize(vectors)
    frame = frame.reset_index(drop=True)
    rows, pooled = [], []
    for key, group in frame.groupby(KEYS, sort=True):
        for role, roles in (
            ("user", ["user"]),
            ("assistant", ["assistant"]),
            ("mixed", ["user", "assistant"]),
        ):
            arm = group[group.role.isin(roles)]
            if arm.empty:
                continue
            weighted = normalize(
                (vectors[arm.index] * arm.content_tokens.to_numpy()[:, None]).sum(0)
            )
            turn_vectors = []
            for _, turn in arm.groupby(["role", "turn_sequence"], sort=True):
                turn_vectors.append(
                    normalize(
                        (
                            vectors[turn.index]
                            * turn.content_tokens.to_numpy()[:, None]
                        ).sum(0)
                    )
                )
            for pooling, vec in (
                ("token", weighted),
                ("equal_turn", normalize(np.sum(turn_vectors, axis=0))),
            ):
                rows.append(
                    dict(zip(KEYS, key))
                    | {
                        "role": role,
                        "pooling": pooling,
                        "chars": int((arm.end - arm.start).sum()),
                        "content_tokens": int(arm.content_tokens.sum()),
                        "turns": len(turn_vectors),
                        "windows": len(arm),
                    }
                )
                pooled.append(vec)
    base = pd.DataFrame(rows)
    for (person, pooling), group in base[
        (base.view_kind == "direct_answer") & (base.role == "user")
    ].groupby(["participant_code", "pooling"], sort=True):
        if set(group.condition) != {"meps_q1", "meps_q2", "meps_q3"}:
            raise ValueError("Missing or duplicate fixed answers")
        if len(group) != 3:
            raise ValueError("Duplicate fixed answer keys")
        rows.append(
            {
                "participant_code": person,
                "view_kind": "direct_all",
                "condition": "direct_all",
                "role": "user",
                "pooling": pooling,
                **{
                    k: int(group[k].sum())
                    for k in ("chars", "content_tokens", "turns", "windows")
                },
            }
        )
        pooled.append(normalize(np.sum([pooled[i] for i in group.index], axis=0)))
    return pd.DataFrame(rows), np.stack(pooled)


def weighted_node_stat(matrix, counts):
    """Bidirectional percentile and gap with same-original-author edges excluded."""
    s = np.asarray(matrix, dtype=float)
    w = np.asarray(counts, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or len(s) != len(w):
        raise ValueError("Require a square paired-author similarity matrix")
    if not np.all(np.isfinite(s)) or np.any(w < 0) or np.count_nonzero(w) < 2:
        raise ValueError("Need at least two finite, distinct participant nodes")
    total = w.sum()
    valid = w > 0
    denom = total - w[valid]
    scores = []
    for a in (s, s.T):
        own = np.diag(a)
        other = (a @ w - own * w)[valid] / denom
        comparison = (own[:, None] > a).astype(float)
        comparison += 0.5 * (own[:, None] == a)
        np.fill_diagonal(comparison, 0)
        percentile = (comparison @ w)[valid] / denom
        scores.append(
            [
                np.average(own[valid] - other, weights=w[valid]),
                np.average(percentile, weights=w[valid]),
            ]
        )
    return np.mean(scores, axis=0)


def length_matched(matrix, left_chars, right_chars, neighbors=5):
    """Compare against other authors nearest the true target's length, not vectors."""
    n = len(matrix)
    if min(left_chars) <= 0 or min(right_chars) <= 0 or n < 2:
        raise ValueError("Invalid matching lengths or population")
    effects = []
    for s, chars in ((matrix, right_chars), (matrix.T, left_chars)):
        logchars = np.log(np.asarray(chars, dtype=float))
        for i in range(n):
            order = np.argsort(np.abs(logchars - logchars[i]), kind="stable")
            selected = order[order != i][: min(neighbors, n - 1)]
            own, other = s[i, i], s[i, selected]
            effects.append(
                (
                    own - other.mean(),
                    np.mean((own > other) + 0.5 * (own == other)),
                    np.mean(np.abs(logchars[selected] - logchars[i])),
                )
            )
    return np.mean(effects, axis=0)


def correspondence(matrix, left_chars, right_chars, *, draws=1000, permutations=1999):
    """Node bootstrap and complete-link permutation without pair-edge N inflation."""
    s = np.asarray(matrix, dtype=float)
    n = len(s)
    if n < 3:
        raise ValueError("Need three paired authors")
    point = weighted_node_stat(s, np.ones(n))
    rng = np.random.default_rng(731)
    estimates = []
    while len(estimates) < draws:
        counts = np.bincount(rng.integers(n, size=n), minlength=n)
        if np.count_nonzero(counts) >= 2:
            estimates.append(weighted_node_stat(s, counts))
    ci = np.quantile(estimates, [0.025, 0.975], axis=0)
    rng = np.random.default_rng(1731)
    nulls = np.empty(permutations)
    # Under a bijection only the matched trace changes, not the total matrix sum.
    for k in range(permutations):
        trace = s[np.arange(n), rng.permutation(n)].sum()
        nulls[k] = trace / n - (s.sum() - trace) / (n * (n - 1))
    matched = length_matched(s, left_chars, right_chars)
    return {
        "participants": n,
        "cosine_gap": float(point[0]),
        "gap_ci_low": float(ci[0, 0]),
        "gap_ci_high": float(ci[1, 0]),
        "same_person_percentile": float(point[1]),
        "percentile_ci_low": float(ci[0, 1]),
        "percentile_ci_high": float(ci[1, 1]),
        "same_cosine": float(np.trace(s) / n),
        "other_cosine": float((s.sum() - np.trace(s)) / (n * (n - 1))),
        "top_one_fraction": float(
            np.mean([np.mean(np.argmax(a, axis=1) == np.arange(n)) for a in (s, s.T)])
        ),
        "gap_permutation_p": float(
            (1 + np.sum(nulls >= point[0])) / (permutations + 1)
        ),
        "length_matched_gap": float(matched[0]),
        "length_matched_percentile": float(matched[1]),
        "length_match_mean_log_error": float(matched[2]),
        "bootstrap_draws": draws,
        "permutations": permutations,
    }


def holm(values):
    """Holm family correction retaining every frozen primary contrast."""
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    result = np.empty_like(p)
    result[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    )
    return result


def compare_views(metadata, vectors, *, draws=1000, permutations=1999):
    """Produce seven primary comparisons plus predeclared role/pooling controls."""
    rows, matrices, pair_metadata = [], {}, {}
    specs = []
    for pooling in ("token", "equal_turn"):
        for left, right in PRIMARY:
            specs.append((left, "user", right, "user", pooling, pooling == "token"))
        for role in ("assistant", "mixed"):
            specs.append(("direct_all", "user", "free_chat", role, pooling, False))
        specs.append(("free_chat", "user", "free_chat", "assistant", pooling, False))
        for q in ("meps_q1", "meps_q2", "meps_q3"):
            for role in ("user", "assistant"):
                specs.append((q, "user", "split:" + q, role, pooling, False))

    def select(condition, role, pooling):
        if condition.startswith("split:"):
            kind, condition = "split_chat", condition[6:]
        else:
            kind = "direct_answer" if condition.startswith("meps_") else condition
        return metadata[
            (metadata.view_kind == kind)
            & (metadata.condition == condition)
            & (metadata.role == role)
            & (metadata.pooling == pooling)
        ]

    for left, left_role, right, right_role, pooling, primary in specs:
        a = select(left, left_role, pooling)
        b = select(right, right_role, pooling)
        if (
            a.participant_code.duplicated().any()
            or b.participant_code.duplicated().any()
        ):
            raise ValueError("Duplicate author/view correspondence")
        ids = sorted(set(a.participant_code) & set(b.participant_code))
        name = f"{left}_{left_role}__{right}_{right_role}__{pooling}"
        if len(ids) < 3:
            rows.append(
                {
                    "contrast": name,
                    "primary": primary,
                    "status": "INSUFFICIENT",
                    "participants": len(ids),
                }
            )
            continue
        ia = [int(a.index[a.participant_code == i][0]) for i in ids]
        ib = [int(b.index[b.participant_code == i][0]) for i in ids]
        s = normalize(vectors[ia]) @ normalize(vectors[ib]).T
        result = correspondence(
            s,
            metadata.loc[ia, "chars"],
            metadata.loc[ib, "chars"],
            draws=draws,
            permutations=permutations,
        )
        rows.append(
            {"contrast": name, "primary": primary, "status": "COMPLETE", **result}
        )
        matrices[name] = s
        pair_metadata[name] = {"participants": ids, "left_rows": ia, "right_rows": ib}
    frame = pd.DataFrame(rows)
    primary = frame.primary
    if primary.sum() != 7 or (frame.loc[primary, "status"] != "COMPLETE").any():
        raise ValueError("Primary family incomplete")
    frame.loc[primary, "gap_holm_p"] = holm(frame.loc[primary, "gap_permutation_p"])
    return frame, matrices, pair_metadata
