#!/usr/bin/env python3
"""Exact original-text sidecars for frozen nonoverlapping GRAIN observations."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from suica_core.suica import TOKEN_RE, tokenize


def token_index(raw):
    """Index original Unicode character spans without normalizing source text."""
    result = {}
    for row, record in raw.items():
        matches = list(TOKEN_RE.finditer(record["body"]))
        result[int(row)] = {
            **record,
            "tokens": [m.group() for m in matches],
            "starts": np.array([m.start() for m in matches], int),
            "ends": np.array([m.end() for m in matches], int),
        }
    return result


def check_spec(spec):
    """Refuse operators for which monotonically consumed source tokens are false."""
    if spec.kind not in {"whole", "native", "semantic", "fixed"}:
        raise ValueError("Unsupported sidecar operator")
    if spec.kind == "fixed" and (
        spec.stride_tokens != spec.window_tokens
        or spec.offset_tokens != 0
        or spec.offset_strategy != "fixed"
        or spec.source_boundary_mode not in {"cross", "within"}
    ):
        raise ValueError("Sidecar requires nonoverlapping offset-zero windows")


def align_units(units, index, spec):
    """Recover monotonic source segments; never search for a matching substring.

    Tail truncation is permitted but skipping then resuming inside one comment
    is not. Token comparisons fail closed on such ambiguous operators. Raw
    formatting is preserved through offsets, not substituted into frozen units.
    """
    check_spec(spec)
    if units.unit_id.duplicated().any():
        raise ValueError("Duplicate observation identity")
    cursors, rows = defaultdict(int), []
    for unit in units.sort_values(["user_id", "unit_index"]).itertuples():
        if unit.operator != spec.name:
            raise ValueError("Operator provenance mismatch")
        sources = [int(x) for x in str(unit.source_rows).split(",")]
        weights = [int(x) for x in str(unit.source_token_weights).split(",")]
        tokens = tokenize(unit.text)
        if (
            len(sources) != len(weights)
            or len(set(sources)) != len(sources)
            or any(w <= 0 for w in weights)
            or sum(weights) != len(tokens)
            or len(tokens) != unit.token_count
        ):
            raise ValueError("Invalid token provenance weights")
        consumed = 0
        for source, width in zip(sources, weights):
            original = index[source]
            if str(original["author"]) != str(unit.user_id):
                raise ValueError("Cross-author provenance")
            start, end = cursors[source], cursors[source] + width
            segment = tokens[consumed : consumed + width]
            if original["tokens"][start:end] != segment:
                raise ValueError("Unit/source token sequence mismatch")
            cstart, cend = (
                int(original["starts"][start]),
                int(original["ends"][end - 1]),
            )
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "unit_index": int(unit.unit_index),
                    "user_id": str(unit.user_id),
                    "source_row": source,
                    "token_start": start,
                    "token_end": end,
                    "unit_token_start": consumed,
                    "unit_token_end": consumed + width,
                    "char_start": cstart,
                    "char_end": cend,
                    "format_changed": original["body"][cstart:cend]
                    != " ".join(segment),
                }
            )
            cursors[source] = end
            consumed += width
    return pd.DataFrame(rows)


def sentence_coverage(sentences, segments, index):
    """Measure source-token retention and one-unit completeness, not validity."""
    by_source = defaultdict(list)
    for r in segments.itertuples():
        by_source[int(r.source_row)].append(r)
    rows = []
    for sentence_id, s in enumerate(sentences):
        source = int(s["source_row"])
        raw = index[source]
        if raw["body"][s["start"] : s["end"]] != s["text"]:
            raise ValueError("Stale evidence span")
        hits = np.flatnonzero((raw["starts"] < s["end"]) & (raw["ends"] > s["start"]))
        if not len(hits):
            raise ValueError("Evidence span has no project tokens")
        retained = set()
        unit_counts = defaultdict(int)
        for segment in by_source[source]:
            overlap = hits[(hits >= segment.token_start) & (hits < segment.token_end)]
            retained.update(overlap.tolist())
            unit_counts[segment.unit_id] += len(overlap)
        best = max(unit_counts.values(), default=0)
        rows.append(
            {
                "sentence_index": sentence_id,
                "source_row": source,
                "n_tokens": len(hits),
                "retained_tokens": len(retained),
                "retention": len(retained) / len(hits),
                "one_unit_completeness": best / len(hits),
                "full_retention": len(retained) == len(hits),
                "one_unit_complete": best == len(hits),
                "touching_units": sum(n > 0 for n in unit_counts.values()),
                "partial_source_token": bool(
                    raw["starts"][hits[0]] < s["start"]
                    or raw["ends"][hits[-1]] > s["end"]
                ),
            }
        )
    return pd.DataFrame(rows)


def author_rank_auc(left, right, *, draws=500, seed=916):
    """Own-versus-stranger cosine rank, with paired author-cluster bootstrap.

    One author has equal weight regardless of text count. Bootstrap jointly
    resamples anchor and gallery authors; repeated copies of an identity are
    never counted as strangers. This is not pair-independent ROC uncertainty.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Matched author matrices required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite coordinates")
    ln, rn = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
    valid = (ln > 1e-12) & (rn > 1e-12)
    a, b = left[valid] / ln[valid, None], right[valid] / rn[valid, None]
    n = len(a)
    if n < 3:
        raise ValueError("Too few nonzero matched authors")
    similarity = a @ b.T
    own = np.diag(similarity)
    scores = []
    for sim in (similarity, similarity.T):
        diff = own[:, None] - sim
        rank = (diff > 1e-12).astype(float) + 0.5 * (np.abs(diff) <= 1e-12)
        np.fill_diagonal(rank, 0)
        scores.append(rank)
    rank = (scores[0] + scores[1]) / 2
    per_author = rank.sum(1) / (n - 1)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=draws).astype(float)
    denominators = n - weights
    candidate_scores = weights @ rank.T
    boot = (
        np.sum(
            weights
            * np.divide(
                candidate_scores,
                denominators,
                out=np.zeros_like(candidate_scores),
                where=denominators > 0,
            ),
            axis=1,
        )
        / n
    )
    boot = boot[(weights > 0).sum(1) >= 2]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "authors": n,
        "zero_norm_excluded": int((~valid).sum()),
        "auc": float(per_author.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "own_cosine": float(own.mean()),
        "stranger_cosine": float((similarity.sum() - own.sum()) / (n * (n - 1))),
        "bootstrap_draws": len(boot),
    }
