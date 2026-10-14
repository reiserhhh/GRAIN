#!/usr/bin/env python3
"""Typed event evidence and condition-preserving aggregation.

NLI outputs are reader scores, not calibrated response or personality probabilities.
Events and conditions must have declared provenance; this module does not infer
missing opportunities or convert a single episode into a habitual self-report.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def split_controlled_reported_speech(text):
    """Route explicit friend quotations in the declared fixture grammar.

    All quoted spans and offsets are retained as other-person evidence. This
    limited grammar demonstration is not an open-domain coreference system;
    unmatched text remains untouched and unresolved rather than silently removed.
    """
    pattern = r"my friend (?:said|told me): '(?P<quote>.*?)'(?= (?:I have not described my own response\.|My own experience was different\.))"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
    other, writer, end = [], [], 0
    for match in matches:
        writer.append(text[end : match.start()])
        other.append(
            {
                "owner": "reported_friend",
                "start": match.start("quote"),
                "end": match.end("quote"),
                "text": match.group("quote"),
            }
        )
        end = match.end()
    writer.append(text[end:])
    return {
        "writer_text": "".join(writer),
        "other_spans": other,
        "rule": "declared_controlled_reporting_grammar",
        "matched_quotes": len(matches),
    }


def paired_evidence(positive, negative, threshold=0.5, margin=0.25):
    """Contrast two explicit propositions; abstain on weak or conflicting support."""
    positive, negative = np.broadcast_arrays(positive, negative)
    if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError("Threshold and margin must be in [0, 1]")
    if any(
        not np.isfinite(v).all() or np.any((v < 0) | (v > 1))
        for v in (positive, negative)
    ):
        raise ValueError("Finite NLI probabilities in [0, 1] required")
    contrast = positive - negative
    conflict = (positive >= threshold) & (negative >= threshold)
    accepted = (
        (np.maximum(positive, negative) >= threshold)
        & (abs(contrast) >= margin)
        & ~conflict
    )
    return {
        "contrast": contrast,
        "accepted": accepted,
        "conflict": conflict,
        "direction": np.where(accepted, np.sign(contrast), np.nan),
    }


def aggregate_events(frame):
    """One event/query contribution; return coverage separately from signed scores.

    Event IDs are only unique within an author and a query. Conflicting duplicates
    raise rather than being averaged. Context absence and reader abstention stay NaN.
    """
    keys = ["author_id", "query", "event_id"]
    values = ["condition", "sentence_hash", "score", "accepted"]
    if not set(keys + values).issubset(frame):
        raise ValueError("Missing event identity/provenance/scoring fields")
    if frame[keys + ["condition", "sentence_hash"]].isna().any().any():
        raise ValueError("Missing identity or design condition")
    if not pd.api.types.is_bool_dtype(frame.accepted):
        raise ValueError("Accepted must be boolean")
    if not np.isfinite(frame.score).all() or (frame.score.abs() > 1 + 1e-10).any():
        raise ValueError("Finite signed reader scores in [-1, 1] required")
    duplicates = frame[frame.duplicated(keys, keep=False)]
    if (
        len(duplicates)
        and (duplicates.groupby(keys)[values].nunique(dropna=False) > 1).any().any()
    ):
        raise ValueError("Conflicting copies of event")
    unique = frame.drop_duplicates(keys).copy()
    unique["observed_score"] = unique.score.where(unique.accepted)
    result = (
        unique.groupby(["author_id", "query", "condition"], sort=False)
        .agg(
            n_events=("event_id", "size"),
            n_accepted=("accepted", "sum"),
            mean_score=("observed_score", "mean"),
        )
        .reset_index()
    )
    result["coverage"] = result.n_accepted / result.n_events
    return result


def conditional_profile(aggregated):
    """Return balanced two-condition level and response contrast, not trait labels."""
    if not set(aggregated.condition).issubset({-1, 1}):
        raise ValueError("This reference profile requires declared -1/+1 conditions")
    table = aggregated.pivot(
        index=["author_id", "query"], columns="condition", values="mean_score"
    )
    table = table.reindex(columns=[-1, 1])
    return pd.DataFrame(
        {
            "level": (table[1] + table[-1]) / 2,
            "response": (table[1] - table[-1]) / 2,
        }
    ).reset_index()
