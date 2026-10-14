#!/usr/bin/env python3
"""Source-bound observation selection; quoted content remains in the ledger."""

from __future__ import annotations

import numpy as np

HYPOTHESES = {
    "self": "The quoted passage is presented as the writer's own account of their views or experiences.",
    "other": "The quoted passage is presented as another person's account of their views or experiences, rather than the writer's own account.",
}
WRAPPERS = {
    "original": {
        "self": "This is my own account, not a friend's account: ",
        "other": "This is a friend's account, not my own account: ",
    },
    "written": {
        "self": "I wrote the following about what I personally experienced: ",
        "other": "A friend wrote the following about what they personally experienced: ",
    },
    "explained": {
        "self": "I explained what happened to me in these words: ",
        "other": "My friend explained what happened to them in these words: ",
    },
}


def quote_premise(text, quoted_passage):
    """Scope an attribution claim to a verbatim passage within its full source."""
    if (
        not isinstance(text, str)
        or not isinstance(quoted_passage, str)
        or not quoted_passage.strip()
        or quoted_passage not in text
    ):
        raise ValueError("Nonempty exact quote in source text required")
    return (
        "Source text:\n"
        + text
        + "\n\nQuoted passage under examination:\n"
        + quoted_passage
    )


def classify_source(probabilities):
    """Require exactly one entailed source claim; unresolved is not an other label."""
    if set(probabilities) != set(HYPOTHESES):
        raise ValueError("Both source hypotheses required")
    winners = {}
    for owner, values in probabilities.items():
        if set(values) != {"entailment", "contradiction", "neutral"}:
            raise ValueError("Unexpected NLI label encoding")
        p = np.asarray(list(values.values()), float)
        if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-5:
            raise ValueError("Finite normalized NLI probabilities required")
        # A tied winning class is not a determinate entailment.
        best = max(values.values())
        winners[owner] = [k for k, v in values.items() if abs(v - best) < 1e-12]
    supported = [owner for owner, labels in winners.items() if labels == ["entailment"]]
    route = supported[0] if len(supported) == 1 else "unresolved"
    return {
        "source_owner": route,
        "entailed_claims": supported,
        "reason": "unique_entailment"
        if len(supported) == 1
        else "conflicting_claims"
        if supported
        else "neither_claim_entailed",
    }


def select_observations(excerpts, decisions):
    """Retain original order and all context records, never substitute a zero score."""
    if len(excerpts) != len(decisions) or not excerpts:
        raise ValueError("Matched nonempty excerpt/decision sequences required")
    if any(not isinstance(t, str) or not t.strip() for t in excerpts):
        raise ValueError("Empty text is not an observation")
    owners = [d["source_owner"] for d in decisions]
    if not set(owners) <= {"self", "other", "unresolved"}:
        raise ValueError("Unknown source route")
    selected = [i for i, owner in enumerate(owners) if owner == "self"]
    return {
        "selected_indices": selected,
        "selected_texts": [excerpts[i] for i in selected],
        "context_indices": [i for i in range(len(excerpts)) if i not in selected],
        "source_owners": owners,
        "status": "OBSERVED" if selected else "NO_OBSERVED_SELF_EVIDENCE",
        "excerpt_coverage": len(selected) / len(excerpts),
        "score_when_unobserved": None,
        "all_texts_retained": list(excerpts),
    }
