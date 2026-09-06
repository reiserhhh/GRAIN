#!/usr/bin/env python3
"""Independent matching reconstruction and explicit lexical-copy heuristic tests."""

import numpy as np

from grain_research.vector_mechanisms.history_overlap import copy_overlap, shingles
from grain_research.vector_mechanisms.source_history import (
    profile,
    select_donor,
    split_view,
)
from scripts.audit_grain_source_history_material import (
    independent_match,
    independent_view,
)


class Tokenizer:
    """Word IDs suffice for independent selection arithmetic testing."""

    def encode(self, text, **kwargs):
        return text.split()


def view(uid, offset):
    """Synthetic time/thread-separated dictionary member."""
    return [
        {
            "author_id": uid,
            "comment_id": f"{uid}_{i}",
            "thread_id": f"t_{uid}_{i}",
            "body": f"unique text {uid} item {i}",
            "timestamp": i + offset,
            "subreddit": "general",
        }
        for i in range(12)
    ]


def test_independent_selection_is_equal_on_synthetic_dictionary():
    tok = Tokenizer()
    target = view("target", 100)
    h, current, pf = independent_view(target, tok)
    own, cur = split_view(target)
    assert h == own and cur == current
    produced = []
    independent = []
    for uid, offset in [("a", 0), ("b", 30), ("future", 200)]:
        h, _, p = independent_view(view(uid, offset), tok)
        independent.append({"author_id": uid, "view": 0, "history": h, "profile": p})
        produced.append(
            {
                "author_id": uid,
                "view": 0,
                "history": h,
                "profile": profile(h, lambda s: len(tok.encode(s))),
            }
        )
    a = select_donor(
        profile(own, lambda s: len(tok.encode(s))), current, target, produced
    )
    b, n = independent_match(pf, current, target, independent)
    assert (a["donor"]["author_id"], a["donor"]["view"]) == b[1:3]
    assert a["eligible_donors"] == n == 2
    np.testing.assert_allclose(
        [a[k] for k in ("cost", "length_error", "time_error", "subreddit_jaccard")],
        [b[0], b[4], b[5], b[6]],
        atol=1e-14,
    )


def test_copy_rule_retains_thresholds_and_does_not_name_semantic_noise():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    a = shingles(text)
    assert copy_overlap(a, shingles(text.upper()))["flagged"]
    assert copy_overlap(
        a, shingles("preface extra words " + text + " ending appendix words")
    )["flagged"]
    assert not copy_overlap(
        shingles("one two three four five"), shingles("one two three four five")
    )["flagged"]
    assert not copy_overlap(
        a,
        shingles(
            "unrelated conceptual sentence with different vocabulary and no repeated long fragment whatsoever"
        ),
    )["flagged"]
    altered = shingles(text.replace("eight", "CHANGED"))
    score = copy_overlap(a, altered)
    assert 0 < score["jaccard"] < 0.8 and not score["flagged"]
