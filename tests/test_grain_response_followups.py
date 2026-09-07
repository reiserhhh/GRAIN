#!/usr/bin/env python3
"""Pairing and immutable-text checks for the bounded functional follow-ups."""

import pandas as pd
import pytest

from grain_research.vector_mechanisms.direct_traits import render
from scripts.connect_grain_response_student import paired_source
from scripts.run_grain_numeric_query_probe import CONTROL_QUESTION, control_prompt


def test_external_connection_requires_identical_control_labels():
    rows = [
        {
            "author_id": str(i),
            "task": "big5",
            "trait": "neuroticism",
            "arm": "cue_mask",
            "observer": o,
            "y_true": i / 3,
            "prediction": i / 4,
        }
        for i in range(4)
        for o in ("native_query", "e5_large")
    ]
    frame = pd.DataFrame(rows)
    assert len(paired_source(frame, "neuroticism")) == 4
    frame.loc[0, "y_true"] = 42
    with pytest.raises(ValueError, match="disagree"):
        paired_source(frame, "neuroticism")


def test_new_question_leaves_excerpt_bodies_and_offsets_unchanged():
    texts = ["A short account of something.", "Another original excerpt."]
    before, spans_before = render(texts, "neuroticism")
    after, spans_after = control_prompt(texts)
    assert spans_before == spans_after
    assert CONTROL_QUESTION in after
    for lo, hi in spans_before:
        assert before[lo:hi] == after[lo:hi]
