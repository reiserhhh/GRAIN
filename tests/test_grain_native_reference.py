#!/usr/bin/env python3
"""Minimal native-operator comparability and identity checks."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_grain_native_reference import align, native_observations


def test_native_operator_retains_provenance_and_enforces_budget():
    config = {
        "min_tokens_per_unit": 3,
        "max_units_per_user": 128,
        "max_source_comments_per_user": 3,
        "max_source_tokens_per_user": 12,
        "operators": [{"name": "native", "kind": "native"}],
    }
    comments = [
        {
            "body": "I enjoy a quiet afternoon reading books.",
            "timestamp": float(i),
            "subreddit": "books",
        }
        for i in range(6)
    ]
    b = [{"author_id": "a", "source_group": "big5_only", "views": [comments]}]
    obs, spec = native_observations(b, config)
    assert obs.token_count.sum() <= 12
    assert len(obs) <= 3
    assert obs.source_rows.is_unique
    assert set(obs.user_id) == {"a"}
    assert spec.kind == "native"
    b[0]["source_group"] = "bridge"
    with pytest.raises(ValueError, match="Big5-only"):
        native_observations(b, config)


def test_alignment_never_silently_reorders_or_drops_authors():
    f = pd.DataFrame({"user_id": ["b", "a"], "value": [2, 1]})
    assert align(f, np.array(["a", "b"])).value.tolist() == [1, 2]
    with pytest.raises(ValueError, match="mismatch"):
        align(f, np.array(["a", "c"]))
    with pytest.raises(ValueError, match="mismatch"):
        align(pd.concat([f, f]), np.array(["a", "b"]))
