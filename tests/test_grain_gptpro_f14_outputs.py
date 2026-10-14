#!/usr/bin/env python3
"""Independent aggregate audit rejects broken identities and keeps X tags."""

import pandas as pd
import pytest

from scripts.audit_grain_gptpro_f14_outputs import recount_x, validate_aggregate_counts


def test_aggregate_identities_reject_corruption():
    p = {
        "rows": 3,
        "events": 2,
        "duplicate_event_extra_rows": 1,
        "missing_event_rows": 0,
        "calendar_rows": 2,
        "invalid_time_rows": 0,
        "missing_time_rows": 1,
        "role_rows": {"author": 3},
        "language_rows": {"en": 3},
    }
    validate_aggregate_counts(p, 3)
    for key in ("events", "calendar_rows", "missing_time_rows"):
        with pytest.raises(ValueError):
            validate_aggregate_counts(p | {key: p[key] + 1}, 3)
    with pytest.raises(ValueError, match="row total"):
        validate_aggregate_counts(p, 4)
    with pytest.raises(ValueError, match="role_rows"):
        validate_aggregate_counts(p | {"role_rows": {"author": 2}}, 3)


def test_recount_preserves_collection_tags_without_creating_events(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame(
        {
            "post_id": ["p1", "p1", "p2", "p3"],
            "account_id": ["SIM_A", "SIM_A", "SIM_A", "SIM_B"],
            "timestamp": ["2024-01-01T00:00:00Z"] * 2 + ["2024-01-02T00:00:00Z"] * 2,
            "symbol": ["A", "B", "A", ""],
            "query_group": ["Q", "Q", "Q", ""],
            "lang": ["en", "en", "ja", "en"],
            "text": ["PRIVATE_NOT_READ"] * 4,
        }
    ).to_csv(path, index=False)
    r = recount_x([path])
    assert (r["rows"], r["events"], r["duplicate_event_extra_rows"]) == (4, 3, 1)
    assert r["events_multiple_conditions"] == 1
    assert r["events_without_any_complete_condition"] == 1
    assert r["same_author_condition_multiple_days"] == 1
    assert r["authors_multiple_days"] == r["authors_multiple_languages"] == 1
    assert r["event_author_conflicts"] == 0
    assert "PRIVATE_NOT_READ" not in str(r)
