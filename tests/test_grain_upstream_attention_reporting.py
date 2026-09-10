#!/usr/bin/env python3
"""Report identities must not collide with numeric attention channel names."""

import pandas as pd

from scripts.run_grain_upstream_attention_channels import CHANNELS, identity_fields


def test_numeric_channels_preserve_author_and_query_groups():
    rows = []
    for author in range(2):
        for q in ("neuroticism", "confirmation_seeking"):
            for f in range(4):
                r = identity_fields(
                    {"key": f"author{author}", "cohort": "fixture"}, "own", f, q
                )
                assert not set(r).intersection(CHANNELS)
                rows.append(r | dict.fromkeys(CHANNELS, float(f)))
    df = pd.DataFrame(rows)
    assert df.author_key.nunique() == 2
    assert df.query_name.nunique() == 2
    assert df.groupby(["cohort", "mode", "query_name"]).size().tolist() == [8, 8]
    assert df.groupby(
        ["cohort", "mode", "query_name"]
    ).author_key.nunique().tolist() == [2, 2]
