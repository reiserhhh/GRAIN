#!/usr/bin/env python3
"""Reader composition keeps original events and oracle truth out of real arms."""

import gzip
import json

import pandas as pd
import pytest

from scripts.run_grain_gptpro_f16_reader_composition import compose_readings, histories


def test_nonoracle_inputs_do_not_use_designed_truth():
    readers = {
        name: pd.DataFrame({"pole_score": [p], "qualification_score": [q]}, index=["m"])
        for name, p, q in (
            ("nli", 0.8, 0.3),
            ("grain", 0.5, 0.9),
            ("lexical", 0.7, 0.8),
        )
    }
    a, b = compose_readings("m", readers, 0, 0), compose_readings("m", readers, 1, 1)
    assert all(a[k] == b[k] for k in a if k != "oracle")
    assert a["nli_pole_grain_qualification"] == (0.8, 0.9)
    assert a["nli_pole_lexical_qualification"] == (0.8, 0.8)
    assert a["oracle"] != b["oracle"]


def test_histories_group_events_not_each_event_id(tmp_path):
    path = tmp_path / "events.gz"
    rows = [
        {
            "role": "application",
            "panel": "mixed",
            "author_id": "SIM",
            "event_id": f"c2_xor_20260915_{i:08}",
            "t": i,
        }
        for i in range(48)
    ]
    with gzip.open(path, "wt") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    assert list(histories(path)) == [rows]
    with gzip.open(path, "wt") as out:
        for row in rows[:-1]:
            out.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="missing or reordered"):
        list(histories(path))
