#!/usr/bin/env python3
"""Executed negative cases for metadata units, joins and source projections."""

import csv
import sqlite3

import pandas as pd
import pytest

from grain_research.gptpro_foundation_v3.f14_metadata import (
    MetadataProfile,
    MetadataSpec,
    audit_join,
    inspect_schema,
    normalize_metadata,
    read_projected,
)
from scripts.run_grain_gptpro_f14_metadata import cache_author_keys, placeholder_checks


def test_real_adapter_placeholder_checks():
    result = placeholder_checks()
    assert len(result["checks"]) == 9
    assert result["all_passed"]


def test_original_pandora_anonymizer_not_fuzzy_identity():
    import hashlib

    values = pd.DataFrame({"author": ["SIM_A", "SIM_B"]})
    expected = [
        hashlib.sha256(f"grain_vm02_raw_contexts_20260905::{v}".encode()).hexdigest()[
            :24
        ]
        for v in values.author
    ]
    mapped = cache_author_keys(values, "pandora")
    assert mapped.author.tolist() == expected
    result = audit_join(
        mapped,
        pd.DataFrame({"author": expected}),
        ["author"],
        require="one_to_one",
        complete_left=True,
    )
    assert result["intersection_keys"] == 2
    assert cache_author_keys(values, "essays").equals(values)


@pytest.mark.parametrize("field", ["body", "text", "Extraversion", "score", "email"])
def test_prohibited_value_projection(tmp_path, field):
    with pytest.raises(ValueError, match="content/questionnaire"):
        MetadataSpec("bad", "author", ("id",), extras=(field,)).columns()
    with pytest.raises(ValueError, match="content/questionnaire"):
        inspect_schema(tmp_path / "none.csv", [field])


def test_csv_header_only_and_projection(tmp_path):
    path = tmp_path / "tiny.csv"
    with path.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["author", "id", "text", "Openness"])
        w.writerow(["NA", "1", "PRIVATE_MARKER", "4.9"])
    spec = MetadataSpec("tiny", "author", ("id",))
    schema = inspect_schema(path, spec.columns())
    assert schema["missing_columns"] == []
    frame = next(read_projected(path, spec))
    assert list(frame.columns) == ["author", "id"]
    assert frame.author.iloc[0] == "NA"
    assert "PRIVATE_MARKER" not in str(frame)
    assert inspect_schema(path, ["absent"])["missing_columns"] == ["absent"]
    with pytest.raises(ValueError, match="missing metadata"):
        next(read_projected(path, MetadataSpec("bad", "absent", ("id",))))


def test_parquet_projection(tmp_path):
    path = tmp_path / "tiny.parquet"
    pd.DataFrame({"author": ["SIM"], "id": ["e"], "body": ["PRIVATE"]}).to_parquet(path)
    spec = MetadataSpec("tiny", "author", ("id",))
    assert inspect_schema(path, spec.columns())["types"]["author"] in {
        "string",
        "large_string",
    }
    assert list(next(read_projected(path, spec)).columns) == ["author", "id"]


def test_missing_and_wrong_time_units():
    spec = MetadataSpec("tiny", "author", ("id",), time="t", time_kind="unix_seconds")
    frame = pd.DataFrame(
        {
            "author": ["SIM"] * 4,
            "id": list("abcd"),
            "t": ["", "0", "1704067200000", "1704067200"],
        }
    )
    out = normalize_metadata(frame, spec)
    assert out.time_missing.tolist() == [1, 0, 0, 0]
    assert out.time_invalid.tolist() == [0, 1, 1, 0]
    assert out.time.isna().tolist() == [True, True, True, False]


def test_clock_only_and_timezone_conversion():
    spec = MetadataSpec("tiny", "u", ("id",), time="t", time_kind="iso8601_tz")
    frame = pd.DataFrame(
        {
            "u": ["SIM"] * 3,
            "id": list("abc"),
            "t": ["09:00:00", "2024-01-01T09:00:00", "2024-01-01T09:00:00+09:00"],
        }
    )
    out = normalize_metadata(frame, spec)
    assert out.time_invalid.tolist() == [1, 1, 0]
    assert out.time.iloc[-1] == "2024-01-01T00:00:00.000000Z"


def test_compound_keys_cannot_collide():
    spec = MetadataSpec("tiny", "u", ("a", "b"))
    frame = pd.DataFrame(
        {"u": ["SIM"] * 3, "a": ["a|b", "a", ""], "b": ["c", "b|c", "c"]}
    )
    out = normalize_metadata(frame, spec)
    assert out.event.iloc[0] != out.event.iloc[1]
    assert pd.isna(out.event.iloc[2])


def test_actual_join_multiplicity_and_conflicts():
    left = pd.DataFrame({"id": ["a", "a", "b"], "author": ["SIM1", "SIM1", "SIM2"]})
    right = pd.DataFrame({"id": ["a", "b"], "author": ["SIM1", "SIM2"]})
    result = audit_join(
        left,
        right,
        ["id"],
        require="many_to_one",
        complete_left=True,
        agree=("author",),
    )
    assert result["joined_rows"] == 3
    assert result["intersection_keys"] == 2
    with pytest.raises(ValueError, match="context mismatch"):
        audit_join(left, right.assign(author="WRONG"), ["id"], agree=("author",))


def test_duplicate_views_not_new_events(tmp_path):
    spec = MetadataSpec("tiny", "u", ("id",), ("c",), "t", "unix_seconds", "lang")
    frame = pd.DataFrame(
        {
            "u": ["SIM1", "SIM1", "SIM1", "SIM2"],
            "id": ["e1", "e1", "e2", "e3"],
            "c": ["A", "B", "A", "A"],
            "t": ["1704067200", "1704067200", "1704153600", ""],
            "lang": ["en", "en", "ja", "en"],
        }
    )
    profile = MetadataProfile(tmp_path / "data.sqlite")
    try:
        profile.add(normalize_metadata(frame, spec))
        r = profile.availability_report()
        assert (r["rows"], r["events"], r["duplicate_event_extra_rows"]) == (4, 3, 1)
        assert r["authors_multiple_days"] == 1
        assert r["authors_multiple_languages"] == 1
        assert r["events_multiple_conditions"] == 1
        assert r["same_author_condition_multiple_days"] == 1
        assert r["event_author_conflicts"] == 0
        assert not r["all_opportunities_observed"]
    finally:
        profile.close()
    with pytest.raises(sqlite3.ProgrammingError):
        profile.db.execute("SELECT 1")


def test_source_conflict_is_not_silently_deduplicated(tmp_path):
    spec = MetadataSpec("tiny", "u", ("id",), time="t", time_kind="unix_seconds")
    frame = pd.DataFrame(
        {"u": ["SIM1", "SIM2"], "id": ["e", "e"], "t": ["1704067200", "1704153600"]}
    )
    profile = MetadataProfile(tmp_path / "data.sqlite")
    try:
        profile.add(normalize_metadata(frame, spec))
        r = profile.availability_report()
        assert r["event_author_conflicts"] == 1
        assert r["event_time_conflicts"] == 1
    finally:
        profile.close()
