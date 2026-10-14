#!/usr/bin/env python3
"""F14-03 role negative checks: role is source metadata, never inferred meaning."""

import pandas as pd
import pytest

from grain_research.gptpro_foundation_v3.f14_metadata import (
    MetadataSpec,
    audit_join,
    normalize_metadata,
)


def frame(role_values):
    return pd.DataFrame({"author": ["a1", "a2", "a3"], "post": ["p1", "p2", "p3"], "who": role_values})


def test_role_conflict_between_views_is_rejected():
    left = normalize_metadata(frame(["author", "author", "author"]), MetadataSpec("x", "author", ("post",), role="who"))
    right = normalize_metadata(frame(["author", "quoted_speaker", "author"]), MetadataSpec("x", "author", ("post",), role="who"))
    with pytest.raises(ValueError, match="join context mismatch: role"):
        audit_join(left, right, ["author", "event"], require="one_to_one", agree=("role",))


def test_absent_role_defaults_to_author_and_is_not_qualification():
    spec = MetadataSpec("x", "author", ("post",))
    out = normalize_metadata(frame(["x", "y", "z"]).drop(columns="who"), spec)
    assert set(out["role"]) == {"author"}
    assert spec.contract()["source_role"] == "not semantic self-qualification"


def test_role_values_are_passed_through_not_mapped_to_self_or_other():
    out = normalize_metadata(frame(["self", "quoted", None]), MetadataSpec("x", "author", ("post",), role="who"))
    assert out["role"].tolist() == ["self", "quoted", None]
