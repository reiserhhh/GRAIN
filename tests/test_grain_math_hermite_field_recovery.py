#!/usr/bin/env python3
"""Tests for cache compatibility, including refusal of mismatched score aliases."""

import json

import pytest

from scripts.run_grain_math_hermite_field_r2 import read_json


def test_alias_does_not_mutate_cache(tmp_path):
    path = tmp_path / "node.json"
    payload = {"native": {"conditional_score": 0.25}, "native_derivative": 0.03}
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    assert read_json(path)["native"]["score"] == 0.25
    assert path.read_bytes() == original


def test_ambiguous_alias_rejected(tmp_path):
    path = tmp_path / "node.json"
    path.write_text(json.dumps({"native": {"conditional_score": 0.25, "score": 0.3}}))
    with pytest.raises(ValueError):
        read_json(path)
