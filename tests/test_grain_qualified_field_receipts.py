#!/usr/bin/env python3
"""Test serialization recovery and independent saved-endpoint accounts."""

import json

import pytest

from scripts import recover_grain_qualified_target_field_hm155 as C


def test_csv_scalar_types_are_json_safe(tmp_path):
    """CSV recovery must not reproduce the NumPy integer serialization failure."""
    path = tmp_path / "characters.csv"
    path.write_text(
        "group,bank,condition,order,mode,axis,family,mask,logodds\n"
        "0,1,0,1,natural,extraversion,named,5,0.25\n"
        "1,0,1,0,natural,neuroticism,named,6,-0.05\n",
        encoding="utf-8",
    )
    rows = C.table(path)
    count = sum(bool(row["logodds"] > 0) for row in rows)
    payload = {"positive": count, "rows": rows}
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert type(count) is int
    assert type(rows[0]["mask"]) is int
    assert rows[1]["logodds"] < 0


def test_undefined_action_is_preserved(tmp_path):
    """Undefined directions are null, never silently imputed as zero."""
    path = tmp_path / "actions.csv"
    path.write_text(
        "group,mode,cosine,signed_projection\n0,e_local,,\n", encoding="utf-8"
    )
    assert C.table(path)[0]["cosine"] is None
    assert C.table(path)[0]["signed_projection"] is None


def test_saved_endpoint_accounts_without_model_forwards():
    """When the finite bundle is available, recheck every scalar aggregation."""
    if not (C.R.DEFAULT / "capture_seal.json").exists():
        pytest.skip("HM155 finite public synthetic bundle is not present")
    audit, mean_mass, minimum_mass, self_error = C.scalar_accounts(C.R.DEFAULT)
    assert audit["checks"] == 448
    assert audit["max_error"] < 1e-10
    assert audit["new_forwards"] == 0
    assert 0 < minimum_mass <= mean_mass <= 1
    assert self_error == 0
