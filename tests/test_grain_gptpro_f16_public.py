#!/usr/bin/env python3
"""Public contracts must preserve actual scores and reject semantic substitutions."""

import copy
import json

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3 import f16_public as public
from tests.test_grain_gptpro_f16_dynamic_text import bundle as dynamic_fixture
from tests.test_grain_gptpro_f16_dynamic_text import raw as dynamic_raw


def test_archived_comparator_handles_numpy_scalars():
    from scripts.run_grain_gptpro_f16_public import difference

    assert difference(np.float64(1.0), np.float64(1.0 + 2e-16)) < 1e-10
    with pytest.raises(ValueError):
        difference(np.float64("nan"), 1.0)


def build(tmp_path, target="T2"):
    """Create a small real engine with complete declared source provenance."""
    base = dynamic_fixture()
    base.update(
        target_ids=["T2", "T3"],
        object_classes=["dynamic_state", "future_response"],
        qualification_rate=0.5,
        source_hash={"source": "a" * 64},
        source_bundle_sha256="b" * 64,
        reader_bundle_sha256="c" * 64,
        source_reference_authors=8,
        unit="numeric source unit",
        invalidating_result="wrong observation model",
        support_rule="prior if empty",
        error_model="ADF working moments",
        explanation_rule="target relative",
    )
    base["reader_hash"] = public.object_hash(
        {"materials": base["materials"], "channels": base["channel_draws"]}
    )
    path = tmp_path / "source.json"
    path.write_text(json.dumps(base))
    descriptor = public.describe(base, target)
    brief = {
        k: descriptor[k]
        for k in (
            "target_id",
            "object_class",
            "unit",
            "world_contract",
            "observed_role",
        )
    }
    result = public.build_instrument(
        brief,
        {"path": str(path), "sha256": public.file_hash(path), "role": "source_fitted"},
    )
    raw = dynamic_raw(47)
    return (
        result,
        {"contract": public.request_contract(result), "data": raw},
        path,
        brief,
    )


@pytest.mark.parametrize("target", ["T2", "T3"])
def test_real_cold_scoring_matches_and_maps_target(tmp_path, target):
    b, r, _, _ = build(tmp_path, target)
    before = copy.deepcopy(b)
    score = public.cold_score(r, b)
    old = public.dynamic.cold_scores([r["data"]], b["base_bundle"])[0]
    assert score["engine_output"] == old
    assert score["score"] == old["targets"][target]["score"]
    assert score["object_class"] == public.CLASSES[target]
    assert b == before
    assert b["base_bundle"]["object_classes"][-1] == "future_response"


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_id", "T1"),
        ("object_class", "reader_judgment"),
        ("unit", "calendar score"),
        ("reader_hash", "d" * 64),
        ("source_hash", "e" * 64),
        ("world_contract", "other"),
        ("reference_mu", 999),
        ("reference_sigma", 1e-3),
        ("time_contract", {"unit": "days"}),
        ("condition_contract", {}),
        ("scorer_code_hash", "f" * 64),
        ("observed_role", "numeric_observation"),
    ],
)
def test_changed_metadata_is_not_a_valid_new_instrument(tmp_path, field, value):
    b, r, _, _ = build(tmp_path)
    b[field] = value
    with pytest.raises(ValueError):
        public.cold_score(r, b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_id", "T3"),
        ("unit", "renamed"),
        ("reader_hash", "e" * 64),
        ("world_contract", "F09"),
        ("time_contract", {"unit": "days"}),
        ("condition_contract", {"support": [0], "future": []}),
    ],
)
def test_request_declarations_enforced(tmp_path, field, value):
    b, r, _, _ = build(tmp_path)
    r["contract"][field] = value
    with pytest.raises(ValueError):
        public.cold_score(r, b)


def test_wrong_reader_cache_rejected_even_if_nested_hash_recomputed(tmp_path):
    b, r, _, _ = build(tmp_path)
    b["base_bundle"]["materials"]["a"]["pole_readout"] = 0.1
    b["base_bundle_hash"] = public.object_hash(b["base_bundle"])
    with pytest.raises(ValueError, match="reader identity"):
        public.cold_score(r, b)


def test_hidden_truth_missing_field_bad_time_and_role(tmp_path):
    b, r, path, brief = build(tmp_path)
    for key, value in [("truth", 0.2), ("target_time", 46), ("target_time", True)]:
        wrong = copy.deepcopy(r)
        wrong["data"][key] = value
        with pytest.raises(ValueError):
            public.cold_score(wrong, b)
    wrong = copy.deepcopy(b)
    del wrong["error_model"]
    with pytest.raises(ValueError):
        public.cold_score(r, wrong)
    with pytest.raises(ValueError):
        public.build_instrument(
            brief,
            {
                "path": str(path),
                "sha256": public.file_hash(path),
                "role": "application",
            },
        )
    with pytest.raises(ValueError):
        public.build_instrument(
            brief, {"path": str(path), "sha256": "0" * 64, "role": "source_fitted"}
        )


def test_batch_identity_reference_and_empty_history(tmp_path):
    b, r, _, _ = build(tmp_path)
    other = copy.deepcopy(r)
    other["data"]["author_id"] = "stranger"
    for row in other["data"]["history"]:
        row["observation"]["author_id"] = "stranger"
    first = public.cold_score(r, b)
    batch = public.cold_scores([r, other], b)
    np.testing.assert_allclose(first["score"], batch[0]["score"], atol=1e-12)
    np.testing.assert_allclose(first["score"], batch[1]["score"], atol=1e-12)
    empty = copy.deepcopy(r)
    empty["data"]["history"] = []
    assert public.cold_score(empty, b)["engine_output"]["prior_only"]
    assert public.cold_scores([], b) == []
