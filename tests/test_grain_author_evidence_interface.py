#!/usr/bin/env python3
"""The author interface requires a declared measurement object and valid law."""

import numpy as np
import pytest

from scripts.compose_grain_author_evidence import score_profiles


def fixture():
    """Two local scores have equal author mass but different partition mass."""
    observations = {
        "readings": np.array([[[0.0]], [[1.0]]]),
        "log_mass": np.log([[[1.0]], [[3.0]]]),
    }
    declaration = {
        "targets": ["anonymous_target"],
        "atom_convention": "token_sum",
        "atom_counts": [1, 1],
        "profiles": [
            {
                "name": "anonymous_author",
                "record_indices": [0, 1],
                "weights": [0.5, 0.5],
            }
        ],
    }
    return observations, declaration


def test_explicit_author_expectation_is_not_attention_weighted():
    observations, declaration = fixture()
    result = score_profiles(observations, declaration, "declared_expectation")
    assert result["scores"][0]["score_by_target"]["anonymous_target"] == 0.5


def test_partition_requires_scale_contract_and_remains_separate():
    observations, declaration = fixture()
    with pytest.raises(ValueError, match="scale contract"):
        score_profiles(observations, declaration, "calibrated_partition")
    declaration["partition_scale_contract"] = "same registered source scale"
    result = score_profiles(observations, declaration, "calibrated_partition")
    assert result["scores"][0]["score_by_target"]["anonymous_target"] == 0.75


def test_duplicates_do_not_silently_become_independent_events():
    observations, declaration = fixture()
    declaration["profiles"][0]["record_indices"] = [0, 0]
    with pytest.raises(ValueError, match="unique existing"):
        score_profiles(observations, declaration, "declared_expectation")


def test_empty_profiles_are_missing_not_zero():
    observations, declaration = fixture()
    declaration["profiles"] = []
    with pytest.raises(ValueError, match="missing observations"):
        score_profiles(observations, declaration, "declared_expectation")
