#!/usr/bin/env python3
"""Synthetic checks for natural-source function construction, not psychology."""

from copy import deepcopy

import numpy as np
import pytest

from grain_research.construction import math_observed_family as F


def source():
    """Construct a complete eight-cell law with two authors and joint outcomes."""
    x = np.zeros((16, 15))
    x[:, 0] = np.tile([-1, 1, -1, 1], 4)
    x[:, 5] = np.tile([-1, -1, 1, 1], 4)
    c = np.tile(np.repeat([0, 1], 4), 2)
    ids = np.repeat([0, 1], 8)
    y = np.column_stack([c, x[:, 0] > 0, x[:, 5] > 0])
    return x, c, y, ids


def record(x, c):
    """Make a strict outcome-free numeric author window."""
    return {
        "author_key": "new_author",
        "events": [
            {"event_id": str(i), "condition": int(ci), "coordinates": row.tolist()}
            for i, (row, ci) in enumerate(zip(x, c, strict=True))
        ],
    }


def test_oriented_observed_law_not_covariance_witness():
    x, c, y, ids = source()
    law = F.compile_law(x, c, y, ids)
    np.testing.assert_allclose(F.event_probabilities(law, x, c), y)
    np.testing.assert_allclose(law["reference"], [0.5, 0.5, 0.5])
    assert law["source_authors"] == 2


def test_author_composition_keeps_both_components_and_fixed_reference():
    x, c, y, ids = source()
    law = F.compile_law(x, c, y, ids)
    out = F.cold_author(law, record(x[[7, 7]], c[[7, 7]]))
    np.testing.assert_allclose(out["relative"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(out["choice"], [0.5, 0, 0])
    np.testing.assert_allclose(out["within"], [0, 0.5, 0.5])
    np.testing.assert_allclose(np.array(out["choice"]) + out["within"], out["relative"])
    other = record(x[[7, 7]], c[[7, 7]])
    other["author_key"] = "completely_different_identifier"
    assert F.cold_author(law, other)["relative"] == out["relative"]


def test_source_author_weight_not_publication_count():
    x, c, y, ids = source()
    x = np.concatenate([x[:8], np.repeat(x[8:9], 80, axis=0)])
    c = np.r_[c[:8], np.repeat(c[8], 80)]
    y = np.concatenate([y[:8], np.repeat(y[8:9], 80, axis=0)])
    ids = np.r_[ids[:8], np.ones(80, int)]
    law = F.compile_law(x, c, y, ids)
    np.testing.assert_allclose(
        law["reference"], 0.5 * y[:8].mean(0) + 0.5 * y[8:].mean(0)
    )
    assert not np.allclose(law["reference"], y.mean(0))


def test_unseen_support_is_missing_not_zero_or_prior():
    x, c, y, ids = source()
    law = F.compile_law(x[c == 0], c[c == 0], y[c == 0], ids[c == 0])
    out = F.cold_author(law, record(x[[7]], c[[7]]))
    assert out["status"] == "UNOBSERVED_SOURCE_CELLS"
    assert out["relative"] is None
    assert F.cold_author(law, record(x[:0], c[:0]))["status"] == "NO_OBSERVATIONS"


def test_future_fields_and_duplicate_carriers_rejected():
    x, c, y, ids = source()
    law = F.compile_law(x, c, y, ids)
    bad = record(x[:2], c[:2])
    bad["outcomes"] = y[:2].tolist()
    with pytest.raises(ValueError, match="excludes"):
        F.cold_author(law, bad)
    bad = record(x[:2], c[:2])
    bad["events"][1]["event_id"] = bad["events"][0]["event_id"]
    with pytest.raises(ValueError, match="identities"):
        F.cold_author(law, bad)
    bad = record(x[:1], c[:1])
    bad["events"][0]["personality"] = "high"
    with pytest.raises(ValueError, match="Current"):
        F.cold_author(law, bad)


@pytest.mark.parametrize("field", ["reference", "mass", "probabilities"])
def test_changed_units_or_source_reference_rejected(field):
    law = F.compile_law(*source())
    bad = deepcopy(law)
    if field == "reference":
        bad[field][0] += 0.01
    elif field == "mass":
        bad[field][0][0] += 0.01
    else:
        bad[field][0][0][0] = 0.25
    with pytest.raises(ValueError):
        F.verify_law(bad)


def test_observed_polarity_reverses_when_actual_outcome_reverses():
    x, c, y, ids = source()
    first, second = F.compile_law(x, c, y, ids), F.compile_law(x, c, 1 - y, ids)
    out, rev = (
        F.cold_author(first, record(x[[7]], c[[7]])),
        F.cold_author(second, record(x[[7]], c[[7]])),
    )
    np.testing.assert_allclose(out["relative"], -np.array(rev["relative"]))


def test_no_latent_truth_field_and_nonfinite_rejection():
    x, c, y, ids = source()
    law = F.compile_law(x, c, y, ids)
    bad = deepcopy(law)
    bad["latent_theta"] = [0.1]
    with pytest.raises(ValueError, match="latent"):
        F.verify_law(bad)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        F.compile_law(x, c, y, ids)
