#!/usr/bin/env python3
"""Known positive/confounded/null worlds for conditional target readers."""

import copy
import itertools

import numpy as np
import pytest

from grain_research.construction import math_profile_reader as M


def record():
    """Supply one distinct numerical event per addressed role."""
    return {
        "frame_key": "fixture",
        "view": 0,
        "events": [
            {"event_id": str(t), "slot": t, "coordinates": [0.0] * 15}
            for t in (0, 4, 8)
        ],
    }


def world():
    """Give all32profiles support,with target0 correlated with target1."""
    y = np.concatenate(
        [
            np.tile(p, (8 if p[0] == p[1] else 2, 1))
            for p in itertools.product((0, 1), repeat=5)
        ]
    )
    x = np.column_stack(
        (y[:, 1], y[:, 0] + 3 * y[:, 1], y[:, 0] * y[:, 1], np.ones(len(y)))
    )
    return x.astype(float), y


def bundle():
    """Build a constant-feature fixture with no hidden application labels."""
    fields = np.zeros((2, 5, 4, 4096))
    fields[:, :, :, 0] = 1
    edges = np.zeros((2, 5, 16, 4096))
    edges[:, :, :, 0] = 2
    return {
        "version": M.VERSION,
        "code_hash": M.code_hash(),
        "targets": list(M.TARGETS),
        "components": list(M.COMPONENTS),
        "fields": fields.tolist(),
        "edges": edges.tolist(),
        "omega": np.full((5, 16), 1 / 16).tolist(),
        "interpretation": "source_descriptive_profile_contrasts_not_probability_or_causal_trait",
    }


def test_confounded_only_and_direct_signal_are_distinguished():
    x, y = world()
    result = M.construct(x, y)
    assert result["fields"][0, 0, 0] > 0.5
    np.testing.assert_allclose(result["fields"][0, 1, :2], [0, 1], atol=1e-14)
    np.testing.assert_allclose(result["fields"][0, :, 3], 0, atol=1e-14)


def test_exact_balance_and_opposite_group_weight_formula():
    x, y = world()
    r = M.construct(x, y)
    for q in range(5):
        c = M.profile_codes(y, q)
        for g in (0, 1):
            mass = np.bincount(
                c[y[:, q] == g],
                weights=np.abs(r["weights"][q, y[:, q] == g]),
                minlength=16,
            )
            np.testing.assert_allclose(mass, r["omega"][q], atol=1e-14)
        np.testing.assert_allclose(r["weights"][q] @ x, r["fields"][q, 1], atol=1e-14)


def test_three_term_identity_and_author_order():
    x, y = world()
    a = M.construct(x, y)
    np.testing.assert_allclose(a["fields"][:, 0], a["fields"][:, 1:].sum(1), atol=1e-14)
    order = np.random.default_rng(4).permutation(len(y))
    b = M.construct(x[order], y[order])
    np.testing.assert_allclose(a["fields"], b["fields"], atol=1e-14)


def test_missing_profile_support_is_not_smoothed():
    x, y = world()
    with pytest.raises(ValueError, match="Unobserved"):
        M.construct(x[~np.all(y == 0, axis=1)], y[~np.all(y == 0, axis=1)])


def test_cold_inputs_no_opportunity_and_labels():
    b, r = bundle(), record()
    M.validate_bundle(b)
    out = M.cold_record(b, r)
    np.testing.assert_allclose(out["retained_fields"], 1 / 64)
    np.testing.assert_allclose(out["conditional_edges"], 2)
    r["events"] = r["events"][:2]
    out = M.cold_record(b, r)
    assert (
        out["conditional_fields"] is None and np.max(np.abs(out["retained_edges"])) == 0
    )


@pytest.mark.parametrize(
    "kind", ["label", "duplicate", "extra_bundle", "bad_label", "bad_feature"]
)
def test_input_contract_rejects_leakage_and_bad_values(kind):
    b, r = copy.deepcopy(bundle()), record()
    x, y = world()
    if kind == "label":
        r["target"] = 1
        with pytest.raises(ValueError):
            M.cold_record(b, r)
    elif kind == "duplicate":
        r["events"][1]["event_id"] = r["events"][0]["event_id"]
        with pytest.raises(ValueError):
            M.cold_record(b, r)
    elif kind == "extra_bundle":
        b["source_ids"] = [1]
        with pytest.raises(ValueError):
            M.validate_bundle(b)
    elif kind == "bad_label":
        y[0, 0] = 2
        with pytest.raises(ValueError):
            M.construct(x, y)
    else:
        x[0, 0] = np.nan
        with pytest.raises(ValueError):
            M.construct(x, y)
