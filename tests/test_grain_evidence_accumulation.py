#!/usr/bin/env python3
"""Synthetic invariants for condition profiles, dependent evidence and revisions."""

import numpy as np
import pandas as pd
import pytest

from grain_research.evidence_accumulation import (
    calibration_fit,
    profile_fit,
    resolve_revisions,
)


def test_perfect_linear_profile():
    z = np.tile([-1.0, 1.0], 8)
    fit = profile_fit(
        z, 0.2 + 0.3 * z, np.arange(16), balance_events=True, cluster_errors=True
    )
    np.testing.assert_allclose(fit["beta"], [0.2, 0.3], atol=1e-14)


def test_repetitions_do_not_change_event_balanced_point():
    z = np.tile([-1.0, 1.0], 8)
    y = np.random.default_rng(1).normal(size=16)
    copies = np.repeat(np.arange(16), [21] + [1] * 15)
    a = profile_fit(z, y, np.arange(16), balance_events=True, cluster_errors=True)
    b = profile_fit(
        z[copies], y[copies], copies, balance_events=True, cluster_errors=True
    )
    np.testing.assert_allclose(a["beta"], b["beta"])
    assert b["effective_events_perfect_dependence"] == pytest.approx(16)


def test_clustered_duplicates_do_not_claim_independent_precision():
    rng = np.random.default_rng(2)
    z = np.tile([-1.0, 1.0], 12)
    y = 0.2 * z + rng.normal(size=24)
    ix = np.repeat(np.arange(24), 8)
    independent = profile_fit(
        z[ix], y[ix], ix, balance_events=False, cluster_errors=False
    )
    clustered = profile_fit(z[ix], y[ix], ix, balance_events=False, cluster_errors=True)
    assert np.trace(clustered["covariance"]) > 6 * np.trace(independent["covariance"])
    np.testing.assert_allclose(independent["beta"], clustered["beta"])


def test_single_condition_is_not_identified():
    assert (
        profile_fit(
            np.ones(8),
            np.arange(8),
            np.arange(8),
            balance_events=True,
            cluster_errors=True,
        )
        is None
    )


def test_revision_is_order_invariant_and_retractions_preserve_history():
    frame = pd.DataFrame(
        {
            "author": ["a"] * 3,
            "event": ["x", "y", "x"],
            "version": [1, 1, 2],
            "active": [True, True, False],
            "score": [1.0, 0.5, -1.0],
        }
    )
    a, history = resolve_revisions(frame)
    b, _ = resolve_revisions(frame.iloc[::-1])
    pd.testing.assert_frame_equal(a, b)
    assert a.event.tolist() == ["y"]
    assert len(history) == 3


def test_conflicting_revision_fails():
    f = pd.DataFrame(
        {
            "author": ["a"] * 2,
            "event": ["x"] * 2,
            "version": [1, 1],
            "active": [True, True],
            "score": [1.0, -1.0],
        }
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_revisions(f)


def test_source_calibration():
    coef = calibration_fit(np.array([-0.7, -0.7, 0.9, 0.9]), [-1, -1, 1, 1])
    np.testing.assert_allclose(coef, [-0.125, 1.25])


def test_wording_partition_and_condition_outcome_balance():
    import json

    from scripts.run_grain_condition_accumulation import ROOT, build_bank

    c = json.loads((ROOT / "configs/grain_condition_accumulation.json").read_text())
    f = build_bank(c)
    assert len(f) == 128
    assert not set(f[f.partition.eq("source")].context_variant) & set(
        f[f.partition.eq("held")].context_variant
    )
    assert not set(f[f.partition.eq("source")].response_variant) & set(
        f[f.partition.eq("held")].response_variant
    )
    assert set(f.groupby(["query", "partition", "condition"]).sign.mean()) == {0}
