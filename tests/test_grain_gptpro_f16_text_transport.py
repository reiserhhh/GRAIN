#!/usr/bin/env python3
"""Isolation, missing support, actual categories and transport-mixture contracts."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f16_text_transport import (
    calibrated_channel,
    channel_diagnostics,
    channels,
    material_bank,
    observation_cache,
)


def scored_bank():
    """Provide deterministic fake READOUTS for algebra tests, not experiment results."""
    return [
        dict(
            r,
            pole_readout=float(r["pole"]),
            grain_qualification=float(r["qualification"]),
            nli_qualification=0.7,
        )
        for r in material_bank()
    ]


def test_unique_disjoint_materials_and_pilot_partition():
    rows = material_bank()
    assert len(rows) == len({r["text"] for r in rows}) == 380
    assert [
        sum(r["role"] == role for r in rows)
        for role in ("source", "pilot", "application")
    ] == [80, 100, 200]
    pilot = [r for r in scored_bank() if r["pilot_partition"] == "fit"]
    result = calibrated_channel(
        pilot, "grain_qualification", required_role="pilot", partition="fit"
    )
    assert len(pilot) == 60
    assert np.array_equal(np.sum(result["counts"], -1), np.full((2, 2), 15))


def test_no_anchor_cannot_be_created_by_prior():
    pilot = [
        r for r in scored_bank() if r["role"] == "pilot" and r["qualification"] == 1
    ]
    result = calibrated_channel(pilot, "grain_qualification", required_role="pilot")
    assert result["status"] == "MISSING_ANCHOR"
    with pytest.raises(ValueError, match="MISSING_ANCHOR"):
        channels(result, 1, 200)


def test_application_refused_for_pilot_fit():
    with pytest.raises(ValueError, match="role"):
        calibrated_channel(
            scored_bank()[-8:], "grain_qualification", required_role="pilot"
        )


def test_no_designed_labels_in_observation_export():
    cache = observation_cache(scored_bank(), "new")
    for r in cache.values():
        assert set(r) == {
            "text",
            "text_sha256",
            "pole_readout",
            "grain_qualification",
            "nli_qualification",
        }
        assert r["grain_qualification"] == 0.7


def test_channel_rank_and_plugin_draws():
    source = [r for r in scored_bank() if r["role"] == "source"]
    old = calibrated_channel(source, "grain_qualification", required_role="source")
    new = calibrated_channel(source, "nli_qualification", required_role="source")
    assert channel_diagnostics(old)["empirical_rank"] == 4
    assert channel_diagnostics(new)["empirical_rank"] == 2
    assert np.allclose(channels(old, 1, 200).sum(-1), 1)
    assert np.array_equal(channels(old, 1, 200), channels(old, 1, 200))
    plugin = channels(old, 1, 200, "plugin")
    assert np.array_equal(plugin[0], plugin[-1])
    assert np.array_equal(channels(old, 1, 1, "known_pool")[0].reshape(4, 4), np.eye(4))
