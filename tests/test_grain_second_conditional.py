#!/usr/bin/env python3
"""Check score reuse and fixed-cohort fidelity without private material or models."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_grain_second_conditional import ARMS, compact_packet, fidelity


def test_compact_packet_preserves_scores_and_rejects_bad_units():
    packet = {
        "author_id": "synthetic",
        "fold": 1,
        "views": [
            {"view": i, "scores": {arm: i / 10 for arm in ARMS}, "replay_error": 1e-16}
            for i in range(3)
        ],
    }
    result = compact_packet(packet, True)
    assert result["reused_s16"] is True
    assert result["views"] == packet["views"]
    packet["views"][2]["view"] = 1
    with pytest.raises(ValueError, match="distinct"):
        compact_packet(packet, True)
    packet["views"][2]["view"] = 2
    packet["views"][2]["scores"]["exact_mass_second"] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        compact_packet(packet, False)


def test_fidelity_keeps_bias_in_mse():
    x = np.arange(10.0)
    frame = pd.DataFrame(
        {"exact": x, "exact_mass_cov": x + 2, "exact_mass_second": x + 0.5}
    )
    rows = fidelity(frame, "synthetic")
    assert rows[0]["mse_over_exact_variance"] == 0
    assert rows[1]["exact_r"] == pytest.approx(1)
    assert rows[1]["mse_over_exact_variance"] == pytest.approx(4 / np.var(x))
    assert rows[2]["mean_error"] == pytest.approx(0.5)
