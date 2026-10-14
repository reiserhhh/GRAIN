"""Time anchor and data equality for the missing-time paired experiment."""

import numpy as np

from grain_research.gptpro_foundation_v3.f05 import eta_from_rho
from grain_research.gptpro_foundation_v3.f05_recovery import (
    dense_posterior,
    generate_dynamics,
)
from scripts.run_grain_gptpro_f05_missing_time import paired_predictions


def test_same_data_correct_clock_dense_and_future_cutoff():
    events, _ = generate_dynamics(
        "core", 5, {"source": 1, "validation": 0, "application": 0}, 64
    )
    model = {
        "prior": {"mu": [0.0, 0.5, eta_from_rho(0.7)], "Sigma": np.eye(3).tolist()},
        "log_sigma": np.log([0.5, 0.5]).tolist(),
    }
    result, last, count = paired_predictions(events, model, 48, False)
    assert last == 45 and count == 16
    dense = dense_posterior(model["prior"]["mu"], events[:48:3], model["log_sigma"])
    np.testing.assert_allclose(
        result["correct_gap3"]["state_mean"], dense["state_mean"], atol=1e-12
    )
    changed = [{**r, "y": r["y"] + (1000 if r["t"] >= 48 else 0)} for r in events]
    assert paired_predictions(changed, model, 48, False) == (result, last, count)
    assert result["correct_gap3"]["state_mean"] != result["wrong_gap1"]["state_mean"]
