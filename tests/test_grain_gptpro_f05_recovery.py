"""Dense Gaussian, shared-shock and observed-only F05 recovery tests."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f05 import eta_from_rho, grouped
from grain_research.gptpro_foundation_v3.f05_recovery import (
    dense_posterior,
    fit_person,
    fit_source_dynamics,
    generate_dynamics,
    kalman_loglik,
)


def person():
    """A fixed observed prefix, with no latent fields."""
    rows, _ = generate_dynamics(
        "core", 91, {"source": 4, "validation": 0, "application": 0}, 32
    )
    return next(iter(grouped(rows).values()))


@pytest.mark.parametrize("gap", [1, 3])
def test_filter_matches_dense_irregular_time(gap):
    rows = person()[::gap]
    psi = [0.3, -0.2, eta_from_rho(0.7)]
    logs = np.log([0.4, 0.6])
    actual = kalman_loglik(psi, rows, logs)
    expected = dense_posterior(psi, rows, logs)
    assert actual["loglik"] == pytest.approx(expected["loglik"], abs=1e-10)
    assert actual["state_mean"][-1] == pytest.approx(expected["state_mean"], abs=1e-10)
    assert actual["state_variance"][-1] == pytest.approx(
        expected["state_variance"], abs=1e-10
    )


def test_shared_shock_is_shared_and_role_isolated():
    _, truth = generate_dynamics(
        "shared_shock", 17, {"source": 4, "validation": 3, "application": 4}, 32
    )
    source = [x for x in truth if x["role"] == "source"]
    application = [x for x in truth if x["role"] == "application"]
    np.testing.assert_array_equal(
        source[0]["common_innovations"], source[1]["common_innovations"]
    )
    assert source[0]["common_innovations"] != application[0]["common_innovations"]
    assert source[0]["innovations"] != source[1]["innovations"]


def test_budget_and_source_count_do_not_change_application():
    counts = {"source": 4, "validation": 0, "application": 3}
    full, _ = generate_dynamics("core", 18, counts, 32)
    small, _ = generate_dynamics("core", 18, {**counts, "source": 8}, 16)
    a = [r for r in full if r["role"] == "application" and r["t"] < 16]
    b = [r for r in small if r["role"] == "application"]
    assert a == b


def test_person_fit_uses_observations_not_truth():
    rows = person()
    baseline = fit_person(rows, np.log([0.5, 0.5]))
    poisoned = fit_person([{**r, "truth": 12345} for r in rows], np.log([0.5, 0.5]))
    assert baseline == poisoned
    assert len(baseline["attempts"]) == 2
    assert np.all(np.isfinite(baseline["psi"]))


def test_source_two_rounds_and_role_gate():
    rows, _ = generate_dynamics(
        "core", 19, {"source": 4, "validation": 0, "application": 0}, 16
    )
    model = fit_source_dynamics(rows)
    assert len(model["rounds"]) == 2
    assert all(r["total_authors"] == 4 for r in model["rounds"])
    assert np.linalg.eigvalsh(model["prior"]["Sigma"]).min() >= 1e-4 - 1e-10
    assert all(np.log(0.05) <= v <= np.log(5) for v in model["log_sigma"])
    with pytest.raises(ValueError, match="source"):
        fit_source_dynamics([{**r, "role": "application"} for r in rows])


def test_duplicate_and_reversed_times_rejected():
    rows = person()
    psi = [0, 0.5, eta_from_rho(0.5)]
    with pytest.raises(ValueError):
        kalman_loglik(psi, rows + [rows[0]], np.log([0.5, 0.5]))
    with pytest.raises(ValueError):
        kalman_loglik(psi, rows[::-1], np.log([0.5, 0.5]))


def test_source_future_excluded():
    rows, _ = generate_dynamics(
        "core", 52, {"source": 4, "validation": 0, "application": 0}, 64
    )
    expected = fit_source_dynamics(rows)
    changed = fit_source_dynamics(
        [{**r, "y": r["y"] + (10000 if r["t"] >= 48 else 0)} for r in rows]
    )
    assert expected == changed
    assert expected["source_events_used"] == 4 * 48


def test_four_step_forecast_dense_state_reference():
    from grain_research.gptpro_foundation_v3.f05_recovery import score_prefix

    psi = [0.2, 0.3, eta_from_rho(0.6)]
    model = {
        "prior": {"mu": psi, "Sigma": np.eye(3).tolist()},
        "log_sigma": np.log([0.4, 0.5]).tolist(),
    }
    rows = person()
    actual = score_prefix(rows, model, [1, -1, 1, -1], fit_personal=False)
    state = dense_posterior(psi, rows, model["log_sigma"])
    prediction = actual["forecasts"][3]
    assert prediction["mean"] == pytest.approx(0.2 - 0.3 + 0.6**4 * state["state_mean"])
    assert prediction["variance"] == pytest.approx(
        0.6**8 * state["state_variance"] + 0.16 * (1 + 0.6**2 + 0.6**4 + 0.6**6) + 0.25
    )
