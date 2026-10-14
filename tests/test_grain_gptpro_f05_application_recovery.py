"""Observed-worker future isolation and explicit optimizer failure routing."""

import json
from types import SimpleNamespace

import numpy as np

from grain_research.gptpro_foundation_v3.f05 import eta_from_rho
from grain_research.gptpro_foundation_v3.f05_recovery import generate_dynamics
from scripts.run_grain_gptpro_f05_application_recovery import observed_worker


def test_future_values_do_not_change_prefix_outputs(tmp_path, monkeypatch):
    rows, _ = generate_dynamics(
        "core", 431, {"source": 2, "validation": 0, "application": 1}, 64
    )
    source = tmp_path / "source"
    source.mkdir()
    prior = {
        "mu": [0.0, 0.5, eta_from_rho(0.5)],
        "Sigma": np.diag([1.0, 0.25, 1.0]).tolist(),
    }
    (source / "source_model.json").write_text(
        json.dumps({"log_sigma": np.log([0.5, 0.5]).tolist(), "prior": prior})
    )
    for role in ("source", "application"):
        (source / f"{role}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows if row["role"] == role)
        )
    lmm = {
        "mode": "linear",
        "mu": [0.0, 0.5],
        "Sigma": np.diag([1.0, 0.25]).tolist(),
        "sigma2": 0.5,
        "fallback": False,
    }
    monkeypatch.setattr(
        "scripts.run_grain_gptpro_f05_application_recovery.fit_source_lmm",
        lambda _: lmm,
    )
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    observed_worker(source, first)
    poisoned = [
        {**row, "y": row["y"] + (1000 if row["t"] >= 48 else 0)}
        for row in rows
        if row["role"] == "application"
    ]
    (source / "application.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in poisoned)
    )
    observed_worker(source, second)
    a = json.loads((first / "observed_predictions.json").read_text())
    b = json.loads((second / "observed_predictions.json").read_text())
    assert [r for r in a if r["n"] <= 48] == [r for r in b if r["n"] <= 48]
    assert [r for r in a if r["n"] > 48] != [r for r in b if r["n"] > 48]


def test_person_failure_returns_declared_source_prior(monkeypatch):
    from grain_research.gptpro_foundation_v3.f05_recovery import fit_person

    rows, _ = generate_dynamics(
        "core", 12, {"source": 1, "validation": 0, "application": 0}, 16
    )
    prior = {"mu": [1.0, 0.3, eta_from_rho(0.4)], "Sigma": np.eye(3).tolist()}

    def fail(fun, x0, **kwargs):
        return SimpleNamespace(
            success=False,
            fun=float(fun(np.asarray(x0))),
            nit=100,
            message="test failure",
            x=np.asarray(x0),
        )

    monkeypatch.setattr(
        "grain_research.gptpro_foundation_v3.f05_recovery.minimize", fail
    )
    result = fit_person(rows, np.log([0.5, 0.5]), prior)
    assert not result["success"]
    assert result["psi"] == prior["mu"]
    assert result["fallback"] == "source_prior"
    assert len(result["attempts"]) == 2


def test_nullable_numpy_coverage_has_arithmetic_mean():
    import pandas as pd

    from scripts.audit_grain_gptpro_f05_application_recovery import numeric_coverage

    data = pd.DataFrame(
        {
            "group": [0] * 4,
            "covered": pd.Series(
                [np.bool_(True), np.bool_(True), np.bool_(False), None], dtype=object
            ),
        }
    )
    data["covered"] = numeric_coverage(data.covered)
    assert data.groupby("group").covered.mean().iloc[0] == 2 / 3
