#!/usr/bin/env python3
"""Explicit clock, no-imputation, source isolation and dense numerical tests."""

import copy
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3 import f16_cutoff as adapter
from grain_research.gptpro_foundation_v3 import f16_independent_t3 as t3
from grain_research.gptpro_foundation_v3 import f16_state_instrument as t2
from grain_research.gptpro_foundation_v3.f16_cutoff_audit import dense_moments


@pytest.fixture(scope="module")
def sources():
    """Build small actual numerical instruments with no repository data files."""
    seed = 20260915
    records = {
        f"a{a}": [
            {
                "author_id": f"a{a}",
                "role": "source",
                "world": "core",
                "episode_id": "e",
                "event_id": f"a{a}:{i}",
                "t": i,
                "z": (-1) ** i,
                "y": float(a - 1 + 0.4 * (-1) ** i + 0.2 * np.sin(i)),
            }
            for i in range(8)
        ]
        for a in range(3)
    }
    ensemble = [
        {
            "status": "FITTED",
            "seed": seed,
            "world": "core",
            "bootstrap": b,
            "source_ids": sorted(records),
            "fit": {
                "prior": {
                    "mu": [0, 0.3, 0.4],
                    "Sigma": np.diag([1, 0.5, 0.3]).tolist(),
                },
                "log_sigma": [-0.6 + 0.03 * b, -0.8],
            },
        }
        for b in range(2)
    ]
    brief = {
        "world": "core",
        "seeds": [seed],
        "source_authors": 3,
        "reference_prefix": 8,
        "application_prefix": 8,
        "source_bootstraps": 2,
        "personal_draws": 12,
        "T3": {
            "target_id": "T3",
            "future_conditions": [1, -1, 1, -1],
            "unit": "future observation units",
        },
    }
    provenance = {"scope": "synthetic unit test"}
    manifest = {
        "seed": seed,
        "world": "core",
        "source_prefixes": records,
        "ensemble": ensemble,
        "provenance": provenance,
        "source_hash": t3.object_hash(provenance),
    }
    base3, _ = t3.build_instrument(brief, manifest)
    base2 = {
        "target_id": "T2",
        "object_class": "dynamic_state",
        "formula_version": t2.FORMULA,
        "world_contract": "F05_core_single_state",
        "world": "core",
        "seed": seed,
        "reader_hash": t2.READER_HASH,
        "scorer_code_hash": t2.code_hash(),
        "source_hash": {},
        "unit": "state",
        "observed_role": "synthetic response",
        "invalidating_result": "model mismatch",
        "support_rule": "borrowed",
        "error_model": "Laplace",
        "explanation_rule": "state",
        "reference_mu": 0.0,
        "reference_sigma": 1.0,
        "source_reference_authors": 3,
        "personal_draws": 12,
        "source_models": [
            {"bootstrap": e["bootstrap"], "model": e["fit"]} for e in ensemble
        ],
    }
    rows = copy.deepcopy(records["a1"])
    for row in rows:
        row["role"] = "application"
    return {"T2": base2, "T3": base3}, rows


def make_raw(base, rows, cutoff=7):
    """Construct the declared input, not an implicit last-record target."""
    return {
        "world": "core",
        "reader_hash": base["reader_hash"],
        "target_time": cutoff,
        "records": rows,
    }


@pytest.mark.parametrize("target", ["T2", "T3"])
def test_complete_prefix_matches_unchanged_legacy(sources, target):
    bases, rows = sources
    base = bases[target]
    result = adapter.cold_score(make_raw(base, rows), adapter.build_adapter(base))
    old = (
        t2.cold_score(
            {
                "world_contract": base["world_contract"],
                "reader_hash": base["reader_hash"],
                "records": rows,
            },
            base,
        )
        if target == "T2"
        else t3.cold_score(
            {"world": "core", "reader_hash": base["reader_hash"], "prefix": rows}, base
        )
    )
    score = old["score"] if target == "T2" else old["raw_score"]
    variance = (
        old["variance"] if target == "T2" else old["variance_components"]["total"]
    )
    assert result["supported"]
    assert result["score"] == score
    # Same finite formula; summation wrappers can differ by one floating-point bit.
    np.testing.assert_allclose(result["variance"], variance, atol=1e-10, rtol=1e-8)
    assert result["propagation_steps"] == 0
    assert result["calibration_status"] == "NOT_EVALUATED"


@pytest.mark.parametrize("target", ["T2", "T3"])
@pytest.mark.parametrize("rho", [-0.98, -0.4, 0.0, 0.6, 0.989])
@pytest.mark.parametrize("empty", [False, True])
def test_requested_time_matches_full_covariance(sources, target, rho, empty):
    _, rows = sources
    rows = [] if empty else [rows[i] for i in (0, 2, 5)]
    params = np.array(
        [[0.2, -0.3, np.arctanh(rho / 0.99)], [-0.5, 0.8, np.arctanh(-rho / 0.99)]]
    )
    noise = np.log([0.5, 0.3])
    mean, var, _, _ = adapter.conditional_moments(target, params, rows, noise, 13)
    expected_m, expected_v = dense_moments(params, rows, noise, 13, target)
    np.testing.assert_allclose(mean, expected_m, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(var, expected_v, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("target", ["T2", "T3"])
def test_trailing_missingness_preserves_target_and_is_not_zero(sources, target):
    bases, rows = sources
    base = bases[target]
    bundle = adapter.build_adapter(base)
    result = adapter.cold_score(make_raw(base, rows[:4]), bundle)
    assert result["target_time"] == 7
    assert result["last_observed_time"] == 3
    assert result["propagation_steps"] == 4
    assert result["observed_count"] == 4
    assert abs(result["score"] - result["old_last_observation_target"]["score"]) > 1e-4
    assert result["future_times"] == ([8, 9, 10, 11] if target == "T3" else None)
    fabricated = copy.deepcopy(rows)
    for row in fabricated[4:]:
        row["y"] = 0.0
    wrong = adapter.cold_score(make_raw(base, fabricated), bundle)
    assert abs(wrong["score"] - result["score"]) > 1e-4


@pytest.mark.parametrize("target", ["T2", "T3"])
def test_empty_is_stationary_prior_not_observed_zero(sources, target):
    bases, _ = sources
    base = bases[target]
    bundle = adapter.build_adapter(base)
    result = adapter.cold_score(make_raw(base, [], 47), bundle)
    assert result["supported"] and result["prior_only"]
    assert result["observed_count"] == 0 and result["variance"] > 0
    assert result["last_observed_time"] is None and result["propagation_steps"] is None
    assert result["old_last_observation_target"] is None
    other = adapter.cold_score(make_raw(base, [], 100), bundle)
    assert result["score"] == other["score"] and result["variance"] == other["variance"]


@pytest.mark.parametrize("target", ["T2", "T3"])
def test_no_failed_fit_dropped(sources, target, monkeypatch):
    bases, rows = sources
    monkeypatch.setattr(
        adapter, "parameter_draws", lambda *a: (None, {"status": "FAILED"})
    )
    result = adapter.cold_score(
        make_raw(bases[target], rows), adapter.build_adapter(bases[target])
    )
    assert not result["supported"] and result["unsupported_count"] == 2
    assert result["score"] is None and result["variance"] is None


@pytest.mark.parametrize("cutoff", [True, -1, 4, 7.5, "7", None])
def test_invalid_cutoff_rejected(sources, cutoff):
    bases, rows = sources
    with pytest.raises(ValueError):
        adapter.cold_score(
            make_raw(bases["T2"], rows, cutoff), adapter.build_adapter(bases["T2"])
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_id", "other"),
        ("author_id", "other"),
        ("y", None),
        ("z", 0.2),
        ("t", True),
    ],
)
def test_row_contract_rejects_mixed_or_missing(sources, field, value):
    bases, rows = sources
    changed = copy.deepcopy(rows)
    changed[0][field] = value
    with pytest.raises(ValueError):
        adapter.cold_score(
            make_raw(bases["T2"], changed), adapter.build_adapter(bases["T2"])
        )


def test_reader_truth_fields_and_hash_rejected(sources):
    bases, rows = sources
    bundle = adapter.build_adapter(bases["T2"])
    raw = make_raw(bases["T2"], rows)
    for change in ({"reader_hash": "wrong"}, {"true_state": 1}, {"world": "F09"}):
        with pytest.raises(ValueError):
            adapter.cold_score(dict(raw, **change), bundle)
    changed = copy.deepcopy(bundle)
    changed["base_bundle"]["reference_mu"] = 10.0
    with pytest.raises(ValueError):
        adapter.cold_score(raw, changed)


@pytest.mark.parametrize("target", ["T2", "T3"])
def test_time_translation_and_identity_independence(sources, target):
    bases, rows = sources
    base = bases[target]
    bundle = adapter.build_adapter(base)
    original = adapter.cold_score(make_raw(base, rows[:4]), bundle)
    changed = copy.deepcopy(rows[:4])
    for row in changed:
        row.update(
            t=row["t"] + 100,
            author_id="new",
            episode_id="new",
            event_id="new" + row["event_id"],
        )
    translated = adapter.cold_score(make_raw(base, changed, 107), bundle)
    assert original["score"] == translated["score"]
    assert original["variance"] == translated["variance"]


def test_propagation_composes():
    p = np.array([[0.0, 0.0, 0.9], [0.0, 0.0, -0.9]])
    m, v, noise = np.array([0.8, -0.3]), np.array([0.2, 0.3]), [-0.6, -0.8]
    m1, v1 = adapter.propagate(m, v, p, noise, 3)
    actual = adapter.propagate(m1, v1, p, noise, 7)
    expected = adapter.propagate(m, v, p, noise, 10)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_real_cold_subprocess(sources, tmp_path):
    bases, rows = sources
    base = bases["T2"]
    bundle = adapter.build_adapter(base)
    raw = make_raw(base, rows[:4])
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    (tmp_path / "input.jsonl").write_text(json.dumps(raw) + "\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            adapter.__name__,
            "--bundle",
            str(path),
            "--sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "--input",
            str(tmp_path / "input.jsonl"),
            "--output",
            str(tmp_path / "scores.jsonl"),
        ],
        check=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    assert json.loads((tmp_path / "scores.jsonl").read_text()) == adapter.cold_score(
        raw, bundle
    )
