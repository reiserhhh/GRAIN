#!/usr/bin/env python3
"""T2 score isolation, total variance and strict prefix input tests."""

import copy

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f16_state_instrument import (
    FORMULA,
    READER_HASH,
    code_hash,
    cold_score,
    finalize_reference,
    state_moments,
    validate_records,
)


@pytest.fixture
def case():
    """Tiny real two-source ensemble, with positive definite parameter priors."""
    models = [
        {
            "bootstrap": b,
            "model": {
                "prior": {
                    "mu": [b * 0.1, 0.5, 0.4],
                    "Sigma": np.diag([1.0, 0.25, 0.1]).tolist(),
                },
                "log_sigma": np.log([0.5, 0.5]).tolist(),
            },
        }
        for b in range(2)
    ]
    bundle = {
        "target_id": "T2",
        "object_class": "dynamic_state",
        "formula_version": FORMULA,
        "world_contract": "F05_core_single_state",
        "world": "core",
        "seed": 20260915,
        "reader_hash": READER_HASH,
        "scorer_code_hash": code_hash(),
        "source_hash": {},
        "unit": "state",
        "observed_role": "response",
        "invalidating_result": "misspecification",
        "support_rule": "inferred",
        "error_model": "Laplace",
        "explanation_rule": "model state",
        "reference_mu": 0.0,
        "reference_sigma": 1.0,
        "source_reference_authors": 3,
        "source_models": models,
        "personal_draws": 20,
    }
    rows = [
        {
            "author_id": "a",
            "event_id": str(i),
            "role": "application",
            "world": "core",
            "t": i,
            "z": (-1.0) ** i,
            "y": 0.2 + (-1.0) ** i * 0.4 + 0.03 * i,
        }
        for i in range(12)
    ]
    raw = {
        "world_contract": bundle["world_contract"],
        "reader_hash": READER_HASH,
        "records": rows,
    }
    return bundle, raw


def test_total_variance_and_prior(case):
    bundle, raw = case
    result = cold_score(raw, bundle)
    assert result["status"] == "MOMENTS_COMPUTED"
    assert result["variance"] == pytest.approx(sum(result["components"].values()))
    assert result["source_uncertainty_included"]
    assert result["calibration_check"] == "NOT_EVALUATED"
    prior = cold_score(dict(raw, records=[]), bundle)
    assert prior["prior_only"] and prior["score"] == 0 and prior["variance"] > 0


def test_id_and_batch_invariance(case):
    bundle, raw = case
    original = cold_score(raw, bundle)
    renamed = copy.deepcopy(raw)
    for row in renamed["records"]:
        row["author_id"] = "z"
        row["event_id"] += "new"
    assert cold_score(renamed, bundle) == original
    stranger = copy.deepcopy(raw)
    for row in stranger["records"]:
        row["y"] += 10
    cold_score(stranger, bundle)
    assert cold_score(raw, bundle) == original


@pytest.mark.parametrize("time", [True, 1.5, "1", None, -1])
def test_bad_time(case, time):
    _, raw = case
    raw["records"][0]["t"] = time
    with pytest.raises(ValueError):
        validate_records(raw["records"], "core")


def test_scope_and_future_fields(case):
    bundle, raw = case
    with pytest.raises(ValueError):
        cold_score(dict(raw, world_contract="F09"), bundle)
    raw["records"][0]["true_state"] = 1
    with pytest.raises(ValueError):
        cold_score(raw, bundle)


def test_source_norm_and_failure_retention(case):
    bundle, _ = case
    results = [{"status": "MOMENTS_COMPUTED", "score": s} for s in (-1, 0, 1)]
    fixed = finalize_reference(bundle, results)
    assert fixed["reference_mu"] == 0 and fixed["reference_sigma"] == 1
    results[-1]["status"] = "CURVATURE_UNSUPPORTED"
    with pytest.raises(ValueError):
        finalize_reference(bundle, results)


def test_no_source_omission(case):
    bundle, raw = case
    bundle["source_models"][1]["bootstrap"] = 5
    with pytest.raises(ValueError):
        cold_score(raw, bundle)
    bundle["source_models"][1]["bootstrap"] = 1
    result = state_moments(raw["records"], bundle)
    assert len(result["source_components"]) == 2


def test_failed_personal_fit_is_not_renormalized(case, monkeypatch):
    """A bad source-conditioned approximation does not disappear from a mixture."""
    import grain_research.gptpro_foundation_v3.f16_state_instrument as module

    bundle, raw = case
    monkeypatch.setattr(
        module, "laplace_person", lambda *_: {"status": "CURVATURE_UNSUPPORTED"}
    )
    result = cold_score(raw, bundle)
    assert result["status"] == "UNSUPPORTED_PERSONAL_FIT"
    assert result["bootstrap"] == 0
    assert "score" not in result and result["components"] is None


def test_duplicate_mixed_author_and_nonfinite(case):
    bundle, raw = case
    with pytest.raises(ValueError):
        cold_score(dict(raw, records=raw["records"] * 2), bundle)
    mixed = copy.deepcopy(raw)
    mixed["records"][0]["author_id"] = "someone_else"
    with pytest.raises(ValueError):
        cold_score(mixed, bundle)
    mixed = copy.deepcopy(raw)
    mixed["records"][0]["y"] = float("nan")
    with pytest.raises(ValueError):
        cold_score(mixed, bundle)


def test_time_translation_and_gap_use(case):
    """Ordinal time origin does not matter, but actual elapsed gaps do."""
    bundle, raw = case
    first = cold_score(raw, bundle)
    translated = copy.deepcopy(raw)
    for row in translated["records"]:
        row["t"] += 100
    assert cold_score(translated, bundle) == first
    gapped = copy.deepcopy(raw)
    for row in gapped["records"]:
        row["t"] *= 3
    changed = cold_score(gapped, bundle)
    assert not np.isclose(changed["score"], first["score"], atol=1e-6)


def test_public_factory_and_trust(case, tmp_path):
    """Public dispatch preserves actual scores and rejects unsupported targets."""
    import hashlib
    import json

    from grain_research.gptpro_foundation_v3.f16_factory import (
        build_instrument,
        load_bundle,
    )
    from grain_research.gptpro_foundation_v3.f16_factory import (
        cold_score as public_score,
    )

    bundle, raw = case
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    loaded = load_bundle(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert public_score(raw, loaded) == cold_score(raw, bundle)
    with pytest.raises(ValueError):
        load_bundle(path, "bad")
    with pytest.raises(ValueError):
        public_score(raw, dict(bundle, target_id="T3"))
    with pytest.raises(ValueError):
        build_instrument({"target_id": "T3"}, {})
