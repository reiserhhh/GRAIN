"""Independent scalar audit fixtures and targeted readout ledger attacks."""

import copy

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.response_readout import (
    decompose_logits,
    summarize,
)
from scripts.audit_grain_response_depth import compare
from scripts.audit_grain_response_readout import audit_frame


@pytest.fixture
def ledger():
    rows, checks = [], []
    rng = np.random.default_rng(731)
    for i, length in enumerate((4, 10, 17)):
        a, m = rng.normal(size=(2, length - 1, 23))
        values, error = decompose_logits(a, m, rng.integers(23, size=length - 1))
        rows.extend(
            {
                "participant_code": str(i),
                "response_token_position": j + 2,
                **{k: float(v[j]) for k, v in values.items()},
                "state_cosine": 0.9,
                "state_displacement_norm": 1.0,
            }
            for j in range(length - 1)
        )
        checks.append(
            {
                "participant_code": str(i),
                "response_tokens": length,
                "head_batches": 2 * ((length + 6) // 8),
                "algebra_max_error": error,
                "head_replay_max_error": 0,
                "mean_predictor_state_cosine": 0.99,
            }
        )
    return pd.DataFrame(rows), checks


def test_full_person_weighted_replay(ledger):
    frame, checks = ledger
    persons, effects, error = audit_frame(frame, checks)
    expected, summary = summarize(frame, checks)
    pd.testing.assert_frame_equal(persons, expected)
    compare(effects, summary)
    assert error < 1e-9


@pytest.mark.parametrize(
    "attack",
    [
        "drop",
        "duplicate",
        "position",
        "kl",
        "gradient",
        "cosine",
        "head_count",
        "algebra_error",
        "nan",
    ],
)
def test_inconsistent_readout_record_fails(ledger, attack):
    f, checks = ledger
    checks = copy.deepcopy(checks)
    if attack == "drop":
        f = f.iloc[1:]
    elif attack == "duplicate":
        f = pd.concat([f.iloc[1:], f.iloc[1:2]])
    elif attack == "position":
        f.loc[0, "response_token_position"] = 1
    elif attack == "kl":
        f.loc[0, "kl_matched_to_actual"] = -1.0
    elif attack == "gradient":
        f.loc[0, "actual_gradient_dot_delta"] += 1.0
    elif attack == "cosine":
        f.loc[0, "state_cosine"] = 1.1
    elif attack == "head_count":
        checks[0]["head_batches"] = 0
    elif attack == "algebra_error":
        checks[0]["algebra_max_error"] = 0.1
    else:
        f.loc[0, "delta_logp"] = float("nan")
    with pytest.raises(ValueError):
        audit_frame(f, checks)
