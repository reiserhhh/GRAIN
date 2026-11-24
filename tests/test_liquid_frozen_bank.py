#!/usr/bin/env python3
import numpy as np
import pytest
from grain_core import liquid_frozen_bank as bank


def test_conditional_transport_is_causal_and_commutes_with_affine_coordinates():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(7, 3))
    r = rng.random((7, 4))
    a = rng.normal(size=(3, 5))
    b = rng.normal(size=5)
    y = bank.mix(x, r)
    np.testing.assert_allclose(bank.mix(x @ a + b, r), y @ a + b, atol=1e-14)
    w = bank.mix(np.eye(7), r)
    np.testing.assert_allclose(w.sum(1), 1)
    np.testing.assert_array_equal(np.triu(w, k=1), 0)
    np.testing.assert_allclose(w @ x, y, atol=1e-14)
    np.testing.assert_allclose(bank.mix(x[:3], r[:3]), y[:3])
    np.testing.assert_array_equal(x[:1], bank.mix(x[:1], r[:1]))
    assert not np.allclose(y, bank.mix(x, r.mean(1, keepdims=True)))


def test_nonfinite_and_untrusted_artifact_fail(tmp_path):
    with pytest.raises(ValueError):
        bank.mix([[1], [2]], [[1], [float("nan")]])
    fake = tmp_path / "f.joblib"
    fake.write_bytes(b"not a released artifact")
    with pytest.raises(ValueError, match="known released"):
        bank.old_runtime("native", fake)
