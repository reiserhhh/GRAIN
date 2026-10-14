import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sct", ROOT / "scripts/run_grain_r14_soft_channel_trace.py")
sct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sct)

FITTED = {"b1": {"source_p": [0.2, 0.7, 0.6, 0.3]}, "b2": {"coef": [0.1, 0.2, 0.0, 0.1]},
          "m": {"mean": [0.4, 0.5, 0.5, 0.4], "right": np.eye(4)[:2].tolist()}}
INIT = np.array([[0.3, 0.2], [0.2, 0.3]])


def test_noinfo_adds_no_statistics_and_equals_soft_half_b1_in_filtered_mode():
    a = sct.NoInfoHistory(FITTED, INIT, 0.45, mode="filtered")
    b = sct.r14.QualifiedHistory(FITTED, INIT, 0.45, mode="filtered")
    for t in range(20):
        if t >= 2:
            assert abs(a.predict()["probabilities"]["b1"] - b.predict()["probabilities"]["b1"]) < 1e-15
        a.observe(0.5, 0.7)
        b.observe(0.5, 0.7)
    assert a.mass == 0 and a.counts.sum() == 0 and np.all(a.gram == 0)
    assert b.mass > 0 and b.counts.sum() > 0  # archived estimator recycles model-imputed counts


def test_half_reading_likelihood_is_flat():
    for q in (0.0, 0.3, 1.0):
        like = q * np.array([0.5, 0.5]) + (1 - q) / 2
        assert like[0] == like[1]


def test_noinfo_prediction_is_prior_only_status():
    a = sct.NoInfoHistory(FITTED, INIT, 0.45, mode="independent")
    for _ in range(5):
        a.observe(0.9, 1.0)
    assert a.predict()["status"] == "prior_only"
