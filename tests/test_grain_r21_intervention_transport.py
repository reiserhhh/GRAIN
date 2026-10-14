import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("it", ROOT / "scripts/run_grain_r21_intervention_transport.py")
it = importlib.util.module_from_spec(spec)
spec.loader.exec_module(it)


def frame(n=6, seed=0, corpus="x"):
    rng = np.random.default_rng(seed)
    d = {"key": [f"{corpus}{i}" for i in range(n)], "corpus": corpus}
    for c in it.CONDS:
        d[f"N_{c}"] = rng.normal(size=n)
        d[f"Q_{c}"] = 0.3 + 0.2 * d[f"N_{c}"] + rng.normal(scale=0.01, size=n)
    return pd.DataFrame(d)


def test_fit_uses_only_orig_and_neu():
    src = frame()
    m1 = it.fit_map(src)
    src2 = src.copy()
    src2[["Q_pos", "N_pos", "Q_neg", "N_neg", "Q_oth", "N_oth"]] = 99.0
    assert it.fit_map(src2) == m1


def test_singular_source_reports_nonidentified_slope():
    src = frame()
    src["N_orig"] = 1.0
    src["N_neu"] = 1.0
    m = it.fit_map(src)
    assert not m["slope_identified"] and m["b"] == 0.0


def test_dmap_equals_b_times_dN_and_permutation_preserves_means():
    app = frame(seed=1)
    m = {"a": 0.1, "b": 0.5}
    dQ, dM = it.signatures(app, m)
    assert np.allclose(dM[:, 0], 0.5 * (app.N_pos - app.N_neu))
    perm = np.random.default_rng(2).permutation(len(app))
    assert np.allclose(dM[perm].mean(0), dM.mean(0))


def test_perfect_transport_has_zero_signature_discrepancy():
    app = frame(seed=3)
    for c in it.CONDS:
        app[f"Q_{c}"] = 0.2 + 0.4 * app[f"N_{c}"]
    s = it.sig_stats(*it.signatures(app, {"a": 0.2, "b": 0.4}))
    assert abs(s["population_signature_discrepancy_norm"]) < 1e-12 and s["S_matched"] < 1e-20
