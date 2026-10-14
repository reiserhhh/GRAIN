"""Focused tests for R10/R29 common-shock episodes (protocol V1 + A1)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import run_grain_r10_common_shock_episodes as rr  # noqa: E402
from grain_research.gptpro_foundation_v3.f05_joint import joint_filter, learn_common_variance  # noqa: E402
from grain_research.gptpro_foundation_v3.f05_recovery import filter_arrays, fit_source_dynamics  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F05/common_shock_episodes_r1"


def _panel(seed=3, n=4, T=12):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(T, n))
    z = np.array([[-1.0 if (t // 4) % 2 == 0 else 1.0] * n for t in range(T)])
    psi = np.column_stack([rng.normal(size=n), rng.normal(0.5, 0.2, n), rng.normal(0.5, 0.3, n)])
    return y, z, psi


def test_checks_gate_passed_exactly():
    c = json.loads((OUT / "checks.json").read_text())
    assert c["status"] == "PASS" and c["C1a_generator_rows_equal"]
    assert max(c["C1b_source_refit_max_err"].values()) == 0.0 and c["C1d_personal_psi_max_err"] == 0.0
    assert c["C1e_panel_rows"] == 6400 and max(c["C1e_panel_max_err"].values()) == 0.0
    assert c["C4_independence"]["distinct_authors"] == 32000 and c["C4_independence"]["common_arrays_all_distinct"]


def test_dense_versus_efficient_filter():
    d = rr.dense_check()
    assert d["status"] == "PASS" and d["max_mean_err"] < 1e-10 and d["max_var_err"] < 1e-10


def test_zero_common_equals_univariate_filter():
    y, z, psi = _panel()
    q, r = 0.3, 0.2
    f = joint_filter(y, z, psi, q, r, 0.0)
    ls = np.log(np.sqrt([q, r]))
    for i in range(y.shape[1]):
        u = filter_arrays(psi[i], y[:, i], z[:, i], np.arange(len(y)), ls)
        assert np.allclose(u["state_mean"], f["state_mean"][:, i], atol=1e-12)
        assert np.allclose(u["state_variance"], f["state_variance"][:, i], atol=1e-12)


def test_joint_filter_uses_no_future_rows():
    y, z, psi = _panel(seed=5)
    f1 = joint_filter(y, z, psi, 0.25, 0.25, 0.1)
    y2 = y.copy(); y2[7:] = -9.0
    f2 = joint_filter(y2, z, psi, 0.25, 0.25, 0.1)
    assert np.array_equal(f1["state_mean"][:7], f2["state_mean"][:7])
    assert not np.allclose(f1["state_mean"][7:], f2["state_mean"][7:])


def test_source_functions_reject_application_rows():
    rows = [{"role": "application", "author_id": "x", "t": 0, "y": 0.0, "z": -1.0, "event_id": "x:0"}]
    with pytest.raises(ValueError):
        fit_source_dynamics(rows)
    with pytest.raises(ValueError):
        learn_common_variance(rows, {"rounds": [{"personal_fits": {}}], "log_sigma": [0.0, 0.0]})


def test_episode_completeness_and_disjoint_authors():
    eps = sorted(p for p in (OUT / "episodes").iterdir() if p.is_dir() and not p.name.startswith("._"))
    assert len(eps) == 64
    seen = set()
    for p in eps:
        m = pd.read_csv(p / "author_metrics.csv")
        assert len(m) == 200 * 3 * 2 * 2
        ids = set(m.author_id)
        assert len(ids) == 200 and not (ids & seen)
        seen |= ids
        info = json.loads((p / "episode.json").read_text())
        assert info["source_authors"] == 300 and sum(info["personal_fit_failures"].values()) == 0


def test_primary_contrast_recomputed_independently():
    rows = []
    for p in sorted((OUT / "episodes").glob("shared_shock_*")):
        if p.name.startswith("._"):
            continue
        ev = pd.read_csv(p / "events.csv.gz")
        ev = ev[(ev.budget == 48) & (ev.method == "personal_MAP")]
        e2 = (ev.state_mean - ev.state_truth) ** 2
        per = ev.assign(e2=e2).groupby(["author_id", "arm"]).e2.mean().unstack("arm")
        rows.append((per.joint - per.independent).mean())
    c = pd.read_csv(OUT / "paired_contrasts.csv")
    prim = c[(c.world == "shared_shock") & (c.budget == 48) & (c.method == "personal_MAP") & (c.metric == "state_err2")].iloc[0]
    assert len(rows) == 32
    assert prim.mean_diff == pytest.approx(float(np.mean(rows)), abs=1e-14)
    assert prim.episode_se == pytest.approx(float(np.std(rows, ddof=1) / np.sqrt(32)), abs=1e-14)


def test_cold_replay_of_one_episode():
    info, auth = rr.run_episode("shared_shock", 20261001, None)
    saved = pd.read_csv(OUT / "episodes/shared_shock_20261001/author_metrics.csv", float_precision="round_trip")
    cols = [c for c in saved.columns if c.endswith(("_err2", "_cover", "_width"))]
    a = auth.sort_values(["author_id", "budget", "method", "arm"]).reset_index(drop=True)
    s = saved.sort_values(["author_id", "budget", "method", "arm"]).reset_index(drop=True)
    assert (a[["author_id", "budget", "method", "arm"]].values == s[["author_id", "budget", "method", "arm"]].values).all()
    assert np.nanmax(np.abs(a[cols].to_numpy(float) - s[cols].to_numpy(float))) <= 1e-12
    old = json.loads((OUT / "episodes/shared_shock_20261001/episode.json").read_text())
    assert info["source_rows_sha256"] == old["source_rows_sha256"] and info["moment_common_variance"] == old["moment_common_variance"]


def test_analysis_hashes_match():
    a = json.loads((OUT / "analysis.json").read_text())
    for f, h in {**a["output_sha256"], **a["episode_files_sha256"]}.items():
        assert hashlib.sha256((OUT / f).read_bytes()).hexdigest() == h, f
