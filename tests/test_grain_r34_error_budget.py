"""Focused tests for packet20 A (R34 cached error-budget reconciliation)."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F16/error_budget_reconciliation_r1"
PROTOCOL = ROOT / "docs/GRAIN_R34_ERROR_BUDGET_PROTOCOL_V1.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(OUT / "per_person_components.csv", low_memory=False)


def test_protocol_and_input_bindings_hold():
    assert sha(PROTOCOL) == "5e55f6145e36d9301157e4e042ee35c49d43881ba770995ec807463eb188e827"
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == h
    res = json.loads((OUT / "results.json").read_text())
    assert res["inputs_verified"] == 244 and res["neural_calls"] == 0


def test_mixture_identity_formula_on_synthetic_components():
    rng = np.random.default_rng(0)
    m, v = rng.normal(size=200), rng.uniform(.1, 1, size=200)
    # law of total variance for an equal-weight mixture of Gaussians, checked by brute-force sampling-free moments
    second = np.mean(v + m ** 2)
    assert abs((second - m.mean() ** 2) - (v.mean() + np.var(m, ddof=0))) < 1e-12


def test_every_readout_within_tolerance(df):
    res = json.loads((OUT / "results.json").read_text())
    assert res["readouts"] == len(df) == 45600
    assert all(v == 0 for v in res["readouts_over_tolerance"].values())
    for c in ("err_sum", "err_recon_components", "err_recon_score", "err_recon_total", "err_interval"):
        assert df[c].dropna().max() <= 1e-12


def test_combined_source_reader_is_never_split(df):
    text = df[df.route.str.startswith("text")]
    assert "c_source" not in text.columns or text["c_source"].isna().all()
    assert (text.filter(like="c_source_and_reader_combined").notna().any(axis=None)
            or text.filter(like="c_source_reader_combined").notna().any(axis=None))


def test_t2_and_t3_components_are_labelled_differently(df):
    t3 = df[(df.route == "numeric_t3")]
    assert t3["c_future_innovation"].notna().all() and t3["c_state_carried"].notna().all()
    t2 = df[(df.route == "numeric_t2")]
    assert "c_future_innovation" not in t2.columns or t2["c_future_innovation"].isna().all()


def test_cold_replays_reproduce_identity():
    rp = json.loads((OUT / "replay_checks.json").read_text())
    assert rp["text_t1"]["components_captured"] == 200 and rp["text_t1"]["err_vs_production"] <= 1e-12
    assert rp["text_t1"]["err_vs_archived"] <= 1e-12
    for t in ("T2", "T3"):
        assert rp["text_dynamic"]["targets"][t]["err_vs_production"] <= 1e-12
        assert rp["text_dynamic"]["targets"][t]["err_vs_archived"] <= 1e-12
    z = np.load(OUT / "replay_components.npz")
    assert abs(z["t1_within"].mean() + z["t1_means"].var() - (rp["text_t1"]["E_within"] + rp["text_t1"]["Var_means_ddof0"])) < 1e-12


def test_aliases_equal_and_superseded_reported():
    a = json.loads((OUT / "alias_checks.json").read_text())
    for k, v in a.items():
        if k.startswith("dyn_gauss_superseded"):
            assert "max_abs_difference_REPORTED_NOT_EXPECTED_EQUAL" in v
        elif k.startswith("public_contract"):
            assert v["matched_within_tol"] == v["records"]
        else:
            assert v["max_abs"] is not None and v["max_abs"] <= 1e-12


def test_unsupported_routes_are_named_not_filled(df):
    un = df[df.note.astype(str).str.contains("UNSUPPORTED")]
    assert set(un.route) == {"numeric_t1_plugin_nosource", "text_t1_old_reader"}
    g = un[un.route == "text_t1_old_reader"]
    assert set(g.variant) == {"naive_gaussian", "eligible_continuous"}
    assert g.filter(like="c_").isna().all(axis=None)
