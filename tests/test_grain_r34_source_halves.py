"""R34 source-half reconstruction: disjoint source-only halves, real refits, role firewall, cold replay, T2/T3 shared fit, no pseudo-bootstrap."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from grain_research.gptpro_foundation_v3 import f05_recovery as f05  # noqa: E402
from grain_research.gptpro_foundation_v3 import f16_t1_source as t1  # noqa: E402
import run_grain_r34_source_halves as rh  # noqa: E402

V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "F16/source_half_reconstruction_r1"
R = json.loads((OUT / "results.json").read_text())
SPLITS = json.loads((OUT / "splits.json").read_text())
P = pd.read_csv(OUT / "per_author_predictions.csv")


def _src_ids(path):
    return sorted({r["author_id"] for r in rh.jl(path) if r["role"] == "source"})


def test_disjoint_source_only_halves_and_counts():
    for key, sps in SPLITS.items():
        fam, seed = key.split("|")
        path = V3 / (f"F04/source_full_r1/L_{seed}/source.jsonl" if fam == "T1" else f"F05/source_full_r2/core_{seed}/source.jsonl")
        ids = set(_src_ids(path))
        assert len(sps) == 20
        for sp in sps:
            assert not set(sp["A"]) & set(sp["B"]) and set(sp["A"]) | set(sp["B"]) == ids and len(sp["A"]) == 150
    assert R["fit_counts"] == {"T1_half": 120, "dyn_half": 120, "T1_full": 3, "dyn_full": 3} and R["neural_calls"] == 0


def test_fits_respond_to_source_changes():
    rows = rh.jl(V3 / "F04/source_full_r1/L_20260915/source.jsonl")
    by = {}
    for r in rows:
        by.setdefault(r["author_id"], []).append(r)
    ids = sorted(by); st = [t1.sufficient(by[a]) for a in ids]; S = tuple(np.array([s[i] for s in st]) for i in range(3))
    sp = SPLITS["T1|20260915"][0]
    fa = t1.fit_source(S, np.array([ids.index(a) for a in sp["A"]])); fb = t1.fit_source(S, np.array([ids.index(a) for a in sp["B"]]))
    assert not np.allclose(fa["mu"], fb["mu"])
    S2 = (S[0], S[1].copy(), S[2].copy()); S2[1][0, 0] += 5.0; S2[2][0, 0] += 50.0
    assert not np.allclose(t1.fit_source(S2, np.arange(300))["mu"], t1.fit_source(S, np.arange(300))["mu"])


def test_application_rows_cannot_enter_source_fits():
    rows = rh.jl(V3 / "F05/source_full_r2/core_20260915/source.jsonl")[:480]
    app = rh.jl(V3 / "F05/source_full_r2/core_20260915/validation.jsonl")[:5]
    with pytest.raises(ValueError):
        f05.fit_source_dynamics(rows + app)


def test_cold_replay_of_saved_predictions():
    seed = 20260916; sp = 3
    fit = json.loads((OUT / "dyn_half_fits" / f"{seed}_split{sp:02d}.json").read_text())["A"]
    val = rh.jl(V3 / f"F05/source_full_r2/core_{seed}/validation.jsonl")
    author = sorted({r["author_id"] for r in val})[7]
    prefix = sorted([r for r in val if r["author_id"] == author and r["t"] < 48], key=lambda r: r["t"])
    r = f05.score_prefix(prefix, fit, future_conditions=rh.PLAN)
    m3, v3 = rh.t3_mean_of_four(r, fit["log_sigma"])
    row2 = P[(P.seed == seed) & (P.split == sp) & (P.author == author) & (P.target == "T2")].iloc[0]
    row3 = P[(P.seed == seed) & (P.split == sp) & (P.author == author) & (P.target == "T3")].iloc[0]
    assert abs(row2.pred_A - r["state_mean"]) < 1e-9 and abs(row3.pred_A - m3) < 1e-9 and abs(row3.var_A - v3) < 1e-9


def test_T2_T3_share_fit_but_differ_and_no_pseudo_bootstrap():
    fit = json.loads((OUT / "dyn_half_fits" / "20260915_split00.json").read_text())
    for h in ("A", "B"):
        assert len(fit[h]["prior"]["mu"]) == 3 and np.array(fit[h]["prior"]["Sigma"]).shape == (3, 3)   # one point fit, not an ensemble
    t2 = P[(P.target == "T2") & (P.seed == 20260915) & (P.split == 0)].sort_values("author")
    t3 = P[(P.target == "T3") & (P.seed == 20260915) & (P.split == 0)].sort_values("author")
    assert list(t2.author) == list(t3.author) and not np.allclose(t2.pred_A.to_numpy(), t3.pred_A.to_numpy())
    assert R["model_label"].startswith("POINT_FIT_PLUGIN")
