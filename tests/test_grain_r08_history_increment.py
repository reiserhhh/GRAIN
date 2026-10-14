"""Focused tests for packet22 B (R08 incremental history information). Pre-run tests + post-run test_output_ tests."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r08_history_increment as h  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F14/history_increment_r1"
F14 = ROOT / "results/grain_gptpro_foundation_v3/F14"
PROTOCOL = ROOT / "docs/GRAIN_R08_HISTORY_INCREMENT_PROTOCOL_V1.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_ridge_normal_equations_are_the_declared_objective():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(200, 60)), rng.normal(size=(200, 30))
    B = h.ridge_fixed(X, Y)
    assert np.abs(h.objective_grad(X, Y, B)).max() < 1e-12
    obj = lambda B_: np.sum((Y - X @ B_) ** 2) / 200 + np.sum(B_ ** 2)
    assert obj(B) <= obj(B + 1e-3 * rng.normal(size=B.shape))


def test_features_use_prefix_only_and_future_target_changes_nothing_earlier():
    rng = np.random.default_rng(1)
    Zs = rng.normal(size=(36, 30))
    pop = {"phi": rng.normal(size=30) * .1, "V5": np.linalg.qr(rng.normal(size=(30, 5)))[0].T, "cmean": np.zeros(30)}
    Bl, Bh = rng.normal(size=(30, 30)) * .01, rng.normal(size=(60, 30)) * .01
    before = [h.keyed_predict(Zs[:t], pop, Bl, Bh, .5, .5) for t in range(25, 36)]
    Z2 = Zs.copy(); Z2[30:] += 5.0                       # change target at t=30 and later
    after = [h.keyed_predict(Z2[:t], pop, Bl, Bh, .5, .5) for t in range(25, 36)]
    for i, t in enumerate(range(25, 36)):
        same = all(np.array_equal(before[i][k], after[i][k]) for k in before[i])
        assert same == (t <= 30)                         # prediction at t uses only events < t


def test_oom_filter_prediction_at_t_does_not_see_y_t():
    f = h.load_oom(F14 / "pandora_venue_oom_r1/oom_fit_source.npz")
    seq = np.random.default_rng(2).integers(0, h.C, 36)
    a = h.oom_records(seq, f)
    s2 = seq.copy(); s2[30] = (s2[30] + 1) % h.C
    b = h.oom_records(s2, f)
    for t in range(25, 31):
        assert np.array_equal(a[t]["p"], b[t]["p"])
    assert not np.array_equal(a[31]["p"], b[31]["p"])


def test_permutation_sees_prefix_only_and_keeps_mean():
    pre = np.random.default_rng(3).normal(size=(27, 30))
    pp = h.permuted_prefix(pre, "held", 7, 27)
    assert sorted(map(tuple, pp)) == sorted(map(tuple, pre)) and pp.shape == pre.shape
    assert np.array_equal(pp, h.permuted_prefix(pre, "held", 7, 27))          # deterministic
    Bh = np.random.default_rng(4).normal(size=(60, 30)) * .01
    m = pre.mean(0)
    assert np.allclose(h.hist_readout(pp, Bh, 0.0, mean_override=m), m)       # w=0 -> exactly the prefix mean


def test_mix_nested_fallback():
    pB = np.full(h.C, 1 / h.C); pO = np.eye(h.C)[0] * .9 + .1 / h.C
    assert np.array_equal(h.mix(pB, None, .5), pB) and np.array_equal(h.mix(pB, pO, 0.0), pB)
    assert np.isclose(h.mix(pB, pO, .25).sum(), 1)


# ------------------------------------------------------------------ outputs
def test_output_protocol_gates_roles():
    assert sha(PROTOCOL) == json.loads((OUT / "results.json").read_text())["protocol_sha256"]
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == hh
    res = json.loads((OUT / "results.json").read_text())
    g = res["replay_gates"]
    assert g["rows"] == [1320, 1320, 1320] and max(v for k, v in g.items() if k != "rows") <= 1e-12
    assert res["roles"] == {"source_train": 403, "source_validation": 101, "held": 120} and res["neural_calls"] == 0
    assert res["cold_replay_max_abs"] <= 1e-12


def test_output_selection_rules_and_redundancy_flag():
    res = json.loads((OUT / "results.json").read_text())
    v = {float(k): x for k, x in res["B1_venue"]["validation_mean_logloss_by_w"].items()}
    assert res["B1_venue"]["w_selected"] == min(w for w, x in v.items() if x == min(v.values()))
    for name in ("coupled_last", "coupled_hist"):
        s = res["B2_keyed"]["selection"][name]; vv = {float(k): x for k, x in s["validation_mse_by_w"].items()}
        assert s["w_selected"] == min(w for w, x in vv.items() if x == min(vv.values()))
    pc = res["B2_keyed"]["permutation_control"]
    assert pc["redundant_because_w_hist_is_0"] == (res["B2_keyed"]["selection"]["coupled_hist"]["w_selected"] == 0.0)


def test_output_private_files_and_pairing():
    for f in OUT.glob("*.csv"):
        if f.name.startswith("._"):          # macOS AppleDouble metadata on this volume, not an output
            continue
        assert f.name.startswith("private_")
    H = pd.read_csv(OUT / "private_held_keyed_event_losses.csv")
    assert len(H) == 1320 and H.groupby("author").size().eq(11).all()
