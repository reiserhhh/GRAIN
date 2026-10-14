"""R15-M1 speaker identifiability: falsifiers G1-G4 and independent recomputations."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r15_speaker_identifiability as m  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/speaker_identifiability_r1"
R = json.loads((OUT / "results.json").read_text())
RT = {(r["design"], r["observation"]): r for r in R["rank_table"]}


def test_jacobian_fd_and_independent_probability():
    assert R["fd_max_abs_error"] < 1e-6
    th, c = m.REF, m.ctx("A1")
    b = 1 / (1 + math.exp(-th[0]))
    U = []
    for i in range(3):
        ui = b * math.log(m.L0T[i]) + (1 - b) * math.log(1 - m.L0T[i])
        k = th[2] * c["a"] + th[3] * c["s"] * (1 - b) + th[4] + th[5] * c["f"]
        U.append(ui - k * (m.L0T[i] - .5) + th[6] * c["p"] * m.M[i] + th[7] * math.log(m.STYLE[c["m"]][i]) - m.COST[i])
    ex = [math.exp(th[1] * u) for u in U]
    manual = [e / sum(ex) for e in ex]
    assert np.allclose(manual, m.probs(th, c)[0], atol=1e-14)


def test_G3_passive_text_rank_two_and_G2_all_probes_full_rank():
    assert RT[("passive", "text_only")]["max_rank"] == 2 and RT[("passive", "joint")]["max_rank"] == 3
    assert RT[("C0+all", "joint")]["points_at_max_rank"] == 25 and RT[("C0+all", "joint")]["max_rank"] == 8
    assert RT[("C0+E", "text_only")]["max_rank"] == 4          # evidence moves belief and the (1-b) term


def test_G1_commitment_aliasing_exact_and_only_audience_separates():
    e1 = R["E1_commitment_aliasing"]
    assert e1["TV_passive_text"] < 1e-12 and e1["TV_passive_joint"] < 1e-12
    sep = {k for k, v in e1["TV_by_context_joint"].items() if v > 1e-9}
    assert sep == {"A0", "A1"}


def test_E2_equivalence_needs_negative_weights_and_B_separates():
    e2 = R["E2_epistemic_vs_preference"]
    assert e2["TV_passive_text"] < 1e-12 and not e2["feasible_nonnegative"] and e2["TV_passive_joint"] > 0.1


def test_E3_single_observation_mixture_equivalence_broken_by_repeated_or_probed_observations():
    e3 = R["E3_mixture_vs_homogeneous"]
    assert e3["feasible_nonnegative"] and e3["TV_single_observation_passive"] < 1e-12
    assert e3["TV_pair_C0_C0"] > 0 and e3["TV_pair_A0_A1"] > e3["TV_pair_C0_C0"]


def test_functional_identification_pattern():
    f = pd.DataFrame(R["functionals"])
    get = lambda d, o, fn: bool(f[(f.design == d) & (f.observation == o) & (f.functional == fn)].identified.iloc[0])  # noqa: E731
    assert get("passive", "text_only", "P(hedge|C0)") and not get("passive", "text_only", "w_face")
    assert get("passive", "joint", "P(B=1|C0)") and not get("passive", "text_only", "P(B=1|C0)")
    assert get("C0+A", "joint", "P(hedge|a=0)") and not get("C0+A", "joint", "w_face")
    assert all(get("C0+all", "joint", fn) for fn in ("w_face", "w_opt", "w_lc", "kappa0_at_C0"))


def test_G4_recorded_probe_design_not_better_at_this_n():
    rc = pd.read_csv(OUT / "recovery.csv")
    rmse = {d: {n: float(np.sqrt(((g[f"est_{n}"] - g[f"true_{n}"]) ** 2).mean())) for n in ("w_face", "w_opt", "beta0", "w_acc")} for d, g in rc.groupby("design")}
    assert rmse["F_joint"]["w_face"] >= rmse["P_joint"]["w_face"]          # G4 fired: reported, not redesigned
    assert rmse["P_joint"]["beta0"] < rmse["P_text"]["beta0"]               # behaviour identifies belief
    ss = json.loads((OUT / "sample_size_posthoc.json").read_text())
    assert ss["C0+all|joint"]["n_per_context_for_sd_0.5"]["w_face"] > 10
