"""R08 conditional information bottleneck witness: exact checks and declared endpoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_conditional_ib as ib  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/conditional_ib_r1"
R = json.loads((OUT / "results.json").read_text())


def test_exact_checks_and_endpoints():
    for w in ("W1", "W6"):
        c = R[w]["checks"]
        assert abs(c["joint_mass"] - 1) < 1e-12 and c["entropy_vs_kl_full"] < 1e-12 and c["data_processing"] and c["refinement_monotone"]
        assert c["partitions"] == 4140
        f = R[w]["frontier"]
        assert f[0]["blocks"] == 1 and abs(f[0]["I_ZF_given_C"]) < 1e-12 and f[-1]["blocks"] == 8
        assert abs(f[-1]["I_ZF_given_C"] - R[w]["I_full_code"]) < 1e-12 and R[w]["I_full_code"] < R[w]["I_oracle_latent_state"]


def test_independent_recomputation_of_full_code_information():
    PH, _ = ib.joint("W1")
    # independent: conditional MI via explicit dictionaries
    tot = 0.0
    for a in (0, 1):
        pc = PH[:, a, :].sum()
        J = PH[:, a, :] / pc
        pz, pf = J.sum(1), J.sum(0)
        tot += pc * sum(J[z, f] * np.log2(J[z, f] / (pz[z] * pf[f])) for z in range(8) for f in range(4) if J[z, f] > 0)
    assert abs(tot - R["W1"]["I_full_code"]) < 1e-12


def test_fixed_content_summary_collides_futures_and_loses_information():
    for w in ("W1", "W6"):
        s = R[w]["fixed_summaries"]["mean_content"]
        assert s["gap_to_best_same_k"] > 0.01 and s["n_collisions"] > 0
        assert max(max(x["TV_a0"], x["TV_a1"]) for x in s["future_collisions"]) > 0.1
    assert R["W6"]["fixed_summaries"]["last_behaviour"]["gap_to_best_same_k"] < 1e-12       # behaviour is the optimal 1-bit code in W6
