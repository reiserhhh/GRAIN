"""R08 controlled PSR witness: source-only fitting, support refusal, gate classification, independent estimator check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_controlled_psr_witness as w  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/controlled_psr_witness_r1"
R = json.loads((OUT / "results.json").read_text())
ARMS = {a["arm"]: a for a in R["arms"]}


def test_budget_and_unsupported_primitive_refusal():
    assert R["total_generated_steps"] <= 1_000_000 and R["Q3_unsupported_refusal_ok"]
    u = ARMS["W6_V_unsup"]
    assert u["supported_actions"] == [0] and u["refused_next_step"]["psr"] >= u["a1_steps_in_application"]
    assert u["refused_next_step"]["static"] == u["a1_steps_in_application"] and u["refused_next_step"]["oracle"] == 0


def test_gate_passing_arms_psr_beats_static_order1_pca_but_not_oracle():
    passing = [a for a in R["arms"] if a["support_gate_pass"] and a["src_policy"] == "random"]
    assert len(passing) == 3
    for a in passing:
        ns = a["next_step"]
        for m in ("static", "order1", "pca"):
            assert ns[m]["delta_ci_vs_psr"][0] > 0, (a["arm"], m)
        assert ns["oracle"]["delta_ci_vs_psr"][1] < 0


def test_failed_gates_are_classified_not_hidden():
    failing = [a["arm"] for a in R["arms"] if not a["support_gate_pass"]]
    assert set(failing) == {"W1_20261101", "W6_20261101", "W6_20261102"}
    for n in failing:
        assert max(ARMS[n]["source_reconstruction_TV_by_action_seq"].values()) > 0.02


def test_novel_combination_transport():
    c = ARMS["W6_V_comb"]
    assert c["novel_combination_steps"] > 1000
    gap_novel = c["novel_combination_logloss"]["psr"] - c["novel_combination_logloss"]["oracle"]
    gap_sup = c["supported_combination_logloss"]["psr"] - c["supported_combination_logloss"]["oracle"]
    assert gap_novel < 0.02 and abs(gap_novel - gap_sup) < 0.01


def test_estimator_matches_independent_counting_and_is_source_only():
    K, init = w.construct_kernel("W1")
    rng = np.random.default_rng(7)
    A, O = w.generate(K, init, 400, "random", rng)
    P = w.fit_spectral_stats(A, O)
    Z = 4 * A + O
    # independent count of P_TH for one history/test cell
    h, t = 8 * 1 + 5, 8 + 8 * 2 + 6
    num = den = 0
    for i in range(L := w.L):
        pass
    for s in range(Z.shape[0]):
        for i in range(w.L - 4):
            if (A[s, i], A[s, i + 1], A[s, i + 2], A[s, i + 3]) == (0, 1, 0, 1):
                den += 1
                if (Z[s, i], Z[s, i + 1], Z[s, i + 2], Z[s, i + 3]) == (1, 5, 2, 6):
                    num += 1
    assert abs(P["TH"][t, h] - num / den) < 1e-12
    import inspect
    src = inspect.getsource(w.run_arm)
    k_block = src[src.index("for k in range(1, 9)"):src.index("k_best = ")]
    assert "oracle" not in k_block and "K[" not in k_block          # rank selection never touches the kernel/oracle


def test_posthoc_collision_example_psr_separates_static_merges():
    c = json.loads((OUT / "collision_examples_posthoc.json").read_text())
    ex = c["W1_20261102"]["examples"][0]
    assert ex["static_TV"] == 0 and ex["psr_TV"] > 0.05 and ex["oracle_TV"] > 0.05
