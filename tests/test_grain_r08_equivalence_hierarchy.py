"""R08 amendment companion: symmetry is sufficient not necessary; predictive equivalence is coarser than lumpability."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_equivalence_hierarchy as h  # noqa: E402
import run_grain_r08_text_behaviour_quotient as r1  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/equivalence_hierarchy_r1"
RES = json.loads((OUT / "results.json").read_text())


def test_four_state_lumpable_without_symmetry():
    w = h.four_state()
    assert r1.lumpability(w, "raw", (0,), [[0, 1], [2, 3]])["lumpable"]
    assert h.transition_automorphism_error(w, {0: 1, 1: 0, 2: 3, 3: 2}) == h.q(.3)
    assert r1.tv(r1.future_law(w, "raw", 0, (0,)), r1.future_law(w, "raw", 2, (0,))) > 0      # blocks keep a non-constant distinction
    assert RES["four_state"]["predictive_classes"] == [["x0r0", "x0r1"], ["x1r0", "x1r1"]]


def test_predictive_equivalence_without_lumpable_partition():
    t = h.trace_not_bisim()
    for k in range(1, 6):
        assert r1.tv(r1.future_law(t, "raw", 0, (0,) * k), r1.future_law(t, "raw", 1, (0,) * k)) == 0
    classes, _ = r1.krylov_classes_exact(t, "raw", (0,))
    assert [0, 1] in classes and not r1.lumpability(t, "raw", (0,), classes)["lumpable"]
    assert [0] in r1.bisimulation(t, "raw", (0,))


def test_bayes_error_identity():
    b = RES["bayes_error_identity"]
    assert b["match"] and abs(b["direct_bayes_error_single_observation"] - 0.35) < 1e-12


def test_r1_results_untouched_and_failed_attempt_kept():
    assert (ROOT / "results/grain_gptpro_foundation_v3/F15/equivalence_hierarchy_r1_failed_attempt1/FAILURE.md").exists()
    rep = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F15/text_behaviour_quotient_r1/checks.json").read_text())
    assert rep["status"] == "PASS"
