"""R08 natural venue OOM: role split, source-only fit, exact baseline reproduction, operator replay, support accounting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_pandora_venue_oom as vo  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1"
R = json.loads((OUT / "results.json").read_text())
ROWS = pd.read_csv(OUT / "held_event_rows.csv", float_precision="round_trip")
ROLES = json.loads((OUT / "author_roles.json").read_text())
FIT = np.load(OUT / "oom_fit_source.npz")


def _data():
    m, _X, _ = vo.ph.load()
    vo.roles_and_dictionary(m)
    return m


def test_roles_disjoint_and_counts():
    tr, va, he = map(set, (ROLES["source_train"], ROLES["source_validation"], ROLES["held"]))
    assert (len(tr), len(va), len(he)) == (403, 101, 120) and not (tr & va) and not (tr & he) and not (va & he)
    assert len(ROWS) == 1320 and set(ROWS.author) == he and ROWS.t.min() == 25 and ROWS.t.max() == 35


def test_exact_baseline_reproduction_against_packet15_rows():
    prev = pd.read_csv(ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_history_comparison_r1/held_event_scores.csv", float_precision="round_trip")
    mg = ROWS.merge(prev, on=["author", "t"])
    for n in ("global", "static", "order1", "combined"):
        assert np.abs(mg[f"ll_{n}"] - mg[f"venue_ll_{n}"]).max() <= 1e-12


def test_fit_uses_source_train_only_and_operator_replay():
    m = _data()
    cats = m.cat.to_numpy(); aidx = m.author_index.to_numpy()
    seqs = {a: cats[aidx == a] for a in set(aidx)}
    P1, P21, P3 = vo.fit_stats([seqs[a] for a in sorted(ROLES["source_train"])], 21)
    assert np.allclose(P1, FIT["P1"], atol=0) and np.allclose(P21, FIT["P21"], atol=0)
    k = R["chosen_rank"]
    f = {"b1": FIT[f"k{k}_b1"], "binf": FIT[f"k{k}_binf"], "B": FIT[f"k{k}_B"]}
    for a in ROLES["held"][:5]:
        recs, _ = vo.oom_filter(seqs[a], f)
        got = ROWS[ROWS.author == a].sort_values("t").p_oom.to_numpy()
        want = np.array([r["p"][seqs[a][r["t"]]] for r in recs])
        assert np.abs(got - want).max() < 1e-12


def test_rank_rule_and_support_accounting():
    v = R["validation"]
    assert R["chosen_rank"] == min(R["eligible_ranks"], key=lambda k: v[str(k)]["logloss_nonrefused"])
    assert all(v[str(k)]["gate_G1_pass"] == (v[str(k)]["neg_share_mean"] <= .05 and v[str(k)]["refusal_rate"] <= .01
                                              and v[str(k)]["reset_rate_scored_updates"] <= .01) for k in R["ranks_candidates"])
    H = R["held_oom"]
    assert H["coverage"] == 1 - ROWS.oom_refused.mean() and H["floored_events"] == int(ROWS.oom_floored.sum())
    assert (ROWS.ll_oom_fulldenom.notna()).all()


def test_independent_reaggregation_and_decomposition():
    H = R["held_oom"]; ok = ROWS[~ROWS.oom_refused]
    per = ok.groupby("author")[["ll_oom", "ll_static"]].mean()
    assert abs((per.ll_oom - per.ll_static).mean() - H["delta_logloss_paired"]["static"]["mean"]) < 1e-12
    for ref in ("static", "order1", "combined"):
        tot = sum(H["decomposition"][s][f"contribution_to_pooled_delta_vs_{ref}"] for s in ("outcome_top20", "outcome_OTHER"))
        assert abs(tot - (ok.ll_oom - ok[f"ll_{ref}"]).mean()) < 1e-12
