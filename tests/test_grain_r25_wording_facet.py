"""R25 designed wording facet: pair audit, df/estimability, SS identity, independent balanced-formula recomputation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F12/designed_wording_facet_r1"
R = json.loads((OUT / "results.json").read_text())
AN = pd.read_csv(OUT / "anova_rows.csv")
WD = pd.read_csv(OUT / "wording_contrast_rows.csv")
AUD = pd.read_csv(OUT / "pair_audit.csv")
ROWS = [json.loads(l) for l in (ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_commitment_reader_r1/readings.jsonl").read_text().splitlines()]


def test_pair_audit_complete_no_drops():
    assert len(AUD) == 48 and R["pair_audit"]["dropped"] == 0 and R["pair_audit"]["exceptions"] == []
    assert AUD[["same_frame_prefix", "owner_wrapper_consistent", "proposition_in_both", "frame_text_found"]].all().all()
    assert R["reader_calls"] == 0


def test_df_and_estimability():
    for (sp, q, resp), g in AN.groupby(["split", "query", "response"]):
        assert g.df.sum() == (63 if sp == "source" else 31) and g.rank_ok.all()
        assert int(g[g.term == "wording_x_context_lumped"].df.iloc[0]) == (28 if sp == "source" else 12)
    assert R["max_abs_sum_check"] < 1e-12 and R["max_abs_orthogonality_gap"] < 1e-10


def test_wording_ss_by_balanced_formula_independent():
    for sp, cores in (("source", {1, 2}), ("test", {3})):
        for q in ("no_commitment", "truth_diagnostic", "uncertainty"):
            d = {}
            for r in ROWS:
                if r["role"] == q and r["core"] in cores:
                    d.setdefault((r["core"], r["form"], r["owner"], r["frame"]), {})[r["wording"]] = r["p_entailment"]
            diffs = np.array([v["original"] - v["paraphrase"] for v in d.values()])
            n = 2 * len(diffs)
            ss_w = n * diffs.mean() ** 2 / 4
            ss_within = (diffs ** 2).sum() / 2
            g = AN[(AN.split == sp) & (AN["query"] == q) & (AN.response == "ent")].set_index("term")
            assert abs(ss_w - g.loc["wording", "ss"]) < 1e-12
            assert abs(ss_within - g.loc[["wording", "wording_x_form", "wording_x_context_lumped"], "ss"].sum()) < 1e-12


def test_raw_contrast_rows_and_merged_sensitivity_labels():
    assert len(WD) == 48 and WD.filter(like="d_").shape[1] == 12
    se = pd.DataFrame(R["merged_functional_sensitivity"])
    assert len(se) == 18 and not se[se.core == 3].in_sample_for_channel.any()
