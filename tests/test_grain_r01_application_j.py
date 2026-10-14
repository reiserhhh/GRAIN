"""R01 application-J diagnostic: accounting, frozen bindings, pairing, decomposition identity, independent tangent recomputation."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_grain_r01_application_j as aj  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/P05/application_j_diagnostic_r1"
R = json.loads((OUT / "results.json").read_text())
ST = json.loads((OUT / "neural_status.json").read_text())
ACC = [json.loads(x) for x in (OUT / "accounting.jsonl").read_text().splitlines()]
C = pd.read_csv(OUT / "components_by_change.csv")
X = pd.read_csv(OUT / "decomposition_vs_donor.csv")
IDS = json.loads((ROOT / "docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json").read_text())


def test_accounting_counts_and_cap():
    started = [a for a in ACC if a["status"] == "STARTED"]
    assert sum(a["passes"] for a in started) == 312 <= 320 and not any(a["status"] == "FAILED" for a in ACC)
    assert Counter((a["step"], a["passes"]) for a in started) == {("replay", 1): 16, ("app_jacobian", 2): 144, ("fd", 1): 8}
    assert ST["status"] == "COMPLETE" and ST["passes_attempted"] == 312 and ST["replay_fails"] == []
    assert ST["grad_pass_score_replay_max_abs"] <= 1e-5 and all(f["pass"] and f["rel_err"] <= .10 for f in ST["fd_check"])
    assert ST["planned"] == {"replay_forwards": 16, "gradient_forwards": 144, "gradient_backwards": 144, "fd_forwards": 8, "total_passes": 312, "cap": 320}


def test_frozen_bindings_and_protocol():
    aj.verify()
    assert R["protocol_sha256"] == aj.PROTOCOL_SHA


def test_pairing_and_coverage():
    assert len(C) == 72 * 2 * 9 and set(C.pair) == set(IDS["application_pairs"])
    assert C.groupby(["pair", "query"]).size().eq(9).all()
    jfiles = [a for a in ACC if a["status"] == "OK" and a["item"].startswith("app_jacobian")]
    assert len(jfiles) == 144 and all(abs(a["score"] - a["cached_low"]) <= 1e-5 for a in jfiles)


def test_decomposition_identity_and_share_sum():
    assert (X.E - X["T"] - X.dR).abs().max() <= 1e-12
    for q in ("extraversion", "neuroticism"):
        o = R["per_query"][q]["E24"]
        h = X[(X["query"] == q) & (X.change == "E24")]
        e2, t2, r2, xx = (h.E ** 2).mean(), (h["T"] ** 2).mean(), (h.dR ** 2).mean(), 2 * (h["T"] * h.dR).mean()
        assert abs(t2 / e2 - o["share_T"]) < 1e-10 and abs(t2 + r2 + xx - e2) < 1e-15


def test_independent_tangent_recomputation_from_saved_gradients():
    arr = np.load(ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2/action_arrays.npz")
    order = json.loads((ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2/pair_order.json").read_text())
    prep = np.load(ROOT / "results/grain_gptpro_foundation_v3/P05/native_action_r1/private_prepared.npz")
    off = {"extraversion": slice(0, 7552), "neuroticism": slice(7552, 15232)}
    for p in IDS["application_pairs"][:6]:
        i = order.index(p)
        for q, sl in off.items():
            J = np.load(OUT / "private_app_jacobians" / f"{i:03d}_{q}.npy").ravel()
            row = C[(C.pair == p) & (C["query"] == q)].set_index("change")
            assert abs(J @ arr["dh"][i, sl] - row.loc["donor", "L"]) < 1e-12
            assert abs(J @ prep["E24"][i, sl] - row.loc["E24", "L"]) < 1e-12
