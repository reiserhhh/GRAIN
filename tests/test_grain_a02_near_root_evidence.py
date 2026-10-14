"""A02 conditional evidence repair: independent certificate checks and per-cell (no mixing) near-root re-derivation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F05 = ROOT / "results/grain_gptpro_foundation_v3/F05"
OUT = F05 / "near_root_identification_r1"
R = json.loads((OUT / "results.json").read_text())


def test_certificates_verified_independently():
    c = R["certificates"]
    assert c["pass"] and c["stationary_P_max_abs_diff_vs_recorded"] < 1e-10
    assert all(t["autocov_max_abs_diff"] < 1e-12 for t in c["transforms"])
    assert {k: (v["independent_rank"], v["recorded_rank"]) for k, v in c["ranks"].items()} == {"all": (5, 5), "fixed_CQ": (5, 5), "fixed_CQR": (4, 4)}


def test_rederivation_matches_audit_and_is_per_cell():
    assert R["rederivation_pass"] and max(R["rederivation_max_abs_diff_vs_audit"].values()) < 1e-12
    C = pd.read_csv(OUT / "near_root_vs_core_by_cell.csv")
    assert len(C) == R["cells"] == 90 and not C.duplicated(["n", "phase", "method"]).any()


def test_one_cell_recomputed_from_application_rows():
    st = pd.read_csv(F05 / "application_full_r1/state_parameter_metrics.csv")
    g = st[(st.n == 32) & (st.phase == "prefix") & (st.method == "personal_MAP")]
    near = g[g.world == "near_root"].state_squared_error.mean(); core = g[g.world == "core"].state_squared_error.mean()
    C = pd.read_csv(OUT / "near_root_vs_core_by_cell.csv")
    row = C[(C.n == 32) & (C.phase == "prefix") & (C.method == "personal_MAP")].iloc[0]
    assert abs(row.state_squared_error_near - near) < 1e-12 and abs(row.state_squared_error_core - core) < 1e-12
