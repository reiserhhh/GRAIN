#!/usr/bin/env python3
"""dif_contrast reproduces the stored F12 DIF table and detects condition DIF."""

import csv
from pathlib import Path

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f12_dif import dif_contrast

RUN = Path("results/grain_gptpro_foundation_v3/F12/transport_full_r1")
TABLE = Path("results/grain_gptpro_foundation_v3/F12/dif_table_r1/dif_table.csv")


@pytest.mark.skipif(not TABLE.exists(), reason="archived DIF table not present")
def test_reproduces_stored_dif_table_row():
    seed, world, method = 20260916, "conditional", "global"
    z = np.load(RUN / f"observations_{seed}_application.npz")["conditions"]
    theta = np.load(RUN / f"truth_{seed}_application.npz")["theta"]
    d = np.load(RUN / f"scores_{seed}_{world}_application_{method}.npz")
    out = dif_contrast(d["score"], d["support"], theta[:, [0]] + theta[:, [1]] * z[None, :], z)
    row = next(r for r in csv.DictReader(TABLE.open()) if r["seed"] == str(seed) and r["world"] == world and r["method"] == method)
    assert abs(out["dif_range"] - float(row["dif_range"])) < 1e-12
    assert abs(out["dif_range_after_renorm"] - float(row["dif_range_after_oracle_renorm"])) < 1e-12


def test_global_shift_removed_by_renorm_but_condition_shift_is_not():
    rng = np.random.default_rng(0)
    z = np.tile([-1.0, 1.0], 16)
    ref = rng.normal(size=(50, 1)) + 0.5 * z
    global_scores = 2 * ref + 1
    cond_scores = ref + 0.8 * (z > 0)
    ok = np.ones_like(ref, dtype=bool)
    assert dif_contrast(global_scores, ok, ref, z)["dif_range_after_renorm"] < 1e-9
    assert dif_contrast(cond_scores, ok, ref, z)["dif_range_after_renorm"] > 0.3


def test_unsupported_readings_excluded_not_imputed():
    z = np.array([-1.0, 1.0])
    out = dif_contrast(np.array([[np.nan, 1.0]]), np.array([[False, True]]), np.array([[0.0, 1.0]]), z)
    assert out == {"status": "NO_SUPPORTED_SCORES"}
