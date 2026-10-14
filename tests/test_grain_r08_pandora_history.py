"""R08 observational PANDORA history comparison: row identity, prequential no-lookahead, independent delta."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_pandora_history as ph  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_history_comparison_r1"
R = json.loads((OUT / "results.json").read_text())


def test_row_identity_and_roles():
    m, X, ident = ph.load()
    assert ident["one_to_one"] and len(m) == 22464 and X.shape == (22464, 30)
    assert R["roles"]["held_scored_events"] == 1320 and R["roles"]["held_authors"] == 120
    assert R["roles"]["source_train_authors"] + R["roles"]["source_validation_authors"] == 504
    assert R["facets"][0] == "A1" and R["facets"][-1] == "O6"


def test_independent_recomputation_of_venue_delta_and_keyed_means():
    df = pd.read_csv(OUT / "held_event_scores.csv")
    assert len(df) == 1320 and df.t.min() == 25 and df.t.max() == 35
    per = df.groupby("author").mean(numeric_only=True)
    d = (per["venue_ll_combined"] - per["venue_ll_static"]).mean()
    assert abs(d - R["venue"]["combined"]["delta_vs_static"]["mean"]) < 1e-12
    assert abs(df["keyed_mse_static"].mean() - R["keyed"]["static"]["mse"]) < 1e-12


def test_prequential_prediction_uses_only_prefix():
    m, X, _ = ph.load()
    ix = np.where(m.author_index.to_numpy() == m[m.split != "source"].author_index.iloc[0])[0]
    Z = X[ix]
    pm_before = Z[:30].mean(0)
    Z2 = Z.copy(); Z2[30:] += 100.0                          # perturb the future
    assert np.array_equal(Z2[:30].mean(0), pm_before)
    assert R["novelty_split"]["known_raw_venue"]["events"] + R["novelty_split"]["novel_raw_venue"]["events"] == 1320
