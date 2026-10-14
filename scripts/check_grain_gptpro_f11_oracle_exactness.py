#!/usr/bin/env python3
"""Large-sample check that the F11 oracle interval is exact (generator = model).

Simulates 200,000 fresh authors per world with the same generator form as
`run_grain_gptpro_f11_mtmm.py` (separate stream SeedSequence([11, 99, world]))
and scores them with the oracle parameters. If coverage is .95 here, the
3-seed oracle under-coverage is sampling variation, not model mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f11_mtmm import designs, predict_traits

N = 200_000


def main():
    """Print and save coverage and z-variance per world and panel."""
    sigma = np.full((3, 3), 0.3)
    np.fill_diagonal(sigma, 1.0)
    out = {}
    for wi, world in enumerate(("base", "shared_event", "conditional_loading")):
        conditional, event = world == "conditional_loading", 0.0 if world == "base" else 0.25
        f, m, c = designs(conditional)
        rng = np.random.default_rng(np.random.SeedSequence([11, 99, wi]))
        trait = rng.multivariate_normal(np.zeros(3), sigma, size=N)
        y = trait @ f.T + rng.normal(scale=0.5, size=(N, 3)) @ m.T + rng.normal(scale=np.sqrt(event), size=(N, 2)) @ c.T + rng.normal(scale=0.5, size=(N, 18))
        oracle = {"sigma": sigma, "method": 0.25, "event": event, "error": 0.25, "mean": np.zeros(18)}
        for panel, cells in (("all_two_occasions", None), ("same_event_three_readers", np.arange(0, 18, 2)), ("one_reader_two_occasions", np.array([0, 1, 6, 7, 12, 13]))):
            pred, var = predict_traits(oracle, y, conditional, cells)
            z = (pred - trait) / np.sqrt(np.diag(var))
            out[f"{world}:{panel}"] = {"coverage": float((np.abs(z) <= 1.96).mean()), "z_var": z.var(0).round(4).tolist()}
    (ROOT / "results/grain_gptpro_foundation_v3/F11/source_uncertainty_r1/oracle_exactness_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
