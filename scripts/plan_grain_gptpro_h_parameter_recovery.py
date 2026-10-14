#!/usr/bin/env python3
"""Parameter-recovery simulation for human-stage sample planning (P03 p.32).

Replaces conventional power formulas in the H contracts with recovery curves
under declared simulation inputs (taken from the project's worlds; all inputs
must be replaced by pilot estimates before real planning):
- H01: reliability of a person's condition slope (split-half over days)
  versus reliability of a single-text score, by number of days.
- H03: recovery of the functional distance between two language environments
  (sup over contexts of |response-law difference|) by persons per language,
  with equal and unequal true laws.
- H04: G-coefficient of openness-level profiles by repeats per level.
No human data; synthetic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(np.random.SeedSequence([20260926, 17, 3]))


def h01(days_grid=(2, 4, 8, 14, 30, 60), persons=200, per_day=4, sims=100):
    """Split-half reliability of slope b vs single-text reliability (F04 L-world inputs)."""
    out = []
    for days in days_grid:
        slope_r, single_r = [], []
        for _ in range(sims):
            a = RNG.normal(0, 1, persons)
            b = RNG.normal(0.5, 0.5, persons)
            z = np.tile([-1.0, 1.0], per_day // 2)
            ys = a[:, None, None] + b[:, None, None] * z[None, None, :] + RNG.normal(0, 0.5, (persons, days, per_day))
            half = days // 2 if days >= 2 else 1
            def slope(y):
                return ((y * z).mean(axis=(1, 2)) / (z**2).mean())
            s1, s2 = slope(ys[:, :half]), slope(ys[:, half:])
            slope_r.append(np.corrcoef(s1, s2)[0, 1])
            single_r.append(np.corrcoef(ys[:, 0, 0], ys[:, -1, 0])[0, 1])
        out.append({"days": days, "readings_per_day": per_day, "slope_split_half_r": float(np.mean(slope_r)),
                    "slope_spearman_brown": float(np.mean([2 * r / (1 + r) for r in slope_r])), "single_text_retest_r": float(np.mean(single_r))})
    return out


def h03(persons_grid=(10, 20, 50, 100, 200), contexts=6, sims=200, readings=4):
    """Recovery of sup-context distance between two language response laws."""
    out = []
    for delta in (0.0, 0.3):
        for n in persons_grid:
            est = []
            for _ in range(sims):
                c = np.linspace(-1, 1, contexts)
                law1 = 0.5 * c
                law2 = 0.5 * c + delta * (c > 0.5)
                y1 = law1[None, :] + RNG.normal(0, 1, (n, 1)) + RNG.normal(0, 0.5 / np.sqrt(readings), (n, contexts))
                y2 = law2[None, :] + RNG.normal(0, 1, (n, 1)) + RNG.normal(0, 0.5 / np.sqrt(readings), (n, contexts))
                d = (y2 - y2.mean(1, keepdims=True)).mean(0) - (y1 - y1.mean(1, keepdims=True)).mean(0)
                est.append(np.abs(d - d.mean()).max())
            est = np.array(est)
            out.append({"true_shift": delta, "persons_per_language": n, "mean_sup_distance": float(est.mean()), "q95_sup_distance": float(np.quantile(est, 0.95))})
    return out


def h04(repeats_grid=(1, 2, 4, 8), persons=200, levels=5, sims=100):
    """G-coefficient for person x openness-level profiles (var comps from F13 scale)."""
    out = []
    for rep in repeats_grid:
        g = []
        for _ in range(sims):
            person = RNG.normal(0, 1, persons)
            pxl = RNG.normal(0, 0.5, (persons, levels))
            y = person[:, None, None] + pxl[:, :, None] + RNG.normal(0, 1.0, (persons, levels, rep))
            m = y.mean(2)
            vp = np.var(m.mean(1), ddof=1) - np.mean(np.var(m, axis=1, ddof=1)) / levels
            vres = np.mean(np.var(m, axis=1, ddof=1))
            g.append(max(vp, 0) / (max(vp, 0) + vres / levels))
        out.append({"repeats_per_level": rep, "levels": levels, "G_person_over_levels": float(np.mean(g))})
    return out


def main():
    """Write recovery_curves.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    res = {"H01_slope_vs_single_reliability": h01(), "H03_functional_distance_recovery": h03(), "H04_G_coefficient": h04(),
           "inputs": {"H01": "a~N(0,1), b~N(.5,.5), reading noise SD .5 (F04 L world)", "H03": "person SD 1, reading SD .5, 6 contexts, 4 readings; unequal law = +.3 shift on the top context",
                      "H04": "person var 1, person x level var .25, residual var 1 (declared)"},
           "note": "synthetic parameter recovery for planning; replace inputs with pilot estimates"}
    (args.output / "recovery_curves.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
