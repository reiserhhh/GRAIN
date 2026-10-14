#!/usr/bin/env python3
"""Held-out-context prediction (R09/R25) and rank stability / same-person
identification (R24). Frozen before running (2026-09-26, Claude Opus 5.5).

Worlds (F04-style): L linear m(z)=a+bz; T curved m(z)=a+bz+c(z^2-5/9) with
c~N(0,.5). Conditions z in {-1,-1/3,1/3,1}; 8 readings per condition; noise
SD .5; 600 persons; SeedSequence([20260926, 4, 20+k]).
1. Held-out context: fit each person on z in {-1,-1/3,1/3}; predict mean
   response at z=1 with (i) personal linear extrapolation, (ii) partial-pooling
   linear (population slope prior), (iii) the four-cell/transport rule, which
   refuses the unseen condition; MSE against the true m(1) and 95% interval
   coverage for the model-based predictors.
2. Rank stability: Spearman correlation of person slope estimates from two
   independent halves; same-person re-identification: nearest-neighbour match
   of half-1 profiles (four cell means) to half-2 profiles, accuracy vs chance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

Z = np.array([-1, -1 / 3, 1 / 3, 1.0])


def world(kind, persons, rng):
    a = rng.normal(0, 1, persons)
    b = rng.normal(0.5, 0.5, persons)
    c = rng.normal(0, 0.5, persons) if kind == "T" else np.zeros(persons)
    mean = a[:, None] + b[:, None] * Z[None, :] + c[:, None] * (Z[None, :] ** 2 - 5 / 9)
    return mean


def heldout(kind, rng, persons=600, reps=8, noise=0.5):
    mean = world(kind, persons, rng)
    y = mean[:, :, None] + rng.normal(0, noise, (persons, 4, reps))
    cell = y.mean(2)
    zt, target = Z[:3], mean[:, 3]
    x = np.column_stack([np.ones(3), zt])
    coef = np.linalg.lstsq(x, cell[:, :3].T, rcond=None)[0].T
    pred_personal = coef[:, 0] + coef[:, 1] * 1.0
    xtx_inv = np.linalg.inv(x.T @ x)
    var_personal = (noise**2 / reps) * np.array([1, 1.0]) @ xtx_inv @ np.array([1, 1.0])
    # partial pooling: shrink the slope toward the population slope (source prior b ~ N(.5,.25))
    se2_b = (noise**2 / reps) * xtx_inv[1, 1]
    w = 0.25 / (0.25 + se2_b)
    b_pool = w * coef[:, 1] + (1 - w) * 0.5
    a_hat = cell[:, :3].mean(1) - b_pool * zt.mean()
    pred_pool = a_hat + b_pool * 1.0
    out = {}
    for name, pred, var in (("personal_linear", pred_personal, var_personal), ("partial_pooling", pred_pool, var_personal * w**2 + (noise**2 / reps) / 3)):
        err = pred - target
        out[name] = {"mse": float(np.mean(err**2)), "coverage95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(var)))}
    out["four_cell_transport_rule"] = {"status": "REFUSED_UNSEEN_CONDITION", "supported_share": 0.0}
    out["population_mean_at_z1"] = {"mse": float(np.mean((cell[:, :3].mean() + 0.5 * (1 - zt.mean()) - target) ** 2))}
    return out


def rank_reid(rng, persons=600, reps=8, noise=0.5):
    mean = world("L", persons, rng)
    h1 = (mean[:, :, None] + rng.normal(0, noise, (persons, 4, reps))).mean(2)
    h2 = (mean[:, :, None] + rng.normal(0, noise, (persons, 4, reps))).mean(2)
    zc = Z - Z.mean()
    s1, s2 = (h1 * zc).sum(1) / (zc**2).sum(), (h2 * zc).sum(1) / (zc**2).sum()
    d = ((h1[:, None, :] - h2[None, :, :]) ** 2).sum(2)
    acc = float(np.mean(np.argmin(d, axis=1) == np.arange(persons)))
    return {"slope_rank_spearman": float(spearmanr(s1, s2)[0]), "level_rank_spearman": float(spearmanr(h1.mean(1), h2.mean(1))[0]),
            "reidentification_accuracy": acc, "chance": 1 / persons}


def main():
    """Write heldout_rank.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = {"heldout_L": heldout("L", np.random.default_rng(np.random.SeedSequence([20260926, 4, 21]))),
           "heldout_T": heldout("T", np.random.default_rng(np.random.SeedSequence([20260926, 4, 22]))),
           "rank_reidentification": rank_reid(np.random.default_rng(np.random.SeedSequence([20260926, 4, 23])))}
    (args.output / "heldout_rank.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
