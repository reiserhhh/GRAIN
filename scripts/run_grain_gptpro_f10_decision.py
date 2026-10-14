#!/usr/bin/env python3
"""F10 decision packet: PV01 action vs zero with paired CIs; cross-fitted shrinkage.

No new model calls. Uses frozen caches only:
- results/grain_measurement_discovery_v1/pv01/r1/application_effects.csv
- results/grain_vector_mechanisms/conditional_operator_v1/r2 (pairs, action_arrays)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PV01 = ROOT / "results/grain_measurement_discovery_v1/pv01/r1/application_effects.csv"
CACHE = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
GRID = np.array([0, 0.25, 0.5, 0.75, 1.0])
REPS = 2000


def sha(path):
    """File hash."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boot(rng, units, values, hierarchical, groups=None):
    """Mean of values under pair or task-then-pair resampling."""
    idx = np.arange(len(values))
    out = np.empty(REPS)
    for r in range(REPS):
        if not hierarchical:
            s = rng.choice(idx, len(idx), replace=True)
        else:
            tasks = sorted(set(groups))
            s = []
            for t in rng.choice(tasks, len(tasks), replace=True):
                g = idx[np.asarray(groups) == t]
                s.extend(rng.choice(g, len(g), replace=True))
            s = np.array(s)
        out[r] = values[s].mean()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def decide(lo, hi):
    """COMMON rule for loss differences new - reference."""
    if hi < 0:
        return "SUPPORTED_LOWER_LOSS"
    if lo > 0:
        return "HIGHER_LOSS"
    return "INCONCLUSIVE"


def pv01(rows_out):
    """Paired PV01 contrasts on donor and full-contrast targets."""
    rows = list(csv.DictReader(PV01.open()))
    rng = np.random.default_rng(20260915)
    for query in ("extraversion", "neuroticism"):
        sub = [r for r in rows if r["query"] == query]
        groups = [r["pair"].split("|")[2] for r in sub]
        for target, key in (
            ("donor_effect", "donor_effect"),
            ("full_contrast", "contrast"),
        ):
            y = np.array([float(r[key]) for r in sub])
            pred = {
                "zero": np.zeros_like(y),
                "euclidean": np.array([float(r["euclidean_effect"]) for r in sub]),
                "weighted": np.array([float(r["weighted_effect"]) for r in sub]),
            }
            se = {k: (v - y) ** 2 for k, v in pred.items()}
            for name, a, b in (
                ("weighted_minus_zero", "weighted", "zero"),
                ("euclidean_minus_zero", "euclidean", "zero"),
                ("weighted_minus_euclidean", "weighted", "euclidean"),
            ):
                d = se[a] - se[b]
                lo_p, hi_p = boot(rng, None, d, False)
                lo_t, hi_t = boot(rng, None, d, True, groups)
                rows_out.append(
                    {
                        "part": "pv01_action_vs_zero",
                        "query": query,
                        "target": target,
                        "contrast": name,
                        "n_pairs": len(d),
                        "mse_a": float(se[a].mean()),
                        "mse_b": float(se[b].mean()),
                        "delta_mse": float(d.mean()),
                        "pair_lower": lo_p,
                        "pair_upper": hi_p,
                        "task_pair_lower": lo_t,
                        "task_pair_upper": hi_t,
                        "decision_pair": decide(lo_p, hi_p),
                        "decision_task_pair": decide(lo_t, hi_t),
                    }
                )


def direct_predictions(dx, ds, fit_idx):
    """Original source ridge (scale on fit rows, identity-augmented least squares)."""
    scale = dx[fit_idx].std(axis=0)
    scale[scale < 1e-12] = 1
    x = dx[fit_idx] / scale
    coef = np.linalg.lstsq(
        np.vstack([x, np.eye(4)]),
        np.vstack([ds[fit_idx], np.zeros((4, 2))]),
        rcond=None,
    )[0]
    return (dx / scale) @ coef


def crossfit(rows_out, choices_out):
    """Leave-one-source-task-out selection of c versus in-sample selection."""
    pairs = list(csv.DictReader((CACHE / "pairs.csv").open()))
    with np.load(CACHE / "action_arrays.npz") as z:
        dx, ds, source = (
            z["dx"].astype(float),
            z["ds"].astype(float),
            z["source"].astype(bool),
        )
    src, app = np.flatnonzero(source), np.flatnonzero(~source)
    tasks = np.array([p["situation"] for p in pairs])
    src_tasks = sorted(set(tasks[src]))
    full = direct_predictions(dx, ds, src)
    in_sample = []
    cross = []
    for q in range(2):
        loss_in = [np.mean((c * full[src, q] - ds[src, q]) ** 2) for c in GRID]
        c_in = float(GRID[int(np.argmin(loss_in))])
        held = np.zeros(len(GRID))
        for t in src_tasks:
            fit = src[tasks[src] != t]
            test = src[tasks[src] == t]
            pred = direct_predictions(dx, ds, fit)
            held += np.array(
                [np.sum((c * pred[test, q] - ds[test, q]) ** 2) for c in GRID]
            )
        c_cf = float(GRID[int(np.argmin(held))])
        in_sample.append(c_in)
        cross.append(c_cf)
        choices_out.append(
            {
                "query": ["extraversion", "neuroticism"][q],
                "c_in_sample": c_in,
                "c_crossfit": c_cf,
                "in_sample_losses": [float(v) for v in loss_in],
                "crossfit_heldout_sse": held.tolist(),
                "source_tasks": src_tasks,
            }
        )
    rng = np.random.default_rng(20260916)
    groups = [pairs[i]["situation"] for i in app]
    for q, query in enumerate(("extraversion", "neuroticism")):
        y = ds[app, q]
        se_zero = y**2
        se_cf = (cross[q] * full[app, q] - y) ** 2
        se_in = (in_sample[q] * full[app, q] - y) ** 2
        for name, d in (
            ("crossfit_minus_zero", se_cf - se_zero),
            ("in_sample_minus_zero", se_in - se_zero),
            ("crossfit_minus_in_sample", se_cf - se_in),
        ):
            lo_p, hi_p = boot(rng, None, d, False)
            lo_t, hi_t = boot(rng, None, d, True, groups)
            rows_out.append(
                {
                    "part": "direct_text_contrast_shrinkage",
                    "query": query,
                    "target": "full_contrast",
                    "contrast": name,
                    "n_pairs": len(d),
                    "mse_a": float(
                        (se_cf if name.startswith("crossfit") else se_in).mean()
                    ),
                    "mse_b": float(
                        (se_zero if name.endswith("zero") else se_in).mean()
                    ),
                    "delta_mse": float(d.mean()),
                    "pair_lower": lo_p,
                    "pair_upper": hi_p,
                    "task_pair_lower": lo_t,
                    "task_pair_upper": hi_t,
                    "decision_pair": decide(lo_p, hi_p),
                    "decision_task_pair": decide(lo_t, hi_t),
                }
            )


def main():
    """Write decision tables and a decision note into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, choices = [], []
    pv01(rows)
    crossfit(rows, choices)
    with (args.output / "paired_decisions.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.output / "shrinkage_choices.json").write_text(
        json.dumps(choices, indent=1) + "\n"
    )
    inputs = {
        "pv01_application_effects": sha(PV01),
        "pairs": sha(CACHE / "pairs.csv"),
        "action_arrays": sha(CACHE / "action_arrays.npz"),
        "code": sha(Path(__file__)),
    }
    (args.output / "input_manifest.json").write_text(
        json.dumps(inputs, indent=1) + "\n"
    )
    for r in rows:
        print(
            f"{r['part'][:12]} {r['query'][:4]} {r['target']:13s} {r['contrast']:26s} d={r['delta_mse']:+.6f} pair[{r['pair_lower']:+.6f},{r['pair_upper']:+.6f}] task[{r['task_pair_lower']:+.6f},{r['task_pair_upper']:+.6f}] {r['decision_task_pair']}"
        )
    print(json.dumps(choices, indent=0)[:800])


if __name__ == "__main__":
    main()
