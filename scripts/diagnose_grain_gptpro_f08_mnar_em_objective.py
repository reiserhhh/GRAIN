#!/usr/bin/env python3
"""Post-hoc EM objective diagnostic for `F08/mnar_source_r1`.

The frozen acceptance asked for a non-decreasing log-likelihood. The M-step is
the smoothed (MAP: Beta(2,2) on p, Dirichlet(2) on the initial pair) update, so
EM guarantees ascent of the penalized objective
loglik + sum log p + log(1-p) + sum log init, not of the likelihood alone.
This re-implements the EM step independently (joint-state forward-backward
written separately), records the parameter history, proves the final
parameters and log-likelihood equal the frozen run, and reports both checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from run_grain_gptpro_f08_end_to_end import COUNTS, READERS, SEEDS, render
from run_grain_gptpro_f08_mnar_source import emission_law

from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.measurement_discovery.estimators import ordered_rows


def em_step(codes, law, p, init):
    """One EM step; state (y_{t-1}, y_t) indexed as 2*y_{t-1}+y_t."""
    n, horizon = codes.shape
    trans = np.zeros((4, 4))
    for s in range(4):
        trans[s, 2 * (s % 2) + 1] = p[s]
        trans[s, 2 * (s % 2)] = 1 - p[s]
    obs = law[:, codes]  # (2, n, T): P(r_t | y)
    emit = obs[[0, 1, 0, 1]].transpose(1, 2, 0)  # y_t = s % 2
    first = obs[[0, 0, 1, 1], :, 0].T  # y_0 = s // 2
    fwd = np.empty((n, horizon, 4))
    c = np.empty((n, horizon))
    v = init * first * emit[:, 1]
    c[:, 1] = v.sum(1)
    fwd[:, 1] = v / c[:, 1, None]
    for t in range(2, horizon):
        v = np.einsum("ns,sk->nk", fwd[:, t - 1], trans) * emit[:, t]
        c[:, t] = v.sum(1)
        fwd[:, t] = v / c[:, t, None]
    bwd = np.ones((n, horizon, 4))
    for t in range(horizon - 2, 0, -1):
        bwd[:, t] = np.einsum("sk,nk->ns", trans, emit[:, t + 1] * bwd[:, t + 1]) / c[:, t + 1, None]
    joint = np.zeros((4, 4))
    for t in range(2, horizon):
        joint += np.einsum("ns,sk,nk,n->sk", fwd[:, t - 1], trans, emit[:, t] * bwd[:, t], 1 / c[:, t])
    c1 = np.array([joint[s, 2 * (s % 2) + 1] for s in range(4)])
    c0 = np.array([joint[s, 2 * (s % 2)] for s in range(4)])
    pair = (fwd[:, 1] * bwd[:, 1]).sum(0)
    return float(np.log(c[:, 1:]).sum()), (c1 + 1) / (c0 + c1 + 2), (pair + 1) / (pair.sum() + 4)


def main():
    """Recompute objectives for all 24 fits; write objective_check.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.run / "em_diagnostics.json").read_text())
    report = []
    for reader, folder in READERS.items():
        with (folder / "reader_scores.csv").open() as f:
            cache = {
                (r["role"], int(r["pole"]), r["subject_status"], int(r["variant"])): r
                for r in csv.DictReader(f)
            }
        laws = {"em_validation_law": emission_law(cache, "validation"), "em_source_law": emission_law(cache, "source")}
        for world_index, world in enumerate(("c2_xor", "c2_iid")):
            for seed in SEEDS:
                records, _ = generate_world(world, seed, counts=COUNTS)
                source = [r for r in records if r["role"] == "source"]
                codes = []
                for s_index, rows in enumerate(ordered_rows(source).values()):
                    ys = [r["observation"] for r in rows]
                    rd = render(np.random.SeedSequence([seed, 8, 100 + world_index, s_index]), "mixed", ys, cache, "source")
                    codes.append([0 if x["hard"][1] < 0.5 else 1 + int(x["hard"][0]) for x in rd])
                codes = np.asarray(codes)
                for arm, law in laws.items():
                    p, init = np.full(4, 0.5), np.full(4, 0.25)
                    ll, pen = [], []
                    for _ in range(1000):
                        loglik, new_p, new_init = em_step(codes, law, p, init)
                        ll.append(loglik)
                        pen.append(loglik + float(np.sum(np.log(p) + np.log1p(-p)) + np.sum(np.log(init))))
                        change = max(np.abs(new_p - p).max(), np.abs(new_init - init).max())
                        p, init = new_p, new_init
                        if change < 1e-10:
                            break
                    ref = next(d for d in frozen if (d["reader"], d["world"], d["seed"], d["arm"]) == (reader, world, seed, arm))
                    report.append(
                        {
                            "reader": reader,
                            "world": world,
                            "seed": seed,
                            "arm": arm,
                            "iterations": len(ll),
                            "frozen_iterations": ref["iterations"],
                            "final_b1_max_abs_diff_vs_frozen": float(np.abs(p - np.asarray(ref["b1_table"])).max()),
                            "final_loglik_abs_diff_vs_frozen": float(abs(ll[-1] - ref["final_loglik"])),
                            "min_loglik_step": float(np.diff(ll).min()),
                            "min_penalized_step": float(np.diff(pen).min()),
                        }
                    )
                    print(json.dumps(report[-1]), flush=True)
    (args.run / "objective_check.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
