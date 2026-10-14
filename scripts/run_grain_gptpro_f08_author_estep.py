#!/usr/bin/env python3
"""F08 author-level E-step for M under a measured emission law.

See docs/GRAIN_F08_AUTHOR_ESTEP_PROTOCOL_V1.md. Mixed panel only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from run_grain_gptpro_f08_end_to_end import (
    COUNTS,
    READERS,
    SEEDS,
    render,
    score_application,
)
from run_grain_gptpro_f08_mnar_source import em_fit, emission_law, fitted_from_counts

from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f02_recovery import fit_observed_source
from grain_research.measurement_discovery.estimators import ordered_rows

F08 = ROOT / "results/grain_gptpro_foundation_v3/F08"


def author_em(codes, law, p0, init, max_iter=1000, tol=1e-10):
    """Per-author MAP EM for the second-order table; initial pair law fixed.

    Returns tables (n,4), expected counts (n,4,2), iterations (n,), and the
    per-author penalized objective history (iterations x n).
    """
    n, horizon = codes.shape
    like = law.T[codes]  # (n, T, 2)
    emit = np.stack([like[:, :, s & 1] for s in range(4)], axis=2)
    first = np.stack([like[:, 0, s >> 1] for s in range(4)], axis=1)
    p = np.tile(np.asarray(p0, float), (n, 1))
    active = np.ones(n, dtype=bool)
    iterations = np.zeros(n, dtype=int)
    history = []
    counts = np.zeros((n, 4, 2))
    for _ in range(max_iter):
        trans = np.zeros((n, 4, 4))
        for s in range(4):
            b = s & 1
            trans[:, s, 2 * b] = 1 - p[:, s]
            trans[:, s, 2 * b + 1] = p[:, s]
        alpha = np.zeros((n, horizon, 4))
        scale = np.zeros((n, horizon))
        alpha[:, 1] = init[None, :] * first * emit[:, 1]
        scale[:, 1] = alpha[:, 1].sum(1)
        alpha[:, 1] /= scale[:, 1, None]
        for t in range(2, horizon):
            alpha[:, t] = np.einsum("ns,nsk->nk", alpha[:, t - 1], trans) * emit[:, t]
            scale[:, t] = alpha[:, t].sum(1)
            alpha[:, t] /= scale[:, t, None]
        loglik = np.log(scale[:, 1:]).sum(1)
        history.append(loglik + np.sum(np.log(p) + np.log1p(-p), axis=1))
        beta = np.ones((n, horizon, 4))
        for t in range(horizon - 2, 0, -1):
            beta[:, t] = np.einsum("nsk,nk->ns", trans, emit[:, t + 1] * beta[:, t + 1]) / scale[:, t + 1, None]
        new_counts = np.zeros((n, 4, 2))
        for t in range(2, horizon):
            xi = alpha[:, t - 1, :, None] * trans * (emit[:, t] * beta[:, t])[:, None, :] / scale[:, t, None, None]
            for s in range(4):
                b = s & 1
                new_counts[:, s, 0] += xi[:, s, 2 * b]
                new_counts[:, s, 1] += xi[:, s, 2 * b + 1]
        new_p = (new_counts[:, :, 1] + 1) / (new_counts.sum(2) + 2)
        change = np.abs(new_p - p).max(1)
        p = np.where(active[:, None], new_p, p)
        counts = np.where(active[:, None, None], new_counts, counts)
        iterations += active
        active &= change >= tol
        if not active.any():
            break
    return p, counts, iterations, np.asarray(history)


def main():
    """Readers x worlds x seeds; M from author-level E-step vs pooled vs oracle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((F08 / "mnar_source_r1/em_diagnostics.json").read_text())
    rows_out, diagnostics = [], []
    for reader, folder in READERS.items():
        with (folder / "reader_scores.csv").open() as f:
            cache = {(r["role"], int(r["pole"]), r["subject_status"], int(r["variant"])): r for r in csv.DictReader(f)}
        laws = {"validation_law": emission_law(cache, "validation"), "source_law": emission_law(cache, "source")}
        for world_index, world in enumerate(("c2_xor", "c2_iid")):
            for seed in SEEDS:
                records, _ = generate_world(world, seed, counts=COUNTS)
                source = [r for r in records if r["role"] == "source"]
                oracle_fit = fit_observed_source(source, world)
                init0 = np.ones((2, 2))
                for g in ordered_rows(source).values():
                    init0[g[0]["observation"], g[1]["observation"]] += 1
                init0 /= init0.sum()
                mean0 = float(np.mean([r["observation"] for r in source]))
                codes = []
                for s_index, rows in enumerate(ordered_rows(source).values()):
                    ys = [r["observation"] for r in rows]
                    rd = render(np.random.SeedSequence([seed, 8, 100 + world_index, s_index]), "mixed", ys, cache, "source")
                    codes.append([0 if x["hard"][1] < 0.5 else 1 + int(x["hard"][0]) for x in rd])
                codes = np.asarray(codes)
                arms = {"oracle_source": (oracle_fit, init0, mean0)}
                for name, law in laws.items():
                    pooled = em_fit(codes, law)
                    ref = next(d for d in frozen if (d["reader"], d["world"], d["seed"], d["arm"]) == (reader, world, seed, f"em_{name}"))
                    replay_diff = float(np.abs(pooled["p"] - np.asarray(ref["b1_table"])).max())
                    pooled_fit = fitted_from_counts(pooled["counts"], world)
                    pooled_init = pooled["init"].reshape(2, 2)
                    if name == "validation_law":
                        arms["em_pooled_validation_law"] = (pooled_fit, pooled_init, pooled["y_mean"])
                    tables, _counts, iters, hist = author_em(codes, law, pooled["p"], pooled["init"])
                    mean = tables.mean(0)
                    _, singular, right = np.linalg.svd(tables - mean, full_matrices=False)
                    author_fit = dict(pooled_fit, m={"mean": mean, "right": right[:2], "singular_values": singular, "rank": 2})
                    arms[f"em_author_{name}"] = (author_fit, pooled_init, pooled["y_mean"])
                    steps = np.diff(hist, axis=0) if len(hist) > 1 else np.zeros((1, len(codes)))
                    oracle_tables = []
                    for rows in ordered_rows(source).values():
                        c = np.zeros((4, 2))
                        v = [r["observation"] for r in rows]
                        for i in range(2, len(v)):
                            c[2 * v[i - 2] + v[i - 1], v[i]] += 1
                        oracle_tables.append((c[:, 1] + 1) / (c.sum(1) + 2))
                    oracle_tables = np.asarray(oracle_tables)
                    pooled_tables = (pooled["counts"][:, :, 1] + 1) / (pooled["counts"].sum(2) + 2)
                    diagnostics.append({
                        "reader": reader, "world": world, "seed": seed, "law": name,
                        "pooled_replay_max_abs_diff": replay_diff,
                        "author_iterations_max": int(iters.max()), "author_iterations_median": float(np.median(iters)),
                        "min_penalized_step": float(steps.min()),
                        "author_table_rmse_vs_oracle_tables": float(np.sqrt(np.mean((tables - oracle_tables) ** 2))),
                        "pooled_table_rmse_vs_oracle_tables": float(np.sqrt(np.mean((pooled_tables - oracle_tables) ** 2))),
                        "author_table_sd": float(tables.std(0).mean()), "pooled_table_sd": float(pooled_tables.std(0).mean()),
                        "oracle_table_sd": float(oracle_tables.std(0).mean()),
                        "singular_values_author": singular[:3].tolist(),
                        "singular_values_oracle": np.linalg.svd(oracle_tables - oracle_tables.mean(0), compute_uv=False)[:3].tolist(),
                    })
                apps = ordered_rows([r for r in records if r["role"] != "source"])
                for a_index, (author, rows) in enumerate(apps.items()):
                    if rows[0]["role"] != "application":
                        continue
                    ys = [r["observation"] for r in rows]
                    rd = render(np.random.SeedSequence([seed, 8, world_index, a_index]), "mixed", ys, cache, "application")
                    for arm, (fitted, init, mean) in arms.items():
                        res = score_application(fitted, init, mean, "mixed", rd, ys, "hard")
                        for cand, (brier, logloss) in res.items():
                            rows_out.append({"reader": reader, "world": world, "seed": seed, "author_id": author, "arm": arm, "candidate": cand, "brier": float(brier), "log_loss": float(logloss)})
                print(json.dumps({"reader": reader, "world": world, "seed": seed}), flush=True)
    with (args.output / "metrics_by_author.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    (args.output / "author_em_diagnostics.json").write_text(json.dumps(diagnostics, indent=1) + "\n")
    (args.output / "input_manifest.json").write_text(json.dumps(
        {r: hashlib.sha256((f / "reader_scores.csv").read_bytes()).hexdigest() for r, f in READERS.items()}
        | {"code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "mnar_em_diagnostics": hashlib.sha256((F08 / "mnar_source_r1/em_diagnostics.json").read_bytes()).hexdigest()},
        indent=1) + "\n")


if __name__ == "__main__":
    main()
