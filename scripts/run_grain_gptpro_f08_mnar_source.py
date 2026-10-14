#!/usr/bin/env python3
"""F08 MNAR-aware source fit: EM with a measured reader emission law.

See docs/GRAIN_F08_MNAR_SOURCE_PROTOCOL_V1.md. Mixed panel only.
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
    STATUSES,
    fit_read_source,
    render,
    score_application,
)

from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f02_recovery import fit_observed_source
from grain_research.gptpro_foundation_v3.f08_estimators import FEATURES
from grain_research.measurement_discovery.estimators import ordered_rows

QUALIFIED = ("actual", "negated")


def emission_law(cache, role):
    """P(r | y) with r = 0 missing, 1 read 0, 2 read 1, under the render law."""
    law = np.zeros((2, 3))
    for y in (0, 1):
        for status in STATUSES:
            for variant in range(4):
                poles = (y,) if status in QUALIFIED else (0, 1)
                for pole in poles:
                    m = cache[(role, pole, status, variant)]
                    weight = 0.25 * 0.25 / len(poles)
                    if float(m["qualification_score"]) < 0.5:
                        law[y, 0] += weight
                    else:
                        law[y, 1 + int(float(m["pole_score"]) >= 0.5)] += weight
    return law


def em_fit(readings, law, max_iter=1000, tol=1e-10):
    """Pooled second-order chain EM; returns parameters and expected counts.

    readings: int array (authors, T) in {0, 1, 2}. Pair state s = 2*y_{t-1}+y_t.
    """
    n, horizon = readings.shape
    like = law.T[readings]  # (n, T, 2): P(r_t | y_t)
    p = np.full(4, 0.5)
    init = np.full(4, 0.25)
    history = []
    for iteration in range(max_iter):
        trans = np.zeros((4, 4))
        for s in range(4):
            b = s & 1
            trans[s, 2 * b] = 1 - p[s]
            trans[s, 2 * b + 1] = p[s]
        emit = np.stack([like[:, :, s & 1] for s in range(4)], axis=2)  # y_t = s & 1
        first = np.stack([like[:, 0, s >> 1] for s in range(4)], axis=1)
        alpha = np.zeros((n, horizon, 4))
        scale = np.zeros((n, horizon))
        alpha[:, 1] = init[None, :] * first * emit[:, 1]
        scale[:, 1] = alpha[:, 1].sum(axis=1)
        alpha[:, 1] /= scale[:, 1, None]
        for t in range(2, horizon):
            alpha[:, t] = (alpha[:, t - 1] @ trans) * emit[:, t]
            scale[:, t] = alpha[:, t].sum(axis=1)
            alpha[:, t] /= scale[:, t, None]
        loglik = float(np.log(scale[:, 1:]).sum())
        history.append(loglik)
        beta = np.ones((n, horizon, 4))
        for t in range(horizon - 2, 0, -1):
            beta[:, t] = (emit[:, t + 1] * beta[:, t + 1]) @ trans.T / scale[:, t + 1, None]
        post = alpha * beta
        counts = np.zeros((n, 4, 2))
        for t in range(2, horizon):
            xi = (
                alpha[:, t - 1, :, None]
                * trans[None]
                * (emit[:, t] * beta[:, t])[:, None, :]
                / scale[:, t, None, None]
            )
            for s in range(4):
                b = s & 1
                counts[:, s, 0] += xi[:, s, 2 * b]
                counts[:, s, 1] += xi[:, s, 2 * b + 1]
        pooled = counts.sum(axis=0)
        new_p = (pooled[:, 1] + 1) / (pooled.sum(axis=1) + 2)
        pair_counts = post[:, 1].sum(axis=0)
        new_init = (pair_counts + 1) / (pair_counts.sum() + 4)
        change = max(np.abs(new_p - p).max(), np.abs(new_init - init).max())
        p, init = new_p, new_init
        if change < tol:
            break
    y_mean = float(
        (post[:, 1:, 1].sum() + post[:, 1:, 3].sum() + post[:, 1, 2].sum() + post[:, 1, 3].sum())
        / (n * horizon)
    )
    return {
        "p": p,
        "init": init,
        "counts": counts,
        "y_mean": y_mean,
        "iterations": iteration + 1,
        "loglik": history,
    }


def fitted_from_counts(counts, world):
    """B1/B2/M from expected per-author counts (declared forms)."""
    pooled = counts.sum(axis=0)
    b1 = (pooled[:, 1] + 1) / (pooled.sum(axis=1) + 2)
    gram = FEATURES.T @ (pooled.sum(axis=1)[:, None] * FEATURES) + np.eye(4)
    coef = np.linalg.solve(gram, FEATURES.T @ pooled[:, 1])
    tables = (counts[:, :, 1] + 1) / (counts.sum(axis=2) + 2)
    mean = tables.mean(axis=0)
    _, singular, right = np.linalg.svd(tables - mean, full_matrices=False)
    return {
        "world": world,
        "b1": {"counts": pooled, "source_p": b1},
        "b2": {"coef": coef, "alpha": 1.0},
        "m": {"mean": mean, "right": right[:2], "singular_values": singular, "rank": 2},
    }


def main():
    """Readers x worlds x seeds, mixed panel; four source arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows_out, diagnostics, laws = [], [], {}
    for reader, folder in READERS.items():
        with (folder / "reader_scores.csv").open() as f:
            cache = {
                (r["role"], int(r["pole"]), r["subject_status"], int(r["variant"])): r
                for r in csv.DictReader(f)
            }
        reader_laws = {
            "em_validation_law": emission_law(cache, "validation"),
            "em_source_law": emission_law(cache, "source"),
        }
        laws[reader] = {k: v.tolist() for k, v in reader_laws.items()}
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
                read_hist, codes, read_vals = {}, [], []
                for s_index, (author, rows) in enumerate(ordered_rows(source).items()):
                    ys = [r["observation"] for r in rows]
                    rd = render(
                        np.random.SeedSequence([seed, 8, 100 + world_index, s_index]),
                        "mixed",
                        ys,
                        cache,
                        "source",
                    )
                    seq, code = [], []
                    for r, x in zip(rows, rd, strict=True):
                        p, q = x["hard"]
                        if q < 0.5:
                            seq.append(None)
                            code.append(0)
                        else:
                            seq.append(dict(r, observation=int(p)))
                            code.append(1 + int(p))
                            read_vals.append(int(p))
                    read_hist[author] = seq
                    codes.append(code)
                codes = np.asarray(codes)
                gap_fit = fit_read_source(read_hist, world)
                init1 = np.ones((2, 2))
                for seq in read_hist.values():
                    if seq[0] is not None and seq[1] is not None:
                        init1[seq[0]["observation"], seq[1]["observation"]] += 1
                init1 /= init1.sum()
                arms = {
                    "oracle_source": (oracle_fit, init0, mean0),
                    "gap_aware_read_source": (gap_fit, init1, float(np.mean(read_vals))),
                }
                for arm, law in reader_laws.items():
                    em = em_fit(codes, law)
                    ll = np.asarray(em["loglik"])
                    fitted = fitted_from_counts(em["counts"], world)
                    arms[arm] = (fitted, em["init"].reshape(2, 2), em["y_mean"])
                    diagnostics.append(
                        {
                            "reader": reader,
                            "world": world,
                            "seed": seed,
                            "arm": arm,
                            "iterations": em["iterations"],
                            "final_loglik": float(ll[-1]),
                            "min_loglik_step": float(np.diff(ll).min()) if ll.size > 1 else 0.0,
                            "b1_table": fitted["b1"]["source_p"].tolist(),
                            "oracle_b1_table": oracle_fit["b1"]["source_p"].tolist(),
                            "max_abs_b1_shift": float(
                                np.abs(fitted["b1"]["source_p"] - oracle_fit["b1"]["source_p"]).max()
                            ),
                            "y_mean": em["y_mean"],
                            "oracle_y_mean": mean0,
                        }
                    )
                apps = ordered_rows([r for r in records if r["role"] != "source"])
                for a_index, (author, rows) in enumerate(apps.items()):
                    if rows[0]["role"] != "application":
                        continue
                    ys = [r["observation"] for r in rows]
                    rd = render(
                        np.random.SeedSequence([seed, 8, world_index, a_index]),
                        "mixed",
                        ys,
                        cache,
                        "application",
                    )
                    for arm, (fitted, init, mean) in arms.items():
                        res = score_application(fitted, init, mean, "mixed", rd, ys, "hard")
                        for cand, (brier, logloss) in res.items():
                            rows_out.append(
                                {
                                    "reader": reader,
                                    "world": world,
                                    "seed": seed,
                                    "author_id": author,
                                    "arm": arm,
                                    "candidate": cand,
                                    "brier": float(brier),
                                    "log_loss": float(logloss),
                                }
                            )
                print(json.dumps({"reader": reader, "world": world, "seed": seed}), flush=True)
    with (args.output / "metrics_by_author.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    (args.output / "em_diagnostics.json").write_text(json.dumps(diagnostics, indent=1) + "\n")
    (args.output / "emission_laws.json").write_text(json.dumps(laws, indent=1) + "\n")
    (args.output / "input_manifest.json").write_text(
        json.dumps(
            {
                r: hashlib.sha256((f / "reader_scores.csv").read_bytes()).hexdigest()
                for r, f in READERS.items()
            }
            | {"code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
