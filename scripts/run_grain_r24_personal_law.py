#!/usr/bin/env python
"""R24 personal laws and honest uncertainty (docs/GRAIN_R24_PERSONAL_LAW_PROTOCOL_V1.md). CPU only, 0 neural calls.

Stages: smoke (timing only) | full (sizes from amendment A1) | replay (canonical W0 check).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f02 import c2_next_probability, generate_world, opaque_author  # noqa: E402
from grain_research.gptpro_foundation_v3.f02_recovery import fit_observed_source  # noqa: E402
from grain_research.gptpro_foundation_v3.f13_binary_ensemble import EnsembleHistory, components  # noqa: E402
from grain_research.gptpro_foundation_v3.f13_binary_source import author_statistics, refit_statistics  # noqa: E402
from grain_research.measurement_discovery.estimators import ordered_rows  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R24_PERSONAL_LAW_PROTOCOL_V1.md"
PROTOCOL_SHA = "5df0cdb08afa385bc4e13906fe3cee7331d5b15c33fca2c17434ba4abd0d7b1f"
R = ROOT / "results/grain_gptpro_foundation_v3"
OUT = R / "F13/personal_law_r1"
BOOT = R / "F13/binary_source_full_r2"
SEEDS = (20260915, 20260916, 20260917)
COUNTS = {"source": 300, "validation": 100, "application": 200}
MECHS = ("none", "MCAR", "MAR", "MNAR")
P_XOR = np.array([0.25, 0.75, 0.75, 0.25])
P_COPY = np.array([0.25, 0.75, 0.25, 0.75])
WORLD_CODE = {"W0": 0, "H1": 1, "H2": 2}
Z = 1.959963984540054


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- worlds
def personal_world(world: str, seed: int, counts=COUNTS):
    """Records in F02 schema plus an evaluation-only truth table {author_id: p_u(s) (4,)}."""
    if world == "W0":
        records, _ = generate_world("c2_xor", seed, counts=counts)
        return records, {r["author_id"]: P_XOR.copy() for r in records}
    code = WORLD_CODE[world]
    records, truth, index = [], {}, 0
    for role_number, role in enumerate(("source", "validation", "application")):
        author_rng, material_rng, noise_rng = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 24, code, role_number]).spawn(3)]
        for a in range(counts[role]):
            author_id = opaque_author(seed, f"personal_{world}", role, a)
            if world == "H1":
                p = sigmoid(logit(P_XOR) + author_rng.normal(0, 1.0))
            else:
                w = author_rng.beta(2, 2)
                p = w * P_XOR + (1 - w) * P_COPY
            truth[author_id] = p
            ys = [int(material_rng.integers(2)), int(material_rng.integers(2))]
            for t in range(48):
                if t >= 2:
                    ys.append(int(noise_rng.random() < p[2 * ys[-2] + ys[-1]]))
                y = ys[t]
                records.append({"schema_version": "grain-gptpro-record-v3", "author_id": author_id, "event_id": f"{world}_{seed}_{index:08d}",
                                "episode_id": f"{author_id}_episode", "t": t, "condition": {"z": -1 if t % 2 == 0 else 1}, "opportunity": 1,
                                "opportunity_status": "assigned", "selected": 1, "observation": y, "text_ref": f"synthetic://{world}/{index}",
                                "reader_id": "synthetic_identity_reader", "role": role})
                index += 1
    return records, truth


# ---------------------------------------------------------------- EB personal law
def eb_hyper(counts: np.ndarray):
    """Per-state method-of-moments Beta prior from source author counts (N,4,2)."""
    x, n = counts[:, :, 1], counts.sum(axis=2)
    alpha, beta, tau2 = np.zeros(4), np.zeros(4), np.zeros(4)
    for s in range(4):
        keep = n[:, s] > 0
        xs, ns = x[keep, s], n[keep, s]
        k, N = len(ns), ns.sum()
        pbar = xs.sum() / N
        S = (ns * (xs / ns - pbar) ** 2).sum()
        denom = N - (ns ** 2).sum() / N
        t2 = max(0.0, (S - pbar * (1 - pbar) * (k - 1)) / denom)
        ab = min(1e6, pbar * (1 - pbar) / t2 - 1) if t2 > 0 else 1e6
        ab = max(ab, 1e-3)
        alpha[s], beta[s], tau2[s] = pbar * ab, (1 - pbar) * ab, t2
    return alpha, beta, tau2


def eb_state(alpha, beta, x, n):
    """Posterior mean and variance per state (vectorised over refits)."""
    a, b = alpha + x, beta + (n - x)
    m = a / (a + b)
    v = a * b / ((a + b) ** 2 * (a + b + 1))
    return m, v


# ---------------------------------------------------------------- truth-side filter (evaluation only)
def true_integrated(p_u, observed_ys, observed_flags, t):
    """E[p_u(s_t) | observed prefix, true law], forward filter over pair states."""
    pair = np.full((2, 2), 0.25)
    for i in range(t):
        like = np.array([1 - observed_ys[i], observed_ys[i]], float) if observed_flags[i] else np.array([0.5, 0.5])
        if i == 0:
            pair = pair * like[:, None]
        elif i == 1:
            pair = pair * like[None, :]
        else:
            trans = p_u.reshape(2, 2)
            joint = pair[:, :, None] * np.stack([1 - trans, trans], -1) * like
            pair = joint.sum(axis=0)
        pair = pair / pair.sum()
    return float(pair.ravel() @ p_u)


def mask(ys, mechanism, rng):
    observed, last = [], None
    u = rng.random(len(ys))
    for t, y in enumerate(ys):
        p = {"none": 0.0, "MCAR": 0.4}.get(mechanism)
        if mechanism == "MAR":
            p = 0.6 if last == 1 else 0.2
        elif mechanism == "MNAR":
            p = 0.6 if y == 1 else 0.2
        seen = u[t] >= p
        observed.append(seen)
        if seen:
            last = y
    return observed


# ---------------------------------------------------------------- source side
def source_side(world, seed, records, n_refits):
    source = [r for r in records if r["role"] == "source"]
    assert all(r["role"] == "source" for r in source)
    canonical = fit_observed_source(source, "c2_xor")
    init0 = np.ones((2, 2))
    groups = ordered_rows(source)
    for g in groups.values():
        init0[g[0]["observation"], g[1]["observation"]] += 1
    init0 /= init0.sum()
    ids = sorted(groups)
    stats = author_statistics([groups[a] for a in ids])
    fits, inits, hypers = [canonical], [init0], [eb_hyper(stats["counts"])]
    if world == "W0":
        files = sorted(BOOT.glob(f"c2_xor_{seed}_b*.json"))[:n_refits]
        for p in files:
            b = json.loads(p.read_text())
            fits.append(b["fit"])
            inits.append(np.asarray(b["fit"]["initial_pair"], float))
            hypers.append(eb_hyper(stats["counts"][np.asarray(b["sampled_indices"], int)]))
    else:
        for b in range(n_refits):
            idx = np.random.default_rng(np.random.SeedSequence([seed, 13, 8, 10 + WORLD_CODE[world], b])).integers(0, len(ids), len(ids))
            f = refit_statistics(stats, idx)
            fits.append(f)
            inits.append(np.asarray(f["initial_pair"], float))
            hypers.append(eb_hyper(stats["counts"][idx]))
    smean = float(np.mean([r["observation"] for r in source]))
    return fits, inits, hypers, smean, len(files) if world == "W0" else n_refits


# ---------------------------------------------------------------- scoring one author / mechanism
def score_author(ys, p_u, seen, fits, inits, hypers, run_eb):
    ens = EnsembleHistory(fits, inits)
    A = np.array([h[0] for h in hypers])
    Bt = np.array([h[1] for h in hypers])
    x, n = np.zeros(4), np.zeros(4)
    acc = {c: {k: [] for k in ("se_lat", "bias_lat", "se_int", "bias_int", "brier", "cover_lat", "cover_int", "width", "state", "personal", "source", "event", "clipped")}
           for c in (("b1", "b2", "m", "eb") if run_eb else ("b1", "b2", "m"))}
    obs_vals = [y if s else 0 for y, s in zip(ys, seen)]
    for t, y in enumerate(ys):
        if t >= 12:
            values, weights, personal = ens.per_state()
            comp = components(values, weights, personal)
            lat = float(p_u[2 * ys[t - 2] + ys[t - 1]])
            integ = true_integrated(p_u, obs_vals, seen, t)
            preds = {c: (comp["point"][i], comp["state"][i], comp["personal"][i], comp["source"][i], comp["event"][i]) for i, c in enumerate(("b1", "b2", "m"))}
            if run_eb:
                m, v = eb_state(A, Bt, x[None, :], n[None, :])          # (F,4)
                pred_f = np.einsum("fs,fs->f", weights, m)
                w0 = weights[0]
                point = float(np.clip(pred_f[0], 1e-6, 1 - 1e-6))
                state = float(w0 @ (m[0] - pred_f[0]) ** 2)
                pers = float(w0 @ v[0])
                src = float(pred_f[1:].var()) if len(pred_f) > 1 else 0.0
                event = float(w0 @ (m[0] * (1 - m[0])))
                preds["eb"] = (point, state, pers, src, event)
            for c, (pt, st, pe, so, ev) in preds.items():
                half = Z * np.sqrt(st + pe + so)
                lo, hi = max(0.0, pt - half), min(1.0, pt + half)
                a = acc[c]
                a["se_lat"].append((pt - lat) ** 2); a["bias_lat"].append(pt - lat)
                a["se_int"].append((pt - integ) ** 2); a["bias_int"].append(pt - integ)
                a["brier"].append((pt - y) ** 2)
                a["cover_lat"].append(float(lo <= lat <= hi)); a["cover_int"].append(float(lo <= integ <= hi))
                a["width"].append(hi - lo)
                a["state"].append(st); a["personal"].append(pe); a["source"].append(so); a["event"].append(ev)
                a["clipped"].append(float(pt - half < 0 or pt + half > 1))
        # observation happens after prediction (prediction-before-observation)
        q = 1.0 if seen[t] else 0.0
        ens.observe(float(y) if seen[t] else 0.5, q)
        if t >= 2 and seen[t] and seen[t - 1] and seen[t - 2]:
            s = 2 * ys[t - 2] + ys[t - 1]
            n[s] += 1
            x[s] += y
    return {c: {k: float(np.mean(v)) for k, v in a.items()} for c, a in acc.items()}


def run_world_seed(world, seed, n_refits, n_app, mechs):
    records, truth = personal_world(world, seed)
    fits, inits, hypers, smean, used = source_side(world, seed, records, n_refits)
    rows = []
    apps = [(a, g) for a, g in ordered_rows([r for r in records if r["role"] != "source"]).items() if g[0]["role"] == "application"][:n_app]
    for ai, (author, group) in enumerate(apps):
        ys = [int(r["observation"]) for r in group]
        for mi, mech in enumerate(MECHS):
            if mech not in mechs:
                continue
            seen = mask(ys, mech, np.random.default_rng(np.random.SeedSequence([seed, 13, 31, WORLD_CODE[world], ai, mi])))
            res = score_author(ys, truth[author], seen, fits, inits, hypers, run_eb=mech in ("none", "MCAR"))
            for c, v in res.items():
                rows.append({"world": world, "seed": seed, "author_id": author, "mechanism": mech, "candidate": c, "missing_rate": 1 - float(np.mean(seen)), **v})
    # source-validation observed-outcome calibration (canonical fits only; no truth)
    val = []
    for author, group in ordered_rows([r for r in records if r["role"] == "validation"]).items():
        ys = [int(r["observation"]) for r in group]
        res = score_author(ys, truth[author], [True] * 48, fits[:1], inits[:1], hypers[:1], run_eb=True)
        val.append({"world": world, "seed": seed, "author_id": author, **{f"{c}_brier": v["brier"] for c, v in res.items()}})
    tau2 = hypers[0][2].tolist()
    return rows, val, {"world": world, "seed": seed, "tau2_by_state": tau2, "refits_used": used, "source_mean": smean,
                       "eb_alpha": hypers[0][0].tolist(), "eb_beta": hypers[0][1].tolist()}


def replay_check():
    arch = pd.read_csv(R / "F13/recovery3_r1/author_metrics.csv", float_precision="round_trip")
    arch = arch[(arch.world == "c2_xor") & (arch.seed == 20260915) & (arch.mechanism == "none")]
    rows, _v, _h = run_world_seed("W0", 20260915, 200, 200, ("none",))
    new = pd.DataFrame(rows)
    new = new[new.candidate.isin(["b1", "b2", "m"])]
    m = new.merge(arch, on=["author_id", "candidate"], suffixes=("", "_a"), validate="one_to_one")
    diffs = {k: float(np.abs(m[k1] - m[k2]).max()) for k, k1, k2 in (("se", "se_lat", "se"), ("bias", "bias_lat", "bias"), ("brier", "brier", "brier_a"),
                                                                    ("cover", "cover_lat", "cover"), ("width", "width", "width_a"))}
    return {"rows": len(m), "max_abs_diff": diffs, "status": "PASS" if len(m) == 600 and max(diffs.values()) <= 1e-12 else "STOP_REPLAY"}


def main() -> int:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "replay", "full"], required=True)
    ap.add_argument("--refits", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.stage == "smoke":
        tm = {}
        for w in ("W0", "H1"):
            t1 = time.time()
            run_world_seed(w, SEEDS[0], 20, 20, ("none",))
            tm[w] = time.time() - t1
        est = {"smoke_seconds": tm, "note": "timing only; no scientific output inspected",
               "per_author_mech_seconds_at_20_refits": {w: v / 20 for w, v in tm.items()}}
        (OUT / "smoke_timing.json").write_text(json.dumps(est, indent=1) + "\n")
        print(json.dumps(est))
        return 0
    if a.stage == "replay":
        out = replay_check()
        (OUT / "replay_check.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out))
        return 0 if out["status"] == "PASS" else 2
    if json.loads((OUT / "replay_check.json").read_text())["status"] != "PASS":
        raise RuntimeError("replay gate not passed")
    from joblib import Parallel, delayed
    jobs = [(w, s) for w in ("W0", "H1", "H2") for s in SEEDS[: a.seeds]]
    res = Parallel(n_jobs=2)(delayed(run_world_seed)(w, s, a.refits, 200, MECHS) for w, s in jobs)
    pd.DataFrame([r for rr, _, _ in res for r in rr]).to_csv(OUT / "author_metrics.csv", index=False)
    pd.DataFrame([v for _, vv, _ in res for v in vv]).to_csv(OUT / "source_validation_calibration.csv", index=False)
    (OUT / "source_side.json").write_text(json.dumps([h for _, _, h in res], indent=1) + "\n")
    (OUT / "runtime_full.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0, "jobs": len(jobs), "refits": a.refits,
                                                       "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}) + "\n")
    print(json.dumps({"seconds": time.time() - t0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
