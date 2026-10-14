#!/usr/bin/env python
"""R24 target repair (docs/GRAIN_R24_TARGET_REPAIR_PROTOCOL_V2.md): replay personal_law_r1 exactly and add
selection-aware integrated targets and fixed-working-weight companions. CPU only, 0 neural calls.

Stages: full (replay + gate + event trace).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r24_personal_law as r1  # noqa: E402  (unchanged r1 module)
from grain_research.gptpro_foundation_v3.f08_estimators import FEATURES  # noqa: E402
from grain_research.gptpro_foundation_v3.f13_binary_ensemble import SIGMA2, EnsembleHistory, components  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R24_TARGET_REPAIR_PROTOCOL_V2.md"
PROTOCOL_SHA = "b9feb3ea2c9fc522aca12f428fc36c20ed97bbf8948b4c8c82726270a440fecb"
R1_RUNNER_SHA = "92e4e91a256a1332fc3a01cbdf2b0dcf050ca6ca80e9272a34360e29bc0960c9"
R1 = r1.OUT
OUT = r1.R / "F13/personal_law_r2"
Z = r1.Z
CANDS = ("b1", "b2", "m", "eb")
LEGACY = ("se_lat", "bias_lat", "se_int", "bias_int", "brier", "cover_lat", "cover_int", "width", "state", "personal", "source", "event", "clipped")
NEW = ("se_int_sel", "bias_int_sel", "cover_int_sel_L", "pers_fw", "width_fw", "clipped_fw", "cover_fw_own", "cover_fw_ign", "cover_fw_sel",
       "se_fw", "bias_fw", "diag_consistency_err", "fw_unsupported")
IU = np.triu_indices(4)
EV_COLS = (["t", "y", "seen", "w0", "w1", "w2", "w3", "t_lat", "t_ign", "t_sel", "t_fw"]
           + [f"{c}_{k}" for c in CANDS for k in ("v0", "v1", "v2", "v3", "point", "state", "personal", "source", "pers_fw")]
           + [f"b2_C{i}{j}" for i, j in zip(*IU)] + [f"m_C{i}{j}" for i, j in zip(*IU)] + ["eb_var0", "eb_var1", "eb_var2", "eb_var3"])


def integrated_sel(p_u, observed_ys, observed_flags, t, mechanism):
    """E[p_u(s_t) | observed prefix, missingness indicators < t, true law, declared mask law] (evaluation only)."""
    pair = np.full((2, 2), 0.25)
    last = None
    for i in range(t):
        if observed_flags[i]:
            like = np.array([1 - observed_ys[i], observed_ys[i]], float)
            last = observed_ys[i]
        elif mechanism == "MNAR":
            like = np.array([0.2, 0.6])
        elif mechanism == "MAR":
            pi = 0.6 if last == 1 else 0.2
            like = np.array([pi, pi])
        elif mechanism == "MCAR":
            like = np.array([0.4, 0.4])
        else:
            raise ValueError(f"missing step under mechanism {mechanism}")
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


def working_covariances(ens):
    """Full working personal covariance at the canonical fit (orientation from EnsembleHistory.per_state)."""
    a = ens.gram[:1] + np.eye(4)[None]
    cov = SIGMA2 * np.linalg.inv(a)
    c_b2 = np.einsum("si,fij,tj->fst", FEATURES, cov, FEATURES)[0]
    n = ens.counts[:1].sum(2)
    raw = (ens.counts[:1, :, 1] + 1) / (n + 2)
    rv = raw * (1 - raw) / (n + 3)
    c_m = np.einsum("fjs,fj,fjt->fst", ens.projector[:1], rv, ens.projector[:1])[0]
    return c_b2, c_m


def score_author_r2(ys, p_u, seen, fits, inits, hypers, run_eb, mechanism):
    """Legacy lines copied verbatim from r1.score_author, plus evaluation-side additions."""
    ens = EnsembleHistory(fits, inits)
    A = np.array([h[0] for h in hypers])
    Bt = np.array([h[1] for h in hypers])
    x, n = np.zeros(4), np.zeros(4)
    cands = ("b1", "b2", "m", "eb") if run_eb else ("b1", "b2", "m")
    acc = {c: {k: [] for k in LEGACY + NEW} for c in cands}
    obs_vals = [y if s else 0 for y, s in zip(ys, seen)]
    events = []
    for t, y in enumerate(ys):
        if t >= 12:
            values, weights, personal = ens.per_state()
            comp = components(values, weights, personal)
            lat = float(p_u[2 * ys[t - 2] + ys[t - 1]])
            integ = r1.true_integrated(p_u, obs_vals, seen, t)
            preds = {c: (comp["point"][i], comp["state"][i], comp["personal"][i], comp["source"][i], comp["event"][i]) for i, c in enumerate(("b1", "b2", "m"))}
            if run_eb:
                m, v = r1.eb_state(A, Bt, x[None, :], n[None, :])          # (F,4)
                pred_f = np.einsum("fs,fs->f", weights, m)
                w0 = weights[0]
                point = float(np.clip(pred_f[0], 1e-6, 1 - 1e-6))
                state = float(w0 @ (m[0] - pred_f[0]) ** 2)
                pers = float(w0 @ v[0])
                src = float(pred_f[1:].var()) if len(pred_f) > 1 else 0.0
                event = float(w0 @ (m[0] * (1 - m[0])))
                preds["eb"] = (point, state, pers, src, event)
            # ---- evaluation-side additions (never fed back into any estimator)
            wc = weights[0]
            integ_sel = integrated_sel(p_u, obs_vals, seen, t, mechanism)
            t_fw = float(wc @ p_u)
            c_b2, c_m = working_covariances(ens)
            cmat = {"b1": np.zeros((4, 4)), "b2": c_b2, "m": c_m}
            legacy_diag = {"b1": personal[0, 0], "b2": personal[0, 1], "m": personal[0, 2]}
            if run_eb:
                cmat["eb"] = np.diag(v[0])
                legacy_diag["eb"] = v[0]
            ev = [t, y, float(seen[t]), *wc, lat, integ, integ_sel, t_fw]
            for c in CANDS:
                if c not in preds:
                    ev += [np.nan] * 9
                    continue
                pt, st, pe, so, _e = preds[c]
                diag_err = float(np.max(np.abs(np.diag(cmat[c]) - legacy_diag[c])))
                pers_fw = float(wc @ cmat[c] @ wc)
                vals = values[0, ("b1", "b2", "m").index(c)] if c != "eb" else m[0]
                ev += [*vals, pt, st, pe, so, pers_fw]
                a = acc[c]
                half = Z * np.sqrt(st + pe + so)
                lo, hi = max(0.0, pt - half), min(1.0, pt + half)
                a["se_int_sel"].append((pt - integ_sel) ** 2); a["bias_int_sel"].append(pt - integ_sel)
                a["cover_int_sel_L"].append(float(lo <= integ_sel <= hi))
                a["diag_consistency_err"].append(diag_err)
                a["se_fw"].append((pt - t_fw) ** 2); a["bias_fw"].append(pt - t_fw)
                if diag_err <= 1e-12:
                    hf = Z * np.sqrt(pers_fw + so)
                    lf, hf_ = max(0.0, pt - hf), min(1.0, pt + hf)
                    a["pers_fw"].append(pers_fw); a["width_fw"].append(hf_ - lf); a["clipped_fw"].append(float(pt - hf < 0 or pt + hf > 1))
                    a["cover_fw_own"].append(float(lf <= t_fw <= hf_)); a["cover_fw_ign"].append(float(lf <= integ <= hf_))
                    a["cover_fw_sel"].append(float(lf <= integ_sel <= hf_)); a["fw_unsupported"].append(0.0)
                else:
                    for k in ("pers_fw", "width_fw", "clipped_fw", "cover_fw_own", "cover_fw_ign", "cover_fw_sel"):
                        a[k].append(np.nan)
                    a["fw_unsupported"].append(1.0)
            ev += [*c_b2[IU], *c_m[IU], *(v[0] if run_eb else [np.nan] * 4)]
            events.append(ev)
            # ---- legacy scoring, verbatim
            for c, (pt, st, pe, so, ev_) in preds.items():
                half = Z * np.sqrt(st + pe + so)
                lo, hi = max(0.0, pt - half), min(1.0, pt + half)
                a = acc[c]
                a["se_lat"].append((pt - lat) ** 2); a["bias_lat"].append(pt - lat)
                a["se_int"].append((pt - integ) ** 2); a["bias_int"].append(pt - integ)
                a["brier"].append((pt - y) ** 2)
                a["cover_lat"].append(float(lo <= lat <= hi)); a["cover_int"].append(float(lo <= integ <= hi))
                a["width"].append(hi - lo)
                a["state"].append(st); a["personal"].append(pe); a["source"].append(so); a["event"].append(ev_)
                a["clipped"].append(float(pt - half < 0 or pt + half > 1))
        # observation happens after prediction (prediction-before-observation)
        q = 1.0 if seen[t] else 0.0
        ens.observe(float(y) if seen[t] else 0.5, q)
        if t >= 2 and seen[t] and seen[t - 1] and seen[t - 2]:
            s = 2 * ys[t - 2] + ys[t - 1]
            n[s] += 1
            x[s] += y
    res = {}
    for c, a in acc.items():
        res[c] = {k: float(np.mean(a[k])) for k in LEGACY}
        for k in NEW:
            vals_ = np.asarray(a[k], float)
            res[c][k] = float(np.max(vals_)) if k == "diag_consistency_err" else (float(np.sum(vals_)) if k == "fw_unsupported" else float(np.nanmean(vals_)) if np.isfinite(vals_).any() else np.nan)
    return res, np.asarray(events, float)


def run_world_seed(world, seed, n_refits, n_app, mechs):
    records, truth = r1.personal_world(world, seed)
    fits, inits, hypers, smean, used = r1.source_side(world, seed, records, n_refits)
    rows, ev_blocks, hist = [], [], []
    apps = [(a, g) for a, g in r1.ordered_rows([r for r in records if r["role"] != "source"]).items() if g[0]["role"] == "application"][:n_app]
    for ai, (author, group) in enumerate(apps):
        ys = [int(r["observation"]) for r in group]
        for mi, mech in enumerate(r1.MECHS):
            if mech not in mechs:
                continue
            seen = r1.mask(ys, mech, np.random.default_rng(np.random.SeedSequence([seed, 13, 31, r1.WORLD_CODE[world], ai, mi])))
            res, ev = score_author_r2(ys, truth[author], seen, fits, inits, hypers, mech in ("none", "MCAR"), mech)
            for c, v in res.items():
                rows.append({"world": world, "seed": seed, "author_id": author, "mechanism": mech, "candidate": c, "missing_rate": 1 - float(np.mean(seen)), **v})
            ev_blocks.append((author, mech, ev))
            hist += [{"world": world, "seed": seed, "author_id": author, "mechanism": mech, "t": t, "y": y, "seen": int(s)} for t, (y, s) in enumerate(zip(ys, seen))]
    val = []
    for author, group in r1.ordered_rows([r for r in records if r["role"] == "validation"]).items():
        ys = [int(r["observation"]) for r in group]
        res = r1.score_author(ys, truth[author], [True] * 48, fits[:1], inits[:1], hypers[:1], run_eb=True)
        val.append({"world": world, "seed": seed, "author_id": author, **{f"{c}_brier": v["brier"] for c, v in res.items()}})
    side = {"world": world, "seed": seed, "tau2_by_state": hypers[0][2].tolist(), "refits_used": used, "source_mean": smean,
            "eb_alpha": hypers[0][0].tolist(), "eb_beta": hypers[0][1].tolist()}
    frames = []
    for author, mech, ev in ev_blocks:
        f = pd.DataFrame(ev, columns=EV_COLS)
        f.insert(0, "mechanism", mech); f.insert(0, "author_id", author); f.insert(0, "seed", seed); f.insert(0, "world", world)
        frames.append(f)
    truth_rows = [{"world": world, "seed": seed, "author_id": a, **{f"p{s}": float(truth[a][s]) for s in range(4)}} for a, _g in apps]
    return rows, val, side, pd.concat(frames, ignore_index=True), pd.DataFrame(hist), pd.DataFrame(truth_rows)


def file_hashes(d: Path):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file() and not p.name.startswith("._")}


def gate(new_rows: pd.DataFrame, val: pd.DataFrame, side: list):
    old = pd.read_csv(R1 / "author_metrics.csv", float_precision="round_trip")
    keys = ["world", "seed", "author_id", "mechanism", "candidate"]
    cols = list(LEGACY) + ["missing_rate"]
    m = old.merge(new_rows[keys + cols], on=keys, how="outer", suffixes=("_r1", "_r2"), indicator=True, validate="one_to_one")
    out = {"rows_r1": len(old), "rows_r2": len(new_rows), "unmatched": int((m["_merge"] != "both").sum())}
    diffs = {}
    for c in cols:
        a, b = m[f"{c}_r1"].to_numpy(float), m[f"{c}_r2"].to_numpy(float)
        same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        diffs[c] = float(np.nanmax(np.abs(a - b))) if same_nan else float("inf")
    out["max_abs_diff"] = diffs
    vold = pd.read_csv(R1 / "source_validation_calibration.csv", float_precision="round_trip")
    vnew = val.sort_values(["world", "seed", "author_id"]).reset_index(drop=True)
    vold = vold.sort_values(["world", "seed", "author_id"]).reset_index(drop=True)
    out["validation_equal"] = bool(vold.shape == vnew.shape and (vold.columns == vnew.columns).all()
                                   and all(np.array_equal(vold[c].to_numpy(), vnew[c].to_numpy()) for c in vold.columns))
    sold = json.loads((R1 / "source_side.json").read_text())
    key = lambda h: (h["world"], h["seed"])  # noqa: E731
    out["source_side_equal"] = sorted(sold, key=key) == sorted(json.loads(json.dumps(side)), key=key)
    out["status"] = "PASS" if (out["unmatched"] == 0 and max(diffs.values()) == 0.0 and out["validation_equal"] and out["source_side_equal"]) else "STOP_GATE"
    return out


def main() -> int:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    if hashlib.sha256((ROOT / "scripts/run_grain_r24_personal_law.py").read_bytes()).hexdigest() != R1_RUNNER_SHA:
        raise RuntimeError("r1 runner changed")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["full"], required=True)
    ap.parse_args()
    if OUT.exists():
        raise FileExistsError("personal_law_r2 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    r1_before = file_hashes(R1)
    (OUT / "source_archive").mkdir()
    for p in (Path(__file__), ROOT / "scripts/run_grain_r24_personal_law.py", PROTOCOL):
        shutil.copyfile(p, OUT / "source_archive" / f"{hashlib.sha256(p.read_bytes()).hexdigest()}_{p.name}")
    t0 = time.time()
    from joblib import Parallel, delayed
    jobs = [(w, s) for w in ("W0", "H1", "H2") for s in r1.SEEDS]
    res = Parallel(n_jobs=2)(delayed(run_world_seed)(w, s, 200, 200, r1.MECHS) for w, s in jobs)
    rows = pd.DataFrame([r for rr, *_ in res for r in rr])
    val = pd.DataFrame([v for _, vv, *_ in res for v in vv])
    side = [h for _, _, h, *_ in res]
    g = gate(rows, val, side)
    target = OUT if g["status"] == "PASS" else OUT / "failed_gate"
    target.mkdir(exist_ok=True)
    rows.to_csv(target / "author_metrics_r2.csv", index=False)
    val.to_csv(target / "source_validation_calibration.csv", index=False)
    (target / "source_side.json").write_text(json.dumps(side, indent=1) + "\n")
    pd.concat([e for *_, e, _h, _t in res], ignore_index=True).to_parquet(target / "event_trace.parquet", index=False, compression="zstd")
    pd.concat([h for *_, _e, h, _t in res], ignore_index=True).to_parquet(target / "histories.parquet", index=False, compression="zstd")
    pd.concat([tt for *_, tt in res], ignore_index=True).to_parquet(target / "truth_evaluation_only.parquet", index=False, compression="zstd")
    (target / "gate.json").write_text(json.dumps(g, indent=1) + "\n")
    r1_after = file_hashes(R1)
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0, "jobs": len(jobs), "refits": 200, "workers": 2,
                                                  "r1_files_unchanged": r1_before == r1_after, "r1_file_hashes": r1_before,
                                                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps({"gate": g["status"], "seconds": time.time() - t0, "r1_unchanged": r1_before == r1_after}))
    return 0 if g["status"] == "PASS" and r1_before == r1_after else 2


if __name__ == "__main__":
    raise SystemExit(main())
