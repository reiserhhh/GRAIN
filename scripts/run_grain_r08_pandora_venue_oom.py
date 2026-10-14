#!/usr/bin/env python
"""R08 natural venue-sequence spectral OOM vs static/order-one/blend (docs/GRAIN_R08_PANDORA_VENUE_OOM_PROTOCOL_V1.md).
CPU only, 0 neural calls. Source-train fit, source-validation rank rule (3 candidates), prequential held scoring."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_pandora_history as ph  # noqa: E402  (pinned loader and row-identity checks)

PROTOCOL = ROOT / "docs/GRAIN_R08_PANDORA_VENUE_OOM_PROTOCOL_V1.md"
PROTOCOL_SHA = "59ea431fe4ab9e9fbea21d338118ef85d22416482b1c962b8cf035559a45673a"
PREV = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_history_comparison_r1"
PREV_ROWS_SHA = "7e086a0ebfd5a097ea9f63320f3bd7e74207b20524e570598aea7ec357941199"
PH_CODE_SHA = "128f5b648cda42b0f71fd2292ccb659de77874f92181b6776c010b7d9f2e9b01"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1"
K_TOP, ALPHA, WV, T0 = 20, 3, 0.2, 25
RANKS = (5, 10, 20)
EPS_MASS, FLOOR, EPS_UPD = 1e-12, 1e-4, 1e-12
GATE = {"neg_share": .05, "refusal": .01, "reset": .01}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def roles_and_dictionary(m):
    src_auth = sorted(m[m.split == "source"].author_index.unique())
    val_auth = set(a for j, a in enumerate(src_auth) if j % 5 == 0)
    train_auth = set(src_auth) - val_auth
    held_auth = sorted(m[m.split != "source"].author_index.unique())
    tr = m.author_index.isin(train_auth).to_numpy()
    top = m[tr].condition.value_counts().index[:K_TOP].tolist()
    cat = {v: i for i, v in enumerate(top)}
    m["cat"] = m.condition.map(lambda v: cat.get(v, K_TOP))
    return sorted(train_auth), sorted(val_auth), held_auth, tr, top


def fit_stats(seqs, C):
    P1 = np.zeros(C); P21 = np.zeros((C, C)); P3 = np.zeros((C, C, C))
    for s in seqs:
        np.add.at(P1, s, 1)
        np.add.at(P21, (s[1:], s[:-1]), 1)
        np.add.at(P3, (s[1:-1], s[2:], s[:-2]), 1)  # P3[x][k, i] = P(x_{t+2}=k, x_{t+1}=x, x_t=i)
    return P1 / P1.sum(), P21 / P21.sum(), P3 / P3.sum()


def fit_oom(P1, P21, P3, k):
    U, s, _ = np.linalg.svd(P21)
    U = U[:, :k]
    b1 = U.T @ P1
    binf = np.linalg.pinv(P21.T @ U) @ P1
    Mp = np.linalg.pinv(U.T @ P21)
    B = np.stack([U.T @ P3[x] @ Mp for x in range(P1.size)])
    return {"U": U, "s": s, "b1": b1, "binf": binf, "B": B, "k": k}


def oom_filter(seq, f, t_from=T0):
    """Prequential: predict event t from events < t, then update. Returns per-scored-event records."""
    b = f["b1"].copy(); out = []; resets_all = 0
    for t in range(len(seq)):
        q = (f["B"] @ b) @ f["binf"]
        if t >= t_from:
            fin = np.all(np.isfinite(q))
            absq = np.abs(q).sum() if fin else np.nan
            neg = float(np.clip(-q, 0, None).sum() / absq) if fin and absq > 0 else float("nan")
            qp = np.clip(q, 0, None) if fin else None
            refused = (not fin) or qp.sum() <= EPS_MASS
            p = None
            floored = False
            if not refused:
                p = qp / qp.sum()
                floored = bool(p[seq[t]] < FLOOR)
                p = np.maximum(p, FLOOR); p = p / p.sum()
            rec = {"t": t, "p": p, "neg_share": neg, "norm_dev": float(abs(q.sum() - 1)) if fin else float("nan"),
                   "refused": bool(refused), "floored": floored}
        y = seq[t]; qy = q[y]
        nb = f["B"][y] @ b / qy if np.isfinite(qy) and qy > EPS_UPD else None
        reset = nb is None or not np.all(np.isfinite(nb))
        b = f["b1"].copy() if reset else nb
        resets_all += int(reset)
        if t >= t_from:
            rec["reset_after"] = bool(reset)
            out.append(rec)
    return out, resets_all


def implied_bigram(f, C):
    J = np.array([[f["binf"] @ f["B"][j] @ f["B"][i] @ f["b1"] for j in range(C)] for i in range(C)])  # J[i, j] = P(x1=i, x2=j)
    J = np.clip(J, 0, None)
    return J / J.sum()


def main() -> int:
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    if sha(PROTOCOL) != PROTOCOL_SHA or sha(PREV / "held_event_scores.csv") != PREV_ROWS_SHA or sha(ROOT / "scripts/run_grain_r08_pandora_history.py") != PH_CODE_SHA:
        raise RuntimeError("pinned protocol / previous rows / previous code changed")
    if OUT.exists():
        raise FileExistsError("pandora_venue_oom_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    m, _X, ident = ph.load()
    train_auth, val_auth, held_auth, tr, top = roles_and_dictionary(m)
    assert (len(train_auth), len(val_auth), len(held_auth)) == (403, 101, 120)
    C = K_TOP + 1
    cats, raw, ts = m.cat.to_numpy(), m.condition.to_numpy(), m.timestamp_unix_seconds.to_numpy()
    by_author = {a: np.where(m.author_index.to_numpy() == a)[0] for a in m.author_index.unique()}
    assert all(len(by_author[a]) == 36 for a in train_auth + val_auth + held_auth)
    # baselines (identical to packet 15 Part C)
    g = np.bincount(cats[tr], minlength=C) + 1.0; g /= g.sum()
    T = np.ones((C, C))
    for a in train_auth:
        c = cats[by_author[a]]
        np.add.at(T, (c[:-1], c[1:]), 1)
    T /= T.sum(1, keepdims=True)

    def base_preds(ix, t):
        prev = cats[ix[:t]]
        cnt = np.bincount(prev, minlength=C)
        static = (cnt + ALPHA * g) / (cnt.sum() + ALPHA)
        o1 = T[prev[-1]]
        return {"global": g, "static": static, "order1": o1, "combined": WV * o1 + (1 - WV) * static}
    # OOM fit on source-train only
    P1, P21, P3 = fit_stats([cats[by_author[a]] for a in train_auth], C)
    fits = {k: fit_oom(P1, P21, P3, k) for k in RANKS}
    emp_tr = P21.T  # [i, j]
    vp = np.zeros((C, C))
    for a in val_auth:
        s = cats[by_author[a]]; np.add.at(vp, (s[:-1], s[1:]), 1)
    emp_val = vp / vp.sum()
    val = {}
    for k, f in fits.items():
        recs, resets = [], 0
        for a in val_auth:
            s = cats[by_author[a]]
            r_, ra = oom_filter(s, f); resets += ra
            for r in r_:
                r["y"] = s[r["t"]]
            recs += r_
        ok = [r for r in recs if not r["refused"]]
        J = implied_bigram(f, C)
        val[k] = {"events": len(recs), "logloss_nonrefused": float(np.mean([-np.log(r["p"][r["y"]]) for r in ok])) if ok else None,
                  "neg_share_mean": float(np.nanmean([r["neg_share"] for r in recs])), "refusal_rate": float(np.mean([r["refused"] for r in recs])),
                  "reset_rate_scored_updates": float(np.mean([r["reset_after"] for r in recs])), "reset_rate_all_updates": resets / (36 * len(val_auth)),
                  "floored_rate": float(np.mean([r["floored"] for r in recs])), "norm_dev_median": float(np.nanmedian([r["norm_dev"] for r in recs])),
                  "bigram_TV_train_insample": float(.5 * np.abs(J - emp_tr).sum()), "bigram_TV_validation": float(.5 * np.abs(J - emp_val).sum()),
                  "singular_values_P21_top": np.round(f["s"][:k], 6).tolist()}
        v = val[k]
        v["gate_G1_pass"] = bool(v["neg_share_mean"] <= GATE["neg_share"] and v["refusal_rate"] <= GATE["refusal"]
                                 and v["reset_rate_scored_updates"] <= GATE["reset"])  # protocol: scored events (all-update rate reported only)
    eligible = [k for k in RANKS if val[k]["gate_G1_pass"] and val[k]["logloss_nonrefused"] is not None]
    k_star = min(eligible, key=lambda k: val[k]["logloss_nonrefused"]) if eligible else None
    # validation baselines for context (not used for selection)
    vb = {n: [] for n in ("global", "static", "order1", "combined")}
    for a in val_auth:
        ix = by_author[a]
        for t in range(T0, 36):
            bp = base_preds(ix, t)
            for n in vb:
                vb[n].append(-np.log(bp[n][cats[ix[t]]]))
    res = {"protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": PROTOCOL_SHA, "row_identity": ident,
           "roles": {"source_train": len(train_auth), "source_validation": len(val_auth), "held": len(held_auth)},
           "dictionary_top20": top, "ranks_candidates": list(RANKS), "gate": GATE, "validation": val,
           "validation_baseline_logloss": {n: float(np.mean(v)) for n, v in vb.items()}, "eligible_ranks": eligible, "chosen_rank": k_star, "neural_calls": 0}
    roles = {"source_train": [int(a) for a in train_auth], "source_validation": [int(a) for a in val_auth], "held": [int(a) for a in held_auth]}
    (OUT / "author_roles.json").write_text(json.dumps(roles) + "\n")
    np.savez(OUT / "oom_fit_source.npz", P1=P1, P21=P21, P3=P3, **{f"k{k}_{n}": fits[k][n] for k in RANKS for n in ("U", "b1", "binf", "B")}, s=fits[RANKS[0]]["s"])
    # held baselines + exact reproduction gate
    rows, dists = [], {n: [] for n in ("global", "static", "order1", "combined", "oom")}
    prev = pd.read_csv(PREV / "held_event_scores.csv", float_precision="round_trip")
    oom_by_author = {}
    if k_star is not None:
        for a in held_auth:
            recs, _ = oom_filter(cats[by_author[a]], fits[k_star])
            oom_by_author[a] = {r["t"]: r for r in recs}
    for a in held_auth:
        ix = by_author[a]
        for t in range(T0, 36):
            y = cats[ix[t]]
            bp = base_preds(ix, t)
            r = {"author": int(a), "t": t, "y": int(y), "y_is_other": bool(y == K_TOP), "novel_raw_venue": bool(raw[ix[t]] not in set(raw[ix[:t]])),
                 "gap_days": float((ts[ix[t]] - ts[ix[t - 1]]) / 86400)}
            for n, p in bp.items():
                r[f"p_{n}"] = float(p[y]); r[f"ll_{n}"] = float(-np.log(p[y])); r[f"brier_{n}"] = float(((p - np.eye(C)[y]) ** 2).sum())
                dists[n].append(p)
            if k_star is not None:
                o = oom_by_author[a][t]
                r.update({"oom_refused": o["refused"], "oom_floored": o["floored"], "oom_neg_share": o["neg_share"], "oom_norm_dev": o["norm_dev"],
                          "oom_reset_after": o["reset_after"]})
                if o["refused"]:
                    r.update({"p_oom": np.nan, "ll_oom": np.nan, "brier_oom": np.nan, "ll_oom_fulldenom": float(-np.log(g[y]))})
                    dists["oom"].append(np.full(C, np.nan))
                else:
                    p = o["p"]
                    r.update({"p_oom": float(p[y]), "ll_oom": float(-np.log(p[y])), "brier_oom": float(((p - np.eye(C)[y]) ** 2).sum()),
                              "ll_oom_fulldenom": float(-np.log(p[y]))})
                    dists["oom"].append(p)
            rows.append(r)
    df = pd.DataFrame(rows)
    mg = df.merge(prev, on=["author", "t"], suffixes=("", "_prev"))
    assert len(mg) == len(df) == 1320
    repro = {n: float(np.abs(mg[f"ll_{n}"] - mg[f"venue_ll_{n}"]).max()) for n in ("global", "static", "order1", "combined")}
    res["baseline_reproduction_max_abs_diff"] = repro
    if max(repro.values()) > 1e-12:
        (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
        raise RuntimeError(f"baseline reproduction failed: {repro}")
    df.to_csv(OUT / "held_event_rows.csv", index=False)
    np.savez(OUT / "held_predictive_distributions.npz", **{n: np.array(v) for n, v in dists.items() if v})
    if k_star is None:
        res["held_oom"] = "NOT RUN: no candidate rank passed gate G1 (declared estimator/support failure)"
    else:
        rng = np.random.default_rng(20260926)
        auths = np.array(sorted(df.author.unique()))
        draws = rng.integers(0, len(auths), size=(2000, len(auths)))
        ok = df[~df.oom_refused]

        def paired(col, ref, data):
            per = data.groupby("author")[[col, ref]].mean().reindex(auths)
            d = (per[col] - per[ref]).to_numpy()
            bs = np.nanmean(d[draws], 1)
            return {"mean": float(np.nanmean(d)), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        H = {"coverage": float(1 - df.oom_refused.mean()), "scored_events": int(len(ok)), "refused_events": int(df.oom_refused.sum()),
             "floored_events": int(df.oom_floored.sum()), "resets_after_scored": int(df.oom_reset_after.sum()),
             "neg_share_mean": float(df.oom_neg_share.mean()), "norm_dev_median": float(df.oom_norm_dev.median())}
        H["logloss_paired_set"] = {n: float(ok[f"ll_{n}"].mean()) for n in ("global", "static", "order1", "combined", "oom")}
        H["brier_paired_set"] = {n: float(ok[f"brier_{n}"].mean()) for n in ("global", "static", "order1", "combined", "oom")}
        H["delta_logloss_paired"] = {ref: paired("ll_oom", f"ll_{ref}", ok) for ref in ("static", "order1", "combined")}
        H["delta_brier_paired"] = {ref: paired("brier_oom", f"brier_{ref}", ok) for ref in ("static", "order1", "combined")}
        H["full_denominator_logloss_fallback_global"] = {"oom": float(df.ll_oom_fulldenom.mean()), "static": float(df.ll_static.mean()),
                                                          "combined": float(df.ll_combined.mean()),
                                                          "delta_vs_static": paired("ll_oom_fulldenom", "ll_static", df),
                                                          "delta_vs_combined": paired("ll_oom_fulldenom", "ll_combined", df)}
        dec = {}
        for name, mask in (("outcome_top20", ~ok.y_is_other), ("outcome_OTHER", ok.y_is_other), ("novel_raw_venue", ok.novel_raw_venue),
                           ("known_raw_venue", ~ok.novel_raw_venue)):
            sub = ok[mask]
            dec[name] = {"events": int(len(sub)), "share": float(len(sub) / len(ok)),
                         **{f"ll_{n}": float(sub[f"ll_{n}"].mean()) for n in ("static", "order1", "combined", "oom")},
                         **{f"delta_oom_minus_{n}": float((sub.ll_oom - sub[f"ll_{n}"]).mean()) for n in ("static", "order1", "combined")},
                         **{f"contribution_to_pooled_delta_vs_{n}": float((sub.ll_oom - sub[f"ll_{n}"]).sum() / len(ok)) for n in ("static", "order1", "combined")}}
        H["decomposition"] = dec
        ok2 = ok.copy(); ok2["gap_tercile"] = pd.qcut(ok2.gap_days, 3, labels=["short", "mid", "long"])
        H["gap_terciles"] = {str(k): v for k, v in ok2.groupby("gap_tercile", observed=True)[["gap_days", "ll_static", "ll_combined", "ll_oom"]].mean().round(5).to_dict(orient="index").items()}
        res["held_oom"] = H
    res["seconds"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "finished": datetime.now(timezone.utc).isoformat(), "seconds": res["seconds"],
                                                  "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)), "protocol_sha256": PROTOCOL_SHA,
                                                  "inputs": {str(p.relative_to(ROOT)): h for p, h in ph.PINS.items()} | {"held_event_scores.csv(prev)": PREV_ROWS_SHA}},
                                                 indent=1) + "\n")
    print(json.dumps({"validation": {k: {kk: v[kk] for kk in ("logloss_nonrefused", "neg_share_mean", "refusal_rate", "reset_rate_all_updates", "gate_G1_pass")} for k, v in val.items()},
                      "chosen": k_star, "repro": repro, "seconds": res["seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
