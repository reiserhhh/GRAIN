#!/usr/bin/env python
"""Cold verification and impact analysis for personal_law_r2 (protocol V2). Reads only saved files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F13/personal_law_r2"
Z = 1.959963984540054
CANDS = ("b1", "b2", "m", "eb")
IU = np.triu_indices(4)


def forward_expect(p, ys, seen, t, mechanism, selection_aware):
    """Independent 4-state forward algorithm (state index 2a+b); returns E[p(s_t) | data < t]."""
    def like(i, last):
        if seen[i]:
            return np.array([1.0 - ys[i], float(ys[i])])
        if not selection_aware:
            return np.array([1.0, 1.0])
        if mechanism == "MNAR":
            return np.array([0.2, 0.6])
        if mechanism == "MAR":
            q = 0.6 if last == 1 else 0.2
            return np.array([q, q])
        return np.array([0.4, 0.4])
    last = None
    l0 = like(0, last); last = ys[0] if seen[0] else last
    l1 = like(1, last); last = ys[1] if seen[1] else last
    alpha = np.array([0.25 * l0[a] * l1[b] for a in (0, 1) for b in (0, 1)])
    for i in range(2, t):
        li = like(i, last)
        new = np.zeros(4)
        for a in (0, 1):
            for b in (0, 1):
                s = 2 * a + b
                for y in (0, 1):
                    new[2 * b + y] += alpha[s] * (p[s] if y else 1 - p[s]) * li[y]
        alpha = new / new.sum()
        if seen[i]:
            last = ys[i]
    return float(alpha @ p / alpha.sum())


def cold_verify(ev, hist, truth, am):
    tr = truth.set_index(["world", "seed", "author_id"])
    hs = {k: g.sort_values("t") for k, g in hist.groupby(["world", "seed", "author_id", "mechanism"])}
    rec = []
    max_target_err = {"t_lat": 0.0, "t_ign": 0.0, "t_sel": 0.0, "t_fw": 0.0}
    for key, g in ev.groupby(["world", "seed", "author_id", "mechanism"]):
        h = hs[key]
        ys, seen = h.y.to_numpy(), h.seen.to_numpy().astype(bool)
        p = tr.loc[key[:3], ["p0", "p1", "p2", "p3"]].to_numpy(float)
        W = g[["w0", "w1", "w2", "w3"]].to_numpy()
        for j, t in enumerate(g.t.astype(int)):
            lat = p[2 * ys[t - 2] + ys[t - 1]]
            ign = forward_expect(p, ys, seen, t, key[3], False)
            sel = forward_expect(p, ys, seen, t, key[3], True)
            fw = float(W[j] @ p)
            for k, v in (("t_lat", lat), ("t_ign", ign), ("t_sel", sel), ("t_fw", fw)):
                max_target_err[k] = max(max_target_err[k], abs(v - g[k].iloc[j]))
        row = {"world": key[0], "seed": key[1], "author_id": key[2], "mechanism": key[3]}
        for c in CANDS:
            if g[f"{c}_point"].isna().all():
                continue
            pt, st, pe, so = (g[f"{c}_{k}"].to_numpy() for k in ("point", "state", "personal", "source"))
            if c == "b1":
                C = np.zeros((len(g), 4, 4))
            elif c == "eb":
                C = np.stack([np.diag(r) for r in g[[f"eb_var{s}" for s in range(4)]].to_numpy()])
            else:
                C = np.zeros((len(g), 4, 4))
                up = g[[f"{c}_C{i}{j}" for i, j in zip(*IU)]].to_numpy()
                C[:, IU[0], IU[1]] = up
                C[:, IU[1], IU[0]] = up
            pfw = np.einsum("es,est,et->e", W, C, W)
            hl, hf = Z * np.sqrt(st + pe + so), Z * np.sqrt(pfw + so)
            lo, hi = np.maximum(0, pt - hl), np.minimum(1, pt + hl)
            lf, uf = np.maximum(0, pt - hf), np.minimum(1, pt + hf)
            tl, ti, ts, tf = (g[k].to_numpy() for k in ("t_lat", "t_ign", "t_sel", "t_fw"))
            row.update({f"{c}|se_lat": np.mean((pt - tl) ** 2), f"{c}|cover_lat": np.mean((lo <= tl) & (tl <= hi)),
                        f"{c}|cover_int": np.mean((lo <= ti) & (ti <= hi)), f"{c}|se_int_sel": np.mean((pt - ts) ** 2),
                        f"{c}|cover_int_sel_L": np.mean((lo <= ts) & (ts <= hi)), f"{c}|pers_fw": np.mean(pfw),
                        f"{c}|width_fw": np.mean(uf - lf), f"{c}|cover_fw_own": np.mean((lf <= tf) & (tf <= uf)),
                        f"{c}|cover_fw_ign": np.mean((lf <= ti) & (ti <= uf)), f"{c}|cover_fw_sel": np.mean((lf <= ts) & (ts <= uf)),
                        f"{c}|se_fw": np.mean((pt - tf) ** 2), f"{c}|width": np.mean(hi - lo)})
        rec.append(row)
    cv = pd.DataFrame(rec).melt(id_vars=["world", "seed", "author_id", "mechanism"], var_name="k", value_name="v").dropna()
    cv[["candidate", "metric"]] = cv.k.str.split("|", expand=True)
    cv = cv.pivot_table(index=["world", "seed", "author_id", "mechanism", "candidate"], columns="metric", values="v").reset_index()
    mm = am.merge(cv, on=["world", "seed", "author_id", "mechanism", "candidate"], suffixes=("", "_cold"), validate="one_to_one")
    metric_err = {c: float(np.nanmax(np.abs(mm[c] - mm[f"{c}_cold"]))) for c in cv.columns if c not in ("world", "seed", "author_id", "mechanism", "candidate")}
    return {"max_target_err": max_target_err, "max_author_metric_err": metric_err, "rows_compared": len(mm),
            "status": "PASS" if max(max_target_err.values()) <= 1e-12 and max(metric_err.values()) <= 1e-12 and len(mm) == len(am) else "FAIL"}


def main() -> int:
    ev = pd.read_parquet(OUT / "event_trace.parquet")
    hist = pd.read_parquet(OUT / "histories.parquet")
    truth = pd.read_parquet(OUT / "truth_evaluation_only.parquet")
    am = pd.read_csv(OUT / "author_metrics_r2.csv", float_precision="round_trip")
    cold = cold_verify(ev, hist, truth, am)
    (OUT / "cold_verification.json").write_text(json.dumps(cold, indent=1) + "\n")
    # target relations by mechanism (event level)
    rel = []
    for mech, g in ev.groupby("mechanism"):
        rel.append({"mechanism": mech, "events": len(g), "max_abs_sel_minus_ign": float((g.t_sel - g.t_ign).abs().max()),
                    "mean_abs_sel_minus_ign": float((g.t_sel - g.t_ign).abs().mean()), "max_abs_sel_minus_lat": float((g.t_sel - g.t_lat).abs().max()),
                    "max_abs_fw_minus_sel": float((g.t_fw - g.t_sel).abs().max()), "mean_abs_fw_minus_sel": float((g.t_fw - g.t_sel).abs().mean())})
    pd.DataFrame(rel).to_csv(OUT / "target_relations.csv", index=False)
    # complete-history equality of L and FW (event level)
    comp = {}
    g = ev[ev.mechanism == "none"]
    for c in CANDS:
        hl = Z * np.sqrt(g[f"{c}_state"] + g[f"{c}_personal"] + g[f"{c}_source"])
        hf = Z * np.sqrt(g[f"{c}_pers_fw"] + g[f"{c}_source"])
        comp[c] = {"max_abs_state": float(g[f"{c}_state"].abs().max()), "max_abs_half_L_minus_FW": float((hl - hf).abs().max())}
    comp["max_abs_targets_lat_ign_sel_fw"] = float(np.max(np.abs(np.stack([g.t_ign - g.t_lat, g.t_sel - g.t_lat, g.t_fw - g.t_lat]))))
    comp["weights_one_hot"] = bool(((g[["w0", "w1", "w2", "w3"]].max(axis=1) == 1.0)).all())
    # off-diagonal contribution for B2/M under masks (diagnostic only; never used as an interval)
    offd = []
    for (world, mech), gg in ev[ev.mechanism != "none"].groupby(["world", "mechanism"]):
        W = gg[["w0", "w1", "w2", "w3"]].to_numpy()
        for c in ("b2", "m"):
            up = gg[[f"{c}_C{i}{j}" for i, j in zip(*IU)]].to_numpy()
            C = np.zeros((len(gg), 4, 4)); C[:, IU[0], IU[1]] = up; C[:, IU[1], IU[0]] = up
            full = np.einsum("es,est,et->e", W, C, W)
            diag_only = np.einsum("es,ess,es->e", W, C, W)
            latent_personal = np.einsum("es,ess->e", W, C)
            offd.append({"world": world, "mechanism": mech, "candidate": c, "mean_wCw": float(full.mean()), "mean_diag_only_sum_w2Css": float(diag_only.mean()),
                         "mean_legacy_sum_wCss": float(latent_personal.mean()), "mean_ratio_full_over_diag_only": float(np.mean(full[diag_only > 0] / diag_only[diag_only > 0])),
                         "max_abs_full_minus_diag_only": float(np.abs(full - diag_only).max())})
    pd.DataFrame(offd).to_csv(OUT / "offdiagonal_diagnostic.csv", index=False)
    # author-level summaries and seed-stratified author-cluster bootstrap for coverage columns
    cols = ["se_lat", "se_int", "se_int_sel", "bias_int", "bias_int_sel", "se_fw", "cover_lat", "cover_int", "cover_int_sel_L",
            "cover_fw_own", "cover_fw_ign", "cover_fw_sel", "width", "width_fw", "clipped", "clipped_fw", "state", "personal", "pers_fw", "source", "fw_unsupported"]
    summ = am.groupby(["world", "mechanism", "candidate"])[cols].mean().reset_index()
    summ.to_csv(OUT / "summary_r2.csv", index=False)
    rng = np.random.default_rng(20260926)
    boot = []
    for (world, mech, cand), gg in am.groupby(["world", "mechanism", "candidate"]):
        gg = gg.sort_values(["seed", "author_id"])
        strata = [np.where(gg.seed.to_numpy() == s)[0] for s in sorted(gg.seed.unique())]
        draws = np.stack([np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata]) for _ in range(2000)])
        for met in ("cover_lat", "cover_int", "cover_int_sel_L", "cover_fw_own", "cover_fw_ign", "cover_fw_sel", "se_int", "se_int_sel"):
            v = gg[met].to_numpy()
            bs = v[draws].mean(axis=1)
            boot.append({"world": world, "mechanism": mech, "candidate": cand, "metric": met, "mean": float(v.mean()),
                         "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5))})
    pd.DataFrame(boot).to_csv(OUT / "coverage_intervals_r2.csv", index=False)
    # MNAR target correction: paired per-author difference se_int_sel - se_int
    mn = am[am.mechanism == "MNAR"].assign(d_se=lambda d: d.se_int_sel - d.se_int, d_bias=lambda d: d.bias_int_sel - d.bias_int)
    mn.groupby(["world", "candidate"])[["se_int", "se_int_sel", "d_se", "bias_int", "bias_int_sel", "d_bias"]].mean().reset_index().to_csv(OUT / "mnar_target_correction.csv", index=False)
    files = ["author_metrics_r2.csv", "event_trace.parquet", "histories.parquet", "truth_evaluation_only.parquet", "gate.json", "source_side.json",
             "source_validation_calibration.csv", "cold_verification.json", "target_relations.csv", "offdiagonal_diagnostic.csv", "summary_r2.csv",
             "coverage_intervals_r2.csv", "mnar_target_correction.csv"]
    (OUT / "complete_history_equality.json").write_text(json.dumps(comp, indent=1) + "\n")
    files.append("complete_history_equality.json")
    (OUT / "analysis.json").write_text(json.dumps({"output_sha256": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files},
                                                   "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps({"cold": cold["status"], "complete": comp}, indent=1))
    return 0 if cold["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
