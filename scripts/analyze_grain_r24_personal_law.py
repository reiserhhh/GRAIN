#!/usr/bin/env python
"""Analysis for R24 personal laws (protocol V1 + A1). Author-cluster, seed-stratified bootstrap (2,000 draws, seed 20260926)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F13/personal_law_r1"
METS = ("se_lat", "se_int", "bias_lat", "brier", "cover_lat", "cover_int", "width", "state", "personal", "source", "event", "clipped")
CONTRASTS = [("eb", "b2"), ("eb", "m"), ("eb", "b1"), ("b2", "b1"), ("m", "b1")]


def homogeneous_and_tau_checks():
    """W0 point-level |EB - B1| on complete histories (canonical fit only) and evaluation-only true tau2 vs source MoM."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_grain_r24_personal_law as rr
    from grain_research.measurement_discovery.estimators import ordered_rows

    out = {}
    for seed in rr.SEEDS:
        records, truth = rr.personal_world("W0", seed)
        fits, inits, hypers, _sm, _u = rr.source_side("W0", seed, records, 0)
        a0, b0, _t = hypers[0]
        apps = [g for g in ordered_rows([r for r in records if r["role"] != "source"]).values() if g[0]["role"] == "application"][:200]
        diffs, by_state = [], {s: [] for s in range(4)}
        for g in apps:
            ys = [int(r["observation"]) for r in g]
            ens = rr.EnsembleHistory(fits, inits)
            x, n = np.zeros(4), np.zeros(4)
            for t, y in enumerate(ys):
                if t >= 12:
                    values, weights, personal = ens.per_state()
                    b1 = float(rr.components(values, weights, personal)["point"][0])
                    m, _v = rr.eb_state(a0, b0, x, n)
                    eb = float(np.clip(weights[0] @ m, 1e-6, 1 - 1e-6))
                    s = 2 * ys[t - 2] + ys[t - 1]
                    diffs.append(abs(eb - b1)); by_state[s].append(abs(eb - b1))
                ens.observe(float(y), 1.0)
                if t >= 2:
                    s = 2 * ys[t - 2] + ys[t - 1]; n[s] += 1; x[s] += y
        out[str(seed)] = {"max_abs_eb_minus_b1": float(max(diffs)), "mean_abs_eb_minus_b1": float(np.mean(diffs)),
                          "max_by_state": {str(s): float(max(v)) for s, v in by_state.items()}, "predictions": len(diffs),
                          "tau2": _t.tolist()}
    (OUT / "homogeneous_reduction.json").write_text(json.dumps(out, indent=1) + "\n")
    # evaluation-only population tau2: H1 logit-normal by Gauss-Hermite; H2 closed form Var(w)*(.5)^2 for states 2/3
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(80)
    gh_w = gh_w / gh_w.sum()
    rows = []
    side = json.loads((OUT / "source_side.json").read_text())
    for h in side:
        for s in range(4):
            p0 = rr.P_XOR[s]
            if h["world"] == "H1":
                vals = rr.sigmoid(rr.logit(p0) + gh_x)
                true = float(gh_w @ vals ** 2 - (gh_w @ vals) ** 2)
            elif h["world"] == "H2":
                true = 0.05 * 0.25 if s >= 2 else 0.0
            else:
                true = 0.0
            rows.append({"world": h["world"], "seed": h["seed"], "state": s, "true_population_tau2": true, "source_mom_tau2": h["tau2_by_state"][s]})
    pd.DataFrame(rows).to_csv(OUT / "tau2_truth_vs_source.csv", index=False)


def main() -> int:
    m = pd.read_csv(OUT / "author_metrics.csv", float_precision="round_trip")
    summ = m.groupby(["world", "mechanism", "candidate"])[list(METS)].mean().reset_index()
    summ.to_csv(OUT / "summary.csv", index=False)
    rng = np.random.default_rng(20260926)
    rows, cover_rows = [], []
    for (world, mech), g in m.groupby(["world", "mechanism"]):
        piv = {c: g[g.candidate == c].set_index(["seed", "author_id"]).sort_index() for c in g.candidate.unique()}
        keys = next(iter(piv.values())).index
        seeds = keys.get_level_values(0).to_numpy()
        strata = [np.where(seeds == s)[0] for s in np.unique(seeds)]
        draws = np.stack([np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata]) for _ in range(2000)])
        for c, df in piv.items():
            for met in ("cover_lat", "cover_int"):
                v = df.loc[keys, met].to_numpy()
                bs = v[draws].mean(axis=1)
                cover_rows.append({"world": world, "mechanism": mech, "candidate": c, "metric": met, "mean": float(v.mean()),
                                   "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5)), "authors": len(v)})
        for a, b in CONTRASTS:
            if a not in piv or b not in piv:
                continue
            for met in ("se_lat", "brier", "cover_lat", "width"):
                d = piv[a].loc[keys, met].to_numpy() - piv[b].loc[keys, met].to_numpy()
                bs = d[draws].mean(axis=1)
                rows.append({"world": world, "mechanism": mech, "contrast": f"{a}-{b}", "metric": met, "mean": float(d.mean()),
                             "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5)), "authors": len(d)})
    pd.DataFrame(rows).to_csv(OUT / "paired_contrasts.csv", index=False)
    pd.DataFrame(cover_rows).to_csv(OUT / "coverage_intervals.csv", index=False)
    homogeneous_and_tau_checks()
    val = pd.read_csv(OUT / "source_validation_calibration.csv")
    vs = val.groupby("world")[[c for c in val.columns if c.endswith("_brier")]].mean().reset_index()
    vs.to_csv(OUT / "source_validation_summary.csv", index=False)
    files = ["author_metrics.csv", "summary.csv", "paired_contrasts.csv", "coverage_intervals.csv", "source_validation_summary.csv",
             "source_side.json", "homogeneous_reduction.json", "tau2_truth_vs_source.csv"]
    (OUT / "analysis.json").write_text(json.dumps({"output_sha256": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files},
                                                   "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
