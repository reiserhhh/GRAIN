#!/usr/bin/env python
"""Declared analysis for the counterfactual reader packet (docs/GRAIN_COUNTERFACTUAL_READER_PACKET_V1.md + A1).

Every quantity is a per-unit difference in reader score, aggregated per corpus and pooled with a unit bootstrap
(2,000 draws, seed 20260926; one unit per author, so this is an author bootstrap). Pooled draws are stratified by
corpus. No ratios are taken to small totals.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/grain_gptpro_foundation_v3/R14/counterfactual_reader_r1"
DRAWS, SEED = 2000, 20260926


def per_unit(rows: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"key": rows.key, "corpus": rows.corpus, "set": rows["set"]})
    for r in ("qwen", "nli"):
        d[f"{r}_E+"] = rows[f"{r}_pos"] - rows[f"{r}_neu"]
        d[f"{r}_E-"] = rows[f"{r}_neg"] - rows[f"{r}_neu"]
        d[f"{r}_own(other-own)"] = rows[f"{r}_oth"] - rows[f"{r}_pos"]
        d[f"{r}_neutral-orig"] = rows[f"{r}_neu"] - rows[f"{r}_orig"]
    for dn, lab in (("pos", "E+"), ("neg", "E-")):
        d[f"T_H0_{lab}"] = rows[f"patch_{dn}_h0"] - rows["qwen_neu"]
        d[f"T_H6_{lab}"] = rows[f"patch_{dn}_hc"] - rows["qwen_neu"]
        d[f"T_H0-T_H6_{lab}"] = d[f"T_H0_{lab}"] - d[f"T_H6_{lab}"]
        d[f"placebo_{lab}"] = rows[f"placebo_{dn}"] - rows["qwen_neu"]
    d["self_patch"] = rows["self_patch"] - rows["qwen_neu"]
    return d


def boot(d: pd.DataFrame, cols: list[str], rng) -> pd.DataFrame:
    out = []
    groups = {"pandora": d[d.corpus == "pandora"], "essays": d[d.corpus == "essays"]}
    idx = {g: np.arange(len(x)) for g, x in groups.items()}
    draws = {g: rng.integers(0, len(x), size=(DRAWS, len(x))) for g, x in groups.items() if len(x)}
    for col in cols:
        for g, x in groups.items():
            if not len(x):
                continue
            v = x[col].to_numpy(float)
            b = v[draws[g]].mean(axis=1)
            out.append({"quantity": col, "scope": g, "n": len(v), "mean": float(v.mean()), "ci_low": float(np.percentile(b, 2.5)),
                        "ci_high": float(np.percentile(b, 97.5)), "mean_abs": float(np.abs(v).mean()), "n_positive": int((v > 0).sum())})
        vs = [groups[g][col].to_numpy(float) for g in groups if len(groups[g])]
        pooled = np.concatenate(vs)
        bp = np.concatenate([groups[g][col].to_numpy(float)[draws[g]] for g in groups if len(groups[g])], axis=1).mean(axis=1)
        out.append({"quantity": col, "scope": "pooled", "n": len(pooled), "mean": float(pooled.mean()), "ci_low": float(np.percentile(bp, 2.5)),
                    "ci_high": float(np.percentile(bp, 97.5)), "mean_abs": float(np.abs(pooled).mean()), "n_positive": int((pooled > 0).sum())})
    return pd.DataFrame(out)


def main() -> int:
    rows = pd.DataFrame([json.loads(x) for x in (OUT / "results.jsonl").read_text().splitlines()])
    d = per_unit(rows)
    d.to_csv(OUT / "per_unit_effects.csv", index=False)
    cols = [c for c in d.columns if c not in ("key", "corpus", "set")]
    rng = np.random.default_rng(SEED)
    summary = boot(d, cols, rng)
    summary.to_csv(OUT / "effects_summary.csv", index=False)
    infl = []
    for reader in ("qwen", "nli"):
        for corpus, g in rows.groupby("corpus"):
            sd = float(g[f"{reader}_orig"].std(ddof=1))
            for q in ("E+", "E-", "own(other-own)"):
                v = d.loc[g.index, f"{reader}_{q}"].to_numpy(float)
                infl.append({"reader": reader, "corpus": corpus, "contrast": q, "orig_sd": sd,
                             "mean_over_sd": (float(v.mean() / sd) if sd >= 1e-6 else None),
                             "median_abs_over_sd": (float(np.median(np.abs(v)) / sd) if sd >= 1e-6 else None)})
    pd.DataFrame(infl).to_csv(OUT / "sentence_influence.csv", index=False)
    by_set = d.groupby(["corpus", "set"])[[c for c in cols if not c.startswith(("placebo", "self"))]].mean().reset_index()
    by_set.to_csv(OUT / "by_phrasing_set_descriptive.csv", index=False)
    mass = rows[[c for c in rows.columns if c.startswith("mass_")]].agg(["mean", "min"]).T.reset_index().rename(columns={"index": "condition"})
    mass.to_csv(OUT / "candidate_mass.csv", index=False)
    files = ["per_unit_effects.csv", "effects_summary.csv", "sentence_influence.csv", "by_phrasing_set_descriptive.csv", "candidate_mass.csv"]
    meta = {"units": int(len(rows)), "draws": DRAWS, "seed": SEED,
            "output_sha256": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files},
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "analysis.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
