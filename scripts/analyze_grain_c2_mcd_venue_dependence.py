#!/usr/bin/env python
"""Declared analysis for the C2 paired MCD venue-dependence packet (c2_v1).

Protocol: docs/GRAIN_C2_MCD_VENUE_DEPENDENCE_PROTOCOL_V1.md. It uses a paired author
bootstrap (2,000 draws, seed 20260926), resampling authors with replacement within each
fold's test set. The same draw is shared by all traits, arms and archived references.
The pooled-r and fold-average-r estimands are reported separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PERSONA = Path("/Volumes/mobile3/projects/project persona")
OUT = REPO / "results/grain_gptpro_foundation_v3/P05/c2_venue_dependence_r1"
ARCH = PERSONA / "results/pandora_baseline_mcd_integration"
BASELINE = PERSONA / "data_sets/PANDORA_official/pandora_baseline"
TRAITS = ["agreeableness", "openness", "conscientiousness", "neuroticism", "extraversion"]
ARMS = ["A", "B", "C"]
DRAWS = 2000
SEED = 20260926


def load_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds, metrics = [], []
    for path in sorted(p for p in (OUT / "cells").glob("*.json") if not p.name.startswith("._")):
        cell = json.loads(path.read_text())
        p = pd.DataFrame(cell["predictions"])
        for key in ("condition", "trait", "fold", "arm"):
            p[key] = cell[key]
        preds.append(p)
        metrics.append({k: cell[k] for k in ("condition", "trait", "fold", "arm", "pearson", "n_train", "n_val", "n_test",
                                              "total_features", "best_val_pearson", "best_alpha", "best_feat_n",
                                              "n_unobserved_test", "n_unobserved_train", "n_unobserved_val",
                                              "grid_n_iter_max", "grid_convergence_warnings", "cell_seconds", "mcd_fit_seconds")})
    return pd.concat(preds, ignore_index=True), pd.DataFrame(metrics)


def references(condition: str) -> dict[str, pd.DataFrame]:
    arch = pd.read_csv(ARCH / "predictions.csv")
    refs = {"ArchA": arch[arch.condition_key == condition][["trait", "fold", "author", "true", "pred"]]}
    if condition == "np_mcd":
        rows = []
        for trait in TRAITS:
            df = pd.read_csv(BASELINE / "res" / f"{trait}-LR-NP" / "preds.csv")
            df["trait"] = trait
            rows.append(df[["trait", "fold", "author", "true", "pred"]])
        refs["LRNP"] = pd.concat(rows, ignore_index=True)
    return refs


def pearson_rows(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Pearson r of y against each column of P (rows are items)."""
    yc = y - y.mean()
    Pc = P - P.mean(axis=0)
    num = yc @ Pc
    den = np.sqrt((yc @ yc) * np.sum(Pc * Pc, axis=0))
    return np.where(den > 1e-12, num / np.maximum(den, 1e-300), np.nan)


def analyze(condition: str, preds: pd.DataFrame) -> dict:
    sub = preds[preds.condition == condition]
    refs = references(condition)
    systems = ARMS + list(refs)
    # aligned per-trait tables: one row per author, columns = systems
    tables, folds_of = {}, {}
    for trait in TRAITS:
        base = sub[(sub.trait == trait) & (sub.arm == "A")][["author", "fold", "true"]].set_index("author")
        cols = {}
        for arm in ARMS:
            a = sub[(sub.trait == trait) & (sub.arm == arm)].set_index("author")
            assert set(a.index) == set(base.index), (trait, arm)
            assert np.allclose(a.loc[base.index, "true"], base["true"]), (trait, arm)
            cols[arm] = a.loc[base.index, "pred"].to_numpy(float)
        for name, ref in refs.items():
            r = ref[ref.trait == trait].set_index("author")
            assert set(r.index) == set(base.index), (trait, name, len(set(r.index) ^ set(base.index)))
            assert np.allclose(r.loc[base.index, "true"], base["true"]), (trait, name)
            assert (r.loc[base.index, "fold"].to_numpy(int) == base["fold"].to_numpy(int)).all(), (trait, name)
            cols[name] = r.loc[base.index, "pred"].to_numpy(float)
        tables[trait] = (base.index.to_numpy(), base["fold"].to_numpy(int), base["true"].to_numpy(float),
                         np.column_stack([cols[s] for s in systems]))
    # author universe per fold (union over traits) for a shared draw
    universe = sorted({(a, int(f)) for t in TRAITS for a, f in zip(tables[t][0], tables[t][1])})
    uni_author = np.array([a for a, _ in universe])
    uni_fold = np.array([f for _, f in universe])
    same_sets = all(set(tables[t][0]) == set(tables[TRAITS[0]][0]) for t in TRAITS)
    pos = {t: pd.Series(np.arange(len(tables[t][0])), index=tables[t][0]) for t in TRAITS}
    maps = {t: pos[t].reindex(uni_author).fillna(-1).to_numpy(int) for t in TRAITS}

    def estimands(uidx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pooled = np.zeros((len(TRAITS), len(systems)))
        foldavg = np.zeros((len(TRAITS), len(systems)))
        for ti, t in enumerate(TRAITS):
            _, fo, y, P = tables[t]
            rows = maps[t][uidx]
            rows = rows[rows >= 0]
            pooled[ti] = pearson_rows(y[rows], P[rows])
            fr = [pearson_rows(y[rows[fo[rows] == f]], P[rows[fo[rows] == f]]) for f in range(5)]
            foldavg[ti] = np.nanmean(np.vstack(fr), axis=0)
        return pooled, foldavg

    full_idx = np.arange(len(universe))
    point_pooled, point_fold = estimands(full_idx)
    rng = np.random.default_rng(SEED)
    strata = [np.where(uni_fold == f)[0] for f in range(5)]
    boot_pooled = np.zeros((DRAWS, len(TRAITS), len(systems)))
    boot_fold = np.zeros_like(boot_pooled)
    for b in range(DRAWS):
        uidx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        boot_pooled[b], boot_fold[b] = estimands(uidx)
    sidx = {s: i for i, s in enumerate(systems)}
    contrasts = [("A", "B"), ("A", "C"), ("B", "C"), ("A", "ArchA")]
    if "LRNP" in sidx:
        contrasts = [("A", "LRNP"), ("B", "LRNP"), ("C", "LRNP"), ("ArchA", "LRNP")] + contrasts
    rows = []
    for est_name, point, boot in (("pooled_r", point_pooled, boot_pooled), ("fold_average_r", point_fold, boot_fold)):
        for s in systems:
            v = boot[:, :, sidx[s]].mean(axis=1)
            rows.append({"condition": condition, "estimand": est_name, "quantity": s, "trait": "MEAN_BIG5",
                         "point": float(point[:, sidx[s]].mean()), "ci_low": float(np.percentile(v, 2.5)),
                         "ci_high": float(np.percentile(v, 97.5)), "share_draws_gt0": float(np.mean(v > 0))})
        for x, y in contrasts:
            d_boot = boot[:, :, sidx[x]] - boot[:, :, sidx[y]]
            d_point = point[:, sidx[x]] - point[:, sidx[y]]
            for ti, t in enumerate(TRAITS + ["MEAN_BIG5"]):
                v = d_boot.mean(axis=1) if t == "MEAN_BIG5" else d_boot[:, ti]
                p = float(d_point.mean()) if t == "MEAN_BIG5" else float(d_point[ti])
                rows.append({"condition": condition, "estimand": est_name, "quantity": f"{x}-{y}", "trait": t, "point": p,
                             "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)),
                             "share_draws_gt0": float(np.mean(v > 0))})
    per_trait = []
    for est_name, point in (("pooled_r", point_pooled), ("fold_average_r", point_fold)):
        for ti, t in enumerate(TRAITS):
            per_trait.append({"condition": condition, "estimand": est_name, "trait": t,
                              **{s: float(point[ti, sidx[s]]) for s in systems}})
    return {"summary": rows, "per_trait": per_trait, "n_authors_universe": len(universe), "same_author_sets_across_traits": same_sets,
            "systems": systems}


def main() -> int:
    preds, metrics = load_cells()
    mask = pd.read_csv(OUT / "completion_mask.csv")
    preds.to_csv(OUT / "predictions.csv", index=False)
    metrics.sort_values(["condition", "trait", "fold", "arm"]).to_csv(OUT / "metrics_by_fold.csv", index=False)
    out = {"completion": {c: {"complete": int(g["complete"].sum()), "total": int(len(g))} for c, g in mask.groupby("condition")}}
    summary, per_trait = [], []
    for condition in ("np_mcd", "mcd_only"):
        m = mask[mask.condition == condition]
        if not m["complete"].all():
            out[condition] = "INCOMPLETE: not analyzed (no partial estimand silently reported)"
            continue
        res = analyze(condition, preds)
        summary += res["summary"]
        per_trait += res["per_trait"]
        out[condition] = {k: res[k] for k in ("n_authors_universe", "same_author_sets_across_traits", "systems")}
    pd.DataFrame(summary).to_csv(OUT / "bootstrap_summary.csv", index=False)
    pd.DataFrame(per_trait).to_csv(OUT / "per_trait_point_estimates.csv", index=False)
    sel = metrics.groupby(["condition", "arm"]).agg(
        cells=("pearson", "size"), mean_fold_r=("pearson", "mean"), max_n_iter=("grid_n_iter_max", "max"),
        conv_warnings=("grid_convergence_warnings", "sum"), unobserved_test=("n_unobserved_test", "sum"),
        cell_seconds=("cell_seconds", "sum")).reset_index()
    sel.to_csv(OUT / "arm_cell_summary.csv", index=False)
    files = ["predictions.csv", "metrics_by_fold.csv", "bootstrap_summary.csv", "per_trait_point_estimates.csv", "arm_cell_summary.csv"]
    out["output_sha256"] = {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files}
    out["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out["draws"], out["seed"] = DRAWS, SEED
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
