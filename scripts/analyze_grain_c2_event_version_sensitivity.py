#!/usr/bin/env python
"""Declared analysis for C2 event-version sensitivities (protocol V1).

It reuses r1's bootstrap exactly: seed 20260926, stratified by fold, a fresh RNG per condition, and the same author
universe order. Draws are therefore shared with r1's A/B/C, ArchA and LRNP. It first checks that r1's B−C is
reproduced exactly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r1a", REPO / "scripts/analyze_grain_c2_mcd_venue_dependence.py")
r1a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1a)
OUT = REPO / "results/grain_gptpro_foundation_v3/P05/c2_event_version_sensitivity_r1"
R1 = REPO / "results/grain_gptpro_foundation_v3/P05/c2_venue_dependence_r1"
TRAITS = r1a.TRAITS


def load_new() -> pd.DataFrame:
    rows = []
    for view, tag in (("first", "f"), ("last", "l")):
        for p in sorted(x for x in (OUT / f"cells_{view}").glob("*.json") if not x.name.startswith("._")):
            c = json.loads(p.read_text())
            d = pd.DataFrame(c["predictions"])
            d["condition"], d["trait"], d["fold"], d["arm"] = c["condition"], c["trait"], c["fold"], c["arm"][0] + tag
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def analyze(condition: str, preds: pd.DataFrame, systems: list[str], contrasts: dict[str, dict[str, float]]):
    refs = r1a.references(condition)
    allsys = systems + list(refs)
    tables = {}
    for trait in TRAITS:
        base = preds[(preds.condition == condition) & (preds.trait == trait) & (preds.arm == "A")][["author", "fold", "true"]].set_index("author")
        cols = {}
        for s in systems:
            a = preds[(preds.condition == condition) & (preds.trait == trait) & (preds.arm == s)].set_index("author")
            assert set(a.index) == set(base.index), (trait, s)
            assert np.allclose(a.loc[base.index, "true"], base["true"]), (trait, s)
            cols[s] = a.loc[base.index, "pred"].to_numpy(float)
        for name, ref in refs.items():
            r = ref[ref.trait == trait].set_index("author")
            assert set(r.index) == set(base.index) and np.allclose(r.loc[base.index, "true"], base["true"])
            assert (r.loc[base.index, "fold"].to_numpy(int) == base["fold"].to_numpy(int)).all()
            cols[name] = r.loc[base.index, "pred"].to_numpy(float)
        tables[trait] = (base.index.to_numpy(), base["fold"].to_numpy(int), base["true"].to_numpy(float), np.column_stack([cols[s] for s in allsys]))
    universe = sorted({(a, int(f)) for t in TRAITS for a, f in zip(tables[t][0], tables[t][1])})
    uni_author = np.array([a for a, _ in universe])
    uni_fold = np.array([f for _, f in universe])
    pos = {t: pd.Series(np.arange(len(tables[t][0])), index=tables[t][0]) for t in TRAITS}
    maps = {t: pos[t].reindex(uni_author).fillna(-1).to_numpy(int) for t in TRAITS}

    # r1's systems are evaluated in exactly r1's column block (A, B, C, then references) so that r1 numbers reproduce
    # bit-for-bit; the new systems form a second block. Bootstrap draws do not depend on the blocks.
    r1_cols = [allsys.index(x) for x in ["A", "B", "C", *refs]]
    new_cols = [i for i in range(len(allsys)) if i not in r1_cols]

    def pr(y, P):
        out = np.empty(P.shape[1])
        out[r1_cols] = r1a.pearson_rows(y, np.ascontiguousarray(P[:, r1_cols]))
        out[new_cols] = r1a.pearson_rows(y, np.ascontiguousarray(P[:, new_cols]))
        return out

    def estimands(uidx):
        pooled = np.zeros((len(TRAITS), len(allsys)))
        foldavg = np.zeros_like(pooled)
        for ti, t in enumerate(TRAITS):
            _, fo, y, P = tables[t]
            rws = maps[t][uidx]
            rws = rws[rws >= 0]
            pooled[ti] = pr(y[rws], P[rws])
            fr = [pr(y[rws[fo[rws] == f]], P[rws[fo[rws] == f]]) for f in range(5)]
            foldavg[ti] = np.nanmean(np.vstack(fr), axis=0)
        return pooled, foldavg

    pp, pf = estimands(np.arange(len(universe)))
    rng = np.random.default_rng(r1a.SEED)
    strata = [np.where(uni_fold == f)[0] for f in range(5)]
    bp = np.zeros((r1a.DRAWS, len(TRAITS), len(allsys)))
    bf = np.zeros_like(bp)
    for b in range(r1a.DRAWS):
        uidx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        bp[b], bf[b] = estimands(uidx)
    idx = {s: i for i, s in enumerate(allsys)}
    rows = []
    for est, point, boot in (("pooled_r", pp, bp), ("fold_average_r", pf, bf)):
        for s in allsys:
            v = boot[:, :, idx[s]].mean(axis=1)
            rows.append({"condition": condition, "estimand": est, "quantity": s, "trait": "MEAN_BIG5", "point": float(point[:, idx[s]].mean()),
                         "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)), "share_draws_gt0": float(np.mean(v > 0))})
        for name, coef in contrasts.items():
            if any(k not in idx for k in coef):
                continue
            dp = sum(c * point[:, idx[k]] for k, c in coef.items())
            db = sum(c * boot[:, :, idx[k]] for k, c in coef.items())
            for ti, t in enumerate(TRAITS + ["MEAN_BIG5"]):
                v = db.mean(axis=1) if t == "MEAN_BIG5" else db[:, ti]
                pt = float(dp.mean()) if t == "MEAN_BIG5" else float(dp[ti])
                rows.append({"condition": condition, "estimand": est, "quantity": name, "trait": t, "point": pt, "ci_low": float(np.percentile(v, 2.5)),
                             "ci_high": float(np.percentile(v, 97.5)), "share_draws_gt0": float(np.mean(v > 0))})
    return rows


def main() -> int:
    r1p = pd.read_csv(R1 / "predictions.csv", float_precision="round_trip")
    new = load_new()
    preds = pd.concat([r1p[["condition", "trait", "fold", "arm", "author", "true", "pred"]], new[["condition", "trait", "fold", "arm", "author", "true", "pred"]]], ignore_index=True)
    systems = ["A", "B", "C", "Af", "Bf", "Cf", "Al", "Bl", "Cl"]
    con = {"B-C": {"B": 1, "C": -1}, "A-C": {"A": 1, "C": -1}}
    for v in ("f", "l"):
        con.update({f"A'{v}-LRNP": {f"A{v}": 1, "LRNP": -1}, f"B'{v}-LRNP": {f"B{v}": 1, "LRNP": -1}, f"C'{v}-LRNP": {f"C{v}": 1, "LRNP": -1},
                    f"B'{v}-C'{v}": {f"B{v}": 1, f"C{v}": -1}, f"A'{v}-C'{v}": {f"A{v}": 1, f"C{v}": -1}, f"A'{v}-A": {f"A{v}": 1, "A": -1},
                    f"(B'{v}-C'{v})-(B-C)": {f"B{v}": 1, f"C{v}": -1, "B": -1, "C": 1}})
    con.update({"A'f-A'l": {"Af": 1, "Al": -1}, "(B'f-C'f)-(B'l-C'l)": {"Bf": 1, "Cf": -1, "Bl": -1, "Cl": 1}})
    rows = []
    for condition in ("np_mcd", "mcd_only"):
        rows += analyze(condition, preds, systems, con)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "bootstrap_summary.csv", index=False)
    r1s = pd.read_csv(R1 / "bootstrap_summary.csv", float_precision="round_trip")
    chk = {}
    for cond in ("np_mcd", "mcd_only"):
        for est in ("pooled_r", "fold_average_r"):
            a = r1s[(r1s.condition == cond) & (r1s.estimand == est) & (r1s.quantity == "B-C") & (r1s.trait == "MEAN_BIG5")].iloc[0]
            b = out[(out.condition == cond) & (out.estimand == est) & (out.quantity == "B-C") & (out.trait == "MEAN_BIG5")].iloc[0]
            chk[f"{cond}|{est}"] = {"point_diff": abs(a.point - b.point), "ci_low_diff": abs(a.ci_low - b.ci_low), "ci_high_diff": abs(a.ci_high - b.ci_high)}
    reproduced = all(max(v.values()) == 0.0 for v in chk.values())
    lr = {t: hashlib.sha256((r1a.BASELINE / "res" / f"{t}-LR-NP" / "preds.csv").read_bytes()).hexdigest() for t in TRAITS}
    meta = {"r1_B_minus_C_reproduced_exactly": reproduced, "r1_check": chk, "lrnp_pred_sha256": lr,
            "output_sha256": hashlib.sha256((OUT / "bootstrap_summary.csv").read_bytes()).hexdigest(),
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "analysis.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    return 0 if reproduced else 2


if __name__ == "__main__":
    raise SystemExit(main())
