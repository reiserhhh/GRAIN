#!/usr/bin/env python3
"""R12 dual-report: mean effect + reliability of unit-level effect differences (cached)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R12_DUAL_REPORT_PROTOCOL_V1.md"
PROTOCOL_SHA = "a3e7df8303b4ff8bf055849e9ec1cc3005492832ed4c61d4584dc28344ceeed6"
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "F04/dual_report_r1"
SEED, BOOT = 20260926, 2000


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def icc(groups):
    """groups: list of equal-length replicate arrays (one per unit)."""
    g = np.array(groups, float)
    n, k = g.shape
    grand = g.mean()
    msb = k * ((g.mean(1) - grand) ** 2).sum() / (n - 1)
    msw = ((g - g.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    icc1 = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else float("nan")
    icck = (msb - msw) / msb if msb > 0 else float("nan")
    return float(icc1), float(icck)


def report(name, groups, rng):
    g = np.array(groups, float)
    n = len(g)
    idx = rng.integers(0, n, (BOOT, n))
    means = g[idx].mean(axis=(1, 2))
    iccs = np.array([icc(g[i])[0] for i in idx])
    i1, ik = icc(g)
    lo = float(np.nanpercentile(iccs, 2.5))
    return {"effect": name, "units": n, "replicates": g.shape[1], "mean_effect": float(g.mean()),
            "mean_ci": [float(x) for x in np.percentile(means, [2.5, 97.5])],
            "icc1": i1, "icc1_ci": [lo, float(np.nanpercentile(iccs, 97.5))], "icc_k": ik,
            "individually_reliable": bool(lo > 0.10),
            "label": "individual + population" if lo > 0.10 else "population effect only"}


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    eff = pd.DataFrame([json.loads(x) for x in (OP / "native_effects.jsonl").read_text().splitlines()])
    eff["cell"] = eff.axis + "|" + eff.family + "|" + eff.situation
    for (q, kind), g in eff.groupby(["query", "kind"]):
        groups = [grp.sort_values("seed")["effect"].to_numpy() for _, grp in g.groupby("cell")]
        rows.append({"dataset": "F10_PV01_native", **report(f"{kind}_action|{q}", groups, rng)})
    for q, g in eff[eff.kind == "donor"].groupby("query"):
        groups = [grp.sort_values("seed")["contrast"].to_numpy() for _, grp in g.groupby("cell")]
        rows.append({"dataset": "F10_PV01_native", **report(f"full_text_contrast|{q}", groups, rng)})
    ie = pd.read_csv(V3 / "F11/reader_concordance_r1/intervention_effects.csv")
    for inter, g in ie.groupby("intervention"):
        for col in ("pole_readout", "grain_pole", "nli_qualification", "grain_qualification"):
            groups = [grp[col].to_numpy() for _, grp in g.groupby("template") if len(grp) == 2]
            if np.std(np.concatenate(groups)) < 1e-12:
                rows.append({"dataset": "F16_reader_interventions", "effect": f"{inter}|{col}",
                             "label": "ABSENT_CHANNEL", "mean_effect": float(np.mean(groups))})
                continue
            rows.append({"dataset": "F16_reader_interventions", **report(f"{inter}|{col}", groups, rng)})
    dif = pd.read_csv(V3 / "F12/dif_table_r1/dif_table.csv")
    zcols = [c for c in dif.columns if c.startswith("bias_z")]
    for (method, world), g in dif[dif.status == "OK"].groupby(["method", "world"]):
        if g.seed.nunique() < 2:
            continue
        groups = [g.sort_values("seed")[c].to_numpy() for c in zcols]
        rows.append({"dataset": "F12_DIF_conditions", **report(f"dif_bias|{method}|{world}", groups, rng)})
    sim = pd.read_csv(V3 / "F09/rival_person_simulation_r1/rival_person_metrics.csv")
    cited = sim[(sim.design == "D1")][["world", "coordinate", "mean_estimate", "retest_r"]].to_dict("records")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dual_report.csv", index=False)
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "rows": len(df),
           "cited_F09_D1_mean_and_retest": cited,
           "cited_F04": "F04/excitation_reliability_r2 reliability vs tokens and vs context diversity at fixed tokens",
           "not_estimable": {"R17_natural_edits": "one unit per author; no replicate"},
           "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "dual_report.json").write_text(json.dumps(out, indent=1) + "\n")
    cols = ["dataset", "effect", "units", "replicates", "mean_effect", "mean_ci", "icc1", "icc1_ci", "icc_k", "label"]
    print(df.reindex(columns=cols).to_string())


if __name__ == "__main__":
    main()
