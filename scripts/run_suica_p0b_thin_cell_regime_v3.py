#!/usr/bin/env python
"""OP-10b / P0-B: estimator validation in the THIN-CELL regime.

Round-4 audit proved the MoM decomposition is blind to interaction (state)
shares below ~0.05 on E5/E7-like layouts (2-8 slices per cell, residual ~0.9)
and that the original P0 harness never probed this regime. This harness
closes that gap at the statistical layer (synthetic scores; the text->score
layer is already P0-validated).

Worlds: planted state share s in {0.00, 0.02, 0.05, 0.10}, trait 0.10,
residual = remainder; 300 users x ~10 occasions x 2-8 slices; 4 reps each.
Estimators: MoM (mom_from_cells), MixedLM (REML, user + user:occasion VCs,
150-user subsample), and the estimator-free contiguous split-half detector.

Pre-committed criteria:
  C1 (MoM boundary documented): mean MoM state estimate <= 0.005 at s=0.02
     (blindness reproduced) and underestimation at s=0.05.
  C2 (MixedLM validity): mean MixedLM state estimate within +-40% relative
     at s=0.05 and s=0.10.
  C3 (false positives): at s=0, MixedLM state estimates stay <= 0.02 and
     split-half detector flags no state (CI includes 0) in >= 3/4 reps.
  P0B-verdict pass = C1 & C2 & C3 -> MixedLM certified as the headline
  estimator for thin-cell decompositions; MoM demoted to screening.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.suica_v2_lib import cell_table, mom_from_cells  # noqa: E402

OUT_DIR = ROOT / "results" / "suica_p0b_thin_cell_regime_v3"
REPORT = ROOT / "reports" / "suica_p0b_thin_cell_regime_v3.md"
SEED = 42
N_USERS = 300
MIXEDLM_USERS = 150
STATE_SHARES = [0.0, 0.02, 0.05, 0.10]
TRAIT = 0.10
REPS = 4


def gen_world(rng: np.random.Generator, state: float) -> pd.DataFrame:
    resid = 1.0 - TRAIT - state
    rows = []
    for u in range(N_USERS):
        b = rng.normal(0, np.sqrt(TRAIT))
        n_occ = max(6, rng.poisson(10))
        for o in range(n_occ):
            s = rng.normal(0, np.sqrt(state)) if state > 0 else 0.0
            n_slices = int(rng.integers(2, 9))
            for i in range(n_slices):
                rows.append({"user_id": f"u{u:04d}", "condition": f"o{o:02d}", "slice_i": i,
                             "y": b + s + rng.normal(0, np.sqrt(resid))})
    return pd.DataFrame(rows)


def mixedlm_state(data: pd.DataFrame) -> float:
    """Multi-start REML with best-restricted-loglik selection (round-6 audit fix).

    lbfgs alone collapses to the vc=0 boundary in 2/4 thin-cell worlds (one
    with a falsely True converged flag); selecting the max-llf fit across
    optimizers recovers the planted component.
    """
    import statsmodels.formula.api as smf
    d = data.loc[data["user_id"].isin(sorted(data["user_id"].unique())[:MIXEDLM_USERS])].copy()
    d["cell"] = d["user_id"] + ":" + d["condition"]
    d["g"] = 1
    md = smf.mixedlm("y ~ 1", d, groups="g", vc_formula={"u": "0 + C(user_id)", "uc": "0 + C(cell)"})
    best = None
    for method in ("bfgs", "lbfgs", "cg"):
        try:
            fit = md.fit(reml=True, method=method, maxiter=500)
        except Exception:
            continue
        llf = float(fit.llf) if np.isfinite(fit.llf) else -np.inf
        if best is None or llf > best[0]:
            best = (llf, fit)
    if best is None:
        return float("nan")
    fit = best[1]
    names = list(md.exog_vc.names)
    vcs = dict(zip(names, [float(v) for v in np.atleast_1d(fit.vcomp)]))
    total = sum(vcs.values()) + float(fit.scale)
    return vcs.get("uc", np.nan) / total


def split_half_state(data: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    d = data.assign(par=data["slice_i"] % 2)
    cell = d.groupby(["user_id", "condition", "par"])["y"].agg(["mean", "size"]).reset_index()
    wide = cell.pivot_table(index=["user_id", "condition"], columns="par", values="mean").dropna()
    for col in (0, 1):
        wide[col] = wide[col] - wide.groupby(level=0)[col].transform("mean")
    r = float(np.corrcoef(wide[0], wide[1])[0, 1])
    idx = np.arange(len(wide))
    boots = []
    for _ in range(300):
        t = rng.choice(idx, size=len(idx), replace=True)
        a, b = wide[0].to_numpy()[t], wide[1].to_numpy()[t]
        if a.std() > 1e-12 and b.std() > 1e-12:
            boots.append(np.corrcoef(a, b)[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return r, float(lo), float(hi)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for state in STATE_SHARES:
        for rep in range(REPS):
            rng = np.random.default_rng(1000 * int(state * 100) + rep)
            data = gen_world(rng, state)
            mom = mom_from_cells(cell_table(data.rename(columns={"y": "v"}), "v"))
            try:
                mlm = mixedlm_state(data)
            except Exception:
                mlm = np.nan
            sh_r, sh_lo, sh_hi = split_half_state(data, rng)
            rows.append({"planted_state": state, "rep": rep,
                         "mom_state": mom["interaction_share"], "mixedlm_state": mlm,
                         "split_half_r": sh_r, "split_half_ci_lo": sh_lo, "split_half_ci_hi": sh_hi})
            print(f"s={state} rep={rep}: mom={mom['interaction_share']:.4f} mixedlm={mlm:.4f} split={sh_r:.3f}")
    out = pd.DataFrame(rows)
    g = out.groupby("planted_state").mean(numeric_only=True)
    c1 = bool(g.loc[0.02, "mom_state"] <= 0.005 and g.loc[0.05, "mom_state"] < 0.05 * 0.7)
    c2 = bool(abs(g.loc[0.05, "mixedlm_state"] - 0.05) <= 0.02 and abs(g.loc[0.10, "mixedlm_state"] - 0.10) <= 0.04)
    zero = out.loc[out["planted_state"].eq(0.0)]
    c3 = bool((zero["mixedlm_state"] <= 0.02).all() and (zero["split_half_ci_lo"] <= 0).sum() >= 3)
    criteria = {"C1_mom_boundary_documented": c1, "C2_mixedlm_valid": c2, "C3_false_positive_control": c3,
                "P0B_verdict": "pass" if (c1 and c2 and c3) else "fail",
                "mean_by_share": {str(k): {"mom": float(v["mom_state"]), "mixedlm": float(v["mixedlm_state"])}
                                  for k, v in g.iterrows()}}
    out.to_csv(OUT_DIR / "p0b_thin_cell.csv", index=False)
    (OUT_DIR / "p0b_results.json").write_text(json.dumps(criteria, indent=2, default=float) + "\n")
    REPORT.write_text("# GRAIN P0-B Thin-Cell Estimator Regime (OP-10b)\n\n"
                      + out.round(4).to_markdown(index=False)
                      + "\n\n```json\n" + json.dumps(criteria, indent=2, default=float) + "\n```\n")
    print(json.dumps(criteria, indent=2, default=float))


if __name__ == "__main__":
    main()
