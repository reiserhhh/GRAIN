#!/usr/bin/env python
"""R13 packet 30 B runner (docs/GRAIN_R13_FREEDOM_COMPONENTS_PROTOCOL_V1.md). Declared-world simulation; 0 neural calls.
The generator (observe) and the audit (truth_targets / reference) are the ONLY places that touch generating values;
estimation uses y and z only."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r13_freedom_components as m  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R13_FREEDOM_COMPONENTS_PROTOCOL_V1.md"
PROTOCOL_SHA = "ba674296fc1f4d783abd1cb1f42f1b3973bd2c285db9e01957fd39e71d67c723"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F07/freedom_components_r1"
nA, nC, nO, nR = 200, 6, 2, 3
Z = np.linspace(-1.0, 1.0, nC)
SEEDS, BOOT = (0, 1, 2), 300
MECH = {"M1": 0.3, "M2": 0.0}
REGIME = {"LOW": (0.10, 1.0), "HIGH": (0.40, 3.0)}
VB, SR2, SAR2, SACR2 = 0.25, 0.05, 0.02, 0.02
CONTRASTS = ((1.0, 0.0), (0.5, 0.0), (1.0, 0.5))


def lam(f): return 0.6 + 0.4 * f
def kap(f): return 0.8 * f
def gam(f): return 1.0 - 0.6 * f
def sW2(f): return 0.10 + 0.20 * f
def sO2(f): return 0.20 + 0.30 * f
def se2(f, reg): b, h = REGIME[reg]; return b * (1 + h * f)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def base_draws(seed):
    rng = np.random.default_rng(np.random.SeedSequence([20260927, 30, seed]))
    return {"theta": rng.standard_normal(nA), "s0": rng.standard_normal(nA), "beta0": rng.standard_normal(nA), "w": rng.standard_normal((nA, nC)),
            "rr": rng.standard_normal(nR), "u": rng.standard_normal((nA, nR)), "v": rng.standard_normal((nA, nC, nR)), "o": rng.standard_normal((nA, nC, nO)),
            "e": rng.standard_normal((nA, nC, nO, nR))}


def latent(B, mech):
    rho = MECH[mech]
    return B["theta"], rho * B["theta"] + np.sqrt(1 - rho ** 2) * B["s0"], np.sqrt(VB) * B["beta0"]


def observe(B, f, mech, reg):
    """Generator: returns observations only."""
    th, s, beta = latent(B, mech)
    y = (0.5 * Z[None, :, None, None] + (lam(f) * th + kap(f) * s)[:, None, None, None] + (gam(f) * beta)[:, None, None, None] * Z[None, :, None, None]
         + np.sqrt(sW2(f)) * B["w"][:, :, None, None] + np.sqrt(SR2) * B["rr"][None, None, None, :] + np.sqrt(SAR2) * B["u"][:, None, None, :]
         + np.sqrt(SACR2) * B["v"][:, :, None, :] + np.sqrt(sO2(f)) * B["o"][:, :, :, None] + np.sqrt(se2(f, reg)) * B["e"])
    return y


def truth_targets(B, mech):
    """AUDIT ONLY."""
    th, s, beta = latent(B, mech)
    return {"T1": th, "T2": beta, "T3": th + s}


def true_components(f, mech, reg):
    """AUDIT ONLY: generating (population) variances in the estimator's parametrisation."""
    rho = MECH[mech]
    return {"e": se2(f, reg), "O": sO2(f), "ACR": SACR2, "W": sW2(f), "SLOPE": gam(f) ** 2 * VB, "AR": SAR2, "CR": 0.0,
            "A": lam(f) ** 2 + kap(f) ** 2 + 2 * lam(f) * kap(f) * rho, "R": SR2}


def reference(f, mech, reg):
    """AUDIT ONLY: analytic repeatability, fidelity and aligned variance under the declared model."""
    rho, tc = MECH[mech], true_components(f, mech, reg)
    el = sum(w * tc[c] for c, w in m.error_weights_level((nA, nC, nO, nR)).items())
    es = sum(w * tc[c] for c, w in m.error_weights_slope((nA, nC, nO, nR), float(Z @ Z)).items())
    vx, vb = tc["A"] + el, tc["SLOPE"] + es
    c1 = lam(f) + kap(f) * rho; c3 = c1 + lam(f) * rho + kap(f)
    a1, f1 = m.aligned(c1, 1.0, vx); a3, f3 = m.aligned(c3, 2 + 2 * rho, vx); a2, f2 = m.aligned(gam(f) * VB, VB, vb)
    naive3 = (lam(f) ** 2 + kap(f) ** 2) / vx     # FU2-b style "sum of component variances in the target" (for contrast only)
    return {"G_level_ref": tc["A"] / vx, "G_slope_ref": tc["SLOPE"] / vb, "fid_T1_ref": f1, "fid_T2_ref": f2, "fid_T3_ref": f3,
            "aligned_T1_ref": a1, "aligned_T2_ref": a2, "aligned_T3_ref": a3, "naive_sum_T3_ref": naive3}


def recovery(x, b, T):
    """AUDIT ONLY: empirical squared correlation of observed scores with truth."""
    c = lambda p, q: float(np.corrcoef(p, q)[0, 1] ** 2)
    return {"rec_T1": c(x, T["T1"]), "rec_T2": c(b, T["T2"]), "rec_T3": c(x, T["T3"])}


def closed_form(ms):
    """Independent check: explicit MS-difference formulas (not the EMS solve)."""
    M = ms["MS"]; Szz = ms["Szz"]
    return {"e": M["e"], "O": (M["O"] - M["e"]) / nR, "ACR": (M["ACR"] - M["e"]) / nO, "W": (M["ACrem"] - M["ACR"] - M["O"] + M["e"]) / (nO * nR),
            "SLOPE": (M["Alin"] - M["ACrem"]) / (nO * nR * Szz), "AR": (M["AR"] - M["ACR"]) / (nC * nO), "A": (M["A"] - M["ACrem"] - M["AR"] + M["ACR"]) / (nC * nO * nR)}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("freedom_components_r1 exists")
    OUT.mkdir(parents=True)
    comp_rows, ident_rows, tr_rows, con_rows, checks = [], [], [], [], {"ss_identity_max": 0.0, "closed_form_max_abs": 0.0, "g_recompute_max_abs": 0.0}
    draws = {}
    ystore = {}
    for seed in SEEDS:
        B = base_draws(seed)
        idx = np.random.default_rng(np.random.SeedSequence([20260927, 30, seed, 7])).integers(0, nA, (BOOT, nA))
        for mech in MECH:
            T = truth_targets(B, mech)
            for reg in REGIME:
                for f in m.LEVELS:
                    y = observe(B, f, mech, reg)
                    ystore[f"s{seed}_{mech}_{reg}_f{f:.2f}"] = y.astype(np.float32)
                    fit = m.estimate(y, Z); G = m.g_coefficients(fit)
                    checks["ss_identity_max"] = max(checks["ss_identity_max"], fit["ms"]["ss_identity_error"])
                    cf = closed_form(fit["ms"])
                    checks["closed_form_max_abs"] = max(checks["closed_form_max_abs"], max(abs(cf[c] - fit["components"][c]) for c in cf))
                    el = sum(w * fit["components"][c] for c, w in m.error_weights_level(fit["ms"]["shape"]).items())
                    checks["g_recompute_max_abs"] = max(checks["g_recompute_max_abs"], abs(fit["components"]["A"] / (fit["components"]["A"] + el) - G["G_level"]))
                    neg = m.estimate(y[:, :, :1, :], Z); Gn = m.g_coefficients(neg)
                    x, b = m.scores(y, Z); rec = recovery(x, b, T); ref = reference(f, mech, reg); tc = true_components(f, mech, reg)
                    # bootstrap (paired: same idx across levels)
                    bc = np.zeros((BOOT, len(m.COMPONENTS))); bg = np.zeros((BOOT, 2)); br = np.zeros((BOOT, 3))
                    for k in range(BOOT):
                        ii = idx[k]; fb = m.estimate(y[ii], Z); gb = m.g_coefficients(fb)
                        bc[k] = [fb["components"][c] for c in m.COMPONENTS]; bg[k] = [gb["G_level"], gb["G_slope"]]
                        xb, bb = m.scores(y[ii], Z); rb = recovery(xb, bb, {t: v[ii] for t, v in T.items()})
                        br[k] = [rb["rec_T1"], rb["rec_T2"], rb["rec_T3"]]
                    draws[(seed, mech, reg, f)] = (bc, bg, br)
                    for j, c in enumerate(m.COMPONENTS):
                        lo, hi = np.percentile(bc[:, j], [2.5, 97.5])
                        comp_rows.append({"seed": seed, "mechanism": mech, "reader_regime": reg, "freedom": f, "component": c, "estimate": fit["components"][c],
                                          "flag": fit["flags"][c], "truth_generating": tc[c], "boot_lo": lo, "boot_hi": hi, "boot_negative_share": float((bc[:, j] < 0).mean())})
                    for design, ft, gg in (("nO=2 (main)", fit, G), ("nO=1 (aliased negative control)", neg, Gn)):
                        for c in m.COMPONENTS:
                            ident_rows.append({"seed": seed, "mechanism": mech, "reader_regime": reg, "freedom": f, "design": design, "quantity": c,
                                               "identifiable": ft["flags"][c] != "UNIDENTIFIED", "flag": ft["flags"][c], "estimate": ft["components"][c]})
                        for q in ("G_level", "G_slope"):
                            ident_rows.append({"seed": seed, "mechanism": mech, "reader_regime": reg, "freedom": f, "design": design, "quantity": q,
                                               "identifiable": gg[q] is not None, "flag": "OK" if gg[q] is not None else "UNIDENTIFIED", "estimate": gg[q]})
                    row = {"seed": seed, "mechanism": mech, "reader_regime": reg, "freedom": f, "G_level_est": G["G_level"], "G_slope_est": G["G_slope"],
                           "G_level_lo": float(np.percentile(bg[:, 0], 2.5)), "G_level_hi": float(np.percentile(bg[:, 0], 97.5)),
                           "G_slope_lo": float(np.percentile(bg[:, 1], 2.5)), "G_slope_hi": float(np.percentile(bg[:, 1], 97.5)), **rec, **ref}
                    for t, j in (("T1", 0), ("T2", 1), ("T3", 2)):
                        row[f"rec_{t}_lo"], row[f"rec_{t}_hi"] = (float(v) for v in np.percentile(br[:, j], [2.5, 97.5]))
                    tr_rows.append(row)
    # ---- paired contrasts (same author indices across levels within a seed)
    qn = {"G_level": ("g", 0), "G_slope": ("g", 1), "rec_T1": ("r", 0), "rec_T2": ("r", 1), "rec_T3": ("r", 2),
          "comp_A": ("c", m.COMPONENTS.index("A")), "comp_W": ("c", m.COMPONENTS.index("W")), "comp_SLOPE": ("c", m.COMPONENTS.index("SLOPE")), "comp_e": ("c", m.COMPONENTS.index("e"))}
    TR = pd.DataFrame(tr_rows); CP = pd.DataFrame(comp_rows)
    for seed in SEEDS:
        for mech in MECH:
            for reg in REGIME:
                for f1, f0 in CONTRASTS:
                    d1, d0 = draws[(seed, mech, reg, f1)], draws[(seed, mech, reg, f0)]
                    for q, (kind, j) in qn.items():
                        a1 = {"c": d1[0], "g": d1[1], "r": d1[2]}[kind][:, j]; a0 = {"c": d0[0], "g": d0[1], "r": d0[2]}[kind][:, j]
                        if q.startswith("comp_"):
                            p1 = CP[(CP.seed == seed) & (CP.mechanism == mech) & (CP.reader_regime == reg) & (CP.freedom == f1) & (CP.component == q[5:])].estimate.iloc[0]
                            p0 = CP[(CP.seed == seed) & (CP.mechanism == mech) & (CP.reader_regime == reg) & (CP.freedom == f0) & (CP.component == q[5:])].estimate.iloc[0]
                        else:
                            col = q + "_est" if q.startswith("G_") else q
                            p1 = TR[(TR.seed == seed) & (TR.mechanism == mech) & (TR.reader_regime == reg) & (TR.freedom == f1)][col].iloc[0]
                            p0 = TR[(TR.seed == seed) & (TR.mechanism == mech) & (TR.reader_regime == reg) & (TR.freedom == f0)][col].iloc[0]
                        lo, hi = np.percentile(a1 - a0, [2.5, 97.5])
                        con_rows.append({"seed": seed, "mechanism": mech, "reader_regime": reg, "contrast": f"{f1:g}-{f0:g}", "quantity": q, "point": float(p1 - p0),
                                         "ci_lo": float(lo), "ci_hi": float(hi), "ci_contains_0": bool(lo <= 0 <= hi)})
    CN = pd.DataFrame(con_rows)
    CP.to_csv(OUT / "component_estimates.csv", index=False); TR.to_csv(OUT / "target_recovery.csv", index=False); CN.to_csv(OUT / "paired_contrasts.csv", index=False)
    pd.DataFrame(ident_rows).to_csv(OUT / "identification_checks.csv", index=False)
    np.savez_compressed(OUT / "simulated_observations.npz", z=Z, **ystore)
    # ---- pooled summaries and construction decision (declared rule: sign disagreement across targets for the same comparison)
    pooled = CN.groupby(["mechanism", "reader_regime", "contrast", "quantity"]).agg(mean_point=("point", "mean"), min_point=("point", "min"), max_point=("point", "max"),
                                                                                  seeds_ci_excl_0=("ci_contains_0", lambda s: int((~s).sum()))).reset_index()
    pooled.to_csv(OUT / "paired_contrasts_pooled.csv", index=False)
    decisions = []
    for (mech, reg, con), g in pooled.groupby(["mechanism", "reader_regime", "contrast"]):
        sg = {q: float(g[g.quantity == q].mean_point.iloc[0]) for q in ("rec_T1", "rec_T2", "rec_T3", "G_level", "G_slope")}
        signs = {q: int(np.sign(round(v, 4))) for q, v in sg.items() if q.startswith("rec_")}
        decisions.append({"mechanism": mech, "reader_regime": reg, "contrast": con, **{f"mean_{k}": v for k, v in sg.items()},
                          "target_signs_differ": len(set(signs.values()) - {0}) > 1})
    neg = pd.DataFrame(ident_rows); negc = neg[neg.design.str.startswith("nO=1")]
    checks["negative_control_unidentified_all"] = bool(negc[negc.quantity.isin(["e", "O", "ACR", "W"])].identifiable.eq(False).all())
    checks["negative_control_never_zero"] = bool(negc[negc.quantity.isin(["e", "O", "ACR", "W"])].estimate.isna().all())
    checks["main_all_identifiable"] = bool(neg[neg.design.str.startswith("nO=2")].identifiable.all())
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "cells": len(TR), "fits_point": len(TR) * 2, "fits_bootstrap": len(TR) * BOOT,
           "observations_per_cell": nA * nC * nO * nR, "checks": checks, "decisions": decisions, "seconds": time.time() - t0,
           "created": datetime.now(timezone.utc).isoformat(), "conditioning": "author-cluster bootstrap; conditions, readers and occasions fixed; declared world"}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/r13_freedom_components.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print(json.dumps({"checks": checks, "seconds": res["seconds"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
