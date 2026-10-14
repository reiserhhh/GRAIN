#!/usr/bin/env python
"""R01 packet 27: common donor shift versus text-specific vector action (docs/GRAIN_R01_SOURCE_AFFINE_PROTOCOL_V1.md).
Cached numeric arrays only; 0 neural calls; four fixed models (ZERO / COMMON / LINEAR / AFFINE); no search.
T_ij are cached directional derivatives, NOT new persons, finite native patch outcomes or labels.
Usage: run_grain_r01_source_affine.py            (main run; refuses an existing output root)
       run_grain_r01_source_affine.py --cold-replay   (separate process; rescoring from saved coefficients only)"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r01_source_affine as a  # noqa: E402
from grain_research import r01_source_crossed as x  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R01_SOURCE_AFFINE_PROTOCOL_V1.md"
PROTOCOL_SHA = "070039eed61781be08ba06db76bd78af3f2845a4a2ad4e39510ebc02fc29843e"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
NA, AJ, P24 = V3 / "P05/native_action_r1", V3 / "P05/application_j_diagnostic_r1", V3 / "P05/source_crossed_action_r1"
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
IDS = ROOT / "docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json"
OUT = V3 / "P05/source_affine_action_r1"
QUERIES, OFF = ("extraversion", "neuroticism"), {"extraversion": slice(0, 7552), "neuroticism": slice(7552, 15232)}
MODELS = ("ZERO", "COMMON", "LINEAR", "AFFINE")
CONTRASTS = (("AFFINE", "LINEAR"), ("AFFINE", "COMMON"), ("AFFINE", "ZERO"), ("LINEAR", "ZERO"), ("COMMON", "ZERO"), ("LINEAR", "COMMON"))
TREES = {"private_jacobians": "35e1b6d4d305a8e0aacf0f47827e42bf99921515d0fcf295c19c2e8ea649d751",
         "private_app_jacobians": "8601b303be50ad1b0279ab4d7aad89c831c25cd6401d703a27955bc36adb60cc"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_sha(d):
    fs = sorted(p for p in Path(d).glob("*.npy") if not p.name.startswith("._"))
    return hashlib.sha256("".join(f"{p.name}:{sha(p)}\n" for p in fs).encode()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")
    got = {"private_jacobians": tree_sha(NA / "private_jacobians"), "private_app_jacobians": tree_sha(AJ / "private_app_jacobians")}
    if got != TREES:
        raise RuntimeError("jacobian tree mismatch")
    return got


def cell_boot(per_cell_a, per_cell_b, seed=20260927, n=2000):
    """Packet-24 paired 12-cell bootstrap scheme (same seed, same draws for every contrast); descriptive."""
    d = np.asarray(per_cell_a) - np.asarray(per_cell_b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "ci_contains_0": bool(lo <= 0 <= hi)}


def load_panel():
    arr = np.load(OP / "action_arrays.npz"); prep = np.load(NA / "private_prepared.npz")
    order = json.loads((OP / "pair_order.json").read_text()); ids = json.loads(IDS.read_text()); fit = json.loads((NA / "fit.json").read_text())
    si = np.flatnonzero(arr["source"]); names = [order[i] for i in si]
    assert sorted(names) == sorted(ids["source_pairs"]) and len(si) == 36
    assert all(v == k.rsplit("|", 1)[0] for k, v in ids["cells"].items())
    cell_of = [nm.rsplit("|", 1)[0] for nm in names]; seed_of = [nm.split("|")[-1] for nm in names]
    cells = sorted(set(cell_of)); assert len(cells) == 12 and all(cell_of.count(c) == 3 for c in cells)
    ai = [order.index(p) for p in ids["application_pairs"]]
    return arr, prep, fit, si, names, cell_of, seed_of, cells, ai, ids["application_pairs"]


def cold_replay():
    """Separate process: application predictions from saved coefficients + declared numeric arrays only."""
    arr, prep, fit, si, names, cell_of, seed_of, cells, ai, app = load_panel()
    A = pd.read_csv(OUT / "private_application_predictions.csv")
    worst = 0.0
    for q in QUERIES:
        Ja = np.stack([np.load(AJ / "private_app_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in ai])
        for inp in ("4", "24"):
            co = np.load(OUT / f"private_coefficients_{q}_z{inp}.npz")
            s = float(co["s"]); Ua = np.sqrt(s) * Ja; Za = prep[f"z{inp}"][ai].astype(float)
            got = a.cold_score({k: co[k] for k in ("Q", "B_linear", "B_affine", "c_affine", "dbar")}, Za, Ua, s)
            g = A[(A["query"] == q) & (A["input"] == f"z{inp}")].set_index("pair").loc[app]
            ta = (Ua * arr["dh"][ai, OFF[q]].astype(float)).sum(1) / np.sqrt(s)
            worst = max(worst, float(np.abs(ta - g.truth.to_numpy()).max()))
            for m in MODELS:
                worst = max(worst, float(np.abs(got[m] - g[f"pred_{m}"].to_numpy()).max()))
    (OUT / "cold_replay.json").write_text(json.dumps({"max_abs": worst, "pid": os.getpid(), "inputs": "saved coefficients + prep z + app jacobians + dh"}) + "\n")


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    trees = verify()
    if OUT.exists():
        raise FileExistsError("source_affine_action_r1 exists; preserve earlier runs")
    arr, prep, fit, si, names, cell_of, seed_of, cells, ai, app = load_panel()
    P24pred = pd.read_csv(P24 / "private_heldcell_predictions.csv"); P24sup = pd.read_csv(P24 / "private_supplement_heldonor_mse.csv")
    R24 = json.loads((P24 / "results.json").read_text())
    OUT.mkdir(parents=True)
    gates = {"linear_heldcell_replay_max_abs": 0.0, "linear_supplement_replay_max_abs": 0.0, "linear_application_replay_max_abs": 0.0,
             "affine_vs_joint_fold0_max_abs": 0.0, "max_stationarity_grad": 0.0, "c_identity_max_abs": 0.0, "origin_invariance_affine_max_abs": 0.0,
             "orientation_identity_max_abs": 0.0, "projection_identity_max": 0.0, "T_equals_DQ_V_max_rel": 0.0, "centered_residual_identity_max": 0.0,
             "held_exclusion_ok": True}
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "trees": trees, "source_pairs": 36, "cells": 12, "application_pairs": 72,
           "labels": {"T": "cached directional derivatives d_i.u_j; dependent, NOT new persons / independent interventions / finite patch outcomes / labels",
                      "cv": "conditional on frozen source z and s transforms; not end-to-end", "intercept": "AFFINE intercept c is unpenalized (documented change)",
                      "application": "descriptive, one full-36 fit per model, derivative target J_a.d_a; no ranking", "bootstrap": "paired 12-cell, 2000 draws, seed 20260927; descriptive"},
           "source_diagnostics_full36": {}, "fold_diagnostics": {}, "orientation_full36": {}, "summary": {}, "application_descriptive": {}}
    pred_rows, sup_rows, app_rows, fold_rows = [], [], [], []
    for q in QUERIES:
        sl = OFF[q]
        Js = np.stack([np.load(NA / "private_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in si])
        s = float(fit["cells"][f"J24|{q}"]["metric_scale_s"]); assert abs(Js.shape[1] / np.mean((Js ** 2).sum(1)) - s) <= 1e-9 * s
        rs = np.sqrt(s); U = rs * Js; Dn = arr["dh"][si, sl].astype(float); T = Dn @ U.T
        Ja = np.stack([np.load(AJ / "private_app_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in ai]); Ua = rs * Ja
        Da = arr["dh"][ai, sl].astype(float); ta = (Ua * Da).sum(1) / rs
        for inp in ("4", "24"):
            Z = prep[f"z{inp}"][si].astype(float); Za = prep[f"z{inp}"][ai].astype(float)
            ps = a.projection_shares(Z, T); gates["projection_identity_max"] = max(gates["projection_identity_max"], ps["orthogonal_identity_error"])
            fold_coefs = {"B_linear": [], "B_affine": [], "c_affine": [], "dbar": []}
            fdiag = []
            for fi, hc in enumerate(cells):
                tr, h = a.split(cell_of, hc)
                ok = len(h) == 3 and len(tr) == 33 and not set(tr) & set(h) and all(cell_of[k] != hc for k in tr)
                gates["held_exclusion_ok"] = gates["held_exclusion_ok"] and ok
                n = len(tr); lam = 1.0 / n
                Zt, Ut, Dt = Z[tr], U[tr], Dn[tr]; Tt = T[np.ix_(tr, tr)]
                Q, V, _ = x.orth_basis(Ut); DQ = Dt @ Q
                gates["T_equals_DQ_V_max_rel"] = max(gates["T_equals_DQ_V_max_rel"], float(np.abs(DQ @ V.T - Tt).max() / np.abs(Tt).max()))
                Bl = a.fit_linear(Zt, V, Tt, lam); Ba, ca = a.fit_affine(Zt, V, Tt, lam); dbar = a.common_coef(DQ); zbar = Zt.mean(0)
                gates["c_identity_max_abs"] = max(gates["c_identity_max_abs"], float(np.abs(ca - (dbar - zbar @ Ba)).max() / max(1e-300, np.abs(dbar).max())))
                El = Zt @ Bl @ V.T - Tt; gl = float(np.abs((2 / n ** 2) * Zt.T @ El @ V + 2 * lam * Bl).max())
                scale = float(np.abs(Tt).max())
                gates["max_stationarity_grad"] = max(gates["max_stationarity_grad"], gl / scale, a.affine_grad(Zt, V, Tt, Ba, ca, lam) / scale)
                if fi == 0:
                    Bj, cj = a.fit_affine_joint(Zt, V, Tt, lam)
                    gates["affine_vs_joint_fold0_max_abs"] = max(gates["affine_vs_joint_fold0_max_abs"],
                                                                 float(np.abs((cj[None, :] + Zt @ Bj) @ V.T - (ca[None, :] + Zt @ Ba) @ V.T).max() / scale))
                Ba2, ca2 = a.fit_affine(Zt - zbar, V, Tt, lam); Bl2 = a.fit_linear(Zt - zbar, V, Tt, lam)
                for k_, v_ in (("B_linear", Bl), ("B_affine", Ba), ("c_affine", ca), ("dbar", dbar)):
                    fold_coefs[k_].append(v_)
                # training-only diagnostics
                fps = a.projection_shares(Zt, Tt); gates["projection_identity_max"] = max(gates["projection_identity_max"], fps["orthogonal_identity_error"])
                Pc = np.outer(np.ones(n), dbar @ V.T); Pl = Zt @ Bl @ V.T; Pa = (ca[None, :] + Zt @ Ba) @ V.T
                fdiag.append({"fold": fi, "held_cell": hc, **{k_: v_ for k_, v_ in fps.items() if k_ != "residual_singular_values_top5"},
                              "insample_rmse_COMMON": float(np.sqrt(np.mean((Pc - Tt) ** 2)) / rs), "insample_rmse_LINEAR": float(np.sqrt(np.mean((Pl - Tt) ** 2)) / rs),
                              "insample_rmse_AFFINE": float(np.sqrt(np.mean((Pa - Tt) ** 2)) / rs), "insample_rmse_ZERO": float(np.sqrt(np.mean(Tt ** 2)) / rs),
                              "cos_COMMON_LINEAR_pred": float(np.sum(Pc * Pl) / (np.linalg.norm(Pc) * np.linalg.norm(Pl))),
                              "zbar_norm": float(np.linalg.norm(zbar)), "zc_rms_norm": float(np.sqrt(np.mean(np.sum((Zt - zbar) ** 2, 1))))})
                lin_orig_change = []
                for k in h:
                    zh, uh = Z[k], U[k]; vh = Q.T @ uh
                    truth = T[k, k] / rs
                    pr = {"ZERO": 0.0, "COMMON": float(dbar @ vh) / rs, "LINEAR": float(zh @ Bl @ vh) / rs, "AFFINE": a.predict(zh, Ba, ca, vh) / rs}
                    gates["origin_invariance_affine_max_abs"] = max(gates["origin_invariance_affine_max_abs"], abs(a.predict(zh - zbar, Ba2, ca2, vh) / rs - pr["AFFINE"]) / (np.abs(np.diag(T)).max() / rs))
                    lin_orig_change.append(float((zh - zbar) @ Bl2 @ vh) / rs - pr["LINEAR"])
                    pred_rows.append({"query": q, "input": f"z{inp}", "fold": fi, "cell": hc, "pair": names[k], "truth": truth, **{f"pred_{m}": pr[m] for m in MODELS},
                                      "held_grad_share_in_train_span": float(vh @ vh / (uh @ uh))})
                    tj = T[k, tr] / rs
                    sp = {"ZERO": np.zeros(n), "COMMON": (V @ dbar) / rs, "LINEAR": (zh @ Bl @ V.T) / rs, "AFFINE": ((ca + zh @ Ba) @ V.T) / rs}
                    sup_rows.append({"query": q, "input": f"z{inp}", "cell": hc, "pair": names[k], **{f"mse_{m}": float(np.mean((sp[m] - tj) ** 2)) for m in MODELS},
                                     **{f"r_{m}": float(np.corrcoef(sp[m], tj)[0, 1]) for m in ("COMMON", "LINEAR", "AFFINE")}})
                fdiag[-1]["linear_origin_change_rms"] = float(np.sqrt(np.mean(np.square(lin_orig_change))))
            res["fold_diagnostics"][f"{q}|z{inp}"] = fdiag; fold_rows += [{"query": q, "input": f"z{inp}", **r_} for r_ in fdiag]
            # ---- full-36 source fit: diagnostics, orientation, coefficients, application (descriptive)
            Q36, V36, _ = x.orth_basis(U); lam36 = 1 / 36; DQ36 = Dn @ Q36
            Bl36 = a.fit_linear(Z, V36, T, lam36); Ba36, ca36 = a.fit_affine(Z, V36, T, lam36); dbar36 = a.common_coef(DQ36)
            Pc = np.outer(np.ones(36), dbar36 @ V36.T); Pl = Z @ Bl36 @ V36.T; Pa = (ca36[None, :] + Z @ Ba36) @ V36.T
            res["source_diagnostics_full36"][f"{q}|z{inp}"] = {**ps, "residual_singular_values_top5_native": [v_ / rs for v_ in ps["residual_singular_values_top5"]],
                                                               "insample_rmse": {"ZERO": float(np.sqrt(np.mean(T ** 2)) / rs), "COMMON": float(np.sqrt(np.mean((Pc - T) ** 2)) / rs),
                                                                                 "LINEAR": float(np.sqrt(np.mean((Pl - T) ** 2)) / rs), "AFFINE": float(np.sqrt(np.mean((Pa - T) ** 2)) / rs)},
                                                               "cos_COMMON_LINEAR_pred": float(np.sum(Pc * Pl) / (np.linalg.norm(Pc) * np.linalg.norm(Pl))),
                                                               "zbar": Z.mean(0).tolist()}
            odd_err, odd_tag, odd_c, own = [], [], [], []
            for i in range(36):
                v = V36[i]
                o = a.oddness(Z[i], Ba36, ca36, v); odd_err.append(abs(o - 2 * ca36 @ v)); odd_tag.append(abs(a.oddness(Z[i], Ba36, ca36, v, sigma=1.0)))
                odd_c.append(o / rs); own.append(T[i, i] / rs)
            gates["orientation_identity_max_abs"] = max(gates["orientation_identity_max_abs"], float(max(odd_err) / np.abs(T).max()), float(max(odd_tag) / np.abs(T).max()))
            res["orientation_full36"][f"{q}|z{inp}"] = {"untagged_affine_oddness_rms_native": float(np.sqrt(np.mean(np.square(odd_c)))),
                                                        "own_derivative_rms_native": float(np.sqrt(np.mean(np.square(own)))),
                                                        "common_oddness_rms_native": float(np.sqrt(np.mean(np.square([2 * dbar36 @ V36[i] / rs for i in range(36)])))),
                                                        "linear_oddness": 0.0, "note": "sigma tag, if used, is paired-experiment metadata, not a recovered attribute; LOW/HIGH base swap not tested"}
            coef = {"Q": Q36, "B_linear": Bl36, "B_affine": Ba36, "c_affine": ca36, "dbar": dbar36}
            np.savez(OUT / f"private_coefficients_{q}_z{inp}.npz", **coef, zbar=Z.mean(0), s=np.array(s), lam=np.array(lam36),
                     source_pairs=np.array(names), application_pairs=np.array(app), fold_cells=np.array(cells),
                     **{f"fold_{k_}": np.stack(v_) for k_, v_ in fold_coefs.items()})
            got = a.cold_score(coef, Za, Ua, s)
            Va = Ua @ Q36
            direct_lin = np.einsum("ia,ab,ib->i", Za, Bl36, Va) / rs
            gates["linear_application_replay_max_abs"] = max(gates["linear_application_replay_max_abs"],
                                                             abs(float(np.sqrt(np.mean((direct_lin - ta) ** 2))) - R24["application_descriptive"][f"{q}|z{inp}"]["F"]["rmse"]),
                                                             abs(float(np.corrcoef(direct_lin, ta)[0, 1]) - R24["application_descriptive"][f"{q}|z{inp}"]["F"]["r"]),
                                                             float(np.abs(direct_lin - got["LINEAR"]).max()))
            appd = {}
            for m in MODELS:
                p_ = got[m]
                appd[m] = {"rmse": float(np.sqrt(np.mean((p_ - ta) ** 2))), "r": float(np.corrcoef(p_, ta)[0, 1]) if p_.std() > 0 else None}
                if m in ("LINEAR", "AFFINE"):
                    pc_, tc_ = p_ - got["COMMON"], ta - got["COMMON"]
                    appd[m]["centered_r"] = float(np.corrcoef(pc_, tc_)[0, 1]); appd[m]["centered_amplitude"] = float(pc_.std() / tc_.std())
            res["application_descriptive"][f"{q}|z{inp}"] = {"target": "application diagonal DERIVATIVE J_a.d_a (native); finite outcomes not evaluated", **appd}
            for j, p in enumerate(app):
                app_rows.append({"query": q, "input": f"z{inp}", "pair": p, "truth": float(ta[j]), **{f"pred_{m}": float(got[m][j]) for m in MODELS}})
    P = pd.DataFrame(pred_rows); P.to_csv(OUT / "private_heldcell_predictions.csv", index=False)
    Sx = pd.DataFrame(sup_rows); Sx.to_csv(OUT / "private_supplement_heldonor.csv", index=False)
    pd.DataFrame(app_rows).to_csv(OUT / "private_application_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_source_diagnostics.csv", index=False)
    # LINEAR exact replay of packet 24
    mg = P.merge(P24pred[["query", "input", "pair", "truth", "pred_F", "pred_ZERO"]], on=["query", "input", "pair"], suffixes=("", "_p24"))
    assert len(mg) == len(P) == 144
    gates["linear_heldcell_replay_max_abs"] = float(max(np.abs(mg.pred_LINEAR - mg.pred_F).max(), np.abs(mg.truth - mg.truth_p24).max()))
    ms = Sx.merge(P24sup[["query", "input", "pair", "mse_F", "mse_ZERO"]], on=["query", "input", "pair"], suffixes=("", "_p24"))
    gates["linear_supplement_replay_max_abs"] = float(max(np.abs(ms.mse_LINEAR - ms.mse_F).max(), np.abs(ms.mse_ZERO - ms.mse_ZERO_p24).max()))
    for (q, inp), g in P.groupby(["query", "input"]):
        blk = {"pooled": {}, "per_cell_mse": {}, "paired": {}, "failure_cells_worse_than_zero": {}, "affine_worse_than_common_cells": [], "centered_minus_common": {}}
        pc = {}
        for m in MODELS:
            e = g[f"pred_{m}"] - g.truth
            blk["pooled"][m] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "r": float(np.corrcoef(g[f"pred_{m}"], g.truth)[0, 1]) if g[f"pred_{m}"].std() > 0 else None,
                                "amplitude_sd_ratio": float(g[f"pred_{m}"].std() / g.truth.std())}
            pc[m] = g.assign(se=e ** 2).groupby("cell").se.mean().reindex(cells).to_numpy()
            blk["per_cell_mse"][m] = dict(zip(cells, pc[m].round(10).tolist()))
        for m1, m2 in CONTRASTS:
            blk["paired"][f"{m1}-{m2}"] = cell_boot(pc[m1], pc[m2])
        for m in ("COMMON", "LINEAR", "AFFINE"):
            blk["failure_cells_worse_than_zero"][m] = [c for c, v1, v0 in zip(cells, pc[m], pc["ZERO"]) if v1 > v0]
        blk["affine_worse_than_common_cells"] = [c for c, v1, v0 in zip(cells, pc["AFFINE"], pc["COMMON"]) if v1 > v0]
        tc = g.truth - g.pred_COMMON
        for m in ("ZERO", "LINEAR", "AFFINE"):
            pc_ = g[f"pred_{m}"] - g.pred_COMMON
            gates["centered_residual_identity_max"] = max(gates["centered_residual_identity_max"], float(np.abs((pc_ - tc) - (g[f"pred_{m}"] - g.truth)).max()))
            blk["centered_minus_common"][m] = {"r": float(np.corrcoef(pc_, tc)[0, 1]), "amplitude_sd_ratio": float(pc_.std() / tc.std())}
        blk["centered_minus_common"]["target_centered_sd_over_raw_sd"] = float(tc.std() / g.truth.std())
        sg = Sx[(Sx["query"] == q) & (Sx["input"] == inp)]
        spc = {m: sg.groupby("cell")[f"mse_{m}"].mean().reindex(cells).to_numpy() for m in MODELS}
        blk["supplement_heldonor_x_train_recipients"] = {"pooled_rmse": {m: float(np.sqrt(sg[f"mse_{m}"].mean())) for m in MODELS},
                                                          "mean_within_donor_r": {m: float(sg[f"r_{m}"].mean()) for m in ("COMMON", "LINEAR", "AFFINE")},
                                                          "paired": {f"{m1}-{m2}": cell_boot(spc[m1], spc[m2]) for m1, m2 in CONTRASTS},
                                                          "affine_worse_than_common_cells": [c for c, v1, v0 in zip(cells, spc["AFFINE"], spc["COMMON"]) if v1 > v0]}
        fd = pd.DataFrame(res["fold_diagnostics"][f"{q}|{inp}"])
        blk["fold_training_means"] = {k_: float(fd[k_].mean()) for k_ in fd.columns if k_ not in ("fold", "held_cell")}
        blk["held_grad_share_in_train_span_mean"] = float(g.held_grad_share_in_train_span.mean())
        res["summary"][f"{q}|{inp}"] = blk
    res["gates"] = gates
    res["seconds_main"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    # cold replay in a separate process
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--cold-replay"], check=True)
    cr = json.loads((OUT / "cold_replay.json").read_text())
    res["gates"]["cold_replay_max_abs"] = cr["max_abs"]; res["gates"]["cold_replay_pid_differs"] = cr["pid"] != os.getpid()
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/r01_source_affine.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print(json.dumps(res["gates"], indent=1, default=float)); print("done", round(res["seconds"], 1))


if __name__ == "__main__":
    if "--cold-replay" in sys.argv:
        verify(); cold_replay()
    else:
        main()
