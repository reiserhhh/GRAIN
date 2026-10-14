#!/usr/bin/env python
"""R01 source-crossed derivative design (docs/GRAIN_R01_SOURCE_CROSSED_PROTOCOL_V1.md). Cached arrays only; 0 neural calls;
ordinary linear fits. Crossed T_ij are cached directional derivatives, NOT finite native patch outcomes or new samples."""
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
from grain_research import r01_source_crossed as x  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R01_SOURCE_CROSSED_PROTOCOL_V1.md"
PROTOCOL_SHA = "49ddc5b19f1e3a79337d91d470ba3393ce72d63a72032c54b1b552765baa57ab"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
NA, AJ = V3 / "P05/native_action_r1", V3 / "P05/application_j_diagnostic_r1"
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
IDS = ROOT / "docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json"
OUT = V3 / "P05/source_crossed_action_r1"
QUERIES, OFF = ("extraversion", "neuroticism"), {"extraversion": slice(0, 7552), "neuroticism": slice(7552, 15232)}
DESIGNS = ("D", "C", "R", "F", "E", "ZERO")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def tree_sha(d):
    fs = sorted(p for p in Path(d).glob("*.npy") if not p.name.startswith("._"))
    return hashlib.sha256("".join(f"{p.name}:{sha(p)}\n" for p in fs).encode()).hexdigest()


def cell_boot(per_cell_a, per_cell_b, seed=20260927, n=2000):
    d = np.asarray(per_cell_a) - np.asarray(per_cell_b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "ci_contains_0": bool(lo <= 0 <= hi)}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    trees = {"private_jacobians": tree_sha(NA / "private_jacobians"), "private_app_jacobians": tree_sha(AJ / "private_app_jacobians")}
    if OUT.exists():
        raise FileExistsError("source_crossed_action_r1 exists; preserve earlier runs")
    arr = np.load(OP / "action_arrays.npz"); prep = np.load(NA / "private_prepared.npz")
    order = json.loads((OP / "pair_order.json").read_text()); ids = json.loads(IDS.read_text()); fit = json.loads((NA / "fit.json").read_text())
    si = np.flatnonzero(arr["source"]); names = [order[i] for i in si]
    assert sorted(names) == sorted(ids["source_pairs"]) and len(si) == 36
    # attempt 1 failed here: the IDS "cells" map covers non-source pairs only; a pair's cell is its name without the seed suffix
    # (the same convention as that map's values). Checked against the map for every pair it does contain.
    assert all(v == k.rsplit("|", 1)[0] for k, v in ids["cells"].items())
    cell_of = [nm.rsplit("|", 1)[0] for nm in names]; seed_of = [nm.split("|")[-1] for nm in names]
    cells = sorted(set(cell_of)); assert len(cells) == 12 and all(cell_of.count(c) == 3 for c in cells)
    ai = [order.index(p) for p in ids["application_pairs"]]
    OUT.mkdir(parents=True)
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "trees": trees, "source_pairs": 36, "cells": 12, "replay": {}, "diagnostics": {}, "folds": {}, "summary": {}, "application_descriptive": {},
           "labels": {"T": "cached directional derivatives d_i.u_j; NOT independent samples, finite patch outcomes or W-generated labels; effective material count 36 pairs / 12 cells",
                      "fullcross": "donor regression under aggregate training-gradient metric U'U (same donors), not new observations",
                      "cv": "conditional on frozen source coordinate transforms (z, s); not end-to-end representation validation"}}
    pred_rows, supp_rows, rowlists = [], [], {}
    for qi, q in enumerate(QUERIES):
        sl = OFF[q]
        Js = np.stack([np.load(NA / "private_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in si])
        assert Js.shape[1] == sl.stop - sl.start
        s = float(fit["cells"][f"J24|{q}"]["metric_scale_s"]); assert abs(fit["cells"][f"J4|{q}"]["metric_scale_s"] - s) <= 1e-9 * s
        assert abs(Js.shape[1] / np.mean((Js ** 2).sum(1)) - s) <= 1e-9 * s
        U = np.sqrt(s) * Js; Dn = arr["dh"][si, sl].astype(float)
        T = Dn @ U.T                                              # T[i, j] = d_i . u_j
        np.save(OUT / f"private_T_{q}.npy", T)
        t_old = (U * Dn).sum(1)
        rep = {"diag_T_vs_old_t_max": float(np.abs(np.diag(T) - t_old).max())}
        for inp in ("4", "24"):
            Z = prep[f"z{inp}"][si].astype(float); Zall = prep[f"z{inp}"].astype(float)
            K = (Z @ Z.T) * (U @ U.T); c = np.linalg.solve(K + np.eye(36), t_old)
            rep[f"J{inp}_compiled_replay_max"] = float(np.abs((Zall @ Z.T * c) @ U - prep[f"J{inp}"][:, sl]).max())
            rep[f"E{inp}_compiled_replay_max"] = float(np.abs(Zall @ x.euclid(Z, Dn) - prep[f"E{inp}"][:, sl]).max())
            # ---- diagnostics on the full 36-source panel
            rz, ru, rk = x.kron_rank(Z, U)
            Qa, Va, Sa = x.orth_basis(U)
            Phi = np.einsum("jb,ia->ijba", Va, Z).reshape(36 * 36, Va.shape[1] * Z.shape[1])
            sphi = np.linalg.svd(Phi, compute_uv=False); rphi = int((sphi > sphi.max() * x.RANK_TOL).sum())
            prod = np.sort(np.outer(np.linalg.svd(Z, compute_uv=False), np.linalg.svd(Va, compute_uv=False)).ravel())[::-1][:len(sphi)]
            wK = np.linalg.eigvalsh(K); rK = int((wK > wK.max() * x.RANK_TOL).sum())
            share, resid = x.pz_share(Z, T)
            sT = np.linalg.svd(T, compute_uv=False)
            res["diagnostics"][f"{q}|z{inp}"] = {"rank_Z": rz, "rank_U": ru, "fullcross_rank_expected": rk, "fullcross_rank_numerical": rphi,
                                                 "product_gram_identity_max_rel": float(np.max(np.abs(sphi - prod)) / sphi.max()), "diagonal_design_rank": rK,
                                                 "fullcross_condition_kept": float(sphi.max() / sphi[rphi - 1]), "PZ_column_share": share,
                                                 "residual_singular_values_top5": resid[:5].tolist(), "T_rank_p_energy_share": float((sT[:Z.shape[1]] ** 2).sum() / (sT ** 2).sum())}
            # ---- source-cell-held-out folds
            for fi, hc in enumerate(cells):
                h = [k for k in range(36) if cell_of[k] == hc]; tr = [k for k in range(36) if cell_of[k] != hc]
                n = len(tr); lam = 1.0 / n
                Zt, Ut, Dt = Z[tr], U[tr], Dn[tr]; Tt = T[np.ix_(tr, tr)]
                Q, V, _ = x.orth_basis(Ut)
                tcell = [cell_of[k] for k in tr]; tseed = [seed_of[k] for k in tr]
                rows = {"D": [(i, i) for i in range(n)], "C": x.cyclic_rows(tcell, tcell, tseed), "R": x.random_rows(tcell, tcell, tseed, [24011, fi]),
                        "F": [(i, j) for i in range(n) for j in range(n)]}
                for k_ in ("C", "R"):
                    assert sorted(i for i, _ in rows[k_]) == list(range(n)) and sorted(j for _, j in rows[k_]) == list(range(n))
                    assert all(tcell[i] != tcell[j] for i, j in rows[k_])
                rowlists[f"{q}|z{inp}|fold{fi}"] = {k_: [[tr[i], tr[j]] for i, j in rows[k_]] for k_ in ("D", "C", "R")}
                B = {k_: x.fit_rows(Zt, V, rows[k_], Tt, lam) for k_ in ("D", "C", "R")}
                B["F"] = x.fit_full(Zt, V, Tt, lam)
                WE = x.euclid(Zt, Dt)
                grads = {k_: x.loss_grad(Zt, V, rows[k_], Tt, B[k_], lam) for k_ in ("D", "C", "R", "F")}
                fold = {"held_cell": hc, "n_train": n, "lambda": lam, "summed_alpha": {k_: len(rows[k_]) * lam for k_ in rows}, "max_grad": grads, "rank_V": int(V.shape[1])}
                if fi == 0:
                    fold["fullcross_vs_primal_max"] = float(np.abs(B["F"] - x.fit_full_primal(Zt, V, Tt, lam)).max())
                    fold["fullcross_vs_dual_max"] = float(np.abs(B["F"] - x.fit_rows(Zt, V, rows["F"], Tt, lam)).max())
                res["folds"][f"{q}|z{inp}|fold{fi}"] = fold
                # exposure of held diagonal rows
                Kz, Ku = Zt @ Zt.T, Ut @ Ut.T
                for k in h:
                    zh, uh = Z[k], U[k]; vh = Q.T @ uh
                    kzz, kuu = float(zh @ zh), float(uh @ uh)
                    zshare = float(np.linalg.norm(Zt.T @ np.linalg.pinv(Zt @ Zt.T, rcond=1e-10) @ Zt @ zh) ** 2 / kzz)
                    ushare = float((vh @ vh) / kuu)
                    expo = {}
                    for k_ in ("D", "C", "R"):
                        rr = np.asarray(rows[k_]); Kd = Kz[rr[:, 0]][:, rr[:, 0]] * Ku[rr[:, 1]][:, rr[:, 1]]
                        kv = (Zt[rr[:, 0]] @ zh) * (Ut[rr[:, 1]] @ uh)
                        expo[k_] = float(kv @ np.linalg.pinv(Kd, rcond=1e-10, hermitian=True) @ kv / (kzz * kuu))
                    expo["F"] = zshare * ushare
                    truth = T[k, k] / np.sqrt(s)
                    pr = {k_: x.predict(zh, B[k_], vh) / np.sqrt(s) for k_ in ("D", "C", "R", "F")}
                    pr["E"] = float(zh @ WE @ uh) / np.sqrt(s); pr["ZERO"] = 0.0
                    pred_rows.append({"query": q, "input": f"z{inp}", "fold": fi, "cell": hc, "pair": names[k], "truth": truth, **{f"pred_{k_}": v for k_, v in pr.items()},
                                      **{f"exposure_{k_}": v for k_, v in expo.items()}})
                    # supplement: held donor k x training recipients
                    tj = T[k, tr] / np.sqrt(s)
                    sp = {}
                    for k_ in ("D", "C", "R", "F"):
                        sp[k_] = float(np.mean(((zh @ B[k_] @ V.T) / np.sqrt(s) - tj) ** 2))
                    sp["E"] = float(np.mean(((zh @ WE @ Ut.T) / np.sqrt(s) - tj) ** 2)); sp["ZERO"] = float(np.mean(tj ** 2))
                    supp_rows.append({"query": q, "input": f"z{inp}", "cell": hc, "pair": names[k], **{f"mse_{k_}": v for k_, v in sp.items()}})
            # ---- application panel, descriptive only (full-36 fits of every frozen design)
            Qa2, Va2, _ = x.orth_basis(U); lam36 = 1 / 36
            rows36 = {"D": [(i, i) for i in range(36)], "C": x.cyclic_rows(cell_of, cell_of, seed_of), "R": x.random_rows(cell_of, cell_of, seed_of, [24012, 0])}
            B36 = {k_: x.fit_rows(Z, Va2, rows36[k_], T, lam36) for k_ in rows36}; B36["F"] = x.fit_full(Z, Va2, T, lam36); WE36 = x.euclid(Z, Dn)
            Ja = np.stack([np.load(AJ / "private_app_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in ai]); Ua = np.sqrt(s) * Ja
            Za = prep[f"z{inp}"][ai].astype(float); Da = arr["dh"][ai, sl].astype(float)
            ta = (Ua * Da).sum(1) / np.sqrt(s)
            app = {}
            for k_ in ("D", "C", "R", "F"):
                p_ = np.einsum("ia,ab,ib->i", Za, B36[k_], Ua @ Qa2) / np.sqrt(s)
                app[k_] = {"rmse": float(np.sqrt(np.mean((p_ - ta) ** 2))), "r": float(np.corrcoef(p_, ta)[0, 1])}
            pe = np.einsum("ia,ad,id->i", Za, WE36, Ua) / np.sqrt(s)
            app["E"] = {"rmse": float(np.sqrt(np.mean((pe - ta) ** 2))), "r": float(np.corrcoef(pe, ta)[0, 1])}; app["ZERO"] = {"rmse": float(np.sqrt(np.mean(ta ** 2)))}
            res["application_descriptive"][f"{q}|z{inp}"] = {"target": "application diagonal DERIVATIVE J_a.d_a (native units); finite native outcomes NOT evaluated (finite-outcome cache semantics not established in this packet)", **app}
        res["replay"][q] = rep
    P = pd.DataFrame(pred_rows); P.to_csv(OUT / "private_heldcell_predictions.csv", index=False)
    Sx = pd.DataFrame(supp_rows); Sx.to_csv(OUT / "private_supplement_heldonor_mse.csv", index=False)
    (OUT / "row_lists.json").write_text(json.dumps(rowlists) + "\n")
    for (q, inp), g in P.groupby(["query", "input"]):
        blk = {"pooled": {}, "per_cell_mse": {}, "paired_vs_D": {}, "paired_vs_ZERO": {}, "failure_cells_worse_than_zero": {}, "exposure_mean": {}}
        pc = {}
        for k_ in DESIGNS:
            e = g[f"pred_{k_}"] - g.truth
            blk["pooled"][k_] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "r": float(np.corrcoef(g[f"pred_{k_}"], g.truth)[0, 1]) if k_ != "ZERO" and g[f"pred_{k_}"].std() > 0 else None}
            pc[k_] = g.assign(se=e ** 2).groupby("cell").se.mean().reindex(cells).to_numpy()
            blk["per_cell_mse"][k_] = dict(zip(cells, pc[k_].round(8).tolist()))
        for k_ in ("C", "R", "F", "E", "ZERO"):
            blk["paired_vs_D"][f"{k_}-D"] = cell_boot(pc[k_], pc["D"])
        for k_ in ("D", "C", "R", "F", "E"):
            blk["paired_vs_ZERO"][f"{k_}-ZERO"] = cell_boot(pc[k_], pc["ZERO"])
            blk["failure_cells_worse_than_zero"][k_] = [c for c, a, z in zip(cells, pc[k_], pc["ZERO"]) if a > z]
        for k_ in ("D", "C", "R", "F"):
            blk["exposure_mean"][k_] = float(g[f"exposure_{k_}"].mean())
        sg = Sx[(Sx["query"] == q) & (Sx["input"] == inp)]
        spc = {k_: sg.groupby("cell")[f"mse_{k_}"].mean().reindex(cells).to_numpy() for k_ in DESIGNS}
        blk["supplement_heldonor_x_train_recipients"] = {"pooled_rmse": {k_: float(np.sqrt(sg[f"mse_{k_}"].mean())) for k_ in DESIGNS},
                                                          "paired_vs_D": {f"{k_}-D": cell_boot(spc[k_], spc["D"]) for k_ in ("C", "R", "F", "E", "ZERO")}}
        res["summary"][f"{q}|{inp}"] = blk
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/r01_source_crossed.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print("done", round(res["seconds"], 1))


if __name__ == "__main__":
    main()
