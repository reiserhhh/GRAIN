#!/usr/bin/env python3
"""R01 native application-J diagnostic (docs/GRAIN_R01_APPLICATION_J_DIAGNOSTIC_PROTOCOL_V1.md).
Stages: neural (replay 16 + gradients 288 + finite differences 8; cap 320) and analyze (CPU)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.native_action import Accounting  # noqa: E402
from scripts.run_grain_p05_native_action import (  # noqa: E402  (frozen helpers, unchanged)
    OP, QUERIES, check_suffix, grad_score, load_panel, observer_and_alphabets, segments, state, texts,
)

PROTOCOL = ROOT / "docs/GRAIN_R01_APPLICATION_J_DIAGNOSTIC_PROTOCOL_V1.md"
PROTOCOL_SHA = "b056ab512d1cc6922abff99c71c32ef8520ea61c58a4a0f8c2b0992f7377ddf2"
NA = ROOT / "results/grain_gptpro_foundation_v3/P05/native_action_r1"
GEO = ROOT / "results/grain_gptpro_foundation_v3/P05/native_geometry_r1/native_geometry.json"
OUT = ROOT / "results/grain_gptpro_foundation_v3/P05/application_j_diagnostic_r1"
SRC_TREE_SHA = "35e1b6d4d305a8e0aacf0f47827e42bf99921515d0fcf295c19c2e8ea649d751"
CAP, H_FD, TOL, TOL_FD_REL, NEAR0, TOL_FD_ABS = 320, 0.1, 1e-5, 0.10, 1e-6, 1e-7
FD_PAIRS = ("extraversion|anchored|advice|s11", "extraversion|anchored|advice|s23")
WALL = 1800
SEED, BOOT = 20260926, 2000
CHANGES = ("donor", "E4", "J4", "E24", "J24", "RAND_J24", "r2_compiled", "r2_random", "zero")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def src_tree_sha():
    jd = sorted(p for p in (NA / "private_jacobians").glob("*.npy") if not p.name.startswith("._"))
    return hashlib.sha256("".join(f"{p.name}:{digest(p)}\n" for p in jd).encode()).hexdigest()


def verify():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if digest(ROOT / rel) != h:
            raise ValueError(f"Binding mismatch: {rel}")
    if src_tree_sha() != SRC_TREE_SHA:
        raise ValueError("source Jacobian tree changed")


def call(acc, item, step, passes, fn):
    if item in acc.done():
        return acc.done()[item]
    acc.start(item, step, passes)
    try:
        payload = fn()
    except Exception as exc:
        acc.fail(item, repr(exc))
        raise
    acc.finish(item, **payload)
    return acc.done()[item]


def write(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n")
    os.replace(tmp, path)


def stage_neural(_args):
    from scripts.run_grain_conditional_operator_native import context, intervene
    from scripts.run_grain_persona_reverse_trace import read_score

    verify()
    OUT.mkdir(parents=True, exist_ok=True)
    arr, pairs, ids = load_panel()
    shapes, off = segments(pairs)
    order = pairs.pair.tolist()
    app = ids["application_pairs"]
    assert all(p in app for p in ids["replay_pairs"]) and all(p in app for p in FD_PAIRS) and len(app) == 72
    status = {"started": datetime.now(timezone.utc).isoformat(), "protocol_sha256": PROTOCOL_SHA, "stage": "neural",
              "planned": {"replay_forwards": 16, "gradient_forwards": 144, "gradient_backwards": 144, "fd_forwards": 8,
                          "total_passes": 312, "cap": CAP}}
    t_pre = time.monotonic()
    obs, alpha = observer_and_alphabets()
    t_load = time.monotonic(); status["model_load_seconds"] = t_load - t_pre
    status["model_manifest"] = obs.manifest()
    tx = texts()
    check_suffix(obs, pairs, [pairs.low_id[order.index(p)] for p in app])  # tokenizer only, no forward
    status["suffix_binding_72_app_low"] = "OK"
    acc = Accounting(OUT / "accounting.jsonl", CAP)
    eff = pd.DataFrame([json.loads(x) for x in (OP / "native_effects.jsonl").read_text().splitlines()])
    donor_eff = eff[eff.kind == "donor"].set_index(["pair", "query"])["effect"]

    def timed_out():
        return time.monotonic() - t_load > WALL

    def stop(reason):
        status.update({"status": reason, "passes_attempted": acc.passes_attempted(), "finished": datetime.now(timezone.utc).isoformat(),
                       "seconds_after_load": time.monotonic() - t_load})
        write(OUT / "neural_status.json", status)
        print(json.dumps({k: v for k, v in status.items() if k != "model_manifest"}, indent=1, default=float))
        raise SystemExit(0 if reason == "COMPLETE" else 3)
    # 1. replay (16 forwards)
    fails = []
    for p in ids["replay_pairs"]:
        i = order.index(p); low = pairs.low_id[i]
        for qi, q in enumerate(QUERIES):
            if timed_out():
                stop("PARTIAL_TIME")
            c_ids, pos = context(obs, tx[low], q)
            cached = float(state(low, q)["score"])
            r = call(acc, f"replay_low|{p}|{q}", "replay", 1, lambda c_ids=c_ids, q=q: {"score": read_score(obs, c_ids, alpha[q])["score"]})
            if abs(r["score"] - cached) > TOL:
                fails.append(("low", p, q, r["score"], cached))
            d = arr["dh"][i, off[qi]:off[qi + 1]].reshape(shapes[qi])
            r2 = call(acc, f"replay_donor|{p}|{q}", "replay", 1, lambda c_ids=c_ids, pos=pos, q=q, d=d: {"score": intervene(obs, c_ids, pos, alpha[q], d)["score"]})
            if abs((r2["score"] - cached) - donor_eff[(p, q)]) > TOL:
                fails.append(("donor", p, q, r2["score"] - cached, float(donor_eff[(p, q)])))
    status["replay_fails"] = fails
    write(OUT / "neural_status.json", status)
    if fails:
        stop("BLOCKED_REPLAY")
    # 2. application gradients (144 items x 2 passes)
    jdir = OUT / "private_app_jacobians"; jdir.mkdir(exist_ok=True)
    worst = 0.0
    for p in app:
        i = order.index(p); low = pairs.low_id[i]
        for q in QUERIES:
            if timed_out():
                stop("PARTIAL_TIME")
            target = jdir / f"{i:03d}_{q}.npy"

            def fn(low=low, q=q, target=target):
                c_ids, pos = context(obs, tx[low], q)
                s, g = grad_score(obs, c_ids, pos, alpha[q])
                tmp = target.with_name(target.name + ".tmp")
                with open(tmp, "wb") as fh:
                    np.save(fh, g)
                os.replace(tmp, target)
                return {"score": s, "cached_low": float(state(low, q)["score"]), "grad_norm": float(np.linalg.norm(g)), "sha256": digest(target)}
            r = call(acc, f"app_jacobian|{p}|{q}", "app_jacobian", 2, fn)
            err = abs(r["score"] - r["cached_low"]); worst = max(worst, err)
            if err > TOL:
                status["grad_score_fail"] = {"pair": p, "query": q, "abs_err": err}
                stop("BLOCKED_GRAD_SCORE")
    status["grad_pass_score_replay_max_abs"] = worst
    write(OUT / "neural_status.json", status)
    # 3. central finite differences (8 forwards)
    fd = []
    for p in FD_PAIRS:
        i = order.index(p); low = pairs.low_id[i]
        for q in QUERIES:
            if timed_out():
                stop("PARTIAL_TIME")
            g = np.load(jdir / f"{i:03d}_{q}.npy"); gn = float(np.linalg.norm(g))
            unit = g / gn if gn > 0 else g
            c_ids, pos = context(obs, tx[low], q)
            sp = call(acc, f"fd_plus|{p}|{q}", "fd", 1, lambda c_ids=c_ids, pos=pos, q=q, unit=unit: {"score": intervene(obs, c_ids, pos, alpha[q], H_FD * unit)["score"]})
            sm = call(acc, f"fd_minus|{p}|{q}", "fd", 1, lambda c_ids=c_ids, pos=pos, q=q, unit=unit: {"score": intervene(obs, c_ids, pos, alpha[q], -H_FD * unit)["score"]})
            cd = (sp["score"] - sm["score"]) / (2 * H_FD)
            abs_err = abs(cd - gn)
            ok = (abs_err / gn <= TOL_FD_REL) if gn >= NEAR0 else (abs_err <= TOL_FD_ABS)
            fd.append({"pair": p, "query": q, "central_diff": cd, "grad_norm": gn, "abs_err": abs_err,
                       "rel_err": abs_err / gn if gn > 0 else None, "rule": "relative" if gn >= NEAR0 else "absolute", "pass": bool(ok)})
    status["fd_check"] = fd
    if not all(x["pass"] for x in fd):
        stop("BLOCKED_FD")
    stop("COMPLETE")


# ------------------------------------------------------------------ CPU analysis
def basis(M):
    u, s, _ = np.linalg.svd(M.T, full_matrices=False)
    return u[:, s > s.max() * 1e-10]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else float("nan")


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def stage_analyze(_args):
    from grain_research.vector_mechanisms.reverse_trace import equal_norm_random
    from scipy.stats import spearmanr

    verify()
    st = json.loads((OUT / "neural_status.json").read_text())
    if st.get("status") != "COMPLETE":
        raise SystemExit(f"neural stage not complete: {st.get('status')}")
    arr, pairs, ids = load_panel()
    shapes, off = segments(pairs)
    order = pairs.pair.tolist(); src = arr["source"]; si = np.flatnonzero(src)
    app = ids["application_pairs"]; cells = ids["cells"]
    prep = np.load(NA / "private_prepared.npz")
    r1 = pd.DataFrame([json.loads(x) for x in (NA / "effects.jsonl").read_text().splitlines()]).set_index(["pair", "query", "kind"])
    r2 = pd.DataFrame([json.loads(x) for x in (OP / "native_effects.jsonl").read_text().splitlines()]).set_index(["pair", "query", "kind"])
    zmax = json.loads((NA / "prepare.json").read_text())["zmax_source_app"]
    geo = json.loads(GEO.read_text())
    rows, contract, grads = [], [], {}
    Qs = {}
    for qi, q in enumerate(QUERIES):
        sl = slice(off[qi], off[qi + 1]); D = off[qi + 1] - off[qi]
        Jsrc = np.stack([np.load(NA / "private_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in si])
        Q = basis(Jsrc); Qs[q] = Q
        for p in app:
            i = order.index(p); low = pairs.low_id[i]
            J = np.load(OUT / "private_app_jacobians" / f"{i:03d}_{q}.npy").ravel()
            grads[(p, q)] = J
            Jin = Q @ (Q.T @ J); Jout = J - Jin
            base = float(state(low, q)["score"])
            ch = {"donor": arr["dh"][i, sl].astype(float), **{c: prep[c][i, sl].astype(float) for c in ("E4", "J4", "E24", "J24")},
                  "r2_compiled": arr["predicted"][i, sl].astype(float), "zero": np.zeros(D)}
            ch["RAND_J24"] = equal_norm_random(np.zeros(shapes[qi]), ch["J24"].reshape(shapes[qi]), f"jw24|{p}|{q}").ravel()
            ch["r2_random"] = equal_norm_random(np.zeros(shapes[qi]), ch["r2_compiled"].reshape(shapes[qi]), f"{p}|{q}").ravel()
            meas = {c: float(r1.loc[(p, q, c), "effect"]) for c in ("E4", "J4", "E24", "J24", "RAND_J24")}
            for c, kind in (("donor", "donor"), ("r2_compiled", "compiled"), ("r2_random", "random"), ("zero", "self")):
                meas[c] = float(r2.loc[(p, q, kind), "score"]) - base
            for c, (tab, kind) in {"E4": (r1, "E4"), "J4": (r1, "J4"), "E24": (r1, "E24"), "J24": (r1, "J24"), "RAND_J24": (r1, "RAND_J24"),
                                   "donor": (r2, "donor"), "r2_compiled": (r2, "compiled"), "r2_random": (r2, "random")}.items():
                cached = float(tab.loc[(p, q, kind), "action_norm"]); got = float(np.linalg.norm(ch[c]))
                contract.append({"pair": p, "query": q, "change": c, "cached_norm": cached, "norm": got,
                                 "rel_err": abs(got - cached) / max(cached, 1e-300)})
            for c in CHANGES:
                L = float(J @ ch[c]); Lin = float(Jin @ ch[c])
                rows.append({"pair": p, "query": q, "cell": cells[p], "split": pairs.split[i], "change": c, "meas": meas[c], "L": L, "R": meas[c] - L,
                             "L_in": Lin, "L_out": L - Lin, "norm": float(np.linalg.norm(ch[c])), "grad_norm": float(np.linalg.norm(J)),
                             "app_share_in_source_span": float((Q.T @ J) @ (Q.T @ J) / (J @ J)),
                             "zmax24": float(np.abs(prep["z24"][i]).max()), "zmax4": float(np.abs(prep["z4"][i]).max())})
    C = pd.DataFrame(rows)
    K = pd.DataFrame(contract)
    unavailable = sorted({c for c in ("RAND_J24", "r2_random") if (K[K.change == c].rel_err > 1e-9).any()})
    # decomposition rows vs donor
    dcols = C[C.change == "donor"].set_index(["pair", "query"])[["meas", "L", "R", "L_in", "L_out"]].add_suffix("_d")
    X = C[C.change != "donor"].join(dcols, on=["pair", "query"])
    X["E"] = X.meas - X.meas_d; X["T"] = X.L - X.L_d; X["dR"] = X.R - X.R_d
    X["T_in"] = X.L_in - X.L_in_d; X["T_out"] = X.L_out - X.L_out_d
    ident = float((X.E - X["T"] - X.dR).abs().max())
    res = {"protocol_sha256": PROTOCOL_SHA, "identity_max_abs_error": ident, "random_controls_unavailable": unavailable,
           "norm_contract_max_rel_err": K.groupby("change").rel_err.max().to_dict(), "neural_status": {k: st[k] for k in ("status", "passes_attempted", "grad_pass_score_replay_max_abs", "fd_check", "replay_fails", "seconds_after_load", "model_load_seconds")},
           "per_query": {}, "geometry": {}, "alignment_control": {}, "extrapolation": {}}
    rng = np.random.default_rng(SEED)
    for q in QUERIES:
        g = X[X["query"] == q]; cq = C[C["query"] == q]
        donor = cq[cq.change == "donor"].set_index("pair")
        uc = sorted(set(cells[p] for p in app)); pidx = {p: k for k, p in enumerate(app)}
        idx = [np.array([pidx[p] for p in app if cells[p] == c]) for c in uc]
        draws = [np.concatenate([idx[j] for j in rng.integers(0, len(uc), len(uc))]) for _ in range(BOOT)]
        dm = donor.loc[app, "meas"].to_numpy(); dL = donor.loc[app, "L"].to_numpy()
        out = {"donor": {"rms_meas": rms(dm), "rms_L": rms(dL), "rms_R": rms(dm - dL), "corr_L_meas": corr(dL, dm),
                         "slope_meas_on_L": float((dL @ dm) / (dL @ dL)) if dL @ dL > 0 else float("nan")}}
        for c in CHANGES[1:]:
            h = g[g.change == c].set_index("pair").loc[app]
            E, T, dR = h.E.to_numpy(), h["T"].to_numpy(), h.dR.to_numpy()
            m, Lc = h.meas.to_numpy(), h.L.to_numpy()
            mE, mT, mR, mX = np.mean(E**2), np.mean(T**2), np.mean(dR**2), 2 * np.mean(T * dR)

            def shares(b):
                e = np.mean(E[b] ** 2)
                return (np.mean(T[b] ** 2) / e, np.mean(dR[b] ** 2) / e, 2 * np.mean(T[b] * dR[b]) / e) if e > 0 else (np.nan,) * 3
            bs = np.array([shares(b) for b in draws])
            ci = lambda k: [float(v) for v in np.nanpercentile(bs[:, k], [2.5, 97.5])]
            sT, sR, sX = (mT / mE, mR / mE, mX / mE) if mE > 0 else (np.nan,) * 3
            if c == "zero" or mE == 0:
                label = "n/a"
            else:
                cT, cR = ci(0), ci(1)
                label = "tangent mismatch dominant" if cT[0] > cR[1] else ("finite-step remainder dominant" if cR[0] > cT[1] else "combination")
            out[c] = {"mean_E2": float(mE), "mean_T2": float(mT), "mean_dR2": float(mR), "cross_2TdR": float(mX),
                      "sum_check": float(mT + mR + mX - mE), "share_T": float(sT), "share_R": float(sR), "share_X": float(sX),
                      "share_T_ci": ci(0), "share_R_ci": ci(1), "share_X_ci": ci(2), "label": label,
                      "rms_E": rms(E), "rms_T": rms(T), "rms_dR": rms(dR), "mae_E": float(np.abs(E).mean()), "mae_T": float(np.abs(T).mean()),
                      "mae_dR": float(np.abs(dR).mean()), "mean_E": float(E.mean()), "mean_T": float(T.mean()), "mean_dR": float(dR.mean()),
                      "rms_T_in": rms(h.T_in), "rms_T_out": rms(h.T_out),
                      "linearization": {"corr_L_meas": corr(Lc, m), "rms_R_over_rms_meas": rms(m - Lc) / rms(m) if rms(m) > 0 else float("nan"),
                                        "slope_meas_on_L": float((Lc @ m) / (Lc @ Lc)) if Lc @ Lc > 0 else float("nan")},
                      "ranking": {"corr_meas_c_meas_d": corr(m, dm), "corr_L_c_L_d": corr(Lc, dL)},
                      "overshoot": {"rms_L_c_over_rms_L_d": rms(Lc) / rms(dL), "rms_meas_c_over_rms_meas_d": rms(m) / rms(dm),
                                    "median_norm_ratio": float(np.median(h.norm.to_numpy() / donor.loc[app, "norm"].to_numpy()))},
                      "relative_to_donor_amplitude": {"rms_E": rms(E) / rms(dm), "rms_T": rms(T) / rms(dm), "rms_dR": rms(dR) / rms(dm)}}
        res["per_query"][q] = out
        sh = cq[cq.change == "donor"].app_share_in_source_span.to_numpy()
        Q = Qs[q]; D = Q.shape[0]
        res["geometry"][q] = {"D": D, "source_span_rank": Q.shape[1], "random_reference_rank_over_D": Q.shape[1] / D,
                              "app_share_mean": float(sh.mean()), "app_share_q10_q50_q90": [float(v) for v in np.percentile(sh, [10, 50, 90])],
                              "app_share_ci": [float(v) for v in np.percentile([sh[b].mean() for b in draws], [2.5, 97.5])],
                              "source_loo_coverage_mean": geo["queries"][q]["Q1_loo_gradient_coverage"]["mean"],
                              "app_grad_norm_median": float(np.median(cq[cq.change == "donor"].grad_norm)),
                              "source_grad_norm_median": geo["queries"][q]["gradient_norm_median"]}
        # alignment control: gradient of the next application pair (cyclic) applied to this pair's donor change
        shifted = []
        for k, p in enumerate(app):
            pn = app[(k + 1) % len(app)]
            i = order.index(p); qi = QUERIES.index(q)
            d = arr["dh"][i, off[qi]:off[qi + 1]].astype(float)
            shifted.append(float(grads[(pn, q)] @ d))
        shifted = np.array(shifted)
        res["alignment_control"][q] = {"matched_corr_L_meas_donor": corr(dL, dm), "shifted_corr_L_meas_donor": corr(shifted, dm),
                                       "matched_rms_R_donor": rms(dm - dL), "shifted_rms_R_donor": rms(dm - shifted),
                                       "note": "descriptive alignment control (cyclic shift within query/dimension/role); not a p-value"}
        ex = {}
        for c in ("E24", "J24"):
            h = g[g.change == c].set_index("pair").loc[app]
            ex[c] = {"spearman_zmax24_absT": float(spearmanr(h.zmax24, np.abs(h["T"])).statistic),
                     "spearman_zmax24_absdR": float(spearmanr(h.zmax24, np.abs(h.dR)).statistic)}
        zz = g[g.change == "E24"].set_index("pair").loc[app]
        ex["pairs_beyond_source_support_z24"] = int((zz.zmax24 > zmax[1][0]).sum())
        ex["pairs_beyond_source_support_z4"] = int((zz.zmax4 > zmax[0][0]).sum())
        ex["source_max"] = {"z4": zmax[0][0], "z24": zmax[1][0]}
        res["extrapolation"][q] = ex
    # decision flags (frozen rule)
    dec = {}
    for q in QUERIES:
        sh = res["geometry"][q]["app_share_mean"]
        dec[q] = {"application_tangent_transport": "weakened (share >= .9)" if sh >= .9 else ("supported (share < .5)" if sh < .5 else "unresolved (.5-.9)"),
                  "cell_labels": {c: res["per_query"][q][c]["label"] for c in ("E4", "J4", "E24", "J24", "RAND_J24", "r2_compiled", "r2_random")},
                  "E24_overshoot_location": "tangent" if res["per_query"][q]["E24"]["overshoot"]["rms_L_c_over_rms_L_d"] >= 1.5 else "remainder"}
    res["decision_flags"] = dec
    C.to_csv(OUT / "components_by_change.csv", index=False)
    X.to_csv(OUT / "decomposition_vs_donor.csv", index=False)
    K.to_csv(OUT / "norm_identity_contract.csv", index=False)
    res["code_sha256"] = digest(Path(__file__)); res["created"] = datetime.now(timezone.utc).isoformat()
    write(OUT / "results.json", res)
    print(json.dumps({"identity": ident, "unavailable": unavailable, "geometry": res["geometry"], "decision_flags": dec}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["neural", "analyze"])
    a = ap.parse_args()
    {"neural": stage_neural, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
