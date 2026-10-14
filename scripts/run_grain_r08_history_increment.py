#!/usr/bin/env python
"""R08 incremental history information on existing PANDORA caches (docs/GRAIN_R08_HISTORY_INCREMENT_PROTOCOL_V1.md).
CPU only, 0 neural calls, no raw-text scan. Replay gates first; selection on source-validation only; held scored once."""
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
from grain_research import r08_history_increment as h  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R08_HISTORY_INCREMENT_PROTOCOL_V1.md"
PROTOCOL_SHA = "705c55ad5c6781eb2dda5511b00127fcd58e69288db9b2deb0f7e1df04974ffb"
F14 = ROOT / "results/grain_gptpro_foundation_v3/F14"
OUT = F14 / "history_increment_r1"
TOL = 1e-12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def boot_draws(n):
    return np.random.default_rng(20260926).integers(0, n, size=(2000, n))


def paired(df, col, ref, auths, draws):
    per = df.groupby("author")[[col, ref]].mean().reindex(auths)
    d = (per[col] - per[ref]).to_numpy()
    bs = d[draws].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "ci_contains_0": bool(lo <= 0 <= hi)}


def strata(df, col, ref, masks):
    n = len(df); out = {}
    for name, mask in masks.items():
        sub = df[mask]
        out[name] = {"events": int(len(sub)), "mean_delta": float((sub[col] - sub[ref]).mean()) if len(sub) else None,
                     "contribution_to_pooled_delta": float((sub[col] - sub[ref]).sum() / n)}
    return out


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("history_increment_r1 exists; preserve earlier runs")
    D = h.load_all()
    train, val, held = D["train"], D["val"], D["held"]
    assert (len(train), len(val), len(held)) == (403, 101, 120)
    assert not (set(train) & set(val) or set(train) & set(held) or set(val) & set(held))
    by, cats, raw, Z = D["by_author"], D["cats"], D["raw"], D["Z"]
    assert all(len(by[a]) == h.NEV for a in train + val + held)
    OUT.mkdir(parents=True)
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "row_identity": D["ident"], "roles": {"source_train": 403, "source_validation": 101, "held": 120},
           "held_disclosure": "same 120 held authors used in packets 15-16: exploratory contrast, NOT an untouched confirmatory lockbox"}
    # ================= replay gates
    g, T = h.venue_population(D)
    f = h.load_oom(F14 / "pandora_venue_oom_r1/oom_fit_source.npz")
    pop = h.keyed_population(D)
    prev15 = pd.read_csv(F14 / "pandora_history_comparison_r1/held_event_scores.csv", float_precision="round_trip")
    prev16 = pd.read_csv(F14 / "pandora_venue_oom_r1/held_event_rows.csv", float_precision="round_trip")
    oomres = json.loads((F14 / "pandora_venue_oom_r1/results.json").read_text())
    oom_held = {a: h.oom_records(cats[by[a]], f) for a in held}
    oom_val = {a: h.oom_records(cats[by[a]], f) for a in val}
    rep = []
    for a in held:
        ix = by[a]
        for t in range(h.T0, h.NEV):
            y = cats[ix[t]]
            vb = h.venue_base(cats[ix[:t]], g, T)
            kp = h.keyed_predict(Z[ix[:t]], pop, np.zeros((30, 30)), np.zeros((60, 30)), 0.0, 0.0)
            o = oom_held[a][t]
            rep.append({"author": int(a), "t": t, **{f"ll_{k}": float(-np.log(p[y])) for k, p in vb.items()},
                        "ll_oom": float(-np.log(o["p"][y])) if not o["refused"] else np.nan,
                        **{f"kmse_{k}": float(((kp[k] - Z[ix[t]]) ** 2).mean()) for k in ("global", "static", "pca5", "ar1", "combined")}})
    R = pd.DataFrame(rep)
    m15 = R.merge(prev15, on=["author", "t"]); m16 = R.merge(prev16, on=["author", "t"], suffixes=("", "_p16"))
    gates = {"rows": [int(len(R)), int(len(m15)), int(len(m16))],
             "venue_vs_p15": max(float(np.abs(m15[f"ll_{k}"] - m15[f"venue_ll_{k}"]).max()) for k in ("global", "static", "order1", "combined")),
             "venue_vs_p16": max(float(np.abs(m16[f"ll_{k}"] - m16[f"ll_{k}_p16"]).max()) for k in ("global", "static", "order1", "combined")),
             "oom_held_vs_p16": float(np.nanmax(np.abs(m16["ll_oom"] - m16["ll_oom_p16"]))),
             "keyed_vs_p15": max(float(np.abs(m15[f"kmse_{k}"] - m15[f"keyed_mse_{k}"]).max()) for k in ("global", "static", "pca5", "ar1", "combined"))}
    vll = [-np.log(oom_val[a][t]["p"][cats[by[a][t]]]) for a in val for t in range(h.T0, h.NEV) if not oom_val[a][t]["refused"]]
    gates["oom_validation_logloss_vs_p16"] = float(abs(np.mean(vll) - oomres["validation"]["20"]["logloss_nonrefused"]))
    res["replay_gates"] = gates
    if gates["rows"] != [1320, 1320, 1320] or max(v for k, v in gates.items() if k != "rows") > TOL:
        (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        raise RuntimeError(f"replay gate failed: {gates}")
    # ================= B1 nested venue mixture: selection on source-validation
    vrows = []
    for a in val:
        ix = by[a]
        for t in range(h.T0, h.NEV):
            y = cats[ix[t]]; pB = h.venue_base(cats[ix[:t]], g, T)["combined"]; o = oom_val[a][t]
            pO = None if o["refused"] else o["p"]
            vrows.append({"author": int(a), "t": t, "refused": o["refused"], **{f"ll_w{w}": float(-np.log(h.mix(pB, pO, w)[y])) for w in h.W_VENUE_GRID}})
    V = pd.DataFrame(vrows)
    vsel = {w: float(V[f"ll_w{w}"].mean()) for w in h.W_VENUE_GRID}
    best = min(vsel.values()); w_mix = min(w for w, v in vsel.items() if v == best)
    exc_val = {"refused": int(sum(oom_val[a][t]["refused"] for a in val for t in range(h.T0, h.NEV))), "floored": int(sum(oom_val[a][t]["floored"] for a in val for t in range(h.T0, h.NEV))),
               "reset_after": int(sum(oom_val[a][t]["reset_after"] for a in val for t in range(h.T0, h.NEV)))}
    hrows = []
    for a in held:
        ix = by[a]
        for t in range(h.T0, h.NEV):
            y = cats[ix[t]]; pB = h.venue_base(cats[ix[:t]], g, T)["combined"]; o = oom_held[a][t]
            pO = None if o["refused"] else o["p"]; pM = h.mix(pB, pO, w_mix); e = np.eye(h.C)[y]
            hrows.append({"author": int(a), "t": t, "y_is_other": bool(y == h.K_TOP), "novel_raw_venue": bool(raw[ix[t]] not in set(raw[ix[:t]])), "oom_refused": o["refused"],
                          "oom_floored": o["floored"], "ll_B": float(-np.log(pB[y])), "ll_mix": float(-np.log(pM[y])),
                          "ll_oom": float(-np.log(pO[y])) if pO is not None else np.nan, "brier_B": float(((pB - e) ** 2).sum()), "brier_mix": float(((pM - e) ** 2).sum()),
                          "brier_oom": float(((pO - e) ** 2).sum()) if pO is not None else np.nan})
    H1 = pd.DataFrame(hrows)
    auths = np.array(sorted(held)); draws = boot_draws(len(auths))
    ok = H1[~H1.oom_refused]
    masks = {"known_raw_venue": ~H1.novel_raw_venue, "novel_raw_venue": H1.novel_raw_venue, "outcome_top20": ~H1.y_is_other, "outcome_OTHER": H1.y_is_other}
    res["B1_venue"] = {"validation_mean_logloss_by_w": vsel, "w_selected": w_mix, "validation_numerical_exceptions": exc_val,
                       "held_coverage": float(1 - H1.oom_refused.mean()), "held_numerical_exceptions": {"refused": int(H1.oom_refused.sum()), "floored": int(H1.oom_floored.sum())},
                       "held_logloss": {"pB": float(H1.ll_B.mean()), "pMix": float(H1.ll_mix.mean()), "pOOM_nonrefused": float(ok.ll_oom.mean()), "pB_nonrefused": float(ok.ll_B.mean())},
                       "held_brier": {"pB": float(H1.brier_B.mean()), "pMix": float(H1.brier_mix.mean()), "pOOM_nonrefused": float(ok.brier_oom.mean())},
                       "delta_logloss_mix_minus_B": paired(H1, "ll_mix", "ll_B", auths, draws), "delta_brier_mix_minus_B": paired(H1, "brier_mix", "brier_B", auths, draws),
                       "delta_logloss_oom_minus_B_nonrefused": paired(ok, "ll_oom", "ll_B", auths, draws),
                       "strata_mix_minus_B": strata(H1, "ll_mix", "ll_B", masks), "strata_oom_minus_B": strata(H1, "ll_oom", "ll_B", masks),
                       "note": "w=0 means this convex nested contrast adds nothing on source-validation; it does not show all longer-history information is absent"}
    # ================= B2 coupled semantic-history readouts
    Xl, Xh, Y = h.design_rows(Z, by, train)
    B_last, B_hist = h.ridge_fixed(Xl, Y), h.ridge_fixed(Xh, Y)
    grad = {"last": float(np.abs(h.objective_grad(Xl, Y, B_last)).max()), "hist": float(np.abs(h.objective_grad(Xh, Y, B_hist)).max())}

    def kpreds(a, wl, wh, permute_role=None):
        ix = by[a]; out = []
        for t in range(h.T0, h.NEV):
            pre = Z[ix[:t]]
            kp = h.keyed_predict(pre, pop, B_last, B_hist, wl, wh)
            r = {"author": int(a), "t": t, "novel_raw_venue": bool(raw[ix[t]] not in set(raw[ix[:t]])),
                 **{f"kmse_{k}": float(((kp[k] - Z[ix[t]]) ** 2).mean()) for k in ("static", "ar1", "combined", "coupled_last", "coupled_hist")},
                 **{f"sq_{k}_{j}": float((kp[k][j] - Z[ix[t]][j]) ** 2) for k in ("static", "coupled_last", "coupled_hist") for j in range(30)}}
            if permute_role:
                pp = h.permuted_prefix(pre, permute_role, a, t)
                ph_ = h.hist_readout(pp, B_hist, wh, mean_override=pre.mean(0))
                r["kmse_coupled_hist_permuted"] = float(((ph_ - Z[ix[t]]) ** 2).mean())
            out.append(r)
        return out
    vsel_k = {}
    for name, key in (("coupled_last", 0), ("coupled_hist", 1)):
        vals = {}
        for w in h.W_KEYED_GRID:
            wl, wh = (w, 0.0) if key == 0 else (0.0, w)
            vals[w] = float(np.mean([r[f"kmse_{name}"] for a in val for r in kpreds(a, wl, wh)]))
        bestv = min(vals.values())
        vsel_k[name] = {"validation_mse_by_w": vals, "w_selected": min(w for w, v in vals.items() if v == bestv)}
    wl, wh = vsel_k["coupled_last"]["w_selected"], vsel_k["coupled_hist"]["w_selected"]
    Vk = pd.DataFrame([r for a in val for r in kpreds(a, wl, wh, "source_validation")])
    Hk = pd.DataFrame([r for a in held for r in kpreds(a, wl, wh, "held")])
    km = {"known_raw_venue": ~Hk.novel_raw_venue, "novel_raw_venue": Hk.novel_raw_venue}
    facets = h.ph.FACETS
    per_facet = {k: {facets[j]: float(Hk[f"sq_{k}_{j}"].mean()) for j in range(30)} for k in ("static", "coupled_last", "coupled_hist")}
    vauths = np.array(sorted(val)); vdraws = np.random.default_rng(20260927).integers(0, len(vauths), size=(2000, len(vauths)))
    res["B2_keyed"] = {"fit_rows": int(len(Y)), "objective": "(1/n)||Y-XB||_F^2 + 1.0*||B||_F^2; loss sums the 30 output dims and averages rows; B=(X'X+n I)^-1 X'Y",
                       "max_abs_gradient_at_solution": grad, "B_frobenius": {"last": float(np.linalg.norm(B_last)), "hist": float(np.linalg.norm(B_hist))},
                       "B_hist_block_frobenius": {"last_block": float(np.linalg.norm(B_hist[:30])), "recent4_block": float(np.linalg.norm(B_hist[30:]))},
                       "selection": vsel_k,
                       "held_mse": {k: float(Hk[f"kmse_{k}"].mean()) for k in ("static", "ar1", "combined", "coupled_last", "coupled_hist")},
                       "deltas": {f"{k}_minus_{ref}": paired(Hk, f"kmse_{k}", f"kmse_{ref}", auths, draws) for k in ("coupled_last", "coupled_hist") for ref in ("static", "ar1", "combined")},
                       "incremental_history_hist_minus_last": paired(Hk, "kmse_coupled_hist", "kmse_coupled_last", auths, draws),
                       "strata_hist_minus_last": strata(Hk, "kmse_coupled_hist", "kmse_coupled_last", km), "strata_last_minus_static": strata(Hk, "kmse_coupled_last", "kmse_static", km),
                       "per_facet_held_mse": per_facet,
                       "permutation_control": {"redundant_because_w_hist_is_0": bool(wh == 0.0),
                                               "validation_permuted_minus_original": paired(Vk, "kmse_coupled_hist_permuted", "kmse_coupled_hist", vauths, vdraws),
                                               "held_permuted_minus_original": paired(Hk, "kmse_coupled_hist_permuted", "kmse_coupled_hist", auths, draws),
                                               "note": "tests the frozen readout's use of prefix order; not an exchangeable-null p-value or causal time dynamics"}}
    # ================= cold replay from saved coefficients (no refit)
    np.savez(OUT / "coefficients.npz", B_last=B_last, B_hist=B_hist, mu=D["mu"], sd=D["sd"], phi=pop["phi"], V5=pop["V5"], cmean=pop["cmean"],
             w_last=wl, w_hist=wh, w_mix=w_mix, g=g, T=T)
    with np.load(OUT / "coefficients.npz") as cz:
        Zc = (D["X"] - cz["mu"]) / cz["sd"]; popc = {"phi": cz["phi"], "V5": cz["V5"], "cmean": cz["cmean"]}
        cold = []
        for a in held:
            ix = by[a]
            for t in range(h.T0, h.NEV):
                kp = h.keyed_predict(Zc[ix[:t]], popc, cz["B_last"], cz["B_hist"], float(cz["w_last"]), float(cz["w_hist"]))
                cold.append([float(((kp[k] - Zc[ix[t]]) ** 2).mean()) for k in ("coupled_last", "coupled_hist")])
    cold = np.array(cold)
    res["cold_replay_max_abs"] = float(np.max(np.abs(cold - Hk[["kmse_coupled_last", "kmse_coupled_hist"]].to_numpy())))
    if res["cold_replay_max_abs"] > TOL:
        raise RuntimeError("cold replay failed")
    H1.to_csv(OUT / "private_held_venue_event_losses.csv", index=False)
    Hk[[c for c in Hk.columns if not c.startswith("sq_")]].to_csv(OUT / "private_held_keyed_event_losses.csv", index=False)
    Vk[[c for c in Vk.columns if not c.startswith("sq_")]].to_csv(OUT / "private_validation_keyed_event_losses.csv", index=False)
    V.to_csv(OUT / "private_validation_venue_event_losses.csv", index=False)
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/r08_history_increment.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print("done", round(res["seconds"], 1), "gates", gates, "w_mix", w_mix, "w_last", wl, "w_hist", wh)


if __name__ == "__main__":
    main()
