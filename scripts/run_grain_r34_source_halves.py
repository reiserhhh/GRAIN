#!/usr/bin/env python
"""R34 source-half structural reconstruction of T1/T2/T3 (docs/GRAIN_R34_SOURCE_HALF_RECONSTRUCTION_PROTOCOL_V1.md).
True disjoint-source POINT_FIT_PLUGIN refits; fixed validation panels; original units. CPU only, 0 neural calls."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import itertools
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
from grain_research.gptpro_foundation_v3 import f05_recovery as f05  # noqa: E402
from grain_research.gptpro_foundation_v3 import f16_t1_source as t1  # noqa: E402
from grain_research.gptpro_foundation_v3 import f16_t1_text as tt  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R34_SOURCE_HALF_RECONSTRUCTION_PROTOCOL_V1.md"
PROTOCOL_SHA = "f8df28610c6215113e590b9a9380f644cbed0cdea82329e410880cba9d08fb54"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "F16/source_half_reconstruction_r1"
SEEDS = (20260915, 20260916, 20260917)
NSPLIT, PLAN = 20, (1.0, -1.0, 1.0, -1.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != h:
            raise RuntimeError(f"binding mismatch {rel}")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def stats_validation(rows):
    """Same arithmetic as f16_t1_source.sufficient; differs only in accepting role=validation (fixed scoring panel)."""
    counts, sums, squares = np.zeros(4), np.zeros(4), np.zeros(4)
    for r in rows:
        if r["role"] != "validation" or r["world"] != "L":
            raise ValueError("validation panel rows only")
        k = t1.CONDITIONS.index(r["z"])
        counts[k] += 1; sums[k] += r["y"]; squares[k] += r["y"] ** 2
    return counts, sums, squares


def t3_mean_of_four(res, logs):
    """Mean of the 4 plan forecasts and the variance of that mean with full future covariance (plug-in, includes future noise)."""
    rho = res["rho"]; q, r = np.exp(2 * np.asarray(logs)); V = res["state_variance"]
    K = len(PLAN); C = np.zeros((K, K))
    for k in range(1, K + 1):
        for l in range(k, K + 1):
            c = rho ** (k + l) * V + q * sum(rho ** (l - k + 2 * j) for j in range(k)) + (r if k == l else 0.0)
            C[k - 1, l - 1] = C[l - 1, k - 1] = c
    m = float(np.mean([f["mean"] for f in res["forecasts"]]))
    return m, float(C.sum() / K ** 2)


def splits_for(ids, si):
    out = []
    for sp in range(NSPLIT):
        perm = np.random.default_rng([19001, si, sp]).permutation(len(ids))
        a = sorted(ids[i] for i in perm[:len(ids) // 2]); b = sorted(ids[i] for i in perm[len(ids) // 2:])
        out.append({"split": sp, "rng_key": [19001, si, sp], "A": a, "B": b})
    return out


def summarize_diffs(pa, va, pb, vb):
    d = pa - pb; z = d / np.sqrt((va + vb) / 2)
    return {"mean_abs_d": float(np.abs(d).mean()), "rms_d": float(np.sqrt(np.mean(d ** 2))), "mean_abs_z": float(np.abs(z).mean()),
            "q95_abs_z": float(np.quantile(np.abs(z), .95)), "max_abs_z": float(np.abs(z).max())}


def dist(xs):
    x = np.asarray(xs, float)
    return {"median": float(np.median(x)), "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9)), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("source_half_reconstruction_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "model_label": "POINT_FIT_PLUGIN (ensemble count 1 per half; no bootstrap fabricated)",
           "seeds": {}, "failures": [], "fit_counts": {"T1_half": 0, "dyn_half": 0, "T1_full": 0, "dyn_full": 0}}
    splits_all, per_author_rows = {}, []
    for si, seed in enumerate(SEEDS):
        S = {}
        # ---------------- T1
        src = jl(V3 / f"F04/source_full_r1/L_{seed}/source.jsonl"); val = jl(V3 / f"F04/source_full_r1/L_{seed}/validation.jsonl")
        by = {}
        for r in src:
            by.setdefault(r["author_id"], []).append(r)
        ids = sorted(by); assert len(ids) == 300 and all(by[a][0]["role"] == "source" for a in ids)
        st = [t1.sufficient(by[a]) for a in ids]
        STATS = tuple(np.array([s[i] for s in st]) for i in range(3))
        vby = {}
        for r in val:
            vby.setdefault(r["author_id"], []).append(r)
        vids = sorted(vby); assert not set(vids) & set(ids)
        vstats = {a: stats_validation(vby[a]) for a in vids}
        sp_t1 = splits_for(ids, si)
        full = t1.fit_source(STATS, np.arange(300)); res["fit_counts"]["T1_full"] += 1
        boot = json.loads((V3 / f"F16/t1_source_full_r1/{seed}/bundle.json").read_text())["source_models"]
        boot_mu_sd = np.std([m["mu"] for m in boot], axis=0); boot_sig2_sd = float(np.std([m["sigma2"] for m in boot]))
        boot_cvar_sd = float(np.std([t1.CONTRAST @ np.array(m["Sigma"]) @ t1.CONTRAST for m in boot]))
        c = t1.CONTRAST

        def t1_panel(model):
            out = []
            for a in vids:
                cnt, sm, _ = vstats[a]
                mean, cov = t1.posterior(cnt, sm, model)
                out.append((float(c @ mean), float(c @ cov @ c)))
            return np.array(out)
        Pfull = t1_panel(full)
        t1_split = []
        for sp in sp_t1:
            iA = np.array([ids.index(a) for a in sp["A"]]); iB = np.array([ids.index(a) for a in sp["B"]])
            assert not set(sp["A"]) & set(sp["B"])
            mA, mB = t1.fit_source(STATS, iA), t1.fit_source(STATS, iB); res["fit_counts"]["T1_half"] += 2
            PA, PB = t1_panel(mA), t1_panel(mB)
            SA, SB = np.array(mA["Sigma"]), np.array(mB["Sigma"]); muA, muB = np.array(mA["mu"]), np.array(mB["mu"])
            row = {"split": sp["split"], "d_mu": (muA - muB).tolist(), "d_mu_over_boot_sd": ((muA - muB) / boot_mu_sd).tolist(),
                   "d_contrast_mean": float(c @ (muA - muB)), "sigma_rel_frob": float(np.linalg.norm(SA - SB) / ((np.linalg.norm(SA) + np.linalg.norm(SB)) / 2)),
                   "sigma2_ratio": mA["sigma2"] / mB["sigma2"], "d_sigma2_over_boot_sd": (mA["sigma2"] - mB["sigma2"]) / boot_sig2_sd,
                   "contrast_var_ratio": float((c @ SA @ c) / (c @ SB @ c)), "d_contrast_var_over_boot_sd": float((c @ SA @ c - c @ SB @ c) / boot_cvar_sd),
                   "floor_hits": [int(np.sum(np.array(mA["raw_eigenvalues"]) < 1e-6)), int(np.sum(np.array(mB["raw_eigenvalues"]) < 1e-6))],
                   **{f"target_{k}": v for k, v in summarize_diffs(PA[:, 0], PA[:, 1], PB[:, 0], PB[:, 1]).items()},
                   "half_vs_full_mean_abs": float((np.abs(PA[:, 0] - Pfull[:, 0]).mean() + np.abs(PB[:, 0] - Pfull[:, 0]).mean()) / 2)}
            t1_split.append(row)
            for j, a in enumerate(vids):
                per_author_rows.append({"seed": seed, "target": "T1", "split": sp["split"], "author": a, "pred_A": PA[j, 0], "var_A": PA[j, 1], "pred_B": PB[j, 0], "var_B": PB[j, 1],
                                        "pred_full": Pfull[j, 0], "var_full": Pfull[j, 1]})
        S["T1"] = {"full_fit": {"mu": full["mu"], "Sigma": full["Sigma"], "sigma2": full["sigma2"], "contrast_var": float(c @ np.array(full["Sigma"]) @ c)},
                   "boot_ref_sd": {"mu": boot_mu_sd.tolist(), "sigma2": boot_sig2_sd, "contrast_var": boot_cvar_sd}, "splits": t1_split,
                   "distribution": {k: dist([r[k] for r in t1_split]) for k in ("d_contrast_mean", "sigma_rel_frob", "sigma2_ratio", "contrast_var_ratio",
                                                                                  "target_mean_abs_d", "target_mean_abs_z", "target_q95_abs_z", "half_vs_full_mean_abs")}}
        splits_all[f"T1|{seed}"] = sp_t1
        # ---------------- T2 / T3 (shared half fits)
        src5 = jl(V3 / f"F05/source_full_r2/core_{seed}/source.jsonl"); val5 = jl(V3 / f"F05/source_full_r2/core_{seed}/validation.jsonl")
        by5 = {}
        for r in src5:
            by5.setdefault(r["author_id"], []).append(r)
        ids5 = sorted(by5); assert len(ids5) == 300 and all(r["role"] == "source" for r in src5)
        vby5 = {}
        for r in val5:
            vby5.setdefault(r["author_id"], []).append(r)
        vids5 = sorted(vby5); assert not set(vids5) & set(ids5)
        prefix = {a: sorted([r for r in vby5[a] if r["t"] < 48], key=lambda r: r["t"]) for a in vids5}
        sp_dyn = splits_for(ids5, si)
        boot5 = json.loads((V3 / f"F16/t2_cold_full_r1/{seed}/bundle.json").read_text())["source_models"]
        bmu = np.array([m["model"]["prior"]["mu"] for m in boot5]); bls = np.array([m["model"]["log_sigma"] for m in boot5])
        b_mu_sd = bmu.std(0); b_ls_sd = bls.std(0); b_rho_sd = float((.99 * np.tanh(bmu[:, 2])).std())

        def dyn_fit(rows):
            try:
                return f05.fit_source_dynamics(rows), None
            except Exception as exc:  # recorded, never hidden
                return None, repr(exc)

        def dyn_panel(model):
            out = []
            for a in vids5:
                r = f05.score_prefix(prefix[a], model, future_conditions=PLAN)
                m3, v3 = t3_mean_of_four(r, model["log_sigma"])
                out.append((float(r["state_mean"]), float(r["state_variance"]), float(r["psi"][0]), m3, v3, bool(r["fit_success"])))
            return np.array(out, dtype=float)
        fullm, err = dyn_fit(src5); res["fit_counts"]["dyn_full"] += 1
        if err:
            res["failures"].append({"seed": seed, "fit": "dyn_full", "error": err})
        Dfull = dyn_panel(fullm)
        dyn_split = []
        for sp in sp_dyn:
            A = set(sp["A"]); B = set(sp["B"]); assert not A & B
            mA, eA = dyn_fit([r for r in src5 if r["author_id"] in A]); mB, eB = dyn_fit([r for r in src5 if r["author_id"] in B]); res["fit_counts"]["dyn_half"] += 2
            if eA or eB:
                res["failures"].append({"seed": seed, "split": sp["split"], "A_error": eA, "B_error": eB}); continue
            DA, DB = dyn_panel(mA), dyn_panel(mB)
            muA, muB = np.array(mA["prior"]["mu"]), np.array(mB["prior"]["mu"]); SA, SB = np.array(mA["prior"]["Sigma"]), np.array(mB["prior"]["Sigma"])
            lsA, lsB = np.array(mA["log_sigma"]), np.array(mB["log_sigma"])
            row = {"split": sp["split"], "d_mu_a_b_eta": (muA - muB).tolist(), "d_mu_over_boot_sd": ((muA - muB) / b_mu_sd).tolist(),
                   "d_rho_prior_mean": float(.99 * np.tanh(muA[2]) - .99 * np.tanh(muB[2])), "d_rho_over_boot_sd": float((.99 * np.tanh(muA[2]) - .99 * np.tanh(muB[2])) / b_rho_sd),
                   "sigma_rel_frob": float(np.linalg.norm(SA - SB) / ((np.linalg.norm(SA) + np.linalg.norm(SB)) / 2)),
                   "d_log_sigma": (lsA - lsB).tolist(), "d_log_sigma_over_boot_sd": ((lsA - lsB) / b_ls_sd).tolist(),
                   "noise_success": [all(h["noise_success"] for h in mA["rounds"]), all(h["noise_success"] for h in mB["rounds"])],
                   "personal_fit_failures_panel": int((DA[:, 5] < .5).sum() + (DB[:, 5] < .5).sum())}
            for lab, (ci, cv) in (("T2_state", (0, 1)), ("T3_forecast", (3, 4))):
                row.update({f"{lab}_{k}": v for k, v in summarize_diffs(DA[:, ci], DA[:, cv], DB[:, ci], DB[:, cv]).items()})
                row[f"{lab}_half_vs_full_mean_abs"] = float((np.abs(DA[:, ci] - Dfull[:, ci]).mean() + np.abs(DB[:, ci] - Dfull[:, ci]).mean()) / 2)
            row["T2_intercept_mean_abs_d"] = float(np.abs(DA[:, 2] - DB[:, 2]).mean())
            row["T2_sum_intercept_state_mean_abs_d"] = float(np.abs((DA[:, 2] + DA[:, 0]) - (DB[:, 2] + DB[:, 0])).mean())
            dyn_split.append(row)
            for j, a in enumerate(vids5):
                per_author_rows.append({"seed": seed, "target": "T2", "split": sp["split"], "author": a, "pred_A": DA[j, 0], "var_A": DA[j, 1], "pred_B": DB[j, 0], "var_B": DB[j, 1],
                                        "pred_full": Dfull[j, 0], "var_full": Dfull[j, 1], "intercept_A": DA[j, 2], "intercept_B": DB[j, 2], "intercept_full": Dfull[j, 2]})
                per_author_rows.append({"seed": seed, "target": "T3", "split": sp["split"], "author": a, "pred_A": DA[j, 3], "var_A": DA[j, 4], "pred_B": DB[j, 3], "var_B": DB[j, 4],
                                        "pred_full": Dfull[j, 3], "var_full": Dfull[j, 4]})
            (OUT / "dyn_half_fits").mkdir(exist_ok=True)
            (OUT / "dyn_half_fits" / f"{seed}_split{sp['split']:02d}.json").write_text(json.dumps({"A": {k: mA[k] for k in ("prior", "log_sigma", "source_authors")},
                                                                                                    "B": {k: mB[k] for k in ("prior", "log_sigma", "source_authors")}}) + "\n")
        S["T2T3"] = {"full_fit": {"prior": fullm["prior"], "log_sigma": fullm["log_sigma"]}, "boot_ref_sd": {"mu_a_b_eta": b_mu_sd.tolist(), "log_sigma": b_ls_sd.tolist(), "rho": b_rho_sd},
                     "splits": dyn_split,
                     "distribution": {k: dist([r[k] for r in dyn_split]) for k in ("d_rho_prior_mean", "sigma_rel_frob", "T2_state_mean_abs_d", "T2_state_mean_abs_z", "T2_state_q95_abs_z",
                                                                                    "T3_forecast_mean_abs_d", "T3_forecast_mean_abs_z", "T3_forecast_q95_abs_z", "T2_intercept_mean_abs_d",
                                                                                    "T2_state_half_vs_full_mean_abs", "T3_forecast_half_vs_full_mean_abs")}}
        splits_all[f"T2T3|{seed}"] = sp_dyn
        res["seeds"][str(seed)] = S
        print(seed, "done", round(time.time() - t0, 1), "s", flush=True)
    # ---------------- text channel extension
    nli = pd.read_csv(V3 / "F08/nli_reader_r1/reader_scores.csv").set_index("material_id")
    grain = pd.read_csv(V3 / "F11/grain_comment_reader_r1/reader_scores.csv").set_index("material_id")
    ident = ["text", "role", "family", "pole", "qualification", "subject_status", "variant"]
    assert nli[ident].equals(grain[ident])
    bank = nli[ident].copy(); bank["pole_readout"] = nli.pole_score; bank["nli_qualification"] = nli.qualification_score; bank["grain_qualification"] = grain.qualification_score
    bank = bank.reset_index()
    cal = bank[bank.role == "validation"]; app = bank[bank.role == "application"]
    cores = sorted(set(zip(cal.pole, cal.subject_status)))
    strata = {}
    for pcore in cores:
        q = int(cal[(cal.pole == pcore[0]) & (cal.subject_status == pcore[1])].qualification.iloc[0])
        strata.setdefault((pcore[0], q), []).append(pcore)
    assert all(len(v) == 2 for v in strata.values()) and len(strata) == 4
    keys = sorted(strata)
    parts = []
    for choice in itertools.product([0, 1], repeat=4):
        A = {strata[k][c] for k, c in zip(keys, choice)}
        if choice[0] == 1:
            continue      # unordered: fix the first stratum's assignment -> 8 distinct partitions
        parts.append(sorted(A))
    text = {"calibration_cores": [list(x) for x in cores], "partitions": len(parts), "arms": {}}
    for arm in ("nli_joint", "hybrid_joint"):
        key = "nli_qualification" if arm == "nli_joint" else "grain_qualification"
        rows_out = []
        for pi, A in enumerate(parts):
            inA = cal.apply(lambda r: (r.pole, r.subject_status) in set(A), axis=1)
            cA = tt.channel_counts(cal[inA].to_dict("records"), arm); cB = tt.channel_counts(cal[~inA].to_dict("records"), arm)
            PA = (cA + .5) / (cA + .5).sum(2, keepdims=True); PB = (cB + .5) / (cB + .5).sum(2, keepdims=True)
            tv = .5 * np.abs(PA - PB).sum(2)
            obs = [(int(r.pole), int(r.qualification), tt.category(r.pole_readout, getattr(r, key))) for r in app.itertuples()]
            lA = np.array([np.log(PA[b, q, k]) for b, q, k in obs]); lB = np.array([np.log(PB[b, q, k]) for b, q, k in obs])
            rows_out.append({"partition": pi, "half_A_cores": [list(x) for x in A], "tv_by_stratum": tv.tolist(), "tv_mean": float(tv.mean()), "tv_max": float(tv.max()),
                             "app_logprob_A": float(lA.mean()), "app_logprob_B": float(lB.mean()), "app_logprob_mean_abs_diff": float(np.abs(lA - lB).mean())})
        text["arms"][arm] = {"partitions": rows_out, "tv_mean_dist": dist([r["tv_mean"] for r in rows_out]), "tv_max_dist": dist([r["tv_max"] for r in rows_out]),
                             "app_logprob_abs_diff_dist": dist([r["app_logprob_mean_abs_diff"] for r in rows_out])}
    text["author_level_text_T1_T2_T3"] = "NOT_SUPPORTED_BY_EXISTING_BUILDER: text bundles embed channel draws built at bundle time; substituting a half channel requires a new bundle build path (not done); no numeric score substituted"
    res["text_channel"] = text
    pd.DataFrame(per_author_rows).to_csv(OUT / "per_author_predictions.csv", index=False)
    (OUT / "splits.json").write_text(json.dumps(splits_all) + "\n")
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "protocol_sha256": PROTOCOL_SHA, "fit_counts": res["fit_counts"], "failures": len(res["failures"])}, indent=1) + "\n")
    print("done", round(res["seconds"], 1), res["fit_counts"], "failures", len(res["failures"]))


if __name__ == "__main__":
    main()
