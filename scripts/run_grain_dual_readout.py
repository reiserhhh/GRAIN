#!/usr/bin/env python3
"""GRAIN-C1-02 runner (r4): dual readouts vs fixed pilot_pcr; estimator stage before truth."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import dual_readout as DR
from grain_research.construction import hidden_evidence_map as H

_spec = importlib.util.spec_from_file_location("legacy_c1", ROOT / "scripts/run_grain_hidden_evidence_map.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

SEEDS = (20261011, 20261012, 20261013)
ARMS = ("pilot_pcr", "lcmv_dual", "pilot_dual")
PROTOCOL = ROOT / "docs/GRAIN_C1_HIDDEN_EVIDENCE_MAP_PROTOCOL_V1_3_SUPPLEMENT.md"
TAU_F, B_JOINT, B_AUTH = 0.20, 30, 500


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(arm, view, aug, train_mask=None, k=None):
    if arm == "pilot_pcr":
        return H.fit_arm(arm, view, train_mask=train_mask, fixed={"k": k} if k else None)
    return DR.fit_dual(arm, aug if arm == "pilot_dual" else view, train_mask=train_mask)


def boot_views(view, aug, rng):
    n = H.N_SOURCE
    pick = rng.integers(0, n, n)
    xs = view.source_x.reshape(n, H.EVENTS, H.D)[pick].reshape(-1, H.D)
    tr = np.flatnonzero(view.pilot_writer < H.PILOT_TRAIN_WRITERS)
    writers = np.unique(view.pilot_writer[tr])
    pw = rng.choice(writers, len(writers), replace=True)
    rows = np.concatenate([tr[view.pilot_writer[tr] == w] for w in pw])
    neww = np.concatenate([np.full(int(np.sum(view.pilot_writer[tr] == w)), j) for j, w in enumerate(pw)])
    kw = {"source_x": xs, "pilot_x": view.pilot_x[rows], "pilot_writer": neww, "pilot_label": view.pilot_label[rows]}
    return (dataclasses.replace(view, **kw), dataclasses.replace(aug, **kw, pilot_z=aug.pilot_z[rows]),
            np.ones(len(rows), bool))


def estimator_stage(view, aug, seed, wi):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1611]))
    design_r2 = L.pilot_design_r2(view) if view.pilot_label is not None else None
    res = {}
    for arm in ARMS:
        r = fit(arm, view, aug)
        e = {"readout": r, "status": r.status, "flags": [], "info": r.info or {}}
        res[arm] = e
        if r.direction is None:
            continue
        kern = H.fit_moments(r.phi(view.source_x), view.source_z, view.source_author)
        e.update(fit=kern, scores=L.score_all(r, kern, view),
                 held=L.score_all(r, kern, view, mask=view.app_z[view.app_author == 0] != 1 / 3))
        g, se = L.validation_gain(r, view, rng)
        e.update(validation_gain=g, validation_gain_se=se, tau_T=max(0.15, 2 * se), design_r2=design_r2,
                 anchor_t=e["info"].get("anchor_t"))
        if design_r2 is not None and design_r2 > 0.5:
            e["flags"].append("CONFOUNDED_PILOT_DESIGN")
        draws, ranks, fails = {n: [] for n in H.TARGET_VECTORS}, [], 0
        k_main = e["info"].get("k")
        for b in range(B_JOINT):
            bv, ba, mask = boot_views(view, aug, np.random.default_rng(np.random.SeedSequence([seed, 1640, wi, b])))
            br = fit(arm, bv, ba, train_mask=mask, k=k_main)
            ranks.append((br.info or {}).get("rank_k", k_main))
            if br.direction is None:
                fails += 1
                continue
            bk = H.fit_moments(br.phi(bv.source_x), bv.source_z, bv.source_author)
            sc = L.score_all(br, bk, view)
            for n in H.TARGET_VECTORS:
                draws[n].append(sc[n]["score"])
        e["between_var"] = {n: np.var(np.array(v), axis=0, ddof=1) for n, v in draws.items()}
        e["boot_ranks"], e["boot_failures"] = ranks, fails
        e["nonconstant_min"] = {n: 0.1 * float(np.sqrt(t @ kern["Sigma"] @ t)) for n, t in H.TARGET_VECTORS.items()}
    return res


def evaluate(store, out, worlds_targets):
    rec, ivr, pair, side = [], [], [], []
    for (seed, world), (view, aug, truth, est) in store.items():
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1630]))
        idx = [rng.integers(0, H.N_APP, H.N_APP) for _ in range(B_AUTH)]
        arms = dict(est)
        for orc in (H.oracle_readouts(view, truth), DR.oracle_perp_style(view, truth)):
            k = H.fit_moments(orc.phi(view.source_x), view.source_z, view.source_author)
            arms[orc.arm] = {"readout": orc, "status": "PRIVILEGED", "flags": [], "fit": k, "info": {},
                             "scores": L.score_all(orc, k, view),
                             "held": L.score_all(orc, k, view, mask=view.app_z[view.app_author == 0] != 1 / 3)}
        slope = worlds_targets[world]["slope"]
        kinds = ["level", "fingerprint", "style", "wording", "material"] + (["slope"] if slope else [])
        iv = {k: L.intervene(truth, k, seed) for k in kinds}
        targets = list(worlds_targets[world]["targets"]) + (["held_m_1/3"] if slope else [])
        base_scores = {}
        for arm, e in arms.items():
            r = e["readout"]
            if "scores" not in e:
                rec.append({"world": world, "seed": seed, "arm": arm, "status": e["status"], "target": "ALL", "material": "-"})
                continue
            mech = H.readout_mechanism(r, truth.maps)
            mat = L.score_all(r, e["fit"], view, x=iv["material"][0])
            for name in targets:
                for material in ("same_family", "held_topic_family"):
                    if material == "held_topic_family" and name == "held_m_1/3":
                        continue
                    s = (mat if material == "held_topic_family" else (e["held"] if name == "held_m_1/3" else e["scores"]))[name]
                    score, tru = s["score"], L.latent_truth(truth.app_latent, name)
                    var = s["var"] + e.get("between_var", {}).get(name, np.zeros(H.N_APP))
                    sd = float(np.std(score))
                    rb = L.rboot(score, tru, rng, idx)
                    err, half = score - tru, 1.959964 * np.sqrt(var)
                    row = {"world": world, "seed": seed, "arm": arm, "status": e["status"], "flags": ";".join(e["flags"]),
                           "target": name, "material": material, "n": H.N_APP, "sd_score": sd,
                           "r": float(np.corrcoef(score, tru)[0, 1]), "r_lo": float(np.percentile(rb, 2.5)),
                           "r_hi": float(np.percentile(rb, 97.5)), "rmse": float(np.sqrt(np.mean(err ** 2))),
                           "mse": float(np.mean(err ** 2)), "bias": float(err.mean()),
                           "cal_slope": float(np.polyfit(score, tru, 1)[0]),
                           "coverage": float(np.mean(np.abs(err) <= half)), "mean_width": float(np.mean(2 * half))}
                    row.update({f"mech_{k}": v for k, v in mech.items()})
                    rec.append(row)
                    if material == "same_family":
                        base_scores[(arm, name)] = (score, tru)
            base = L.score_all(r, e["fit"], view)
            for name in worlds_targets[world]["targets"]:
                tkind = "slope" if name == "delta" else "level"
                ch_t = L.score_all(r, e["fit"], view, x=iv[tkind][0])[name]["ols"] - base[name]["ols"]
                gain = float(ch_t.mean())
                gb = np.array([ch_t[i].mean() for i in idx])
                ivr.append({"world": world, "seed": seed, "arm": arm, "status": e["status"], "flags": ";".join(e["flags"]),
                            "target": name, "intervention": f"target_{tkind}", "intended_change": 1.0,
                            "statistic": "gain_T", "value": gain, "ci_lo": float(np.percentile(gb, 2.5)),
                            "ci_hi": float(np.percentile(gb, 97.5)), "analytic": mech["gain_expression"],
                            "tolerance": e.get("tau_T"),
                            "within_tolerance": (abs(gain - 1) <= e["tau_T"]) if e.get("tau_T") else None})
                for kind in ("fingerprint", "style", "wording", "material"):
                    ch = L.score_all(r, e["fit"], view, x=iv[kind][0])[name]["ols"] - base[name]["ols"]
                    ratio = float(np.sqrt(np.mean(ch ** 2)) / abs(ch_t.mean()))
                    rbs = np.array([np.sqrt(np.mean(ch[i] ** 2)) / abs(ch_t[i].mean()) for i in idx])
                    if kind == "fingerprint":
                        df = (iv["fingerprint"][1]["f"] - truth.app_latent["f"]) @ truth.maps["U_f"].T @ r.direction / r.unit
                        analytic = float(np.sqrt(np.mean(df ** 2))) if name == "level" else 0.0
                    elif kind == "style":
                        analytic = 0.0 if name == "level" else float(abs(2 * mech["style_response_per_sd"]))
                    else:
                        analytic = None
                    rf = kind in ("fingerprint", "style")
                    ivr.append({"world": world, "seed": seed, "arm": arm, "status": e["status"], "flags": ";".join(e["flags"]),
                                "target": name, "intervention": kind, "intended_change": 0.0,
                                "statistic": "R_F" if rf else "rms_ratio_to_target", "value": ratio,
                                "ci_lo": float(np.percentile(rbs, 2.5)), "ci_hi": float(np.percentile(rbs, 97.5)),
                                "mean_change_ols": float(ch.mean()), "rms_change_ols": float(np.sqrt(np.mean(ch ** 2))),
                                "analytic_rms_change_ols": analytic, "tolerance": TAU_F if rf else None,
                                "within_tolerance": (ratio <= TAU_F) if rf else None})
            if arm == "pilot_dual" and slope:
                sh = e["info"]["style_hat"]
                y = (view.app_x @ sh).reshape(H.N_APP, -1)
                zz = view.app_z[view.app_author == 0]
                sl = y @ (H.design(zz) @ np.linalg.inv(H.design(zz).T @ H.design(zz)))[:, 1]
                side.append({"world": world, "seed": seed, "coordinate": "pilot_dual s_hat'x per-author slope",
                             "corr_true_style_r": float(np.corrcoef(sl, truth.app_latent["s"])[0, 1]),
                             "corr_true_slope_b": float(np.corrcoef(sl, truth.app_latent["b"])[0, 1]),
                             "cos_s_hat_true_u_s": float(abs(sh @ truth.maps["u_s"]) / np.linalg.norm(sh)),
                             "note": "numeric side coordinate; not a named or pure psychological score"})
        for (arm, name), (score, tru) in base_scores.items():
            if arm == "pilot_pcr" or ("pilot_pcr", name) not in base_scores:
                continue
            s0, _ = base_scores[("pilot_pcr", name)]
            dr = [np.corrcoef(score[i], tru[i])[0, 1] - np.corrcoef(s0[i], tru[i])[0, 1] for i in idx]
            dm = [np.mean((score[i] - tru[i]) ** 2) - np.mean((s0[i] - tru[i]) ** 2) for i in idx]
            pair.append({"world": world, "seed": seed, "arm": arm, "target": name, "reference": "pilot_pcr",
                         "delta_r": float(np.corrcoef(score, tru)[0, 1] - np.corrcoef(s0, tru)[0, 1]),
                         "delta_r_lo": float(np.percentile(dr, 2.5)), "delta_r_hi": float(np.percentile(dr, 97.5)),
                         "delta_mse": float(np.mean((score - tru) ** 2) - np.mean((s0 - tru) ** 2)),
                         "delta_mse_lo": float(np.percentile(dm, 2.5)), "delta_mse_hi": float(np.percentile(dm, 97.5))})
    for name, rows in (("recovery_by_mechanism.csv", rec), ("intervention_checks.csv", ivr),
                       ("paired_differences.csv", pair), ("side_coordinates.csv", side)):
        write_csv(out / name, rows)


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(out, worlds, builder, worlds_targets):
    out.mkdir(parents=True, exist_ok=False)
    store, digests, tol, cold, diag = {}, {}, [], [], []
    for seed in SEEDS:
        for wi, world in enumerate(worlds):
            view, aug, truth = builder(seed, world)
            digests[f"{world}/{seed}"] = {"base_view": H.view_digest(view), "augmented_pilot_z_sha256":
                                          hashlib.sha256(np.ascontiguousarray(aug.pilot_z).tobytes()).hexdigest()}
            est = estimator_stage(view, aug, seed, wi)
            store[(seed, world)] = (view, aug, truth, est)
            for arm, e in est.items():
                i = e["info"]
                tol.append({"world": world, "seed": seed, "arm": arm, "status": e["status"], "flags": e["flags"],
                            "anchor_t": e.get("anchor_t"), "pilot_design_r2": e.get("design_r2"),
                            "validation_gain": e.get("validation_gain"), "validation_gain_se": e.get("validation_gain_se"),
                            "tau_T": e.get("tau_T"), "tau_F": TAU_F, "nonconstant_min": e.get("nonconstant_min"),
                            "k": i.get("k"), "lambda": None})
                diag.append({"world": world, "seed": seed, "arm": arm, "status": e["status"], "rank_k": i.get("rank_k"),
                             "rank_tau": i.get("rank_tau"), "eigenvalue_min": i.get("eigenvalue_min"),
                             "eigenvalues_top": json.dumps([round(x, 5) for x in i.get("eigenvalues_top", [])]),
                             "range_fraction": i.get("range_fraction"), "anchor_t": i.get("anchor_t"),
                             "style_eigenvalue": i.get("style_eigenvalue"), "pcr_k": i.get("k"),
                             "boot_ranks": json.dumps(e.get("boot_ranks")), "boot_failures": e.get("boot_failures")})
                if "scores" in e:
                    for name in worlds_targets[world]["targets"]:
                        s = e["scores"][name]
                        for u in range(H.N_APP):
                            cold.append([world, seed, arm, name, u, s["score"][u],
                                         s["var"] + e["between_var"][name][u], s["ols"][u]])
    (out / "frozen_tolerances.json").write_text(json.dumps(tol, indent=1, default=float))
    with open(out / "cold_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["world", "seed", "arm", "target", "app_author_index", "score", "interval_var", "ols"])
        w.writerows(cold)
    write_csv(out / "rank_anchor_diagnostics.csv", diag)
    frozen = sha(out / "frozen_tolerances.json")
    evaluate(store, out, worlds_targets)
    return digests, frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grain_construction_v1/C1/hidden_map_r4")
    args = ap.parse_args()
    out = ROOT / args.out
    t0, c0 = time.time(), time.process_time()
    proto = sha(PROTOCOL)
    code = {p: sha(ROOT / p) for p in ("grain_research/construction/dual_readout.py",
                                       "grain_research/construction/hidden_evidence_map.py",
                                       "scripts/run_grain_dual_readout.py", "scripts/run_grain_hidden_evidence_map.py")}
    wt = {w: {"slope": c["slope"], "targets": c["targets"]} for w, c in H.WORLDS.items()}
    digests, frozen = run(out, list(H.WORLDS), DR.build_world_augmented, wt)
    (out / "hidden_map_protocol.md").write_text(PROTOCOL.read_text() + f"\n\n<!-- copied; sha256 {proto} -->\n")
    bwt = {"W5_correlated_rho06": {"slope": True, "targets": ("delta", "level")}}
    bdig, bfrozen = run(out / "correlated_boundary", list(bwt), lambda s, w: DR.build_world_correlated(s), bwt)
    contract = {
        "task": "GRAIN-C1-02", "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": proto,
        "base_protocol": "docs/GRAIN_C1_HIDDEN_EVIDENCE_MAP_PROTOCOL_V1.md", "code_sha256": code,
        "frozen_tolerances_sha256_before_evaluation": frozen, "boundary_frozen_tolerances_sha256": bfrozen,
        "seeds": SEEDS, "arms": ARMS, "primary_worlds": list(H.WORLDS), "boundary_world": "W5_correlated_rho06 (rho .6)",
        "estimator_inputs": {"pilot_pcr": list(H.ESTIMATOR_FIELDS), "lcmv_dual": list(H.ESTIMATOR_FIELDS),
                             "pilot_dual": list(DR.AUGMENTED_FIELDS)},
        "augmented_field": {"pilot_z": "recorded assigned pilot condition, exported by assigned_pilot_conditions(seed) before the evaluator boundary"},
        "evaluator_only": ["maps", "a", "b", "f", "s", "topic", "eps", "noise", "true expression", "pilot latents"],
        "privileged_arms": ["oracle_map", "oracle_perp_style"], "view_digests": digests, "boundary_view_digests": bdig,
        "stage_order": "estimator stage wrote frozen_tolerances.json, cold_scores.csv, rank_anchor_diagnostics.csv before truth was read",
        "pilot_purity_assumption": "labels independent of writer fingerprint/style; conditions assigned independently of labels (declared, not proven by writer R2)",
    }
    (out / "source_application_contract.json").write_text(json.dumps(contract, indent=1))
    (out / "estimator_manifest.json").write_text(json.dumps({"role": "estimator", "inputs_by_arm": contract["estimator_inputs"],
        "augmented_field": contract["augmented_field"], "frozen_tolerances_sha256": frozen, "view_digests": digests}, indent=1))
    (out / "evaluator_manifest.json").write_text(json.dumps({"role": "evaluator", "reads_hidden": contract["evaluator_only"],
        "privileged_reference_arms": contract["privileged_arms"],
        "rule": "hidden truth is not a feature, selector, cutoff, unit or interval input"}, indent=1))
    run_log = {"total_wall_s": time.time() - t0, "total_cpu_s": time.process_time() - c0, "python": sys.version,
               "numpy": np.__version__, "platform": platform.platform(),
               "threads": {v: os.environ[v] for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    (out / "run_log.json").write_text(json.dumps(run_log, indent=1))
    print(json.dumps(run_log, indent=1))


if __name__ == "__main__":
    main()
