#!/usr/bin/env python3
"""GRAIN-C1-01 runner: estimator stage (no truth) then evaluator stage."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import hidden_evidence_map as H  # noqa: E402

SEEDS = (20261002, 20261003, 20261004)
ARMS = ("pilot_ridge", "pilot_pcr", "keyed", "svd1", "stability_author", "stability_response")
PROTOCOL = ROOT / "docs/GRAIN_C1_HIDDEN_EVIDENCE_MAP_PROTOCOL_V1.md"
KEYED_DECLARED_SIGN = False
TAU_F = 0.20
B_PILOT = 30
B_AUTH = 500


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_of(readout):
    i = readout.info or {}
    if readout.arm == "pilot_ridge":
        return {"lam": i["lambda"]}
    if readout.arm == "pilot_pcr":
        return {"k": i["k"], "vt": i["vt"]}
    if readout.arm == "svd1":
        return {"vt": i["vt"]}
    if readout.arm in ("stability_author", "stability_response"):
        return {"d": i["d"]}
    if readout.arm == "np_ridge":
        return {"lam": i["lambda"], "nuisance": i["nuisance"]}
    if readout.arm == "np_pcr":
        return {"k": i["k"], "vt": i["vt"], "nuisance": i["nuisance"]}
    return {}


def pilot_design_r2(view):
    tr = view.pilot_writer < H.PILOT_TRAIN_WRITERS
    y, w = view.pilot_label[tr], view.pilot_writer[tr]
    means = np.array([y[w == k].mean() for k in np.unique(w)])
    fitted = means[np.searchsorted(np.unique(w), w)]
    return float(1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))


def validation_gain(readout, view, rng):
    va = view.pilot_writer >= H.PILOT_TRAIN_WRITERS
    phi, lab = readout.phi(view.pilot_x[va]), view.pilot_label[va]

    def slope(idx):
        xm = np.column_stack([np.ones(len(idx)), lab[idx]])
        return float(np.linalg.lstsq(xm, phi[idx], rcond=None)[0][1])

    g = slope(np.arange(len(lab)))
    boots = [slope(rng.integers(0, len(lab), len(lab))) for _ in range(500)]
    return g, float(np.std(boots, ddof=1))


def score_all(readout, fit, view, x=None, mask=None):
    x = view.app_x if x is None else x
    return H.score_targets(readout, fit, x, view.app_z, view.app_author, mask)


def pilot_bootstrap(arm, readout, view, seed):
    """Refit unit (and supervised direction) on resampled pilot train writers."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1610, ARMS.index(arm)]))
    tr = np.flatnonzero(view.pilot_writer < H.PILOT_TRAIN_WRITERS)
    writers = np.unique(view.pilot_writer[tr])
    out = []
    for _ in range(B_PILOT):
        pick = rng.choice(writers, len(writers), replace=True)
        rows = np.concatenate([tr[view.pilot_writer[tr] == w] for w in pick])
        newwriter = np.concatenate([np.full(np.sum(view.pilot_writer[tr] == w), j) for j, w in enumerate(pick)])
        bview = dataclasses.replace(view, pilot_x=view.pilot_x[rows], pilot_writer=newwriter,
                                    pilot_label=view.pilot_label[rows])
        r = H.fit_arm(arm, bview, train_mask=np.ones(len(rows), bool), fixed=fixed_of(readout))
        fit = H.fit_moments(r.phi(view.source_x), view.source_z, view.source_author)
        out.append((r, fit))
    return out


def estimator_stage(view, seed, world):
    """Everything here reads only the EstimatorView."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1611]))
    res = {}
    design_r2 = pilot_design_r2(view) if view.pilot_label is not None else None
    for arm in ARMS:
        r = H.fit_arm(arm, view, keyed_declared_sign=KEYED_DECLARED_SIGN)
        entry = {"readout": r, "status": r.status, "flags": []}
        if r.direction is None:
            res[arm] = entry
            continue
        fit = H.fit_moments(r.phi(view.source_x), view.source_z, view.source_author)
        scores = score_all(r, fit, view)
        held = score_all(r, fit, view, mask=view.app_z[view.app_author == 0] != 1 / 3)
        entry.update(fit=fit, scores=scores, held=held)
        if view.pilot_label is not None:
            g, se = validation_gain(r, view, rng)
            entry["validation_gain"], entry["validation_gain_se"] = g, se
            entry["tau_T"] = max(0.15, 2 * se)
            entry["design_r2"] = design_r2
            if design_r2 > 0.5:
                entry["flags"].append("CONFOUNDED_PILOT_DESIGN")
            boots = pilot_bootstrap(arm, r, view, seed)
            between = {}
            for name in H.TARGET_VECTORS:
                arr = np.array([score_all(br, bf, view)[name]["score"] for br, bf in boots])
                between[name] = arr.var(axis=0, ddof=1)
            entry["between_var"] = between
            entry["anchor_t"] = r.info.get("anchor_t")
        for name in H.TARGET_VECTORS:
            t = H.TARGET_VECTORS[name]
            entry.setdefault("nonconstant_min", {})[name] = 0.1 * float(np.sqrt(t @ fit["Sigma"] @ t))
        res[arm] = entry
    return res


def rboot(score, truth, rng, idx):
    out = []
    for i in idx:
        out.append(np.corrcoef(score[i], truth[i])[0, 1])
    return np.array(out)


def latent_truth(lat, name):
    a, b = lat["a"], lat["b"]
    return {"level": a, "delta": 2 * b, "held_m_1/3": a + b / 3}[name]


def intervene(truth, kind, seed):
    lat = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in truth.app_latent.items()}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1620, ["level", "slope", "fingerprint", "style", "wording", "material"].index(kind)]))
    n = len(lat["a"])
    if kind == "level":
        lat["a"] = lat["a"] + 1.0
    elif kind == "slope":
        lat["b"] = lat["b"] + 0.5
    elif kind == "fingerprint":
        u = rng.normal(size=(n, H.KF))
        lat["f"] = lat["f"] + H.FP_SD * u / np.linalg.norm(u, axis=1, keepdims=True)
    elif kind == "style":
        lat["s"] = lat["s"] + H.STYLE_SD
    elif kind == "wording":
        lat["noise"] = rng.normal(size=lat["noise"].shape)
    elif kind == "material":
        lat["topic"] = rng.choice(H.HELD_TOPICS, size=lat["topic"].shape)
    x, _ = H.render_authors(truth.maps, lat)
    return x.reshape(-1, H.D), lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grain_construction_v1/C1/hidden_map_r1")
    ap.add_argument("--seeds", default=None, help="comma list; default = r1 seeds")
    ap.add_argument("--extra-arms", default="", help="comma list of supplement arms")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--keyed-declared-sign", action="store_true")
    ap.add_argument("--pilot-writers", type=int, default=None, help="supplement V1.2 pilot design")
    args = ap.parse_args()
    if args.pilot_writers:
        H.PILOT_WRITERS = args.pilot_writers
        H.PILOT_TRAIN_WRITERS = args.pilot_writers * 3 // 4
    global SEEDS, ARMS, PROTOCOL, KEYED_DECLARED_SIGN
    if args.seeds:
        SEEDS = tuple(int(x) for x in args.seeds.split(","))
    if args.extra_arms:
        ARMS = ARMS + tuple(args.extra_arms.split(","))
    if args.protocol:
        PROTOCOL = ROOT / args.protocol
    KEYED_DECLARED_SIGN = args.keyed_declared_sign
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    t0, c0 = time.time(), time.process_time()
    proto_sha = sha(PROTOCOL)
    code_sha = {p: sha(ROOT / p) for p in ("grain_research/construction/hidden_evidence_map.py",
                                           "scripts/run_grain_hidden_evidence_map.py")}
    (out / "hidden_map_protocol.md").write_text(PROTOCOL.read_text() + f"\n\n<!-- copied; sha256 {proto_sha} -->\n")

    # ------------------------------------------------------------ estimator stage
    store, digests, tol_rows, cold_rows = {}, {}, [], []
    for seed in SEEDS:
        for world in H.WORLDS:
            view, truth = H.build_world(seed, world)
            digests[f"{world}/{seed}"] = H.view_digest(view)
            est = estimator_stage(view, seed, world)
            store[(seed, world)] = (view, truth, est)
            for arm, e in est.items():
                tol_rows.append({
                    "world": world, "seed": seed, "arm": arm, "status": e["status"], "flags": e["flags"],
                    "anchor_t": e.get("anchor_t"), "pilot_design_r2": e.get("design_r2"),
                    "validation_gain": e.get("validation_gain"), "validation_gain_se": e.get("validation_gain_se"),
                    "tau_T": e.get("tau_T"), "tau_F": TAU_F, "nonconstant_min": e.get("nonconstant_min"),
                    "lambda": (e["readout"].info or {}).get("lambda"), "k": (e["readout"].info or {}).get("k"),
                })
                if "scores" in e:
                    for name in H.WORLDS[world]["targets"]:
                        s = e["scores"][name]
                        bv = e.get("between_var", {}).get(name, np.zeros(H.N_APP))
                        for u in range(H.N_APP):
                            cold_rows.append([world, seed, arm, name, u, s["score"][u], s["var"] + bv[u], s["ols"][u]])
    (out / "frozen_tolerances.json").write_text(json.dumps(tol_rows, indent=1, default=float))
    with open(out / "cold_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["world", "seed", "arm", "target", "app_author_index", "score", "interval_var", "ols"])
        w.writerows(cold_rows)
    frozen_sha = sha(out / "frozen_tolerances.json")
    stage_a = {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}

    # ------------------------------------------------------------ evaluator stage
    rec_rows, int_rows, pair_rows = [], [], []
    for (seed, world), (view, truth, est) in store.items():
        cfg = H.WORLDS[world]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1630]))
        idx = [rng.integers(0, H.N_APP, H.N_APP) for _ in range(B_AUTH)]
        arms = dict(est)
        orc = H.oracle_readouts(view, truth)
        ofit = H.fit_moments(orc.phi(view.source_x), view.source_z, view.source_author)
        arms["oracle_map"] = {"readout": orc, "status": "PRIVILEGED", "flags": [], "fit": ofit,
                              "scores": score_all(orc, ofit, view),
                              "held": score_all(orc, ofit, view, mask=view.app_z[view.app_author == 0] != 1 / 3)}
        # oracle expression: true event expression through the same kernel
        e_src, e_app = truth.source_latent["e"].reshape(-1), truth.app_latent["e"].reshape(-1)
        ident = H.Readout("oracle_expression", None, status="PRIVILEGED")
        efit = H.fit_moments(e_src, view.source_z, view.source_author)
        ez = view.app_z[view.app_author == 0]
        eb, ep, eo = H.blup(efit, H.design(ez), e_app.reshape(H.N_APP, -1))
        arms["oracle_expression"] = {"readout": ident, "status": "PRIVILEGED", "flags": [], "fit": efit,
                                     "scores": {n: {"score": eb @ t, "var": float(t @ ep @ t), "ols": eo @ t}
                                                for n, t in H.TARGET_VECTORS.items()}}
        kinds_all = ["level", "fingerprint", "style", "wording", "material"] + (["slope"] if cfg["slope"] else [])
        iv_all = {k: intervene(truth, k, seed) for k in kinds_all}
        material_x = iv_all["material"][0]
        targets = list(cfg["targets"]) + (["held_m_1/3"] if cfg["slope"] else [])
        base_scores = {}
        for arm, e in arms.items():
            r = e["readout"]
            mech = H.readout_mechanism(r, truth.maps) if r.direction is not None else {}
            if "scores" not in e:
                rec_rows.append({"world": world, "seed": seed, "arm": arm, "status": e["status"], "target": "ALL",
                                 "material": "-", **{k: None for k in ("r", "r_lo", "r_hi")}})
                continue
            unit_ok = e["status"] in ("OK", "PRIVILEGED", "ANCHOR_UNSUPPORTED_DIRECTION")
            mat_scores = score_all(r, e["fit"], view, x=material_x) if r.direction is not None else None
            for name in targets:
                for material in ("same_family", "held_topic_family"):
                    if material == "held_topic_family" and mat_scores is None:
                        continue
                    src = e["held"] if name == "held_m_1/3" and "held" in e else e["scores"]
                    if name == "held_m_1/3" and "held" not in e:
                        continue
                    s = (mat_scores if material == "held_topic_family" else src)[name]
                    if material == "held_topic_family" and name == "held_m_1/3":
                        continue
                    score = s["score"]
                    var = s["var"] + e.get("between_var", {}).get(name, np.zeros(H.N_APP))
                    tru = latent_truth(truth.app_latent, name)
                    sd = float(np.std(score))
                    r_ = float(np.corrcoef(score, tru)[0, 1]) if sd > 0 else float("nan")
                    rb = rboot(score, tru, rng, idx) if sd > 0 else np.array([np.nan])
                    row = {"world": world, "seed": seed, "arm": arm, "status": e["status"],
                           "flags": ";".join(e["flags"]), "target": name, "material": material,
                           "n": H.N_APP, "sd_score": sd, "r": r_,
                           "r_lo": float(np.nanpercentile(rb, 2.5)), "r_hi": float(np.nanpercentile(rb, 97.5))}
                    if unit_ok:
                        err = score - tru
                        half = 1.959964 * np.sqrt(var)
                        row.update(rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(err.mean()),
                                   cal_slope=float(np.polyfit(score, tru, 1)[0]) if sd > 0 else None,
                                   coverage=float(np.mean(np.abs(err) <= half)), mean_width=float(np.mean(2 * half)))
                    else:
                        row.update(abs_r_orientation_unidentified=abs(r_), rmse=None, bias=None,
                                   cal_slope=None, coverage=None, mean_width=None)
                    row.update({f"mech_{k}": v for k, v in mech.items()})
                    rec_rows.append(row)
                    if material == "same_family":
                        base_scores[(arm, name)] = (score, tru)
            # interventions
            if r.direction is None and arm != "oracle_expression":
                continue
            if arm == "oracle_expression":
                continue
            ivx = {k: v[0] for k, v in iv_all.items()}
            base = score_all(r, e["fit"], view)
            for name in cfg["targets"]:
                tkind, delta = ("slope", 1.0) if name == "delta" else ("level", 1.0)
                ch_t = score_all(r, e["fit"], view, x=ivx[tkind])[name]["ols"] - base[name]["ols"]
                ch_t_blup = score_all(r, e["fit"], view, x=ivx[tkind])[name]["score"] - base[name]["score"]
                gain = float(ch_t.mean() / delta)
                gb = np.array([ch_t[i].mean() / delta for i in idx])
                exp_gain = mech.get("gain_expression")
                tau_T = e.get("tau_T")
                int_rows.append({"world": world, "seed": seed, "arm": arm, "status": e["status"],
                                 "flags": ";".join(e["flags"]), "target": name, "intervention": f"target_{tkind}",
                                 "intended_change": delta, "statistic": "gain_T", "value": gain,
                                 "ci_lo": float(np.percentile(gb, 2.5)), "ci_hi": float(np.percentile(gb, 97.5)),
                                 "blup_mean_change": float(ch_t_blup.mean()), "analytic": exp_gain,
                                 "tolerance": tau_T, "within_tolerance": (abs(gain - 1) <= tau_T) if tau_T else None})
                for kind in ("fingerprint", "style", "wording", "material"):
                    ch = score_all(r, e["fit"], view, x=ivx[kind])[name]["ols"] - base[name]["ols"]
                    chb = score_all(r, e["fit"], view, x=ivx[kind])[name]["score"] - base[name]["score"]
                    ratio = float(np.sqrt(np.mean(ch ** 2)) / abs(ch_t.mean())) if abs(ch_t.mean()) > 0 else float("inf")
                    rbs = np.array([np.sqrt(np.mean(ch[i] ** 2)) / abs(ch_t[i].mean()) for i in idx])
                    if kind == "fingerprint":
                        uf = truth.maps["U_f"]
                        df = (iv_all["fingerprint"][1]["f"] - truth.app_latent["f"]) @ uf.T @ r.direction / r.unit
                        analytic = float(np.sqrt(np.mean(df ** 2))) if name == "level" else 0.0
                    elif kind == "style":
                        analytic = 0.0 if name == "level" else float(abs(2 * mech["style_response_per_sd"]))
                    else:
                        analytic = None
                    int_rows.append({"world": world, "seed": seed, "arm": arm, "status": e["status"],
                                     "flags": ";".join(e["flags"]), "target": name, "intervention": kind,
                                     "intended_change": 0.0, "statistic": "R_F" if kind in ("fingerprint", "style") else "rms_ratio_to_target",
                                     "value": ratio, "ci_lo": float(np.percentile(rbs, 2.5)), "ci_hi": float(np.percentile(rbs, 97.5)),
                                     "mean_change_ols": float(ch.mean()), "rms_change_ols": float(np.sqrt(np.mean(ch ** 2))),
                                     "blup_mean_change": float(chb.mean()), "analytic_rms_change_ols": analytic,
                                     "tolerance": TAU_F if kind in ("fingerprint", "style") else None,
                                     "within_tolerance": (ratio <= TAU_F) if kind in ("fingerprint", "style") else None})
        # paired differences vs pilot_ridge
        for (arm, name), (score, tru) in base_scores.items():
            if arm == "pilot_ridge" or ("pilot_ridge", name) not in base_scores:
                continue
            s0, _ = base_scores[("pilot_ridge", name)]
            dr = [np.corrcoef(score[i], tru[i])[0, 1] - np.corrcoef(s0[i], tru[i])[0, 1] for i in idx]
            dm = [np.sqrt(np.mean((score[i] - tru[i]) ** 2)) - np.sqrt(np.mean((s0[i] - tru[i]) ** 2)) for i in idx]
            pair_rows.append({"world": world, "seed": seed, "arm": arm, "target": name, "reference": "pilot_ridge",
                              "delta_r": float(np.corrcoef(score, tru)[0, 1] - np.corrcoef(s0, tru)[0, 1]),
                              "delta_r_lo": float(np.nanpercentile(dr, 2.5)), "delta_r_hi": float(np.nanpercentile(dr, 97.5)),
                              "delta_rmse": float(np.sqrt(np.mean((score - tru) ** 2)) - np.sqrt(np.mean((s0 - tru) ** 2))),
                              "delta_rmse_lo": float(np.percentile(dm, 2.5)), "delta_rmse_hi": float(np.percentile(dm, 97.5))})

    def write(name, rows):
        keys = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write("recovery_by_mechanism.csv", rec_rows)
    write("intervention_checks.csv", int_rows)
    write("paired_differences.csv", pair_rows)
    contract = {
        "task": "GRAIN-C1-01", "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": proto_sha,
        "code_sha256": code_sha, "frozen_tolerances_sha256_before_evaluation": frozen_sha,
        "seeds": SEEDS, "arms": ARMS, "keyed_declared_sign": KEYED_DECLARED_SIGN, "worlds": {k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()} for k, v in H.WORLDS.items()},
        "estimator_inputs": list(H.ESTIMATOR_FIELDS),
        "evaluator_only": ["maps(u_e,U_f,u_s,U_t,u_c)", "a", "b", "f", "s(style)", "topic", "eps", "noise", "true expression e", "pilot latents"],
        "privileged_arms": ["oracle_map", "oracle_expression"],
        "pilot_writers": H.PILOT_WRITERS, "pilot_train_writers": H.PILOT_TRAIN_WRITERS,
        "splits": {"source_authors": H.N_SOURCE, "application_authors": H.N_APP, "events": H.EVENTS,
                   "pilot": f"{H.PILOT_WRITERS} writers x {H.PILOT_PER_WRITER}; train writers 0-{H.PILOT_TRAIN_WRITERS - 1}, validation the rest",
                   "material": "source/pilot topics 0-7; held topic family 8-11 for the material check"},
        "estimator_view_sha256": digests,
        "stage_order": "estimator stage wrote frozen_tolerances.json and cold_scores.csv before any truth was read",
    }
    (out / "source_application_contract.json").write_text(json.dumps(contract, indent=1))
    run = {"stage_a": stage_a, "total_wall_s": time.time() - t0, "total_cpu_s": time.process_time() - c0,
           "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
           "threads": {v: os.environ[v] for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    (out / "run_log.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
