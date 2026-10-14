#!/usr/bin/env python
"""R34 cached error-budget reconciliation (docs/GRAIN_R34_ERROR_BUDGET_PROTOCOL_V1.md).
Reconstructs each archived readout's variance identity from its saved component fields; two one-person cold replays
(one per text scorer) expose per-component parts that were not cached. CPU only, 0 neural calls."""
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
from grain_research.gptpro_foundation_v3 import f16_dynamic_text as dt  # noqa: E402
from grain_research.gptpro_foundation_v3 import f16_t1_text as tt  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R34_ERROR_BUDGET_PROTOCOL_V1.md"
PROTOCOL_SHA = "5e55f6145e36d9301157e4e042ee35c49d43881ba770995ec807463eb188e827"
INPUTS = ROOT / "docs/GRAIN_R34_ERROR_BUDGET_INPUTS_V1.json"
F16 = ROOT / "results/grain_gptpro_foundation_v3/F16"
OUT = F16 / "error_budget_reconciliation_r1"
SEEDS = ("20260915", "20260916", "20260917")
TOL = 1e-12
CAT_ARMS = ("hybrid_joint", "nli_joint", "hybrid_factorized", "oracle")
GAUSS_ARMS = ("naive_gaussian", "eligible_continuous")
E_T2, E_T3 = dt.TARGETS[0], dt.TARGETS[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != h:
            raise RuntimeError(f"binding mismatch {rel}")
    manifest = json.loads(INPUTS.read_text())
    n = 0
    for group in manifest["groups"].values():
        for rel, h in group.items():
            if sha(ROOT / rel) != h:
                raise RuntimeError(f"input changed {rel}")
            n += 1
    return n


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def pvar(x):
    x = np.asarray(x, float)
    return float(np.mean((x - x.mean()) ** 2))


ROWS = []


def emit(**kw):
    comps = kw.pop("components")
    total = kw["total_saved"]
    row = dict(kw)
    for k, v in comps.items():
        row[f"c_{k}"] = v
        row[f"share_{k}"] = v / total if total > 0 else np.nan
    ROWS.append(row)


# ---------------------------------------------------------------- historical numeric
def do_h_t1(seed):
    for i, r in enumerate(jl(F16 / f"t1_source_full_r1/{seed}/cold_scores.jsonl")):
        sc = r["source_components"]
        m = np.array([c["mean"] for c in sc]); cv = np.array([c["conditional_variance"] for c in sc])
        pers, src = float(cv.mean()), pvar(m)
        emit(set="historical", route="numeric_t1", seed=seed, variant="native_L", target="T1", person=r.get("author_id", f"idx{i:03d}"),
             score=r["score"], total_saved=r["variance"], genuine_components=len(sc), interval="Gaussian_moment_interval",
             components={"personal_conditional": r["components"]["personal_conditional"], "source": r["components"]["source"]},
             err_sum=abs(r["components"]["personal_conditional"] + r["components"]["source"] - r["variance"]),
             err_recon_components=max(abs(pers - r["components"]["personal_conditional"]), abs(src - r["components"]["source"])),
             err_recon_score=abs(float(m.mean()) - r["score"]), err_recon_total=abs(pers + src - r["variance"]),
             err_interval=abs(r["lower"] - (r["score"] - 1.96 * np.sqrt(r["variance"]))),
             note="mc_components are a finite-MC comparator, not a psychological component")


def do_h_t2(seed):
    for i, r in enumerate(jl(F16 / f"t2_cold_full_r1/{seed}/cold_scores.jsonl")):
        sc = r["source_components"]
        m = np.array([c["mean"] for c in sc]); es = np.array([c["event_state"] for c in sc]); pp = np.array([c["personal_parameter"] for c in sc])
        c = r["components"]
        rec = {"event_state": float(es.mean()), "personal_parameter": float(pp.mean()), "source": pvar(m)}
        emit(set="historical", route="numeric_t2", seed=seed, variant="native_core", target="T2", person=r.get("author_id", f"idx{i:03d}"),
             score=r["score"], total_saved=r["variance"], genuine_components=len(sc), interval=r.get("interval_kind"),
             components=dict(c), err_sum=abs(sum(c.values()) - r["variance"]),
             err_recon_components=max(abs(rec[k] - c[k]) for k in rec), err_recon_score=abs(float(m.mean()) - r["score"]),
             err_recon_total=abs(sum(rec.values()) - r["variance"]), err_interval=abs(r["lower"] - (r["score"] - 1.96 * np.sqrt(r["variance"]))), note="")


def do_h_t3(seed):
    bundle = json.loads((F16 / f"t3_application_full_r1/{seed}/bundle.json").read_text())
    rs = bundle["reference_sigma"]
    for i, r in enumerate(jl(F16 / f"t3_application_full_r1/{seed}/cold_scores.jsonl")):
        ps = r["per_source"]
        if not r["supported"]:
            ROWS.append({"set": "historical", "route": "numeric_t3", "seed": seed, "variant": "native_core", "target": "T3", "person": f"idx{i:03d}",
                         "note": "UNSUPPORTED readout", "total_saved": np.nan})
            continue
        m = np.array([p["mean"] for p in ps])
        es = np.array([p["event_state"] for p in ps]); pp = np.array([p["personal_parameter"] for p in ps])
        ts = np.array([p["terminal_state"] for p in ps]); fi = np.array([p["future_innovation"] for p in ps]); fo = np.array([p["future_observation"] for p in ps])
        vc = r["variance_components"]
        rec = {"event_state": float(es.mean()), "personal_parameter": float(pp.mean()), "source": pvar(m), "terminal_state": float(ts.mean()),
               "future_innovation": float(fi.mean()), "future_observation": float(fo.mean())}
        per_b_split = float(np.max(np.abs(es - (ts + fi + fo))))
        comps = {"state_carried": vc["terminal_state"], "future_innovation": vc["future_innovation"], "future_observation": vc["future_observation"],
                 "personal_parameter": vc["personal_parameter"], "source": vc["source"]}
        emit(set="historical", route="numeric_t3", seed=seed, variant="native_core", target="T3", person=f"idx{i:03d}",
             score=r["raw_score"], total_saved=vc["total"], genuine_components=len(ps), interval="gaussian_moment;WORKING_BOOTSTRAP_LAPLACE",
             components=comps, err_sum=abs(vc["event_state"] + vc["personal_parameter"] + vc["source"] - vc["total"]),
             err_recon_components=max(max(abs(rec[k] - vc[k]) for k in rec), per_b_split, abs(vc["terminal_state"] + vc["future_innovation"] + vc["future_observation"] - vc["event_state"])),
             err_recon_score=abs(float(m.mean()) - r["raw_score"]), err_recon_total=abs(rec["event_state"] + rec["personal_parameter"] + rec["source"] - vc["total"]),
             err_interval=max(abs(r["source_standardized_variance"] - vc["total"] / rs ** 2), abs(r["interval"][0] - (r["raw_score"] - 1.96 * np.sqrt(vc["total"])))),
             note="event_state = terminal_state (current-state uncertainty carried forward) + future innovation + future observation")


def do_h_t1p(seed):
    for i, r in enumerate(jl(F16 / f"t1_cold_full_r1/{seed}/scores.jsonl")):
        emit(set="historical", route="numeric_t1_plugin_nosource", seed=seed, variant="plugin_single_model", target="T1", person=f"idx{i:03d}",
             score=r["score"], total_saved=r["variance"], genuine_components=1, interval=r.get("interval_kind"), components={"personal_conditional_only": r["variance"]},
             err_sum=np.nan, err_recon_components=np.nan, err_recon_score=np.nan, err_recon_total=np.nan,
             err_interval=abs(r["lower"] - (r["score"] - 1.96 * np.sqrt(r["variance"]))), note="UNSUPPORTED_DECOMPOSITION: source_uncertainty_included=false; total only")


# ---------------------------------------------------------------- historical text
def do_h_x1(seed):
    arms = json.loads((F16 / f"t1_text_full_r1/{seed}/arm_scores.json").read_text())
    for arm in CAT_ARMS:
        for i, r in enumerate(arms[arm]):
            c = {"personal_conditional": r["personal_conditional_variance"], "source_and_reader_combined": r["source_and_reader_variance"]}
            emit(set="historical", route="text_t1_old_reader", seed=seed, variant=arm, target="T1", person=f"idx{i:03d}", score=r["score"], total_saved=r["variance"],
                 genuine_components=r["source_components"], interval="moment_normal;QMC error separate", components=c, err_sum=abs(sum(c.values()) - r["variance"]),
                 err_recon_components=np.nan, err_recon_score=np.nan, err_recon_total=np.nan, err_interval=abs(r["lower"] - (r["score"] - 1.96 * np.sqrt(r["variance"]))),
                 note="per-b parts not cached (see replay)")
    for arm in GAUSS_ARMS:
        for i, r in enumerate(arms[arm]):
            emit(set="historical", route="text_t1_old_reader", seed=seed, variant=arm, target="T1", person=f"idx{i:03d}", score=r["score"], total_saved=r["variance"],
                 genuine_components=200, interval="moment_normal", components={}, err_sum=np.nan, err_recon_components=np.nan, err_recon_score=np.nan,
                 err_recon_total=np.nan, err_interval=abs(r["lower"] - (r["score"] - 1.96 * np.sqrt(r["variance"]))),
                 note="UNSUPPORTED_DECOMPOSITION: numeric comparator mixture computed but parts not saved")


def dyn_rows(records, set_, route, seed, arm, src_note):
    for i, r in enumerate(records):
        J = {k: np.asarray(v) for k, v in r["joint_components"].items()}
        tot = sum(J.values()); jc = np.asarray(r["joint_covariance"]); jm = np.asarray(r["joint_mean"])
        err_joint = float(np.max(np.abs(tot - jc)))
        for tname, e in (("T2", E_T2), ("T3", E_T3)):
            tg = r["targets"][tname]
            proj = {k: float(e @ v @ e) for k, v in J.items()}
            c = dict(tg["components"])
            rename = {"source_reader": "source_reader_combined"}
            emit(set=set_, route=route, seed=seed, variant=arm, target=tname, person=f"idx{i:03d}", score=tg["score"], total_saved=tg["variance"],
                 genuine_components=200, interval="moment_normal;" + r.get("approximation", ""), components={rename.get(k, k): v for k, v in c.items()},
                 err_sum=abs(sum(c.values()) - tg["variance"]), err_recon_components=max(abs(proj[k] - c[k]) for k in c),
                 err_recon_score=abs(float(e @ jm) - tg["score"]), err_recon_total=max(abs(float(e @ jc @ e) - tg["variance"]), err_joint),
                 err_interval=abs(tg["lower"] - (tg["score"] - 1.96 * np.sqrt(tg["variance"]))),
                 note=src_note + ("; T2 = current state (filter variance only)" if tname == "T2" else "; T3 = future mean (includes future innovation + observation)"))


def do_h_x2(seed):
    for arm in CAT_ARMS:
        dyn_rows(json.loads((F16 / f"dynamic_text_full_r1/{seed}/{arm}_scores.json").read_text()), "historical", "text_dynamic_old_reader", seed, arm, "dynamic_text_full_r1")
    for arm in GAUSS_ARMS:
        dyn_rows(json.loads((F16 / f"dynamic_text_gaussian_r1/{seed}/{arm}_scores.json").read_text()), "historical", "text_dynamic_old_reader", seed, arm,
                 "dynamic_text_gaussian_r1 (supersedes integration in full_r1)")


# ---------------------------------------------------------------- serialized public
def do_serialized():
    index = json.loads((F16 / "serialize_full_r1/bundle_index.json").read_text())
    joint_T2T3 = {}
    for b in index:
        recs = jl(F16 / f"serialize_full_r1/cold/{b['name']}.scores.jsonl")
        variant = b["name"].replace(f"{b['route']}_{b['seed']}_", "").rsplit("_", 1)[0]
        for i, r in enumerate(recs):
            if r.get("status") != "SCORED":
                ROWS.append({"set": "serialized", "route": b["route"], "seed": str(b["seed"]), "variant": variant, "target": b["target"], "person": r.get("author_id"),
                             "note": f"NOT_SCORED {r.get('status')}", "total_saved": np.nan})
                continue
            c = dict(r["components"])
            ren = {"source_and_reader": "source_and_reader_combined", "source_reader": "source_reader_combined"}
            sup = r.get("support", {})
            extra = {}
            err_rc, err_rs, err_rt = np.nan, np.nan, np.nan
            if b["route"] == "text_dynamic":
                eo = r["engine_output"]
                J = {k: np.asarray(v) for k, v in eo["joint_components"].items()}
                e = E_T2 if b["target"] == "T2" else E_T3
                err_rc = max(abs(float(e @ J[k] @ e) - c[k]) for k in c)
                err_rt = max(abs(float(e @ np.asarray(eo["joint_covariance"]) @ e) - r["variance"]), float(np.max(np.abs(sum(J.values()) - np.asarray(eo["joint_covariance"])))))
                err_rs = abs(float(e @ np.asarray(eo["joint_mean"])) - r["score"])
            elif b["route"] == "text_t1":
                eo = r["engine_output"]
                err_rc = max(abs(eo["personal_conditional_variance"] - c["personal_conditional"]), abs(eo["source_and_reader_variance"] - c["source_and_reader"]))
                err_rs = abs(eo["score"] - r["score"]); err_rt = abs(eo["variance"] - r["variance"])
            elif b["route"] == "numeric_dynamic" and "joint" in r:
                joint_T2T3[(str(b["seed"]), variant, r["author_id"], b["target"])] = r["joint"]
            emit(set="serialized", route=b["route"], seed=str(b["seed"]), variant=variant, target=b["target"], person=r["author_id"], score=r["score"], total_saved=r["variance"],
                 genuine_components=sup.get("integration_components"), calibrators=sup.get("calibrators"), interval="moment_normal", components={ren.get(k, k): v for k, v in c.items()},
                 err_sum=abs(sum(c.values()) - r["variance"]), err_recon_components=err_rc, err_recon_score=err_rs, err_recon_total=err_rt,
                 err_interval=abs(r["lower"] - (r["score"] - 1.96 * np.sqrt(r["variance"]))), note="", **extra)
    # joint T2/T3 diagonal vs companion readouts
    lookup = {(r["seed"], r["variant"], r["person"], r["target"]): r for r in ROWS if r.get("set") == "serialized" and r.get("route") == "numeric_dynamic"}
    errs, n = [], 0
    for (s, v, a, t), j in joint_T2T3.items():
        cov = np.asarray(j["covariance_T2_T3"])
        for k, tn in enumerate(("T2", "T3")):
            other = lookup.get((s, v, a, tn))
            if other is not None:
                errs.append(abs(cov[k, k] - other["total_saved"])); n += 1
    return {"joint_diag_checks": n, "joint_diag_max_err": float(max(errs)) if errs else None,
            "joint_keys": sorted(set(k for j in joint_T2T3.values() for k in j))}


# ---------------------------------------------------------------- aliases
def max_abs(pairs):
    return float(max(abs(a - b) for a, b in pairs)) if pairs else None


def alias_checks():
    out = {}
    for seed in SEEDS:
        arms = json.loads((F16 / f"t1_text_full_r1/{seed}/arm_scores.json").read_text())["hybrid_joint"]
        app = jl(F16 / f"t1_text_full_r1/{seed}/application_cold_scores.jsonl")
        out[f"t1_text_app_cold_vs_arm_hybrid|{seed}"] = {"n": len(app), "max_abs": max_abs([(a[k], b[k]) for a, b in zip(app, arms, strict=True)
                                                                                                for k in ("score", "variance", "personal_conditional_variance", "source_and_reader_variance")])}
        hj = json.loads((F16 / f"dynamic_text_full_r1/{seed}/hybrid_joint_scores.json").read_text())
        dap = jl(F16 / f"dynamic_text_full_r1/{seed}/application_cold_scores.jsonl")
        out[f"dyn_app_cold_vs_hybrid|{seed}"] = {"n": len(dap), "max_abs": max_abs([(a["targets"][t][k], b["targets"][t][k]) for a, b in zip(dap, hj, strict=True)
                                                                                      for t in ("T2", "T3") for k in ("score", "variance")])}
        for arm in GAUSS_ARMS:
            old = json.loads((F16 / f"dynamic_text_full_r1/{seed}/{arm}_scores.json").read_text())
            new = json.loads((F16 / f"dynamic_text_gaussian_r1/{seed}/{arm}_scores.json").read_text())
            out[f"dyn_gauss_superseded_full_vs_refined|{arm}|{seed}"] = {"n": len(old), "max_abs_difference_REPORTED_NOT_EXPECTED_EQUAL":
                                                                         max_abs([(a["targets"][t][k], b["targets"][t][k]) for a, b in zip(old, new, strict=True) for t in ("T2", "T3") for k in ("score", "variance")])}
        for est in ("mixture", "plugin"):
            tr = json.loads((F16 / f"text_transport_full_r1/scores/{seed}/T1_changed_pilot_{est}.json").read_text())
            ser = jl(F16 / f"serialize_full_r1/cold/text_t1_{seed}_{est}_T1.scores.jsonl")
            aligned = all(a.get("active_materials") == b["engine_output"].get("active_materials") for a, b in zip(tr, ser, strict=True))
            out[f"text_transport_T1_{est}_vs_serialized|{seed}"] = {"n": len(tr), "aligned_by_active_materials": aligned,
                                                                    "max_abs": max_abs([(a["score"], b["score"]) for a, b in zip(tr, ser)] + [(a["variance"], b["variance"]) for a, b in zip(tr, ser)]
                                                                                       + [(a["personal_conditional_variance"], b["components"]["personal_conditional"]) for a, b in zip(tr, ser)]
                                                                                       + [(a["source_and_reader_variance"], b["components"]["source_and_reader"]) for a, b in zip(tr, ser)])}
            trd = json.loads((F16 / f"text_transport_full_r1/scores/{seed}/dynamic_changed_pilot_{est}.json").read_text())
            for t in ("T2", "T3"):
                ser = jl(F16 / f"serialize_full_r1/cold/text_dynamic_{seed}_{est}_{t}.scores.jsonl")
                out[f"text_transport_dyn_{est}_{t}_vs_serialized|{seed}"] = {"n": len(trd), "max_abs": max_abs(
                    [(a["targets"][t][k], b[k]) for a, b in zip(trd, ser, strict=True) for k in ("score", "variance")]
                    + [(a["targets"][t]["components"][k], b["components"][k]) for a, b in zip(trd, ser) for k in a["targets"][t]["components"]])}
        pay = json.loads((F16 / f"t1_transport_full_r1/{seed}/application_scores_before_truth.json").read_text())
        errs, nkeys = [], 0
        for item in pay:
            w, mode, est = item["key"]
            f = F16 / f"serialize_full_r1/cold/numeric_t1_{seed}_{w}_{mode}_{est}_T1.scores.jsonl"
            if not f.exists():
                continue
            nkeys += 1
            ser = jl(f); mo = item["moments"]
            for j, r in enumerate(ser):
                errs += [abs(mo["score"][j] - r["score"]), abs(mo["variance"][j] - r["variance"])] + [abs(mo[k][j] - r["components"][k]) for k in ("conditional", "pilot", "source")]
        out[f"t1_transport_payload_vs_serialized|{seed}"] = {"keys_matched": nkeys, "max_abs": float(max(errs)) if errs else None}
        natives = {1: jl(F16 / f"t1_source_full_r1/{seed}/cold_scores.jsonl"), 2: jl(F16 / f"t2_cold_full_r1/{seed}/cold_scores.jsonl"),
                   3: jl(F16 / f"t3_application_full_r1/{seed}/cold_scores.jsonl")}
        for k in (1, 2, 3):
            pc = jl(F16 / f"public_contract_full_r1/numeric_T{k}_{seed}/cold.jsonl")
            matched, errs = 0, []
            for rec in pc:
                eo = rec.get("engine_output", rec)
                ec = eo.get("components") or eo.get("variance_components") or {}
                best = None
                for nat in natives[k]:
                    nc = nat.get("components") or nat.get("variance_components") or {}
                    keys = [x for x in ec if x in nc and isinstance(ec[x], (int, float))]
                    if keys:
                        d = max(abs(ec[x] - nc[x]) for x in keys)
                        best = d if best is None or d < best else best
                if best is not None and best <= TOL:
                    matched += 1
                if best is not None:
                    errs.append(best)
            out[f"public_contract_T{k}_vs_native|{seed}"] = {"records": len(pc), "matched_within_tol": matched, "best_match_max_abs": float(max(errs)) if errs else None}
    return out


# ---------------------------------------------------------------- cold replays (one person per text scorer)
def replay_t1():
    bundle = json.loads((F16 / "t1_text_full_r1/20260915/bundle.json").read_text())
    inputs = jl(F16 / "t1_text_full_r1/20260915/application_inputs.jsonl")
    pos = int(np.argmin([r["author_id"] for r in inputs]))
    raw = inputs[pos]
    tt.verify_bundle(bundle)
    counts, support = tt.counts_from_input(raw, bundle)
    captured = []
    orig = tt.posterior_from_nodes

    def wrap(*a, **k):
        res = orig(*a, **k); captured.append((res[0].copy(), res[1].copy())); return res
    t0 = time.time()
    tt.posterior_from_nodes = wrap
    try:
        prod = tt.score_counts(counts[None], bundle, "hybrid_joint")[0]
    finally:
        tt.posterior_from_nodes = orig
    secs = time.time() - t0
    m = np.array([c[0][0] for c in captured]); v = np.array([c[1][0] for c in captured])
    archived = json.loads((F16 / "t1_text_full_r1/20260915/arm_scores.json").read_text())["hybrid_joint"][pos]
    app = jl(F16 / "t1_text_full_r1/20260915/application_cold_scores.jsonl")[pos]
    return {"scorer": "f16_t1_text.score_counts (unchanged; posterior_from_nodes wrapped to record outputs)", "bundle": "t1_text_full_r1/20260915", "arm": "hybrid_joint",
            "author_id": raw["author_id"], "position": pos, "components_captured": len(m), "seconds": secs,
            "alignment_active_materials_equal": app.get("active_materials") == support.get("active_materials") if isinstance(support, dict) else None,
            "E_within": float(v.mean()), "Var_means_ddof0": pvar(m), "score_recon": float(m.mean()),
            "err_vs_production": max(abs(float(v.mean()) - prod["personal_conditional_variance"]), abs(pvar(m) - prod["source_and_reader_variance"]), abs(float(m.mean()) - prod["score"])),
            "err_vs_archived": max(abs(prod["score"] - archived["score"]), abs(prod["variance"] - archived["variance"]),
                                   abs(prod["personal_conditional_variance"] - archived["personal_conditional_variance"]), abs(prod["source_and_reader_variance"] - archived["source_and_reader_variance"])),
            "per_b_mean_sd": float(m.std()), "per_b_within_range": [float(v.min()), float(v.max())]}, m, v


def replay_dyn():
    bundle = json.loads((F16 / "dynamic_text_full_r1/20260915/bundle.json").read_text())
    inputs = jl(F16 / "dynamic_text_full_r1/20260915/application_inputs.jsonl")
    pos = int(np.argmin([r["author_id"] for r in inputs]))
    raw = inputs[pos]
    cap_f, cap_m = [], []
    of, om = dt.filter_nodes, dt.future_moments

    def wf(*a, **k):
        res = of(*a, **k); cap_f.append(res[2].copy()); return res

    def wm(*a, **k):
        res = om(*a, **k); cap_m.append((res[0].copy(), res[1].copy())); return res
    t0 = time.time()
    dt.filter_nodes, dt.future_moments = wf, wm
    try:
        prod = dt.score_panel([raw], bundle, "hybrid_joint")[0]
    finally:
        dt.filter_nodes, dt.future_moments = of, om
    secs = time.time() - t0
    centers, within, personal = [], [], []
    for ll, (means, cov) in zip(cap_f, cap_m, strict=True):
        w = np.exp(ll[:, 0] - (ll[:, 0].max() + np.log(np.sum(np.exp(ll[:, 0] - ll[:, 0].max())))))
        c = w @ means[:, 0, :]; centers.append(c)
        within.append(np.einsum("n,nkl->kl", w, cov[:, 0])); d = means[:, 0, :] - c; personal.append(np.einsum("n,nk,nl->kl", w, d, d))
    C = np.array(centers); W = np.array(within); P = np.array(personal)
    src = np.einsum("bk,bl->kl", C - C.mean(0), C - C.mean(0)) / len(C)
    archived = json.loads((F16 / "dynamic_text_full_r1/20260915/hybrid_joint_scores.json").read_text())[pos]
    out = {"scorer": "f16_dynamic_text.score_panel (unchanged; filter_nodes/future_moments wrapped to record outputs)", "bundle": "dynamic_text_full_r1/20260915",
           "arm": "hybrid_joint", "author_id": raw["author_id"], "position": pos, "components_captured": len(C), "seconds": secs, "targets": {}}
    for tn, e in (("T2", E_T2), ("T3", E_T3)):
        m = C @ e; within_t = np.einsum("k,bkl,l->b", e, W, e); pers_t = np.einsum("k,bkl,l->b", e, P, e)
        tg = prod["targets"][tn]; ar = archived["targets"][tn]
        out["targets"][tn] = {"score_recon": float(m.mean()), "E_cond_state_future": float(within_t.mean()), "E_personal_parameter": float(pers_t.mean()), "Var_centers_ddof0": pvar(m),
                              "identity_total": float(within_t.mean() + pers_t.mean() + pvar(m)),
                              "err_vs_production": max(abs(float(m.mean()) - tg["score"]), abs(float(within_t.mean()) - tg["components"]["conditional_state_future"]),
                                                       abs(float(pers_t.mean()) - tg["components"]["personal_parameter"]), abs(pvar(m) - tg["components"]["source_reader"]),
                                                       abs(float(within_t.mean() + pers_t.mean() + pvar(m)) - tg["variance"])),
                              "err_vs_archived": max(abs(tg["score"] - ar["score"]), abs(tg["variance"] - ar["variance"])),
                              "per_b_center_sd": float(m.std())}
    out["joint_source_matrix_err"] = float(np.max(np.abs(src - np.asarray(prod["joint_components"]["source_reader"]))))
    return out, C, W, P


# ---------------------------------------------------------------- summaries
def summarize(df):
    out = []
    comp_cols = [c for c in df.columns if c.startswith("share_")]
    for key, g in df.groupby(["set", "route", "seed", "variant", "target"], dropna=False):
        row = {"set": key[0], "route": key[1], "seed": key[2], "variant": key[3], "target": key[4], "n": int(len(g)),
               "n_missing_total": int(g["total_saved"].isna().sum()), "mean_total": float(g["total_saved"].mean()),
               "genuine_components": sorted(set(int(x) for x in g["genuine_components"].dropna())) if "genuine_components" in g else None,
               "interval": sorted(set(str(x) for x in g["interval"].dropna()))}
        for c in comp_cols:
            s = g[c].dropna()
            if len(s):
                row[c.replace("share_", "mean_share_")] = float(s.mean()); row[c.replace("share_", "median_share_")] = float(s.median())
                row[c.replace("share_", "max_share_")] = float(s.max())
        for e in ("err_sum", "err_recon_components", "err_recon_score", "err_recon_total", "err_interval"):
            if e in g and g[e].notna().any():
                row[f"max_{e}"] = float(g[e].max())
        out.append(row)
    return out


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    n_inputs = verify()
    if OUT.exists():
        raise FileExistsError("error_budget_reconciliation_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    for seed in SEEDS:
        do_h_t1(seed); do_h_t2(seed); do_h_t3(seed); do_h_t1p(seed); do_h_x1(seed); do_h_x2(seed)
    t_hist = time.time() - t0
    joint = do_serialized()
    t_ser = time.time() - t0
    aliases = alias_checks()
    t_alias = time.time() - t0
    r1, m1, v1 = replay_t1()
    r2, C, W, P = replay_dyn()
    np.savez_compressed(OUT / "replay_components.npz", t1_means=m1, t1_within=v1, dyn_centers=C, dyn_within=W, dyn_personal=P)
    df = pd.DataFrame(ROWS)
    df.to_csv(OUT / "per_person_components.csv", index=False)
    summ = summarize(df)
    (OUT / "summaries.json").write_text(json.dumps(summ, indent=1, default=float) + "\n")
    (OUT / "alias_checks.json").write_text(json.dumps(aliases, indent=1, default=float) + "\n")
    (OUT / "replay_checks.json").write_text(json.dumps({"text_t1": r1, "text_dynamic": r2}, indent=1, default=float) + "\n")
    errcols = ["err_sum", "err_recon_components", "err_recon_score", "err_recon_total", "err_interval"]
    fails = {c: int((df[c] > TOL).sum()) for c in errcols if c in df}
    res = {"protocol_sha256": PROTOCOL_SHA, "inputs_verified": n_inputs, "neural_calls": 0, "readouts": int(len(df)),
           "readouts_by_set": df.groupby("set").size().to_dict(), "max_errors": {c: float(df[c].max()) for c in errcols if c in df},
           "tolerance": TOL, "readouts_over_tolerance": fails, "serialized_joint": joint, "unsupported": df[df["note"].astype(str).str.contains("UNSUPPORTED")].groupby(["route", "variant"]).size().to_dict(),
           "timing_seconds": {"historical": t_hist, "serialized_cum": t_ser, "alias_cum": t_alias, "replay_t1": r1["seconds"], "replay_dyn": r2["seconds"]}}
    res["unsupported"] = {f"{k[0]}|{k[1]}": int(v) for k, v in res["unsupported"].items()}
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    inv = json.loads(INPUTS.read_text())
    (OUT / "inventory.json").write_text(json.dumps({"input_manifest_sha256": sha(INPUTS), "counts": inv["counts"], "canonical_sets": sorted(set(zip(df["set"], df["route"]))),
                                                    "aliases": sorted(aliases), "non_canonical_not_decomposed": ["text transport known_pool/uncalibrated/old_matched/oracle arms",
                                                                                                                  "numeric transport uncalibrated/known_inverse arms", "cards_* and source_scores* panels"]},
                                                   indent=1, default=str) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print("done", round(res["seconds"], 1), "s readouts", len(df), "over_tol", fails)


if __name__ == "__main__":
    main()
