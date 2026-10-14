#!/usr/bin/env python
"""R34 text half-channel sensitivity pilot (docs/GRAIN_R34_TEXT_HALF_CHANNEL_PROTOCOL_V1.md).
One-component diagnostic views (source b=0, b=0 integration seed) with half-calibration channel means; unchanged
production primitives. CPU only, 0 neural calls, no new reader outputs."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import copy
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

PROTOCOL = ROOT / "docs/GRAIN_R34_TEXT_HALF_CHANNEL_PROTOCOL_V1.md"
PROTOCOL_SHA = "6469a1885d7e16801a7615b337cb09b74c46a8f4cb1dc50b02b2fdc48ba46829"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "F16/text_half_channel_sensitivity_r1"
BUDGET_S, NPER, TOL = 600.0, 20, 1e-12
ARMS = ("nli_joint", "hybrid_joint")
LABEL = "DIAGNOSTIC_ONE_COMPONENT_VIEW (source b=0; not the production 200-component ensemble)"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != h:
            raise RuntimeError(f"binding mismatch {rel}")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def bank():
    nli = pd.read_csv(V3 / "F08/nli_reader_r1/reader_scores.csv").set_index("material_id")
    grain = pd.read_csv(V3 / "F11/grain_comment_reader_r1/reader_scores.csv").set_index("material_id")
    ident = ["text", "role", "family", "pole", "qualification", "subject_status", "variant"]
    assert nli[ident].equals(grain[ident])
    b = nli[ident].copy(); b["pole_readout"] = nli.pole_score; b["nli_qualification"] = nli.qualification_score; b["grain_qualification"] = grain.qualification_score
    return b.reset_index()


def first_partition():
    parts = json.loads((V3 / "F16/source_half_reconstruction_r1/results.json").read_text())["text_channel"]["arms"]["nli_joint"]["partitions"]
    A = sorted([p["half_A_cores"] for p in parts])[0]
    return [tuple(x) for x in A]


def smoothed(counts):
    c = np.asarray(counts, float) + .5
    return c / c.sum(2, keepdims=True)


def view_t1(bundle, arm, channel):
    v = copy.copy(bundle)
    v["source_models"] = [bundle["source_models"][0]]
    key = "nli_joint" if arm == "nli_joint" else "hybrid_joint"
    v["channel_draws"] = {**bundle["channel_draws"], key: [np.asarray(channel).tolist()]}
    v["source_components"] = 1
    return v


def view_dyn(bundle, arm, channel):
    v = copy.copy(bundle)
    v["source_models"] = [bundle["source_models"][0]]; v["source_components"] = 1
    draws = {k: [bundle["channel_draws"][k][0]] for k in ("hybrid_joint", "nli_joint")}
    draws["nli_joint" if arm == "nli_joint" else "hybrid_joint"] = [np.asarray(channel).tolist()]
    v["channel_draws"] = draws
    return v


def score_t1(bundle, raws, arm, channel):
    v = view_t1(bundle, arm, channel)
    counts = np.stack([tt.counts_from_input(r, bundle, arm)[0] for r in raws])
    res = tt.score_counts(counts, v, arm)
    return [(x["score"], x["personal_conditional_variance"], x["source_and_reader_variance"]) for x in res]


def score_dyn(bundle, raws, arm, channel):
    v = view_dyn(bundle, arm, channel)
    res = dt.score_panel(raws, v, arm)
    return [{t: (x["targets"][t]["score"], x["targets"][t]["variance"], x["targets"][t]["components"]["source_reader"]) for t in ("T2", "T3")} for x in res]


def production_component0_t1(bundle, raw, arm):
    counts, _ = tt.counts_from_input(raw, bundle, arm)
    cap = []; orig = tt.posterior_from_nodes

    def wrap(*a, **k):
        r = orig(*a, **k); cap.append((float(r[0][0]), float(r[1][0]))); return r
    tt.posterior_from_nodes = wrap
    try:
        tt.score_counts(counts[None], bundle, arm)
    finally:
        tt.posterior_from_nodes = orig
    return cap[0]


def production_component0_dyn(bundle, raw, arm):
    cf, cm = [], []; of, om = dt.filter_nodes, dt.future_moments

    def wf(*a, **k):
        r = of(*a, **k); cf.append(r[2].copy()); return r

    def wm(*a, **k):
        r = om(*a, **k); cm.append((r[0].copy(), r[1].copy())); return r
    dt.filter_nodes, dt.future_moments = wf, wm
    try:
        dt.score_panel([raw], bundle, arm)
    finally:
        dt.filter_nodes, dt.future_moments = of, om
    ll = cf[0][:, 0]; w = np.exp(ll - ll.max()); w /= w.sum()
    means, cov = cm[0][0][:, 0, :], cm[0][1][:, 0]
    c = w @ means; within = np.einsum("n,nkl->kl", w, cov); d = means - c; pers = np.einsum("n,nk,nl->kl", w, d, d)
    out = {}
    for t, e in (("T2", dt.TARGETS[0]), ("T3", dt.TARGETS[1])):
        out[t] = (float(e @ c), float(e @ (within + pers) @ e))
    return out


def support(bundle, raw, arm, roles):
    counts, sup = tt.counts_from_input(raw, bundle, arm)
    rows, _ = tt.replay_history(raw, bundle)
    cats = counts.sum(0)
    return {"counts_condition_by_category": counts.astype(int).tolist(), "category_totals": cats.astype(int).tolist(),
            "pole_high_share": float((cats[2] + cats[3]) / cats.sum()) if cats.sum() else None, "qual_high_share": float((cats[1] + cats[3]) / cats.sum()) if cats.sum() else None,
            "both_endpoints": bool(counts[0].sum() and counts[3].sum()), "events": len(rows),
            "event_roles": sorted(set(roles[r["material_id"]] for r in rows)),
            "event_addresses": [[r.get("event_id"), r["material_id"], r["t"], r["z"]] for r in rows]}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("text_half_channel_sensitivity_r1 exists; preserve earlier runs")
    B = bank(); roles = dict(zip(B.material_id, B.role)); cal = B[B.role == "validation"]
    A_cores = set(first_partition())
    inA = cal.apply(lambda r: (int(r.pole), r.subject_status) in A_cores, axis=1)
    b1 = json.loads((V3 / "F16/t1_text_full_r1/20260915/bundle.json").read_text()); tt.verify_bundle(b1)
    b2 = json.loads((V3 / "F16/dynamic_text_full_r1/20260915/bundle.json").read_text()); dt.validate_bundle(b2)
    chan, chan_info = {}, {}
    for arm in ARMS:
        cA = tt.channel_counts(cal[inA].to_dict("records"), arm); cB = tt.channel_counts(cal[~inA].to_dict("records"), arm)
        cF = tt.channel_counts(cal.to_dict("records"), arm)
        assert np.array_equal(cF, np.asarray(b1["channel_counts"][arm])), "bundle calibration counts differ from bank"
        PA, PB, PF = smoothed(cA), smoothed(cB), smoothed(cF)
        chan[arm] = {"A": PA, "B": PB, "FULL": PF}
        tv = .5 * np.abs(PA - PB).sum(2)
        chan_info[arm] = {"counts_A": cA.astype(int).tolist(), "counts_B": cB.astype(int).tolist(), "counts_FULL": cF.astype(int).tolist(),
                          "mean_A": PA.tolist(), "mean_B": PB.tolist(), "mean_FULL": PF.tolist(), "tv_by_stratum_pole_qual": tv.tolist(), "tv_mean": float(tv.mean()),
                          "tv_max": float(tv.max()), "dyn_bundle_draw0_equals_t1_bundle_draw0": bool(np.allclose(np.asarray(b2["channel_draws"][arm][0]), np.asarray(b1["channel_draws"][arm][0]), atol=0))}
    split = {"half_A_cores": sorted([list(x) for x in A_cores]), "half_A_materials": sorted(cal[inA].material_id), "half_B_materials": sorted(cal[~inA].material_id)}
    persons = {}
    for route, b in (("T1", b1), ("dynamic", b2)):
        inp = jl(V3 / f"F16/{'t1_text_full_r1' if route == 'T1' else 'dynamic_text_full_r1'}/20260915/application_inputs.jsonl")
        sel = sorted(inp, key=lambda r: r["author_id"])[:NPER]
        for r in sel:
            mats = [h["observation"]["material_id"] for h in r["history"] if "observation" in h]
            if any(roles[m] != "application" for m in mats):
                raise RuntimeError(f"calibration material in evaluation for {r['author_id']}")
        persons[route] = sel
    OUT.mkdir(parents=True)
    rows, sup_rows, checks = [], [], {}
    # ---------------- pilot (person 1 both routes, all arms) + cold replay check
    tp = time.time()
    for arm in ARMS:
        raw = persons["T1"][0]
        m0, v0 = production_component0_t1(b1, raw, arm)
        (s, v, sr), = score_t1(b1, [raw], arm, np.asarray(b1["channel_draws"]["nli_joint" if arm == "nli_joint" else "hybrid_joint"][0]))
        checks[f"T1|{arm}"] = {"author_id": raw["author_id"], "prod_component0": [m0, v0], "view_draw0": [s, v], "err": max(abs(s - m0), abs(v - v0)), "view_between_term": sr}
        raw = persons["dynamic"][0]
        pc = production_component0_dyn(b2, raw, arm)
        vw = score_dyn(b2, [raw], arm, np.asarray(b2["channel_draws"][arm][0]))[0]
        checks[f"dynamic|{arm}"] = {"author_id": raw["author_id"], "prod_component0": pc, "view_draw0": {t: vw[t][:2] for t in vw},
                                    "err": max(max(abs(vw[t][0] - pc[t][0]), abs(vw[t][1] - pc[t][1])) for t in ("T2", "T3")), "view_between_term": {t: vw[t][2] for t in vw}}
        for h in ("A", "B", "FULL"):
            score_t1(b1, persons["T1"][:1], arm, chan[arm][h]); score_dyn(b2, persons["dynamic"][:1], arm, chan[arm][h])
    pilot_s = time.time() - tp
    projected = pilot_s * NPER
    n_use = NPER if projected <= BUDGET_S - (time.time() - t0) else max(1, int((BUDGET_S - (time.time() - t0)) / pilot_s))
    # ---------------- main panel
    for arm in ARMS:
        res = {h: score_t1(b1, persons["T1"][:n_use], arm, chan[arm][h]) for h in ("A", "B", "FULL")}
        for j, raw in enumerate(persons["T1"][:n_use]):
            (sA, vA, _), (sB, vB, _), (sF, vF, _) = res["A"][j], res["B"][j], res["FULL"][j]
            rows.append({"route": "text_T1", "target": "T1", "arm": arm, "author_id": raw["author_id"], "mean_A": sA, "var_A": vA, "mean_B": sB, "var_B": vB,
                         "mean_FULL": sF, "var_FULL": vF, "d": sA - sB, "z": (sA - sB) / np.sqrt((vA + vB) / 2), "abs_A_FULL": abs(sA - sF), "abs_B_FULL": abs(sB - sF)})
            sup_rows.append({"route": "text_T1", "arm": arm, "author_id": raw["author_id"], **support(b1, raw, arm, roles)})
        resd = {h: score_dyn(b2, persons["dynamic"][:n_use], arm, chan[arm][h]) for h in ("A", "B", "FULL")}
        for j, raw in enumerate(persons["dynamic"][:n_use]):
            for t in ("T2", "T3"):
                (sA, vA, _), (sB, vB, _), (sF, vF, _) = resd["A"][j][t], resd["B"][j][t], resd["FULL"][j][t]
                rows.append({"route": "text_dynamic", "target": t, "arm": arm, "author_id": raw["author_id"], "mean_A": sA, "var_A": vA, "mean_B": sB, "var_B": vB,
                             "mean_FULL": sF, "var_FULL": vF, "d": sA - sB, "z": (sA - sB) / np.sqrt((vA + vB) / 2), "abs_A_FULL": abs(sA - sF), "abs_B_FULL": abs(sB - sF)})
            sup_rows.append({"route": "text_dynamic", "arm": arm, "author_id": raw["author_id"], **support(b2, raw, arm, roles)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "per_person.csv", index=False)
    summ = {}
    for (route, t, arm), g in df.groupby(["route", "target", "arm"]):
        summ[f"{route}|{t}|{arm}"] = {"n": int(len(g)), "mean_abs_d": float(g.d.abs().mean()), "rms_d": float(np.sqrt((g.d ** 2).mean())), "mean_abs_z": float(g.z.abs().mean()),
                                      "max_abs_z": float(g.z.abs().max()), "share_d_exactly_0": float((g.d == 0).mean()), "mean_abs_A_FULL": float(g.abs_A_FULL.mean()),
                                      "mean_abs_B_FULL": float(g.abs_B_FULL.mean()), "mean_var_FULL": float(g.var_FULL.mean()), "mean_abs_d_over_sd_FULL": float((g.d.abs() / np.sqrt(g.var_FULL)).mean())}
    sp = pd.DataFrame(sup_rows)
    supp = {f"{r}|{a}": {"persons": int(len(g)), "both_endpoints_share": float(g.both_endpoints.mean()), "events_mean": float(g.events.mean()),
                         "pole_high_share_mean": float(g.pole_high_share.mean()), "qual_high_share_mean": float(g.qual_high_share.mean()),
                         "category_totals_sum": np.sum(np.array(g.category_totals.tolist()), axis=0).astype(int).tolist(), "event_roles": sorted(set(x for l in g.event_roles for x in l))}
            for (r, a), g in sp.groupby(["route", "arm"])}
    (OUT / "support.json").write_text(json.dumps(sup_rows, indent=0, default=float) + "\n")
    (OUT / "channels.json").write_text(json.dumps({"split": split, "arms": chan_info}, indent=1) + "\n")
    (OUT / "config.json").write_text(json.dumps({"label": LABEL, "seed": 20260915, "source_component": 0, "qmc_power": {"T1": b1["qmc_power"], "dynamic": b2["qmc_power"]},
                                                 "integration_seed": {"T1": "SeedSequence([20260915,161,0,0])", "dynamic": "SeedSequence([20260915,164,0,0])"},
                                                 "source_reference": {"T1": {"bundle": "F16/t1_text_full_r1/20260915/bundle.json", "model0_mu": b1["source_models"][0]["mu"], "model0_sigma2": b1["source_models"][0]["sigma2"]},
                                                                      "dynamic": {"bundle": "F16/dynamic_text_full_r1/20260915/bundle.json", "model0_prior_mu": b2["source_models"][0]["prior"]["mu"], "model0_log_sigma": b2["source_models"][0]["log_sigma"]}},
                                                 "channel_rule": "smoothed mean (counts+.5)/sum, per (pole,qualification) row", "actual_component_count": 1,
                                                 "persons": {k: [r["author_id"] for r in v[:n_use]] for k, v in persons.items()}}, indent=1) + "\n")
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "label": LABEL, "pilot_seconds": pilot_s, "projected_seconds_20": projected, "persons_used": n_use,
           "stopped_early": n_use < NPER, "cold_replay_checks": checks, "cold_replay_max_err": float(max(c["err"] for c in checks.values())),
           "summaries": summ, "support_summary": supp, "channel_tv": {a: {"tv_mean": chan_info[a]["tv_mean"], "tv_max": chan_info[a]["tv_max"]} for a in ARMS},
           "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print("done", round(res["seconds"], 2), "pilot", round(pilot_s, 3), "persons", n_use, "replay_err", res["cold_replay_max_err"])


if __name__ == "__main__":
    main()
