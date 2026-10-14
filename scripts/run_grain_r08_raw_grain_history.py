#!/usr/bin/env python
"""R08 exact-cache raw-vector / VM05 lexical-SVD mean-std GRAIN history comparators (docs/GRAIN_R08_RAW_GRAIN_HISTORY_PROTOCOL_V1.md).
CPU only, 0 neural calls, no text/labels. Bridge gates first; four affine ridge readouts; held scored once."""
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
from grain_research import r08_history_increment as h22  # noqa: E402
from grain_research import r08_raw_grain_history as g  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R08_RAW_GRAIN_HISTORY_PROTOCOL_V1.md"
PROTOCOL_SHA = "9c28e25ffea6067d7324f8ded6a7e7ad6580700cca0ebbb1c3e287c121f0457d"
F14 = ROOT / "results/grain_gptpro_foundation_v3/F14"
OUT = F14 / "raw_grain_history_r1"
TOL_RAW, TOL = 1e-6, 1e-12
ARMS = ("KEYED_STATIC", "RAW_STATIC", "GRAIN_STATIC", "RAW_ORDERED")
ROLE = {"source_validation": 1, "held": 2}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def paired(df, col, ref, auths, draws):
    per = df.groupby("author")[[col, ref]].mean().reindex(auths)
    d = (per[col] - per[ref]).to_numpy()
    bs = d[draws].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "ci_contains_0": bool(lo <= 0 <= hi)}


def strata(df, col, ref):
    n = len(df); out = {}
    for name, mask in (("known_raw_venue", ~df.novel_raw_venue), ("novel_raw_venue", df.novel_raw_venue)):
        sub = df[mask]
        out[name] = {"events": int(len(sub)), "mean_delta": float((sub[col] - sub[ref]).mean()), "contribution": float((sub[col] - sub[ref]).sum() / n)}
    return out


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("raw_grain_history_r1 exists; preserve earlier runs")
    D = h22.load_all()
    m, X, Z, by = D["m"], D["X"], D["Z"], D["by_author"]
    train, val, held = D["train"], D["val"], D["held"]
    roles16 = json.loads((F14 / "pandora_venue_oom_r1/author_roles.json").read_text())
    assert [int(a) for a in train] == roles16["source_train"] and [int(a) for a in val] == roles16["source_validation"] and [int(a) for a in held] == roles16["held"]
    assert not (set(train) & set(val) or set(train) & set(held) or set(val) & set(held))
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "roles": {"source_train": 403, "source_validation": 101, "held": 120},
           "held_disclosure": "same 120 held authors as packets 15/16/22: exploratory reuse, not a fresh lockbox", "arms_executed": [], "arms_not_executed": {}}
    au = m.author_index.to_numpy(); un = m.unit.to_numpy()
    # ================= A1 raw E5 bridge
    comments, ids = g.load_raw_comments()
    cref_ids = json.loads((g.CREF / "pandora/private_authors.json").read_text())["ids"]
    mm = json.loads((g.MAT / "material_manifest.json").read_text())
    direction = g.load_directions()
    keyed_re = g.keyed_from_raw(comments[au, un], direction)
    err = np.abs(keyed_re - X)
    raw_gate = {"author_ids_equal_cohort_order": bool(list(ids) == cref_ids), "model": mm["model"], "model_revision_ok": mm["model"][1] == g.E5_REVISION,
                "units_per_author_unique": bool(m.groupby("author_index").unit.apply(lambda s: sorted(s) == list(range(36))).all()),
                "rows_compared": int(err.shape[0]), "max_abs_error": float(err.max()), "rows_over_tol": int((err.max(1) > TOL_RAW).sum()), "tolerance": TOL_RAW,
                "formula": "keyed = (v/||v||)_float64 @ private_directions.direction, row (author_index, unit)"}
    raw_ok = raw_gate["author_ids_equal_cohort_order"] and raw_gate["model_revision_ok"] and raw_gate["units_per_author_unique"] and raw_gate["rows_over_tol"] == 0
    res["A1_raw_bridge"] = raw_gate
    # ================= A2 GRAIN bridge
    gunits, gviews, gall36, names = g.load_grain_units(list(ids))
    with np.load(g.MAT / "private_arrays.npz", allow_pickle=False) as z:
        mat_grain = np.asarray(z["grain"], float)                            # representation array only (not y/folds)
    common_sha = sha(g.PRE / "common_material_manifest.json"); parent_sha = sha(g.PARENT / "manifest.json")
    pre_man = json.loads((g.PRE / "manifest.json").read_text()); parent_man = json.loads((g.PARENT / "manifest.json").read_text())
    common = json.loads((g.PRE / "common_material_manifest.json").read_text())
    grain_gate = {"preflight_ids_unique_and_cover_cohort": True, "units_shape": list(gunits.shape), "feature_names": names,
                  "all36_recomputed_vs_preflight_max": float(np.abs(g.named_mean_std(gunits, names) - gall36).max()),
                  "all36_recomputed_vs_material_grain_max": float(np.abs(g.named_mean_std(gunits, names) - mat_grain).max()),
                  "view_blocks_vs_preflight_max": float(max(np.abs(g.named_mean_std(gunits[:, v * 12:(v + 1) * 12], names) - gviews[v]).max() for v in range(3))),
                  "address_view_index_equals_unit_div_12": bool((m.view_index.to_numpy() == un // 12).all()),
                  "material_manifest_common_sha_matches_preflight_file": mm["sources"]["common_manifest"] == common_sha,
                  "material_manifest_parent_sha_matches_parent_file": mm["sources"]["parent_manifest"] == parent_sha,
                  "preflight_source_identity_equals_parent_identity": pre_man.get("source_identity") == parent_man.get("identity"),
                  "common_parent_identity_equals_parent_identity": common.get("parent_identity") == parent_man.get("identity"),
                  "order_basis": "hash-chained manifests + identical views-flatten construction code in preflight, material and event addresses; per-comment ID sequence NOT content-verified (mask ledger covers only cue comments; view files contain text and were not loaded)"}
    grain_ok = (grain_gate["all36_recomputed_vs_preflight_max"] <= TOL and grain_gate["all36_recomputed_vs_material_grain_max"] <= TOL and grain_gate["view_blocks_vs_preflight_max"] <= TOL
                and grain_gate["address_view_index_equals_unit_div_12"] and grain_gate["material_manifest_common_sha_matches_preflight_file"]
                and grain_gate["material_manifest_parent_sha_matches_parent_file"] and grain_gate["preflight_source_identity_equals_parent_identity"])
    # ================= A3 representation fit-population audit
    bud = pd.read_csv(g.PRE / "budget_by_author.csv", usecols=["author_id", "old_reference_member", "old_e1_member"]).drop_duplicates("author_id").set_index("author_id")
    e1 = json.loads((g.E1 / "run_manifest.json").read_text())
    role_of = {int(a): "train" for a in train} | {int(a): "validation" for a in val} | {int(a): "held" for a in held}
    ov = {r: {"old_e1_member": 0, "old_reference_member": 0} for r in ("train", "validation", "held")}
    for a in range(624):
        rr = bud.loc[ids[a]]
        for k in ("old_e1_member", "old_reference_member"):
            ov[role_of[a]][k] += int(rr[k] == True)  # noqa: E712
    res["A3_representation_audit"] = {"representation": "VM05 lexical/SVD: TF-IDF (max 5000, 1-2 grams) + SVD 24 from e1_v72_full_20260715 common_source_comment_representation.joblib",
                                      "e1_population": {"fresh_e1_authors": e1.get("fresh_e1_authors"), "excluded_e0_authors": e1.get("excluded_e0_authors"), "split_counts": e1.get("author_split_counts")},
                                      "cohort_overlap_by_role": ov, "clean": all(v["old_e1_member"] == 0 for v in ov.values())}
    grain_ok = grain_ok and res["A3_representation_audit"]["clean"]
    res["A2_grain_bridge"] = grain_gate
    if not raw_ok:
        res["arms_not_executed"].update({"RAW_STATIC": "raw bridge failed", "RAW_ORDERED": "raw bridge failed"})
    if not grain_ok:
        res["arms_not_executed"]["GRAIN_STATIC"] = "GRAIN bridge or representation audit failed"
    # ================= per-author time-ordered arrays
    Rn = {}; Gu = {}; Zk = {}
    for a in train + val + held:
        ix = by[a]
        Zk[a] = Z[ix]
        if raw_ok:
            Rn[a] = g.unit_normalise(comments[a, un[ix]])
        if grain_ok:
            Gu[a] = gunits[a, un[ix]]
    pca = None
    if raw_ok:
        pca = g.fit_pca(np.concatenate([Rn[a] for a in train]))
        P = {a: g.pca_coords(Rn[a], pca) for a in Rn}

    def feats(a, t, arm, order=None):
        if arm == "KEYED_STATIC":
            return g.feat_keyed_static(Zk[a][:t])
        if arm == "RAW_STATIC":
            return g.feat_raw_static(P[a][:t])
        if arm == "GRAIN_STATIC":
            return g.feat_grain_static(Gu[a][:t], names)
        pre = P[a][:t]
        if order is None:
            return g.feat_raw_ordered(pre)
        pp = pre[order]
        return g.feat_raw_ordered(pp, mean_override=pre.mean(0), std_override=pre.std(0, ddof=0))
    arms = [a for a in ARMS if a not in res["arms_not_executed"]]
    res["arms_executed"] = arms

    def design(auths, arm):
        Xd, Yd, keys = [], [], []
        for a in auths:
            for t in range(g.T0, g.NEV):
                Xd.append(feats(a, t, arm)); Yd.append(Zk[a][t]); keys.append((int(a), t))
        return np.array(Xd), np.array(Yd), keys
    models, fitinfo = {}, {}
    for arm in arms:
        Xtr, Ytr, _ = design(train, arm)
        sc = g.fit_scaler(Xtr); Xs = g.apply_scaler(Xtr, sc)
        B, b = g.fit_affine_ridge(Xs, Ytr)
        gB, gb = g.ridge_gradients(Xs, Ytr, B, b)
        models[arm] = {"sc": sc, "B": B, "b": b}
        fitinfo[arm] = {"columns": int(Xtr.shape[1]), "rows": int(Xtr.shape[0]), "zero_sd_columns": int(sc["zero_sd"].sum()), "max_grad_B": gB, "max_grad_b": gb,
                        "B_frobenius": float(np.linalg.norm(B))}
    res["fits"] = fitinfo
    if pca is not None:
        res["pca"] = {"k": g.K_PCA, "retained_variance_share": pca["retained_variance_share"], "fit_vectors": int(sum(len(Rn[a]) for a in train)), "note": "reported, not tuned"}

    def predict(arm, x):
        mo = models[arm]
        return g.apply_scaler(x[None, :], mo["sc"])[0] @ mo["B"] + mo["b"]
    # frozen references (packet 22 parameterization)
    with np.load(F14 / "history_increment_r1/coefficients.npz") as cz:
        pop = {"phi": cz["phi"], "V5": cz["V5"], "cmean": cz["cmean"]}
        Bl, Bh, wl, wh = cz["B_last"], cz["B_hist"], float(cz["w_last"]), float(cz["w_hist"])
    raw_cond = D["raw"]

    def score(auths, role):
        rows = []
        for a in auths:
            ix = by[a]
            for t in range(g.T0, g.NEV):
                y = Zk[a][t]
                r = {"author": int(a), "t": t, "novel_raw_venue": bool(raw_cond[ix[t]] not in set(raw_cond[ix[:t]]))}
                ref = h22.keyed_predict(Zk[a][:t], pop, Bl, Bh, wl, wh)
                r["mse_REF_prefix_mean"] = float(((ref["static"] - y) ** 2).mean()); r["mse_REF_coupled_hist"] = float(((ref["coupled_hist"] - y) ** 2).mean())
                for arm in arms:
                    e = predict(arm, feats(a, t, arm)) - y
                    r[f"mse_{arm}"] = float((e ** 2).mean())
                    for j in range(30):
                        r[f"sq_{arm}_{j}"] = float(e[j] ** 2)
                if "RAW_ORDERED" in arms:
                    order = g.permuted_order(t, ROLE[role], a)
                    pre = P[a][:t]; pp = pre[order]
                    r["perm_static_invariance_err"] = float(max(np.abs(pp.mean(0) - pre.mean(0)).max(), np.abs(pp.std(0, ddof=0) - pre.std(0, ddof=0)).max()))
                    r["mse_RAW_ORDERED_permuted"] = float(((predict("RAW_ORDERED", feats(a, t, "RAW_ORDERED", order)) - y) ** 2).mean())
                rows.append(r)
        return pd.DataFrame(rows)
    Vd = score(val, "source_validation")
    Hd = score(held, "held")
    # replay packet-22 references on held
    p22 = pd.read_csv(F14 / "history_increment_r1/private_held_keyed_event_losses.csv")
    mg = Hd.merge(p22, on=["author", "t"])
    res["reference_replay_vs_packet22"] = {"rows": int(len(mg)), "prefix_mean_max": float(np.abs(mg.mse_REF_prefix_mean - mg.kmse_static).max()),
                                           "coupled_hist_max": float(np.abs(mg.mse_REF_coupled_hist - mg.kmse_coupled_hist).max())}
    if len(mg) != 1320 or max(res["reference_replay_vs_packet22"]["prefix_mean_max"], res["reference_replay_vs_packet22"]["coupled_hist_max"]) > TOL:
        OUT.mkdir(parents=True); (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        raise RuntimeError("reference replay failed")
    auths = np.array(sorted(held)); draws = np.random.default_rng(20260926).integers(0, len(auths), size=(2000, len(auths)))
    vauths = np.array(sorted(val)); vdraws = np.random.default_rng(20260927).integers(0, len(vauths), size=(2000, len(vauths)))
    cols = [f"mse_{a}" for a in arms] + ["mse_REF_prefix_mean", "mse_REF_coupled_hist"]
    res["validation_mse_descriptive"] = {c: float(Vd[c].mean()) for c in cols}
    res["held_mse"] = {c: float(Hd[c].mean()) for c in cols}
    contrasts = []
    for a, b_ in (("RAW_STATIC", "KEYED_STATIC"), ("GRAIN_STATIC", "KEYED_STATIC"), ("RAW_STATIC", "GRAIN_STATIC"), ("RAW_ORDERED", "RAW_STATIC"),
                  ("RAW_ORDERED", "KEYED_STATIC"), ("GRAIN_STATIC", "RAW_ORDERED")):
        if a in arms and b_ in arms:
            contrasts.append((a, b_))
    for a in arms:
        contrasts += [(a, "REF_prefix_mean"), (a, "REF_coupled_hist")]
    res["held_paired_deltas"] = {f"{a}-{b_}": paired(Hd, f"mse_{a}", f"mse_{b_}", auths, draws) for a, b_ in contrasts}
    res["held_strata"] = {f"{a}-{b_}": strata(Hd, f"mse_{a}", f"mse_{b_}") for a, b_ in contrasts if not b_.startswith("REF_")}
    res["held_per_coordinate_mse"] = {a: {h22.ph.FACETS[j]: float(Hd[f"sq_{a}_{j}"].mean()) for j in range(30)} for a in arms}
    if "RAW_ORDERED" in arms:
        res["order_control"] = {"static_invariance_max_err_validation": float(Vd.perm_static_invariance_err.max()), "static_invariance_max_err_held": float(Hd.perm_static_invariance_err.max()),
                                "validation_permuted_minus_original": paired(Vd, "mse_RAW_ORDERED_permuted", "mse_RAW_ORDERED", vauths, vdraws),
                                "held_permuted_minus_original": paired(Hd, "mse_RAW_ORDERED_permuted", "mse_RAW_ORDERED", auths, draws),
                                "note": "same frozen scaler/readout; not an exchangeable-null p-value"}
    # ================= cold bundle replay (saved arrays only, no fitting)
    OUT.mkdir(parents=True)
    bundle = {"names_json": np.array(json.dumps(names))}
    if pca is not None:
        bundle.update(pca_mu=pca["mu"], pca_W=pca["W"])
    for arm in arms:
        mo = models[arm]
        bundle.update({f"{arm}_sc_mu": mo["sc"]["mu"], f"{arm}_sc_sd": mo["sc"]["sd"], f"{arm}_sc_zero": mo["sc"]["zero_sd"], f"{arm}_B": mo["B"], f"{arm}_b": mo["b"]})
    np.savez(OUT / "private_bundle.npz", **bundle)
    with np.load(OUT / "private_bundle.npz") as cb:
        cold = []
        for a in held:
            Pa = ((g.unit_normalise(comments[a, un[by[a]]]) - cb["pca_mu"]) @ cb["pca_W"]) if pca is not None else None
            for t in range(g.T0, g.NEV):
                row = []
                for arm in arms:
                    if arm == "KEYED_STATIC":
                        x = g.feat_keyed_static(Zk[a][:t])
                    elif arm == "RAW_STATIC":
                        x = g.feat_raw_static(Pa[:t])
                    elif arm == "GRAIN_STATIC":
                        x = g.feat_grain_static(gunits[a, un[by[a]]][:t], json.loads(str(cb["names_json"])))
                    else:
                        x = g.feat_raw_ordered(Pa[:t])
                    xs = (x - cb[f"{arm}_sc_mu"]) / cb[f"{arm}_sc_sd"]; xs[cb[f"{arm}_sc_zero"]] = 0.0
                    row.append(float(((xs @ cb[f"{arm}_B"] + cb[f"{arm}_b"] - Zk[a][t]) ** 2).mean()))
                cold.append(row)
    res["cold_replay_max_abs"] = float(np.abs(np.array(cold) - Hd[[f"mse_{a}" for a in arms]].to_numpy()).max())
    Hd[[c for c in Hd.columns if not c.startswith("sq_")]].to_csv(OUT / "private_held_event_losses.csv", index=False)
    Vd[[c for c in Vd.columns if not c.startswith("sq_")]].to_csv(OUT / "private_validation_event_losses.csv", index=False)
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/r08_raw_grain_history.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print("done", round(res["seconds"], 1), "arms", arms, "raw_err", raw_gate["max_abs_error"], "grain_ok", grain_ok, "cold", res["cold_replay_max_abs"])


if __name__ == "__main__":
    main()
