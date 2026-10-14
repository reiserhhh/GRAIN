#!/usr/bin/env python3
"""R17 packet 1 analysis: no model calls; geometric sensitivity only."""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh

from scripts.run_grain_r17_natural_sensitivity import (
    BOOT_SEED,
    EDITS_A1,
    LB,
    PROTOCOL,
    PROTOCOL_SHA_A1,
    _ledger,
    _versions_for,
    digest,
    load_manifest,
    load_material,
    write,
)

N_BOOT = 2000


def facet_directions():
    from grain_research.local_bilinear import keyed_directions
    from scripts.prepare_grain_predecessor_reference import DEFAULT

    vectors, start = [], 0
    for batch in sorted((DEFAULT / "sentence_batches").glob("batch_*.npz")):
        meta = json.loads(batch.with_suffix(".json").read_text())
        if digest(batch) != meta["sha256"] or meta["start"] != start:
            raise ValueError("Item cache mismatch")
        with np.load(batch, allow_pickle=False) as z:
            vectors.append(z["vectors"])
        start = meta["stop"]
        if start >= 300:
            break
    q = np.concatenate(vectors)[:300]
    items = json.loads((DEFAULT / "ipip_neo300.json").read_text())["items"]
    return keyed_directions(q, items)


def grain_runtime():
    from scripts.run_grain_native_reference import E1

    return joblib.load(E1 / "artifacts/common_source_comment_representation.joblib")


def cosdist(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 1.0 - np.sum(a * b, axis=-1)


def eucl(a, b):
    return np.linalg.norm(a - b, axis=-1)


def boot_ratio(num, den, rng_seed=BOOT_SEED):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(rng_seed)
    n = len(num)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num[idx].mean(1) / den[idx].mean(1)
    point = num.mean() / den.mean() if den.mean() > 0 else np.nan
    return point, *np.nanpercentile(r, [2.5, 97.5])


def boot_mean(x, rng_seed=BOOT_SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def classify(lo, hi, mode, max_edit=None):
    if mode == "TOKENIZER_IDENTICAL":
        return "TOKENIZER_IDENTICAL"
    if max_edit is not None and max_edit == 0.0:
        return "PREPROCESSING_IDENTICAL"
    if not np.isfinite(hi):
        return "UNDEFINED"
    if hi < 0.5:
        return "SMALL"
    if lo > 1.0:
        return "LARGE"
    if lo <= 1.0 <= hi:
        return "COMPARABLE"
    return "INTERMEDIATE"


def analyze(args):
    out = args.output
    if digest(PROTOCOL) != PROTOCOL_SHA_A1:
        raise ValueError("Protocol changed")
    status = json.loads((out / "accounting" / "status.json").read_text())
    if status["status"] != "COMPLETE":
        raise SystemExit("Encoding not complete")
    _, started, _ok, failed = _ledger(out)
    events = [json.loads(x) for x in (out / "accounting/attempts.jsonl").read_text().splitlines()]
    identical = {e["item"] for e in events if e.get("mode") == "TOKENIZER_IDENTICAL"}
    manifest = load_manifest(out)
    material = load_material()
    direction, facet_names = facet_directions()
    runtime = grain_runtime()
    domains = sorted({f[0] for f in facet_names})
    dom_w = np.column_stack([[1 / 6 if f[0] == d else 0 for f in facet_names] for d in domains])
    compat, per_unit, summary, signed, readout_rows, eigen = {}, [], [], [], [], {}
    for corpus in ("pandora", "essays"):
        m = manifest[manifest.corpus == corpus].reset_index(drop=True)
        with np.load(LB / corpus / "private_units.npz", allow_pickle=False) as z:
            cg, ch, csem, cauth = z["g"], z["h"], z["semantic"], z["authors"]
        with np.load(LB / corpus / "private_blocks.npz", allow_pickle=False) as z:
            blocks = {k: z[k] for k in z.files}
        with np.load(LB / corpus / "private_predictions.npz", allow_pickle=False) as z:
            preds = {k: z[k] for k in ("semantic", "grain")}
            held_ix = z["held_indices"]
        texts = {v: [] for v in ("base", *EDITS_A1, "partner")}
        for rec in m.to_dict("records"):
            vs = _versions_for(material, rec)
            for v, bucket in texts.items():
                bucket.append(vs[v])
        vec = {r: {} for r in ("e5", "bge")}
        for r, store in vec.items():
            for v in ("base", *EDITS_A1, "partner"):
                if r == "e5" and v == "partner":
                    continue
                store[v] = np.stack([
                    np.load(out / "private_vectors" / r / f"{corpus}_{a}_{v}.npy")
                    for a in m.author_index])
        ur, pr = m.unit_row.to_numpy(), m.partner_row.to_numpy()
        # compatibility and drift
        g_local = {v: runtime.transform(t) for v, t in texts.items()}
        drift = cosdist(vec["e5"]["base"], ch[ur])
        sem_proj_cached = (ch[ur] / np.linalg.norm(ch[ur], axis=1, keepdims=True)) @ direction
        compat[corpus] = {
            "grain_base_max_abs_diff_vs_cache": float(np.max(np.abs(g_local["base"] - cg[ur]))),
            "grain_partner_max_abs_diff_vs_cache": float(np.max(np.abs(g_local["partner"] - cg[pr]))),
            "ipip_projection_replay_max_abs_diff": float(np.max(np.abs(sem_proj_cached - csem[ur]))),
            "e5_local_vs_cached_base_cosdist_mean": float(drift.mean()),
            "e5_local_vs_cached_base_cosdist_max": float(drift.max()),
            "e5_cache_device": json.loads((LB / corpus / "material.json").read_text())["encoder"].get("device"),
        }
        coords = {}
        # e5 raw: edits local-local; natural cached-cached
        coords["e5_raw"] = {
            "edit": {v: cosdist(vec["e5"]["base"], vec["e5"][v]) for v in EDITS_A1},
            "within": cosdist(ch[ur], ch[pr]),
            "base_nat": ch[ur], "metric": cosdist,
            "delta": {v: vec["e5"][v] - vec["e5"]["base"] for v in EDITS_A1},
        }
        coords["bge_raw"] = {
            "edit": {v: cosdist(vec["bge"]["base"], vec["bge"][v]) for v in EDITS_A1},
            "within": cosdist(vec["bge"]["base"], vec["bge"]["partner"]),
            "base_nat": vec["bge"]["base"], "metric": cosdist,
            "delta": {v: vec["bge"][v] - vec["bge"]["base"] for v in EDITS_A1},
        }
        proj = {v: vec["e5"][v] @ direction for v in ("base", *EDITS_A1)}
        cproj_b = csem[ur]
        cproj_p = (ch[pr] / np.linalg.norm(ch[pr], axis=1, keepdims=True)) @ direction
        coords["e5_ipip30"] = {
            "edit": {v: eucl(proj["base"], proj[v]) for v in EDITS_A1},
            "within": eucl(cproj_b, cproj_p), "base_nat": cproj_b, "metric": eucl,
            "delta": {v: proj[v] - proj["base"] for v in EDITS_A1},
        }
        coords["grain24"] = {
            "edit": {v: eucl(g_local["base"], g_local[v]) for v in EDITS_A1},
            "within": eucl(cg[ur], cg[pr]), "base_nat": cg[ur], "metric": eucl,
            "delta": {v: g_local[v] - g_local["base"] for v in EDITS_A1},
        }
        n = len(m)
        for cname, c in coords.items():
            base_nat = c["base_nat"]
            dmat = np.array([[float(c["metric"](base_nat[i], base_nat[j])) for j in range(n)]
                             for i in range(n)])
            between = (dmat.sum(1)) / (n - 1)
            for v in EDITS_A1:
                reader = cname.split("_")[0]
                mode = ("TOKENIZER_IDENTICAL"
                        if reader in ("e5", "bge") and v == "case_fold"
                        and all(f"{reader}:{corpus}:{a}:case_fold" in identical
                                or (reader == "e5" and corpus == "pandora" and a == 551)
                                for a in m.author_index)
                        else "COMPUTED")
                e = c["edit"][v]
                rw, rwl, rwh = boot_ratio(e, c["within"])
                rb, rbl, rbh = boot_ratio(e, between)
                d = c["delta"][v]
                msq = np.mean(np.sum(d ** 2, 1))
                sys_share = float(np.sum(d.mean(0) ** 2) / msq) if msq > 0 else np.nan
                summary.append({
                    "corpus": corpus, "coordinate": cname, "edit": v, "n_units": n,
                    "edit_mean": e.mean(), "edit_median": np.median(e), "edit_sd": e.std(ddof=1),
                    "within_author_mean": c["within"].mean(),
                    "between_author_mean": between.mean(),
                    "ratio_edit_within": rw, "ratio_edit_within_lo": rwl, "ratio_edit_within_hi": rwh,
                    "ratio_edit_between": rb, "ratio_edit_between_lo": rbl, "ratio_edit_between_hi": rbh,
                    "share_edit_gt_within": float(np.mean(e > c["within"])),
                    "systematic_share": sys_share,
                    "e5_device_drift_mean": float(drift.mean()) if cname == "e5_raw" else np.nan,
                    "classification_vs_within": classify(rwl, rwh, mode, float(np.max(np.abs(e)))),
                    "max_edit_distance": float(np.max(np.abs(e))),
                    "mode": mode,
                })
                for i, a in enumerate(m.author_index):
                    per_unit.append({"corpus": corpus, "author_index": int(a),
                                     "coordinate": cname, "edit": v,
                                     "edit_distance": float(e[i]),
                                     "within_author_distance": float(c["within"][i]),
                                     "between_author_mean_distance": float(between[i])})
        # signed means in between-author SD units
        std_sets = {
            "e5_ipip_domain": (proj["base"] @ dom_w, {v: (proj[v] - proj["base"]) @ dom_w for v in EDITS_A1}, domains),
            "grain24": (g_local["base"], coords["grain24"]["delta"], [f"g{j:02d}" for j in range(24)]),
        }
        for sname, (base, deltas, names) in std_sets.items():
            sd = base.std(0, ddof=1)
            for v in EDITS_A1:
                for j, nm in enumerate(names):
                    z = deltas[v][:, j] / sd[j] if sd[j] > 0 else np.full(n, np.nan)
                    mu, lo, hi = boot_mean(z)
                    signed.append({"corpus": corpus, "coordinate_set": sname, "coordinate": nm,
                                   "edit": v, "mean_delta_sd_units": mu, "ci_lo": lo, "ci_hi": hi,
                                   "rms_delta_sd_units": float(np.sqrt(np.mean(z ** 2))),
                                   "ci_excludes_zero": bool(lo > 0 or hi < 0)})
        # historical supported readouts by single-unit substitution
        from grain_research.vector_mechanisms.external_heads import TRAITS

        pos = {int(a): k for k, a in enumerate(held_ix)}
        for arm in ("semantic", "grain"):
            obj = joblib.load(LB / corpus / f"private_model_{arm}.joblib")
            feat, head = obj["features"], obj["head"]
            psd = preds[arm].std(0, ddof=1)
            for i, rec in enumerate(m.to_dict("records")):
                a = int(rec["author_index"])
                rows = np.flatnonzero(cauth == a)
                k = int(np.flatnonzero(rows == rec["unit_row"])[0])
                n_a = len(rows)

                def score(block_row, arm=arm, a=a, blocks=blocks, feat=feat, head=head):
                    b = {kk: vv for kk, vv in blocks.items()}
                    b[arm] = blocks[arm].copy()
                    b[arm][a] = block_row
                    return head.predict(feat.transform(b, None, [a]))[0]

                base_score = score(blocks[arm][a])
                if not np.allclose(base_score, preds[arm][pos[a]], atol=1e-9):
                    raise ValueError("Readout replay mismatch")
                variants = {}
                if arm == "semantic":
                    for v in EDITS_A1:
                        variants[v] = blocks[arm][a] + (vec["e5"][v][i] - vec["e5"]["base"][i]) / n_a
                    variants["partner_substitution"] = blocks[arm][a] + (ch[rec["partner_row"]] - ch[rec["unit_row"]]) / n_a
                else:
                    G = cg[rows].astype(float)
                    for v in (*EDITS_A1, "partner_substitution"):
                        G2 = G.copy()
                        G2[k] = g_local[v][i] if v != "partner_substitution" else cg[rec["partner_row"]]
                        variants[v] = np.r_[G2.mean(0), G2.std(0, ddof=0)]
                for v, row in variants.items():
                    dz = (score(row) - base_score) / psd
                    readout_rows.append({"corpus": corpus, "arm": arm, "author_index": a,
                                         "variant": v, "n_units_author": n_a,
                                         **{f"d_{t}": float(x) for t, x in zip(TRAITS, dz, strict=True)}})
        # generalized eigenproblem
        for cname, base, deltas in (("e5_ipip30", proj["base"], coords["e5_ipip30"]["delta"]),
                                    ("grain24", g_local["base"], coords["grain24"]["delta"])):
            sets = {"pooled_nonidentical": [v for v in EDITS_A1
                                            if not (cname.startswith("e5") and v == "case_fold")],
                    **{v: [v] for v in EDITS_A1}}
            SB = np.cov(base, rowvar=False)
            res = {}
            for sname, vs in sets.items():
                D = np.concatenate([deltas[v] for v in vs])
                SN = D.T @ D / len(D)
                tr = np.trace(SN)
                if tr <= 0:
                    res[sname] = {"status": "ZERO_NUISANCE_ENERGY"}
                    continue
                SNr = SN + 1e-3 * tr / SN.shape[0] * np.eye(SN.shape[0])
                lam = eigh(SB, SNr, eigvals_only=True)
                lam = lam[::-1]
                wB, UB = np.linalg.eigh(SB)
                top = UB[:, np.argsort(wB)[::-1][:5]]
                entry = {
                    "n_deltas": len(D), "dim": SN.shape[0],
                    "lambda_top3": lam[:3].tolist(), "lambda_bottom3": lam[-3:].tolist(),
                    "lambda_median": float(np.median(lam)),
                    "nuisance_trace_share_in_top5_natural_pcs": float(np.trace(top.T @ SN @ top) / tr),
                    "natural_trace_share_top5": float(np.sort(wB)[::-1][:5].sum() / wB.sum()),
                }
                if cname == "e5_ipip30":
                    entry["domain_rayleigh_between_over_edit"] = {
                        d: float((dom_w[:, j] @ SB @ dom_w[:, j]) / (dom_w[:, j] @ SNr @ dom_w[:, j]))
                        for j, d in enumerate(domains)}
                res[sname] = entry
            eigen.setdefault(corpus, {})[cname] = res
    pd.DataFrame(per_unit).to_csv(out / "per_unit_distances.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(out / "summary_by_coordinate.csv", index=False)
    pd.DataFrame(signed).to_csv(out / "signed_means.csv", index=False)
    rd = pd.DataFrame(readout_rows)
    rd.to_csv(out / "readout_deltas.csv", index=False)
    from grain_research.vector_mechanisms.external_heads import TRAITS

    rs = []
    for (corpus, arm, variant), g in rd.groupby(["corpus", "arm", "variant"]):
        D = g[[f"d_{t}" for t in TRAITS]].to_numpy()
        norm = np.linalg.norm(D, axis=1)
        partner = rd[(rd.corpus == corpus) & (rd.arm == arm) & (rd.variant == "partner_substitution")]
        pn = np.linalg.norm(partner[[f"d_{t}" for t in TRAITS]].to_numpy(), axis=1)
        r, lo, hi = boot_ratio(norm, pn)
        row = {"corpus": corpus, "arm": arm, "variant": variant, "n": len(g),
               "mean_norm_sd_units": norm.mean(), "partner_mean_norm_sd_units": pn.mean(),
               "ratio_vs_partner": r, "ratio_lo": lo, "ratio_hi": hi,
               }
        for t in TRAITS:
            mu, l2, h2 = boot_mean(g[f"d_{t}"])
            row[f"mean_{t}"], row[f"ci_{t}"] = mu, f"[{l2:.4f},{h2:.4f}]"
        if arm == "semantic" and variant != "partner_substitution":
            # Linear head: an edit of every unit shifts the author block by the mean
            # unit delta; estimated by n_i x single-unit delta averaged over authors.
            S = D * g.n_units_author.to_numpy()[:, None]
            rng = np.random.default_rng(BOOT_SEED)
            idx = rng.integers(0, len(S), size=(N_BOOT, len(S)))
            bn = np.linalg.norm(S[idx].mean(1), axis=1)
            row["all_unit_systematic_shift_norm"] = float(np.linalg.norm(S.mean(0)))
            blo, bhi = np.percentile(bn, [2.5, 97.5])
            row["all_unit_systematic_shift_ci"] = f"[{blo:.4f},{bhi:.4f}]"
            for t, val in zip(TRAITS, S.mean(0), strict=True):
                row[f"all_unit_shift_{t}"] = float(val)
        rs.append(row)
    pd.DataFrame(rs).to_csv(out / "readout_summary.csv", index=False)
    write(out / "compat_checks.json", compat)
    write(out / "generalized_eigen.json", eigen)
    files = ["per_unit_distances.csv", "summary_by_coordinate.csv", "signed_means.csv",
             "readout_deltas.csv", "readout_summary.csv", "compat_checks.json",
             "generalized_eigen.json"]
    write(out / "analysis_manifest.json", {
        "protocol_sha256": PROTOCOL_SHA_A1,
        "code_sha256": digest(Path(__file__)),
        "forward_attempts_total": len(started),
        "forward_attempts_by_reader": dict(collections.Counter(e["reader"] for e in started)),
        "tokenizer_identical_items": len(identical),
        "failed": len(failed),
        "cap_total": 1000,
        "outputs": {f: digest(out / f) for f in files},
        "label": "geometric sensitivity only; not personality selectivity",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    })
    print(summ[["corpus", "coordinate", "edit", "ratio_edit_within", "ratio_edit_within_lo",
                "ratio_edit_within_hi", "ratio_edit_between", "systematic_share",
                "classification_vs_within"]].round(4).to_string())
