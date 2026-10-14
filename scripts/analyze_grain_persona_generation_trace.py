#!/usr/bin/env python3
"""Analysis of a persona generation trace run: audits, blinded readers with
task/family holdouts and paired sign-flip nulls, shrinkage displacement, causal
localization summaries with discovery-only site selection, and the GRAIN
endpoint on regenerated continuations. Never fits on evaluation cells."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.persona_trace import (
    AXES,
    auc,
    balanced_accuracy,
    displacement_summary,
    mahalanobis2,
    paired_signflip_null,
    strip_leak,
    within_condition_covariance,
)


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=float) + "\n")


def tfidf_probe(train_texts, train_y, test_texts, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    pipe.fit(train_texts, train_y)
    return pipe.predict_proba(test_texts)[:, list(pipe.classes_).index(1)]


def linear_probe(train_x, train_y, test_x, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(scaler.transform(train_x), train_y)
    return clf.predict_proba(scaler.transform(test_x))[:, list(clf.classes_).index(1)]


def evaluate_scores(rows, scores, axis):
    """AUC / balanced accuracy / sign-flip null on high-vs-low rows of one axis."""
    sel = [r for r in rows if r["axis"] == axis and r["text_id"] in scores]
    if len(sel) < 4:
        return None
    s = np.array([scores[r["text_id"]] for r in sel])
    y = np.array([r["pole"] for r in sel])
    b = np.array([r["block"] for r in sel])
    thr = np.median(s)
    return {"n": len(sel), "auc": auc(s, y), "balanced_accuracy_median_split": balanced_accuracy(np.where(s > thr, 1, -1), y),
            **{f"null_{k}": v for k, v in paired_signflip_null(s, y, b, n=2000, seed=1).items()}}


def holdout_protocols(rows, roles):
    families = sorted({r["family"] for r in rows if r["family"] != "none"})
    protocols = []
    for fam in families:
        others = [f for f in families if f != fam]
        protocols.append({"name": f"family_holdout:test={fam}", "train": lambda r, fam=fam: r["family"] in [f for f in families if f != fam] and r["situation"] in roles["discovery"] + roles["development"], "test": lambda r, fam=fam: r["family"] == fam})
        _ = others
    protocols.append({"name": "situation_holdout:test=evaluation", "train": lambda r: r["situation"] in roles["discovery"] + roles["development"], "test": lambda r: r["situation"] in roles["evaluation"]})
    protocols.append({"name": "strict:train=discovery_familyA,test=evaluation_familyB", "train": lambda r: r["family"] == families[0] and r["situation"] in roles["discovery"], "test": lambda r: r["family"] == families[-1] and r["situation"] in roles["evaluation"]})
    return protocols


def reader_block(rows, roles, texts, grain, local, judge):
    out = {}
    protocols = holdout_protocols(rows, roles)
    for axis in AXES:
        axis_rows = [r for r in rows if r["axis"] == axis]
        out[axis] = {}
        for variant, text_map in (("raw", texts), ("stripped", {k: strip_leak(v) for k, v in texts.items()})):
            for proto in protocols:
                train = [r for r in axis_rows if proto["train"](r) and r["text_id"] in text_map]
                test = [r for r in axis_rows if proto["test"](r) and r["text_id"] in text_map]
                if len(train) < 4 or len(test) < 4 or len({r["pole"] for r in train}) < 2:
                    continue
                key = f"{proto['name']}|{variant}"
                probs = tfidf_probe([text_map[r["text_id"]] for r in train], [r["pole"] for r in train], [text_map[r["text_id"]] for r in test])
                out[axis].setdefault("tfidf", {})[key] = evaluate_scores(test, {r["text_id"]: p for r, p in zip(test, probs)}, axis)
                gkey = "grain" if variant == "raw" else "grain_stripped"
                gtrain = [r for r in train if grain[gkey].get(r["text_id"], {}).get("g")]
                gtest = [r for r in test if grain[gkey].get(r["text_id"], {}).get("g")]
                if len(gtrain) >= 4 and len(gtest) >= 4:
                    probs = linear_probe(np.array([grain[gkey][r["text_id"]]["g"] for r in gtrain]), [r["pole"] for r in gtrain], np.array([grain[gkey][r["text_id"]]["g"] for r in gtest]))
                    out[axis].setdefault("grain4_probe", {})[key] = evaluate_scores(gtest, {r["text_id"]: p for r, p in zip(gtest, probs)}, axis)
        # training-free readers: full set and evaluation cells
        for name, table in (("local_query_raw", local.get("raw", {})), ("local_query_stripped", local.get("stripped", {}))):
            scores = {k: v[axis] for k, v in table.items() if isinstance(v, dict) and axis in v}
            out[axis][name] = {"all": evaluate_scores(axis_rows, scores, axis), "evaluation_situation": evaluate_scores([r for r in axis_rows if r["situation"] in roles["evaluation"]], scores, axis)}
            other = next(a for a in AXES if a != axis)
            out[axis][name]["selectivity_on_other_axis_pairs"] = evaluate_scores([r for r in rows if r["axis"] == other], scores, other)
        if judge:
            scores = {k: v.get(axis) for k, v in judge.items() if isinstance(v.get(axis), (int, float))}
            out[axis]["deepseek_judge"] = {"all": evaluate_scores(axis_rows, scores, axis), "evaluation_situation": evaluate_scores([r for r in axis_rows if r["situation"] in roles["evaluation"]], scores, axis)}
            other = next(a for a in AXES if a != axis)
            out[axis]["deepseek_judge"]["selectivity_on_other_axis_pairs"] = evaluate_scores([r for r in rows if r["axis"] == other], scores, other)
    return out


def control_positions(rows, local, judge, grain):
    """Where neutral and non-persona texts fall on the training-free readers."""
    out = {}
    for axis in AXES:
        for name, table in (("local_query_raw", local.get("raw", {})), ("deepseek_judge", judge)):
            groups = defaultdict(list)
            for r in rows:
                v = table.get(r["text_id"])
                v = v.get(axis) if isinstance(v, dict) else None
                if isinstance(v, (int, float)):
                    groups[r["condition"]].append(v)
            out[f"{axis}|{name}"] = {k: {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None} for k, v in sorted(groups.items())}
    return out


def displacement_block(rows, grain, texts, roles):
    out = {}
    feats = {k: v["g"] for k, v in grain["grain"].items() if v.get("g")}
    if len(feats) >= 8:
        cov, shrink, n = within_condition_covariance(feats, rows)
        out["grain4"] = {"metric": "within-condition Ledoit-Wolf", "shrinkage": shrink, "residuals": n, **{axis: displacement_summary(feats, rows, axis, cov) for axis in AXES}}
        for axis in AXES:
            neutral = [np.asarray(feats[r["text_id"]]) for r in rows if r["condition"] == "neutral" and r["text_id"] in feats]
            nonp = [np.asarray(feats[r["text_id"]]) for r in rows if r["condition"].startswith("nonpersona") and r["text_id"] in feats]
            high = [np.asarray(feats[r["text_id"]]) for r in rows if r["condition"] == f"{axis}_high" and r["text_id"] in feats]
            low = [np.asarray(feats[r["text_id"]]) for r in rows if r["condition"] == f"{axis}_low" and r["text_id"] in feats]
            if neutral and high and low:
                out["grain4"][axis]["D2_high_vs_neutral"] = mahalanobis2(np.mean(high, 0) - np.mean(neutral, 0), cov)
                out["grain4"][axis]["D2_low_vs_neutral"] = mahalanobis2(np.mean(low, 0) - np.mean(neutral, 0), cov)
                if nonp:
                    out["grain4"][axis]["D2_nonpersona_vs_neutral"] = mahalanobis2(np.mean(nonp, 0) - np.mean(neutral, 0), cov)
    # TF-IDF in a fixed low-rank basis fitted on discovery texts only
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    disc = [r for r in rows if r["situation"] in roles["discovery"] and r["text_id"] in texts]
    if len(disc) >= 12:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([texts[r["text_id"]] for r in disc])
        svd = TruncatedSVD(n_components=8, random_state=0).fit(vec.transform([texts[r["text_id"]] for r in disc]))
        lowrank = {k: svd.transform(vec.transform([v]))[0].tolist() for k, v in texts.items()}
        cov, shrink, n = within_condition_covariance(lowrank, rows)
        out["tfidf_svd8"] = {"metric": "within-condition Ledoit-Wolf on discovery-fitted SVD8", "shrinkage": shrink, "residuals": n, **{axis: displacement_summary(lowrank, rows, axis, cov) for axis in AXES}}
    return out


def pooled_slope(df, seed=20260914, draws=2000, den=None):
    """Least-squares slope of delta_lp on the prompt effect through the origin, pair-bootstrapped.

    Small prompt effects make per-row recovery ratios explode; the slope weights each row by the
    size of the effect it is trying to recover.  Bootstrap resamples pairs (both continuations).
    """
    den = (df.lp_high_prompt - df.lp_low_prompt).to_numpy() if den is None else np.asarray(den, dtype=float)
    num = df.delta_lp.to_numpy()
    if not len(den) or float(np.sum(den * den)) < 1e-12:
        return {"slope": None, "ci95": None, "n": len(den)}
    slope = float(np.sum(num * den) / np.sum(den * den))
    pairs = df.pair.to_numpy()
    uniq = np.unique(pairs)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.concatenate([np.flatnonzero(pairs == p) for p in pick])
        d, n = den[mask], num[mask]
        if float(np.sum(d * d)) > 1e-12:
            boots.append(float(np.sum(n * d) / np.sum(d * d)))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None
    return {"slope": slope, "ci95": ci, "n": len(den), "pairs": len(uniq)}


def localization_block(run, roles):
    effects = read_jsonl(run / "localize_effects.jsonl")
    rows = [e for e in effects if e.get("kind") not in (None, "baseline", "skipped_unaligned", "skipped_short")]
    base = [e for e in effects if e.get("kind") == "baseline"]
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    df["axis"] = df.pair.str.split("|").str[0]
    df["positive"] = df.recovery > 0
    summary = df.groupby(["role", "kind", "site"]).agg(recovery_mean=("recovery", "mean"), recovery_median=("recovery", "median"), positive_share=("positive", "mean"), delta_lp_mean=("delta_lp", "mean"), n=("recovery", "size")).reset_index()
    by_axis = df.groupby(["role", "axis", "kind", "site"]).agg(recovery_mean=("recovery", "mean"), recovery_median=("recovery", "median"), n=("recovery", "size")).reset_index()
    disc = summary[(summary.role == "discovery") & (summary.kind == "persona_patch")].copy()
    disc["depth"] = disc.site.str.extract(r"(\d+)").astype(int)
    disc = disc.sort_values("depth")
    rand = summary[(summary.role == "discovery") & (summary.kind == "random_equal_norm")].set_index("site").recovery_mean
    # Selection rule (discovery only): the deepest sites where persona positions still carry
    # >= 0.5 of the prompt effect (the transfer boundary), plus the first site below 0.5.
    carrying = disc[disc.recovery_median >= 0.5]
    below = disc[disc.recovery_median < 0.5]
    selected = carrying.tail(2).site.tolist() + below.head(1).site.tolist()
    baseline = pd.DataFrame([{"pair": b["pair"], "role": b["role"], "cont": c, "effect_high_minus_low_prompt": v["high_prompt"] - v["low_prompt"]} for b in base for c, v in b["values"].items()])
    slopes = [{"role": role, "kind": kind, "site": site, **pooled_slope(sub)} for (role, kind, site), sub in df.groupby(["role", "kind", "site"])]
    carrying_sites = carrying.site.tolist()
    transition = below.head(1).site.tolist()
    endpoint_site = ([s for s in selected if s in carrying_sites] or selected)[-1] if selected else None
    return {"summary": summary.to_dict("records"), "pooled_slopes": slopes, "by_axis": by_axis.to_dict("records"),
            "baseline_effects": baseline.to_dict("records"), "baseline_effect_mean": {f"{k[0]}|{k[1]}": float(v) for k, v in baseline.groupby(["role", "cont"]).effect_high_minus_low_prompt.mean().items()} if len(baseline) else {},
            "selected_sites_discovery": selected, "endpoint_site": endpoint_site, "transition_site": transition, "random_at_selected": {s: float(rand.get(s, np.nan)) for s in selected},
            "development_confirmation": summary[(summary.role == "development") & (summary.site.isin(selected))].to_dict("records")}, selected


def components_block(comp, loc):
    """Attention/MLP/per-head recoveries at the selected blocks, next to the full-block recovery."""
    cdf = pd.DataFrame([c for c in comp if c.get("recovery") is not None])
    if cdf.empty:
        return {}
    cdf["block"] = cdf.site.str.extract(r"(block_\d+)")
    cdf["part"] = cdf.site.str.replace(r"block_\d+_", "", regex=True)
    cdf["positive"] = cdf.recovery > 0
    # the prompt effect (denominator) is recoverable exactly from the stored ratio
    cdf = cdf[cdf.recovery.abs() > 1e-12].copy()
    cdf["den"] = cdf.delta_lp / cdf.recovery
    summary = cdf.groupby(["block", "part"]).agg(recovery_median=("recovery", "median"), recovery_mean=("recovery", "mean"), positive_share=("positive", "mean"), delta_lp_mean=("delta_lp", "mean"), n=("recovery", "size")).reset_index()
    slopes = [{"block": b, "part": p, **pooled_slope(sub, den=sub.den.to_numpy())} for (b, p), sub in cdf.groupby(["block", "part"])]
    full = {}
    if loc:
        for s in loc["summary"]:
            if s["kind"] == "persona_patch" and s["role"] == "discovery":
                full[s["site"]] = {"recovery_median": s["recovery_median"], "recovery_mean": s["recovery_mean"]}
    heads = summary[summary.part.str.startswith("head")].copy()
    head_rank = {b: sub.sort_values("recovery_median", ascending=False).head(4)[["part", "recovery_median", "positive_share"]].to_dict("records") for b, sub in heads.groupby("block")}
    head_sum = {b: float(sub.recovery_median.sum()) for b, sub in heads.groupby("block")}
    return {"summary": summary.to_dict("records"), "pooled_slopes": slopes, "full_block_discovery": full, "top_heads_by_block": head_rank, "sum_of_head_medians": head_sum,
            "note": "per-head = one head's o_proj input replaced at the persona span; attn/mlp = that sub-block's output replaced at the span; full block from localize_effects (discovery + development pooled here)"}


def readside_block(run, filename="readside_effects.jsonl"):
    """Read-side profile: path-patched direct reads of the persona span, one layer at a time (span kept at the low pole)."""
    eff = [e for e in read_jsonl(run / filename) if e.get("recovery") is not None]
    if not eff:
        return None
    df = pd.DataFrame(eff)
    df["positive"] = df.recovery > 0
    summary = df.groupby(["role", "site"]).agg(recovery_median=("recovery", "median"), recovery_mean=("recovery", "mean"), positive_share=("positive", "mean"), delta_lp_mean=("delta_lp", "mean"), n=("recovery", "size")).reset_index()
    slopes = [{"role": role, "site": site, **pooled_slope(sub)} for (role, site), sub in df.groupby(["role", "site"])]
    pooled = [{"site": site, **pooled_slope(sub)} for site, sub in df.groupby("site")]
    layer_rows = [p for p in pooled if p["site"].startswith("read_layer_") and "head" not in p["site"]]
    ranked = sorted(layer_rows, key=lambda p: -(p["slope"] or 0))
    top3 = [p["site"] for p in ranked[:3]]
    disc_ranked = sorted([s for s in slopes if s["role"] == "discovery" and s["site"].startswith("read_layer_") and "head" not in s["site"]], key=lambda p: -(p["slope"] or 0))
    return {"summary": summary.to_dict("records"), "pooled_slopes_by_role": slopes, "pooled_slopes_all": pooled,
            "top_layers_pooled": ranked[:5], "top_layers_discovery": disc_ranked[:5], "sum_single_layer_slopes": float(sum(p["slope"] or 0 for p in layer_rows)),
            "all_layers_downstream": next((p for p in pooled if p["site"] == "read_all_layers"), None), "suggested_head_layers": [int(s.split("_")[2]) for s in top3],
            "note": "read_layer_XX = downstream positions receive the donor's attention read of the span at layer XX (two-forward path patch); the span's own residual stays at the low pole; read_all_layers = all layers' downstream reads at once"}


def readside_heads_block(run):
    """Per-KV-head (sender) and per-query-head (receiver) direct reads at the chosen layers."""
    eff = [e for e in read_jsonl(run / "readside_head_effects.jsonl") if e.get("recovery") is not None]
    if not eff:
        return None
    df = pd.DataFrame(eff)
    df["positive"] = df.recovery > 0
    df["kind"] = np.where(df.site.str.contains("kvhead"), "kv_head", "query_head")
    summary = df.groupby(["layer", "site", "kind"]).agg(recovery_median=("recovery", "median"), recovery_mean=("recovery", "mean"), positive_share=("positive", "mean"), n=("recovery", "size")).reset_index()
    pooled = [{"layer": int(layer), "site": site, "kind": kind, **pooled_slope(sub)} for (layer, site, kind), sub in df.groupby(["layer", "site", "kind"])]
    by_layer = {}
    for layer in sorted(df.layer.unique()):
        rows = [p for p in pooled if p["layer"] == layer]
        q = sorted([p for p in rows if p["kind"] == "query_head"], key=lambda p: -(p["slope"] or 0))
        kv = sorted([p for p in rows if p["kind"] == "kv_head"], key=lambda p: -(p["slope"] or 0))
        by_layer[str(int(layer))] = {"query_heads_ranked": [{"site": p["site"], "slope": p["slope"], "ci95": p["ci95"]} for p in q], "sum_query_head_slopes": float(sum(p["slope"] or 0 for p in q)),
                                     "kv_heads": [{"site": p["site"], "slope": p["slope"], "ci95": p["ci95"]} for p in kv]}
    return {"summary": summary.to_dict("records"), "pooled_slopes": pooled, "by_layer": by_layer}


def endpoint_block(run, rows, grain_scores, texts_pilot):
    gens = read_jsonl(run / "endpoint_generations.jsonl")
    if not gens or not grain_scores:
        return None
    feats_pilot = {k: v["g"] for k, v in grain_scores["pilot"].items() if v.get("g")}
    cov, shrink, _n = within_condition_covariance(feats_pilot, rows)
    g = {i: v["g"] for i, v in grain_scores["endpoint"].items() if v.get("g")}
    df = pd.DataFrame(gens)
    df["key"] = df.pair + "|" + df.arm + "|s" + df.seed.astype(str)
    out = {"metric": "pilot within-condition Ledoit-Wolf", "shrinkage": shrink, "texts": len(gens), "grain_available": len(g), "per_pair": []}
    arms = ("intervention_high_into_low", "random_equal_norm", "reference_high_prompt")
    agg = defaultdict(list)
    for pair, sub in df.groupby("pair"):
        rec = {"pair": pair}
        for arm in arms:
            deltas = []
            for seed, s2 in sub.groupby("seed"):
                c = s2[s2.arm == "control_low"].key.tolist()
                a = s2[s2.arm == arm].key.tolist()
                if c and a and c[0] in g and a[0] in g:
                    deltas.append(np.asarray(g[a[0]]) - np.asarray(g[c[0]]))
            if deltas:
                d = np.mean(deltas, axis=0)
                rec[arm] = {"seeds": len(deltas), "delta_g": d.tolist(), "D2": mahalanobis2(d, cov), "D2_per_seed_median": float(np.median([mahalanobis2(x, cov) for x in deltas]))}
                agg[arm].append(d)
        out["per_pair"].append(rec)
    out["mean_over_pairs"] = {arm: {"pairs": len(v), "delta_g": np.mean(v, 0).tolist(), "D2": mahalanobis2(np.mean(v, 0), cov)} for arm, v in agg.items()}

    def compare(a, b):
        w = np.linalg.solve(cov, b)
        aa = a @ np.linalg.solve(cov, a)
        return {"cosine_in_metric": float((a @ w) / np.sqrt(aa * (b @ w))) if aa > 0 and (b @ w) > 0 else None, "projection_share": float((a @ w) / (b @ w)) if abs(b @ w) > 1e-12 else None}

    if "intervention_high_into_low" in agg and "reference_high_prompt" in agg:
        out["intervention_vs_reference"] = compare(np.mean(agg["intervention_high_into_low"], 0), np.mean(agg["reference_high_prompt"], 0))
        out["random_vs_reference"] = compare(np.mean(agg["random_equal_norm"], 0), np.mean(agg["reference_high_prompt"], 0)) if "random_equal_norm" in agg else None
    # per axis: the same quantities, plus paired per-pair D2 comparisons
    per_axis = {}
    for axis in AXES:
        recs = [r for r in out["per_pair"] if r["pair"].startswith(axis + "|") and all(arm in r for arm in arms)]
        if not recs:
            continue
        means = {arm: np.mean([np.asarray(r[arm]["delta_g"]) for r in recs], 0) for arm in arms}
        d2 = {arm: [r[arm]["D2"] for r in recs] for arm in arms}
        diff = np.asarray(d2["intervention_high_into_low"]) - np.asarray(d2["random_equal_norm"])
        rng = np.random.default_rng(7)
        boots = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(2000)]
        per_axis[axis] = {"pairs": len(recs), "mean_delta_g": {arm: means[arm].tolist() for arm in arms}, "D2_of_mean": {arm: mahalanobis2(means[arm], cov) for arm in arms},
                          "per_pair_D2_median": {arm: float(np.median(d2[arm])) for arm in arms},
                          "intervention_minus_random_D2": {"mean": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "positive_pairs": int((diff > 0).sum())},
                          "intervention_vs_reference": compare(means["intervention_high_into_low"], means["reference_high_prompt"]),
                          "random_vs_reference": compare(means["random_equal_norm"], means["reference_high_prompt"])}
    out["per_axis"] = per_axis
    return out


def endpoint_readers(run, rows, roles, texts):
    """Blinded readers on regenerated texts: local text-only readout means per arm and a
    TF-IDF probe trained on pilot discovery+development texts (per axis)."""
    gens = read_jsonl(run / "endpoint_generations.jsonl")
    local_path = run / "endpoint_local_query.json"
    local = json.loads(local_path.read_text())["raw"] if local_path.exists() else {}
    out = {}
    for axis in AXES:
        train = [r for r in rows if r["axis"] == axis and r["situation"] in roles["discovery"] + roles["development"] and r["text_id"] in texts]
        sub = [g for g in gens if g["axis"] == axis]
        if not sub:
            continue
        keys = [f"{g['pair']}|{g['arm']}|s{g['seed']}" for g in sub]
        probs = tfidf_probe([texts[r["text_id"]] for r in train], [r["pole"] for r in train], [g["text"] for g in sub]) if len(train) >= 4 else [None] * len(sub)
        by_arm = defaultdict(lambda: {"tfidf_p_high": [], "local_query": []})
        for g, k, p in zip(sub, keys, probs):
            if p is not None:
                by_arm[g["arm"]]["tfidf_p_high"].append(float(p))
            if k in local and axis in local[k]:
                by_arm[g["arm"]]["local_query"].append(float(local[k][axis]))
        out[axis] = {arm: {m: {"n": len(v), "mean": float(np.mean(v)) if v else None, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None} for m, v in d.items()} for arm, d in by_arm.items()}
        # paired per (pair, seed): arm minus control_low, sign-flip null over the paired differences
        vals = {}
        for g, k, p in zip(sub, keys, probs):
            vals[(g["pair"], g["seed"], g["arm"])] = {"tfidf_p_high": float(p) if p is not None else None, "local_query": float(local[k][axis]) if k in local and axis in local[k] else None}
        paired = {}
        for arm in ("intervention_high_into_low", "random_equal_norm", "reference_high_prompt"):
            for m in ("tfidf_p_high", "local_query"):
                diffs = [vals[(pr, sd, arm)][m] - vals[(pr, sd, "control_low")][m] for (pr, sd, ar) in vals if ar == arm and (pr, sd, "control_low") in vals and vals[(pr, sd, arm)][m] is not None and vals[(pr, sd, "control_low")][m] is not None]
                if diffs:
                    d = np.asarray(diffs)
                    rng = np.random.default_rng(11)
                    null = [float(np.mean(d * rng.choice([-1, 1], size=len(d)))) for _ in range(4000)]
                    paired[f"{arm}|{m}"] = {"n": len(d), "mean_diff": float(d.mean()), "positive_share": float((d > 0).mean()), "p_signflip": float((np.sum(np.abs(null) >= abs(d.mean())) + 1) / (len(null) + 1))}
        out[axis]["paired_minus_control"] = paired
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--select-sites", action="store_true", help="write selected_sites.json from discovery pairs")
    parser.add_argument("--exclude-refusals", action="store_true", help="sensitivity variant: drop texts whose audit flagged an AI disclaimer; writes to analysis_no_refusal/")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    roles = manifest["split_roles"]
    rows = json.loads((run / "materials.json").read_text())
    gens = read_jsonl(run / "generations.jsonl")
    n_refusal = sum(1 for g in gens if g["audit"].get("refusal"))
    if args.exclude_refusals:
        gens = [g for g in gens if not g["audit"].get("refusal")]
    texts = {g["text_id"]: g["text"] for g in gens}
    audits = pd.DataFrame([{"condition": g["condition"], **g["audit"], "n_new": g["n_new"]} for g in gens])
    audit_table = audits.groupby("condition").agg(n=("words", "size"), words=("words", "mean"), n_new=("n_new", "mean"), truncated=("truncated", "mean"), leak_share=("leak_hits", lambda x: float((x > 0).mean())), leak_hits=("leak_hits", "mean"), refusal=("refusal", "mean"), topic_hits=("topic_hits", "mean"), first_person=("first_person_share", "mean")).reset_index() if len(audits) else pd.DataFrame()
    grain = json.loads((run / "readout_grain.json").read_text()) if (run / "readout_grain.json").exists() else {"grain": {}, "grain_stripped": {}}
    local = json.loads((run / "readout_local_query.json").read_text()) if (run / "readout_local_query.json").exists() else {}
    judge = {j["text_id"]: j["rating"] for j in read_jsonl(run / "judge_deepseek.jsonl") if "rating" in j}
    out = run / ("analysis_no_refusal" if args.exclude_refusals else "analysis")
    out.mkdir(exist_ok=True)
    write(out / "text_selection.json", {"texts_used": len(texts), "refusal_flagged_total": n_refusal, "excluded": bool(args.exclude_refusals), "note": "primary analysis keeps every generation; this variant only documents sensitivity to AI-disclaimer texts"})
    audit_table.to_csv(out / "audit_by_condition.csv", index=False)
    readers = reader_block([r for r in rows if r["text_id"] in texts], roles, texts, grain, local, judge) if texts else {}
    write(out / "readers.json", readers)
    write(out / "control_positions.json", control_positions(rows, local, judge, grain))
    write(out / "displacement.json", displacement_block([r for r in rows if r["text_id"] in texts], grain, texts, roles) if texts else {})
    loc, selected = localization_block(run, roles)
    if loc:
        write(out / "localization.json", loc)
        if args.select_sites and selected:
            write(run / "selected_sites.json", {"sites": selected, "endpoint_site": loc["endpoint_site"], "transition_site": loc["transition_site"], "selection_basis": "discovery pairs only: deepest two sites with median persona-patch recovery >= .5 plus first site below .5; development reported separately", "kind": "persona_patch recovery median"})
    comp = read_jsonl(run / "component_effects.jsonl")
    if comp:
        write(out / "components.json", components_block(comp, loc))
    rs = readside_block(run)
    if rs:
        write(out / "readside.json", rs)
    rh = readside_heads_block(run)
    if rh:
        write(out / "readside_heads.json", rh)
    ep_scores_path = run / "endpoint_grain.json"
    if ep_scores_path.exists():
        ep = json.loads(ep_scores_path.read_text())
        block = endpoint_block(run, [r for r in rows if r["text_id"] in texts], {"pilot": grain["grain"], "endpoint": ep}, texts)
        block["readers_on_endpoint_texts"] = endpoint_readers(run, rows, roles, texts)
        write(out / "endpoint.json", block)
    print(json.dumps({"texts": len(texts), "audit_conditions": len(audit_table), "readers_axes": list(readers), "selected_sites": selected}, indent=1))


if __name__ == "__main__":
    main()
