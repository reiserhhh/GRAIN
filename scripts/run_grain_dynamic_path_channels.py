#!/usr/bin/env python3
"""Finite cached-state experiment linking GRAIN motion, queries and criteria."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.dynamic_channels import (
    clustered_links,
    path_descriptors,
    residualize_descriptive,
    shapley_mse,
)
from grain_research.vector_mechanisms.external_material import masked_comment
from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.factor_path_routes import (
    SEED,
    fold_diagnostics,
    metrics,
    nested_predictions,
    paired_delta,
)
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle

PRIOR = BASE / "factor_path_routes_v1/source_r2"
SOURCE = BASE / "all_factor_source_m29_v1"
MATERIAL = BASE / "source_remainder_s9_r1/private_material.json"
VIEWS = BASE / "vm05_qualification_r2/common_views.json"
FROZEN = ROOT / "results/v7_operator_boundary_audit/e0_full_20260714"
PLAN = ROOT / "docs/GRAIN_DYNAMIC_PATH_CHANNEL_PLAN_V1.md"
FACTORS = [
    f"{op}:F{i:02d}"
    for op in ("whole", "native", "fixed_128_cross")
    for i in range(1, 5)
]
BLOCK_FACTORS = FACTORS[:8]
PARTS = ["mean", "configuration", "final", "score"]
CONTEXT = [
    "log_words",
    "word_cv",
    "question_rate",
    "exclamation_rate",
    "digit_rate",
    "community_entropy",
    "community_switch",
    "log_span_days",
    "log_median_gap_days",
]


def context_features(comments, texts):
    """Observed opportunity summaries; no label, identity or community-name coding."""
    times = np.array([c["timestamp"] for c in comments], float)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Chronological timestamps required")
    words = np.array([len(t.split()) for t in texts], float)
    chars = max(sum(map(len, texts)), 1)
    subs = [str(c["subreddit"]) for c in comments]
    _, counts = np.unique(subs, return_counts=True)
    p = counts / counts.sum()
    return np.array(
        [
            np.log1p(words.mean()),
            words.std() / max(words.mean(), 1),
            sum(t.count("?") for t in texts) / chars,
            sum(t.count("!") for t in texts) / chars,
            sum(sum(c.isdigit() for c in t) for t in texts) / chars,
            -np.sum(p * np.log(p)),
            np.mean(np.array(subs[:-1]) != subs[1:]),
            np.log1p((times[-1] - times[0]) / 86400),
            np.log1p(np.median(np.diff(times)) / 86400),
        ]
    )


def prepare(out):
    """Restore exact matched metadata and score frozen coordinates without labels."""
    if out.exists():
        raise ValueError("New output directory required; use analyze to resume")
    source_manifest = read(SOURCE / "manifest.json")
    if sha(SOURCE / "private_design.json") != source_manifest["input_sha256"]:
        raise ValueError("Cached query design changed")
    if sha(MATERIAL) != read(MATERIAL.parent / "manifest.json")["material_sha256"]:
        raise ValueError("Cached text material changed")
    previous_manifest = read(PRIOR / "manifest.json")
    for rel, expected in previous_manifest["input_sha256"].items():
        # The prior script/code can evolve, but all data and frozen factors must match.
        if rel.startswith("results/") and sha(ROOT / rel) != expected:
            raise ValueError(f"Prior evidence changed: {rel}")
    authors = pd.read_csv(PRIOR / "private_factor_authors.csv")
    ids = set(authors.author_id)
    if len(ids) != 276 or len(authors) != 276:
        raise ValueError("Expected corrected 276-author panel")
    raw = read(VIEWS)
    selected = {r["author_id"]: r for r in raw if r["author_id"] in ids}
    del raw
    if set(selected) != ids:
        raise ValueError("Missing upstream author")
    material = {r["author_id"]: r for r in read(MATERIAL) if r["author_id"] in ids}
    rows, keys, contexts, timeline = [], [], [], []
    for uid in sorted(ids):
        bundle = selected[uid]
        if len(bundle["views"]) != 3:
            raise ValueError("Three views required")
        all_comments = [c for view in bundle["views"] for c in view]
        if (
            len({c["comment_id"] for c in all_comments}) != 36
            or len({c["thread_id"] for c in all_comments}) != 36
        ):
            raise ValueError("Expected source-disjoint comments and discussions")
        times = np.array([c["timestamp"] for c in all_comments])
        if np.any(np.diff(times) < 0):
            raise ValueError("Reconstructed source order changed")
        timeline.append(
            {
                "author_id": uid,
                "days_span": (times[-1] - times[0]) / 86400,
                "median_gap_days": np.median(np.diff(times)) / 86400,
            }
        )
        for v, comments in enumerate(bundle["views"]):
            texts = [
                masked_comment(c["body"], c["comment_id"])[0]["cue_mask"]
                for c in comments
            ]
            if len(texts) != 12 or texts != material[uid]["views"][v]["texts"]:
                raise ValueError("Exact source-to-cache text match failed")
            contexts.append(
                dict(
                    author_id=uid,
                    view=v,
                    **dict(
                        zip(CONTEXT, context_features(comments, texts), strict=True)
                    ),
                )
            )
            for block in range(4):
                key = f"{uid}::v{v}::b{block}"
                keys.append(
                    {"user_id": key, "author_id": uid, "view": v, "block": block}
                )
                for c, text in zip(
                    comments[3 * block : 3 * block + 3],
                    texts[3 * block : 3 * block + 3],
                    strict=True,
                ):
                    rows.append(
                        {
                            "author": key,
                            "body": text,
                            "created_utc": c["timestamp"],
                            "subreddit": c["subreddit"],
                        }
                    )
    frame = canonicalize_comments(
        pd.DataFrame(rows),
        order_col="created_utc",
        condition_col="subreddit",
        min_tokens=12,
    )
    frame["split"] = "confirmation"
    runtime = joblib.load(
        FROZEN / "artifacts/common_source_comment_representation.joblib"
    )
    scores, support = pd.DataFrame(keys), []
    for op in ("whole", "native"):
        bundle = read_factor_bundle(
            FROZEN / f"operator_bundles/{op}_factor_bundle.json"
        )
        units, _, values = score_operator_from_comments(
            frame,
            spec=ObservationSpec(**bundle.operator),
            representation=runtime,
            bundle=bundle,
        )
        if len(values) != len(keys) or not values.score_status.eq("SCORED").all():
            raise ValueError(f"Unsupported blocks: {op}")
        cols = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
        scores = scores.merge(
            values[["user_id", *cols]].rename(
                columns={c: f"{op}:F{i:02d}" for i, c in enumerate(cols, 1)}
            ),
            on="user_id",
            validate="one_to_one",
        )
        support.append(
            {
                "operator": op,
                "blocks": len(values),
                "units": len(units),
                "min_units": int(values.n_units.min()),
                "median_tokens": float(values.n_tokens.median()),
            }
        )
    out.mkdir(parents=True)
    scores.to_csv(out / "private_factor_blocks.csv", index=False)
    pd.DataFrame(contexts).to_csv(out / "private_context_views.csv", index=False)
    pd.DataFrame(timeline).to_csv(out / "private_timeline.csv", index=False)
    input_files = [
        VIEWS,
        MATERIAL,
        PRIOR / "private_factor_views.csv",
        PRIOR / "private_factor_authors.csv",
        SOURCE / "private_design.json",
        SOURCE / "private_external_links.csv",
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/dynamic_channels.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_DYNAMIC_PATH_CHANNEL_V1",
            "seed": SEED,
            "authors": 276,
            "views_per_author": 3,
            "comments_per_view": 12,
            "blocks_per_view": 4,
            "comments_per_block": 3,
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in input_files},
            "new_model_calls": 0,
            "meps_fitted": False,
            "source_labels_previously_exposed": True,
        },
    )
    t = pd.DataFrame(timeline)
    write(
        out / "restoration_audit.json",
        {
            "authors": 276,
            "views": 828,
            "matched_comments": len(rows),
            "exact_text_match": True,
            "chronological": True,
            "threads_per_author": 36,
            "span_days_median": float(t.days_span.median()),
            "median_gap_days_median": float(t.median_gap_days.median()),
            "operator_support": support,
            "scope": "sampled comments across different discussions; not conversational transitions",
        },
    )
    print("Restored metadata and frozen block trajectories", flush=True)


def derive_features(out):
    """Create fixed dynamic descriptors and decode cached per-view query functions."""
    blocks = pd.read_csv(out / "private_factor_blocks.csv")
    rng, rows = np.random.default_rng(SEED), []
    for (uid, view), g in blocks.groupby(["author_id", "view"], sort=True):
        x = g.sort_values("block")[BLOCK_FACTORS].to_numpy(float)
        sd, order = path_descriptors(x)
        _, shuffled = path_descriptors(x[rng.permutation(4)])
        names = (
            [f"drift::{c}" for c in BLOCK_FACTORS]
            + [f"roughness::{c}" for c in BLOCK_FACTORS]
            + [
                f"area::{op}:F{j + 1:02d}:F{k + 1:02d}"
                for op in ("whole", "native")
                for j in range(4)
                for k in range(j + 1, 4)
            ]
        )
        row = {"author_id": uid, "view": view}
        row.update(
            {f"disp::{k}": float(v) for k, v in zip(BLOCK_FACTORS, sd, strict=True)}
        )
        row.update(dict(zip(names, order, strict=True)))
        row.update(
            {"shuffle::" + k: float(v) for k, v in zip(names, shuffled, strict=True)}
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "private_dynamic_views.csv", index=False)
    manifest = read(SOURCE / "manifest.json")
    functions_file = Path(manifest["functions"]) / "nine_query_functions.npz"
    if sha(functions_file) != manifest["functions_sha256"]:
        raise ValueError("Query functions changed")
    z = np.load(functions_file)
    ids = set(blocks.author_id)
    rows, packet_hashes = [], {}
    for n, d in enumerate(read(SOURCE / "private_design.json")):
        if d["author_id"] not in ids:
            continue
        q = d["query"]
        f = FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
        p = SOURCE / f"private_packets/unit_{d['index']:04d}.npz"
        packet_hashes[p.name] = sha(p)
        packet_meta = read(p.with_suffix(".json"))
        if (
            packet_hashes[p.name] != packet_meta["array_sha256"]
            or packet_meta["author_id"] != d["author_id"]
            or packet_meta["query"] != q
        ):
            raise ValueError("Packet identity/hash mismatch")
        with np.load(p) as values:
            for view in range(3):
                rows.append(
                    dict(
                        author_id=d["author_id"],
                        query=q,
                        view=view,
                        **f.parts(values["P"][view], values["Q"][view]),
                    )
                )
        if n % 250 == 0:
            print(f"Cached query packets {n}", flush=True)
    z.close()
    paths = pd.DataFrame(rows)
    cached = pd.read_csv(SOURCE / "private_external_links.csv")
    check = (
        paths.groupby(["author_id", "query"])[PARTS]
        .mean()
        .join(
            cached.set_index(["author_id", "query"])[PARTS], rsuffix="_old", how="inner"
        )
    )
    error = max(float((check[c] - check[c + "_old"]).abs().max()) for c in PARTS)
    if error > 1e-7:
        raise ValueError(f"Query cache reconstruction differs: {error}")
    paths.to_csv(out / "private_path_views.csv", index=False)
    write(
        out / "packet_integrity.json",
        {
            "packets": packet_hashes,
            "function_sha256": sha(functions_file),
            "max_reconstruction_error": error,
        },
    )


def links(out):
    """Compute author-cluster temporal links and distinct real-label associations."""
    grains = pd.read_csv(PRIOR / "private_factor_views.csv")
    paths = pd.read_csv(out / "private_path_views.csv")
    context = pd.read_csv(out / "private_context_views.csv")
    dynamic = pd.read_csv(out / "private_dynamic_views.csv")
    labels = pd.read_csv(SOURCE / "private_external_links.csv")
    output, criterion, alignment, stability = [], [], [], []
    for query, pp in paths.groupby("query", sort=True):
        ids = sorted(set(pp.author_id))
        index = pd.MultiIndex.from_product([ids, range(3)], names=["author_id", "view"])
        gv = (
            grains.set_index(["author_id", "view"])
            .loc[index, FACTORS]
            .to_numpy()
            .reshape(len(ids), 3, 12)
        )
        pv = (
            pp.set_index(["author_id", "view"])
            .loc[index, PARTS]
            .to_numpy()
            .reshape(len(ids), 3, 4)
        )
        cv = (
            context.set_index(["author_id", "view"])
            .loc[index, CONTEXT]
            .to_numpy()
            .reshape(len(ids), 3, len(CONTEXT))
        )
        dg, dp, dc = (np.diff(v, axis=1) for v in (gv, pv, cv))
        z = np.c_[dc.reshape(-1, len(CONTEXT)), np.tile([0.0, 1.0], len(ids))]
        adjusted_g = residualize_descriptive(dg.reshape(-1, 12), z).reshape(dg.shape)
        for k, part in enumerate(PARTS):
            for mode in ("raw", "context_adjusted"):
                x, y = dg, dp[:, :, k]
                if mode == "context_adjusted":
                    x = adjusted_g
                    y = residualize_descriptive(y.ravel(), z).reshape(y.shape)
                r, lo, hi, p = clustered_links(
                    x, y, permutation_draws=4999 if part == "score" else 499
                )
                for j, factor in enumerate(FACTORS):
                    output.append(
                        {
                            "query": query,
                            "part": part,
                            "adjustment": mode,
                            "factor": factor,
                            "authors": len(ids),
                            "transitions": len(ids) * 2,
                            "r": r[j],
                            "lower": lo[j],
                            "upper": hi[j],
                            "p": p[j],
                        }
                    )
        yy = (
            labels.query("query == @query")
            .set_index("author_id")
            .loc[ids, "y_true"]
            .to_numpy()
        )
        dv = dynamic.groupby("author_id").mean(numeric_only=True).loc[ids]
        raw_links = [
            r
            for r in output
            if r["query"] == query and r["part"] == "score" and r["adjustment"] == "raw"
        ]
        real_links = []
        for j, factor in enumerate(FACTORS):
            a = association(gv[:, :, j].mean(1), yy, binary=query.endswith("_cont"))
            real_links.append(abs(a["pearson_r"]))
            criterion.append(dict(query=query, feature=factor, family="static", **a))
        for col in dv:
            if col.startswith(("disp::", "drift::", "roughness::", "area::")):
                criterion.append(
                    dict(
                        query=query,
                        feature=col,
                        family="dispersion" if col.startswith("disp::") else "order",
                        **association(dv[col], yy, binary=query.endswith("_cont")),
                    )
                )
        alignment.append(
            {
                "query": query,
                "coordinates": 12,
                "descriptive_rank_rho": float(
                    spearmanr([abs(r["r"]) for r in raw_links], real_links).statistic
                ),
                "scope": "dependent coordinate comparison, not an independent 12-subject significance test",
            }
        )
    table = pd.DataFrame(output)
    table["q"] = np.nan
    for _, g in table.groupby(["part", "adjustment"]):
        table.loc[g.index, "q"] = bh_adjust(g.p.to_numpy())
    table.to_csv(out / "cochange_links.csv", index=False)
    table = pd.DataFrame(criterion)
    table["q"] = np.nan
    for _, g in table.groupby("family"):
        table.loc[g.index, "q"] = bh_adjust(g.p.to_numpy())
    table.to_csv(out / "criterion_links.csv", index=False)
    pd.DataFrame(alignment).to_csv(out / "path_criterion_alignment.csv", index=False)
    group = (
        grains[["author_id", "source_group"]]
        .drop_duplicates()
        .set_index("author_id")
        .source_group
    )
    for source, ids in group.groupby(group).groups.items():
        for col in dynamic:
            if col.startswith(("disp::", "drift::", "roughness::", "area::")):
                pivot = dynamic[dynamic.author_id.isin(ids)].pivot(
                    index="author_id", columns="view", values=col
                )
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    stability.append(
                        dict(
                            source_group=source,
                            feature=col,
                            view_a=a,
                            view_b=b,
                            **association(pivot[a], pivot[b]),
                        )
                    )
    pd.DataFrame(stability).to_csv(out / "dynamic_stability.csv", index=False)
    print(
        "All-nine co-change, real-criterion and period stability tables complete",
        flush=True,
    )


def predict(out):
    """Nested source-only fits and exact held-author predictive gap allocation."""
    labels = pd.read_csv(SOURCE / "private_external_links.csv")
    grain = pd.read_csv(PRIOR / "private_factor_authors.csv").set_index("author_id")
    dynamics = (
        pd.read_csv(out / "private_dynamic_views.csv")
        .groupby("author_id")
        .mean(numeric_only=True)
    )
    context = (
        pd.read_csv(out / "private_context_views.csv")
        .groupby("author_id")[CONTEXT]
        .mean()
    )
    ordercols = [
        c for c in dynamics if c.startswith(("drift::", "roughness::", "area::"))
    ]
    dispcols = [c for c in dynamics if c.startswith("disp::")]
    results, predictions, comparisons, accounting, controls, params = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for query, frame in labels.groupby("query", sort=True):
        frame = frame[frame.author_id.isin(grain.index)].sort_values("author_id")
        ids, y = frame.author_id.to_numpy(), frame.y_true.to_numpy(float)
        base = grain.loc[ids, FACTORS].to_numpy(float)
        levelcols = [c for c in dynamics if c.startswith("level::")]
        if levelcols:
            base = np.column_stack([base, dynamics.loc[ids, levelcols]])
        blocks = [
            frame[PARTS[:3]].to_numpy(float),
            context.loc[ids].to_numpy(float),
            dynamics.loc[ids, dispcols + ordercols].to_numpy(float),
        ]
        arms = {
            f"subset_{m}": np.column_stack(
                [base] + [b for j, b in enumerate(blocks) if m & (1 << j)]
            )
            for m in range(8)
        }
        arms["dispersion_only_addition"] = np.column_stack(
            [base, dynamics.loc[ids, dispcols]]
        )
        arms["shuffled_order_addition"] = np.column_stack(
            [base, dynamics.loc[ids, dispcols + ["shuffle::" + c for c in ordercols]]]
        )
        pp, losses = {}, {}
        for arm, x in arms.items():
            pred, folds, chosen = nested_predictions(x, y, ids)
            pp[arm] = pred
            if arm.startswith("subset_"):
                losses[int(arm.split("_")[1])] = float(np.mean((y - pred) ** 2))
            results.append(
                dict(
                    query=query,
                    arm=arm,
                    dimensions=x.shape[1],
                    **metrics(y, pred, binary=query.endswith("_cont")),
                    **fold_diagnostics(y, pred, folds, binary=query.endswith("_cont")),
                )
            )
            predictions.extend(
                {
                    "author_id": uid,
                    "query": query,
                    "arm": arm,
                    "fold": int(fold),
                    "y_true": float(yy),
                    "prediction": float(pr),
                }
                for uid, fold, yy, pr in zip(ids, folds, y, pred, strict=True)
            )
            params.append({"query": query, "arm": arm, "folds": chosen})
        for a, b in [
            ("subset_1", "subset_0"),
            ("subset_2", "subset_0"),
            ("subset_4", "subset_0"),
            ("subset_7", "subset_0"),
            ("subset_7", "subset_1"),
            ("subset_4", "dispersion_only_addition"),
            ("subset_4", "shuffled_order_addition"),
        ]:
            comparisons.append(
                dict(
                    query=query,
                    arm=a,
                    reference=b,
                    **paired_delta(
                        y, pp[a], pp[b], binary=query.endswith("_cont"), draws=499
                    ),
                )
            )
        allocated = shapley_mse(losses)
        for j, name in enumerate(("query_path", "context", "dynamic_and_dispersion")):
            accounting.append(
                {
                    "query": query,
                    "block": name,
                    "mse_gain": float(allocated[j]),
                    "baseline_mse": losses[0],
                    "full_mse": losses[7],
                    "total_gain": losses[0] - losses[7],
                    "accounting_error": float(
                        allocated.sum() - (losses[0] - losses[7])
                    ),
                }
            )
        for seed in (SEED + 100, SEED + 101):
            yy = np.random.default_rng(seed).permutation(y)
            for arm in ("subset_0", "subset_7"):
                pred, folds, _ = nested_predictions(arms[arm], yy, ids)
                controls.append(
                    dict(
                        query=query,
                        arm=arm,
                        permutation_seed=seed,
                        **metrics(yy, pred, binary=query.endswith("_cont")),
                        **fold_diagnostics(
                            yy, pred, folds, binary=query.endswith("_cont")
                        ),
                    )
                )
        print(f"Held-author channel matrix complete: {query} n={len(ids)}", flush=True)
        for name, rows in [
            ("prediction_metrics", results),
            ("private_predictions", predictions),
            ("paired_deltas", comparisons),
            ("gap_accounting", accounting),
            ("label_permutation", controls),
        ]:
            pd.DataFrame(rows).to_csv(out / (name + ".csv"), index=False)
        write(out / "model_selection.json", params)


def channel_audit(out):
    """Use already-authorized external numeric caches, never fit on MEPS."""
    meta_path = BASE / "vm06_meps_preflight_r3/private_view_metadata.csv"
    score_path = BASE / "all_factor_inventory_m27_v1/private_meps_scores.csv"
    meta = pd.read_csv(meta_path)
    scores = pd.read_csv(score_path)
    if (
        meta.duplicated(["participant_code", "condition", "view_kind"]).any()
        or scores.duplicated(["participant_code", "scale"]).any()
    ):
        raise ValueError("Duplicate external metadata/criterion rows")
    inventory, links = [], []
    for (kind, condition), g in meta.groupby(["view_kind", "condition"]):
        inventory.append(
            {
                "view_kind": kind,
                "condition": condition,
                "participants": g.participant_code.nunique(),
                "user_turns_total": int(g.user_turns.sum()),
                "assistant_turns_total": int(g.assistant_turns.sum()),
                "median_user_chars": float(g.user_chars.median()),
                "role_order_preserved": int(g.source_order_preserved.sum()),
            }
        )
        for feature in (
            "user_chars",
            "user_turns",
            "assistant_chars",
            "assistant_turns",
        ):
            for scale, s in scores.groupby("scale"):
                joined = g[["participant_code", feature]].merge(
                    s, on="participant_code", validate="one_to_one"
                )
                if len(joined) >= 15 and joined[feature].nunique() > 1:
                    links.append(
                        dict(
                            view_kind=kind,
                            condition=condition,
                            feature=feature,
                            scale=scale,
                            **association(np.log1p(joined[feature]), joined.score),
                        )
                    )
    pd.DataFrame(inventory).to_csv(out / "meps_channel_inventory.csv", index=False)
    ll = pd.DataFrame(links)
    ll["q"] = bh_adjust(ll.p.fillna(1).to_numpy())
    ll.to_csv(out / "meps_participation_links.csv", index=False)
    write(
        out / "behavior_identifiability.json",
        {
            "participants": int(meta.participant_code.nunique()),
            "sources": {
                str(p.relative_to(ROOT)): sha(p) for p in (meta_path, score_path)
            },
            "observed_channels": [
                "user utterances",
                "AI utterances",
                "participation and text-volume metadata",
                "pre/post self-report scales",
            ],
            "independent_action_outcome_in_analyzed_sources": False,
            "new_meps_model_fit": False,
            "raw_surveys_or_analysis_master_opened": False,
            "identified_gap": "held-author prediction error against recorded self-report, and descriptive differences between text settings",
            "unidentified_gap": "latent true personality vs reporting bias vs everyday enacted action",
            "counterexample": "Given observed text T and questionnaire Y, both B=Y and B=independent_noise generate the same (T,Y) distribution when behavior B is unobserved.",
            "minimum_addition": [
                "same-author independent action criterion with explicit timing and opportunities",
                "predeclared behavioral coding independent of model predictions",
                "repeated or multi-method observations if separating stable trait and measurement error",
            ],
        },
    )


def main():
    """Run a finite cache-only analysis or resume after input preparation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=BASE / "dynamic_path_channels_v1"
    )
    parser.add_argument(
        "--stage", choices=("all", "prepare", "analyze", "predict"), default="all"
    )
    args = parser.parse_args()
    started = time.time()
    with threadpool_limits(limits=1):
        if args.stage in ("all", "prepare"):
            prepare(args.output)
        if args.stage in ("all", "analyze"):
            derive_features(args.output)
            links(args.output)
        if args.stage in ("all", "analyze", "predict"):
            predict(args.output)
            channel_audit(args.output)
            write(
                args.output / "status.json",
                {
                    "status": "COMPLETE_BOUNDED_DYNAMIC_CHANNEL_ANALYSIS",
                    "elapsed_seconds": time.time() - started,
                    "new_model_calls": 0,
                    "questions_with_unobserved_action_identified": False,
                },
            )
    print(
        json.dumps({"stage": args.stage, "seconds": time.time() - started}), flush=True
    )


if __name__ == "__main__":
    main()
