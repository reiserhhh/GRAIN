#!/usr/bin/env python3
"""Within-utterance companion: fixed lexical slices, no elapsed-time assumption."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import itertools

from grain_research.vector_mechanisms.dynamic_channels import (
    clustered_links,
    path_descriptors,
)
from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.run_grain_dynamic_path_channels import (
    FACTORS,
    FROZEN,
    MATERIAL,
    PLAN,
    PRIOR,
    SEED,
    SOURCE,
    predict,
)
from suica_core.suica import TOKEN_RE
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle


def lexical_slices(text):
    """Preserve all original characters while making four equal token-count spans."""
    spans = list(TOKEN_RE.finditer(text))
    if len(spans) < 48:
        return []
    edges = (
        [0] + [spans[int(i * len(spans) / 4)].start() for i in (1, 2, 3)] + [len(text)]
    )
    parts = [text[a:b] for a, b in itertools.pairwise(edges)]
    if "".join(parts) != text or min(len(TOKEN_RE.findall(p)) for p in parts) < 12:
        raise ValueError("Lossless lexical slicing failed")
    return parts


def prepare(out):
    """Project each lexical slice into frozen whole coordinates, without labels."""
    if out.exists():
        raise ValueError("Choose a new result directory")
    ids = set(pd.read_csv(PRIOR / "private_factor_authors.csv").author_id)
    raw, meta, coverage = [], [], []
    for r in read(MATERIAL):
        uid = r["author_id"]
        if uid not in ids:
            continue
        for view, v in enumerate(r["views"]):
            eligible = 0
            for comment, text in enumerate(v["texts"]):
                parts = lexical_slices(text)
                if not parts:
                    continue
                eligible += 1
                for index, part in enumerate(parts):
                    key = f"{uid}::v{view}::c{comment}::s{index}"
                    raw.append({"author": key, "body": part})
                    meta.append(
                        {
                            "user_id": key,
                            "author_id": uid,
                            "view": view,
                            "comment": comment,
                            "slice": index,
                        }
                    )
            coverage.append(
                {
                    "author_id": uid,
                    "view": view,
                    "eligible_comments": eligible,
                    "input_comments": len(v["texts"]),
                }
            )
    cc = pd.DataFrame(coverage)
    if cc.eligible_comments.min() < 2:
        raise ValueError(
            "Insufficient within-comment support; no silent author dropping"
        )
    frame = canonicalize_comments(
        pd.DataFrame(raw), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    bundle = read_factor_bundle(FROZEN / "operator_bundles/whole_factor_bundle.json")
    runtime = joblib.load(
        FROZEN / "artifacts/common_source_comment_representation.joblib"
    )
    _, _, scores = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=runtime,
        bundle=bundle,
    )
    if len(scores) != len(meta) or not scores.score_status.eq("SCORED").all():
        raise ValueError("Every selected slice must score")
    cols = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
    scored = pd.DataFrame(meta).merge(
        scores[["user_id", *cols]], on="user_id", validate="one_to_one"
    )
    rng, features = np.random.default_rng(SEED), []
    names = (
        [f"drift::{k}" for k in FACTORS[:4]]
        + [f"roughness::{k}" for k in FACTORS[:4]]
        + [
            f"area::whole:F{j + 1:02d}:F{k + 1:02d}"
            for j in range(4)
            for k in range(j + 1, 4)
        ]
    )
    for (uid, view, comment), g in scored.groupby(
        ["author_id", "view", "comment"], sort=True
    ):
        x = g.sort_values("slice")[cols].to_numpy(float)
        sd, order = path_descriptors(x, groups=((0, 1, 2, 3),))
        _, shuffled = path_descriptors(x[rng.permutation(4)], groups=((0, 1, 2, 3),))
        row = {"author_id": uid, "view": view, "comment": comment}
        row.update(
            {
                "level::" + k: float(v)
                for k, v in zip(FACTORS[:4], x.mean(0), strict=True)
            }
        )
        row.update(
            {"disp::" + k: float(v) for k, v in zip(FACTORS[:4], sd, strict=True)}
        )
        row.update(dict(zip(names, order, strict=True)))
        row.update(
            {"shuffle::" + k: float(v) for k, v in zip(names, shuffled, strict=True)}
        )
        features.append(row)
    out.mkdir(parents=True)
    scored.to_csv(out / "private_factor_slices.csv", index=False)
    pd.DataFrame(features).to_csv(out / "private_comment_descriptors.csv", index=False)
    view_features = (
        pd.DataFrame(features)
        .drop(columns="comment")
        .groupby(["author_id", "view"])
        .mean()
        .reset_index()
    )
    view_features.to_csv(out / "private_dynamic_views.csv", index=False)
    cc.to_csv(out / "private_coverage.csv", index=False)
    parent = out.parent
    context = pd.read_csv(parent / "private_context_views.csv")
    context.to_csv(out / "private_context_views.csv", index=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_WITHIN_COMMENT_DYNAMIC_V1",
            "authors": len(ids),
            "eligible_comments": int(cc.eligible_comments.sum()),
            "input_comments": int(cc.input_comments.sum()),
            "eligible_min_per_view": int(cc.eligible_comments.min()),
            "slices": len(scored),
            "lexical_tokenizer": "suica_core.suica.TOKEN_RE; not transformer tokens",
            "lossless_character_spans": True,
            "frozen_factors": 4,
            "new_model_calls": 0,
            "new_factor_fit": False,
            "baseline_includes_new_slice_means": True,
            "sources": {
                str(p.relative_to(ROOT)): sha(p)
                for p in [
                    MATERIAL,
                    PLAN,
                    Path(__file__),
                    FROZEN / "operator_bundles/whole_factor_bundle.json",
                    PRIOR / "private_factor_authors.csv",
                ]
            },
        },
    )
    print("Within-comment slice trajectories ready", flush=True)


def relate(out):
    """Relate within-utterance motion to query changes and source criteria separately."""
    dynamic = pd.read_csv(out / "private_dynamic_views.csv")
    path = pd.read_csv(out.parent / "private_path_views.csv")
    labels = pd.read_csv(SOURCE / "private_external_links.csv")
    cols = [
        c
        for c in dynamic
        if c.startswith(("drift::", "roughness::", "area::", "disp::"))
    ]
    links, criterion, stability = [], [], []
    for q, p in path.groupby("query", sort=True):
        ids = sorted(set(p.author_id))
        index = pd.MultiIndex.from_product([ids, range(3)], names=["author_id", "view"])
        d = (
            dynamic.set_index(["author_id", "view"])
            .loc[index, cols]
            .to_numpy()
            .reshape(len(ids), 3, len(cols))
        )
        ypath = (
            p.set_index(["author_id", "view"])
            .loc[index, "score"]
            .to_numpy()
            .reshape(len(ids), 3)
        )
        r, lo, hi, pval = clustered_links(
            np.diff(d, axis=1), np.diff(ypath, axis=1), permutation_draws=4999
        )
        y = (
            labels[labels["query"] == q]
            .set_index("author_id")
            .loc[ids, "y_true"]
            .to_numpy()
        )
        for j, c in enumerate(cols):
            links.append(
                {
                    "query": q,
                    "feature": c,
                    "authors": len(ids),
                    "r": r[j],
                    "lower": lo[j],
                    "upper": hi[j],
                    "p": pval[j],
                }
            )
            criterion.append(
                {
                    "query": q,
                    "feature": c,
                    **association(d[:, :, j].mean(1), y, binary=q.endswith("_cont")),
                }
            )
            if q in ("TF_cont", "neuroticism"):
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    stability.append(
                        {
                            "source_group": "mbti_only"
                            if q.endswith("_cont")
                            else "big5_only",
                            "feature": c,
                            "view_a": a,
                            "view_b": b,
                            **association(d[:, a, j], d[:, b, j]),
                        }
                    )
    for name, rows in [
        ("cochange_links", links),
        ("criterion_links", criterion),
        ("dynamic_stability", stability),
    ]:
        f = pd.DataFrame(rows)
        f["q"] = bh_adjust(f.p.to_numpy())
        f.to_csv(out / (name + ".csv"), index=False)


def main():
    """Execute the separate within-utterance branch, with no LLM forwards."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", type=Path, default=BASE / "dynamic_path_channels_v1/within_comment"
    )
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    start = time.time()
    with threadpool_limits(limits=1):
        if not args.resume:
            prepare(args.output)
        relate(args.output)
        predict(args.output)
    write(
        args.output / "status.json",
        {
            "status": "COMPLETE_WITHIN_COMMENT_CHANNELS",
            "seconds": time.time() - start,
            "new_model_calls": 0,
        },
    )
    print("Within-comment analysis complete", flush=True)


if __name__ == "__main__":
    main()
