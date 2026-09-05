#!/usr/bin/env python3
"""Audit and compare discovery-only multi-comment representations and support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.comparison import (
    conditional_rank,
    describe_representation,
    node_bootstrap_weights,
    pair_concordance,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.multicomment import linear_cka_dual, view_layout
from scripts.run_grain_vector_mechanism_controls import write_json


def fast_null(values, draws=499, seed=919):
    """Permute author correspondence using cached cosines, not re-encoding data."""
    values = np.asarray(values, dtype=np.float64)
    v = values / np.maximum(np.linalg.norm(values, axis=2, keepdims=True), 1e-12)
    n = len(v)
    pairs = ((0, 1), (0, 2), (1, 2))
    sims = [v[:, a] @ v[:, b].T for a, b in pairs]
    mask = ~np.eye(n, dtype=bool)

    def rank(s):
        d = np.diag(s)[:, None] - s
        return np.mean(((d > 1e-10) + 0.5 * (abs(d) <= 1e-10))[mask])

    point = np.mean([rank(s) for s in sims])
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(draws):
        perm = [np.arange(n), rng.permutation(n), rng.permutation(n)]
        null.append(
            np.mean(
                [
                    rank(s[np.ix_(perm[a], perm[b])])
                    for s, (a, b) in zip(sims, pairs, strict=True)
                ]
            )
        )
    return {
        "permutation_p": float(
            (1 + np.sum(np.asarray(null) >= point - 1e-12)) / (draws + 1)
        ),
        "null_mean": float(np.mean(null)),
        "permutations": draws,
    }


def matched_masks(bundles, layout):
    """Match impostors to positive-pair source overlap and target-view length."""
    sources, lengths = [], []
    for b in bundles:
        rows = [c for v in b["views"] for c in v]
        groups = view_layout(b, layout)
        sources.append([{rows[i]["subreddit"] for i in g} for g in groups])
        lengths.append([sum(rows[i]["word_count"] for i in g) for g in groups])
    n = len(bundles)
    masks = []
    zero_overlap = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        m = np.zeros((n, n), dtype=bool)
        z = m.copy()
        for i in range(n):
            left, right = sources[i][a], sources[i][b]
            own_j = len(left & right) / max(1, len(left | right))
            for j in range(n):
                if i == j:
                    continue
                other = sources[j][b]
                other_j = len(left & other) / max(1, len(left | other))
                ratio = lengths[j][b] / lengths[i][b]
                m[i, j] = abs(own_j - other_j) <= 0.10 and 1 / 1.25 <= ratio <= 1.25
                z[i, j] = (
                    not (left & right)
                    and not (left & other)
                    and 1 / 1.25 <= ratio <= 1.25
                )
        masks.append(m)
        zero_overlap.append(z)
    return {
        "source_overlap_length_matched": np.stack(masks),
        "no_shared_source_positive": np.stack(zero_overlap),
    }


def main():
    """Do not report unsupported geometry as a usable measurement or test labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--parent-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.views.read_text(encoding="utf-8"))
    if any(b["split"] != "discovery" for b in source):
        raise ValueError("Discovery summaries must not open other splits")
    refs = np.array([b["discovery_role"] == "reference" for b in source])
    bundles = [b for b, ref in zip(source, refs, strict=True) if not ref]
    features, support, parents = {}, {}, []
    for parent in a.parent_dirs:
        manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["source_view_sha256"] != material_hash(source)
            or manifest["authors"] != [b["author_id"] for b in source]
            or manifest["reference_mask"] != refs.tolist()
        ):
            raise ValueError("Parent evidence or ordering mismatch")
        if (
            json.loads((parent / "run_status.json").read_text(encoding="utf-8"))[
                "status"
            ]
            != "COMPLETE"
        ):
            raise ValueError("Incomplete parent")
        parents.append(manifest["identity"])
        with np.load(parent / "representations.npz") as archive:
            for k in archive.files:
                if k in features:
                    raise ValueError("Repeated representation")
                features[k] = archive[k][~refs]
        if (parent / "support_status.npz").exists():
            with np.load(parent / "support_status.npz") as archive:
                for k in archive.files:
                    support[k] = (
                        np.all(archive[k][~refs] == "GEOMETRY_PROFILE_READY", axis=1)
                        if "UNRESTRICTED" not in str(archive[k].flat[0])
                        else np.ones(len(bundles), dtype=bool)
                    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new empty summary directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    write_json(
        a.output_dir / "manifest.json",
        {
            "source_view_sha256": material_hash(source),
            "parents": parents,
            "code_sha256": code_fingerprint(ROOT),
            "labels_used": False,
            "estimand": "same-author three-view rank, not personality prediction",
        },
    )
    metrics, sensitive, boots = [], [], {}
    weights = node_bootstrap_weights(len(bundles))
    masks = {l: matched_masks(bundles, l) for l in ("chronological", "interleaved")}
    no_clues = np.array(
        [
            not any(any(c["leakage_flags"].values()) for v in b["views"] for c in v)
            for b in bundles
        ]
    )
    for key, v in features.items():
        ready = support.get(key, np.ones(len(bundles), dtype=bool))
        if ready.sum() < 3:
            metrics.append(
                {
                    "method": key,
                    "status": "REFUSED_FROZEN_SUPPORT",
                    "authors": int(ready.sum()),
                    "total_comparison_authors": len(bundles),
                }
            )
            continue
        w = weights if ready.all() else node_bootstrap_weights(int(ready.sum()))
        result, boot = describe_representation(v[ready], w)
        metrics.append(
            {"method": key, "status": "COMPLETE", **result, **fast_null(v[ready])}
        )
        if ready.all():
            boots[key] = boot
        clean = ready & no_clues
        if clean.sum() >= 3:
            result, _ = describe_representation(
                v[clean], node_bootstrap_weights(int(clean.sum()))
            )
            sensitive.append(
                {"method": key, "condition": "no_personality_clue_authors", **result}
            )
        c = pair_concordance(v)
        layout = key.split("__")[-1]
        for condition, mask in masks[layout].items():
            mask = mask & ready[None, :, None] & ready[None, None, :]
            valid_authors = np.count_nonzero(mask.any(axis=(0, 2)))
            if valid_authors < 3:
                sensitive.append(
                    {
                        "method": key,
                        "condition": condition,
                        "status": "INSUFFICIENT",
                        "authors": valid_authors,
                    }
                )
                continue
            point = conditional_rank(c, mask, np.ones(len(bundles)))[0]
            b = conditional_rank(c, mask, weights)
            finite = b[np.isfinite(b)]
            sensitive.append(
                {
                    "method": key,
                    "condition": condition,
                    "author_rank_auc": point,
                    "ci_low": np.quantile(finite, 0.025),
                    "ci_high": np.quantile(finite, 0.975),
                    "authors": valid_authors,
                    "directed_edges": int(mask.sum()),
                    "empty_bootstraps": int(np.sum(~np.isfinite(b))),
                }
            )
    pd.DataFrame(metrics).to_csv(a.output_dir / "comparison.csv", index=False)
    pd.DataFrame(sensitive).to_csv(a.output_dir / "sensitivity.csv", index=False)
    pairs = []
    for key, bootstrap in boots.items():
        if "__interleaved" in key:
            other = key.replace("__interleaved", "__chronological")
        elif "_joint_local__" in key:
            other = key.replace("_joint_local__", "_independent_mean__")
        else:
            continue
        if other in boots:
            d = bootstrap - boots[other]
            point = next(
                r["author_rank_auc"] for r in metrics if r["method"] == key
            ) - next(r["author_rank_auc"] for r in metrics if r["method"] == other)
            pairs.append(
                {
                    "candidate": key,
                    "baseline": other,
                    "delta": point,
                    "ci_low": np.quantile(d, 0.025),
                    "ci_high": np.quantile(d, 0.975),
                }
            )
    for candidate in (
        "qwen_block_10_history_mean__chronological",
        "qwen_block_16_decision_behavior__chronological",
    ):
        for baseline in (
            "bge_large_independent_mean__chronological",
            "e5_large_independent_mean__chronological",
            "tfidf_word__chronological",
            "tfidf_char__chronological",
            "grain_unit_mean__chronological",
        ):
            if candidate in boots and baseline in boots:
                d = boots[candidate] - boots[baseline]
                point = next(
                    r["author_rank_auc"] for r in metrics if r["method"] == candidate
                ) - next(
                    r["author_rank_auc"] for r in metrics if r["method"] == baseline
                )
                pairs.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "delta": point,
                        "ci_low": np.quantile(d, 0.025),
                        "ci_high": np.quantile(d, 0.975),
                    }
                )
    pd.DataFrame(pairs).to_csv(a.output_dir / "paired_differences.csv", index=False)
    similarity = []
    names = [
        name
        for name in features
        if name.endswith("__chronological")
        and (
            "_independent_mean__" in name
            or "_joint_local__" in name
            or name
            in (
                "qwen_block_10_history_mean__chronological",
                "qwen_block_16_decision_behavior__chronological",
                "tfidf_word__chronological",
                "tfidf_char__chronological",
                "grain_unit_mean__chronological",
            )
        )
    ]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            similarity.append(
                {
                    "left": left,
                    "right": right,
                    "authors": len(bundles),
                    "linear_cka": linear_cka_dual(
                        features[left].mean(1), features[right].mean(1)
                    ),
                    "scope": "Descriptive biased linear CKA; author mean over 3 views; not validity or equivalence",
                }
            )
    pd.DataFrame(similarity).to_csv(
        a.output_dir / "representation_similarity.csv", index=False
    )
    write_json(
        a.output_dir / "run_status.json",
        {
            "status": "COMPLETE",
            "methods": len(metrics),
            "reference_authors": int(refs.sum()),
            "comparison_authors": len(bundles),
            "no_clue_comparison_authors": int(no_clues.sum()),
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
