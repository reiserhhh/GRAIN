#!/usr/bin/env python3
"""Bounded I1/I3 local predecessor comparison; no pretrained network fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.predecessors import (
    FACETS,
    TRAITS,
    MeaningExtraction,
    keyed_proportions,
    match_items,
    mean_pearson,
    reference_percentiles,
)
from grain_research.vector_mechanisms.external_heads import metrics
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write

METHODS = (
    "word_ridge",
    "mem_ridge",
    "semantic_pca_ridge",
    "grain_ridge",
    "item_counts_ridge",
    "semantic_items",
    "grain_items",
    "grain_semantic_items",
)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def read_material(material):
    """Read only a checksum-bound, complete source construction package."""
    m = json.loads((material / "material_manifest.json").read_text())
    for name, checksum in m["files"].items():
        if digest(material / name) != checksum:
            raise ValueError("Input material changed")
    state = json.loads((material / "encoding_status.json").read_text())
    if state["status"] != "COMPLETE":
        raise ValueError("Sentence encoding not complete")
    contract = material / "sentence_batches/contract.json"
    if digest(contract) != state["contract_sha256"]:
        raise ValueError("Sentence contract changed")
    vectors, next_start = [], 0
    for path in sorted((material / "sentence_batches").glob("batch_*.json")):
        info = json.loads(path.read_text())
        if (
            info["start"] != next_start
            or digest(path.with_suffix(".npz")) != info["sha256"]
        ):
            raise ValueError("Batch gap or changed sentence vectors")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            v = z["vectors"]
        if len(v) != info["stop"] - info["start"]:
            raise ValueError("Malformed sentence batch")
        vectors.append(v)
        next_start = info["stop"]
    if next_start != state["total"]:
        raise ValueError("Incomplete sentence array")
    with np.load(material / "private_arrays.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    data = json.loads((material / "private_material.json").read_text())
    items = json.loads((material / "ipip_neo300.json").read_text())["items"]
    return m, arrays, data, items, np.concatenate(vectors)


def evidence_features(data, items, vectors, n_authors, threshold):
    """Produce raw keyed counts, half-counts and source-addressed matched evidence."""
    q, h = vectors[: len(items)], vectors[len(items) :]
    sentences = data["sentences"]
    if len(h) != len(sentences):
        raise ValueError("Candidate/vector row mismatch")
    counts = np.zeros((n_authors, len(FACETS) + len(TRAITS), 2))
    half = np.zeros((2, *counts.shape))
    audit = []
    by_author = [[] for _ in range(n_authors)]
    for i, s in enumerate(sentences):
        by_author[s["author_index"]].append(i)
    for u, indices in enumerate(by_author):
        local = h[indices]
        c, best, score, accept = match_items(local, q, items, threshold=threshold)
        counts[u] = c
        for parity in range(2):
            ii = [i for i in indices if sentences[i]["comment_index"] % 2 == parity]
            half[parity, u] = match_items(h[ii], q, items, threshold=threshold)[0]
        # Keep candidate identities, margins and source offsets, not just trait totals.
        for i, j, similarity, accepted in zip(
            indices, best, score, accept, strict=True
        ):
            item = items[int(j)]
            audit.append(
                {
                    k: sentences[i][k]
                    for k in (
                        "author_index",
                        "comment_index",
                        "comment_id",
                        "start",
                        "end",
                    )
                }
                | {
                    "item_id": item["id"],
                    "facet": item["facet"],
                    "trait": item["trait"],
                    "key": item["key"],
                    "similarity": float(similarity),
                    "accepted": bool(accepted),
                }
            )
    return counts, half, pd.DataFrame(audit)


class ReferenceFeatures:
    """Fit all corpus-dependent operations inside the current training partition."""

    def __init__(self, method):
        self.method = method
        self.transforms = {}

    def fit_transform(self, sources, indices):
        """Learn representations from this inner or outer training split only."""
        if self.method == "word_ridge":
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=10000, sublinear_tf=True
            )
            return self.vectorizer.fit_transform(sources["documents"][indices])
        if self.method == "mem_ridge":
            self.mem = MeaningExtraction().fit(sources["documents"][indices])
            x = self.mem.transform(sources["documents"][indices])
        else:
            blocks = []
            if "semantic" in self.method:
                x = sources["semantic"][indices]
                pca = PCA(
                    n_components=min(64, len(x) - 1, x.shape[1]),
                    svd_solver="randomized",
                    random_state=731,
                )
                blocks.append(pca.fit_transform(x))
                self.transforms["semantic"] = pca
            if "grain" in self.method:
                blocks.append(sources["grain"][indices])
            if "item" in self.method:
                counts = sources["items"][indices]
                normalized = counts / np.maximum(counts.sum(1, keepdims=True), 1)
                for label, x in (("items", counts), ("items_normalized", normalized)):
                    pca = PCA(
                        n_components=min(10, len(x) - 1, x.shape[1]), svd_solver="full"
                    )
                    blocks.append(pca.fit_transform(x))
                    self.transforms[label] = pca
            if not blocks:
                raise ValueError("Unknown representation method")
            x = np.concatenate(blocks, axis=1)
        self.scaler = StandardScaler().fit(x)
        return self.scaler.transform(x)

    def transform(self, sources, indices):
        """Apply frozen transformations without test-set centering or selection."""
        if self.method == "word_ridge":
            return self.vectorizer.transform(sources["documents"][indices])
        if self.method == "mem_ridge":
            x = self.mem.transform(sources["documents"][indices])
        else:
            blocks = []
            if "semantic" in self.method:
                blocks.append(
                    self.transforms["semantic"].transform(sources["semantic"][indices])
                )
            if "grain" in self.method:
                blocks.append(sources["grain"][indices])
            if "item" in self.method:
                counts = sources["items"][indices]
                normalized = counts / np.maximum(counts.sum(1, keepdims=True), 1)
                for label, x in (("items", counts), ("items_normalized", normalized)):
                    blocks.append(self.transforms[label].transform(x))
            x = np.concatenate(blocks, axis=1)
        return self.scaler.transform(x)


def fit_fold(sources, y, train, test, method):
    """Select Ridge alpha by 3-fold inner author CV, refitting preprocessing."""
    inner = {alpha: np.full(y[train].shape, np.nan) for alpha in ALPHAS}
    for ii, jj in KFold(3, shuffle=True, random_state=731).split(train):
        feature = ReferenceFeatures(method)
        x = feature.fit_transform(sources, train[ii])
        z = feature.transform(sources, train[jj])
        for alpha in ALPHAS:
            inner[alpha][jj] = (
                Ridge(alpha=alpha, solver="lsqr").fit(x, y[train[ii]]).predict(z)
            )
    scores = {alpha: mean_pearson(y[train], pred) for alpha, pred in inner.items()}
    finite = [alpha for alpha in ALPHAS if np.isfinite(scores[alpha])]
    if not finite:
        raise ValueError("All inner CV scores undefined")
    chosen = max(finite, key=lambda alpha: scores[alpha])
    feature = ReferenceFeatures(method)
    x = feature.fit_transform(sources, train)
    head = Ridge(alpha=chosen, solver="lsqr").fit(x, y[train])
    return (
        feature,
        head,
        head.predict(feature.transform(sources, test)),
        {"alpha": chosen, "inner_mean_r": scores[chosen], "inner_scores": scores},
    )


def paired_delta(y, a, b, draws=1000):
    """Paired author bootstrap, conditional on fitted OOF scores; not refit CI."""
    rng = np.random.default_rng(731)
    values = []
    for _ in range(draws):
        ix = rng.integers(0, len(y), len(y))
        values.append(mean_pearson(y[ix], a[ix]) - mean_pearson(y[ix], b[ix]))
    finite = np.asarray(values)[np.isfinite(values)]
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {
        "delta_mean_r": mean_pearson(y, a) - mean_pearson(y, b),
        "ci_low": lo,
        "ci_high": hi,
        "draws": draws,
        "valid_draws": len(finite),
        "scope": "paired_author_conditional_oof_no_refit",
    }


def run(args):
    """Run a declared finite method table and archive numeric, not diagnostic claims."""
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    _manifest, a, data, items, vectors = read_material(args.material)
    n, y, folds = len(a["author_ids"]), a["y"], a["folds"]
    if len(set(a["author_ids"])) != n or set(folds) != set(range(5)):
        raise ValueError("Unique source authors and five official folds required")
    contract = {
        "material_manifest_sha256": digest(args.material / "material_manifest.json"),
        "encoding_contract_sha256": digest(
            args.material / "sentence_batches/contract.json"
        ),
        "methods": list(METHODS),
        "alphas": ALPHAS,
        "threshold": args.threshold,
        "source": "big5_only",
        "authors": n,
        "outer": "official_5fold_repeat0",
        "inner": "KFold3_seed731",
        "bootstrap": {"draws": args.bootstrap, "seed": 731, "refit": False},
        "shuffle_seed": 913,
        "code": {
            p.name: digest(p)
            for p in [Path(__file__), ROOT / "grain_research/predecessors.py"]
        },
        "scope": "local_adapters_not_original_paper_replication",
        "meps_used": False,
        "bridge_used": False,
    }
    # JSON round-trip makes tuple/list comparisons stable on resume.
    contract = json.loads(json.dumps(contract))
    old = out / "run_contract.json"
    if old.exists() and json.loads(old.read_text()) != contract:
        raise ValueError("Changed experiment contract: use a new output directory")
    write(old, contract)
    counts, half_counts, evidence = evidence_features(
        data, items, vectors, n, args.threshold
    )
    evidence.to_csv(out / "private_evidence.csv", index=False)
    np.savez_compressed(
        out / "private_item_counts.npz", counts=counts, half_counts=half_counts
    )
    sources = {
        "documents": np.asarray(data["documents"]),
        "semantic": a["comments"].mean(1),
        "grain": a["grain"],
        "items": counts.reshape(n, -1),
    }
    half_sources = [
        {
            "documents": np.asarray(data[k]),
            "semantic": a["comments"][:, parity::2].mean(1),
            "items": half_counts[parity].reshape(n, -1),
        }
        for parity, k in enumerate(("half_a", "half_b"))
    ]
    predictions, records, selections, reliability, loadings = {}, [], [], [], []
    started = time.monotonic()
    with threadpool_limits(limits=4):
        for method in METHODS:
            p = np.full_like(y, np.nan)
            halves = [np.full_like(y, np.nan), np.full_like(y, np.nan)]
            for fold in range(5):
                train, test = (
                    np.flatnonzero(folds != fold),
                    np.flatnonzero(folds == fold),
                )
                target = out / f"private_{method}_fold{fold}.joblib"
                if target.exists():
                    saved = joblib.load(target)
                    feature, head, selection = (
                        saved["feature"],
                        saved["head"],
                        saved["selection"],
                    )
                    p[test] = head.predict(feature.transform(sources, test))
                    np.testing.assert_allclose(
                        p[test], saved["predictions"], rtol=1e-8, atol=1e-8
                    )
                else:
                    feature, head, p[test], selection = fit_fold(
                        sources, y, train, test, method
                    )
                    joblib.dump(
                        {
                            "feature": feature,
                            "head": head,
                            "selection": selection,
                            "predictions": p[test],
                        },
                        target,
                    )
                selections.append(
                    {
                        "method": method,
                        "fold": fold,
                        "alpha": selection["alpha"],
                        "inner_mean_r": selection["inner_mean_r"],
                    }
                )
                if "grain" not in method:
                    for parity in range(2):
                        halves[parity][test] = head.predict(
                            feature.transform(half_sources[parity], test)
                        )
                if method == "mem_ridge":
                    names = feature.mem.vectorizer_.get_feature_names_out()
                    for k, column in enumerate(feature.mem.loadings_.T):
                        for idx in np.argsort(-abs(column))[:8]:
                            loadings.append(
                                {
                                    "fold": fold,
                                    "factor_code": f"MEM_{k + 1:02d}",
                                    "term": names[idx],
                                    "loading": column[idx],
                                }
                            )
                print(
                    {
                        "method": method,
                        "fold": fold,
                        "alpha": selection["alpha"],
                        "seconds": round(time.monotonic() - started, 1),
                    },
                    flush=True,
                )
            predictions[method] = p
            records.extend({"method": method, **row} for row in metrics(y, p, "big5"))
            if "grain" not in method:
                for j, trait in enumerate(TRAITS):
                    r = (
                        pearsonr(halves[0][:, j], halves[1][:, j]).statistic
                        if min(np.std(halves[0][:, j]), np.std(halves[1][:, j])) > 1e-12
                        else np.nan
                    )
                    reliability.append(
                        {
                            "method": method,
                            "trait": trait,
                            "half_r": r,
                            "scope": "alternate_comments_frozen_outer_model_not_longitudinal_retest",
                        }
                    )
        # One diagnostic label shuffle is a sanity check, not a permutation p-value.
        shuffled = y[np.random.default_rng(913).permutation(n)]
        null = np.full_like(y, np.nan)
        for fold in range(5):
            tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            _, _, null[te], _ = fit_fold(
                sources, shuffled, tr, te, "semantic_pca_ridge"
            )
        records.extend(
            {"method": "semantic_label_shuffle", **row}
            for row in metrics(shuffled, null, "big5")
        )
    # Fixed-reference unsupervised scores are separate from calibrated regression.
    ratio = keyed_proportions(counts[:, len(FACETS) :])
    percentiles = np.full_like(y, np.nan)
    for fold in range(5):
        tr, te = folds != fold, folds == fold
        percentiles[te] = reference_percentiles(ratio[tr], ratio[te])
    coverage = []
    for j, trait in enumerate(TRAITS):
        valid = np.isfinite(percentiles[:, j])
        r = (
            pearsonr(y[valid, j], percentiles[valid, j]).statistic
            if valid.sum() > 3 and np.std(percentiles[valid, j]) > 1e-12
            else np.nan
        )
        coverage.append(
            {
                "trait": trait,
                "authors": n,
                "scorable": int(valid.sum()),
                "coverage": valid.mean(),
                "unsupervised_percentile_r": r,
                "positive_matches": int(counts[:, len(FACETS) + j, 0].sum()),
                "negative_matches": int(counts[:, len(FACETS) + j, 1].sum()),
            }
        )
    pd.DataFrame(records).to_csv(out / "metrics_by_trait.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selection.csv", index=False)
    pd.DataFrame(reliability).to_csv(out / "split_half.csv", index=False)
    pd.DataFrame(loadings).to_csv(out / "mem_loadings.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "evidence_coverage.csv", index=False)
    np.savez_compressed(
        out / "private_oof.npz",
        y=y,
        folds=folds,
        author_ids=a["author_ids"],
        item_percentiles=percentiles,
        **predictions,
    )
    comparisons = []
    for one, other in [
        ("semantic_pca_ridge", "grain_ridge"),
        ("semantic_items", "semantic_pca_ridge"),
        ("grain_items", "grain_ridge"),
        ("grain_semantic_items", "semantic_pca_ridge"),
        ("mem_ridge", "word_ridge"),
    ]:
        comparisons.append(
            {
                "method": one,
                "reference": other,
                **paired_delta(y, predictions[one], predictions[other], args.bootstrap),
            }
        )
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    summary = (
        pd.DataFrame(records)
        .groupby("method", sort=False)[["pearson", "spearman", "mae", "rmse"]]
        .agg(lambda column: column.mean(skipna=False))
        .reset_index()
    )
    summary.to_csv(out / "comparison.csv", index=False)
    write(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "authors": n,
            "outer_folds": 5,
            "methods": len(METHODS),
            "model_training": "CPU_Ridge_only_no_neural_finetuning",
            "seconds": time.monotonic() - started,
            "scope": contract["scope"],
        },
    )
    print(summary.to_string(index=False), flush=True)


def main():
    """Run or replay the one frozen local-adapter comparison table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument(
        "--output", type=Path, default=BASE / "predecessor_integration_v1/comparison_r1"
    )
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    if not -1 <= args.threshold <= 1 or args.bootstrap < 20:
        p.error("Invalid threshold/bootstrap count")
    run(args)


if __name__ == "__main__":
    main()
