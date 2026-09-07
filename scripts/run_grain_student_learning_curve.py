#!/usr/bin/env python3
"""S4: teacher-only sample-size curves on exposed D1 with outer-author validation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.external_material import unpack_features
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_functional_basis import author_folds
from scripts.fit_grain_response_student import raw_blocks
from scripts.run_grain_real_author_path import BASE, key, read, sha, write


def fit_model(x, y, alpha):
    """Fit training-only scales and at most64 PCA coordinates before teacher Ridge."""
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = (x - center) / scale
    pca = PCA(
        n_components=min(64, len(x) - 1, x.shape[1]),
        svd_solver="randomized",
        random_state=731,
    )
    coordinates = pca.fit_transform(z)
    return center, scale, pca, Ridge(alpha=alpha).fit(coordinates, y)


def predict(x, model):
    """Predict from fixed training geometry, without target access or batch adaptation."""
    center, scale, pca, head = model
    return head.predict(pca.transform((x - center) / scale))


def select_alpha(x, target, ids, salt):
    """Select using only teacher scores in inner-author validation."""
    folds = author_folds(ids, 3, salt)
    losses = {alpha: [] for alpha in (1.0, 10.0, 100.0)}
    for fold in range(3):
        train, val = folds != fold, folds == fold
        center, scale, pca, _ = fit_model(x[train], target[train], 1.0)
        z = pca.transform((x[train] - center) / scale)
        zv = pca.transform((x[val] - center) / scale)
        for alpha, errors in losses.items():
            head = Ridge(alpha=alpha).fit(z, target[train])
            errors.append(float(np.mean((head.predict(zv) - target[val]) ** 2)))
    return min(losses, key=lambda a: (np.mean(losses[a]), a))


def feature_cache(connection, panel):
    """Rebuild cue-mask means/query pools from verified caches without labels."""
    manifest = read(connection / "manifest.json")
    summary = read(connection / "summary.json")
    if sha(connection / "manifest.json") != summary["manifest_sha256"]:
        raise ValueError("Connection provenance changed")
    if (
        sha(panel / "private_encoder.npz")
        != read(panel / "manifest.json")["encoder_sha256"]
    ):
        raise ValueError("Query anchors changed")
    with np.load(panel / "private_encoder.npz", allow_pickle=False) as z:
        poles = z["poles"]
    if (
        sha(BASE / "d1_author_discovery_r1/private_predictions.csv")
        != manifest["source_sha256"]
    ):
        raise ValueError("Source teacher/outcome packet changed")
    hashes = read(connection / "private_cache_hashes.json")
    ids, vectors = [], []
    for name, digest in hashes.items():
        path = ROOT / name
        if sha(path) != digest:
            raise ValueError("Encoder cache changed")
        metadata = read(path.with_suffix(".json"))
        with np.load(path, allow_pickle=False) as z:
            h = unpack_features(dict(z), metadata["feature_aliases"])[
                "cue_mask_comment_vectors"
            ]
        ids.append(path.stem)
        vectors.append(h)
    membership = read(BASE / "d1_author_discovery_r1/membership_snapshot.json")
    if not set(ids) <= set(membership["private_author_ids"]):
        raise ValueError("Only prior D1 source authors may be used")
    output = {}
    for q, query in enumerate(("TF_cont", "neuroticism")):
        matrix = {}
        for family in ("mean", "query"):
            spec = {"family": family, "poles": poles}
            raw = raw_blocks(
                [v for author in vectors for v in author],
                np.full(len(ids) * 3, q),
                spec,
            )
            pieces = [value.reshape(len(ids), 3, -1).mean(1) for value in raw.values()]
            matrix[family] = np.column_stack(pieces)
        output[query] = matrix
    return np.asarray(ids), output


def main():
    """Compare four sample sizes; freeze OOF predictions before joining outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private learning curves belong in ignored results")
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S4_TEACHER_SAMPLE_CURVE_V1",
            "script_sha256": sha(__file__),
            "sizes": [24, 96, 384, "all"],
            "families": ["mean", "query"],
            "pca_max_rank": 64,
            "alphas": [1, 10, 100],
            "fit_target": "cached native cue-mask teacher score only",
            "selection": "inner3 author folds; outer5 authors; same outer tests across sizes",
            "scope": "exposed D1 discovery material, not new external confirmation",
            "time_cap_seconds": 600,
            "new_neural_forwards": 0,
            "connection_predictions_sha256": sha(
                args.connection / "private_predictions.csv"
            ),
        },
    )
    start = time.monotonic()
    ids, features = feature_cache(args.connection, args.panel)
    source = args.connection / "private_predictions.csv"
    targets = pd.read_csv(
        source, dtype={"author_id": str}, usecols=["author_id", "query", "native_query"]
    )
    rows, selections = [], []
    with threadpool_limits(limits=2):
        for query, feature in features.items():
            native = targets[targets["query"] == query].set_index("author_id")
            indices = np.array([j for j, uid in enumerate(ids) if uid in native.index])
            names = ids[indices]
            y = native.loc[names, "native_query"].to_numpy()
            folds = author_folds(names, 5, "s4_outer")
            for family, full in feature.items():
                x = full[indices]
                for size in (24, 96, 384, "all"):
                    if (
                        size != "all"
                        and min(np.sum(folds != f) for f in range(5)) < size
                    ):
                        continue
                    for fold in range(5):
                        train, test = (
                            np.flatnonzero(folds != fold),
                            np.flatnonzero(folds == fold),
                        )
                        train = np.asarray(
                            sorted(train, key=lambda j: key(names[j], f"s4_{fold}"))
                        )
                        if size != "all":
                            train = train[:size]
                        alpha = select_alpha(
                            x[train], y[train], names[train], f"s4_inner_{fold}"
                        )
                        prediction = predict(
                            x[test], fit_model(x[train], y[train], alpha)
                        )
                        selections.append(
                            {
                                "query": query,
                                "family": family,
                                "size": str(size),
                                "fold": fold,
                                "train_authors": len(train),
                                "alpha": alpha,
                            }
                        )
                        rows.extend(
                            {
                                "author_id": names[j],
                                "query": query,
                                "family": family,
                                "size": str(size),
                                "fold": fold,
                                "teacher": float(y[j]),
                                "prediction": float(p),
                                "baseline": float(y[train].mean()),
                            }
                            for j, p in zip(test, prediction, strict=True)
                        )
                        if time.monotonic() - start > 600:
                            pd.DataFrame(rows).to_csv(
                                out / "private_partial.csv", index=False
                            )
                            raise TimeoutError(
                                "Teacher-only learning curve exceeded ten-minute cap"
                            )
                    print(f"S4 {query}/{family}/{size} complete", flush=True)
    frame = pd.DataFrame(rows)
    frozen = out / "private_oof_before_outcomes.csv"
    frame.to_csv(frozen, index=False)
    write(
        out / "oof_frozen.json",
        {"sha256": sha(frozen), "selection_outcomes_used": False},
    )
    outcomes = pd.read_csv(
        source, dtype={"author_id": str}, usecols=["author_id", "query", "y_true"]
    )
    frame = frame.merge(outcomes, on=["author_id", "query"], validate="many_to_one")
    metrics = []
    for (query, family, size), group in frame.groupby(["query", "family", "size"]):
        fidelity = scalar_metrics(group.teacher, group.prediction)
        external = scalar_metrics(group.y_true, group.prediction, query == "TF_cont")
        mse = np.mean((group.prediction - group.teacher) ** 2)
        zero = np.mean((group.baseline - group.teacher) ** 2)
        metrics.append(
            {
                "query": query,
                "family": family,
                "size": size,
                "authors": len(group),
                "teacher_r": fidelity["pearson_r"],
                "teacher_mse_ratio": float(mse / zero),
                "external_r": external["pearson_r"],
                "external_p": external["pearson_p"],
                "external_auc": external.get("auc"),
            }
        )
    table = pd.DataFrame(metrics)
    table.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selected_models.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "seconds": time.monotonic() - start,
            "new_neural_forwards": 0,
            "questionnaire_labels_used_for_fitting": False,
            "outcomes_read_after_oof": True,
            "bridge_or_meps_or_lockbox_read": False,
            "metrics": metrics,
            "oof_sha256": sha(frozen),
        },
    )
    report = "# S4 teacher-only sample-size ladder\n\n"
    report += "PCA/Ridge selected by teacher MSE only, same outer-author tests at every size.\n"
    report += "Three view pools averaged equally. Earlier exposed D1 outcomes are exploratory links only.\n\n"
    report += table.to_markdown(index=False, floatfmt=".4f") + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
