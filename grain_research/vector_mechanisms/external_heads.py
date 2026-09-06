"""Source-exclusive, nested official-fold statistical heads for VM05."""

from __future__ import annotations

import csv
import gc
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from .cohort import ID_SEED
from .data_views import stable_key

TRAITS = (
    "Extraversion",
    "Neuroticism",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
)
AXES = {
    "EI_cont": (0, "E", "I"),
    "SN_cont": (1, "N", "S"),
    "TF_cont": (2, "F", "T"),
    "JP_cont": (3, "J", "P"),
}


def load_source_labels(prepared_root, bundles, task, source_tables):
    """Convert values only for the selected, source-exclusive non-bridge authors."""
    group = "big5_only" if task == "big5" else "mbti_only"
    if not bundles or any(b["source_group"] != group for b in bundles):
        raise ValueError("Wrong source group; bridge labels are forbidden here")
    wanted = {b["author_id"]: b for b in bundles}
    if len(wanted) != len(bundles):
        raise ValueError("Duplicate author")
    source = source_tables[task]
    path = Path(prepared_root) / source["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Prepared source changed since material freeze")
    labels, verified, unverifiable = {}, 0, 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            uid = stable_key(row["user_id"], ID_SEED)[:24]
            if uid not in wanted:
                continue  # No conversion, scoring, or storage of excluded/bridge outcomes.
            if (
                uid in labels
                or int(row["official_fold"]) != wanted[uid]["official_folds"][task]
            ):
                raise ValueError("Source fold mismatch or duplicate outcome")
            if task == "big5":
                value = np.asarray([float(row[t]) for t in TRAITS])
                if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                    raise ValueError("Big5 target must be finite on 0..1")
            else:
                value = float(row["positive_probability"])
                if value not in (0.0, 1.0):
                    raise ValueError("Official MBTI source target must be binary")
                pos, positive, negative = AXES[task]
                code = str(row.get("mbti_type", "")).strip().upper()
                if len(code) == 4 and code[pos] in (positive, negative):
                    if value != float(code[pos] == positive):
                        raise ValueError("MBTI polarity disagrees with recorded type")
                    verified += 1
                else:
                    unverifiable += 1
            labels[uid] = value
    if set(labels) != set(wanted):
        raise ValueError("Missing selected source labels")
    y = np.asarray([labels[b["author_id"]] for b in bundles])
    return y, {
        "source_sha256": source["sha256"],
        "authors": len(bundles),
        "type_polarity_verified": verified,
        "type_polarity_unavailable": unverifiable,
        "bridge_label_values_converted": False,
    }


def correlation(y, p):
    """Return Pearson r without turning undefined constant outputs into success."""
    y, p = np.asarray(y), np.asarray(p)
    if len(y) < 3 or not np.isfinite(y).all() or not np.isfinite(p).all():
        return float("nan")
    if np.std(y) <= 1e-12 or np.std(p) <= 1e-12:
        return float("nan")
    return float(pearsonr(y, p).statistic)


def task_score(y, pred, task):
    """Use exactly the frozen inner-CV objective, not a test-set preference."""
    if task == "big5":
        scores = [correlation(y[:, j], pred[:, j]) for j in range(y.shape[1])]
        return float(np.mean(scores)) if np.isfinite(scores).all() else float("nan")
    return float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")


def metrics(y, p, task):
    """Separate correlation/classification, ranking, calibration, and uncertainty."""
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid prediction shape or values")
    if task == "big5":
        rows = []
        for j, trait in enumerate(TRAITS):
            r = correlation(y[:, j], p[:, j])
            bounds = [float("nan"), float("nan")]
            if len(y) > 3 and np.isfinite(r):
                z = np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12))
                bounds = np.tanh(
                    z + np.asarray([-1, 1]) * norm.ppf(0.975) / np.sqrt(len(y) - 3)
                ).tolist()
            rows.append(
                {
                    "trait": trait,
                    "n": len(y),
                    "pearson": r,
                    "spearman": float(spearmanr(y[:, j], p[:, j]).statistic),
                    "mae": float(np.mean(abs(y[:, j] - p[:, j]))),
                    "rmse": float(np.sqrt(np.mean((y[:, j] - p[:, j]) ** 2))),
                    "fisher_lower": bounds[0],
                    "fisher_upper": bounds[1],
                    "fisher_scope": "iid approximation, not refit uncertainty",
                }
            )
        return rows
    if not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("MBTI requires binary targets and probabilities")
    return [
        {
            "trait": task,
            "n": len(y),
            "auc": task_score(y, p, task),
            "pr_auc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
            "accuracy": float(np.mean((p >= 0.5) == y)),
            "brier": float(brier_score_loss(y, p)),
            "positive_rate_true": float(y.mean()),
            "positive_rate_pred": float(np.mean(p >= 0.5)),
        }
    ]


def make_head(feature_kind, task, parameter, n_train, n_features=None, *, pca=True):
    """Build an unfitted pipeline; vocabulary/scaler/PCA learn from this fit only."""
    head = (
        Ridge(alpha=parameter, solver="lsqr")
        if task == "big5"
        else LogisticRegression(C=parameter, max_iter=5000, solver="lbfgs")
    )
    if feature_kind == "word_char":
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=10000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 5),
                        max_features=10000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        return Pipeline([("features", vectorizer), ("head", head)])
    if feature_kind != "dense" or n_features is None or n_train < 20:
        raise ValueError("Insufficient training support for dense projection")
    steps = [("scale", StandardScaler())]
    if pca:
        steps.append(
            (
                "pca",
                PCA(n_components=min(16, n_train // 10, n_features), svd_solver="full"),
            )
        )
    return Pipeline([*steps, ("head", head)])


def predict_head(model, x, task):
    """Return regression values or explicit positive-class probability."""
    if task == "big5":
        return model.predict(x)
    classes = list(model.named_steps["head"].classes_)
    if classes != [0, 1]:
        raise ValueError("Binary source head lacks both classes")
    return model.predict_proba(x)[:, 1]


def select_head(x, y, ids, feature_kind, task, *, pca=True, on_inner=None):
    """Nested selection returns an estimator refitted on the training authors only."""
    if len(ids) != len(set(ids)) or len(x) != len(ids) or len(y) != len(ids):
        raise ValueError("Require unique aligned training authors")
    grid = (0.1, 1.0, 10.0, 100.0, 1000.0) if task == "big5" else (0.01, 0.1, 1.0, 10.0)
    cv = list(KFold(3, shuffle=True, random_state=731).split(ids))
    collected = {parameter: [] for parameter in grid}
    for train, valid in cv:
        fit_train = train if on_inner is None else on_inner(train, valid)
        if set(ids[fit_train]) & set(ids[valid]):
            raise ValueError("Inner author overlap")
        prototype = make_head(
            feature_kind,
            task,
            grid[0],
            len(fit_train),
            None if feature_kind == "word_char" else x.shape[1],
            pca=pca,
        )
        transform = prototype[:-1]
        fit_x = transform.fit_transform(x[fit_train])
        valid_x = transform.transform(x[valid])
        for parameter in grid:
            candidate = make_head(
                feature_kind,
                task,
                parameter,
                len(fit_train),
                None if feature_kind == "word_char" else x.shape[1],
                pca=pca,
            )
            head = Pipeline([("head", candidate.named_steps["head"])])
            head.fit(fit_x, y[fit_train])
            pred = predict_head(head, valid_x, task)
            collected[parameter].append(task_score(y[valid], pred, task))
        # Do not retain the last inner split's large CSR matrices during the
        # next split or final full-source fit. Only scalar search scores survive.
        del fit_x, valid_x, transform, prototype, candidate, head, pred
        gc.collect()
    scores = [
        {"parameter": p, "inner_scores": values, "inner_mean": float(np.mean(values))}
        for p, values in collected.items()
    ]
    finite = [r for r in scores if np.isfinite(r["inner_mean"])]
    if not finite:
        raise ValueError("Every inner score is undefined")
    best = max(finite, key=lambda r: r["inner_mean"])["parameter"]
    final = make_head(
        feature_kind,
        task,
        best,
        len(ids),
        None if feature_kind == "word_char" else x.shape[1],
        pca=pca,
    )
    final.fit(x, y)
    return final, scores, best


def pooled_moments(means, stds, counts):
    """Combine within-view moments without losing between-view variation."""
    m, s = np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)
    w = np.array(counts, dtype=np.float64, copy=True)
    if m.shape != s.shape or m.ndim != 2 or w.shape != (len(m),) or np.any(w <= 0):
        raise ValueError("Invalid moment shapes/counts")
    if not all(np.isfinite(x).all() for x in (m, s, w)):
        raise ValueError("Nonfinite moments")
    w /= w.sum()
    mean = w @ m
    var = w @ (s * s + m * m) - mean * mean
    if np.min(var) < -1e-8:
        raise ValueError("Inconsistent pooled variance")
    return np.concatenate([mean, np.sqrt(np.maximum(var, 0))])
