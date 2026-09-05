"""Label-free fold transform caches with complete repeated supervised selection."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from .external_heads import make_head, predict_head, task_score


def parameter_grid(task):
    """Use the same ordered grid as the source OOF protocol, including tie order."""
    return (0.1, 1.0, 10.0, 100.0, 1000.0) if task == "big5" else (0.01, 0.1, 1.0, 10.0)


def prepare_nested_folds(x, ids, outer_folds, feature_kind, task, *, pca=True):
    """Fit unsupervised transforms on each INNER training set, never pooled data."""
    if len(set(ids)) != len(ids) or set(outer_folds) != set(range(5)):
        raise ValueError("Require unique authors and all five official folds")
    plans = []
    d = None if feature_kind == "word_char" else x.shape[1]
    for outer in range(5):
        train, test = (
            np.flatnonzero(outer_folds != outer),
            np.flatnonzero(outer_folds == outer),
        )
        inner = []
        for ti, vi in KFold(3, shuffle=True, random_state=731).split(train):
            fit, valid = train[ti], train[vi]
            if set(ids[fit]) & set(ids[valid]) or set(ids[fit]) & set(ids[test]):
                raise ValueError("Nested source overlap")
            transform = make_head(
                feature_kind, task, parameter_grid(task)[0], len(fit), d, pca=pca
            )[:-1]
            inner.append(
                {
                    "fit": fit,
                    "valid": valid,
                    "xfit": transform.fit_transform(x[fit]),
                    "xvalid": transform.transform(x[valid]),
                }
            )
        transform = make_head(
            feature_kind, task, parameter_grid(task)[0], len(train), d, pca=pca
        )[:-1]
        plans.append(
            {
                "outer": outer,
                "train": train,
                "test": test,
                "inner": inner,
                "xtrain": transform.fit_transform(x[train]),
                "xtest": transform.transform(x[test]),
                "n_features": d,
                "feature_kind": feature_kind,
                "pca": pca,
            }
        )
    return plans


def nested_predict_cached(plans, y, task):
    """Refit every label-dependent head/grid selection for one complete label map."""
    pred = np.full(y.shape, np.nan)
    selections = []
    for plan in plans:
        scores = []
        for parameter in parameter_grid(task):
            values = []
            for part in plan["inner"]:
                model = make_head(
                    plan["feature_kind"],
                    task,
                    parameter,
                    len(part["fit"]),
                    plan["n_features"],
                    pca=plan["pca"],
                )
                head = Pipeline([("head", model.named_steps["head"])])
                head.fit(part["xfit"], y[part["fit"]])
                values.append(
                    task_score(
                        y[part["valid"]], predict_head(head, part["xvalid"], task), task
                    )
                )
            scores.append((parameter, float(np.mean(values))))
        finite = [r for r in scores if np.isfinite(r[1])]
        if not finite:
            raise ValueError("No defined inner objective")
        parameter = max(finite, key=lambda r: r[1])[0]
        model = make_head(
            plan["feature_kind"],
            task,
            parameter,
            len(plan["train"]),
            plan["n_features"],
            pca=plan["pca"],
        )
        head = Pipeline([("head", model.named_steps["head"])])
        head.fit(plan["xtrain"], y[plan["train"]])
        pred[plan["test"]] = predict_head(head, plan["xtest"], task)
        selections.append(parameter)
    if not np.isfinite(pred).all():
        raise ValueError("Incomplete permuted OOF")
    return pred, selections
