"""Frozen F03 source-learned selection adjustment and honest observation views."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def fit_source_selection(rows: list[dict]) -> dict:
    """Learn shared z/x slopes and source-author intercepts, never application Y.

    Author one-hot columns are the prescribed source random-effect approximation,
    not an identity feature transported into the application predictor.
    """
    if not rows or any(row["role"] != "source" for row in rows):
        raise ValueError("selection learning requires source-only records")
    observed = [row for row in rows if row["opportunity"] == 1]
    if not observed:
        raise ValueError("no source opportunities")
    authors = sorted({row["author_id"] for row in observed})
    columns = {author: i for i, author in enumerate(authors)}
    covariates = np.array(
        [[row["condition"]["z"], row["condition"]["x"]] for row in observed], float
    )
    indicators = sparse.csr_matrix(
        (
            np.ones(len(observed)),
            (np.arange(len(observed)), [columns[row["author_id"]] for row in observed]),
        ),
        shape=(len(observed), len(authors)),
    )
    design = sparse.hstack((sparse.csr_matrix(covariates), indicators), format="csr")
    selected = np.array([row["selected"] for row in observed])
    if len(np.unique(selected)) != 2:
        raise ValueError("source selection has one class; do not invent fitted slopes")
    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, fit_intercept=False
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(design, selected)
    intercepts = model.coef_[0, 2:]
    return {
        "shared_slopes": model.coef_[0, :2].tolist(),
        "intercept_mean": float(intercepts.mean()),
        "intercept_variance": max(0.1, float(np.var(intercepts, ddof=1)))
        if len(intercepts) > 1
        else 0.1,
        "source_intercepts": dict(zip(authors, intercepts.tolist())),
        "source_authors": len(authors),
        "source_opportunities": len(observed),
        "iterations": int(model.n_iter_[0]),
        "converged": not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        ),
        "warnings": [str(w.message) for w in caught],
        "features": ["z", "x"],
        "estimator": "source_C1_logistic_author_intercepts",
    }


def personal_propensity(prefix: list[dict], source: dict) -> dict:
    """Estimate only a personal intercept from observed opportunities in a prefix."""
    observed = [row for row in prefix if row["opportunity"] == 1]
    mean, variance = source["intercept_mean"], source["intercept_variance"]
    if not observed:
        return {
            "status": "PRIOR_ONLY_NO_OPPORTUNITIES",
            "intercept": mean,
            "events": [],
            "q": np.empty(0),
            "clip_count": 0,
            "low_support_count": 0,
        }
    features = np.array(
        [[row["condition"]["z"], row["condition"]["x"]] for row in observed], float
    )
    offset = features @ np.asarray(source["shared_slopes"])
    labels = np.array([row["selected"] for row in observed], float)

    def objective(intercept):
        linear = offset + intercept
        return float(
            np.sum(np.logaddexp(0, linear) - labels * linear)
            + (intercept - mean) ** 2 / (2 * variance)
        )

    fit = minimize_scalar(
        objective, bounds=(-6.0, 6.0), method="bounded", options={"xatol": 1e-10}
    )
    if not fit.success:
        return {"status": "OPTIMIZATION_FAILED", "reason": str(fit.message)}
    raw_q = expit(offset + fit.x)
    return {
        "status": "FIT",
        "intercept": float(fit.x),
        "events": observed,
        "q": np.clip(raw_q, 0.01, 0.99),
        "clip_count": int(np.sum((raw_q < 0.01) | (raw_q > 0.99))),
        "low_support_count": int(np.sum(raw_q < 0.05)),
        "objective": float(fit.fun),
    }


def observe_view(rows: list[dict], view: str) -> list[dict]:
    """Strip unavailable fields, rather than asking downstream code to ignore them."""
    if view == "all_opportunities":
        return [dict(row) for row in rows]
    if view == "selected_indicator":
        return [
            {
                key: value
                for key, value in row.items()
                if key not in ("opportunity", "opportunity_status")
            }
            for row in rows
        ]
    if view == "published_only":
        # No ordinal t or global event address from which a hidden denominator
        # could be recovered. Local labels are addresses, not source indices.
        return [
            {
                "condition": row["condition"],
                "observation": row["observation"],
                "selected": 1,
                "local_publication_index": i,
            }
            for i, row in enumerate(row for row in rows if row["selected"])
        ]
    raise ValueError(view)


def score_views(prefix: list[dict], source: dict | None, design: str) -> list[dict]:
    """Return points, plug-in bounds, and missing information by condition/view."""
    propensity = personal_propensity(prefix, source) if source is not None else None
    scores = []
    for view in ("all_opportunities", "selected_indicator", "published_only"):
        rows = observe_view(prefix, view)
        for z in (-1, 1):
            cell = [row for row in rows if row["condition"]["z"] == z]
            values = np.array(
                [row["observation"] for row in cell if row["selected"]], float
            )
            base = {
                "view": view,
                "z": z,
                "selected_n": len(values),
                "naive": float(values.mean()) if len(values) else None,
                "fixed_beta": float((values.sum() + 1) / (len(values) + 2))
                if design == "fixed"
                else None,
                "fixed_beta_support": "PRIOR_ONLY"
                if not len(values) and design == "fixed"
                else "OBSERVED",
                "total_opportunities": len(cell) if view != "published_only" else None,
                "pi": None,
                "q": None,
                "f": None,
                "ipw": None,
                "bound_low": None,
                "bound_high": None,
                "independent_O_bound_low": None,
                "independent_O_bound_high": None,
                "q_bound_low": None,
                "q_bound_high": None,
                "clip_count": None,
                "low_support_count": None,
            }
            if view != "published_only" and cell:
                f, g = len(values) / len(cell), values.sum() / len(cell)
                base.update(
                    f=float(f),
                    bound_low=float(g),
                    bound_high=float(g + 1 - f),
                    q_bound_low=float(f),
                    q_bound_high=1.0,
                )
                if view == "all_opportunities":
                    n_o = sum(row["opportunity"] for row in cell)
                    base["pi"] = n_o / len(cell)
                    base["q"] = len(values) / n_o if n_o else None
                    if n_o:
                        base.update(
                            independent_O_bound_low=float(values.sum() / n_o),
                            independent_O_bound_high=float(
                                (values.sum() + n_o - len(values)) / n_o
                            ),
                        )
                    else:
                        base.update(
                            independent_O_bound_low=0.0, independent_O_bound_high=1.0
                        )
            if view == "all_opportunities" and propensity:
                base["propensity_status"] = propensity["status"]
                if propensity["status"] == "FIT":
                    eligible = np.array(
                        [
                            row["selected"] and row["condition"]["z"] == z
                            for row in propensity["events"]
                        ],
                        bool,
                    )
                    if eligible.any():
                        weights = 1 / propensity["q"][eligible]
                        observed_y = np.array(
                            [row["observation"] or 0 for row in propensity["events"]],
                            float,
                        )[eligible]
                        base["ipw"] = float(weights @ observed_y / weights.sum())
                    base.update(
                        clip_count=propensity["clip_count"],
                        low_support_count=propensity["low_support_count"],
                    )
            base["bound_status"] = (
                "NO_DENOMINATOR"
                if view == "published_only"
                else "NO_CONDITION_SUPPORT"
                if not cell
                else "SAMPLE_PLUGIN_NOT_CI"
            )
            scores.append(base)
    return scores
