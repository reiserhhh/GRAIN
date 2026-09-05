"""Scenario/author-clustered summaries, without treating tokens as subjects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def grouped_bootstrap(values, groups, *, seed=731, draws=2000) -> dict:
    """Bootstrap equally weighted cluster means and report their mean interval."""
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    means = frame.groupby("group")["value"].mean().to_numpy()
    if len(means) < 2:
        return {
            "mean": float(means.mean()) if len(means) else None,
            "ci_low": None,
            "ci_high": None,
            "clusters": len(means),
        }
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "clusters": len(means),
    }


def fact_summary(frame: pd.DataFrame) -> list[dict]:
    """Report fact accuracy and prevalence, separately from personality validity."""
    rows = []
    for (family, split), sub in frame.groupby(["family", "split"]):
        truth = sub["expected_yes"].to_numpy()
        margin = sub["margin"].to_numpy()
        pred = margin >= 0
        ci = grouped_bootstrap((truth == pred).astype(float), sub["group_id"])
        rows.append(
            {
                "family": family,
                "split": split,
                "n_prompts": len(sub),
                "positive_rate_true": float(truth.mean()),
                "positive_rate_pred": float(pred.mean()),
                "accuracy": ci["mean"],
                "accuracy_ci_low": ci["ci_low"],
                "accuracy_ci_high": ci["ci_high"],
                "scenario_n": ci["clusters"],
                "balanced_accuracy": float(balanced_accuracy_score(truth, pred))
                if len(set(truth)) == 2
                else None,
                "auc": float(roc_auc_score(truth, margin))
                if len(set(truth)) == 2
                else None,
            }
        )
    return rows


def factorial_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute 2x2 output interactions; not an information-theoretic synergy."""
    rows = []
    for group, sub in frame[frame.family == "composition"].groupby("group_id"):
        if set(sub.variant) != {0, 1, 2, 3}:
            continue
        scores = sub.set_index("variant")["margin"]
        prob = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
        rows.append(
            {
                "group_id": group,
                "split": sub.iloc[0]["split"],
                "logodds_interaction": float(
                    scores[3] - scores[2] - scores[1] + scores[0]
                ),
                "probability_interaction": float(prob[3] - prob[2] - prob[1] + prob[0]),
            }
        )
    return pd.DataFrame(rows)


def probe_layers(frame: pd.DataFrame, features: dict[str, np.ndarray]) -> pd.DataFrame:
    """Fit fixed regularized probes on discovery, never select on held-out stories."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train = (frame.split == "discovery").to_numpy()
    test = ~train
    if set(frame.loc[train, "group_id"]) & set(frame.loc[test, "group_id"]):
        raise ValueError("Scenario leakage in probe split")
    results = []
    for site, values in features.items():
        for family in sorted(frame.family.unique()):
            mask = (frame.family == family).to_numpy()
            tr, te = train & mask, test & mask
            if not tr.any() or not te.any():
                continue
            y = frame.expected_yes.to_numpy()
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            scaler = StandardScaler().fit(values[tr])
            # This instrument probe is intentionally fixed, not a tuned trait head.
            model = LogisticRegression(
                C=0.01, max_iter=1000, solver="liblinear", random_state=731
            )
            model.fit(scaler.transform(values[tr]), y[tr])
            score = model.decision_function(scaler.transform(values[te]))
            results.append(
                {
                    "site": site,
                    "family": family,
                    "heldout_auc": float(roc_auc_score(y[te], score)),
                    "n_train": int(tr.sum()),
                    "n_eval": int(te.sum()),
                    "eval_scenarios": int(frame.loc[te, "group_id"].nunique()),
                    "scope": "fixed_probe_semantic_facts_not_personality",
                }
            )
    return pd.DataFrame(results)
