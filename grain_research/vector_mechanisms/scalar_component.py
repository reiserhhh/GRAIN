#!/usr/bin/env python3
"""Small native-score maps distinguishing scalar, distribution and component use."""

from __future__ import annotations

import numpy as np
import pandas as pd

ARMS = (
    "raw_f",
    "affine_author",
    "cubic_author",
    "components_author",
    "affine_view",
    "cubic_view",
    "components_view",
)
COMPONENTS = ("mean", "configuration", "final")


class SmallReadout:
    """Training-only standardization and weighted mean-loss Ridge, fixed penalty."""

    def __init__(self, kind, penalty=0.01):
        if kind not in ("affine", "cubic", "components") or penalty <= 0:
            raise ValueError("Known finite map and positive penalty required")
        self.kind, self.penalty = kind, penalty

    def _check_x(self, x):
        """Preserve scalar versus three-component input contracts."""
        x = np.asarray(x, float)
        if (
            x.ndim != 2
            or x.shape[1] != (3 if self.kind == "components" else 1)
            or not len(x)
            or not np.isfinite(x).all()
        ):
            raise ValueError("Finite rows with the declared input dimension required")
        return x

    def _basis(self, x):
        """Use powers of one standardized scalar, or unchanged coordinates."""
        z = (self._check_x(x) - self.input_mean) / self.input_scale
        return (
            np.column_stack([z[:, 0] ** i for i in (1, 2, 3)])
            if self.kind == "cubic"
            else z
        )

    def fit(self, x, teacher, weights=None):
        """Fit only the supplied model target; sample weights sum to one."""
        x, y = self._check_x(x), np.asarray(teacher, float)
        w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
        if (
            y.shape != (len(x),)
            or w.shape != y.shape
            or len(x) < 2
            or not np.isfinite([y, w]).all()
            or (w <= 0).any()
        ):
            raise ValueError(
                "Aligned finite teacher values and positive weights required"
            )
        w = w / w.sum()
        self.input_mean = w @ x
        self.input_scale = np.sqrt(w @ (x - self.input_mean) ** 2)
        self.input_scale = np.where(self.input_scale < 1e-12, 1, self.input_scale)
        self.input_min, self.input_max = x.min(0), x.max(0)
        phi = self._basis(x)
        self.feature_mean = w @ phi
        self.feature_scale = np.sqrt(w @ (phi - self.feature_mean) ** 2)
        self.feature_scale = np.where(self.feature_scale < 1e-12, 1, self.feature_scale)
        a = (phi - self.feature_mean) / self.feature_scale
        self.intercept = float(w @ y)
        self.coef = np.linalg.solve(
            a.T @ (w[:, None] * a) + self.penalty * np.eye(a.shape[1]),
            a.T @ (w * (y - self.intercept)),
        )
        return self

    def predict(self, x):
        """Evaluate frozen coefficients without refitting or clipping."""
        a = (self._basis(x) - self.feature_mean) / self.feature_scale
        return self.intercept + a @ self.coef

    def extrapolated(self, x):
        """Flag coordinates outside the training input range without excluding rows."""
        x = self._check_x(x)
        return ((x < self.input_min) | (x > self.input_max)).any(1)

    def jensen_parts(self, grouped):
        """Exact within-author cubic aggregation gap for this one frozen map."""
        x = np.asarray(grouped, float)
        if self.kind not in ("affine", "cubic") or x.ndim != 3 or x.shape[2] != 1:
            raise ValueError("Author/view/scalar arrays required")
        z = ((x - self.input_mean) / self.input_scale)[:, :, 0]
        mu = z.mean(1)
        dev = z - mu[:, None]
        var, third = (dev**2).mean(1), (dev**3).mean(1)
        gamma = self.coef / self.feature_scale
        second = gamma[1] * var if self.kind == "cubic" else np.zeros(len(x))
        cubic = (
            gamma[2] * (3 * mu * var + third)
            if self.kind == "cubic"
            else np.zeros(len(x))
        )
        direct = self.predict(x.reshape(-1, 1)).reshape(x.shape[:2]).mean(
            1
        ) - self.predict(x.mean(1))
        return {
            "direct": direct,
            "second": second,
            "third": cubic,
            "closure_error": np.abs(direct - second - cubic),
        }

    def state(self):
        """Serializable private model state, including training-only geometry."""
        return {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in vars(self).items()
        }


def construct_oof(panel):
    """Fit fixed arms on author-disjoint folds; no questionnaire input exists."""
    required = ["author_id", "query", "view", "fold", "teacher", "score", *COMPONENTS]
    if any(c not in panel for c in required) or "y_true" in panel:
        raise ValueError("Only frozen model inputs and teacher targets are permitted")
    data = panel[required].sort_values(["query", "author_id", "view"]).copy()
    keys = ["author_id", "query"]
    if (
        data.duplicated(keys + ["view"]).any()
        or data.groupby(keys).size().ne(3).any()
        or data.groupby(keys).fold.nunique().ne(1).any()
        or not np.isfinite(data[["teacher", "score", *COMPONENTS]]).all().all()
    ):
        raise ValueError("Complete finite three-view author panels required")
    predictions, models, jensen = [], [], []
    for query, group in data.groupby("query", sort=True):
        for fold in sorted(group.fold.unique()):
            train, test = group[group.fold != fold], group[group.fold == fold]
            if train.author_id.nunique() < 2 or not len(test):
                raise ValueError("Nonempty disjoint author folds required")
            if set(train.author_id) & set(test.author_id):
                raise ValueError("Author leakage")
            train_author = train.groupby("author_id", sort=True)[
                ["teacher", "score", *COMPONENTS]
            ].mean()
            baseline = float(train_author.teacher.mean())
            for arm in ARMS:
                if arm == "raw_f":
                    predicted, extra = test.score.to_numpy(), np.zeros(len(test), bool)
                else:
                    kind, aggregation = arm.rsplit("_", 1)
                    columns = list(COMPONENTS) if kind == "components" else ["score"]
                    fitting = train_author if aggregation == "author" else train
                    weights = np.ones(len(fitting)) / (
                        3 if aggregation == "view" else 1
                    )
                    model = SmallReadout(kind).fit(
                        fitting[columns], fitting.teacher, weights
                    )
                    models.append(
                        {
                            "query": query,
                            "fold": int(fold),
                            "arm": arm,
                            "training_authors": train.author_id.nunique(),
                            **model.state(),
                        }
                    )
                    if aggregation == "author":
                        avg = test.groupby("author_id", sort=True)[columns].mean()
                        values = pd.Series(model.predict(avg), index=avg.index)
                        flags = pd.Series(model.extrapolated(avg), index=avg.index)
                        predicted, extra = (
                            test.author_id.map(values).to_numpy(),
                            test.author_id.map(flags).to_numpy(),
                        )
                    else:
                        predicted, extra = (
                            model.predict(test[columns]),
                            model.extrapolated(test[columns]),
                        )
                        if kind in ("affine", "cubic"):
                            grouped = test[columns].to_numpy().reshape(-1, 3, 1)
                            account = model.jensen_parts(grouped)
                            for i, author in enumerate(test.author_id.unique()):
                                jensen.append(
                                    {
                                        "author_id": author,
                                        "query": query,
                                        "fold": int(fold),
                                        "arm": arm,
                                        **{k: float(v[i]) for k, v in account.items()},
                                    }
                                )
                for i, (_, row) in enumerate(test.iterrows()):
                    predictions.append(
                        {
                            **{k: row[k] for k in (*keys, "view", "fold", "teacher")},
                            "arm": arm,
                            "prediction": float(predicted[i]),
                            "extrapolated": bool(extra[i]),
                            "training_mean": baseline,
                        }
                    )
    return pd.DataFrame(predictions), models, pd.DataFrame(jensen)
