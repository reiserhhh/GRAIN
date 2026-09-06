#!/usr/bin/env python3
"""Saved-readout algebra on artificial data, including the logistic link caveat."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.audit_grain_ready_additivity import audit_head


@pytest.mark.parametrize("task", ["big5", "EI_cont"])
def test_affine_coordinate_pullback_and_zero_logit_interaction(task):
    rng = np.random.default_rng(14)
    x = rng.normal(size=(80, 24))
    y = x[:, :5] if task == "big5" else (x[:, 0] + x[:, 20] > 0).astype(int)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=8)),
            ("head", Ridge() if task == "big5" else LogisticRegression()),
        ]
    ).fit(x, y)
    result = audit_head(model, 8, task)
    assert result["mixed_difference_max"] < 1e-12
    assert result["coefficient_pullback_max"] < 1e-12
    model.named_steps["pca"].whiten = True
    with pytest.raises(ValueError, match="coordinate"):
        audit_head(model, 8, task)


def test_probability_curvature_is_not_logit_interaction():
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    eta = 0.2 + 0.9 * x[:, 0] - 0.7 * x[:, 1]
    probability = 1 / (1 + np.exp(-eta))
    assert abs(eta[3] - eta[1] - eta[2] + eta[0]) < 1e-15
    assert abs(probability[3] - probability[1] - probability[2] + probability[0]) > 1e-3
    interaction = x[:, 0] * x[:, 1]
    assert interaction[3] - interaction[1] - interaction[2] + interaction[0] == 1
    bad = make_pipeline(StandardScaler(), Ridge())
    with pytest.raises(ValueError, match="fixed"):
        audit_head(bad, 1, "big5")
