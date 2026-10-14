#!/usr/bin/env python3
"""Frozen common-factor scoring and matched PCA references for the BLT method card.

These are explicitly local adaptations: MINRES and psych promax, not a claim to
recover the original paper's unspecified estimator/equamax-promax sequence.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from grain_research.predecessors import varimax

R_DRIVER = Path(__file__).resolve().parents[1] / "scripts/fit_grain_common_factors.R"


class FrozenCommonFactors(BaseEstimator, TransformerMixin):
    """Fit x = f Lambda' + epsilon using native psych, then freeze score weights.

    Cov(x) = Lambda Phi Lambda' + diag(psi). New scores use training means,
    standard deviations, and Thurstone regression weights; no new-batch fitting.
    """

    def __init__(self, n_factors=5, timeout=180):
        self.n_factors, self.timeout = n_factors, timeout

    def fit(self, x, y=None):
        """Call the standard R implementation on source rows only; keep diagnostics."""
        x = np.asarray(x, float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Expected finite two-dimensional source data")
        if not 1 <= self.n_factors < min(x.shape):
            raise ValueError("Insufficient source support for requested factor count")
        self.mean_, self.scale_ = x.mean(0), x.std(0, ddof=1)
        if np.any(self.scale_ < 1e-12):
            raise ValueError("Constant source variable")
        rscript = shutil.which("Rscript")
        if rscript is None:
            raise RuntimeError("Rscript with psych/GPArotation is required")
        with tempfile.TemporaryDirectory(prefix="grain_factor_") as directory:
            directory = Path(directory)
            pd.DataFrame(x).to_csv(directory / "source.csv", index=False)
            result = subprocess.run(
                [
                    rscript,
                    str(R_DRIVER),
                    str(directory / "source.csv"),
                    str(directory / "output"),
                    str(self.n_factors),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode:
                raise RuntimeError("Common factor fit failed: " + result.stderr[-3000:])
            out = directory / "output"
            for name in (
                "pattern",
                "phi",
                "structure",
                "weights",
                "uniquenesses",
                "observed_correlation",
            ):
                setattr(self, name + "_", pd.read_csv(out / f"{name}.csv").to_numpy())
            self.uniquenesses_ = self.uniquenesses_.ravel()
            d = pd.read_csv(out / "diagnostics.csv")
            self.diagnostics_ = dict(zip(d.key, d.value, strict=True))
            self.warnings_ = (out / "warnings.txt").read_text().splitlines()
            self.versions_ = (out / "versions.txt").read_text().splitlines()
            source_scores = pd.read_csv(out / "source_scores.csv").to_numpy()
            np.testing.assert_allclose(self.transform(x), source_scores, atol=1e-7)
        if not all(
            np.isfinite(v).all() for v in (self.weights_, self.phi_, self.pattern_)
        ):
            raise ValueError("Nonfinite factor solution")
        if np.linalg.eigvalsh(self.phi_).min() < -1e-8:
            raise ValueError("Invalid factor correlation matrix")
        # Scores may exist for a Heywood solution; do not silently certify it as admissible.
        self.admissible_ = bool(np.all(self.uniquenesses_ >= -1e-8))
        return self

    def transform(self, x):
        """Score arbitrary batches against the frozen training reference."""
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != len(self.mean_) or not np.isfinite(x).all():
            raise ValueError("Invalid scoring matrix")
        return ((x - self.mean_) / self.scale_) @ self.weights_


class WordFactorReference(BaseEstimator, TransformerMixin):
    """Matched source-only word-rate input for common factors or orthogonal PCA.

    Rates use all alphabetic tokens as denominator, not a closed sum across the
    selected vocabulary. Rank five and a 200-term cap are fixed local budgets.
    """

    def __init__(self, method="common_factor", n_factors=5, max_features=200):
        self.method, self.n_factors, self.max_features = method, n_factors, max_features

    def _rates(self, documents):
        """Apply the fixed vocabulary while computing each document's own exposure."""
        documents = list(documents)
        counts = self.vectorizer_.transform(documents).toarray().astype(float)
        tokenizer = self.vectorizer_.build_tokenizer()
        denominator = np.asarray([len(tokenizer(d.lower())) for d in documents])
        return counts / np.maximum(denominator[:, None], 1)

    def fit(self, x, y=None):
        """Discover terms and latent structure without using any trait outcomes."""
        self.vectorizer_ = CountVectorizer(
            stop_words="english",
            min_df=0.05,
            max_df=0.95,
            max_features=self.max_features,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        ).fit(x)
        rates = self._rates(x)
        self.mean_, self.scale_ = rates.mean(0), rates.std(0, ddof=1)
        if np.any(self.scale_ < 1e-12):
            raise ValueError("Constant word-rate feature")
        if self.method == "common_factor":
            self.model_ = FrozenCommonFactors(self.n_factors).fit(rates)
            self.loadings_ = self.model_.pattern_
            self.structure_ = self.model_.structure_
            self.diagnostics_ = self.model_.diagnostics_
        elif self.method == "pca":
            self.model_ = PCA(n_components=self.n_factors, svd_solver="full").fit(
                (rates - self.mean_) / self.scale_
            )
            self.score_scale_ = np.sqrt(self.model_.explained_variance_)
            raw = self.model_.components_.T * self.score_scale_
            self.rotation_, self.diagnostics_ = varimax(raw)
            self.loadings_ = raw @ self.rotation_
            self.structure_ = self.loadings_
        else:
            raise ValueError("Unknown factor method")
        return self

    def transform(self, x):
        """Produce source-referenced scores; held authors cannot alter the axes."""
        rates = self._rates(x)
        if self.method == "common_factor":
            return self.model_.transform(rates)
        z = (rates - self.mean_) / self.scale_
        return (self.model_.transform(z) / self.score_scale_) @ self.rotation_
