"""Independent source-anchor identification of constant shared reader effects."""

import numpy as np

from grain_research.gptpro_foundation_v3.f11_mtmm import designs, mtmm_covariance


def covariance_basis(anchored=True):
    """Nine covariance components; an independent anchor breaks one alias."""
    f, m, _ = designs()
    if anchored:
        f = np.vstack([f, np.eye(3)])
        m = np.vstack([m, np.zeros((3, 3))])
    n = len(f)
    basis = []
    for i in range(3):
        for j in range(i + 1):
            sigma = np.zeros((3, 3))
            sigma[i, j] = sigma[j, i] = 1
            basis.append(f @ sigma @ f.T)
    errors = np.zeros((n, n))
    errors[:18, :18] = np.eye(18)
    actor = np.zeros(n)
    actor[:18] = 1
    basis.extend([m @ m.T, errors, np.outer(actor, actor)])
    return np.column_stack([b.ravel() for b in basis])


def fit_anchor(source_text, source_anchor, anchor_noise=0.25):
    """Fit moments from source text plus independent unit-loading noisy anchors."""
    y, z = np.asarray(source_text), np.asarray(source_anchor)
    if y.ndim != 2 or y.shape[1] != 18 or z.shape != (len(y), 3):
        raise ValueError("paired source n x18 text and n x3 anchor required")
    if (
        len(y) < 22
        or not np.isfinite(y).all()
        or not np.isfinite(z).all()
        or anchor_noise < 0
    ):
        raise ValueError(
            "finite source observations and nonnegative known anchor noise required"
        )
    combined = np.column_stack([y, z])
    empirical = np.cov(combined, rowvar=False)
    empirical[18:, 18:] -= anchor_noise * np.eye(3)
    design = covariance_basis()
    coefficients = np.linalg.lstsq(design, empirical.ravel(), rcond=None)[0]
    sigma = np.zeros((3, 3))
    index = 0
    for i in range(3):
        for j in range(i + 1):
            sigma[i, j] = sigma[j, i] = coefficients[index]
            index += 1
    eigen, vectors = np.linalg.eigh(sigma)
    clean = (vectors * np.maximum(eigen, 1e-8)) @ vectors.T
    return {
        "sigma": clean,
        "method": max(float(coefficients[6]), 1e-8),
        "error": max(float(coefficients[7]), 1e-8),
        "actor": max(float(coefficients[8]), 0.0),
        "mean": y.mean(axis=0),
        "raw_coefficients": coefficients,
        "raw_sigma": sigma,
        "rank": int(np.linalg.matrix_rank(design)),
    }


def text_only_blup(model, application_text):
    """Score new text using source-calibrated covariance, without application anchors."""
    f, _, _ = designs()
    sigma = model["sigma"]
    v = mtmm_covariance(
        sigma, model["method"], 0.0, model["error"], model.get("actor", 0.0)
    )
    cross = sigma @ f.T
    gain = np.linalg.solve(v, cross.T).T
    return (application_text - model["mean"]) @ gain.T, sigma - gain @ cross.T
