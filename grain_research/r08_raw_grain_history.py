"""R08 exact-cache raw-vector and VM05 lexical/SVD mean-std GRAIN history comparators (packet 23).

Common observable target: the next event's 30 source-standardized IPIP-keyed values (identical events/units for every
arm). Features use ONLY prefix events 0..t-1 in the pinned time order. Four standalone affine ridge readouts with fixed
lambda = 1. VM05 GRAIN here = legacy lexical/SVD 24-coordinate comment units summarized by mean/std (48 named columns);
NOT the whole GRAIN framework or anonymous psychological factors. No text, labels or neural calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAT = RES / "grain_vector_mechanisms/predecessor_integration_v1/material"
CREF = RES / "grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2"
PRE = RES / "grain_vector_mechanisms/vm05_preflight_r1"
PARENT = RES / "grain_vector_mechanisms/vm05_full_source_cohort_r1"
E1 = RES / "v7_multiview_projection/e1_v72_full_20260715"
T0, NEV, RECENT, K_PCA, LAMBDA = 25, 36, 4, 30, 1.0
E5_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"


# ------------------------------------------------------------------ bridge loaders (numeric arrays and IDs only)
def load_raw_comments():
    with np.load(MAT / "private_arrays.npz", allow_pickle=False) as z:     # only these two keys are read (not y/folds/grain)
        return np.asarray(z["comments"]), z["author_ids"].astype(str)


def load_directions():
    with np.load(CREF / "private_directions.npz", allow_pickle=False) as z:
        return np.asarray(z["direction"], float)


def keyed_from_raw(vecs, direction):
    """Exact conditional_reference_r2 formula: float64 L2 normalisation then projection on frozen keyed directions."""
    h = np.asarray(vecs, float)
    return (h / np.linalg.norm(h, axis=-1, keepdims=True)) @ direction


def unit_normalise(vecs):
    h = np.asarray(vecs, float)
    return h / np.linalg.norm(h, axis=-1, keepdims=True)


def load_grain_units(cohort_ids):
    with np.load(PRE / "grain_frozen_features.npz", allow_pickle=False) as z:
        ids = z["author_ids"].astype(str)
        if len(set(ids)) != len(ids):
            raise ValueError("non-unique preflight author ids")
        ix = {u: i for i, u in enumerate(ids)}
        sel = np.array([ix[u] for u in cohort_ids])
        units = np.asarray(z["grain_units_raw"][sel], float)
        views = {v: np.asarray(z[f"view{v}_raw_features"][sel], float) for v in range(3)}
        all36 = np.asarray(z["all36_raw_features"][sel], float)
    names = json.loads((PRE / "manifest.json").read_text())["feature_names"]
    return units, views, all36, names


def named_mean_std(units, names):
    """raw_author_features formula (population std) on an arbitrary unit block; names give column order."""
    h = np.asarray(units, float)
    mean, std = h.mean(-2), h.std(-2, ddof=0)
    cols = []
    for name in names:
        prefix, index = name.split("::svd_")
        cols.append((mean if prefix == "mean" else std)[..., int(index)])
    return np.stack(cols, axis=-1)


# ------------------------------------------------------------------ source-only transforms
def fit_pca(train_vectors, k=K_PCA):
    """Centered PCA on source-train unit-normalised event vectors; deterministic SVD; sign: largest |loading| positive."""
    mu = train_vectors.mean(0)
    _, s, vt = np.linalg.svd(train_vectors - mu, full_matrices=False)
    W = vt[:k].T.copy()
    sign = np.sign(W[np.abs(W).argmax(0), np.arange(k)]); sign[sign == 0] = 1
    W *= sign
    var = s ** 2
    return {"mu": mu, "W": W, "retained_variance_share": float(var[:k].sum() / var.sum())}


def pca_coords(vectors, pca):
    return (vectors - pca["mu"]) @ pca["W"]


# ------------------------------------------------------------------ prefix features (events strictly before t)
def feat_keyed_static(Zpre):
    return Zpre.mean(0)


def feat_raw_static(Ppre):
    return np.r_[Ppre.mean(0), Ppre.std(0, ddof=0)]


def feat_grain_static(Gpre, names):
    return named_mean_std(Gpre, names)


def feat_raw_ordered(Ppre, mean_override=None, std_override=None):
    m = Ppre.mean(0) if mean_override is None else mean_override
    s = Ppre.std(0, ddof=0) if std_override is None else std_override
    return np.r_[m, s, Ppre[-1] - m, Ppre[-RECENT:].mean(0) - m]


# ------------------------------------------------------------------ affine ridge
def fit_scaler(X):
    mu, sd = X.mean(0), X.std(0, ddof=0)
    zero = sd < 1e-12
    return {"mu": mu, "sd": np.where(zero, 1.0, sd), "zero_sd": zero}


def apply_scaler(X, sc):
    Xs = (X - sc["mu"]) / sc["sd"]
    Xs[..., sc["zero_sd"]] = 0.0
    return Xs


def fit_affine_ridge(Xs, Y, lam=LAMBDA):
    """argmin_{B,b} (1/n)||Y - Xs B - 1 b'||_F^2 + lam ||B||_F^2, intercept unpenalised.
    Xs columns have mean 0 on fitting rows, so b = mean(Y) - mean(Xs) B and (Xc'Xc + n lam I) B = Xc'Yc."""
    n = Xs.shape[0]
    xm, ym = Xs.mean(0), Y.mean(0)
    Xc, Yc = Xs - xm, Y - ym
    B = np.linalg.solve(Xc.T @ Xc + n * lam * np.eye(Xs.shape[1]), Xc.T @ Yc)
    return B, ym - xm @ B


def ridge_gradients(Xs, Y, B, b, lam=LAMBDA):
    n = Xs.shape[0]
    R = Y - Xs @ B - b
    return float(np.abs(-(2 / n) * Xs.T @ R + 2 * lam * B).max()), float(np.abs(-(2 / n) * R.sum(0)).max())


def permuted_order(t, role_code, author, seed=22310):
    return np.random.default_rng([seed, role_code, int(author), int(t)]).permutation(t)
