"""R08 incremental history information on existing PANDORA caches (packet 22).

Reuses the pinned packet-15 loader/roles/baselines and the packet-16 saved OOM operators and filter unchanged.
B1: one nested convex venue mixture pMix = (1-w) pB + w pOOM (w chosen on source-validation only).
B2: two fixed coupled linear residual readouts of the 30 keyed semantic projections (lambda = 1, no grid), each with
one validation-chosen blend multiplier, plus a prefix-permutation control on the frozen history readout.
Cache values are 30 IPIP-keyed semantic projections, NOT raw embeddings, GRAIN factors or personality scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_pandora_history as ph  # noqa: E402  pinned loader
import run_grain_r08_pandora_venue_oom as vo  # noqa: E402  frozen OOM filter and role/dictionary rule

K_TOP, ALPHA, WV, T0, NEV = 20, 3, 0.2, 25, 36
C = K_TOP + 1
W_VENUE_GRID = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
W_KEYED_GRID = (0.0, 0.25, 0.5, 1.0)
LAMBDA = 1.0
RECENT = 4
OOM_RANK = 20
ROLE_CODE = {"source_validation": 1, "held": 2}


# ------------------------------------------------------------------ data and roles (pinned)
def load_all():
    m, X, ident = ph.load()
    train_auth, val_auth, held_auth, tr, top = vo.roles_and_dictionary(m)
    by_author = {a: np.where(m.author_index.to_numpy() == a)[0] for a in m.author_index.unique()}
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Z = (X - mu) / sd
    return {"m": m, "X": X, "Z": Z, "mu": mu, "sd": sd, "ident": ident, "train": train_auth, "val": val_auth, "held": held_auth, "tr": tr, "top": top,
            "by_author": by_author, "cats": m.cat.to_numpy(), "raw": m.condition.to_numpy()}


# ------------------------------------------------------------------ venue (B1)
def venue_population(D):
    cats, tr = D["cats"], D["tr"]
    g = np.bincount(cats[tr], minlength=C) + 1.0; g /= g.sum()
    T = np.ones((C, C))
    for a in D["train"]:
        c = cats[D["by_author"][a]]
        np.add.at(T, (c[:-1], c[1:]), 1)
    T /= T.sum(1, keepdims=True)
    return g, T


def venue_base(prev_cats, g, T):
    cnt = np.bincount(prev_cats, minlength=C)
    static = (cnt + ALPHA * g) / (cnt.sum() + ALPHA)
    o1 = T[prev_cats[-1]]
    return {"global": g, "static": static, "order1": o1, "combined": WV * o1 + (1 - WV) * static}


def load_oom(path, k=OOM_RANK):
    with np.load(path) as z:
        return {"U": z[f"k{k}_U"], "b1": z[f"k{k}_b1"], "binf": z[f"k{k}_binf"], "B": z[f"k{k}_B"], "k": k}


def oom_records(seq, f):
    recs, _ = vo.oom_filter(np.asarray(seq), f)
    return {r["t"]: r for r in recs}


def mix(pB, pO, w):
    """Nested convex mixture; a refused OOM prediction (pO is None) falls back to pB (declared)."""
    if pO is None:
        return pB
    return (1 - w) * pB + w * pO


# ------------------------------------------------------------------ keyed semantic history (B2)
def keyed_population(D):
    """Old packet-15 objects: diagonal AR phi and PCA5 on train author means (replay only)."""
    Z, by = D["Z"], D["by_author"]
    means = np.array([Z[by[a]].mean(0) for a in D["train"]])
    _, _, Vt = np.linalg.svd(means - means.mean(0), full_matrices=False)
    num, den = np.zeros(30), np.zeros(30)
    for a in D["train"]:
        ix = by[a]
        for t in range(2, len(ix)):
            pm = Z[ix[:t]].mean(0)
            dl, dn = Z[ix[t - 1]] - pm, Z[ix[t]] - pm
            num += dl * dn; den += dl * dl
    return {"phi": num / den, "V5": Vt[:5], "cmean": means.mean(0)}


def features(prefix):
    """prefix: (t, 30) array of events strictly before t (in the given order). Returns m_t, last-m, recent4-m."""
    m = prefix.mean(0)
    return m, prefix[-1] - m, prefix[-RECENT:].mean(0) - m


def design_rows(Z, by, authors, t_from=T0):
    Xl, Xh, Y = [], [], []
    for a in authors:
        ix = by[a]
        for t in range(t_from, len(ix)):
            m, dl, dr = features(Z[ix[:t]])
            Xl.append(dl); Xh.append(np.r_[dl, dr]); Y.append(Z[ix[t]] - m)
    return np.array(Xl), np.array(Xh), np.array(Y)


def ridge_fixed(X, Y, lam=LAMBDA):
    """argmin_B (1/n)||Y - XB||_F^2 + lam ||B||_F^2  (loss SUMS output dims, AVERAGES rows) -> (X'X + n lam I) B = X'Y."""
    n = X.shape[0]
    return np.linalg.solve(X.T @ X + n * lam * np.eye(X.shape[1]), X.T @ Y)


def objective_grad(X, Y, B, lam=LAMBDA):
    n = X.shape[0]
    return -(2 / n) * X.T @ (Y - X @ B) + 2 * lam * B


def keyed_predict(prefix, pop, B_last, B_hist, w_last, w_hist, wk_old=0.5):
    """All predictions for the next event from its prefix only."""
    m, dl, dr = features(prefix)
    return {"global": np.zeros(30), "static": m, "pca5": pop["cmean"] + (m - pop["cmean"]) @ pop["V5"].T @ pop["V5"],
            "ar1": m + pop["phi"] * dl, "combined": m + wk_old * (pop["phi"] * dl),
            "coupled_last": m + w_last * (dl @ B_last), "coupled_hist": m + w_hist * (np.r_[dl, dr] @ B_hist)}


def permuted_prefix(prefix, role, author, t):
    """Deterministic permutation of the already-observed prefix only (no future rows)."""
    perm = np.random.default_rng([22009, ROLE_CODE[role], int(author), int(t)]).permutation(len(prefix))
    return prefix[perm]


def hist_readout(prefix, B_hist, w_hist, mean_override=None):
    m, dl, dr = features(prefix)
    if mean_override is not None:
        m_true = mean_override
        dl = prefix[-1] - m_true; dr = prefix[-RECENT:].mean(0) - m_true
        m = m_true
    return m + w_hist * (np.r_[dl, dr] @ B_hist)
