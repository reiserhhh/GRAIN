"""Frozen rival families, process equivalence classes and finite posterior."""

import itertools

import numpy as np
from scipy.special import expit, logit, logsumexp

BETAS = (0.5, 1.0, 1.5)
CORNERS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
EVEN = CORNERS[np.prod(CORNERS, axis=1) == 1]
ODD = CORNERS[np.prod(CORNERS, axis=1) == -1]


def probability(z, family, beta):
    """Observed response law; belief is a two-stage process, not direct logistic."""
    z = np.asarray(z, dtype=float)
    if z.shape[-1] != 3 or not np.isfinite(z).all():
        raise ValueError("three finite coordinates required")
    if family in ("M1", "M2", "M3"):
        return expit(beta * z[..., int(family[-1]) - 1])
    if family == "M4":
        return expit(beta * z.sum(axis=-1) / np.sqrt(3))
    if family == "audience":
        return expit(beta * z[..., 1])
    if family == "belief":
        return expit(1.2) + (expit(-1.2) - expit(1.2)) * expit(beta * z[..., 0])
    if family == "negative_M1":
        slope = logit(expit(1.2) + (expit(-1.2) - expit(1.2)) * expit(beta))
        return expit(slope * z[..., 0])
    raise ValueError("unknown declared rival")


def candidates(process=False):
    """Return nine main atoms or18 process atoms, with uniform class prior."""
    families = (
        ("M1", "M2", "M3", "belief", "audience", "negative_M1")
        if process
        else ("M1", "M2", "M3")
    )
    atoms = [(family, beta) for family in families for beta in BETAS]
    keys = [
        ("M2" if f == "audience" else "belief" if f == "negative_M1" else f, b)
        for f, b in atoms
    ]
    unique = list(dict.fromkeys(keys))
    classes = np.array([unique.index(k) for k in keys])
    prior = np.array([1 / (len(unique) * keys.count(k)) for k in keys])
    return {"atoms": atoms, "classes": classes, "class_keys": unique, "prior": prior}


def fit_family(z, y, process=False):
    """Compute exact posterior from observed binary outcomes; no hidden family input."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if z.shape != (len(y), 3) or not np.isin(y, [0, 1]).all():
        raise ValueError("aligned binary observations required")
    specification = candidates(process)
    p = np.array([probability(z, f, b) for f, b in specification["atoms"]])
    logweight = np.log(specification["prior"]) + (
        y * np.log(p) + (1 - y) * np.log1p(-p)
    ).sum(axis=1)
    posterior = np.exp(logweight - logsumexp(logweight))
    return {
        **specification,
        "posterior": posterior,
        "class_posterior": np.bincount(
            specification["classes"],
            weights=posterior,
            minlength=len(specification["class_keys"]),
        ),
    }


def predict_family(fitted, z):
    """Posterior prediction at new conditions with unchanged candidate atoms."""
    return fitted["posterior"] @ np.array(
        [probability(z, f, b) for f, b in fitted["atoms"]]
    )
