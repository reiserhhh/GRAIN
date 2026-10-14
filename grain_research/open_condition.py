#!/usr/bin/env python3
"""WP2: source-learned assignment of open condition clauses and its propagation
into the two-cell target kernel.

The instruments in ``contextual_instrument`` require each record to declare a
condition in {-1, +1}. Here the condition is *read* from the clause text: a
source-side assigner maps a clause representation to a posterior pi_e =
P(z=+1 | clause). The Gaussian readout is then evaluated with an expected
design row H_e = [1-pi_e, pi_e] and an inflated observation variance
R_e = (1-pi_e) r_- + pi_e r_+ + pi_e (1-pi_e) (mu_+ - mu_-)^2, which is the
standard soft-assignment (expected design / mixture variance) construction; the
new part is only its connection to the frozen readout, the bootstrap working
interval and an abstention rule.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from grain_research.contextual_instrument import PROFILE_TARGETS, gaussian_readout


def flip_mixture(probabilities, epsilon):
    """Symmetric contamination sensitivity model, not a calibrated truth posterior."""
    p = np.asarray(probabilities, float)
    if (
        not 0 <= epsilon <= 0.5
        or not np.all(np.isfinite(p))
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Finite probabilities and epsilon in [0, .5] required")
    return epsilon + (1 - 2 * epsilon) * p


def source_clause_error(vectors, conditions, groups):
    """Estimate errors by leave-one-clause-out, never by application labels.

    Use unique clause groups as the sampling units. Jeffreys smoothing avoids
    claiming zero error after a tiny, perfectly classified source sample.
    """
    x, z, g = np.asarray(vectors), np.asarray(conditions), np.asarray(groups)
    errors = []
    for group in np.unique(g):
        held = g == group
        if len(np.unique(z[held])) != 1 or set(np.unique(z[~held])) != {-1, 1}:
            raise ValueError("Each clause needs one label and both source classes")
        model = learn_prototype_assigner(x[~held], z[~held])
        errors.append(bool((assign(x[held], model).mean() >= 0.5) != (z[held][0] == 1)))
    return {
        "groups": len(errors),
        "errors": sum(errors),
        "epsilon": min(0.5, (sum(errors) + 0.5) / (len(errors) + 1)),
        "estimator": "leave-one-source-clause-out Jeffreys mean, capped .5",
    }


def learn_prototype_assigner(vectors, conditions, *, posterior="lda1d"):
    """Two class prototypes on unit-normalised clause vectors.

    posterior="isotropic": equal-prior isotropic Gaussian classes in the full
    space (recorded first attempt; underconfident in 768 dimensions).
    posterior="lda1d": project onto the prototype difference direction and fit a
    one-dimensional two-class Gaussian with pooled variance (Fisher's linear
    discriminant); the posterior is the standard logistic of the projected
    score. Neither family has tuned parameters.
    """
    x = np.asarray(vectors, float)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    z = np.asarray(conditions)
    if set(np.unique(z)) - {-1, 1}:
        raise ValueError("Source conditions must be -1/+1")
    mu = {c: x[z == c].mean(axis=0) for c in (-1, 1)}
    scale = float(
        np.mean([np.mean(np.sum((x[z == c] - mu[c]) ** 2, axis=1)) for c in (-1, 1)])
    )
    w = mu[1] - mu[-1]
    s = x @ w
    m = {c: float(s[z == c].mean()) for c in (-1, 1)}
    pooled = float(
        np.mean([s[z == c].var(ddof=1) if (z == c).sum() > 1 else 0.0 for c in (-1, 1)])
    )
    return {
        "posterior": posterior,
        "mu_minus": mu[-1].tolist(),
        "mu_plus": mu[1].tolist(),
        "scale": max(scale, 1e-6),
        "direction": w.tolist(),
        "proj_mean_minus": m[-1],
        "proj_mean_plus": m[1],
        "proj_pooled_var": max(pooled, 1e-8),
        "source_rows": len(x),
    }


def assign(vectors, model):
    """Posterior P(z=+1 | clause) for the recorded posterior family."""
    x = np.asarray(vectors, float)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    if model.get("posterior", "isotropic") == "isotropic":
        dm = np.sum((x - np.asarray(model["mu_minus"])) ** 2, axis=1)
        dp = np.sum((x - np.asarray(model["mu_plus"])) ** 2, axis=1)
        logit = (dm - dp) / (2 * model["scale"])
    else:
        s = x @ np.asarray(model["direction"])
        mid = (model["proj_mean_plus"] + model["proj_mean_minus"]) / 2
        logit = (
            (s - mid)
            * (model["proj_mean_plus"] - model["proj_mean_minus"])
            / model["proj_pooled_var"]
        )
    logit = np.clip(logit, -50, 50)
    return 1.0 / (1.0 + np.exp(-logit))


def gaussian_readout_full(mean, covariance, design, noise_matrix):
    """Same affine readout as ``gaussian_readout`` with a full observation covariance."""
    m, s, h, r = (
        np.asarray(a, float) for a in (mean, covariance, design, noise_matrix)
    )
    if r.shape != (len(h), len(h)) or not np.allclose(r, r.T):
        raise ValueError("Square symmetric observation covariance required")
    if np.linalg.eigvalsh(r).min() <= 0:
        raise ValueError("Positive definite observation covariance required")
    inner = h @ s @ h.T + r
    gain = np.linalg.solve(inner, h @ s).T
    post = s - gain @ h @ s
    return {
        "kernel": gain,
        "offset": m - gain @ h @ m,
        "covariance": (post + post.T) / 2,
    }


def assignment_variance(pis, fitted, *, full_moment=False):
    """Assignment noise from E[(theta_plus-theta_minus)^2].

    Under condition-assignment independence from theta, total covariance adds
    pi(1-pi) times BOTH the squared mean contrast and its prior variance.
    This is a moment-matched linear estimator, not an exact mixture posterior.
    """
    pi = np.asarray(pis, float)
    mean = np.asarray(fitted["mean"], float)
    contrast = float((mean[1] - mean[0]) ** 2)
    if full_moment:
        direction = np.array([-1.0, 1.0])
        contrast += max(
            0.0, float(direction @ np.asarray(fitted["covariance"]) @ direction)
        )
    return pi * (1 - pi) * contrast


def soft_profile_readout(values, pis, fitted, *, clause_ids=None, full_moment=False):
    """Expected-design readout for one person's events with condition posteriors.

    Without clause_ids the assignment uncertainty pi(1-pi)(mu_+ - mu_-)^2 is
    added per event (independent). With clause_ids the same term is shared by
    every event carrying the same clause (block-correlated), because a
    misassigned clause misassigns all of its events together.
    """
    y = np.asarray(values, float)
    pi = np.clip(np.asarray(pis, float), 1e-9, 1 - 1e-9)
    mean = np.asarray(fitted["mean"], float)
    noise = np.asarray(fitted["noise"], float)
    design = np.c_[1 - pi, pi]
    inflation = assignment_variance(pi, fitted, full_moment=full_moment)
    obs_noise = (1 - pi) * noise[0] + pi * noise[1]
    if clause_ids is None:
        result = gaussian_readout(
            mean, fitted["covariance"], design, obs_noise + inflation
        )
    else:
        ids = np.asarray(clause_ids)
        same = ids[:, None] == ids[None, :]
        r = np.diag(obs_noise) + same * np.sqrt(np.outer(inflation, inflation))
        result = gaussian_readout_full(mean, fitted["covariance"], design, r)
    return result, y


def score_person_open(
    values, pis, bundle, *, mode="soft", tau=0.2, clause_ids=None, full_moment=False
):
    """Score with soft, hard or abstaining assignment over the frozen bootstrap models."""
    y = np.asarray(values, float)
    pi = np.asarray(pis, float)
    ids = None if clause_ids is None else np.asarray(clause_ids)
    if mode == "hard":
        pi = (pi >= 0.5).astype(float)
    elif mode == "abstain":
        keep = np.abs(pi - 0.5) >= tau
        y, pi = y[keep], pi[keep]
        ids = None if ids is None else ids[keep]
    elif mode != "soft":
        raise ValueError(mode)
    if len(y) == 0:
        return {
            "status": "insufficient_qualified_evidence",
            "targets": {},
            "used_events": 0,
        }
    scores, variances = defaultdict(list), defaultdict(list)
    for fitted in bundle["bootstrap"]:
        result, yy = soft_profile_readout(
            y, pi, fitted, clause_ids=ids, full_moment=full_moment
        )
        for name, t in PROFILE_TARGETS.items():
            kernel = t @ result["kernel"]
            scores[name].append(float(t @ result["offset"] + kernel @ yy))
            variances[name].append(max(0.0, float(t @ result["covariance"] @ t)))
    answer = {}
    for name in PROFILE_TARGETS:
        s = float(np.mean(scores[name]))
        v = float(np.mean(variances[name])) + float(np.var(scores[name], ddof=1))
        answer[name] = {
            "score": s,
            "variance": v,
            "lower": s - 1.959963984540054 * np.sqrt(v),
            "upper": s + 1.959963984540054 * np.sqrt(v),
        }
    return {
        "status": "open_condition_working_measurement",
        "targets": answer,
        "used_events": len(y),
        "mode": mode,
        "tau": tau if mode == "abstain" else None,
        "clause_correlated": ids is not None,
    }
