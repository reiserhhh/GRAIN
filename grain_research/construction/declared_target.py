"""GRAIN-C3-01: reusable declared-target construction (conditional response on recorded context).

A brief declares: an evidence functional per event (here cached keyed projections), a recorded
context rule mapping each event to c in {-1, +1}, windows and targets (choice composition, level,
contrast). Construction fits per-facet random-coefficient moments on source authors (F04 method
generalized to author-specific designs) and serializes a bundle; cold scoring reads only the
bundle and author event records (event id, venue, evidence vector). No text, labels or truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

VERSION = "GRAIN_declared_target_v1"
TARGETS = {"level": np.array([1.0, 0.0]), "contrast": np.array([0.0, 2.0]),
           "m_plus": np.array([1.0, 1.0]), "m_minus": np.array([1.0, -1.0])}


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def context_rule(venues_top):
    top = sorted(venues_top)
    return {"rule": "c=+1 if venue in TOP else -1", "top": top,
            "top_sha256": hashlib.sha256("\n".join(top).encode()).hexdigest()}


def contexts(venues, rule):
    top = set(rule["top"])
    return np.array([1.0 if v in top else -1.0 for v in venues])


def fit_moments(y_by_author, c_by_author, min_per_class=2):
    """y_by_author: list of (n_i, K) arrays; c_by_author: list of (n_i,) +-1. Per-facet moments."""
    k = y_by_author[0].shape[1]
    betas, inv_info, rss, df = [], [], np.zeros(k), 0
    for y, c in zip(y_by_author, c_by_author):
        if min(np.sum(c > 0), np.sum(c < 0)) < min_per_class:
            continue
        x = np.column_stack([np.ones(len(c)), c])
        xtx_inv = np.linalg.inv(x.T @ x)
        b = xtx_inv @ x.T @ y                      # (2, K)
        r = y - x @ b
        rss += np.sum(r ** 2, axis=0)
        df += len(c) - 2
        betas.append(b)
        inv_info.append(xtx_inv)
    betas = np.array(betas)                       # (A, 2, K)
    sigma2 = rss / df
    samp = np.mean(inv_info, axis=0)
    fits = []
    for j in range(k):
        bj = betas[:, :, j]
        cov = np.cov(bj, rowvar=False)
        raw = cov - sigma2[j] * samp
        vals, vecs = np.linalg.eigh((raw + raw.T) / 2)
        floor = 1e-6 * float(np.mean(np.diag(cov)))
        sig = (vecs * np.maximum(vals, floor)) @ vecs.T
        fits.append({"mu": bj.mean(0).tolist(), "Sigma": sig.tolist(), "sigma2": float(sigma2[j]),
                     "raw_eigenvalues": vals.tolist(), "floor": floor})
    return fits, {"authors_used": int(len(betas)), "noise_df": int(df)}


def blup(fit, y, c):
    """Information-form Gaussian posterior for one author and one facet (any design)."""
    mu, sig, s2 = np.asarray(fit["mu"]), np.asarray(fit["Sigma"]), fit["sigma2"]
    if len(c) == 0:
        return mu.copy(), sig.copy()
    x = np.column_stack([np.ones(len(c)), c])
    prec = np.linalg.inv(sig) + x.T @ x / s2
    cov = np.linalg.inv(prec)
    mean = cov @ (np.linalg.solve(sig, mu) + x.T @ y / s2)
    return mean, (cov + cov.T) / 2


def score_author(fits, y, c):
    """All facets and targets for one author; y (n, K), c (n,)."""
    out = {name: {"mean": [], "var": []} for name in TARGETS}
    for j, fit in enumerate(fits):
        m, v = blup(fit, y[:, j] if len(c) else np.zeros(0), c)
        for name, t in TARGETS.items():
            out[name]["mean"].append(float(t @ m))
            out[name]["var"].append(float(t @ v @ t))
    n_top, n = int(np.sum(c > 0)), len(c)
    out["choice_composition"] = {"mean": (n_top + 1) / (n + 2), "lo": float(beta_dist.ppf(.025, n_top + 1, n - n_top + 1)),
                                 "hi": float(beta_dist.ppf(.975, n_top + 1, n - n_top + 1))}
    out["support"] = ("observed" if min(n_top, n - n_top) >= 1 else "model_borrowed_contrast") if n else "no_events_prior_only"
    out["events"] = n
    return out


def build_bundle(spec, rule, fits, fit_info, reference, provenance):
    return {"version": VERSION, "code_hash": code_hash(), "spec": spec, "context_rule": rule, "fits": fits,
            "fit_info": fit_info, "reference": reference, "provenance": provenance,
            "interpretation": "keyed semantic projection units of published comments; conditional on author-chosen "
                              "venue class (descriptive); not a facet, personality or causal context score"}


def verify_bundle(b):
    if b.get("version") != VERSION or b.get("code_hash") != code_hash():
        raise ValueError("changed declared-target code or version")
    rule = b["context_rule"]
    if hashlib.sha256("\n".join(rule["top"]).encode()).hexdigest() != rule["top_sha256"]:
        raise ValueError("context rule tampered")
    if len(b["fits"]) != b["spec"]["facets"]:
        raise ValueError("facet count mismatch")


def cold_score(bundle, records):
    verify_bundle(bundle)
    k = bundle["spec"]["facets"]
    out = []
    for rec in records:
        if set(rec) != {"author_key", "events"}:
            raise ValueError("record must contain only author_key and events")
        ids = [e["event_id"] for e in rec["events"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event identity")
        for e in rec["events"]:
            if set(e) != {"event_id", "venue", "evidence"} or len(e["evidence"]) != k:
                raise ValueError("event must carry id, venue and the declared evidence vector only")
        y = np.array([e["evidence"] for e in rec["events"]], float).reshape(-1, k)
        c = contexts([e["venue"] for e in rec["events"]], bundle["context_rule"])
        out.append({"author_key": rec["author_key"], **score_author(bundle["fits"], y, c)})
    return out


def save_bundle(b, path):
    data = json.dumps(b, sort_keys=True).encode()
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load_bundle(path, sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("untrusted bundle hash")
    return json.loads(data)
