"""GRAIN-C2-01: plan-polarity x subject-qualification evidence unit and observation law.

Unit: designed pole b in {0,1} of one proposition ("the plan") under one of four subject
statuses (actual, quoted, hypothetical, negated-as-correction; actual/negated qualified).
Observation: reader category c = 2*1[pole>=.5] + 1[qualification>=.5] (fixed thresholds).
Observation law: E[b, status, c] per reader version, calibrated on the SOURCE family only.
Target in the finite world: pi_u = P(pole = 1 | qualified own report, author u).

Standard machinery (Dirichlet-multinomial channel, latent-class marginalisation, grid
posterior); nothing here is a personality unit. Material/variant IDs are not writer IDs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

STATUSES = ("actual", "quoted", "hypothetical", "negated")
QUALIFIED = ("actual", "negated")
UNQUALIFIED = ("quoted", "hypothetical")
FAMILY_OF_ROLE = {"source": "plain", "validation": "contextual", "application": "retrospective"}
PI_GRID = (np.arange(81) + 0.5) / 81
KAPPA_GRID = (np.arange(41) + 0.5) / 41
DRAWS = 200
SMOOTH = 0.5
VERSION = "GRAIN_C2_evidence_unit_v1"
READERS = ("lexical_tfidf_f08", "nli_deberta_v3_base_6c749ce")


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def category(pole_score, qual_score):
    p, q = np.asarray(pole_score, float), np.asarray(qual_score, float)
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise ValueError("finite reader scores required")
    return 2 * (p >= 0.5).astype(int) + (q >= 0.5).astype(int)


def channel_counts(rows):
    """Counts N[b, status, c] from rows with designed pole/status and reader category."""
    n = np.zeros((2, 4, 4))
    seen = set()
    for r in rows:
        if r["material_id"] in seen:
            raise ValueError("duplicate calibration material")
        seen.add(r["material_id"])
        n[int(r["pole"]), STATUSES.index(r["subject_status"]), int(r["category"])] += 1
    if (n.sum(-1) == 0).any():
        raise ValueError("every (pole,status) class needs calibration support")
    return n


def point_channel(n):
    return (n + SMOOTH) / (n + SMOOTH).sum(-1, keepdims=True)


def dirichlet_channels(n, seed, draws=DRAWS):
    rng = np.random.default_rng(seed)
    out = np.empty((draws, 2, 4, 4))
    for b in range(2):
        for s in range(4):
            out[:, b, s] = rng.dirichlet(n[b, s] + SMOOTH, size=draws)
    return out


def variant_bootstrap_channels(rows, seed, draws=DRAWS):
    """Paired cluster bootstrap: resample wrapper variants; each variant spans all classes."""
    rng = np.random.default_rng(seed)
    variants = sorted({int(r["variant"]) for r in rows})
    by_v = {v: [r for r in rows if int(r["variant"]) == v] for v in variants}
    out = np.empty((draws, 2, 4, 4))
    for d in range(draws):
        n = np.zeros((2, 4, 4))
        for v in rng.choice(variants, len(variants), replace=True):
            for r in by_v[v]:
                n[int(r["pole"]), STATUSES.index(r["subject_status"]), int(r["category"])] += 1
        out[d] = point_channel(n)
    return out


def category_probabilities(e):
    """p(c | pi, kappa) on the grid for one channel e[b, status, c]; declared selection law."""
    q = [STATUSES.index(s) for s in QUALIFIED]
    u = [STATUSES.index(s) for s in UNQUALIFIED]
    pi = PI_GRID[:, None, None]
    ka = KAPPA_GRID[None, :, None]
    qual = 0.5 * sum((1 - pi) * e[0, s][None, None, :] + pi * e[1, s][None, None, :] for s in q)
    unq = 0.5 * sum(0.5 * (e[0, s] + e[1, s])[None, None, :] for s in u)
    return ka * qual + (1 - ka) * unq


def log_probabilities(channels):
    return np.stack([np.log(np.maximum(category_probabilities(e), 1e-300)).reshape(-1, 4) for e in channels])


def marginal_posteriors(counts, channels=None, logps=None):
    """Mixture over channel draws of the (pi, kappa) marginal posteriors; counts (authors, 4)."""
    counts = np.asarray(counts, float)
    if logps is None:
        logps = log_probabilities(channels)
    uniq, inv = np.unique(counts, axis=0, return_inverse=True)
    acc_pi = np.zeros((len(uniq), len(PI_GRID)))
    acc_ka = np.zeros((len(uniq), len(KAPPA_GRID)))
    for logp in logps:
        ll = uniq @ logp.T
        ll -= ll.max(1, keepdims=True)
        w = np.exp(ll).reshape(len(uniq), len(PI_GRID), len(KAPPA_GRID))
        w /= w.sum((1, 2), keepdims=True)
        acc_pi += w.sum(2)
        acc_ka += w.sum(1)
    inv = np.asarray(inv).ravel()
    return (acc_pi / len(logps))[inv], (acc_ka / len(logps))[inv]


def summarize_grid(post):
    mean = post @ PI_GRID
    var = post @ PI_GRID ** 2 - mean ** 2
    cdf = np.cumsum(post, axis=1)
    lo = PI_GRID[np.argmax(cdf >= 0.025, axis=1)]
    hi = PI_GRID[np.argmax(cdf >= 0.975, axis=1)]
    return mean, np.maximum(var, 0), lo, hi


def beta_summary(pos, neg):
    a, b = 1 + np.asarray(pos, float), 1 + np.asarray(neg, float)
    return a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1)), beta_dist.ppf(0.025, a, b), beta_dist.ppf(0.975, a, b)


def naive_scores(counts):
    """Reader categories taken at face value: qualified = qualification bit, pole = pole bit."""
    c = np.asarray(counts)
    return beta_summary(c[:, 3], c[:, 1])


# ------------------------------------------------------------------ bundle + cold scoring
def build_bundle(rows, reader_id, seed, provenance):
    if reader_id not in READERS:
        raise ValueError("unknown reader version")
    if any(r["role"] != "source" for r in rows):
        raise ValueError("the frozen observation law is calibrated on source-family rows only")
    n = channel_counts(rows)
    return {
        "version": VERSION, "code_hash": code_hash(), "reader_id": reader_id,
        "unit": "plan polarity x subject qualification (actual/quoted/hypothetical/negated-as-correction)",
        "statuses": list(STATUSES), "qualified": list(QUALIFIED),
        "category_rule": "2*1[pole>=.5]+1[qualification>=.5]",
        "selection_law": "actual/negated kappa/2 each; quoted/hypothetical (1-kappa)/2 each; pole|unqualified Bern(.5)",
        "target": "pi = P(pole=1 | qualified own report)", "priors": "uniform pi, uniform kappa",
        "channel_counts": n.tolist(), "dirichlet_seed": seed, "draws": DRAWS, "smoothing": SMOOTH,
        "calibration_family": "plain (source)", "provenance": provenance,
    }


def verify_bundle(bundle):
    if bundle.get("version") != VERSION or bundle.get("code_hash") != code_hash():
        raise ValueError("changed evidence-unit code or version")
    n = np.asarray(bundle["channel_counts"])
    if n.shape != (2, 4, 4) or (n < 0).any() or (n.sum(-1) == 0).any():
        raise ValueError("invalid channel counts")
    return n


def records_to_counts(records, reader_id):
    """Each record: {author_id, reader_id, events: [{event_id, pole_score, qualification_score}]}."""
    out, status = [], []
    for rec in records:
        if set(rec) != {"author_id", "reader_id", "events"}:
            raise ValueError("unexpected input fields (no labels or truth allowed)")
        if rec["reader_id"] != reader_id:
            raise ValueError("reader version does not match the bundle's observation law")
        ids = [ev["event_id"] for ev in rec["events"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event identity")
        for ev in rec["events"]:
            if set(ev) != {"event_id", "pole_score", "qualification_score"}:
                raise ValueError("event must carry only reader scores and identity")
        c = np.zeros(4)
        if rec["events"]:
            cats = category([e["pole_score"] for e in rec["events"]], [e["qualification_score"] for e in rec["events"]])
            c = np.bincount(cats, minlength=4).astype(float)
        out.append(c)
        status.append("scored" if rec["events"] else "no_events")
    return np.array(out), status


def cold_score(bundle, records, mode="dirichlet"):
    n = verify_bundle(bundle)
    counts, status = records_to_counts(records, bundle["reader_id"])
    channels = point_channel(n)[None] if mode == "point" else dirichlet_channels(n, bundle["dirichlet_seed"], bundle["draws"])
    post_pi, post_ka = marginal_posteriors(counts, channels)
    mean, var, lo, hi = summarize_grid(post_pi)
    kappa_mean = post_ka @ KAPPA_GRID
    res = []
    for i, rec in enumerate(records):
        res.append({"author_id": rec["author_id"], "status": status[i], "pi_mean": float(mean[i]), "pi_var": float(var[i]),
                    "pi_lo": float(lo[i]), "pi_hi": float(hi[i]), "kappa_mean": float(kappa_mean[i]),
                    "events": int(counts[i].sum()),
                    "reader_qualified_events": int(counts[i, 1] + counts[i, 3]),
                    "interpretation": "prior only; NOT an observed neutral trait" if status[i] == "no_events"
                    else "posterior for plan-polarity propensity among qualified own reports"})
    return res


def save_bundle(bundle, path):
    data = json.dumps(bundle, sort_keys=True, indent=1).encode()
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load_bundle(path, sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("untrusted bundle hash")
    return json.loads(data)
