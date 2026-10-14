#!/usr/bin/env python3
"""F15 bounded theory certificates: PSR/Hankel, Blackwell order, auxiliary ICA.

Every object is an exact finite computation on a declared law. Nothing here
learns a neural model or claims psychological identification.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
from scipy.special import expit
from scipy.stats import binom

from .f15 import garbling_certificate, mutual_information

# ---------------------------------------------------------------- A: PSR / Hankel


def markov_world(name):
    """Binary stationary finite-order worlds: C2 XOR, C2 iid, hidden order 3."""
    if name == "c2_xor":
        return 2, lambda w: 0.75 if (w[-2] ^ w[-1]) == 1 else 0.25
    if name == "c2_iid":
        return 2, lambda w: 0.5
    if name == "hidden_order3":
        return 3, lambda w: 0.8 if w[-3] == 1 else 0.2
    raise ValueError("unknown world")


def stationary_windows(order, p_one):
    """Stationary distribution over length-`order` windows (exact linear solve)."""
    states = list(itertools.product((0, 1), repeat=order))
    index = {s: i for i, s in enumerate(states)}
    t = np.zeros((len(states), len(states)))
    for s in states:
        p = p_one(s)
        for y, q in ((1, p), (0, 1 - p)):
            t[index[s], index[(*s[1:], y)]] += q
    a = np.vstack([t.T - np.eye(len(states)), np.ones(len(states))])
    b = np.r_[np.zeros(len(states)), 1.0]
    pi = np.linalg.lstsq(a, b, rcond=None)[0]
    if np.abs(pi @ t - pi).max() > 1e-13 or pi.min() < -1e-15:
        raise ValueError("stationary solve failed")
    return {s: float(pi[index[s]]) for s in states}


@functools.cache
def _stationary(world):
    """Cached stationary window law of a named world."""
    order, p_one = markov_world(world)
    return stationary_windows(order, p_one)


@functools.cache
def sequence_probability(seq, world):
    """Exact stationary probability of a finite observed string."""
    order, p_one = markov_world(world)
    pi = _stationary(world)
    seq = tuple(seq)
    if len(seq) <= order:
        return (
            sum(v for w, v in pi.items() if w[order - len(seq) :] == seq)
            if seq
            else 1.0
        )
    prob = pi[seq[:order]]
    for t in range(order, len(seq)):
        p = p_one(seq[t - order : t])
        prob *= p if seq[t] == 1 else 1 - p
    return prob


def strings(max_len, min_len=0):
    """All binary strings with length in [min_len, max_len]."""
    return [
        s
        for n in range(min_len, max_len + 1)
        for s in itertools.product((0, 1), repeat=n)
    ]


def hankel(world, history_len, test_len):
    """System-dynamics block D[h,t]=P(t|h) (Singh, James & Rudary 2004)."""
    histories = strings(history_len)
    tests = strings(test_len, 1)
    d = np.array(
        [
            [
                sequence_probability(h + t, world) / sequence_probability(h, world)
                for t in tests
            ]
            for h in histories
        ]
    )
    return histories, tests, d


def numerical_rank(matrix):
    """Rank with the standard SVD tolerance max(shape)*eps*smax."""
    s = np.linalg.svd(matrix, compute_uv=False)
    tol = max(matrix.shape) * np.finfo(float).eps * (s[0] if len(s) else 0)
    return int((s > tol).sum()), s.tolist(), tol


def core_tests(d, tests):
    """Greedy linearly independent column basis (core tests)."""
    chosen = []
    for j in range(d.shape[1]):
        trial = d[:, [*chosen, j]]
        if np.linalg.matrix_rank(trial) > len(chosen):
            chosen.append(j)
    return chosen


def fit_linear_psr(world, history_len, test_len):
    """Linear PSR from a finite Hankel block: core tests and projection vectors."""
    histories, tests, d = hankel(world, history_len, test_len)
    core = core_tests(d, tests)
    q = d[:, core]
    needed = {t for t in tests if len(t) <= 1} | {
        (o, *tests[c]) for o in (0, 1) for c in core
    }
    extended_tests = sorted(set(tests) | needed, key=lambda s: (len(s), s))
    col = []
    for t in extended_tests:
        col.append(
            [
                sequence_probability(h + t, world) / sequence_probability(h, world)
                for h in histories
            ]
        )
    col = np.array(col).T
    m, *_ = np.linalg.lstsq(q, col, rcond=None)
    fit_error = float(np.abs(q @ m - col).max())
    return {
        "histories": histories,
        "core_tests": [tests[c] for c in core],
        "q_empty": q[0],
        "m": {t: m[:, i] for i, t in enumerate(extended_tests)},
        "fit_error": fit_error,
        "rank": len(core),
    }


def psr_predict(model, history, test):
    """Predict P(test|history) by iterated linear PSR updates from the empty history."""
    state = model["q_empty"].copy()
    for o in history:
        denom = state @ model["m"][(o,)]
        if denom <= 0:
            return np.nan
        state = (
            np.array([state @ model["m"][(o, *c)] for c in model["core_tests"]]) / denom
        )
    prob = 1.0
    for o in test:
        p = state @ model["m"][(o,)]
        prob *= p
        if p <= 0:
            return prob
        state = np.array([state @ model["m"][(o, *c)] for c in model["core_tests"]]) / p
    return prob


def closure_error(model, world, max_history, test_len):
    """Max |PSR prediction - exact law| over all histories/tests up to the bounds."""
    worst, count = 0.0, 0
    for h in strings(max_history):
        ph = sequence_probability(h, world)
        for t in strings(test_len, 1):
            exact = sequence_probability(h + t, world) / ph
            worst = max(worst, abs(psr_predict(model, h, t) - exact))
            count += 1
    return worst, count


def memory_term(world, short, long):
    """Mori-Zwanzig style residual P(y|long history)-P(y|last `short`)."""
    rows = []
    for h in strings(long, long):
        full = sequence_probability((*h, 1), world) / sequence_probability(h, world)
        tail = h[len(h) - short :] if short else ()
        proj = sequence_probability((*tail, 1), world) / sequence_probability(
            tail, world
        )
        rows.append((h, sequence_probability(h, world), full - proj))
    weights = np.array([r[1] for r in rows])
    values = np.array([r[2] for r in rows])
    mean = float(weights @ values)
    var = float(weights @ (values - mean) ** 2)
    return {
        "rows": [(list(h), w, v) for h, w, v in rows],
        "weighted_mean": mean,
        "weighted_variance": var,
    }


# ---------------------------------------------------------- B: Blackwell / garbling


def f09_family_channel(reps=3):
    """9 hypotheses (family k x beta) x full even-corner count outcomes."""
    corners = [c for c in itertools.product((-1, 1), repeat=3) if np.prod(c) == 1]
    hyps = [(k, b) for k in (0, 1, 2) for b in (0.5, 1.0, 1.5)]
    outcomes = list(itertools.product(range(reps + 1), repeat=len(corners)))
    a = np.zeros((len(hyps), len(outcomes)))
    for i, (k, b) in enumerate(hyps):
        p = [expit(b * c[k]) for c in corners]
        for j, n in enumerate(outcomes):
            a[i, j] = np.prod(
                [binom.pmf(n_c, reps, p_c) for n_c, p_c in zip(n, p, strict=True)]
            )
    return {"hypotheses": hyps, "outcomes": outcomes, "cells": corners, "channel": a}


def f04_binary_channel(reps=2):
    """Constructed binary analog of F04 worlds L/T/Q on the four-condition design."""
    z = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
    worlds = {"L": z, "T": (z > 0).astype(float) * 2 - 1, "Q": 2 * z**2 - 1}
    outcomes = list(itertools.product(range(reps + 1), repeat=4))
    hyps = list(worlds)
    a = np.zeros((len(hyps), len(outcomes)))
    for i, w in enumerate(hyps):
        p = expit(1.5 * worlds[w])
        for j, n in enumerate(outcomes):
            a[i, j] = np.prod(
                [binom.pmf(n_c, reps, p_c) for n_c, p_c in zip(n, p, strict=True)]
            )
    return {"hypotheses": hyps, "outcomes": outcomes, "cells": z.tolist(), "channel": a}


def deterministic_garble(outcomes, statistic):
    """Row-stochastic 0/1 kernel from full outcomes to a statistic's values."""
    values = sorted({statistic(o) for o in outcomes})
    k = np.zeros((len(outcomes), len(values)))
    for i, o in enumerate(outcomes):
        k[i, values.index(statistic(o))] = 1
    return values, k


def minimal_sufficient(channel, tol=1e-12):
    """Group outcomes with proportional likelihood columns (minimal sufficient)."""
    cols = channel / channel.sum(0, keepdims=True)
    labels, keys = [], []
    for j in range(channel.shape[1]):
        match = next(
            (i for i, key in enumerate(keys) if np.abs(key - cols[:, j]).max() <= tol),
            None,
        )
        if match is None:
            keys.append(cols[:, j])
            match = len(keys) - 1
        labels.append(match)
    k = np.zeros((channel.shape[1], len(keys)))
    k[np.arange(channel.shape[1]), labels] = 1
    return k


def blackwell_comparison(spec, target_groups):
    """Garbling certificates and MI for full / mean / minimal-sufficient channels."""
    a = spec["channel"]
    _, k_mean = deterministic_garble(spec["outcomes"], sum)
    b = a @ k_mean
    k_min = minimal_sufficient(a)
    s = a @ k_min
    certs = {
        "mean_from_full": garbling_certificate(a, b),
        "full_from_mean": garbling_certificate(b, a),
        "minimal_from_full": garbling_certificate(a, s),
        "full_from_minimal": garbling_certificate(s, a),
    }
    info = {}
    n = a.shape[0]
    for label, groups in target_groups.items():
        g = np.array(groups)
        classes = sorted(set(groups))
        prior = np.full(len(classes), 1 / len(classes))
        for name, ch in (("full", a), ("mean", b), ("minimal", s)):
            merged = np.array([ch[g == c].mean(0) for c in classes])
            info[f"{label}:{name}"] = mutual_information(merged, prior)
    info["hypothesis:full"] = mutual_information(a, np.full(n, 1 / n))
    info["hypothesis:mean"] = mutual_information(b, np.full(n, 1 / n))
    info["hypothesis:minimal"] = mutual_information(s, np.full(n, 1 / n))
    return {
        "dimensions": {"full": a.shape[1], "mean": b.shape[1], "minimal": s.shape[1]},
        "certificates": certs,
        "mutual_information_nats": info,
    }


# ------------------------------------------------- C: auxiliary-variable identification

CONDITIONS = [(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (0, 0, 2, 1), (0, 0, 1, 2)]


def natural_parameters(mu1, mu2, v1, v2):
    """Gaussian natural parameters (mu/v, -1/(2v)) per source."""
    return np.array([mu1 / v1, -1 / (2 * v1), mu2 / v2, -1 / (2 * v2)])


def auxiliary_rank(conditions):
    """Rank of L=(eta(u_j)-eta(u_0)) (Khemakhem et al. 2020, Thm 1(iv))."""
    eta = np.array([natural_parameters(*c) for c in conditions])
    diff = (eta[1:] - eta[0]).T
    rank, sv, _ = numerical_rank(diff) if diff.size else (0, [], 0.0)
    return {
        "shape": list(diff.shape),
        "rank": rank,
        "singular_values": sv,
        "required_rank": 4,
    }


def variability_rank(conditions, y):
    """Hyvarinen et al. 2019 Thm 1 assumption 3 vectors at a point y (2n=4 dims)."""
    rows = []
    for mu1, mu2, v1, v2 in conditions:
        rows.append([-(y[0] - mu1) / v1, -(y[1] - mu2) / v2, -1 / v1, -1 / v2])
    w = np.array(rows)
    diff = (w[1:] - w[0]).T
    return numerical_rank(diff)[0] if diff.size else 0


def mix(s):
    """Declared invertible mixing x=(s1+.2 s2^3, s2)."""
    s = np.asarray(s)
    return np.column_stack([s[:, 0] + 0.2 * s[:, 1] ** 3, s[:, 1]])


def unmix(x):
    """Analytic inverse (a known inverse, not a learned one)."""
    x = np.asarray(x)
    return np.column_stack([x[:, 0] - 0.2 * x[:, 1] ** 3, x[:, 1]])


def sample_world(seed, role, n, null=False):
    """Equal-frequency auxiliary u; sources Gaussian given u (null: all u identical)."""
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                seed,
                15,
                3,
                {"source": 0, "validation": 1, "application": 2}[role],
                int(null),
            ]
        )
    )
    u = np.repeat(np.arange(5), n // 5)
    rng.shuffle(u)
    params = np.array([CONDITIONS[0] if null else CONDITIONS[k] for k in u], float)
    s = params[:, :2] + rng.standard_normal((len(u), 2)) * np.sqrt(params[:, 2:])
    return u, s, mix(s), rng


def contrastive_design(u, x, rng, features="quadratic_x"):
    """Positive true (x,u) pairs vs within-role permuted u; frozen feature map."""
    base = unmix(x) if features == "quadratic_true_s" else x

    def phi(v):
        return np.column_stack(
            [v[:, 0], v[:, 1], v[:, 0] ** 2, v[:, 1] ** 2, v[:, 0] * v[:, 1]]
        )

    def design(uu):
        onehot = np.eye(5)[uu]
        f = phi(base)
        return np.column_stack(
            [f, onehot, (f[:, :, None] * onehot[:, None, :]).reshape(len(uu), -1)]
        )

    perm = rng.permutation(u)
    xmat = np.vstack([design(u), design(perm)])
    label = np.r_[np.ones(len(u)), np.zeros(len(u))]
    return xmat, label, perm


def kernel_certificate(a, b, k):
    """Exact check that an explicit row-stochastic K satisfies B = A K."""
    k = np.asarray(k, float)
    residual = float(np.abs(a @ k - b).max())
    rows = float(np.abs(k.sum(1) - 1).max())
    ok = residual <= 1e-12 and rows <= 1e-12 and float(k.min()) >= 0
    return {
        "status": "EXACT_KERNEL" if ok else "KERNEL_FAILED",
        "max_equation_residual": residual,
        "max_row_sum_error": rows,
        "min_kernel": float(k.min()),
    }


def reverse_sufficient_kernel(a, k_min):
    """Kernel from minimal-sufficient classes back to full outcomes."""
    s = a @ k_min
    rev = np.zeros((k_min.shape[1], a.shape[1]))
    for o in range(a.shape[1]):
        c = int(np.argmax(k_min[o]))
        rev[c, o] = a[:, o].sum() / s[:, c].sum()
    return rev


def blackwell_certificates(spec, target_groups, run_lp=True):
    """Explicit kernels for feasible directions, LP/Farkas for all (retained)."""
    a = spec["channel"]
    _, k_mean = deterministic_garble(spec["outcomes"], sum)
    b = a @ k_mean
    k_min = minimal_sufficient(a)
    s = a @ k_min
    explicit = {
        "mean_from_full": kernel_certificate(a, b, k_mean),
        "minimal_from_full": kernel_certificate(a, s, k_min),
        "full_from_minimal": kernel_certificate(
            s, a, reverse_sufficient_kernel(a, k_min)
        ),
    }
    lp = {}
    if run_lp:
        for name, (x, y) in {
            "mean_from_full": (a, b),
            "full_from_mean": (b, a),
            "minimal_from_full": (a, s),
            "full_from_minimal": (s, a),
        }.items():
            cert = garbling_certificate(x, y)
            lp[name] = {
                k: v for k, v in cert.items() if k not in ("kernel", "farkas_y")
            }
            if "farkas_y" in cert:
                lp[name]["farkas_y_length"] = len(cert["farkas_y"])
    mi = {}
    n = a.shape[0]
    for label, groups in target_groups.items():
        g = np.array(groups)
        classes = sorted(set(groups))
        prior = np.full(len(classes), 1 / len(classes))
        for name, ch in (("full", a), ("mean", b), ("minimal", s)):
            merged = np.array([ch[g == c].mean(0) for c in classes])
            mi[f"{label}:{name}"] = mutual_information(merged, prior)
    for name, ch in (("full", a), ("mean", b), ("minimal", s)):
        mi[f"hypothesis:{name}"] = mutual_information(ch, np.full(n, 1 / n))
    return {
        "dimensions": {"full": a.shape[1], "mean": b.shape[1], "minimal": s.shape[1]},
        "explicit_kernels": explicit,
        "lp": lp,
        "mutual_information_nats": mi,
    }


def log_density_ratio(u, x):
    """Exact log p(x|u) - log p(x) for the declared world (unit Jacobian)."""
    s = unmix(x)

    def logpdf(k):
        mu1, mu2, v1, v2 = CONDITIONS[k]
        return (
            -0.5 * ((s[:, 0] - mu1) ** 2 / v1 + (s[:, 1] - mu2) ** 2 / v2)
            - 0.5 * np.log(v1 * v2)
            - np.log(2 * np.pi)
        )

    all_k = np.stack([logpdf(k) for k in range(5)])
    marginal = np.logaddexp.reduce(all_k, axis=0) - np.log(5)
    return all_k[u, np.arange(len(u))] - marginal


def contrastive_run(
    seed,
    arm,
    sizes=(("source", 3000), ("validation", 1000), ("application", 2000)),
    return_losses=False,
):
    """Fixed sklearn contrastive discriminator; source-only fit; role-internal permutations."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    null = arm == "null_world"
    features = "quadratic_true_s" if arm == "quadratic_true_s" else "quadratic_x"
    data = {}
    for role, n in sizes:
        u, _s, x, rng = sample_world(seed, role, n, null=null)
        xmat, label, perm = contrastive_design(u, x, rng, features)
        data[role] = (u, x, xmat, label, perm)
    scaler = StandardScaler().fit(data["source"][2])
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(scaler.transform(data["source"][2]), data["source"][3])
    rows = []
    for role in ("validation", "application"):
        u, x, xmat, label, perm = data[role]
        prob = model.predict_proba(scaler.transform(xmat))[:, 1]
        clipped = np.clip(prob, 1e-6, 1 - 1e-6)
        row = {
            "seed": seed,
            "arm": arm,
            "role": role,
            "n_pairs": len(label),
            "accuracy": float(np.mean((prob >= 0.5) == label)),
            "log_loss": float(log_loss(label, clipped)),
            "clip_count": int(np.sum((prob < 1e-6) | (prob > 1 - 1e-6))),
            "auc": float(roc_auc_score(label, prob)),
            "chance_log_loss": float(np.log(2)),
            "converged": bool(model.n_iter_[0] < 1000),
        }
        if return_losses:
            half = len(label) // 2
            pos, neg = clipped[:half], clipped[half:]
            row["pair_losses"] = -(np.log(pos) + np.log(1 - neg)) / 2
        if not null:
            # Bayes reference: positives use true u; negatives use their permuted u.
            logits = np.r_[log_density_ratio(u, x), log_density_ratio(perm, x)]
            bayes = np.clip(1 / (1 + np.exp(-logits)), 1e-12, 1 - 1e-12)
            row["bayes_log_loss"] = float(log_loss(label, bayes))
            row["bayes_accuracy"] = float(np.mean((bayes >= 0.5) == label))
        rows.append(row)
    return rows


# ------------------------------------- B2: F09 process sub-world (seminar-01 addition)

GAMMA = 1.2


def belief_report(beta, z):
    """Belief process: B~Bern(logistic(beta z)), report P(Y=1|B)=logistic(gamma(1-2B))."""
    pb = expit(beta * z)
    return pb * expit(-GAMMA) + (1 - pb) * expit(GAMMA)


def direct_slope(beta):
    """Protocol beta': no-intercept logistic matching the belief law at z=+-1."""
    from scipy.special import logit

    return float(logit(belief_report(beta, 1.0)))


def process_atoms():
    """Belief(beta) and its corner-equivalent direct response for beta in {.5,1,1.5}."""
    return [(proc, beta) for beta in (0.5, 1.0, 1.5) for proc in ("belief", "direct")]


def _trial_law(proc, beta, z, observe_b):
    """Per-trial outcome probabilities; with B observed outcomes are (B,Y) pairs."""
    if not observe_b:
        p = (
            belief_report(beta, z)
            if proc == "belief"
            else expit(direct_slope(beta) * z)
        )
        return np.array([1 - p, p])
    pb = expit(beta * z)
    if proc == "belief":
        py1 = np.array([expit(GAMMA), expit(-GAMMA)])  # given B=0, B=1
    else:
        py1 = np.full(2, expit(direct_slope(beta) * z))  # declared: B independent of Y
    # order (B,Y): (0,0),(0,1),(1,0),(1,1)
    return np.array(
        [(1 - pb) * (1 - py1[0]), (1 - pb) * py1[0], pb * (1 - py1[1]), pb * py1[1]]
    )


def process_experiment(levels, reps, observe_b=False):
    """Channel atom x outcome for reps trials at each level (counts per category)."""
    atoms = process_atoms()
    k = 4 if observe_b else 2
    per_level = [
        c for c in itertools.product(range(reps + 1), repeat=k) if sum(c) == reps
    ]
    outcomes = list(itertools.product(per_level, repeat=len(levels)))
    from scipy.stats import multinomial

    channel = np.zeros((len(atoms), len(outcomes)))
    for i, (proc, beta) in enumerate(atoms):
        laws = [_trial_law(proc, beta, z, observe_b) for z in levels]
        for j, outcome in enumerate(outcomes):
            channel[i, j] = np.prod(
                [
                    multinomial.pmf(c, reps, law)
                    for c, law in zip(outcome, laws, strict=True)
                ]
            )
    return {
        "atoms": atoms,
        "outcomes": outcomes,
        "channel": channel,
        "levels": list(levels),
        "observe_b": observe_b,
    }


def marginal_kernel(outcomes_big, outcomes_small, project):
    """Deterministic kernel mapping big-experiment outcomes to the smaller experiment."""
    index = {o: i for i, o in enumerate(outcomes_small)}
    k = np.zeros((len(outcomes_big), len(outcomes_small)))
    for i, o in enumerate(outcomes_big):
        k[i, index[project(o)]] = 1
    return k


def process_blackwell(reps=2):
    """E1 corners only, E2 corners+intermediate levels, E3 corners with belief B observed."""
    e1 = process_experiment((1.0, -1.0), reps)
    e2 = process_experiment((1.0, -1.0, 1 / 3, -1 / 3), reps)
    e3 = process_experiment((1.0, -1.0), reps, observe_b=True)
    k21 = marginal_kernel(e2["outcomes"], e1["outcomes"], lambda o: o[:2])
    k31 = marginal_kernel(
        e3["outcomes"],
        e1["outcomes"],
        lambda o: tuple((c[0] + c[2], c[1] + c[3]) for c in o),
    )
    certs = {
        "E1_from_E2_explicit": kernel_certificate(e2["channel"], e1["channel"], k21),
        "E1_from_E3_explicit": kernel_certificate(e3["channel"], e1["channel"], k31),
        "E2_from_E1_lp": {
            k: v
            for k, v in garbling_certificate(e1["channel"], e2["channel"]).items()
            if k not in ("kernel", "farkas_y")
        },
        "E3_from_E1_lp": {
            k: v
            for k, v in garbling_certificate(e1["channel"], e3["channel"]).items()
            if k not in ("kernel", "farkas_y")
        },
    }
    proc = np.array([a[0] for a in e1["atoms"]])
    beta = np.array([a[1] for a in e1["atoms"]])
    mi = {}
    for name, ch in (
        ("E1", e1["channel"]),
        ("E2", e2["channel"]),
        ("E3", e3["channel"]),
    ):
        for label, groups in (("process", proc), ("beta", beta)):
            classes = sorted(set(groups.tolist()))
            merged = np.array([ch[groups == c].mean(0) for c in classes])
            mi[f"{label}:{name}"] = mutual_information(
                merged, np.full(len(classes), 1 / len(classes))
            )
    corner_gap = max(
        abs(belief_report(b, z) - expit(direct_slope(b) * z))
        for b in (0.5, 1.0, 1.5)
        for z in (1.0, -1.0)
    )
    middle_gap = {
        b: max(
            abs(belief_report(b, z) - expit(direct_slope(b) * z))
            for z in (1 / 3, -1 / 3)
        )
        for b in (0.5, 1.0, 1.5)
    }
    return {
        "dimensions": {
            "E1": e1["channel"].shape[1],
            "E2": e2["channel"].shape[1],
            "E3": e3["channel"].shape[1],
        },
        "certificates": certs,
        "mutual_information_nats": mi,
        "corner_law_max_gap": corner_gap,
        "intermediate_law_gap_by_beta": middle_gap,
        "declared": "direct process has no latent belief; in E3 its B is declared independent of Y with the same Bernoulli(logistic(beta z)) law",
    }


# ------------------------------------- D2: F10 geometry recomputation (seminar-01 addition)


def f10_geometry_recompute(seeds=(20260915, 20260916, 20260917)):
    """Re-execute the frozen F10 quadratic-world design via f10_geometry functions."""
    from .f10_geometry import (
        action_subspace,
        edit_jacobian,
        project_edits,
        quadratic_response,
    )

    cov = [0.04, 0.09, 4.0, 9.0]
    rows, geometry_rows = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        edits = rng.normal(size=(600, 4)) * np.sqrt(cov)
        c = np.cov(edits[:300], rowvar=False)
        h0, a = np.zeros(4), np.eye(4)
        u = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        _, pca = np.linalg.eigh(c)
        pca = pca[:, -2:]
        for world in ("linear", "null_curvature"):
            b = np.zeros((2, 4, 4))
            if world == "null_curvature":
                b[0, 2, 2], b[1, 3, 3] = 0.1, 0.05
            j = edit_jacobian(np.zeros(4), h0, a, u, b)
            geo = action_subspace(j, c)
            v, m = geo["basis"], geo["denominator"]
            observed = edits[300:]
            full = quadratic_response(observed, h0, a, u, b)
            null = geo["nullspace"]
            second_order = float(
                np.mean(
                    np.sum(
                        (quadratic_response(observed @ null @ null.T, h0, a, u, b))
                        ** 2,
                        axis=1,
                    )
                )
            )
            geometry_rows.append(
                {
                    "seed": seed,
                    "world": world,
                    "rank": geo["rank"],
                    "null_dimension": null.shape[1],
                    "null_direction_second_order_mean_square": second_order,
                }
            )
            for method, rec in (
                ("zero", np.zeros_like(observed)),
                ("pca_rank2", project_edits(observed, pca)),
                ("action_rank2", project_edits(observed, v, m)),
                ("full_chart", observed),
            ):
                pred = quadratic_response(rec, h0, a, u, b)
                for k in range(300):
                    rows.append(
                        {
                            "seed": seed,
                            "world": world,
                            "role": "validation" if k < 100 else "application",
                            "method": method,
                            "se": float(np.sum((full[k] - pred[k]) ** 2)),
                        }
                    )
    return rows, geometry_rows
