"""R26/A07 recovery-parameter observation design (packet 25).

Environment, streams and inference reuse F06 gaussian_full_r1 exactly (f06_gaussian_design.fit_prefix, gaussian_terms,
SeedSequence streams). Three arms at matched observation AND intervention quota (2 actions per 8-observation block,
only in within-block transition slots t mod 8 in 0..6):
  fixed      - alternating z, actions at within-block offsets 2 and 5;
  scalar_a   - scalar-level adaptive REFERENCE (minimise predicted Var(a)); not ALBA/IRT;
  recovery   - recovery-target adaptive (minimise predicted Var(rho), rho = .99 tanh(eta), delta method).
Shared criterion: expected Gaussian Fisher information at the current MAP psi for the prefix extended by the candidate
next observation (z', h_t) and ONE declared neutral continuation observation (-z', no further action), with BOTH mean
and covariance (eta) terms. Candidates only change the mean path, so the covariance term is candidate-invariant here.
Policies see observed prefixes, the frozen prior and the model only; truth is used by the environment and scoring only.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from grain_research.gptpro_foundation_v3.f06_gaussian_design import fit_prefix, gaussian_terms, source_prior

ROLE_INDEX_APPLICATION = 2
EVENTS, BLOCK, QUOTA = 24, 8, 2
FIXED_OFFSETS = (2, 5)
CANDIDATES = ((-1.0, 0.0), (-1.0, 1.0), (1.0, 0.0), (1.0, 1.0))
ARMS = ("fixed", "scalar_a", "recovery")


def author_streams(seed, ordinal, role="application", role_index=ROLE_INDEX_APPLICATION, events=EVENTS):
    """Identical SeedSequence construction to run_grain_gptpro_f06_gaussian_design.run_author."""
    author = "u_" + hashlib.sha256(f"F06B/{seed}/{role}/{ordinal}".encode()).hexdigest()[:16]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 6, 1, role_index, ordinal, 0]))
    a, b, rho, beta = rng.normal(), rng.normal(0.5, 0.5), rng.uniform(0.2, 0.8), rng.normal(0.4, 0.2)
    initial = np.random.default_rng(np.random.SeedSequence([seed, 6, 1, role_index, ordinal, 1])).normal(0, 0.5 / np.sqrt(1 - rho ** 2))
    innovations = np.random.default_rng(np.random.SeedSequence([seed, 6, 1, role_index, ordinal, 2])).normal(0, 0.5, events)
    measurement = np.random.default_rng(np.random.SeedSequence([seed, 6, 1, role_index, ordinal, 3])).normal(0, 0.5, events)
    return {"author_id": author, "a": a, "b": b, "rho": rho, "beta": beta, "initial": initial, "innovations": innovations, "measurement": measurement}


def fisher(psi, z, h, q, r):
    """Expected Gaussian information at psi: mean term D'S^-1 D plus the eta covariance term."""
    terms = gaussian_terms(psi, np.asarray(z, float), np.asarray(h, float), q, r)
    factor = cho_factor(terms["covariance"], lower=True)
    F = terms["mean_derivatives"].T @ cho_solve(factor, terms["mean_derivatives"])
    A = cho_solve(factor, terms["covariance_eta"])
    F_cov = 0.5 * np.trace(A @ A)
    F[2, 2] += F_cov
    return F, float(F_cov), terms


def target_variance(psi, precision, z, h, q, r, target):
    F, F_cov, terms = fisher(psi, z, h, q, r)
    C = np.linalg.inv(precision + F)
    if target == "rho":
        return float(terms["drho"] ** 2 * C[2, 2]), F_cov
    if target == "a":
        return float(C[0, 0]), F_cov
    raise ValueError("unknown target")


def allowed_actions(t, h_history):
    """Quota rule: exactly QUOTA actions per block, only at within-block slots t mod 8 in 0..6; never borrow."""
    o = t % BLOCK
    if o == BLOCK - 1 or t >= EVENTS - 1:
        return (0.0,)
    start = t - o
    used = int(sum(h_history[start:t]))
    remaining_quota = QUOTA - used
    remaining_slots = (BLOCK - 1) - o
    if remaining_quota <= 0:
        return (0.0,)
    if remaining_quota >= remaining_slots:
        return (1.0,)
    return (0.0, 1.0)


def choose(arm, posterior, z_hist, h_hist, t, source):
    """(next_z, h_t) after observing y_t. Uses observed prefix, frozen prior/model only."""
    allowed = allowed_actions(t, h_hist)
    if arm == "archived_fixed_replay":          # gate only: the original gaussian_full_r1 fixed policy (no actions)
        return (-1.0 if (t + 1) % 2 == 0 else 1.0), 0.0, None
    if arm == "fixed":
        next_z = -1.0 if (t + 1) % 2 == 0 else 1.0
        h = 1.0 if (t % BLOCK) in FIXED_OFFSETS and t < EVENTS - 1 else 0.0
        return next_z, h, None
    target = "rho" if arm == "recovery" else "a"
    prior_mean, prior_cov, (q, r) = source_prior(source)
    precision = np.linalg.inv(prior_cov)
    psi = np.asarray(posterior["mean"][:4], float)
    scores = []
    for cz, ch in CANDIDATES:
        if ch not in allowed:
            scores.append(np.inf); continue
        z_ext = list(z_hist) + [cz, -cz]
        h_ext = list(h_hist) + [ch, 0.0, 0.0]
        scores.append(target_variance(psi, precision, z_ext, h_ext[:len(z_ext)], q, r, target)[0])
    scores = np.array(scores)
    best = scores.min()
    k = int(np.flatnonzero(np.isclose(scores, best, atol=1e-14, rtol=0))[0])
    return CANDIDATES[k][0], CANDIDATES[k][1], {"scores": scores.tolist(), "chosen": k}


def first_condition(arm, source):
    if arm in ("fixed", "archived_fixed_replay"):
        return -1.0
    target = "rho" if arm == "recovery" else "a"
    prior_mean, prior_cov, (q, r) = source_prior(source)
    precision = np.linalg.inv(prior_cov)
    s = [target_variance(prior_mean, precision, [cz, -cz], [0.0, 0.0], q, r, target)[0] for cz in (-1.0, 1.0)]
    return (-1.0, 1.0)[int(np.flatnonzero(np.isclose(s, min(s), atol=1e-14, rtol=0))[0])]


def rho_summary(posterior):
    """rho = .99 tanh(eta); delta-method variance; eta-scale Wald interval mapped to rho (monotone)."""
    eta = float(posterior["mean"][2]); var_eta = float(max(posterior["covariance"][2, 2], 0.0))
    drho = 0.99 * (1 - np.tanh(eta) ** 2)
    lo, hi = 0.99 * np.tanh(eta - 1.959963984540054 * np.sqrt(var_eta)), 0.99 * np.tanh(eta + 1.959963984540054 * np.sqrt(var_eta))
    return {"rho_hat": float(0.99 * np.tanh(eta)), "rho_var_delta": float(drho ** 2 * var_eta), "rho_lo": float(lo), "rho_hi": float(hi)}


def run_arm(arm, world, streams, source, budgets=(8, 16, 24)):
    """One arm in one world for one author; returns per-step records and budget metrics."""
    a, b, rho, beta = streams["a"], streams["b"], streams["rho"], streams["beta"]
    effect = beta if world == "distributed" else 0.0
    z = first_condition(arm, source)
    state = reference = streams["initial"]
    prev_h = 0.0
    rows, z_hist, h_hist, steps, metrics = [], [], [], [], []
    perturb, failures, nonfinite = 0.0, 0, 0
    for t in range(EVENTS):
        state = rho * state + effect * prev_h + streams["innovations"][t]
        reference = rho * reference + streams["innovations"][t]
        y = a + b * z + state + streams["measurement"][t]
        rows.append({"author_id": streams["author_id"], "event_id": f"{streams['author_id']}:{t}", "role": "application", "t": t, "z": float(z), "y": float(y), "h": 0})
        z_hist.append(float(z))
        post = fit_prefix(rows, source)
        failures += int(not post["success"])
        nonfinite += int(not np.all(np.isfinite(post["mean"])) or not np.all(np.isfinite(post["covariance"])))
        if t + 1 < EVENTS:
            next_z, h, info = choose(arm, post, z_hist, h_hist, t, source)
        else:
            next_z, h, info = z, 0.0, None
        rows[-1]["h"] = int(h); h_hist.append(float(h))
        perturb += (state - reference) ** 2
        rs = rho_summary(post)
        steps.append({"t": t, "z": float(z), "h": int(h), "rho_hat": rs["rho_hat"], "rho_var": rs["rho_var_delta"], "info": info, "fit_success": bool(post["success"])})
        if t + 1 in budgets:
            eff = int(sum(h_hist[:-1]))
            metrics.append({"arm": arm, "world": world, "budget": t + 1, "rho_true": rho, **rs,
                            "rho_err": rs["rho_hat"] - rho, "rho_cover": float(rs["rho_lo"] <= rho <= rs["rho_hi"]), "rho_width": rs["rho_hi"] - rs["rho_lo"],
                            "a_err": float(post["mean"][0] - a), "b_err": float(post["mean"][1] - b), "state_err": float(post["mean"][4] - state),
                            "beta_err": float(post["mean"][3] - effect), "effective_actions": eff, "observed": t + 1,
                            "positive_condition_share": float(np.mean(np.array(z_hist) > 0)), "sum_sq_perturbation": float(perturb),
                            "fit_failures": failures, "nonfinite": nonfinite})
        prev_h, z = h, next_z
    return {"steps": steps, "metrics": metrics, "h": h_hist, "z": z_hist}


def quota_ok(h_hist):
    h = np.asarray(h_hist)
    for k in range(EVENTS // BLOCK):
        blk = h[k * BLOCK:(k + 1) * BLOCK]
        if int(blk[:BLOCK - 1].sum()) != QUOTA or blk[BLOCK - 1] != 0:
            return False
    return True
