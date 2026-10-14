"""R26/A07 committed two-observation recovery design (packet 26). Only the DESIGN RULE differs from packet 25.

Clock (indices 0..23; budgets 8/16/24 end at y7/y15/y23):
  z0 = -1 in every arm.
  After y0: choose and commit (h0, z1) with a ONE-observation criterion (scored window: y1).
  After y_t, t = 1,3,...,21: choose a JOINT pair ((h_t, z_{t+1}), (h_{t+1}, z_{t+2})) and execute BOTH without
      replanning the second after y_{t+1} (inference may update; the commitment may not). Scored windows 2-3,...,22-23.
  After y23: stop, h23 = 0.  h7, h15 are 0 by the block rule.
Environment: s_t = rho s_{t-1} + beta h_{t-1} + w_t ;  y_t = a + b z_t + s_t + e_t ;  h_{-1} = 0.
A new h_t moves the mean of y_{t+1} by beta and of y_{t+2} by rho*beta; its NEW eta sensitivity starts at y_{t+2}
(existing past actions and the state covariance can already carry eta information).
Criterion: predicted target variance from the expected Gaussian Fisher (mean + eta covariance terms) at the current MAP
and frozen prior, for the observed prefix extended by exactly the committed candidate conditions/actions (no refit on
imaginary y, nothing beyond the window, no phantom action). Tie rule: lexical over (z_{t+1}, h_t, z_{t+2}, h_{t+1}),
z = -1 before +1 and h = 0 before 1, atol 1e-14.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from grain_research import f06_recovery_design as p25
from grain_research.gptpro_foundation_v3.f06_gaussian_design import fit_prefix, source_prior

EVENTS, BLOCK = p25.EVENTS, p25.BLOCK
ARMS = ("fixed", "scalar_a", "recovery")
ZS, HS = (-1.0, 1.0), (0.0, 1.0)
EPOCHS = tuple(range(1, EVENTS - 2, 2))            # 1,3,...,21


def window_arrays(z_hist, h_hist, zs_new, hs_new):
    """Extend observed conditions by the committed new conditions and actions; last h entry is a placeholder that
    affects no scored observation (gaussian_terms uses h[t-1] for observation t)."""
    z_ext = list(z_hist) + list(zs_new)
    h_ext = (list(h_hist) + list(hs_new) + [0.0])[:len(z_ext)]
    return z_ext, h_ext


def feasible_pairs(t, h_hist):
    """Sequential quota feasibility: h_t against actual history, then h_{t+1} against history + hypothetical h_t."""
    out = []
    for h1 in p25.allowed_actions(t, h_hist):
        for h2 in p25.allowed_actions(t + 1, list(h_hist) + [h1]):
            out.append((h1, h2))
    return out


def candidates_pair(t, h_hist):
    """Lexical order over (z_{t+1}, h_t, z_{t+2}, h_{t+1})."""
    feas = set(feasible_pairs(t, h_hist))
    return [(z1, h1, z2, h2) for z1, h1, z2, h2 in product(ZS, HS, ZS, HS) if (h1, h2) in feas]


def candidates_single(t, h_hist):
    return [(z1, h1) for z1, h1 in product(ZS, HS) if h1 in p25.allowed_actions(t, h_hist)]


def score(post, z_hist, h_hist, zs_new, hs_new, source, target):
    prior_mean, prior_cov, (q, r) = source_prior(source)
    precision = np.linalg.inv(prior_cov)
    z_ext, h_ext = window_arrays(z_hist, h_hist, zs_new, hs_new)
    return p25.target_variance(np.asarray(post["mean"][:4], float), precision, z_ext, h_ext, q, r, target)[0]


def plan(arm, t, post, z_hist, h_hist, source):
    """Commitment made after y_t (t = 0 single step; t odd in EPOCHS pair). Returns list of (h, next_z) transitions."""
    target = "a" if arm == "scalar_a" else "rho"
    if t == 0:
        cands = candidates_single(t, h_hist)
        sc = np.array([score(post, z_hist, h_hist, [z1], [h1], source, target) for z1, h1 in cands])
        k = int(np.flatnonzero(np.isclose(sc, sc.min(), atol=1e-14, rtol=0))[0])
        z1, h1 = cands[k]
        return [(h1, z1)], {"candidates": [list(c) for c in cands], "scores": sc.tolist(), "chosen": k}
    cands = candidates_pair(t, h_hist)
    sc = np.array([score(post, z_hist, h_hist, [z1, z2], [h1, h2], source, target) for z1, h1, z2, h2 in cands])
    k = int(np.flatnonzero(np.isclose(sc, sc.min(), atol=1e-14, rtol=0))[0])
    z1, h1, z2, h2 = cands[k]
    return [(h1, z1), (h2, z2)], {"candidates": [list(c) for c in cands], "scores": sc.tolist(), "chosen": k}


def run_arm(arm, world, streams, source, budgets=(8, 16, 24), fixed_schedule=None, record_plans=True):
    """Committed design execution. fixed_schedule=(z_seq, h_seq) replays a given schedule (mechanism add-on / fixed arm)."""
    a, b, rho, beta = streams["a"], streams["b"], streams["rho"], streams["beta"]
    effect = beta if world == "distributed" else 0.0
    z = -1.0
    state = reference = streams["initial"]; prev_h = 0.0
    rows, z_hist, h_hist, ys, metrics, plans = [], [], [], [], [], []
    pending = []                                   # committed future transitions [(h_t, z_{t+1}), ...]
    perturb, failures, nonfinite = 0.0, 0, 0
    for t in range(EVENTS):
        state = rho * state + effect * prev_h + streams["innovations"][t]
        reference = rho * reference + streams["innovations"][t]
        y = a + b * z + state + streams["measurement"][t]
        ys.append(float(y))
        rows.append({"author_id": streams["author_id"], "event_id": f"{streams['author_id']}:{t}", "role": "application", "t": t, "z": float(z), "y": float(y), "h": 0})
        z_hist.append(float(z))
        post = fit_prefix(rows, source)
        failures += int(not post["success"])
        nonfinite += int(not np.all(np.isfinite(post["mean"])) or not np.all(np.isfinite(post["covariance"])))
        if t == EVENTS - 1:
            h, next_z = 0.0, z
        elif fixed_schedule is not None:
            h, next_z = float(fixed_schedule[1][t]), float(fixed_schedule[0][t + 1])
        elif arm == "fixed":
            next_z = -1.0 if (t + 1) % 2 == 0 else 1.0
            h = 1.0 if (t % BLOCK) in p25.FIXED_OFFSETS else 0.0
        else:
            if not pending:
                assert t == 0 or t in EPOCHS, "commitment epoch mismatch"
                trans, info = plan(arm, t, post, z_hist, h_hist, source)
                pending = list(trans)
                if record_plans:
                    plans.append({"t": t, "committed": [[float(hh), float(zz)] for hh, zz in trans], **info})
            h, next_z = pending.pop(0)
        rows[-1]["h"] = int(h); h_hist.append(float(h))
        perturb += (state - reference) ** 2
        if t + 1 in budgets:
            rs = p25.rho_summary(post)
            zarr = np.array(z_hist)
            metrics.append({"arm": arm, "world": world, "budget": t + 1, "rho_true": rho, **rs, "rho_err": rs["rho_hat"] - rho,
                            "rho_cover": float(rs["rho_lo"] <= rho <= rs["rho_hi"]), "rho_width": rs["rho_hi"] - rs["rho_lo"],
                            "a_err": float(post["mean"][0] - a), "b_err": float(post["mean"][1] - b), "state_err": float(post["mean"][4] - state),
                            "beta_err": float(post["mean"][3] - effect), "effective_actions": int(sum(h_hist[:-1])), "observed": t + 1,
                            "positive_condition_share": float(np.mean(zarr > 0)), "z_design_rank": int(np.linalg.matrix_rank(np.column_stack([np.ones(len(zarr)), zarr]))),
                            "sum_sq_perturbation": float(perturb), "fit_failures": failures, "nonfinite": nonfinite,
                            "info_condition": float(np.linalg.cond(np.linalg.inv(post["covariance"][:4, :4]))) if np.all(np.isfinite(post["covariance"])) else float("nan")})
        prev_h, z = h, next_z
    return {"metrics": metrics, "h": h_hist, "z": z_hist, "y": ys, "plans": plans}
