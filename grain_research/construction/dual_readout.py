"""GRAIN-C1-02: target-specific dual readouts (forward loadings vs backward weights).

Forward loading v: how the target expression appears in x (estimated from the controlled
pilot: regression of x on the declared label). Backward readout w: weights applied to x.
For C = G diag(sb^2, sr^2) G^T with G = [v s] of full column rank,
    w = C^+ v / (v^T C^+ v) = G (G^T G)^-1 e1,   so w^T v = 1 and w^T s = 0
even when v and s overlap (standard LCMV / dual-basis algebra; Van Veen et al. 1997;
Haufe et al. 2014). With correlated slopes (rho) the minimum-variance readout instead
satisfies w^T s = -rho sb / sr: covariance alone cannot separate the target intervention.

Arms
- lcmv_dual: source-only noise-corrected between-author slope covariance (one frozen rank
  rule) + pilot forward loading; single unit constraint.
- pilot_dual: AUGMENTED interface. Adds the recorded assigned pilot condition (pilot_z) and
  estimates the context-style loading from between-writer pilot slopes inside the source
  slope range; LCMV with w^T v = 1 and w^T s_hat = 0. Extra observational information.

Fit functions read only EstimatorView / AugmentedEstimatorView fields. Hidden maps, latents
and pilot latents stay in EvaluatorTruth.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from . import hidden_evidence_map as H

RANGE_FRACTION_MIN = 0.5
RHO_BOUNDARY = 0.6
DUAL_ARMS = ("lcmv_dual", "pilot_dual")


@dataclass(frozen=True)
class AugmentedEstimatorView(H.EstimatorView):
    """EstimatorView plus the recorded, assigned condition of each pilot text."""

    pilot_z: np.ndarray


AUGMENTED_FIELDS = tuple(f.name for f in fields(AugmentedEstimatorView))


def assigned_pilot_conditions(seed):
    """Replay the pilot generator's design stream up to the condition assignment.

    This is the assignment record a pilot study keeps (which condition each authored text
    was written under). It reproduces H.pilot_bank's draws in order and stops there; no
    label, fingerprint, style value or map is returned.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1603, 0]))
    n = H.PILOT_WRITERS * H.PILOT_PER_WRITER
    rng.normal(0.0, H.FP_SD, size=(H.PILOT_WRITERS, H.KF))
    rng.normal(0.0, H.STYLE_SD, size=H.PILOT_WRITERS)
    return rng.choice(H.CONDITIONS, size=n)


def augment(view, seed):
    return AugmentedEstimatorView(**{f: getattr(view, f) for f in H.ESTIMATOR_FIELDS},
                                  pilot_z=assigned_pilot_conditions(seed))


def build_world_augmented(seed, world):
    view, truth = H.build_world(seed, world)
    return view, augment(view, seed), truth


def build_world_correlated(seed, rho=RHO_BOUNDARY):
    """Boundary world (not a primary world): W2 latents with corr(b, r) = rho."""
    maps = H.world_maps(seed)
    lat = {}
    for role, n in (("source", H.N_SOURCE), ("application", H.N_APP)):
        x = H.author_latents(seed, "W2_conditional", role, n, True)
        xi = (x["b"] - 0.5) / 0.5
        x["s"] = H.STYLE_SD * (rho * xi + np.sqrt(1 - rho ** 2) * x["s"] / H.STYLE_SD)
        lat[role] = x
    xs, es = H.render_authors(maps, lat["source"])
    xa, ea = H.render_authors(maps, lat["application"])
    px, pw, plab, plat = H.pilot_bank(seed, maps, "expression")
    lat["source"]["e"], lat["application"]["e"] = es, ea
    view = H.EstimatorView(
        source_x=xs.reshape(-1, H.D), source_z=np.tile(lat["source"]["z"], H.N_SOURCE),
        source_author=np.repeat(np.arange(H.N_SOURCE), H.EVENTS),
        app_x=xa.reshape(-1, H.D), app_z=np.tile(lat["application"]["z"], H.N_APP),
        app_author=np.repeat(np.arange(H.N_APP), H.EVENTS),
        pilot_x=px, pilot_writer=pw, pilot_label=plab, pilot_label_kind="declared_expression",
        lexicon_index=maps["lexicon_index"].copy(), lexicon_sign=maps["lexicon_sign"].copy())
    truth = H.EvaluatorTruth(maps=maps, source_latent=lat["source"], app_latent=lat["application"],
                             pilot_latent=plat)
    return view, augment(view, seed), truth


# ------------------------------------------------------------------ algebra
def rank_rule(c):
    """Frozen source-only rule: keep eigenvalues above the largest negative magnitude.

    The corrected covariance is a difference of estimates and is not PSD; the magnitude of
    its most negative eigenvalue is used as a noise-floor witness (heuristic, not a test).
    """
    vals, vecs = np.linalg.eigh((c + c.T) / 2)
    tau = max(-float(vals[0]), 0.0)
    keep = vals > tau
    return {"U": vecs[:, keep][:, ::-1], "L": vals[keep][::-1], "k": int(keep.sum()), "tau": tau,
            "eigenvalues_top": vals[::-1][:8].tolist(), "eigenvalue_min": float(vals[0])}


def lcmv_single(u, lam, v):
    a = u.T @ v
    y = a / lam
    return u @ y / float(a @ y)


def lcmv_two(u, lam, v, s):
    a = np.column_stack([u.T @ v, u.T @ s])
    p = a / lam[:, None]
    coef = np.linalg.solve(a.T @ p, np.array([1.0, 0.0]))
    return u @ (p @ coef)


def dual_basis(v, s):
    """Independent closed form G (G^T G)^-1 e1 (used only in tests/verification)."""
    g = np.column_stack([v, s])
    return g @ np.linalg.solve(g.T @ g, np.array([1.0, 0.0]))


# ------------------------------------------------------------------ estimators
def forward_loading(px, py):
    """Pilot forward loading: OLS slope of every x coordinate on the declared label."""
    yc = py - py.mean()
    return (px - px.mean(0)).T @ yc / float(yc @ yc)


def pilot_style_loading(view, train_mask, u):
    """Between-writer covariance of pilot slopes on recorded z (label partialled out),
    sampling-corrected and restricted to the source slope range u."""
    px, pw, pz, py = (view.pilot_x[train_mask], view.pilot_writer[train_mask],
                      view.pilot_z[train_mask], view.pilot_label[train_mask])
    slopes, inv_zz, rss, df = [], [], np.zeros((H.D, H.D)), 0
    for w in np.unique(pw):
        m = pw == w
        xm = np.column_stack([np.ones(m.sum()), pz[m], py[m]])
        coef, *_ = np.linalg.lstsq(xm, px[m], rcond=None)
        resid = px[m] - xm @ coef
        rss += resid.T @ resid
        df += m.sum() - 3
        slopes.append(coef[1])
        inv_zz.append(np.linalg.inv(xm.T @ xm)[1, 1])
    slopes = np.array(slopes)
    sampling = rss / df * float(np.mean(inv_zz))
    between = np.cov(slopes, rowvar=False) - sampling
    bk = u.T @ between @ u
    vals, vecs = np.linalg.eigh((bk + bk.T) / 2)
    return u @ vecs[:, -1], float(vals[-1]), vals.tolist()


def fit_dual(arm, view, train_mask=None):
    """Build lcmv_dual (EstimatorView) or pilot_dual (AugmentedEstimatorView)."""
    if arm not in DUAL_ARMS:
        raise ValueError(arm)
    if view.pilot_label is None:
        return H.Readout(arm, None, status="MISSING_ANCHOR",
                         info={"reason": "pilot declares no label for the target expression"})
    if arm == "pilot_dual" and not isinstance(view, AugmentedEstimatorView):
        raise TypeError("pilot_dual requires the augmented recorded-pilot-condition interface")
    if train_mask is None:
        train_mask = view.pilot_writer < H.PILOT_TRAIN_WRITERS
    px, py = view.pilot_x[train_mask], view.pilot_label[train_mask]
    between, _ = H._slope_moments(view)
    rr = rank_rule(between)
    v = forward_loading(px, py)
    info = {"rank_k": rr["k"], "rank_tau": rr["tau"], "eigenvalues_top": rr["eigenvalues_top"],
            "eigenvalue_min": rr["eigenvalue_min"],
            "uses": ["source_x", "source_z", "source_author", "pilot_x", "pilot_label"],
            "free_parameters": None}
    if rr["k"] == 0:
        return H.Readout(arm, None, status="RANK_ZERO_UNSUPPORTED", info=info)
    frac = float(np.sum((rr["U"].T @ v) ** 2) / np.sum(v ** 2))
    info["range_fraction"] = frac
    if arm == "lcmv_dual":
        w = lcmv_single(rr["U"], rr["L"], v)
        info["free_parameters"] = f"source covariance rank {rr['k']} + {H.D} pilot loadings"
    else:
        info["uses"] = info["uses"] + ["pilot_writer", "pilot_z (AUGMENTED)"]
        if rr["k"] < 2:
            return H.Readout(arm, None, status="RANK_TOO_LOW_FOR_TWO_CONSTRAINTS", info=info)
        s_hat, s_eig, s_vals = pilot_style_loading(view, train_mask, rr["U"])
        info.update(style_eigenvalue=s_eig, style_eigenvalues=s_vals, style_hat=s_hat)
        if s_eig <= 0:
            return H.Readout(arm, None, status="STYLE_UNSUPPORTED", info=info)
        w = lcmv_two(rr["U"], rr["L"], v, s_hat)
        info["free_parameters"] = f"source covariance rank {rr['k']} + pilot loading + pilot style loading"
    alpha, gamma, t = H.calibrate(px @ w, py, with_t=True)
    info["anchor_t"] = t
    if frac < RANGE_FRACTION_MIN:
        status = "RANGE_UNSUPPORTED"
    elif abs(t) < H.ANCHOR_T_MIN:
        status = "ANCHOR_UNSUPPORTED_DIRECTION"
    else:
        status = "OK"
    return H.Readout(arm, w, alpha, gamma, status, info)


def oracle_perp_style(view, truth):
    """Privileged ceiling: oracle LS readout projected orthogonal to the TRUE style loading."""
    orc = H.oracle_readouts(view, truth)
    us = truth.maps["u_s"]
    d = orc.direction - us * (us @ orc.direction)
    e = truth.source_latent["e"].reshape(-1)
    alpha, gamma = H.calibrate(view.source_x @ d, e)
    return H.Readout("oracle_perp_style", d, alpha, gamma, "PRIVILEGED",
                     {"uses": ["TRUE source expression", "TRUE style loading", "source_x"]})
