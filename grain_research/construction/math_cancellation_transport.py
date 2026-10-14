#!/usr/bin/env python3
"""Source-hidden component compensation in a fixed physical observation space."""

from __future__ import annotations

import itertools

import numpy as np

from grain_research.construction.math_target_active_support_stable import source_basis

ACTION_FIELDS = (
    "full",
    "u",
    "w",
    "remaining",
    "missing_u",
    "random_w",
    "random_remaining",
    "link_complement",
    "decoder_complement",
    "roundoff_complement",
    "c_action",
    "e_action",
)


def validate_basis(basis, dimension, against=None, tolerance=1e-8):
    """Require orthonormal physical coordinates, not a nonorthogonal chart."""
    u = np.asarray(basis, float)
    if u.ndim != 2 or u.shape[0] != dimension or not np.isfinite(u).all():
        raise ValueError("finite aligned physical basis required")
    if u.shape[1] and np.max(abs(u.T @ u - np.eye(u.shape[1]))) > tolerance:
        raise ValueError("orthonormal basis required")
    if against is not None:
        a = np.asarray(against, float)
        if a.shape[0] != dimension or (
            a.shape[1] and u.shape[1] and np.max(abs(a.T @ u)) > tolerance
        ):
            raise ValueError("complement must be orthogonal to the frozen basis")
    return u


def complement(columns, basis):
    """Project column vectors into the fixed basis complement without p-by-p arrays."""
    x = np.asarray(columns, float)
    u = validate_basis(basis, x.shape[0])
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite column vectors required")
    return x - u @ (u.T @ x)


def compensation(link, decoder, basis):
    """Split component complements into antisymmetric c and common e coordinates."""
    left, right = np.asarray(link, float), np.asarray(decoder, float)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("aligned physical component columns required")
    rl, rd = complement(left, basis), complement(right, basis)
    c, e = (rl - rd) / 2, (rl + rd) / 2
    return {"r_link": rl, "r_decoder": rd, "c": c, "e": e}


def compensation_basis(columns, roles, old_basis, roundoff_columns, multiplier=100.0):
    """Construct one source-only complement with fixed machine rank and floor labels."""
    x, r = np.asarray(columns, float), np.asarray(roundoff_columns, float)
    if x.ndim != 2 or r.shape != x.shape or len(roles) != x.shape[1]:
        raise ValueError("aligned source columns and roundoff proxy required")
    if any(role != "source" for role in roles):
        raise ValueError("application cannot determine compensation directions")
    if not np.isfinite(r).all():
        raise ValueError("finite roundoff proxy required")
    projected = complement(complement(x, old_basis), old_basis)
    fitted = source_basis(projected, roles, multiplier)
    u = validate_basis(fitted["basis"], len(x), old_basis)
    scales = fitted["scales"]
    normalized = projected[:, scales > 0] / scales[scales > 0]
    singular = np.linalg.svd(normalized, compute_uv=False)
    raw_singular = np.linalg.svd(projected, compute_uv=False)
    proxy_norms = np.linalg.norm(r, axis=0)
    normalized_floor = float(
        np.linalg.norm(proxy_norms[scales > 0] / scales[scales > 0])
    )
    raw_floor = float(np.linalg.norm(r))
    return {
        **fitted,
        "basis": u,
        "raw": projected,
        "singular_values": singular,
        "raw_singular_values": raw_singular,
        "roundoff_proxy_norms": proxy_norms,
        "normalized_roundoff_proxy_floor": normalized_floor,
        "physical_roundoff_proxy_floor": raw_floor,
        "resolved_against_proxy": singular > normalized_floor,
        "proxy_is_certified_component_error_bound": False,
    }


def random_complement(old_basis, rank, seed):
    """Generate one same-rank control in the old physical complement before evaluation."""
    u = validate_basis(old_basis, len(old_basis))
    if not 0 <= rank <= u.shape[0] - u.shape[1]:
        raise ValueError("matched random rank outside complement dimension")
    if rank == 0:
        return np.empty((len(u), 0))
    x = np.random.default_rng(seed).standard_normal((len(u), rank))
    x = complement(complement(x, u), u)
    q = np.linalg.qr(x, mode="reduced")[0]
    return validate_basis(q, len(u), u)


def transport_sum(link, decoder, link_map, decoder_map):
    """Apply explicit separate linear transports, retaining their signed sum."""
    l, d, gl, gd = (
        np.asarray(x, float) for x in (link, decoder, link_map, decoder_map)
    )
    if l.ndim != 1 or d.shape != l.shape or gl.ndim != 2 or gd.shape != gl.shape:
        raise ValueError("aligned component and transport matrices required")
    if gl.shape[1] != len(l) or not all(np.isfinite(x).all() for x in (l, d, gl, gd)):
        raise ValueError("finite aligned transports required")
    return gl @ l + gd @ d


def parts(total, link, decoder, roundoff, displacement, old_basis, w, random_w):
    """Retain coverage, physical action, compensation, and every signed remainder."""
    v, l, d, r = (np.asarray(x, float) for x in (total, link, decoder, roundoff))
    dp = np.asarray(displacement, float).ravel()
    if v.ndim != 2 or v.shape != l.shape or v.shape != d.shape or v.shape != r.shape:
        raise ValueError("aligned two-arm covectors required")
    if v.shape[1] != len(dp) or not all(np.isfinite(x).all() for x in (v, l, d, r, dp)):
        raise ValueError("finite physical covectors and displacement required")
    u = validate_basis(old_basis, len(dp))
    w, wr = validate_basis(w, len(dp), u), validate_basis(random_w, len(dp), u)
    if w.shape != wr.shape:
        raise ValueError("random control must have the same dimension")
    vc, dc = v @ u, dp @ u
    wc, wd, rc, rd = v @ w, dp @ w, v @ wr, dp @ wr
    full, au, aw, ar = v @ dp, vc @ dc, wc @ wd, rc @ rd
    cr = compensation(l.T, d.T, u)
    rl, rdecoder = cr["r_link"].T, cr["r_decoder"].T
    rroundoff = complement(r.T, u).T
    result = {
        "full": full,
        "u": au,
        "w": aw,
        "remaining": full - au - aw,
        "missing_u": full - au,
        "random_w": ar,
        "random_remaining": full - au - ar,
        "link_complement": rl @ dp,
        "decoder_complement": rdecoder @ dp,
        "roundoff_complement": rroundoff @ dp,
        "c_action": cr["c"].T @ dp,
        "e_action": cr["e"].T @ dp,
        "c_norm": np.linalg.norm(cr["c"], axis=0),
        "e_norm": np.linalg.norm(cr["e"], axis=0),
        "link_complement_norm": np.linalg.norm(rl, axis=1),
        "decoder_complement_norm": np.linalg.norm(rdecoder, axis=1),
        "component_dot": np.sum(rl * rdecoder, axis=1),
        "covector_energy": np.sum(v * v, axis=1),
        "u_covector_energy": np.sum(vc * vc, axis=1),
        "w_covector_energy": np.sum(wc * wc, axis=1),
        "random_covector_energy": np.sum(rc * rc, axis=1),
        "input_energy": np.full(len(v), float(dp @ dp)),
        "u_input_energy": np.full(len(v), float(dc @ dc)),
        "w_input_energy": np.full(len(v), float(wd @ wd)),
        "random_input_energy": np.full(len(v), float(rd @ rd)),
        "component_identity_max": np.max(abs(v - l - d - r), axis=1),
    }
    denominator = result["link_complement_norm"] * result["decoder_complement_norm"]
    result["component_cosine"] = [
        float(dot / n) if n else None
        for dot, n in zip(result["component_dot"], denominator, strict=True)
    ]
    return result


def signed_summary(rows, floor=1e-8):
    """Summarize addressed actions without adding RMS or dividing unresolved totals."""
    if not rows:
        raise ValueError("nonempty addressed rows required")
    x = {k: np.array([r[k] for r in rows], float) for k in ACTION_FIELDS}
    if not all(np.isfinite(v).all() for v in x.values()):
        raise ValueError("finite signed actions required")
    ms = {k: float(np.mean(v * v)) for k, v in x.items()}
    result = {
        "count": len(rows),
        "full_denominator_resolved": ms["full"] > floor**2,
        "missing_u_denominator_resolved": ms["missing_u"] > floor**2,
    }
    for k, v in x.items():
        result[k + "_mean"], result[k + "_rms"] = float(v.mean()), float(np.sqrt(ms[k]))
    for a, b in itertools.combinations(("u", "w", "remaining"), 2):
        result[a + "__" + b + "_cross"] = float(2 * np.mean(x[a] * x[b]))
    result["w_remaining_cross"] = float(2 * np.mean(x["w"] * x["remaining"]))
    result["missing_ms_reduction"] = ms["missing_u"] - ms["remaining"]
    result["random_missing_ms_reduction"] = ms["missing_u"] - ms["random_remaining"]
    for k in ("missing_u", "remaining", "random_remaining", "w"):
        result[k + "_over_full_rms"] = (
            float(np.sqrt(ms[k] / ms["full"]))
            if result["full_denominator_resolved"]
            else None
        )
    result["missing_ms_reduction_fraction"] = (
        result["missing_ms_reduction"] / ms["missing_u"]
        if result["missing_u_denominator_resolved"]
        else None
    )
    result["random_missing_ms_reduction_fraction"] = (
        result["random_missing_ms_reduction"] / ms["missing_u"]
        if result["missing_u_denominator_resolved"]
        else None
    )
    result["sum_identity_error"] = float(
        np.max(abs(x["full"] - x["u"] - x["w"] - x["remaining"]))
    )
    return result


def assert_scope(spec, source_nodes=None, application=False, barrier=False):
    """Refuse changed budgets, neural work, or application before source sealing."""
    if spec["maximum_analysis_seconds"] != 300 or spec["maximum_model_calls"] != 0:
        raise ValueError("fixed CPU-only budget required")
    if source_nodes is not None and source_nodes != 48:
        raise ValueError("all48source nodes required")
    if application and not barrier:
        raise ValueError("source barrier must precede application")
