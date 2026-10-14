#!/usr/bin/env python3
"""Source-posterior updates with target means and retained joint configuration."""

import numpy as np
from scipy.special import logsumexp

BITS = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)
CHARACTERS = np.array(
    [
        np.prod(2 * BITS[:, [q for q in range(5) if m >> q & 1]] - 1, axis=1)
        for m in range(32)
    ]
).T
JOINT_MASKS = np.array([m for m in range(32) if m.bit_count() >= 2])


def normalize(log_mass):
    """Normalize a complete strictly positive profile law without clipping."""
    x = np.asarray(log_mass, float)
    if x.ndim < 1 or x.shape[-1] != 32 or not np.isfinite(x).all():
        raise ValueError("All32 finite log-profile masses required")
    result = x - logsumexp(x, axis=-1, keepdims=True)
    if np.any(np.exp(result) == 0):
        raise ValueError("Unsupported zero-mass boundary; no floor or smoothing")
    return result


def split_score(probability, increment, statistics=BITS):
    """Project a likelihood score onto centered target statistics in L2(p)."""
    p, ell, y = (np.asarray(a, float) for a in (probability, increment, statistics))
    if (
        p.shape != ell.shape
        or p.shape[-1] != 32
        or y.ndim != 2
        or y.shape[0] != 32
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
        or not np.isfinite(ell).all()
        or np.any(p <= 0)
        or not np.allclose(p.sum(-1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError(
            "Complete positive normalized law and finite statistics required"
        )
    mean = p @ y
    centered = y - mean[..., None, :]
    score = ell - np.sum(p * ell, axis=-1, keepdims=True)
    covariance = np.einsum("...g,...gi,...gj->...ij", p, centered, centered)
    slope = np.einsum("...g,...gi,...g->...i", p, centered, score)
    coefficients = np.linalg.solve(covariance, slope[..., None])[..., 0]
    main = np.einsum("...gi,...i->...g", centered, coefficients)
    joint = score - main
    energy = np.sum(p * score**2, -1)
    main_energy = np.sum(p * main**2, -1)
    joint_energy = np.sum(p * joint**2, -1)
    return {
        "mean": mean,
        "covariance": covariance,
        "slope": slope,
        "score": score,
        "main": main,
        "joint": joint,
        "energy": energy,
        "main_energy": main_energy,
        "joint_energy": joint_energy,
    }


def transport(log_start, log_end, *, nodes=16):
    """Return an exact endpoint update and a fixed quadrature response account."""
    a, b = normalize(log_start), normalize(log_end)
    if a.ndim != 2 or a.shape != b.shape or type(nodes) is not int or nodes < 1:
        raise ValueError(
            "Matched batches and a positive fixed quadrature count required"
        )
    p0, p1, ell = np.exp(a), np.exp(b), b - a
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(nodes)
    times, weights = (raw_nodes + 1) / 2, raw_weights / 2
    path_log = normalize(a[:, None, :] + times[None, :, None] * ell[:, None, :])
    path = np.exp(path_log)
    split = split_score(path, np.broadcast_to(ell[:, None, :], path.shape))
    start = split_score(p0, ell)
    main_end = np.exp(normalize(a + start["main"]))
    joint_end = np.exp(normalize(a + start["joint"]))
    delta = (p1 - p0) @ BITS
    main_delta, joint_delta = ((v - p0) @ BITS for v in (main_end, joint_end))
    interaction = delta - main_delta - joint_delta
    integrated_slope = np.einsum("k,nkq->nq", weights, split["slope"])
    total_energy = split["energy"] @ weights
    main_energy = split["main_energy"] @ weights
    joint_energy = split["joint_energy"] @ weights
    jeffreys = np.sum((p1 - p0) * ell, -1)
    chars = CHARACTERS[:, JOINT_MASKS]
    joint_slope = np.einsum("nkg,nkg,gj->nkj", path, split["joint"], chars)
    main_slope = np.einsum("nkg,nkg,gj->nkj", path, split["main"], chars)
    full_character_delta = (p1 - p0) @ chars
    main_character_delta = np.einsum("k,nkj->nj", weights, main_slope)
    joint_character_delta = np.einsum("k,nkj->nj", weights, joint_slope)
    return {
        "log_start": a,
        "log_end": b,
        "probability_start": p0,
        "probability_end": p1,
        "increment": ell,
        "times": times,
        "weights": weights,
        "path_probability": path,
        "path_score": split["score"],
        "path_main": split["main"],
        "path_joint": split["joint"],
        "path_slope": split["slope"],
        "path_covariance": split["covariance"],
        "delta": delta,
        "integrated_slope": integrated_slope,
        "total_energy": total_energy,
        "main_energy": main_energy,
        "joint_energy": joint_energy,
        "jeffreys": jeffreys,
        "main_end": main_end,
        "joint_end": joint_end,
        "main_delta": main_delta,
        "joint_delta": joint_delta,
        "interaction_delta": interaction,
        "character_delta": full_character_delta,
        "character_main_delta": main_character_delta,
        "character_joint_delta": joint_character_delta,
        "main_only_endpoint_KL": np.sum(p1 * (b - np.log(main_end)), -1),
    }


def cold(record):
    """Compute a capsule from bound numerical endpoints, never labels or answers."""
    if set(record) != {"evidence_key", "log_start", "log_end", "source_contract"}:
        raise ValueError(
            "Only qualified numeric endpoints and source provenance permitted"
        )
    if not all(
        isinstance(record[k], str) and record[k]
        for k in ("evidence_key", "source_contract")
    ):
        raise ValueError("Nonempty evidence and source contracts required")
    r = transport(
        np.asarray(record["log_start"])[None], np.asarray(record["log_end"])[None]
    )
    return {
        "evidence_key": record["evidence_key"],
        "source_contract": record["source_contract"],
        "target_order": ["E", "N", "A", "C", "O"],
        "profile_probabilities_start": r["probability_start"][0].tolist(),
        "profile_probabilities_end": r["probability_end"][0].tolist(),
        "target_means_start": (r["probability_start"][0] @ BITS).tolist(),
        "target_means_end": (r["probability_end"][0] @ BITS).tolist(),
        "target_delta": r["delta"][0].tolist(),
        "frozen_main_tilt_delta": r["main_delta"][0].tolist(),
        "frozen_joint_tilt_delta": r["joint_delta"][0].tolist(),
        "finite_interaction_delta": r["interaction_delta"][0].tolist(),
        "joint_character_masks": JOINT_MASKS.tolist(),
        "full_character_delta": r["character_delta"][0].tolist(),
        "joint_character_delta": r["character_joint_delta"][0].tolist(),
        "joint_energy": float(r["joint_energy"][0]),
        "main_energy": float(r["main_energy"][0]),
        "scope": "declared source posterior update,not human trait transition",
    }
