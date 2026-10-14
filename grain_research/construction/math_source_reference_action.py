#!/usr/bin/env python3
"""Separate current reference calibration from finite evidence-response drift."""

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_diagnostic_action_bridge as M
from grain_research.construction import math_relational_diagnostic_update as R


def certificate(source, destination):
    """Match all32 control readings, keeping signed finite-action transport defects."""
    s, d = np.asarray(source, float), np.asarray(destination, float)
    if s.shape != d.shape or s.shape[-2:] != (2, 32):
        raise ValueError("Selected/control potentials for all32 profiles required")
    if not np.isfinite(s).all() or not np.isfinite(d).all():
        raise ValueError("Missing support is not a neutral potential")
    cal = d[..., 1, :] - s[..., 1, :]
    t = s + cal[..., None, :]
    eta = (d[..., 0, :] - d[..., 1, :]) - (s[..., 0, :] - s[..., 1, :])
    source_action = M.odds(s[..., 0, :]) - M.odds(s[..., 1, :])
    dest_action = M.odds(d[..., 0, :]) - M.odds(d[..., 1, :])
    transport_action = M.odds(t[..., 0, :]) - M.odds(t[..., 1, :])
    defects, conditional = [], []
    for q in range(5):
        hi = R.BITS[:, q] == 1
        side = []
        for mask in (hi, ~hi):
            base = t[..., 0, mask]
            side.append(logsumexp(base + eta[..., mask], -1) - logsumexp(base, -1))
        defects.append(side[0] - side[1])
        lo = np.flatnonzero(~hi)
        conditional.append(eta[..., lo + 2**q] - eta[..., lo])
    source_profile = t[..., 1, :] - logsumexp(t[..., 1, :], -1)[..., None]
    dest_profile = d[..., 1, :] - logsumexp(d[..., 1, :], -1)[..., None]
    return {
        "source_action": source_action,
        "destination_action": dest_action,
        "transport_action": transport_action,
        "calibration_action": transport_action - source_action,
        "response_defect": dest_action - transport_action,
        "defect_identity": np.stack(defects, -1),
        "conditional_cross_ratio": np.stack(conditional, -2),
        "profile_relative_eta": eta - eta.mean(-1, keepdims=True),
        "control_profile_error": float(np.max(abs(source_profile - dest_profile))),
        "telescope_error": float(np.max(abs(t[..., 0, :] + eta - d[..., 0, :]))),
    }


def logs(bank, points, view):
    """Read the inherited three laws; no new bandwidth, prior or outcome fit."""
    single, pair, product = M.potentials(bank, points)
    return np.stack((single[:, view], pair, product))


def cold(source, destination, record):
    """Return reference-transport actions from two SOURCE banks and numeric events."""
    if source["prototypes"].shape[-1] != destination["prototypes"].shape[-1]:
        raise ValueError("Different embedding coordinates")
    if float(source["variance"]) != float(destination["variance"]):
        raise ValueError("Reference change cannot silently alter the channel")
    points, view = M.edit_points(record, source["prototypes"].shape[-1])
    result = certificate(logs(source, points, view), logs(destination, points, view))
    return {
        "frame_key": record["frame_key"],
        "view": view,
        "laws": list(M.LAWS),
        "targets": list(R.TARGETS),
        **{
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()
        },
        "meaning": "source_reference_transport_certificate_not_accuracy_or_human_trait_change",
    }


def literal_laws(bank, points, view):
    """Evaluate finite SOURCE densities with explicit distance and product loops."""
    z, g, v = bank["prototypes"], bank["codes"], float(bank["variance"])
    result = np.empty((3, 2, 32))
    for p, point in enumerate(points):
        k = np.exp(-np.sum((point[None] - z) ** 2, -1) / (2 * v))
        for c in range(32):
            a, b = k[g == c].T
            result[0, p, c] = (a if view == 0 else b).mean()
            result[1, p, c] = sum(
                float(x * y) for x, y in zip(a, b, strict=True)
            ) / len(a)
            result[2, p, c] = sum(float(x * y) for x in a for y in b) / len(a) ** 2
    if np.any(result <= 0):
        raise ValueError("Literal audit outside finite positive support")
    return result


def literal_certificate(source, destination):
    """Rebuild calibrated actions and cross-ratios in probability coordinates."""
    s, d = np.asarray(source, float), np.asarray(destination, float)
    t = s * (d[:, 1] / s[:, 1])[:, None, :]
    rows = {k: [] for k in ("source_action", "destination_action", "transport_action")}
    for name, law in zip(rows, (s, d, t), strict=True):
        for q in range(5):
            hi = R.BITS[:, q] == 1
            ratio = law[:, :, hi].sum(-1) / law[:, :, ~hi].sum(-1)
            rows[name].append(np.log(ratio[:, 0] / ratio[:, 1]))
    rows = {k: np.stack(v, -1) for k, v in rows.items()}
    eta_ratio = (d[:, 0] / d[:, 1]) / (s[:, 0] / s[:, 1])
    rows["conditional_cross_ratio"] = np.stack(
        [
            np.log(
                eta_ratio[:, np.flatnonzero(R.BITS[:, q] == 0) + 2**q]
                / eta_ratio[:, R.BITS[:, q] == 0]
            )
            for q in range(5)
        ],
        -2,
    )
    rows["response_defect"] = rows["destination_action"] - rows["transport_action"]
    return rows
