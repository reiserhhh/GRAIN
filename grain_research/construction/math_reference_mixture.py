#!/usr/bin/env python3
"""Common-unit SOURCE mixtures and analytic author-order certificates."""

import numpy as np
from scipy.special import logsumexp

from grain_research.construction.math_relational_diagnostic_update import (
    BITS,
    TARGETS,
    compile_bank,
)


def source_log_mass(bank, points):
    """Recover paired/product joint masses without independent source scaling."""
    bank = compile_bank(bank["prototypes"], bank["codes"], bank["variance"])
    z, g, s2 = bank["prototypes"], bank["codes"], float(bank["variance"])
    x = np.asarray(points, float)
    if x.ndim != 3 or x.shape[1:] != z.shape[1:] or not np.isfinite(x).all():
        raise ValueError("Registered finite two-window vectors required")
    # Only the identical Gaussian density constant is omitted across both sources.
    k = np.array(
        [
            (
                x[:, v] @ z[:, v].T
                - np.sum(z[:, v] ** 2, 1) / 2
                - np.sum(x[:, v] ** 2, 1)[:, None] / 2
            )
            / s2
            for v in (0, 1)
        ]
    )
    result = np.empty((len(x), 2, 32))
    for code in range(32):
        ix = g == code
        a, b = k[:, :, ix]
        count = np.log(ix.sum())
        result[:, 0, code] = logsumexp(a + b, 1) - count - np.log(32)
        result[:, 1, code] = logsumexp(a, 1) + logsumexp(b, 1) - 2 * count - np.log(32)
    return result


def moments(log_mass, statistics=BITS):
    """Retain two source evidence masses and target numerators in shared units."""
    logs, y = np.asarray(log_mass, float), np.asarray(statistics, float)
    if (
        logs.shape != (2, 32)
        or not np.isfinite(logs).all()
        or y.ndim != 2
        or y.shape[0] != 32
        or not np.isfinite(y).all()
    ):
        raise ValueError(
            "Two complete common-unit laws and registered statistics required"
        )
    scale = float(logs.max())
    mass = np.exp(logs - scale)
    if np.any(mass == 0):
        raise ValueError("Unrepresentable likelihood mass, no clipping or floor")
    denominator = mass.sum(-1)
    return {
        "numerator": mass @ y,
        "denominator": denominator,
        "log_evidence": np.log(denominator) + scale,
        "endpoint_means": (mass @ y) / denominator[:, None],
    }


def mix(log_mass, weight):
    """Mix source laws before normalization; report evidence-dependent weights."""
    logs = np.asarray(log_mass, float)
    m = moments(logs)
    if not np.isscalar(weight) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Declared SOURCE1 mixture weight must be in [0,1]")
    if weight == 0:
        mixed = logs[0]
    elif weight == 1:
        mixed = logs[1]
    else:
        mixed = np.logaddexp(np.log1p(-weight) + logs[0], np.log(weight) + logs[1])
    probability = np.exp(mixed - logsumexp(mixed))
    b = m["denominator"]
    responsibility = weight * b[1] / ((1 - weight) * b[0] + weight * b[1])
    return {
        "profile": probability,
        "means": probability @ BITS,
        "source1_responsibility": float(responsibility),
    }


def quadratic_roots(coefficients):
    """Return all real interior roots of a degree-at-most-two polynomial."""
    c, b, a = map(float, coefficients)
    if a == 0:
        roots = [] if b == 0 else [-c / b]
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            roots = []
        elif disc == 0:
            roots = [-b / (2 * a)]
        else:
            q = -0.5 * (b + np.copysign(np.sqrt(disc), b))
            roots = [q / a, c / q]
    return sorted({float(r) for r in roots if np.isfinite(r) and 0 < r < 1})


def certificate(numerator_u, denominator_u, numerator_v, denominator_v, *, tol=1e-12):
    """Certify a comparison on the entire mixture family by finite algebra."""
    au, bu, av, bv = [
        np.asarray(v, float)
        for v in (numerator_u, denominator_u, numerator_v, denominator_v)
    ]
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in (au, bu, av, bv)):
        raise ValueError("Two finite source moments required per author")
    if np.any(bu <= 0) or np.any(bv <= 0) or not np.isfinite(tol) or tol <= 0:
        raise ValueError(
            "Positive evidence denominators and fixed sign tolerance required"
        )
    dau, dbu, dav, dbv = au[1] - au[0], bu[1] - bu[0], av[1] - av[0], bv[1] - bv[0]
    coeff = np.array(
        [
            au[0] * bv[0] - av[0] * bu[0],
            dau * bv[0] + au[0] * dbv - dav * bu[0] - av[0] * dbu,
            dau * dbv - dav * dbu,
        ]
    )
    p_times = [0.0, 1.0]
    if coeff[2] != 0:
        vertex = -coeff[1] / (2 * coeff[2])
        if 0 < vertex < 1:
            p_times.append(float(vertex))
    p_values = np.polynomial.polynomial.polyval(p_times, coeff)
    minimum, maximum = float(min(p_values)), float(max(p_values))
    du, dv = dau * bu[0] - au[0] * dbu, dav * bv[0] - av[0] * dbv
    derivative = [
        du * bv[0] ** 2 - dv * bu[0] ** 2,
        2 * (du * bv[0] * dbv - dv * bu[0] * dbu),
        du * dbv**2 - dv * dbu**2,
    ]
    times = [0.0, 1.0, *quadratic_roots(derivative)]

    def difference(t):
        return float(
            (au[0] + t * dau) / (bu[0] + t * dbu)
            - (av[0] + t * dav) / (bv[0] + t * dbv)
        )

    values = [difference(t) for t in times]
    delta_min, delta_max = min(values), max(values)
    if delta_min >= -tol and delta_max <= tol:
        status = "tie_within_tolerance"
    elif minimum > tol:
        status = "u_above_all"
    elif maximum < -tol:
        status = "u_below_all"
    elif minimum < -tol and maximum > tol:
        status = "crossing"
    else:
        status = "boundary_or_unresolved"
    endpoints = [difference(t) for t in (0.0, 1.0)]
    consensus = bool(
        endpoints[0] * endpoints[1] > 0 and min(abs(v) for v in endpoints) > tol
    )
    return {
        "status": status,
        "endpoint_consensus": consensus,
        "endpoint_deltas": endpoints,
        "quadratic": coeff.tolist(),
        "numerator_min": minimum,
        "numerator_max": maximum,
        "polynomial_extrema_weights": p_times,
        "crossing_weights": []
        if status == "tie_within_tolerance"
        else quadratic_roots(coeff),
        "delta_min": delta_min,
        "delta_max": delta_max,
        "delta_min_weight": times[values.index(delta_min)],
        "delta_max_weight": times[values.index(delta_max)],
        "delta_extrema_weights": times,
        "endpoint_consensus_interior_reversal": consensus and status == "crossing",
    }


def compare(log_mass_u, log_mass_v, *, tol=1e-12):
    """Compare all five targets without selecting a reference mixture weight."""
    u, v = moments(log_mass_u), moments(log_mass_v)
    return [
        certificate(
            u["numerator"][:, q],
            u["denominator"],
            v["numerator"][:, q],
            v["denominator"],
            tol=tol,
        )
        for q in range(5)
    ]


def toy():
    """Positive controlled reversal despite endpoint agreement, no real-data fit."""

    def law(means, evidence):
        return np.array(
            [
                np.log(np.where(BITS[:, 0] == 1, p, 1 - p) / 16 * b)
                for p, b in zip(means, evidence, strict=True)
            ]
        )

    return law((0.9, 0.2), (1, 100)), law((0.8, 0.1), (100, 1))


def cold(record):
    """Accept only two addressed common-unit source laws, not labels or answers."""
    if set(record) != {"evidence_keys", "source_contract", "log_joint_mass"}:
        raise ValueError(
            "Only two addressed likelihood laws and source contract permitted"
        )
    keys = record["evidence_keys"]
    if (
        not isinstance(keys, list)
        or len(keys) != 2
        or not all(isinstance(k, str) and k for k in keys)
        or not isinstance(record["source_contract"], str)
        or not record["source_contract"]
    ):
        raise ValueError(
            "Two evidence addresses and nonempty reference contract required"
        )
    logs = np.asarray(record["log_joint_mass"], float)
    if logs.shape != (2, 2, 32):
        raise ValueError("Each evidence unit needs both common-unit source laws")
    return {
        "evidence_keys": keys,
        "source_contract": record["source_contract"],
        "target_order": list(TARGETS),
        "endpoint_means": [moments(x)["endpoint_means"].tolist() for x in logs],
        "source_log_evidence": [moments(x)["log_evidence"].tolist() for x in logs],
        "comparisons": compare(*logs),
        "scope": "entire declared two-source convex family,not psychological validity or all populations",
    }
