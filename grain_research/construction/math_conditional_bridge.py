#!/usr/bin/env python3
"""Anonymous condition-law bridges, preserving soft response differences."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog


def permutations(k):
    """Return every finite label gauge; column i maps to its new row address."""
    return tuple(np.eye(k)[:, p] for p in itertools.permutations(range(k)))


def empirical_pair(categories, k=3):
    """Count paired, supplied-same-author observations, not unpaired marginals."""
    x = np.asarray(categories)
    if x.ndim != 2 or x.shape[1] != 2 or not len(x):
        raise ValueError("Nonempty paired categorical observations required")
    if not np.issubdtype(x.dtype, np.integer) or np.any((x < 0) | (x >= k)):
        raise ValueError("Invalid categorical addresses")
    return np.bincount(x[:, 0] * k + x[:, 1], minlength=k * k).reshape(k, k) / len(x)


def deconvolve_pair(observed, first_emission, second_emission):
    """Recover the anonymous coupling from full-column-rank observation laws.

    P_ij = M_i J_ij M_j.T. Empirical inverses can be signed; no probability
    clipping or projection is used to manufacture a joint law. With square
    invertible channels, exact reconstruction is automatic, not validation.
    """
    p, a, b = map(np.asarray, (observed, first_emission, second_emission))
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Compatible observation laws required")
    if p.shape != (a.shape[0], b.shape[0]):
        raise ValueError("Observed pair has incompatible dimensions")
    if min(np.linalg.svd(x, compute_uv=False)[-1] for x in (a, b)) <= 1e-10:
        raise ValueError("Unidentified coupling: rank-deficient observation law")
    j = np.linalg.pinv(a) @ p @ np.linalg.pinv(b).T
    return {
        "coupling": j,
        "observation_reconstruction_error": float(np.max(abs(a @ j @ b.T - p))),
        "minimum_entry": float(j.min()),
        "total_mass": float(j.sum()),
    }


def best_transport(coupling, tolerance=1e-10):
    """Return ALL optimal bijections; ties are not secretly broken into factors."""
    j = np.asarray(coupling, dtype=float)
    if j.ndim != 2 or j.shape[0] != j.shape[1] or not np.all(np.isfinite(j)):
        raise ValueError("Finite square coupling/objective required")
    candidates = permutations(len(j))
    scores = np.array([np.sum(j * r.T) for r in candidates])
    best = float(scores.max())
    chosen = np.flatnonzero(abs(scores - best) <= tolerance)
    return {
        "matrices": tuple(candidates[i] for i in chosen),
        "match_mass_or_signed_objective": best,
        "optimum_count": len(chosen),
        "second_best_gap": float(np.sort(scores)[-1] - np.sort(scores)[-2]),
    }


def synchronize(couplings, tolerance=1e-10):
    """Enumerate the exact three-condition gauge problem, not a tuning grid.

    Fix S_0=I, where S_i maps reference component addresses to condition i.
    Maximize sum_ij <J_ij, (S_j S_i.T).T>. All tied frames are retained.
    Synchronization imposes consistency on its OUTPUT; it does not establish
    that the data arise from one stable author component.
    """
    if set(couplings) != {(0, 1), (1, 2), (0, 2)}:
        raise ValueError("The declared three-condition triangle is required")
    k = len(couplings[(0, 1)])
    candidates = permutations(k)
    frames = [(np.eye(k), b, c) for b, c in itertools.product(candidates, repeat=2)]
    objectives = np.array(
        [
            sum(np.sum(j * (s[b] @ s[a].T).T) for (a, b), j in couplings.items())
            for s in frames
        ]
    )
    maximum = float(objectives.max())
    chosen = np.flatnonzero(abs(objectives - maximum) <= tolerance)
    edges = {edge: best_transport(j, tolerance) for edge, j in couplings.items()}
    upper = sum(e["match_mass_or_signed_objective"] for e in edges.values())
    cycle = None
    if all(e["optimum_count"] == 1 for e in edges.values()):
        r10, r21, r20 = (edges[e]["matrices"][0] for e in ((0, 1), (1, 2), (0, 2)))
        cycle = r20.T @ r21 @ r10
    return {
        "frames": tuple(frames[i] for i in chosen),
        "optimum_count": len(chosen),
        "objective": maximum,
        "unconstrained_objective": float(upper),
        "forced_consistency_cost": float(upper - maximum),
        "second_best_gap": float(np.sort(objectives)[-1] - np.sort(objectives)[-2]),
        "edges": edges,
        "cycle": cycle,
        "cycle_fixed_points": None if cycle is None else int(np.trace(cycle)),
    }


def stitch_profiles(profiles, frame):
    """Express each conditional profile in a reference frame, WITHOUT averaging."""
    p = np.asarray(profiles)
    if p.ndim != 3 or p.shape[1] != len(frame) or p.shape[2] != frame[0].shape[0]:
        raise ValueError("Author-by-condition-by-component profiles required")
    return np.stack([p[:, c] @ s for c, s in enumerate(frame)], axis=1)


def triangle_compatibility(couplings, tolerance=1e-10):
    """Check existence of a joint law via its finite marginal polytope.

    A returned feasible table is ONE witness, not an identified higher-order
    law. No signed empirical table is silently turned into probabilities.
    """
    if set(couplings) != {(0, 1), (1, 2), (0, 2)}:
        raise ValueError("A full declared triangle is required")
    k = len(couplings[(0, 1)])
    states = np.array(list(itertools.product(range(k), repeat=3)))
    rows, values = [], []
    for (a, b), j in sorted(couplings.items()):
        if j.shape != (k, k) or not np.all(np.isfinite(j)):
            raise ValueError("Invalid coupling")
        if j.min() < -tolerance or abs(j.sum() - 1) > tolerance:
            raise ValueError("Signed/unnormalized estimate is not a probability law")
        for i, q in itertools.product(range(k), repeat=2):
            rows.append(((states[:, a] == i) & (states[:, b] == q)).astype(float))
            values.append(j[i, q])
    matrix, value = np.stack(rows), np.array(values)
    result = linprog(
        np.zeros(k**3), A_eq=matrix, b_eq=value, bounds=(0, None), method="highs"
    )
    if result.status not in (0, 2):
        raise RuntimeError("Marginal-polytope solver did not settle feasibility")
    return {
        "feasible": bool(result.success),
        "solver_status": int(result.status),
        "joint_witness": None if not result.success else result.x.reshape((k,) * 3),
        "witness_error": None
        if not result.success
        else float(np.max(abs(matrix @ result.x - value))),
    }


def cycle_mismatch_bound(couplings, result):
    """Evaluate a union-bound witness for any common three-condition population.

    If all three chosen hard relations hold, the reference class must be a
    fixed point of their cycle. Thus sum edge-error probabilities is at least
    the reference marginal mass on moved classes. Nonflat modal transports
    can coexist with a perfectly valid SOFT joint law.
    """
    cycle = result["cycle"]
    if cycle is None:
        return None
    errors = sum(
        1 - e["match_mass_or_signed_objective"] for e in result["edges"].values()
    )
    marginal = couplings[(0, 1)].sum(axis=1)
    required = float(marginal @ (1 - np.diag(cycle)))
    return {
        "sum_edge_mismatch_mass": float(errors),
        "minimum_required_mass": required,
        "slack": float(errors - required),
    }


def gauge_audit(couplings, profiles=None, tolerance=1e-10):
    """Check all 216 relabelings of three k=3 local laws, not new data trials."""
    reference = synchronize(couplings, tolerance)
    maximum_score_error, maximum_cost_error, maximum_profile_error = 0.0, 0.0, 0.0
    count = 0
    for gauges in itertools.product(
        permutations(len(next(iter(couplings.values())))), repeat=3
    ):
        changed = {
            edge: gauges[edge[0]] @ j @ gauges[edge[1]].T
            for edge, j in couplings.items()
        }
        new = synchronize(changed, tolerance)
        maximum_score_error = max(
            maximum_score_error, abs(new["objective"] - reference["objective"])
        )
        maximum_cost_error = max(
            maximum_cost_error,
            abs(new["forced_consistency_cost"] - reference["forced_consistency_cost"]),
        )
        if (
            new["optimum_count"] != reference["optimum_count"]
            or new["cycle_fixed_points"] != reference["cycle_fixed_points"]
        ):
            raise AssertionError("Gauge changed an identifiable invariant")
        expected_frames = [
            tuple(gauges[c] @ s[c] @ gauges[0].T for c in range(3))
            for s in reference["frames"]
        ]
        if any(
            not any(
                all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
                for b in new["frames"]
            )
            for a in expected_frames
        ):
            raise AssertionError("Optimal frame SET failed gauge covariance")
        if profiles is not None and reference["optimum_count"] == 1:
            local = np.stack([profiles[:, c] @ gauges[c].T for c in range(3)], axis=1)
            actual = stitch_profiles(local, new["frames"][0])
            expected = stitch_profiles(profiles, reference["frames"][0]) @ gauges[0].T
            maximum_profile_error = max(
                maximum_profile_error, float(np.max(abs(actual - expected)))
            )
        count += 1
    return {
        "relabeling_cases": count,
        "objective_error": maximum_score_error,
        "forced_cost_error": maximum_cost_error,
        "conditional_profile_error": maximum_profile_error,
    }
