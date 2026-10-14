#!/usr/bin/env python3
"""Directed evidence flows paired with a fixed contextual observation kernel."""

from __future__ import annotations

import itertools

import numpy as np


def square(value):
    """Validate a finite square array, without assigning psychological meaning."""
    x = np.asarray(value, float)
    if x.ndim < 2 or x.shape[0] != x.shape[1] or x.shape[0] < 2:
        raise ValueError("At least two aligned state axes required")
    if not np.isfinite(x).all():
        raise ValueError("Finite states required")
    return x


def flow_parts(joint):
    """Separate symmetric traffic, directional current, and endpoint imbalance."""
    j = square(joint)
    if j.ndim != 2 or np.min(j) < 0 or not np.isclose(j.sum(), 1):
        raise ValueError("Nonnegative normalized joint law required")
    return (j + j.T) / 2, (j - j.T) / 2, j.sum(1) - j.sum(0)


def kernel_parts(kernel):
    """Split every output channel by exchanging previous and current evidence."""
    f = square(kernel)
    reverse = f.swapaxes(0, 1)
    return (f + reverse) / 2, (f - reverse) / 2


def contract(joint, kernel):
    """Integrate one fixed observer against an author's directed event law."""
    j, f = square(joint), square(kernel)
    if j.ndim != 2 or j.shape != f.shape[:2]:
        raise ValueError("Author and instrument states must align")
    return np.tensordot(j, f, axes=((0, 1), (0, 1)))


def hodge_current(current):
    """Project a skew flow onto gradients and divergence-free graph cycles.

    The graph is complete and unweighted. This is the graph-level two-part
    decomposition, not a claimed full curl/harmonic simplicial analysis.
    """
    a = square(current)
    if a.ndim != 2 or not np.allclose(a, -a.T, atol=1e-12, rtol=0):
        raise ValueError("Skew state current required")
    n = len(a)
    edges = list(itertools.combinations(range(n), 2))
    incidence = np.zeros((n, len(edges)))
    for e, (i, j) in enumerate(edges):
        incidence[i, e], incidence[j, e] = 1, -1
    values = np.array([a[i, j] for i, j in edges])
    potential = np.linalg.lstsq(incidence.T, values, rcond=None)[0]
    gradient = potential[:, None] - potential[None, :]
    return gradient, a - gradient


def readout_account(joint, kernel):
    """Keep traffic, endpoint, and cycle contributions in the original units."""
    s, a, imbalance = flow_parts(joint)
    even, odd = kernel_parts(kernel)
    gradient, cycle = hodge_current(a)
    terms = [contract(s, even), contract(gradient, odd), contract(cycle, odd)]
    value = contract(joint, kernel)
    return {
        "value": value,
        "traffic": terms[0],
        "endpoint": terms[1],
        "cycle": terms[2],
        "imbalance": imbalance,
        "identity_error": float(np.max(np.abs(value - sum(terms)))),
    }


def cycle_counts(direction):
    """Return matched 45-edge laws with identical occupancy, dwell and traffic."""
    if direction not in (-1, 0, 1):
        raise ValueError("Declared three directions only")
    counts = np.full((3, 3), 5, int)
    for i in range(3):
        counts[i, (i + 1) % 3] += 3 * direction
        counts[i, (i - 1) % 3] -= 3 * direction
    return counts


def euler_path(counts, seed):
    """Build a closed toy path with exactly the supplied directed edge counts."""
    c = np.asarray(counts)
    if c.shape != (3, 3) or np.any(c < 0) or not np.equal(c, np.floor(c)).all():
        raise ValueError("Nonnegative integer three-state counts required")
    if not np.array_equal(c.sum(1), c.sum(0)) or not np.all(c > 0):
        raise ValueError("This toy requires balanced connected positive counts")
    rng = np.random.default_rng(seed)
    todo = [rng.permutation(np.repeat(np.arange(3), row)).tolist() for row in c]
    stack, completed = [int(rng.integers(3))], []
    while stack:
        i = stack[-1]
        if todo[i]:
            stack.append(todo[i].pop())
        else:
            completed.append(stack.pop())
    path = np.asarray(completed[::-1], int)
    if len(path) != c.sum() + 1 or path[0] != path[-1]:
        raise ValueError("Incomplete Euler traversal")
    return path


def path_joint(path, closed=False):
    """Count actual adjacency; wrapping is explicit, never assumed for real text."""
    p = np.asarray(path)
    if p.ndim != 1 or len(p) < 2 or not np.equal(p, np.floor(p)).all():
        raise ValueError("Integer path with at least two points required")
    if np.min(p) < 0 or np.max(p) >= 3:
        raise ValueError("Declared three-state path required")
    p = p.astype(int)
    if closed:
        p = np.r_[p, p[0]]
    counts = np.zeros((3, 3), int)
    np.add.at(counts, (p[:-1], p[1:]), 1)
    return counts / counts.sum()
