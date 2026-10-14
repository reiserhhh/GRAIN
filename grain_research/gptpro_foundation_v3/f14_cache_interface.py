#!/usr/bin/env python3
"""Actual finite-value, row-address and author-partition checks for cached views."""

import numpy as np


def check_interface(arrays, addresses, reference_addresses, source, held, n_authors):
    """Validate declared numerical views without reading text or questionnaire data."""
    required = {"g", "h", "semantic", "scores", "authors", "reference_authors"}
    if set(arrays) != required:
        raise ValueError("unexpected cached array contract")
    if type(n_authors) is not int or n_authors <= 0:
        raise ValueError("positive author count required")
    n = len(arrays["authors"])
    for name, value in arrays.items():
        a = np.asarray(value)
        if a.ndim < 1 or len(a) != n or not n:
            raise ValueError(f"unaligned rows: {name}")
        if not np.isfinite(a).all():
            raise ValueError(f"nonfinite array: {name}")
    authors = np.asarray(arrays["authors"])
    if authors.ndim != 1 or not np.array_equal(authors, arrays["reference_authors"]):
        raise ValueError("author views mismatch")
    if np.any(authors != np.floor(authors)) or np.any(
        (authors < 0) | (authors >= n_authors)
    ):
        raise ValueError("author index outside declared population")
    left, right = np.asarray(addresses), np.asarray(reference_addresses)
    if left.shape != (n, 4) or right.shape != (n, 4) or not np.array_equal(left, right):
        raise ValueError("material address mismatch despite possible author agreement")
    if not np.isfinite(left).all() or np.any(left != np.floor(left)):
        raise ValueError("integer material addresses required")
    if (
        not np.array_equal(left[:, 0], authors)
        or np.any(left[:, 1:] < 0)
        or np.any(left[:, 3] <= left[:, 2])
    ):
        raise ValueError("invalid material address or empty span")
    if len(np.unique(left, axis=0)) != n:
        raise ValueError("duplicate material addresses")
    partitions = []
    for values in (source, held):
        v = np.asarray(values)
        if v.ndim != 1 or not np.isfinite(v).all() or np.any(v != np.floor(v)):
            raise ValueError("integer partition identifiers required")
        if len(set(v.tolist())) != len(v):
            raise ValueError("duplicate partition author")
        partitions.append(set(v.tolist()))
    s, h = partitions
    if s & h or s | h != set(range(n_authors)):
        raise ValueError("overlap or incomplete author partition")
    counts = np.bincount(authors.astype(int), minlength=n_authors)
    return {
        "units": n,
        "authors": n_authors,
        "source_authors": len(s),
        "held_authors": len(h),
        "authors_without_units": int(np.sum(counts == 0)),
        "single_unit_authors": int(np.sum(counts == 1)),
        "all_numeric_values_finite": True,
        "all_material_addresses_equal": True,
        "duplicate_material_addresses": 0,
        "partition_covers_declared_authors": True,
    }
