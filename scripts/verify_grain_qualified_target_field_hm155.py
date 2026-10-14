#!/usr/bin/env python3
"""Recompute finite scalar accounts and fixed full-vocabulary spots read-only."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_qualified_target_field_hm155 as R


def verify(out):
    """Use explicit sums, CPU64 projection and actual primitive logits."""
    start = time.monotonic()
    manifest = R.check(out)
    for section in ("capture_seal.json", "output_seal.json"):
        for name, checksum in R.read(out / section).items():
            if R.H.digest(out / name) != checksum:
                raise ValueError("Immutable output changed: " + name)
    cubes, captured = R.observation_cubes(out)
    weight = R.OLD.weights().cpu()
    queries = R.read(out / "queries.json")
    ids = [queries["answers"][s][0] for s in ("Yes", "No")]
    selected_weight = weight[ids].numpy().astype(float)
    maximum_native, maximum_algebra, checks = 0.0, 0.0, 0
    for observation in captured.values():
        expected = observation["states"].astype(float) @ selected_weight.T
        err = float(np.max(abs(expected - observation["selected_logits"])))
        maximum_native = max(maximum_native, err)
        if not np.allclose(
            expected, observation["selected_logits"], atol=1e-4, rtol=1e-5
        ):
            raise ValueError("Actual endpoint does not reconstruct via W*h")
        checks += expected.size
    for gi, group in enumerate(R.GROUPS):
        a = R.arrays(out / "accounts" / f"{gi:02d}.npz")
        for mode in R.MODES:
            for name in ("states", "selected_logits", "normalizer", "logodds"):
                cube = cubes[group, mode][name]
                for mask in range(8):
                    total = np.zeros_like(cube[0, 0, 0], dtype=float)
                    for ie, inn, ip in itertools.product(range(2), repeat=3):
                        signs = [2 * ie - 1, 2 * inn - 1, 2 * ip - 1]
                        sign = np.prod([signs[j] for j in range(3) if mask & (1 << j)])
                        total += sign * cube[ie, inn, ip] / 8
                    err = float(np.max(abs(total - a[mode + "_" + name][mask])))
                    maximum_algebra = max(maximum_algebra, err)
                    if err > 1e-10:
                        raise ValueError("Independent character sum failed")
                    checks += total.size
        cs = [cubes[group, mode]["logodds"] for mode in R.MODES]
        for axis in range(2):
            b, e, n, both, whole = [c[..., axis, :] for c in cs]
            closure = (
                (e - b) + (n - b) + (both - e - n + b) + (whole - both) - (whole - b)
            )
            maximum_algebra = max(maximum_algebra, float(np.max(abs(closure))))
            checks += closure.size
    spots, max_gram = [], 0.0
    for gi in sorted({g for g, _ in manifest["fixed_vocab_spots"]}):
        a = R.arrays(out / "accounts" / f"{gi:02d}.npz")
        actual = R.arrays(out / "accounts" / f"{gi:02d}_gram.npz")
        # Materialize only seven full fields; chunk CPU64 weights, not a 2GB copy.
        vectors = np.r_[a["qualified_fields"], a["reference_state"][None]]
        projected = np.empty((7, weight.shape[0]), dtype=float)
        for j in range(0, weight.shape[0], 4096):
            if time.monotonic() - start > manifest["audit_seconds_cap"]:
                raise TimeoutError("Original120second audit cap; no renewal")
            projected[:, j : j + 4096] = (
                vectors @ weight[j : j + 4096].numpy().astype(float).T
            )
        q = np.exp(projected[6] - logsumexp(projected[6]))
        means = projected[:6] @ q
        centered = projected[:6] - means[:, None]
        gram = (centered * q[None]) @ centered.T
        for f in range(3):
            err = float(np.max(abs(gram[f] - actual["gram"][f])))
            max_gram = max(max_gram, err)
            if not np.allclose(gram[f], actual["gram"][f], atol=1e-4, rtol=1e-4):
                raise ValueError("Fixed full-vocabulary Gram spot failed")
            spots.append({"group": gi, "family": f, "max_error": err})
            checks += len(gram[f])
    result = {
        "status": "PASS_FINITE_ALTERNATE_NOT_BLIND",
        "numerical_checks": checks,
        "max_selected_logit_error": maximum_native,
        "max_algebra_error": maximum_algebra,
        "full_vocab_spots": spots,
        "max_spot_gram_error": max_gram,
        "seconds": time.monotonic() - start,
        "new_model_forwards": 0,
    }
    path = out / "alternate_audit.json"
    if path.exists():
        print(result)
    else:
        R.H.write_json(path, result)
        print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=R.DEFAULT)
    verify(parser.parse_args().out)
