#!/usr/bin/env python3
"""Post-primary symbolic and coordinate audit, not an outcome-selected model."""

from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_observer_parity_hm96 import (
    check,
    load,
    seal,
    settings,
    sha,
    verify,
    write,
)


def triangle(points):
    """Return an antisymmetric oriented polygon tensor in the given units."""
    return (
        sum(
            np.outer(points[i], points[(i + 1) % 3])
            - np.outer(points[(i + 1) % 3], points[i])
            for i in range(3)
        )
        / 2
    )


def run():
    """Separate exact feature collisions from floating-point approximations."""
    spec, source, out = settings()
    check(out)
    verify(out, load(out / "01_seal.json"))
    verify(out, load(out / "02_seal.json"))
    if load(out / "verification.json")["status"] != "VERIFIED_FINITE_HM96_NOT_GLOBAL":
        raise ValueError("Primary finite verification must precede this diagnostic")
    dest = out / "04_symbolic_companion"
    dest.mkdir()
    rows = [r for r in load(source / "00/rows.json") if r["kind"] == "pair"]
    lookup = {
        (r["lexicon"], r["frame"], r["family"], r["previous"], r["current"]): r
        for r in rows
    }
    analyzer = joblib.load(ROOT / spec["runtime"]).vectorizer.build_analyzer()
    equality = []
    for key, row in lookup.items():
        if row["previous"] >= row["current"]:
            continue
        reverse = lookup[*key[:3], key[4], key[3]]
        left, right = Counter(analyzer(row["body"])), Counter(analyzer(reverse["body"]))
        equality.append(
            {
                "left": row["unit"],
                "right": reverse["unit"],
                "analyzed_counts_equal": left == right,
                "unique_ngrams": len(left),
            }
        )
    units = load(out / "01/units.json")
    scores = np.load(out / "01/coordinates.npz")["factors"]
    index = {(r["unit"], r["view"], r["slice"]): i for i, r in enumerate(units)}
    rotation, _ = np.linalg.qr(np.random.default_rng(20261010).normal(size=(4, 4)))
    errors, norms = [], []
    for row in rows:
        points = scores[[index[row["unit"], "event", i] for i in range(3)]]
        original = triangle(points)
        transformed = triangle(points @ rotation.T)
        errors.append(float(abs(transformed - rotation @ original @ rotation.T).max()))
        norms.append(float(abs(np.linalg.norm(transformed) - np.linalg.norm(original))))
    payload = {
        "status": "VERIFIED_POST_PRIMARY_SYMBOLIC_COMPANION",
        "code_sha256": sha(Path(__file__)),
        "paired_analyzers_equal": sum(r["analyzed_counts_equal"] for r in equality),
        "pairs": len(equality),
        "outside_frozen_vocabulary_counts_checked": True,
        "event_records_rotated": len(rows),
        "orthogonal_coordinate_tensor_max_error": max(errors),
        "orthogonal_coordinate_norm_max_error": max(norms),
        "no_claim_all_affine_norm_invariance": True,
        "no_new_scoring_fit_or_model_calls": True,
        "not_new_preregistered_primary": True,
        "scope_complete": False,
    }
    if (
        not all(r["analyzed_counts_equal"] for r in equality)
        or max(errors + norms) > 1e-10
    ):
        raise ValueError("Symbolic or coordinate diagnostic failed")
    write(dest / "ngram_equality.json", equality)
    write(dest / "summary.json", payload)
    write(out / "04_symbolic_seal.json", seal(out, dest))
    print(payload, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
