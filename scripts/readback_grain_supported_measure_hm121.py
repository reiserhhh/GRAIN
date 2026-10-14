#!/usr/bin/env python3
"""Alternate-formula readback of the supported-measure experiment; no neural run."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H


def audit(out):
    """Recompute marked integrals, endpoint contrasts and every family matching."""
    m, result = H.load(out / "00/manifest.json"), H.load(out / "02/summary.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    H.verify_hashes(out, H.load(out / "01/seal.json"))
    with np.load(out / "02/moments.npz", allow_pickle=False) as stream:
        z = {key: stream[key] for key in stream.files}
    a, g, mask, e = z["action"], z["marks"], z["visible"], z["endpoints"]
    expected = {
        key: np.zeros_like(z[key])
        for key in (
            "mass",
            "body_mass",
            "untagged_mass",
            "first",
            "connected",
            "misaligned_connected",
            "total_variation",
        )
    }
    length = a.shape[-1] // 2
    valid = np.asarray(m["input_descriptor"]["valid"], float)
    reference = (
        np.concatenate((valid[z["ids"][:, 0]], valid[z["ids"][:, 1]]), axis=1) * mask
    )
    if not np.array_equal(reference, mask.astype(float)):
        raise AssertionError(
            "Unweighted center is not the declared physical token measure"
        )
    for p in range(len(a)):
        ix = np.flatnonzero(mask[p])
        average = sum(g[p, i] for i in ix) / len(ix)
        wrong = np.zeros_like(g[p])
        for side in range(2):
            selected = ix[(ix >= side * length) & (ix < (side + 1) * length)]
            for j, i in enumerate(selected):
                wrong[i] = g[p, selected[(j - len(selected) // 2) % len(selected)]]
        for mode, q in np.ndindex(2, 6):
            values = a[p, mode, q]
            expected["mass"][p, mode, q] = sum(values)
            expected["body_mass"][p, mode, q] = sum(values[ix])
            expected["untagged_mass"][p, mode, q] = sum(values[~mask[p]])
            expected["total_variation"][p, mode, q] = sum(abs(values))
            expected["first"][p, mode, q] = sum(values[i] * g[p, i] for i in ix)
            expected["connected"][p, mode, q] = sum(
                values[i] * (g[p, i] - average) for i in ix
            )
            expected["misaligned_connected"][p, mode, q] = sum(
                values[i] * (wrong[i] - average) for i in ix
            )
    errors = {key: float(abs(value - z[key]).max()) for key, value in expected.items()}
    truth = (e[1, :, 0] - e[1, :, 1]) - (e[0, :, 0] - e[0, :, 1])
    residual = (
        np.stack(
            [
                np.array([[sum(a[p, mode, q]) for q in range(6)] for mode in range(2)])
                for p in range(len(a))
            ]
        )
        - truth
    )
    errors["truth"] = float(abs(truth - z["truth"]).max())
    errors["residual"] = float(abs(residual - z["residual"]).max())
    errors["completeness_fraction"] = abs(
        float(np.linalg.norm(residual) / np.linalg.norm(truth))
        - result["completeness_RMS_fraction"]
    )
    src = ROOT / m["spec"]["source"]
    with np.load(src / "02/moments.npz", allow_pickle=False) as stream:
        errors["endpoint_reference"] = abs(
            float(abs(e - stream["endpoints"][:, : len(a)]).max())
            - result["endpoint_reference_max_error"]
        )
    errors["gauge_reading"] = abs(
        float(abs(z["midpoint"][0] - z["midpoint"][1]).max())
        - result["midpoint_gauge_reading_max_error"]
    )
    pairs = H.load(out / "00/pairs.json")
    if z["ids"].tolist() != [[p["low"], p["high"]] for p in pairs[: len(a)]]:
        raise AssertionError("Missing or reordered paired evidence")
    permutations = list(itertools.permutations(range(6)))
    permutation_error = 0.0
    for row in result["comparisons"]:
        field = np.zeros((3, 2, 6, 2, 2, *expected[row["map"]].shape[1:]))
        for p, value in zip(pairs, expected[row["map"]], strict=True):
            v, w, f, c, o = p["key"]
            field[v, w, f, (c + 1) // 2, (o + 1) // 2] = value
        values = field[:, :, :, :, 1]
        if row["component"] == "actor_odd":
            values = (values - field[:, :, :, :, 0]) / 2
        s = values[0, 0, :, :, row["mode"]].reshape(6, 2, -1)
        t = values[row["variant"], row["wording"], :, :, row["mode"]].reshape(6, 2, -1)
        if row["status"] != "OBSERVED_DIRECTION":
            if not (
                (np.linalg.norm(s, axis=-1) <= 1e-12)
                | (np.linalg.norm(t, axis=-1) <= 1e-12)
            ).any():
                raise AssertionError("False unavailable direction")
            continue
        matrix = np.empty((2, 6, 6))
        for c, f, other in np.ndindex(2, 6, 6):
            matrix[c, f, other] = np.dot(s[f, c], t[other, c]) / (
                np.linalg.norm(s[f, c]) * np.linalg.norm(t[other, c])
            )
        checks = np.array(
            [
                sum(matrix[c, f, perm[f]] for c in range(2) for f in range(6)) / 12
                for perm in permutations
            ]
        )
        permutation_error = max(
            permutation_error,
            float(abs(checks - row["permutation_values"]).max()),
            abs(float(checks[0]) - row["matched_mean_cosine"]),
        )
        if int(np.sum(checks >= checks[0] - 1e-12)) != row["matching_rank_with_ties"]:
            raise AssertionError("Permutation rank differs")
    errors["all_matchings"] = permutation_error
    if max(errors.values()) > 1e-10:
        raise AssertionError(errors)
    numeric_ok = (
        result["status"] == "COMPLETE_SUPPORTED_MEASURE_NOT_GLOBAL"
        and result["completeness_RMS_fraction"]
        <= m["spec"]["localization_completeness_RMS_fraction_limit"]
        and result["endpoint_reference_max_error"]
        <= m["spec"]["endpoint_rounding_tolerance"]
        and result["midpoint_gauge_reading_max_error"]
        <= m["spec"]["gauge_reading_tolerance"]
        and result["pack_invariance_max_error"] == 0
    )
    if numeric_ok != result["localization_resolved"]:
        raise AssertionError("Wrong resolution decision")
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise AssertionError("Neural runtime loaded by numerical readback")
    H.write(
        out / "02/readback.json",
        {
            "status": "NUMERIC_READBACK_PASSED_NOT_GLOBAL",
            "errors": errors,
            "pairs": len(a),
            "moment_scalars": sum(value.size for value in expected.values()),
            "comparisons": len(result["comparisons"]),
            "permutations_each": 720,
            "same_agent_not_third_party": True,
            "neural_runtime_loaded": False,
            "audit_code_sha256": H.sha(__file__),
            "primary_failures_retained": True,
        },
    )
    H.write(out / "02/seal.json", H.seals(out, out / "02"))
    print("HM121 numeric readback", errors, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output.resolve())
