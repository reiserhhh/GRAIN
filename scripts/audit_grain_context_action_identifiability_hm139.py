#!/usr/bin/env python3
"""Source-equivalent function extensions; never read fresh native outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_context_action_transport_hm139 as H


def extension_geometry(source, application, scale, rtol=1e-10):
    """Find action directions left unidentified by source, in fixed source units."""
    a, b = np.asarray(source, float) / scale, np.asarray(application, float) / scale
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Matched source/application action dictionaries required")
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    rank = int(np.sum(s > s[0] * rtol)) if len(s) else 0
    n = vt[rank:]
    residual = b @ n.T @ n
    alternate = b - b @ np.linalg.pinv(a, rcond=rtol) @ a
    norms = np.linalg.norm(residual, axis=1)
    index = int(np.argmax(norms))
    direction = residual[index] / norms[index] if norms[index] else np.zeros(b.shape[1])
    effect = b @ direction
    total = float(np.sum(b * b))
    result = {
        "parameters": a.shape[1],
        "source_rank": rank,
        "unidentified_parameters": n.shape[0],
        "application_action_unseen_energy_fraction": float(
            np.sum(residual * residual) / total
        )
        if total
        else None,
        "maximum_unseen_action_norm": float(norms.max()),
        "source_unit_coefficient_witness_max_change": float(np.max(abs(a @ direction))),
        "application_unit_coefficient_witness_max_change": float(np.max(abs(effect))),
        "witness_stratum": index,
        "alternate_projection_error": float(np.max(abs(residual - alternate))),
    }
    return result, {
        "null_basis": n,
        "unseen_actions": residual,
        "coefficient_difference_raw": direction / scale,
        "application_difference": effect,
    }


def compute(out):
    """Only source arrays, frozen coefficients and cold coordinates enter."""
    H.check(out)
    bundle = H.H.read(out / "source_bundle.json")
    records = H.H.read(out / "cold_input.json")
    g = np.array([r["g"] for r in records])
    real = np.array([r["real_context"] for r in records])
    sham = np.array([r["sham_context"] for r in records])
    features = {
        "global": H.M.features(g),
        "real": H.M.features(g, real),
        "sham": H.M.features(g, sham),
    }
    result, arrays = {}, {}
    with np.load(out / "source_actions.npz", allow_pickle=False) as z:
        for name in features:
            model = bundle["models"][name]
            result[name], values = extension_geometry(
                z[name], H.contrasts(features[name]), np.array(model["scale"])
            )
            arrays.update({name + "__" + k: v for k, v in values.items()})
    return result, arrays


def main():
    """Freeze and verify one no-outcome diagnostic; preserve prior artifacts."""
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    out = ROOT / H.H.read(H.SPEC)["out"]
    dst = out / "source_identifiability"
    with threadpool_limits(limits=1):
        if args.verify:
            for path, digest in H.H.read(dst / "seal.json").items():
                if H.H.sha(path) != digest:
                    raise ValueError("Changed diagnostic code/input/output")
        else:
            dst.mkdir()
            inputs = [
                Path(__file__),
                ROOT / "tests/test_grain_context_action_identifiability.py",
                out / "source_bundle.json",
                out / "source_actions.npz",
                out / "cold_input.json",
                out / "cold_seal.json",
            ]
            H.H.write(
                dst / "manifest.json",
                {
                    "inputs": {str(f): H.H.sha(f) for f in inputs},
                    "question": "Source full row rank is not parameter identification on new action vectors",
                    "native_outcomes_read": False,
                    "new_fits": 0,
                    "unit_bound": "unit coefficient difference in source-RMS norm;not psychological CI",
                    "original_scope_complete": False,
                },
            )
        result, arrays = compute(out)
        if max(v["alternate_projection_error"] for v in result.values()) > 1e-8:
            raise ValueError("Alternate source quotient computation disagrees")
        if args.verify:
            if result != H.H.read(dst / "result.json"):
                raise ValueError("Diagnostic summary differs")
            with np.load(dst / "directions.npz", allow_pickle=False) as z:
                if set(z.files) != set(arrays) or any(
                    not np.array_equal(z[k], v) for k, v in arrays.items()
                ):
                    raise ValueError("Full diagnostic array replay differs")
        else:
            H.H.write(dst / "result.json", result)
            np.savez(dst / "directions.npz", **arrays)
            manifest = H.H.read(dst / "manifest.json")
            files = [
                dst / n for n in ("manifest.json", "result.json", "directions.npz")
            ]
            H.H.write(
                dst / "seal.json",
                manifest["inputs"] | {str(f): H.H.sha(f) for f in files},
            )
        print("READONLY VERIFIED" if args.verify else "DIAGNOSTIC", result, flush=True)


if __name__ == "__main__":
    main()
