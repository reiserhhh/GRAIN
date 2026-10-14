#!/usr/bin/env python3
"""HM24 alternate QR/pullback and actual finite-readout arithmetic replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_decoder_fibers import OUT, check
from scripts.run_grain_math_response import write_json

VERIFY = ROOT / "results/grain_math_response_v1/decoder_fibers_verification_r1"


def native_score(record, alphabet):
    """Recompute each native endpoint using extended precision softmax."""
    z = np.asarray(record["candidate_logits"], np.longdouble)
    p = np.exp(z - z.max())
    return float(np.sum(p * alphabet) / p.sum())


def verify(packet, out):
    """Check all addresses and numerical accounts without rerunning the model."""
    if out.exists():
        raise FileExistsError("preserve HM24 verification")
    m = check(packet)
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(packet / name) != sha:
            raise ValueError(f"changed result {name}")
    out.mkdir(parents=True)
    (out / "verifier.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "frozen_primary_artifact_hashes.json", ledger)
    errors, counts = {}, {"units": 0, "directions": 0, "spans": 0}

    def compare(a, b, name):
        err = float(np.max(abs(np.asarray(a) - np.asarray(b)), initial=0))
        errors[name] = max(errors.get(name, 0.0), err)
        if not np.allclose(a, b, atol=1e-10, rtol=1e-7):
            raise ValueError(f"alternate arithmetic {name}: {err}")

    try:
        alphabet = np.asarray(m["alphabet"]["values"], np.longdouble)
        for row in read_json(packet / "00/design.json"):
            path = packet / "01/units" / f"unit_{row['hm24_unit']:03d}.json"
            r = read_json(path)
            if (
                row != r["address"]
                or digest(path.with_suffix(".npz")) != r["arrays_sha256"]
            ):
                raise ValueError("address/hash mismatch")
            with np.load(path.with_suffix(".npz")) as p:
                arrays = {k: p[k].copy() for k in p.files}
            gradients, visible = {}, {}
            tangent, pullback = 0.0, 0.0
            with (
                np.load(ROOT / row["reference_npz"]) as ref,
                np.load(ROOT / row["donor_npz"]) as donor,
            ):
                for name in row["capture"]["spans"]:
                    g = arrays["g_" + name]
                    t = len(g)
                    degree = 3 if name.endswith("property") else 0
                    pos = np.zeros(1) if t == 1 else np.linspace(-1, 1, t)
                    o = np.vander(pos, N=degree + 1, increasing=True).T / t
                    # The actual property lengths are <=4; all other rows are means.
                    v = (
                        g.copy()
                        if degree == 3
                        else np.repeat(g.mean(axis=0, keepdims=True), t, axis=0)
                    )
                    gradients[name], visible[name] = g, v
                    if degree == 3:
                        q, rr = np.linalg.qr(o.T)
                        covector = np.linalg.lstsq(rr, q.T @ g, rcond=None)[0]
                    else:
                        covector = g.sum(axis=0, keepdims=True)
                    compare(o.T @ covector, v, "pullback_reconstruction")
                    geometry = r["span_geometry"][name]
                    if geometry["rank"] != (t if degree == 3 else 1):
                        raise ValueError("rank mismatch")
                    compare(o @ (g - v), 0.0, "null_observation")
                    if name == row["span"]:
                        delta = (
                            donor["L16_" + name].astype(np.float32)
                            - ref["L16_" + name].astype(np.float32)
                        ).astype(float)
                        tangent += float(np.sum(g.astype(np.longdouble) * delta))
                        pullback += float(
                            np.sum(covector.astype(np.longdouble) * (o @ delta))
                        )
                    counts["spans"] += 1
            compare(tangent, r["full_tangent_action"], "full_tangent")
            compare(pullback, r["full_moment_pullback_action"], "full_pullback")
            old = read_json(ROOT / row["unit_path"])["native"]
            reference = read_json(ROOT / row["reference_path"])["native"]
            actual = native_score(old, alphabet) - native_score(reference, alphabet)
            compare(actual, r["full_native_action"], "full_native")
            compare(actual - tangent, r["full_nonlinear_residual"], "full_remainder")
            for d in r["directions"]:
                arm = d["arm"]
                total, vis, observed = 0.0, 0.0, 0.0
                for name, g in gradients.items():
                    delta = (arrays[f"{arm}_1_{name}"] - arrays[f"{arm}_-1_{name}"]) / 2
                    total += float(np.sum(g.astype(np.longdouble) * delta))
                    vis += float(np.sum(visible[name].astype(np.longdouble) * delta))
                    t, degree = len(g), 3 if name.endswith("property") else 0
                    pos = np.zeros(1) if t == 1 else np.linspace(-1, 1, t)
                    o = np.vander(pos, N=degree + 1, increasing=True).T / t
                    observed = max(observed, float(np.max(abs(o @ delta))))
                central = (
                    native_score(d["plus_native"], alphabet)
                    - native_score(d["minus_native"], alphabet)
                ) / 2
                for key, expected in (
                    ("central_action", central),
                    ("gradient_action", total),
                    ("visible_leak_action", vis),
                    ("null_gradient_action", total - vis),
                    ("moment_change_max", observed),
                    ("nonlinear_residual", central - total),
                ):
                    compare(d[key], expected, key)
                counts["directions"] += 1
            counts["units"] += 1
        if counts != {"units": 48, "directions": 144, "spans": 288}:
            raise ValueError(f"incomplete verification {counts}")
        write_json(
            out / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM24_NOT_GLOBAL_SCOPE_COMPLETE",
                "counts": counts,
                "max_errors": errors,
                "new_model_calls": 0,
                "verifier_sha256": digest(Path(__file__)),
                "same_agent_alternate_arithmetic_not_independent_replication": True,
                "gradient_oracle_not_LLM_free_prediction": True,
            },
        )
        print(read_json(out / "summary.json"), flush=True)
    except BaseException as exc:
        write_json(
            out / "failure.json",
            {"error": repr(exc), "counts": counts, "max_errors": errors},
        )
        raise


def main():
    """Run one alternate finite verification in a fresh namespace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=VERIFY)
    args = parser.parse_args()
    verify(args.packet, args.out)


if __name__ == "__main__":
    main()
