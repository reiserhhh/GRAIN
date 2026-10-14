#!/usr/bin/env python3
"""HM25 alternate explicit quadrature and physical-unit contractions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_decoder_paths import OUT, check, node_path
from scripts.run_grain_math_response import write_json

VERIFY = ROOT / "results/grain_math_response_v1/decoder_paths_verification_r1"


def explicit_simpson(values, intervals):
    """Sum in extended precision without the primary quadrature weights."""
    y = np.asarray(values, np.longdouble)
    if len(y) != intervals + 1:
        raise ValueError("unexpected fixed quadrature length")
    return float(
        (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()) / (3 * intervals)
    )


def candidate(logits, alphabet):
    """Recompute native score and reference logit covector independently."""
    z, a = np.asarray(logits, np.longdouble), np.asarray(alphabet, np.longdouble)
    p = np.exp(z - z.max())
    p /= p.sum()
    f = p @ a
    return float(f), p * (a - f)


def verify(packet, out):
    """Verify all432 addressed nodes and48 path accounts without new calls."""
    if out.exists():
        raise FileExistsError("preserve verification results")
    m = check(packet)
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(packet / name) != sha:
            raise ValueError("HM25 artifact changed " + name)
    out.mkdir(parents=True)
    (out / "verifier.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "primary_artifact_hashes.json", ledger)
    errors, counts = {}, {"units": 0, "nodes": 0, "gradient_arrays": 0}

    def compare(a, b, label):
        err = float(np.max(abs(np.asarray(a) - np.asarray(b)), initial=0))
        errors[label] = max(errors.get(label, 0.0), err)
        if not np.allclose(a, b, atol=1e-10, rtol=1e-7):
            raise ValueError(f"alternate arithmetic {label}: {err}")

    try:
        with (packet / "02/path_accounts.csv").open() as f:
            accounts = {int(r["unit"]): r for r in csv.DictReader(f)}
        for address in read_json(packet / "00/design.json"):
            unit, span = address["hm24_unit"], address["span"]
            row = accounts[unit]
            if (int(row["family"]), row["role"], row["query"]) != (
                address["family"],
                address["role"],
                address["query"],
            ):
                raise ValueError("summary address mismatch")
            nodes, ds = [], {"native": [], "frozen": []}
            with (
                np.load(ROOT / address["reference_body_npz"]) as ref,
                np.load(ROOT / address["donor_npz"]) as donor,
            ):
                h0, h1 = (
                    ref["L16_" + span].astype(float),
                    donor["L16_" + span].astype(float),
                )
            delta = h1 - h0
            for k, t in enumerate(m["spec"]["path_nodes"]):
                path = node_path(packet, unit, k)
                r = read_json(path)
                if (
                    r["address"] != address
                    or r["node"] != k
                    or r["t"] != t
                    or digest(path.with_suffix(".npz")) != r["arrays_sha256"]
                ):
                    raise ValueError("path node mismatch")
                with np.load(path.with_suffix(".npz")) as arrays:
                    compare(arrays["path_delta"], delta, "ideal_path_delta")
                    compare(
                        arrays["actual_property_state"],
                        (h0 + t * delta).astype(np.float32).astype(float),
                        "actual_node_state",
                    )
                    for channel in ds:
                        value = float(
                            np.sum(
                                arrays[f"g_{channel}_{span}"].astype(np.longdouble)
                                * delta.astype(np.longdouble)
                            )
                        )
                        compare(
                            r[channel + "_derivative"], value, channel + "_derivative"
                        )
                        ds[channel].append(value)
                    for name in address["capture"]["spans"]:
                        for channel in ds:
                            g = arrays[f"g_{channel}_{name}"]
                            if (
                                g.shape
                                != (
                                    len(address["capture"]["spans"][name]["positions"]),
                                    h0.shape[1],
                                )
                                or not np.isfinite(g).all()
                            ):
                                raise ValueError("gradient shape/value")
                            counts["gradient_arrays"] += 1
                f, _ = candidate(
                    r["native"]["candidate_logits"], m["alphabet"]["values"]
                )
                compare(r["native"]["conditional_score"], f, "native_readout")
                nodes.append(r)
                counts["nodes"] += 1
            z0, z1 = (
                np.asarray(nodes[k]["native"]["candidate_logits"], np.longdouble)
                for k in (0, -1)
            )
            f0, g0 = candidate(z0, m["alphabet"]["values"])
            f1, _ = candidate(z1, m["alphabet"]["values"])
            a, b, d0 = f1 - f0, float(g0 @ (z1 - z0)), ds["native"][0]
            fine, coarse, frozen = (
                explicit_simpson(ds["native"], 8),
                explicit_simpson(ds["native"][::2], 4),
                explicit_simpson(ds["frozen"], 8),
            )
            y = np.asarray(ds["native"], np.longdouble)
            trap = float((y[0] / 2 + y[-1] / 2 + y[1:-1].sum()) / 8)
            expected = {
                "native_action": a,
                "start_tangent_action": d0,
                "frozen_readout_logit_action": b,
                "decoder_nonlinear_increment": b - d0,
                "readout_nonlinear_increment": a - b,
                "full_nonlinear_increment": a - d0,
                "integral_native": fine,
                "integral_frozen": frozen,
                "integral_readout_increment": fine - frozen,
                "fine_completeness_error": fine - a,
                "coarse_completeness_error": coarse - a,
                "fine_minus_coarse": fine - coarse,
                "trapezoid_completeness_error": trap - a,
                "frozen_completeness_error": frozen - b,
                "minimum_derivative": float(y.min()),
                "maximum_derivative": float(y.max()),
                "reference_native_frozen_gradient_gap": d0 - ds["frozen"][0],
            }
            for key, value in expected.items():
                compare(float(row[key]), value, key)
            if int(row["derivative_sign_changes"]) != sum(y[1:] * y[:-1] < 0):
                raise ValueError("path sign count")
            compare(a - d0, (b - d0) + (a - b), "increment_account_identity")
            counts["units"] += 1
        if counts != {"units": 48, "nodes": 432, "gradient_arrays": 5184}:
            raise ValueError("incomplete finite replay")
        write_json(
            out / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM25_NOT_GLOBAL",
                "counts": counts,
                "max_errors": errors,
                "new_model_calls": 0,
                "same_agent_alternate_arithmetic_not_independent_replication": True,
                "quadrature_error_remains_physical_not_a_verification_failure": True,
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
    """Run one alternate finite replay in its own preserved namespace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=VERIFY)
    args = parser.parse_args()
    verify(args.packet, args.out)


if __name__ == "__main__":
    main()
