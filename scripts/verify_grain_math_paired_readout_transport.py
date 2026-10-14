#!/usr/bin/env python3
"""HM21 alternate softmax, Hessian cumulants and complete table/address audit."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_paired_readout_transport import DEFAULT, check
from scripts.run_grain_math_response import write_json
from scripts.run_grain_upstream_routes import collector_guard

OUTPUT = (
    ROOT / "results/grain_math_response_v1/paired_readout_transport_verification_r1"
)


def rows(path):
    """Read the exact original table, excluding filesystem sidecars."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def probability(z):
    """Compute softmax by max/exponent/sum without the primary implementation."""
    x = np.asarray(z, float)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def weights(families):
    """Derive equal-family probability masses from the actual address counts."""
    counts = Counter(families)
    return np.array([1 / len(counts) / counts[f] for f in families])


def covariance(x, y, w):
    """Contract centered scalar columns directly, independently of primary moments."""
    return float(
        np.einsum(
            "i,i,i->", w, x - np.einsum("i,i->", w, x), y - np.einsum("i,i->", w, y)
        )
    )


def verify(packet, out):
    """Verify every native record and both integrated actions with fixed order64."""
    if out.exists():
        raise FileExistsError("preserve earlier verification")
    m = check(packet)
    spec = m["spec"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(packet / name) != sha:
            raise ValueError(f"changed primary artifact: {name}")
    out.mkdir(parents=True)
    write_json(out / "frozen_primary_artifact_hashes.json", ledger)
    (out / "verifier.py").write_bytes(Path(__file__).read_bytes())
    errors, counts = defaultdict(float), Counter()

    def compare(saved, expected, category):
        """Keep unknown values unknown and distinguish addresses from numerical checks."""
        for key, value in expected.items():
            actual = saved[key]
            if value is None:
                if actual != "":
                    raise ValueError(f"{category}/{key}: undefined value filled")
            elif isinstance(value, (bool, np.bool_)):
                if actual != str(bool(value)):
                    raise ValueError(f"{category}/{key}: boolean changed")
            elif isinstance(value, (int, float, np.number)):
                err = abs(float(actual) - value)
                errors[category] = max(errors[category], float(err))
                if not np.isclose(
                    float(actual),
                    value,
                    atol=spec["arithmetic_absolute_tolerance"],
                    rtol=spec["aggregate_relative_tolerance"],
                ):
                    raise ValueError(f"{category}/{key}: {actual} != {value}")
            elif isinstance(value, (list, dict)):
                if read_json_value(actual) != value:
                    raise ValueError(f"{category}/{key}: structured address changed")
            elif actual != str(value):
                raise ValueError(f"{category}/{key}: address changed")
        counts[category] += 1

    try:
        meta = read_json(packet / "00/metadata.json")
        alphabet = read_json(packet / "00/alphabet.json")
        original_alphabet = read_json(ROOT / spec["span_packet"] / "00/manifest.json")[
            "alphabet"
        ]
        if alphabet != original_alphabet:
            raise ValueError("original answer units changed")
        a = np.array(alphabet["values"])
        actions = rows(packet / "02/actions.csv")
        if len(actions) != spec["paired_query_rows"]:
            raise ValueError("missing action addresses")
        nodes, qw = np.polynomial.legendre.leggauss(spec["alternate_quadrature_order"])
        u, qw = (nodes + 1) / 2, qw / 2
        values = {}
        with np.load(packet / "02/endpoints.npz") as data:
            for address, saved in zip(meta, actions, strict=True):
                i = address["row"]
                unit, ref = (
                    read_json(ROOT / address[n])
                    for n in ("unit_path", "reference_path")
                )
                r, t = (np.array(v["native"]["candidate_logits"]) for v in (ref, unit))
                np.testing.assert_array_equal(r, data["reference_logits"][i])
                np.testing.assert_array_equal(t, data["target_logits"][i])
                rc, tc = r - np.sum(r) / len(r), t - np.sum(t) / len(t)
                d = tc - rc
                pr, pt = probability(r), probability(t)
                fr, ft = float(pr @ a), float(pt @ a)
                for record, p, f in ((ref, pr, fr), (unit, pt, ft)):
                    errors["stored_probability"] = max(
                        errors["stored_probability"],
                        float(
                            np.max(
                                abs(p - record["native"]["conditional_probabilities"])
                            )
                        ),
                    )
                    errors["stored_score"] = max(
                        errors["stored_score"],
                        abs(f - record["native"]["conditional_score"]),
                    )
                g = pr * (a - fr)
                y, l = ft - fr, float(g @ d)
                c = y - l
                # The scalar third central moment is the Hessian contraction,
                # avoiding the primary explicit 10x10 Hessian matrices.
                path = probability(rc + u[:, None] * d)
                fpath = path @ a
                pd = path @ d
                h = np.sum(path * (a - fpath[:, None]) * (d - pd[:, None]) ** 2, axis=1)
                c64 = float((qw * (1 - u)) @ h)
                digit64 = d * (qw @ (path * (a - fpath[:, None])))
                gradient64 = float(np.sum(digit64))
                fisher = float(np.sum(pr * (d - pr @ d) ** 2))
                lr, lt = (
                    rc - (np.max(rc) + np.log(np.exp(rc - np.max(rc)).sum())),
                    tc - (np.max(tc) + np.log(np.exp(tc - np.max(tc)).sum())),
                )
                kl = float(pt @ (lt - lr))
                bregman = float(
                    np.max(rc)
                    + np.log(np.exp(rc - np.max(rc)).sum())
                    - np.max(tc)
                    - np.log(np.exp(tc - np.max(tc)).sum())
                    + pt @ d
                )
                for k, expected in (
                    ("reference_gradient", g),
                    ("displacement", d),
                    ("digit_path_action", digit64),
                ):
                    err = float(np.max(abs(data[k][i] - expected)))
                    errors["endpoint_arrays"] = max(errors["endpoint_arrays"], err)
                errors["curvature64"] = max(errors["curvature64"], abs(c - c64))
                errors["gradient64"] = max(errors["gradient64"], abs(y - gradient64))
                compare(
                    saved,
                    {
                        **address,
                        "native": y,
                        "tangent": l,
                        "curvature": c,
                        "gradient_integral": gradient64,
                        "hessian_integral": c64,
                        "reference_score": fr,
                        "target_score": ft,
                        "kl": kl,
                        "bregman": bregman,
                        "fisher_energy": fisher,
                        "centered_displacement_norm": float(np.linalg.norm(d)),
                        "reference_candidate_mass": ref["native"]["candidate_mass"],
                        "target_candidate_mass": unit["native"]["candidate_mass"],
                    },
                    "actions",
                )
                if address["exact_input_noop"] and (np.any(d) or y != 0):
                    raise ValueError("changed exact native no-op")
                if (
                    unit["address"]["query"] != address["query"]
                    or unit["live_body_error"] != 0
                ):
                    raise ValueError("changed native source address")
                values[address["state_index"], address["query"]] = {
                    **address,
                    "native": y,
                    "tangent": l,
                    "curvature": c,
                    "reference_candidate_mass": ref["native"]["candidate_mass"],
                    "target_candidate_mass": unit["native"]["candidate_mass"],
                }
        original = ROOT / spec["exchange_packet"]
        compiled = {
            (int(r["row"]), r["model"]): r for r in rows(original / "02/components.csv")
        }
        previous_joint = {
            (
                r["role"],
                r["design"],
                r["query_a"],
                r["query_b"],
                r["model"],
                r["channel"],
            ): r
            for r in rows(original / "02/native_remainder_covariance.csv")
        }
        stats = {
            (r["role"], r["design"], r["query"]): r
            for r in rows(packet / "02/action_statistics.csv")
        }
        joint = {
            (r["role"], r["design"], r["query_a"], r["query_b"]): r
            for r in rows(packet / "02/readout_covariance.csv")
        }
        gaps = {
            (
                r["role"],
                r["design"],
                r["query_a"],
                r["query_b"],
                r["model"],
                r["channel"],
            ): r
            for r in rows(packet / "02/compiled_gap_decomposition.csv")
        }
        first = [r for r in meta if r["query"] == spec["queries"][0]]
        for role, design in itertools.product(
            spec["roles"], ("all_states", *sorted({r["arm"] for r in first}))
        ):
            selected = [
                r
                for r in first
                if r["role"] == role and (design == "all_states" or r["arm"] == design)
            ]
            ids = [r["state_index"] for r in selected]
            w = weights([r["family"] for r in selected])
            info = {
                "role": role,
                "design": design,
                "states": len(ids),
                "families": len({r["family"] for r in selected}),
            }
            for q in spec["queries"]:
                v = [values[s, q] for s in ids]
                y, l, c = (
                    np.array([r[k] for r in v])
                    for k in ("native", "tangent", "curvature")
                )
                rms = lambda x, w=w: float(np.sqrt(np.einsum("i,i,i->", w, x, x)))
                compare(
                    stats[role, design, q],
                    {
                        **info,
                        "query": q,
                        "native_rms": rms(y),
                        "tangent_rms": rms(l),
                        "curvature_rms": rms(c),
                        "curvature_native_rms_ratio": rms(c) / rms(y)
                        if rms(y)
                        else None,
                        "native_tangent_inner": float(w @ (y * l)),
                        "tangent_curvature_inner": float(w @ (l * c)),
                        "reference_candidate_mass_min": min(
                            r["reference_candidate_mass"] for r in v
                        ),
                        "target_candidate_mass_min": min(
                            r["target_candidate_mass"] for r in v
                        ),
                        "exact_noops": sum(r["exact_input_noop"] for r in v),
                    },
                    "statistics",
                )
            for qa, qb in itertools.combinations(spec["queries"], 2):
                va, vb = ([values[s, q] for s in ids] for q in (qa, qb))
                na, nb, la, lb, ca, cb = (
                    np.array([r[k] for r in v])
                    for v, k in (
                        (va, "native"),
                        (vb, "native"),
                        (va, "tangent"),
                        (vb, "tangent"),
                        (va, "curvature"),
                        (vb, "curvature"),
                    )
                )
                nc, ll, lc, cl, cc = (
                    covariance(x, y, w)
                    for x, y in ((na, nb), (la, lb), (la, cb), (ca, lb), (ca, cb))
                )
                inc = lc + cl + cc
                compare(
                    joint[role, design, qa, qb],
                    {
                        **info,
                        "query_a": qa,
                        "query_b": qb,
                        "native_covariance": nc,
                        "tangent_covariance": ll,
                        "tangent_a_curvature_b": lc,
                        "curvature_a_tangent_b": cl,
                        "curvature_covariance": cc,
                        "link_covariance_increment": inc,
                        "recombination_error": abs(nc - ll - inc),
                        "native_tangent_sign_agree": bool(np.sign(nc) == np.sign(ll)),
                        "link_native_ratio": inc / nc if nc else None,
                    },
                    "covariance",
                )
                for model, channel in itertools.product(
                    spec["models"], ("forward", "odd")
                ):
                    pa, pb = (
                        np.array([float(compiled[r["row"], model][channel]) for r in v])
                        for v in (va, vb)
                    )
                    pc = covariance(pa, pb, w)
                    key = role, design, qa, qb, model, channel
                    compare(
                        gaps[key],
                        {
                            **info,
                            "query_a": qa,
                            "query_b": qb,
                            "model": model,
                            "channel": channel,
                            "native_covariance": nc,
                            "compiled_covariance": pc,
                            "total_gap": nc - pc,
                            "tangent_gap": ll - pc,
                            "link_increment": inc,
                            "link_gap_ratio": inc / (nc - pc) if nc != pc else None,
                            "gap_recombination_error": abs(nc - ll - inc),
                        },
                        "gap",
                    )
                    errors["historical_joint"] = max(
                        errors["historical_joint"],
                        abs(float(previous_joint[key]["joint_gap"]) - (nc - pc)),
                    )
        if dict(counts) != {
            "actions": 2172,
            "statistics": 72,
            "covariance": 72,
            "gap": 432,
        }:
            raise ValueError(f"scope incomplete: {dict(counts)}")
        direct_categories = (
            "stored_probability",
            "stored_score",
            "endpoint_arrays",
            "curvature64",
            "gradient64",
            "historical_joint",
        )
        if any(
            errors[k] > spec["arithmetic_absolute_tolerance"] for k in direct_categories
        ):
            raise ValueError(f"alternate identities failed: {dict(errors)}")
        write_json(
            out / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM21_NOT_GLOBAL_SCOPE_COMPLETE",
                "primary_manifest_sha256": digest(packet / "00/manifest.json"),
                "verifier_sha256": digest(Path(__file__)),
                "checked_counts": dict(counts),
                "native_files_hashed": len(
                    read_json(packet / "00/native_record_hashes.json")
                ),
                "max_errors": dict(errors),
                "alternate_order": spec["alternate_quadrature_order"],
                "new_source_fits": 0,
                "new_model_calls": 0,
                "same_agent_alternate_arithmetic_not_independent_human_replication": True,
            },
        )
        print(read_json(out / "summary.json"), flush=True)
    except Exception as exc:
        write_json(
            out / "failure_attempt_01.json",
            {
                "status": "FAILED_PRESERVED",
                "error": repr(exc),
                "checked_counts": dict(counts),
                "max_errors": dict(errors),
                "verifier_sha256": digest(Path(__file__)),
            },
        )
        raise


def read_json_value(value):
    """Parse CSV structured metadata without changing its original meaning."""
    import json

    return json.loads(value)


def main():
    """Run one finite audit without concurrent writers or overwrite recovery."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, default=DEFAULT)
    p.add_argument("--out", type=Path, default=OUTPUT)
    args = p.parse_args()
    with collector_guard(args.out):
        verify(args.packet, args.out)


if __name__ == "__main__":
    main()
