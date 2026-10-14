#!/usr/bin/env python3
"""HM20 alternate explicit dyads, ambient covectors, address and native-score audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_context_operator_decomposition import inputs
from scripts.run_grain_math_endpoint_exchange_orbit import DEFAULT, check
from scripts.run_grain_math_response import write_json
from scripts.run_grain_upstream_routes import collector_guard

OUTPUT = ROOT / "results/grain_math_response_v1/endpoint_exchange_orbit_verification_r1"


def rows(path):
    """Read original addressed CSV records without sidecar files."""
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def weight(families):
    """Derive probability weights independently from the addressed family counts."""
    count = Counter(families)
    return np.array([1 / len(count) / count[f] for f in families])


def stable_hash(value):
    """Rebuild context identities independently from the saved source design."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def ambient(x, c, xs, cs, alpha, k):
    """Contract ambient coefficient covectors, not the primary dyad Gram evaluator."""
    dc = ((c @ cs.T) * alpha) @ xs
    tc = ((c @ xs.T) * alpha) @ cs
    vc = ((x @ xs.T) * alpha) @ cs
    direct, transpose = np.sum(x * dc, axis=1), np.sum(x * tc, axis=1)
    return {
        "linear": x @ (xs.T @ alpha),
        "symmetric": (direct + transpose) / 2,
        "skew": (direct - transpose) / 2,
        "v": k / 2 * np.sum(x * vc, axis=1),
    }


def verify(packet, out):
    """Verify every source spectrum, action, native address and reported aggregate."""
    if out.exists():
        raise FileExistsError("preserve earlier verification")
    manifest = check(packet, source=True)
    spec = manifest["spec"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(packet / name) != sha:
            raise ValueError(f"primary artifact changed: {name}")
    operator = ROOT / spec["operator_packet"]
    _, _, parameters, support, views, alpha, _ = inputs(operator)
    x, c, midpoint, shuffled, _ = views
    k = float(parameters["delta_scale"] / parameters["context_scale"])
    metadata = read_json(packet / "00/metadata.json")
    first = sorted(
        (r for r in metadata if r["query"] == spec["queries"][0]),
        key=lambda r: r["state_index"],
    )
    by_state = {r["state_index"]: r for r in first}
    errors, counts = defaultdict(float), Counter()

    def compare(saved, expected, category):
        for name, v in expected.items():
            actual = saved[name]
            if v is None:
                if actual != "":
                    raise ValueError(f"{category}/{name}: undefined value filled")
            elif isinstance(v, (bool, np.bool_)):
                if actual != str(bool(v)):
                    raise ValueError(f"{category}/{name}: boolean changed")
            elif isinstance(v, (int, float, np.number)):
                err = abs(float(actual) - float(v))
                errors[category] = max(errors[category], err)
                if not np.isclose(float(actual), v, atol=2e-9, rtol=2e-7):
                    raise ValueError(f"{category}/{name}: {actual} != {v}")
                if (
                    name in ("observable_rank", "negative_beyond_tolerance")
                    and err != 0
                ):
                    raise ValueError("numerical rank classification changed")
            elif str(v) != str(actual):
                raise ValueError(f"{category}/{name}: address changed")
        counts[category] += 1

    with np.load(operator / "01/source_support.npz") as a:
        q = a["basis"].copy()
    xq, cq = x @ q, c @ q
    spectral = {
        (r["design"], r["measure"], r["part"]): r
        for r in rows(packet / "01/source_spectra.csv")
    }
    for design in spec["source_designs"]:
        ids = np.array(
            [
                s
                for s in support
                if design == "all_states" or by_state[s]["arm"] == design
            ]
        )
        dyad = np.einsum("ij,ik->ijk", xq[ids], cq[ids])
        u = ((dyad + dyad.transpose(0, 2, 1)) / 2).reshape(len(ids), -1)
        v = (k / 2 * np.einsum("ij,ik->ijk", xq[ids], xq[ids])).reshape(len(ids), -1)
        reversed_dyad = np.einsum("ij,ik->ijk", -xq[ids], cq[ids] + k * xq[ids])
        ur = ((reversed_dyad + reversed_dyad.transpose(0, 2, 1)) / 2).reshape(
            len(ids), -1
        )
        stack, orbit = np.vstack((u, v)), np.vstack((u, ur))
        direct = {
            "U": u @ u.T,
            "V": v @ v.T,
            "UV": u @ v.T,
            "W": (u + v) @ (u + v).T,
            "U_reverse": ur @ ur.T,
            "U_U_reverse": u @ ur.T,
            "stack_U_V": stack @ stack.T,
            "orbit_U_U_reverse": orbit @ orbit.T,
        }
        with np.load(packet / "01" / (design + "_geometry.npz")) as a:
            np.testing.assert_array_equal(ids, a["state_indices"])
            for name, g in direct.items():
                err = float(np.max(abs(g - a[name])))
                errors["explicit_gram"] = max(errors["explicit_gram"], err)
                if err > 1e-9:
                    raise ValueError("explicit orbit Gram differs")
            for measure in spec["source_measures"]:
                w = (
                    np.full(len(ids), 1 / len(ids))
                    if measure == "uniform"
                    else weight([by_state[s]["family"] for s in ids])
                )
                np.testing.assert_allclose(w, a[measure + "_weights"], atol=1e-15)
                for part in ("U", "V", "W", "stack_U_V", "orbit_U_U_reverse"):
                    gram = direct[part]
                    ww = np.tile(w / 2, 2) if len(gram) == 2 * len(ids) else w
                    ds = np.sqrt(ww)
                    eigen = np.linalg.eigvalsh(ds[:, None] * gram * ds[None, :])
                    tol = (
                        spec["machine_tolerance_multiplier"]
                        * np.finfo(float).eps
                        * max(1.0, eigen[-1])
                        * len(gram)
                    )
                    compare(
                        spectral[design, measure, part],
                        {
                            "design": design,
                            "measure": measure,
                            "part": part,
                            "original_states": len(ids),
                            "virtual_gram_rows": len(gram),
                            "observable_rank": int(sum(eigen > tol)),
                            "tolerance": tol,
                            "eigen_min": float(eigen[0]),
                            "largest": float(eigen[-1]),
                            "negative_beyond_tolerance": int(sum(eigen < -tol)),
                        },
                        "source_spectrum",
                    )
                    err = float(
                        np.max(abs(eigen - a[part + "_" + measure + "_eigenvalues"]))
                    )
                    errors["source_eigenvalues"] = max(
                        errors["source_eigenvalues"], err
                    )
                    if err > 1e-9:
                        raise ValueError("source eigenvalues changed")

    captures = read_json(ROOT / spec["span_packet"] / "00/design.json")
    lookup = {(r["doc_id"], r["query"]): r for r in captures}
    expected_inventory, native, edges = {}, {}, defaultdict(list)
    for a in metadata:
        addr = lookup[a["receiver_doc"], a["query"]]
        key = stable_hash(
            {
                "input_ids": addr["input_ids"],
                "layer": spec["layer"],
                "positions": a["positions"],
                "span": a["span"],
            }
        )
        unit, ref = (read_json(ROOT / a[f]) for f in ("unit_path", "reference_path"))
        native[a["row"]] = (
            unit["native"]["conditional_score"] - ref["native"]["conditional_score"]
        )
        expected_inventory[a["row"]] = {
            **a,
            "context_sha256": key,
            "reference_sha256": a["reference_state_sha256"],
            "target_sha256": a["target_state_sha256"],
            "native_forward": native[a["row"]],
            "native_reverse_inferred_only": -native[a["row"]],
            "reference_score": ref["native"]["conditional_score"],
            "target_score": unit["native"]["conditional_score"],
            "complete_input_ids_sha256": stable_hash(addr["input_ids"]),
            "reference_record_sha256": digest(ROOT / a["reference_path"]),
            "target_record_sha256": digest(ROOT / a["unit_path"]),
        }
        edges[key, a["reference_state_sha256"], a["target_state_sha256"]].append(
            a["row"]
        )
    swap = {
        (r["receiver_doc"], r["donor_doc"], r["query"], r["span"], r["arm"]): r["row"]
        for r in metadata
    }
    for saved in rows(packet / "02/native_inverse_inventory.csv"):
        a = expected_inventory[int(saved["row"])]
        ids = [
            i
            for i in edges[
                a["context_sha256"], a["target_sha256"], a["reference_sha256"]
            ]
            if i != a["row"]
        ]
        noop = a["reference_sha256"] == a["target_sha256"]
        other = swap.get(
            (a["donor_doc"], a["receiver_doc"], a["query"], a["span"], a["arm"])
        )
        a.update(
            independent_inverse_rows=";".join(map(str, ids)),
            independent_inverse_count=len(ids),
            nontrivial_observed_inverse=bool(ids) and not noop,
            observed_inverse_sum_max=max(
                (abs(native[i] + native[a["row"]]) for i in ids), default=None
            ),
            swapped_document_candidate=other,
            swapped_document_is_same_context=other is not None
            and a["context_sha256"] == expected_inventory[other]["context_sha256"],
        )
        compare(saved, a, "native_inventory")
    for name, sha in read_json(packet / "02/consumed_native_hashes.json").items():
        if digest(ROOT / name) != sha:
            raise ValueError("native observation changed")
        counts["native_file_hash"] += 1

    values = {}
    for query, model in itertools.product(spec["queries"], spec["models"]):
        ctx = {
            "receiver_tensor": c,
            "endpoint_potential": midpoint,
            "shuffled_context": shuffled,
        }[model]
        args = x[support], ctx[support], alpha[query, model], k
        common, common_r = ambient(x, c, *args), ambient(-x, c + k * x, *args)
        actual_ctx = midpoint if model == "endpoint_potential" else ctx
        rev_ctx = midpoint if model == "endpoint_potential" else ctx + k * x
        f, r = ambient(x, actual_ctx, *args), ambient(-x, rev_ctx, *args)
        scalar = lambda d, model=model: (
            d["linear"]
            + d["symmetric"]
            + (0 if model == "endpoint_potential" else d["skew"])
        )
        forward, reverse = scalar(f), scalar(r)
        cf, cr = scalar(common), scalar(common_r)
        addr = {a["state_index"]: a for a in metadata if a["query"] == query}
        values[query, model] = {
            "forward": forward,
            "reverse_virtual": reverse,
            "even": (forward + reverse) / 2,
            "odd": (forward - reverse) / 2,
            "v": common["v"],
            "skew_common": common["skew"]
            if model != "endpoint_potential"
            else np.zeros(len(x)),
            "common_forward": cf,
            "common_reverse_virtual": cr,
            "common_even": (cf + cr) / 2,
            "common_odd": (cf - cr) / 2,
            "native": np.array([native[addr[s]["row"]] for s in range(len(x))]),
        }
    component_map = {}
    for saved in rows(packet / "02/components.csv"):
        a = metadata[int(saved["row"])]
        compare(saved, a, "address")
        key = saved["query"], saved["model"]
        s = int(saved["state_index"])
        compare(
            saved, {name: v[s] for name, v in values[key].items()}, "ambient_action"
        )
        if saved["reverse_status"] != (
            "virtual_fixed_context"
            if saved["model"] == "shuffled_context"
            else "formula_not_native_measurement"
        ):
            raise ValueError("virtual/native distinction lost")
        # Aggregate only these independently verified primitive values: tiny ratios amplify roundoff.
        component_map[key, s] = {n: float(saved[n]) for n in values[key]}

    def selected(role, design):
        members = [
            a
            for a in first
            if a["role"] == role and (design == "all_states" or a["arm"] == design)
        ]
        return [a["state_index"] for a in members], weight(
            [a["family"] for a in members]
        )

    def vectors(query, model, ids, parts):
        return [
            np.array([component_map[(query, model), s][p] for s in ids]) for p in parts
        ]

    for saved in rows(packet / "02/action_statistics.csv"):
        ids, w = selected(saved["role"], saved["design"])
        f, e, o, n = vectors(
            saved["query"], saved["model"], ids, ("forward", "even", "odd", "native")
        )
        rms = lambda a, w=w: float(np.sqrt(np.einsum("i,i,i->", w, a, a)))
        original = float(np.einsum("i,i,i->", w, e, o))
        balanced = float(
            np.einsum("i,i,i->", np.r_[w / 2, w / 2], np.r_[e, e], np.r_[o, -o])
        )
        compare(
            saved,
            {
                "states": len(ids),
                "families": len({by_state[s]["family"] for s in ids}),
                "forward_rms": rms(f),
                "even_rms": rms(e),
                "odd_rms": rms(o),
                "native_rms": rms(n),
                "forward_rmse": rms(f - n),
                "odd_rmse": rms(o - n),
                "odd_minus_forward_rmse": rms(o - n) - rms(f - n),
                "even_forward_rms_ratio": rms(e) / rms(f) if rms(f) > 0 else None,
                "unbalanced_even_odd_inner": original,
                "balanced_even_odd_inner": balanced,
            },
            "action_statistic",
        )
    for saved in rows(packet / "02/native_remainder_covariance.csv"):
        ids, w = selected(saved["role"], saved["design"])
        pa, na = vectors(
            saved["query_a"], saved["model"], ids, (saved["channel"], "native")
        )
        pb, nb = vectors(
            saved["query_b"], saved["model"], ids, (saved["channel"], "native")
        )
        v = np.column_stack((pa, pb, na - pa, nb - pb, na, nb))
        centered = v - np.einsum("i,ij->j", w, v)
        cov = np.einsum("i,ij,ik->jk", w, centered, centered)
        compare(
            saved,
            {
                "native_covariance": cov[4, 5],
                "compiled_covariance": cov[0, 1],
                "compiled_a_remainder_b": cov[0, 3],
                "remainder_a_compiled_b": cov[2, 1],
                "remainder_covariance": cov[2, 3],
                "joint_gap": cov[4, 5] - cov[0, 1],
                "recombination_error": abs(
                    cov[4, 5] - cov[0, 1] - cov[0, 3] - cov[2, 1] - cov[2, 3]
                ),
            },
            "covariance",
        )
    summary = read_json(packet / "02/summary.json")
    exact = {
        "component_rows": counts["ambient_action"],
        "statistics_rows": counts["action_statistic"],
        "covariance_rows": counts["covariance"],
        "native_inventory_rows": counts["native_inventory"],
        "nontrivial_observed_inverse_rows": sum(
            a["nontrivial_observed_inverse"] for a in expected_inventory.values()
        ),
        "noop_rows": sum(a["exact_input_noop"] for a in metadata),
        "swapped_document_candidates": sum(
            a["swapped_document_candidate"] is not None
            for a in expected_inventory.values()
        ),
        "same_context_swapped_document_candidates": sum(
            a["swapped_document_is_same_context"] for a in expected_inventory.values()
        ),
    }
    if any(summary[n] != value for n, value in exact.items()):
        raise ValueError("summary counts changed")
    out.mkdir(parents=True)
    write_json(out / "frozen_primary_artifact_hashes.json", ledger)
    write_json(
        out / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM20_NOT_GLOBAL_SCOPE_COMPLETE",
            "checked_rows": dict(counts),
            "max_errors": dict(errors),
            "summary_counts": exact,
            "source_method": "explicit_compressed_coordinate_dyads_and_independent_eigvalsh",
            "action_method": "full_ambient_covector_contraction",
            "inventory_method": "independent_full_input_context_and_exact_assigned_span_hash_index",
            "aggregate_method": "einsum_on_independently_verified_saved_primitive_actions",
            "primary_manifest_sha256": digest(packet / "00/manifest.json"),
            "verifier_sha256": digest(Path(__file__)),
            "new_source_fits": 0,
            "new_model_calls": 0,
            "original_full_scope_complete": False,
        },
    )
    print(read_json(out / "summary.json"), flush=True)


def main():
    """Preserve any failed verification attempt; never mutate primary results."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, default=DEFAULT)
    p.add_argument("--out", type=Path, default=OUTPUT)
    a = p.parse_args()
    with collector_guard(a.out):
        try:
            verify(a.packet, a.out)
        except Exception as error:
            i = 1
            failure = (
                a.out.parent / f"endpoint_exchange_verification_attempt_{i:02d}.json"
            )
            while failure.exists():
                i += 1
                failure = (
                    a.out.parent
                    / f"endpoint_exchange_verification_attempt_{i:02d}.json"
                )
            write_json(
                failure,
                {
                    "status": "VERIFICATION_FAILED",
                    "error": repr(error),
                    "verifier_sha256": digest(Path(__file__)),
                    "packet": str(a.packet),
                },
            )
            raise


if __name__ == "__main__":
    main()
