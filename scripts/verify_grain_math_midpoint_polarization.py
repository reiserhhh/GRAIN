#!/usr/bin/env python3
"""HM19 alternate explicit features, SVD projection, and ambient covector replay."""

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
from scripts.run_grain_math_context_operator_decomposition import inputs
from scripts.run_grain_math_midpoint_polarization import DEFAULT, check
from scripts.run_grain_math_response import write_json
from scripts.run_grain_upstream_routes import collector_guard

OUTPUT = ROOT / "results/grain_math_response_v1/midpoint_polarization_verification_r1"
CHANNELS = ("u", "v", "w", "v_supported", "v_new", "prediction", "native", "remainder")


def rows(path):
    """Read one addressed table, excluding volume metadata sidecars."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def measure(families):
    """Reconstruct source/evaluation family probabilities independently."""
    counts = Counter(families)
    return np.array([1 / len(counts) / counts[f] for f in families])


def eigen_info(gram, multiplier):
    """Use the same declared machine rule, but independent eigvalsh arithmetic."""
    eig = np.linalg.eigvalsh((gram + gram.T) / 2)
    tol = multiplier * np.finfo(float).eps * max(1.0, eig[-1]) * len(gram)
    return eig, tol, int(sum(eig > tol))


def cos(value, aa, bb):
    """Keep undefined zero-action similarities missing."""
    return float(value / (aa * bb) ** 0.5) if aa > 0 and bb > 0 else None


def moments(values, weights):
    """Explicit weighted contractions, independent of the primary moment helper."""
    means = np.einsum("i,ij->j", weights, values)
    gram = np.einsum("i,ij,ik->jk", weights, values, values)
    centered = values - means
    cov = np.einsum("i,ij,ik->jk", weights, centered, centered)
    return means, gram, cov


def verify(packet, out):
    """Verify all source geometry, original addresses, actions, and diagnostics."""
    if out.exists():
        raise FileExistsError("preserve earlier alternate verification")
    manifest = check(packet, source=True)
    spec = manifest["spec"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, expected in ledger.items():
        if digest(packet / name) != expected:
            raise ValueError(f"changed HM19 artifact: {name}")
    parent = ROOT / spec["operator_packet"]
    _, old, parameters, support, views, alpha, _ = inputs(parent)
    x, c, midpoint, shuffled, _ = views
    kappa = float(parameters["delta_scale"] / parameters["context_scale"])
    with np.load(parent / "01/source_support.npz") as a:
        q = a["basis"].copy()
    xq, cq = x @ q, c @ q
    metadata = read_json(packet / "00/metadata.json")
    address = {r["row"]: r for r in metadata}
    first = sorted(
        (r for r in metadata if r["query"] == spec["queries"][0]),
        key=lambda r: r["state_index"],
    )
    by_state = {r["state_index"]: r for r in first}
    errors, checks = defaultdict(float), Counter()

    def compare(actual, expected, category):
        for name, value in expected.items():
            saved = actual[name]
            if value is None:
                if saved != "":
                    raise ValueError(f"{category}/{name}: undefined value replaced")
            elif isinstance(value, (bool, np.bool_)):
                if saved != str(bool(value)):
                    raise ValueError(f"{category}/{name}: Boolean changed")
            elif isinstance(value, (int, float, np.number)):
                error = abs(float(saved) - float(value))
                errors[category] = max(errors[category], error)
                if not np.isclose(float(saved), value, atol=2e-9, rtol=2e-7):
                    raise ValueError(f"{category}/{name}: {saved} != {value}")
                if name in ("observable_rank", "negative_beyond_tolerance") and error:
                    raise ValueError(
                        f"{category}/{name}: numerical classification changed"
                    )
            elif str(saved) != str(value):
                raise ValueError(f"{category}/{name}: address changed")
        checks[category] += 1

    spectral = {
        (r["design"], r["measure"], r["part"]): r
        for r in rows(packet / "01/source_spectra.csv")
    }
    projection_checks = []
    for design in spec["source_designs"]:
        selected = np.array(
            [
                s
                for s in support
                if design == "all_states" or by_state[s]["arm"] == design
            ]
        )
        dyad = np.einsum("ij,ik->ijk", xq[selected], cq[selected])
        u = ((dyad + dyad.transpose(0, 2, 1)) / 2).reshape(len(selected), -1)
        v = (kappa * np.einsum("ij,ik->ijk", xq[selected], xq[selected]) / 2).reshape(
            len(selected), -1
        )
        direct = {"U": u @ u.T, "V": v @ v.T, "UV": u @ v.T, "W": (u + v) @ (u + v).T}
        with np.load(packet / "01" / (design + "_geometry.npz")) as a:
            if not np.array_equal(selected, a["state_indices"]):
                raise ValueError("source projection addresses changed")
            for name, value in direct.items():
                error = float(np.max(abs(value - a[name])))
                errors["explicit_gram"] = max(errors["explicit_gram"], error)
                if error > 1e-9:
                    raise ValueError("explicit Gram differs")
            for name in spec["source_measures"]:
                w = (
                    np.full(len(selected), 1 / len(selected))
                    if name == "uniform"
                    else measure([by_state[s]["family"] for s in selected])
                )
                np.testing.assert_allclose(w, a[name + "_weights"], atol=1e-15)
                d = np.sqrt(w)
                # SVD of explicit observations avoids Schur subtraction and a Gram inverse.
                left, singular, vt = np.linalg.svd(d[:, None] * u, full_matrices=False)
                tol = (
                    spec["machine_tolerance_multiplier"]
                    * np.finfo(float).eps
                    * max(1.0, singular[0] ** 2)
                    * len(selected)
                )
                keep = singular**2 > tol
                orthogonal = vt[keep].T
                projected = (v @ orthogonal) @ orthogonal.T
                remainder = v - projected
                residual = remainder @ remainder.T
                schur_error = float(np.max(abs(residual - a[name + "_residual"])))
                errors["explicit_schur"] = max(errors["explicit_schur"], schur_error)
                if schur_error > 1e-8:
                    raise ValueError(
                        "explicit residual differs from source Schur complement"
                    )
                # Coefficient verification is reported, not mistaken for a well-conditioned action.
                inverse = (left[:, keep] / singular[keep] ** 2) @ left[:, keep].T
                coefficient = d[:, None] * inverse * d[None, :]
                coefficient_error = float(
                    np.max(abs(coefficient - a[name + "_coefficient"]))
                )
                errors["projection_coefficient"] = max(
                    errors["projection_coefficient"], coefficient_error
                )
                range_error = float(
                    np.max(
                        abs(
                            direct["U"] @ a[name + "_coefficient"] @ direct["UV"]
                            - direct["UV"]
                        )
                    )
                )
                orthogonality_error = float(np.max(abs(remainder @ u.T)))
                trace = float(w @ np.diag(residual))
                projection_checks.append(
                    {
                        "design": design,
                        "measure": name,
                        "u_rank": int(sum(keep)),
                        "explicit_schur_max_error": schur_error,
                        "coefficient_max_error": coefficient_error,
                        "explicit_orthogonality_max": orthogonality_error,
                        "residual_trace": trace,
                        "range_defect_max": range_error,
                    }
                )
                grams = {
                    **{k: direct[k] for k in ("U", "V", "W")},
                    "stack_U_V": np.vstack((u, v)) @ np.vstack((u, v)).T,
                    "V_perp": residual,
                }
                for part, gram in grams.items():
                    weights = np.tile(w, 2) if part == "stack_U_V" else w
                    ds = np.sqrt(weights)
                    eigen, threshold, rank = eigen_info(
                        ds[:, None] * gram * ds[None, :],
                        spec["machine_tolerance_multiplier"],
                    )
                    compare(
                        spectral[design, name, part],
                        {
                            "design": design,
                            "measure": name,
                            "part": part,
                            "observations": len(selected),
                            "gram_rows": len(gram),
                            "observable_rank": rank,
                            "tolerance": threshold,
                            "eigen_min": float(eigen[0]),
                            "largest": float(eigen[-1]),
                            "negative_beyond_tolerance": int(sum(eigen < -threshold)),
                            "range_defect_max": range_error,
                            "residual_trace": trace,
                        },
                        "source_spectrum",
                    )
                    errors["source_eigenvalues"] = max(
                        errors["source_eigenvalues"],
                        float(
                            np.max(abs(eigen - a[part + "_" + name + "_eigenvalues"]))
                        ),
                    )
                if spectral[design, name, "stack_U_V"]["observable_rank"] != str(
                    int(sum(keep))
                    + int(spectral[design, name, "V_perp"]["observable_rank"])
                ):
                    raise ValueError(
                        "stack dimension does not match residual increment"
                    )
                checks["source_projection"] += 1

    native = np.empty(len(metadata))
    for role in ("source", "held"):
        with np.load(old / f"00/{role}_targets.npz") as a:
            native[a["indices"]] = a["y"]
    actions = {}
    contexts = {
        "receiver_tensor": c,
        "endpoint_potential": midpoint,
        "shuffled_context": shuffled,
    }
    for query, model in itertools.product(spec["queries"], spec["models"]):
        xs, cs, aa = x[support], contexts[model][support], alpha[query, model]
        direct_covector = ((c @ cs.T) * aa) @ xs
        transpose_covector = ((c @ xs.T) * aa) @ cs
        u = np.einsum("ij,ij->i", x, direct_covector + transpose_covector) / 2
        step_covector = ((x @ xs.T) * aa) @ cs
        v = kappa * np.einsum("ij,ij->i", x, step_covector) / 2
        linear = x @ (aa @ xs)
        ctx = contexts[model]
        forward = np.einsum("ij,ij->i", x, ((ctx @ cs.T) * aa) @ xs)
        reverse = np.einsum("ij,ij->i", x, ((ctx @ xs.T) * aa) @ cs)
        s, h = (forward + reverse) / 2, (forward - reverse) / 2
        if model == "endpoint_potential":
            h = np.zeros(len(x))
        query_meta = {r["state_index"]: r for r in metadata if r["query"] == query}
        n = np.array([native[query_meta[s]["row"]] for s in range(len(x))])
        actions[query, model] = {
            "linear": linear,
            "u": u,
            "v": v,
            "w": u + v,
            "frozen_symmetric": s,
            "frozen_skew": h,
            "prediction": linear + s + h,
            "receiver_potential_same_coefficients": linear + u + v,
            "native": n,
            "remainder": n - linear - s - h,
        }
    components = rows(packet / "02/components.csv")
    for design in spec["source_designs"]:
        with np.load(packet / "01" / (design + "_geometry.npz")) as a:
            selected = a["state_indices"].copy()
            coefficient = a[spec["action_projection_measure"] + "_coefficient"].copy()
        for query, model in itertools.product(spec["queries"], spec["models"]):
            v = actions[query, model]
            cross = kappa * (x @ x[selected].T) * (x @ c[selected].T) / 2
            supported = cross @ coefficient @ v["u"][selected]
            selected_rows = [
                r
                for r in components
                if r["source_projection_design"] == design
                and r["query"] == query
                and r["model"] == model
            ]
            if len(selected_rows) != len(x):
                raise ValueError("component state scope changed")
            for row in selected_rows:
                meta = address[int(row["row"])]
                compare(row, meta, "address")
                s = int(row["state_index"])
                compare(
                    row,
                    {
                        **{p: float(a[s]) for p, a in v.items()},
                        "v_supported": float(supported[s]),
                        "v_new": float(v["v"][s] - supported[s]),
                    },
                    "ambient_action",
                )
    # Once the primitive scalars are replayed, aggregate their frozen decimal values.
    # This avoids assigning stable cosine meanings to 1e-14 numerical-null actions.
    component = {
        (
            r["source_projection_design"],
            r["query"],
            r["model"],
            int(r["state_index"]),
        ): r
        for r in components
    }

    def data(info, query, parts):
        selected = [
            r["state_index"]
            for r in first
            if r["role"] == info["role"]
            and (info["design"] == "all_states" or r["arm"] == info["design"])
        ]
        weights = measure([by_state[s]["family"] for s in selected])
        v = np.array(
            [
                [
                    float(
                        component[
                            info["source_projection_design"], query, info["model"], s
                        ][p]
                    )
                    for p in parts
                ]
                for s in selected
            ]
        )
        return v, weights

    for row in rows(packet / "02/action_statistics.csv"):
        v, w = data(row, row["query"], CHANNELS)
        mean, gram, _ = moments(v, w)
        rms = np.sqrt(np.maximum(0, np.diag(gram)))
        compare(
            row,
            {
                **{p + "_rms": float(rms[j]) for j, p in enumerate(CHANNELS)},
                **{p + "_mean": float(mean[j]) for j, p in enumerate(CHANNELS)},
                "v_new_v_rms_ratio": float(rms[4] / rms[1]) if rms[1] > 0 else None,
                "v_new_prediction_rms_ratio": float(rms[4] / rms[5])
                if rms[5] > 0
                else None,
                "v_new_supported_inner": float(gram[4, 3]),
            },
            "action_statistic",
        )
    for row in rows(packet / "02/target_sharing.csv"):
        va, w = data(row, row["query_a"], [row["part"]])
        vb, _ = data(row, row["query_b"], [row["part"]])
        _, g, cov = moments(np.column_stack((va[:, 0], vb[:, 0])), w)
        compare(
            row,
            {
                "inner_product": float(g[0, 1]),
                "covariance": float(cov[0, 1]),
                "cosine": cos(g[0, 1], g[0, 0], g[1, 1]),
                "correlation": cos(cov[0, 1], cov[0, 0], cov[1, 1]),
            },
            "sharing",
        )
    for row in rows(packet / "02/native_remainder_covariance.csv"):
        va, w = data(row, row["query_a"], ["prediction", "remainder", "native"])
        vb, _ = data(row, row["query_b"], ["prediction", "remainder", "native"])
        _, _, cov = moments(
            np.column_stack(
                (va[:, 0], vb[:, 0], va[:, 1], vb[:, 1], va[:, 2], vb[:, 2])
            ),
            w,
        )
        rebuilt = cov[0, 1] + cov[0, 3] + cov[2, 1] + cov[2, 3]
        compare(
            row,
            {
                "native_covariance": float(cov[4, 5]),
                "compiled_covariance": float(cov[0, 1]),
                "compiled_a_remainder_b": float(cov[0, 3]),
                "remainder_a_compiled_b": float(cov[2, 1]),
                "remainder_covariance": float(cov[2, 3]),
                "joint_gap": float(cov[4, 5] - cov[0, 1]),
                "recombination_error": float(abs(cov[4, 5] - rebuilt)),
            },
            "covariance",
        )
    out.mkdir(parents=True)
    write_json(out / "source_projection_checks.json", projection_checks)
    write_json(out / "frozen_primary_artifact_hashes.json", ledger)
    write_json(
        out / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM19_NOT_GLOBAL_SCOPE_COMPLETE",
            "checked_rows": dict(checks),
            "max_errors": dict(errors),
            "source_projection_method": "explicit_small_coordinate_SVD_no_Schur_subtraction",
            "action_method": "full_ambient_covector_contraction_then_frozen_source_projection",
            "aggregate_method": "independent_einsum_on_verified_frozen_primitive_scalars",
            "roundoff_action_cosines_have_no_scientific_interpretation": True,
            "primary_manifest_sha256": digest(packet / "00/manifest.json"),
            "verifier_sha256": digest(Path(__file__)),
            "new_source_fits": 0,
            "new_model_calls": 0,
            "api_calls": 0,
            "private_text_reads": 0,
            "original_full_scope_complete": False,
        },
    )
    print(read_json(out / "summary.json"), flush=True)


def main():
    """Run one finite alternate replay, never overwriting a prior verifier result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        verify(args.packet, args.out)


if __name__ == "__main__":
    main()
