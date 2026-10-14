#!/usr/bin/env python3
"""HM17 alternate frozen-dyad, potential, source-geometry and metric replay."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_context_operator import DEFAULT, check
from scripts.run_grain_math_native_joint import write_json
from scripts.run_grain_upstream_routes import collector_guard
from scripts.verify_grain_math_ordered_moments import cell_metrics, interval, rows

OUTPUT = ROOT / "results/grain_math_response_v1/context_operator_verification_r1"
KINDS = ("receiver_tensor", "endpoint_potential", "shuffled_context")
COMPARISONS = (
    ("receiver_without_skew", "receiver_tensor"),
    ("receiver_potential", "receiver_tensor"),
    ("receiver_potential", "endpoint_potential"),
    ("shuffled_without_skew", "shuffled_context"),
    ("shuffled_potential", "shuffled_context"),
    ("receiver_tensor", "shuffled_context"),
)


def verify(packet, out):
    """Reconstruct implicit laws using contracted covectors, not the primary helper."""
    if out.exists():
        raise FileExistsError("preserve previous verification")
    manifest = check(packet, geometry=True)
    spec, geometry = manifest["spec"], read_json(packet / "01/geometry_manifest.json")
    old = ROOT / spec["context_packet"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, expected in ledger.items():
        if digest(packet / name) != expected:
            raise ValueError(f"HM17 artifact changed: {name}")
    metadata, functions = (
        read_json(old / "00/metadata.json"),
        read_json(old / "01/functions.json"),
    )
    with np.load(old / "00/endpoints.npz") as arrays:
        r, t, delta, mapping = (
            arrays[k].copy()
            for k in ("reference", "target", "delta", "context_mapping")
        )
    with np.load(old / "01/source_coordinates.npz") as arrays:
        origin, sd, sc, support = (
            arrays[k].copy()
            for k in ("origin", "delta_scale", "context_scale", "source_state_indices")
        )
    x, c, m = delta / sd, (r - origin) / sc, ((r + t) / 2 - origin) / sc
    kappa, contexts = (
        float(sd / sc),
        {"receiver_tensor": c, "endpoint_potential": m, "shuffled_context": c[mapping]},
    )
    a, y = x[support], np.empty(len(metadata))
    for partition in ("source", "held"):
        with np.load(old / f"00/{partition}_targets.npz") as arrays:
            y[arrays["indices"]] = arrays["y"]
    if any(
        metadata[i]["role"] != "source"
        for i in range(len(metadata))
        if metadata[i]["state_index"] in set(support.tolist())
    ):
        raise ValueError("source operator support contains held states")
    with np.load(packet / "01/source_support.npz") as arrays:
        q, eigen, saved_coordinates = (
            arrays[k].copy() for k in ("basis", "eigenvalues", "source_coordinates")
        )
        saved_support = arrays["source_state_indices"].copy()
    u = np.vstack((a, c[support]))
    if not np.array_equal(saved_support, support):
        raise ValueError("support address mismatch")
    gram = u @ u.T
    eig = np.linalg.eigvalsh((gram + gram.T) / 2)
    tol = (
        spec["machine_tolerance_multiplier"]
        * np.finfo(float).eps
        * max(1.0, eig[-1])
        * len(u)
    )
    if (
        q.shape != (x.shape[1], sum(eig > tol))
        or q.shape[1] != geometry["support_rank"]
    ):
        raise ValueError("fixed machine support rank mismatch")
    eigen_error = float(np.max(abs(eig - eigen)))
    orthogonality = float(np.max(abs(q.T @ q - np.eye(q.shape[1]))))
    source_error = float(np.max(abs(u - (u @ q) @ q.T)))
    coordinate_error = float(np.max(abs(saved_coordinates - u @ q)))
    if (
        abs(source_error - geometry["source_reconstruction_max"]) > 1e-10
        or abs(orthogonality - geometry["basis_orthogonality_max"]) > 1e-10
    ):
        raise ValueError("support error reporting differs")
    actions, parts, alphas, operators = {}, {}, {}, {}
    operator_error = spectrum_error = potential_error = 0.0
    xa = x @ a.T
    for query in spec["queries"]:
        with np.load(
            ROOT / functions[query]["full_linear"]["inherited_path"]
        ) as arrays:
            b = sd * arrays["coefficient"]
        with np.load(packet / "01" / f"{query}_full_linear.npz") as arrays:
            operator_error = max(
                operator_error, float(np.max(abs(b - arrays["linear"])))
            )
        linear = np.einsum("ij,j->i", x, b)
        actions[query, "full_linear"] = linear
        parts[query, "full_linear"] = {
            "linear": linear,
            **{k: np.zeros(len(x)) for k in ("symmetric", "skew", "finite_step")},
        }
        for kind, context in contexts.items():
            f = functions[query][kind]
            with np.load(old / f["path"]) as arrays:
                alpha = arrays["weighted_dual"].copy() / f["kernel_unit"]
            cs = context[support]
            alphas[query, kind] = alpha
            # Contract a full covector first, rather than multiplying the primary kernel.
            direct_covectors = (context @ cs.T * alpha) @ a
            transpose_covectors = (context @ a.T * alpha) @ cs
            direct = np.einsum("ij,ij->i", x, direct_covectors)
            transpose = np.einsum("ij,ij->i", x, transpose_covectors)
            linear = np.einsum("ij,j->i", x, alpha @ a)
            symmetric, skew = (direct + transpose) / 2, (direct - transpose) / 2
            step_covectors = (xa * alpha) @ cs
            step = kappa * np.einsum("ij,ij->i", x, step_covectors) / 2
            parts[query, kind] = {
                "linear": linear,
                "symmetric": symmetric,
                "skew": skew,
                "finite_step": step,
            }
            if kind == "endpoint_potential":
                parts[query, kind].update(
                    skew=np.zeros(len(x)), finite_step=np.zeros(len(x))
                )
                expected = actions[query, kind] = linear + symmetric
                begin, end = c, c + kappa * x
            else:
                actions[query, kind] = linear + direct
                stem = "receiver" if kind == "receiver_tensor" else "shuffled"
                actions[query, stem + "_without_skew"] = linear + symmetric
                expected = actions[query, stem + "_potential"] = (
                    linear + symmetric + step
                )
                begin, end = context, context + kappa * x
            pa, pc = a @ q, cs @ q
            compact = np.einsum("ij,i,ik->jk", pa, alpha, pc)
            if kind == "endpoint_potential":
                compact = (compact + compact.T) / 2
            operators[query, kind] = compact
            with np.load(packet / "01" / f"{query}_{kind}.npz") as arrays:
                operator_error = max(
                    operator_error,
                    float(np.max(abs(compact - arrays["operator"]))),
                    float(np.max(abs((alpha @ a) @ q - arrays["linear"]))),
                    float(np.max(abs(alpha - arrays["alpha"]))),
                )
                for name, value in (
                    ("A", compact),
                    ("S", (compact + compact.T) / 2),
                    ("H", (compact - compact.T) / 2),
                ):
                    spectrum_error = max(
                        spectrum_error,
                        float(
                            np.max(
                                abs(
                                    np.linalg.svd(value, compute_uv=False)
                                    - arrays[name]
                                )
                            )
                        ),
                    )
                spectrum_error = max(
                    spectrum_error,
                    float(
                        np.max(
                            abs(
                                np.linalg.eigvalsh((compact + compact.T) / 2)
                                - arrays["S_signed"]
                            )
                        )
                    ),
                )

            def phi(z, cs=cs, alpha=alpha):
                covector = (z @ cs.T * alpha) @ a
                return (
                    z @ (alpha @ a) + np.einsum("ij,ij->i", z, covector) / 2
                ) / kappa

            phi_start, phi_end, phi_mid = phi(begin), phi(end), phi((begin + end) / 2)
            potential_error = max(
                potential_error,
                float(np.max(abs(phi_end - phi_start - expected))),
                float(
                    np.max(abs((phi_mid - phi_start) + (phi_end - phi_mid) - expected))
                ),
            )
    geometry_error = 0.0
    grams = {}
    for kind, context in contexts.items():
        cs, weights = (
            context[support],
            np.asarray([alphas[q0, kind] for q0 in spec["queries"]]),
        )
        gx, gc, xc = a @ a.T, cs @ cs.T, a @ cs.T
        aa = np.einsum("ij,jk,lk->il", weights, gx * gc, weights)
        tr = np.einsum("ij,jk,lk->il", weights, xc * xc.T, weights)
        products = {
            "b": weights @ gx @ weights.T,
            "A": aa,
            "S": (aa + tr) / 2,
            "H": (aa - tr) / 2,
        }
        if kind == "endpoint_potential":
            products.update(A=products["S"], H=np.zeros_like(aa))
        grams[kind] = products
        with np.load(packet / "01" / f"{kind}_cross_grams.npz") as arrays:
            for part, values in products.items():
                geometry_error = max(
                    geometry_error, float(np.max(abs(values - arrays[part])))
                )
    for saved in rows(packet / "01/source_geometry.csv"):
        query, kind, part = saved["query"], saved["model"], saved["part"]
        i, op = spec["queries"].index(query), operators[query, kind]
        value = grams[kind][part][i, i]
        geometry_error = max(
            geometry_error, abs(float(saved["source_gram_squared"]) - value)
        )
        if part != "b":
            op = (
                op
                if part == "A"
                else (op + op.T) / 2
                if part == "S"
                else (op - op.T) / 2
            )
            geometry_error = max(
                geometry_error, abs(np.sum(op**2) - float(saved["frobenius_squared"]))
            )
            singular = np.linalg.svd(op, compute_uv=False)
            threshold = (
                spec["machine_tolerance_multiplier"]
                * np.finfo(float).eps
                * max(1.0, max(singular))
                * len(op)
            )
            if int(saved["rank"]) != sum(singular > threshold):
                raise ValueError("operator machine rank mismatch")
    for saved in rows(packet / "01/cross_target_geometry.csv"):
        kind, qa, qb, part = (saved[k] for k in ("model", "query_a", "query_b", "part"))
        if "_vs_shuffled_context" in kind:
            left = operators[qa, kind.removesuffix("_vs_shuffled_context")]
            right = operators[qb, "shuffled_context"]
            if part != "A":
                sign = 1 if part == "S" else -1
                left, right = (left + sign * left.T) / 2, (right + sign * right.T) / 2
            value, na, nb = np.sum(left * right), np.sum(left**2), np.sum(right**2)
        else:
            matrix = grams[kind][part]
            i, j = spec["queries"].index(qa), spec["queries"].index(qb)
            value, na, nb = matrix[i, j], matrix[i, i], matrix[j, j]
        geometry_error = max(geometry_error, abs(float(saved["inner_product"]) - value))
        expected = value / np.sqrt(na * nb) if na > 0 and nb > 0 else None
        if expected is None:
            if saved["cosine"]:
                raise ValueError("zero operator cosine not missing")
        else:
            geometry_error = max(geometry_error, abs(float(saved["cosine"]) - expected))
    prediction_error = component_error = 0.0
    for saved in rows(packet / "02/components.csv"):
        i, model = int(saved["row"]), saved["model"]
        row = metadata[i]
        for key in ("linear", "symmetric", "skew", "finite_step"):
            component_error = max(
                component_error,
                abs(
                    float(saved[key])
                    - parts[row["query"], model][key][row["state_index"]]
                ),
            )
    checked, lookup = [], {}
    for saved in rows(packet / "02/predictions.csv"):
        i, model = int(saved["row"]), saved["model"]
        row, prediction = (
            metadata[i],
            actions[metadata[i]["query"], model][metadata[i]["state_index"]],
        )
        for name in (
            "query",
            "role",
            "family",
            "arm",
            "state_index",
            "tokens",
            "axis",
            "order",
            "opportunity",
        ):
            if saved[name] != str(row[name]):
                raise ValueError("prediction identity mismatch")
        prediction_error = max(
            prediction_error,
            abs(float(saved["predicted_action"]) - prediction),
            abs(float(saved["observed_action"]) - y[i]),
            abs(float(saved["error"]) - (prediction - y[i])),
        )
        value = {
            **row,
            "model": model,
            "observed_action": float(y[i]),
            "predicted_action": float(prediction),
            "error": float(prediction - y[i]),
            "observation_delta_max": float(max(abs(delta[row["state_index"]]))),
        }
        checked.append(value)
        lookup[i, model] = value
    if len(checked) != len(metadata) * len(spec["models"]) or len(lookup) != len(
        checked
    ):
        raise ValueError("full fixed prediction scope mismatch")
    metric_error, metric_count = 0.0, 0
    for name, keys in (
        ("action_metrics", ("role", "query", "model", "arm")),
        (
            "order_token_metrics",
            ("role", "axis", "query", "model", "arm", "order", "opportunity", "tokens"),
        ),
        ("family_metrics", ("role", "family", "query", "model", "arm")),
    ):
        grouped = defaultdict(list)
        for row in checked:
            grouped[tuple(str(row[k]) for k in keys)].append(row)
        table = rows(packet / f"02/{name}.csv")
        if len(table) != len(grouped):
            raise ValueError("metric coverage mismatch")
        for saved in table:
            computed = cell_metrics(grouped[tuple(saved[k] for k in keys)], spec)
            for k, v in computed.items():
                if v is None:
                    if saved[k] != "":
                        raise ValueError("undefined correlation not missing")
                else:
                    metric_error = max(metric_error, abs(v - float(saved[k])))
            metric_count += 1
    gains = defaultdict(lambda: defaultdict(list))
    for model, comparison in COMPARISONS:
        for row in metadata:
            i = row["row"]
            value = lookup[i, comparison]["error"] ** 2 - lookup[i, model]["error"] ** 2
            for axis in ("all", row["axis"]):
                gains[row["role"], axis, row["query"], row["arm"], model, comparison][
                    row["family"]
                ].append(value)
    gain_table = rows(packet / "02/paired_mechanism_gains.csv")
    if len(gain_table) != len(gains):
        raise ValueError("paired comparison coverage mismatch")
    for saved in gain_table:
        key = tuple(
            saved[k] for k in ("role", "axis", "query", "arm", "model", "comparison")
        )
        for k, value in zip(
            ("mse_gain", "ci_low", "ci_high"),
            interval([np.mean(v) for v in gains[key].values()], spec),
            strict=True,
        ):
            metric_error = max(metric_error, abs(float(saved[k]) - value))
    sharing_error = 0.0
    sharing = rows(packet / "02/shared_action_correlations.csv")
    for saved in sharing:
        qa, qb, kind, part = (saved[k] for k in ("query_a", "query_b", "model", "part"))
        group = [
            r0
            for r0 in metadata
            if r0["query"] == qa
            and r0["role"] == saved["role"]
            and r0["arm"] == saved["arm"]
        ]
        indices = sorted(r0["state_index"] for r0 in group)
        if part == "observed_action":

            def values(q0, indices=indices):
                by_state = {
                    r0["state_index"]: y[r0["row"]]
                    for r0 in metadata
                    if r0["query"] == q0
                }
                return np.asarray([by_state[j] for j in indices])
        elif part == "predicted_action":

            def values(q0, kind=kind, indices=indices):
                return actions[q0, kind][indices]
        else:

            def values(q0, kind=kind, part=part, indices=indices):
                return parts[q0, kind][part][indices]

        left, right = values(qa), values(qb)
        left, right = left - left.mean(), right - right.mean()
        expected = (
            float(left @ right / np.sqrt((left @ left) * (right @ right)))
            if np.std(left) > 0 and np.std(right) > 0
            else None
        )
        if expected is None:
            if saved["correlation"]:
                raise ValueError("undefined action sharing not missing")
        else:
            sharing_error = max(
                sharing_error, abs(float(saved["correlation"]) - expected)
            )
    noops = [r0 for r0 in checked if r0["exact_input_noop"]]
    if any(r0["observed_action"] != 0 or r0["predicted_action"] != 0 for r0 in noops):
        raise ValueError("nonzero identical-state action")
    errors = {
        "prediction": prediction_error,
        "component": component_error,
        "operator": operator_error,
        "spectrum": spectrum_error,
        "source_gram": geometry_error,
        "potential": potential_error,
        "metric": metric_error,
        "sharing": sharing_error,
        "support_eigen": eigen_error,
        "source_coordinates": coordinate_error,
    }
    if max(errors.values()) > 1e-9:
        raise ValueError(f"alternate arithmetic mismatch: {errors}")
    out.mkdir(parents=True)
    write_json(
        out / "frozen_primary_artifact_hashes.json",
        {
            str(p.relative_to(packet)): digest(p)
            for p in sorted(packet.rglob("*"))
            if p.is_file()
        },
    )
    summary = {
        "status": "VERIFIED_FINITE_HM17_NOT_GLOBAL_SCOPE_COMPLETE",
        "prediction_rows": len(checked),
        "component_rows": len(rows(packet / "02/components.csv")),
        "metric_rows": metric_count,
        "paired_ci_rows": len(gain_table),
        "shared_action_rows": len(sharing),
        "source_support_rank": q.shape[1],
        "source_reconstruction_max": source_error,
        "basis_orthogonality_max": orthogonality,
        "max_errors": errors,
        "exact_noop_predictions": len(noops),
        "new_model_calls": 0,
        "new_source_fits": 0,
        "native_cycles_observed": False,
        "same_agent_alternate_arithmetic_not_independent_researcher": True,
        "original_18_protocols_100_cards_44_questions_complete": False,
        "primary_manifest_sha256": digest(packet / "00/manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
    }
    write_json(out / "summary.json", summary)
    print(summary, flush=True)


def main():
    """Run an output-preserving CPU-only verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        verify(args.packet, args.out)


if __name__ == "__main__":
    main()
