#!/usr/bin/env python3
"""HM16 alternate absolute-state, spectral-kernel and endpoint-potential replay."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_context_covector import DEFAULT, check
from scripts.run_grain_math_native_joint import write_json
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_upstream_routes import collector_guard
from scripts.verify_grain_math_ordered_moments import (
    SPANS,
    cell_metrics,
    interval,
    native_score,
    rows,
)
from scripts.verify_grain_math_vector_action_kernel import observe

OUTPUT = ROOT / "results/grain_math_response_v1/context_covector_verification_r1"
FIELDS = (
    "pair_id",
    "family",
    "role",
    "axis",
    "order",
    "polarity",
    "receiver_doc",
    "donor_doc",
    "query",
    "arm",
)


def context_cycle(metadata, seed):
    """Rebuild the frozen pair-cycle without importing the primary shuffle helper."""
    core, arms, addresses = {}, defaultdict(set), {}
    for row in metadata:
        p = row["pair_id"]
        value = tuple(row[k] for k in ("role", "axis", "order", "tokens"))
        if p in core and core[p] != value:
            raise ValueError("pair stratum varies across queries/arms")
        core[p] = value
        arms[p].add(row["arm"])
        key = (p, row["arm"])
        if key in addresses and addresses[key] != row["state_index"]:
            raise ValueError("state identity varies by query")
        addresses[key] = row["state_index"]
    groups = defaultdict(list)
    for p in sorted(core):
        groups[(*core[p], tuple(sorted(arms[p])))].append(p)
    rng, destinations = np.random.default_rng(seed), {}
    for key in sorted(groups):
        order = rng.permutation(groups[key]).tolist()
        destinations.update(zip(order, order[1:] + order[:1], strict=True))
    mapping = np.empty(len(addresses), dtype=int)
    for (p, arm), j in addresses.items():
        mapping[j] = addresses[destinations[p], arm]
    if sorted(mapping.tolist()) != list(range(len(mapping))):
        raise ValueError("context mapping is not a permutation")
    return mapping, destinations, groups


def tensor_parts(x, context, support, model):
    """Use polarization of the receiver Gram to rebuild symmetric tensor products."""
    a, b = x[support], context[support]
    linear, context_gram = x @ a.T, context @ b.T
    coupling = linear * context_gram
    if model == "endpoint_potential":
        cross = context @ a.T
        cross_reverse = x @ b.T
        coupling = (coupling + cross * cross_reverse) / 2
    return linear, coupling


def verify(packet, out):
    """Replay every addressed endpoint, source function, contribution and interval."""
    if out.exists():
        raise FileExistsError("preserve earlier verification; use another output")
    manifest = check(packet, models=True)
    spec = manifest["spec"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, expected in ledger.items():
        if digest(packet / name) != expected:
            raise ValueError(f"primary artifact changed: {name}")
    original = ROOT / spec["vector_packet"]
    metadata = read_json(packet / "00/metadata.json")
    if (packet / "00/metadata.json").read_bytes() != (
        original / "00/metadata.json"
    ).read_bytes():
        raise ValueError("inherited action addresses changed")
    span = ROOT / spec["span_packet"]
    canonical = {
        r["doc_id"]: r
        for r in read_json(span / "00/design.json")
        if r["query"] == spec["queries"][0]
    }
    with np.load(span / "02/basis.npz") as arrays:
        frame = arrays["L16_U"].copy()
    bodies = {}
    for doc, row in canonical.items():
        with np.load(span / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            bodies[doc] = {n: arrays[f"L16_{n}"].copy() for n in SPANS}
    with np.load(packet / "00/endpoints.npz") as arrays:
        r, t, d = (arrays[k].copy() for k in ("reference", "target", "delta"))
        saved_mapping = arrays["context_mapping"].copy()
    y, partitions = np.empty(len(metadata)), {}
    for label in ("source", "held"):
        with np.load(packet / f"00/{label}_targets.npz") as arrays:
            partitions[label] = arrays["indices"].tolist()
            y[arrays["indices"]] = arrays["y"]
        if (packet / f"00/{label}_targets.npz").read_bytes() != (
            original / f"00/{label}_targets.npz"
        ).read_bytes():
            raise ValueError("inherited target file changed")
    source = [i for i, row in enumerate(metadata) if row["role"] == "source"]
    if (
        source != partitions["source"]
        or set(source) & set(partitions["held"])
        or sorted(source + partitions["held"]) != list(range(len(metadata)))
    ):
        raise ValueError("source/held partition mismatch")
    values = np.asarray(read_json(span / "00/manifest.json")["alphabet"]["values"])
    feature_error = action_error = 0.0
    seen, records = {}, {}
    for i, row in enumerate(metadata):
        j, name = row["state_index"], row["span"]
        if row["row"] != i:
            raise ValueError("row identity mismatch")
        if j not in seen:
            h, donor = (bodies[row[k]][name] for k in ("receiver_doc", "donor_doc"))
            ha, da = (
                canonical[row[k]]["spans"][name] for k in ("receiver_doc", "donor_doc")
            )
            if ha["raw"] != da["raw"] or ha["token_ids"] != da["token_ids"]:
                raise ValueError("same-word/token identity mismatch")
            reference, change = h, donor.astype(float) - h.astype(float)
            mean = np.broadcast_to(np.mean(change, axis=0), change.shape)
            arm = row["arm"]
            if arm == "full_property":
                target = donor
            elif arm == "mean_only":
                target = (h.astype(float) + mean).astype(np.float32)
            elif arm == "centered_only":
                target = (h.astype(float) + (change - mean)).astype(np.float32)
            elif arm in ("receiver_reverse", "centered_reverse"):
                if arm == "centered_reverse":
                    reference = (h.astype(float) + (change - mean)).astype(np.float32)
                target = reference[np.arange(len(reference) - 1, -1, -1)]
            elif arm in ("projection", "complement"):
                projection = np.asarray([frame @ (frame.T @ v) for v in change])
                part = projection if arm == "projection" else change - projection
                target = (h.astype(float) + part).astype(np.float32)
            else:
                raise ValueError("undeclared intervention arm")
            if (
                hashlib.sha256(reference.tobytes()).hexdigest()
                != row["reference_state_sha256"]
                or hashlib.sha256(target.tobytes()).hexdigest()
                != row["target_state_sha256"]
                or row["exact_input_noop"] != (reference.tobytes() == target.tobytes())
            ):
                raise ValueError("absolute full-state identity mismatch")
            rb, tb = (
                dict(bodies[row["receiver_doc"]]),
                dict(bodies[row["receiver_doc"]]),
            )
            rb[name], tb[name] = reference, target
            rr, tt = observe(rb), observe(tb)
            feature_error = max(
                feature_error,
                float(np.max(np.abs(rr - r[j]))),
                float(np.max(np.abs(tt - t[j]))),
                float(np.max(np.abs((tt - rr) - d[j]))),
            )
            seen[j] = (row["pair_id"], row["arm"])
        elif seen[j] != (row["pair_id"], row["arm"]):
            raise ValueError("state address reused by a different pair-arm")
        for k in ("unit_path", "reference_path"):
            path = ROOT / row[k]
            if path not in records:
                records[path] = read_json(path)
        unit, ref = (records[ROOT / row[k]] for k in ("unit_path", "reference_path"))
        if (
            any(unit["address"][k] != row[k] for k in FIELDS)
            or unit["live_body_error"] != 0
            or ref["address"]["query"] != row["query"]
            or ref["address"].get("doc_id", ref["address"].get("receiver_doc"))
            != row["receiver_doc"]
        ):
            raise ValueError("native readout identity mismatch")
        action_error = max(
            action_error,
            abs(native_score(unit, values) - native_score(ref, values) - y[i]),
        )
    if len(seen) != 724 or feature_error > 1e-10 or action_error > 1e-12:
        raise ValueError("absolute observation/native action replay failed")
    mapping, destinations, strata = context_cycle(metadata, spec["shuffle_seed"])
    if not np.array_equal(mapping, saved_mapping):
        raise ValueError("frozen context cycle differs")
    pair_table = rows(packet / "00/context_pair_mapping.csv")
    for saved in pair_table:
        p = int(saved["pair_id"])
        if int(saved["context_pair_id"]) != destinations[p] or (
            saved["pair_moved"] == "True"
        ) != (destinations[p] != p):
            raise ValueError("context pair table mismatch")
    with np.load(packet / "01/source_coordinates.npz") as arrays:
        saved_origin, saved_scale, delta_scale = (
            arrays[k].copy() for k in ("origin", "context_scale", "delta_scale")
        )
        support = arrays["source_state_indices"].copy()
        saved_scale2 = float(arrays["context_scale_squared"])
    source_query = [i for i in source if metadata[i]["query"] == spec["queries"][0]]
    if not np.array_equal(support, [metadata[i]["state_index"] for i in source_query]):
        raise ValueError("source coordinates contain non-source states")
    counts = defaultdict(int)
    for i in source_query:
        counts[metadata[i]["family"]] += 1
    w = np.asarray(
        [
            len(source_query) / (len(counts) * counts[metadata[i]["family"]])
            for i in source_query
        ]
    )
    origin = sum(w[j] * r[k] for j, k in enumerate(support)) / sum(w)
    scale2 = sum(
        w[j] * np.dot(r[k] - origin, r[k] - origin) for j, k in enumerate(support)
    ) / sum(w)
    context_scale = np.sqrt(scale2) if scale2 else 1.0
    coordinate_error = max(
        float(np.max(np.abs(origin - saved_origin))),
        abs(context_scale - float(saved_scale)),
        abs(float(scale2) - saved_scale2),
    )
    original_functions = read_json(original / "01/functions.json")
    if any(
        float(delta_scale) != original_functions[q]["ordered_full"]["scale"]
        for q in spec["queries"]
    ):
        raise ValueError("inherited displacement scale changed")
    x = d / delta_scale
    c = (r - origin) / context_scale
    midpoint = ((r + t) / 2 - origin) / context_scale
    contexts = {
        "receiver_tensor": c,
        "endpoint_potential": midpoint,
        "shuffled_context": c[mapping],
    }
    parts = {
        model: tensor_parts(x, ctx, support, model) for model, ctx in contexts.items()
    }
    signs = np.where(np.arange(x.shape[1]) % 2, -1.0, 1.0)
    xx = x[:, ::-1] * signs
    gauge_error = 0.0
    for model, ctx in contexts.items():
        rotated = tensor_parts(xx, ctx[:, ::-1] * signs, support, model)
        gauge_error = max(
            gauge_error, float(np.max(np.abs(sum(rotated) - sum(parts[model]))))
        )
    functions = read_json(packet / "01/functions.json")
    spectral, kernels, fit_rows = {}, {}, []
    unit_error = dual_error = eigen_error = residual_error = 0.0
    for model, (linear, coupling) in parts.items():
        raw = (linear + coupling)[support]
        unit = sum(w[j] * raw[j, j] for j in range(len(w))) / sum(w)
        gram = (raw + raw.T) / (2 * unit)
        eig = np.linalg.eigvalsh(gram)
        tolerance = 100 * np.finfo(float).eps * max(1.0, eig[-1]) * len(w)
        if eig[0] < -tolerance:
            raise ValueError("source tensor Gram not PSD")
        root = np.sqrt(w)
        weighted_gram = root[:, None] * gram * root[None, :]
        eig_w, vectors = np.linalg.eigh(weighted_gram)
        kernels[model] = (unit, eig, gram)
        for query in spec["queries"]:
            indices = [i for i in source if metadata[i]["query"] == query]
            if [metadata[i]["state_index"] for i in indices] != support.tolist():
                raise ValueError("source support differs by query")
            f = functions[query][model]
            dual = root * (
                vectors @ ((vectors.T @ (root * y[indices])) / (eig_w + spec["alpha"]))
            )
            with np.load(packet / f["path"]) as arrays:
                dual_error = max(
                    dual_error, float(np.max(np.abs(dual - arrays["weighted_dual"])))
                )
                eigen_error = max(
                    eigen_error, float(np.max(np.abs(eig - arrays["eigenvalues"])))
                )
                if not np.array_equal(arrays["source_state_indices"], support):
                    raise ValueError("source kernel support changed")
            unit_error = max(unit_error, abs(unit - f["kernel_unit"]))
            residual = float(
                np.max(
                    np.abs(
                        (weighted_gram + np.eye(len(w))) @ (dual / root)
                        - root * y[indices]
                    )
                )
            )
            residual_error = max(residual_error, residual)
            if (
                f["alpha"] != 1
                or f["intercept"] != 0
                or f["source_rows"] != 360
                or f["source_families"] != 8
                or f["source_numerical_rank"] != int(np.sum(eig > tolerance))
            ):
                raise ValueError("source-only function contract mismatch")
            spectral[query, model] = dual / unit
            fit_rows.append(
                {
                    "query": query,
                    "model": model,
                    "source_rows": len(w),
                    "kernel_unit": unit,
                    "source_numerical_rank": int(np.sum(eig > tolerance)),
                    "spectral_residual_max": residual,
                }
            )
    fm = read_json(packet / "01/functions_manifest.json")
    if (
        fm["held_targets_loaded"]
        or fm["source_indices"] != source
        or fm["new_kernel_fits"] != 9
    ):
        raise ValueError("source fit provenance mismatch")
    if (
        max(
            coordinate_error,
            gauge_error,
            unit_error,
            dual_error,
            eigen_error,
            residual_error,
        )
        > 1e-9
    ):
        raise ValueError("source spectral/coordinate/gauge replay failed")
    cached, potential_error, potential_cycle_error = {}, 0.0, 0.0
    for query in spec["queries"]:
        for model in spec["models"]:
            f = functions[query][model]
            if model == "full_linear":
                if f["inherited_path"] != str(
                    (
                        original / original_functions[query]["ordered_full"]["path"]
                    ).relative_to(ROOT)
                ):
                    raise ValueError("baseline source law changed")
                with np.load(ROOT / f["inherited_path"]) as arrays:
                    pl, pc = d @ arrays["coefficient"], np.zeros(len(d))
            else:
                linear, coupling = parts[model]
                alpha = spectral[query, model]
                pl, pc = linear @ alpha, coupling @ alpha
                if model == "endpoint_potential":

                    def phi(z, dual=alpha):
                        za, zb = z @ x[support].T, z @ midpoint[support].T
                        return (
                            (context_scale / delta_scale) * (za + za * zb / 2) @ dual
                        )

                    fr, ft, fm = (
                        phi(c),
                        phi((t - origin) / context_scale),
                        phi(midpoint),
                    )
                    potential_error = max(
                        potential_error, float(np.max(np.abs((ft - fr) - (pl + pc))))
                    )
                    potential_cycle_error = max(
                        potential_cycle_error,
                        float(np.max(np.abs((fm - fr) + (ft - fm) - (ft - fr)))),
                    )
            for i, row in enumerate(metadata):
                if row["query"] != query:
                    continue
                j = row["state_index"]
                p = float(pl[j] + pc[j])
                cached[i, model] = {
                    **row,
                    "model": model,
                    "observed_action": float(y[i]),
                    "predicted_action": p,
                    "error": float(p - y[i]),
                    "linear_contribution": float(pl[j]),
                    "coupling_contribution": float(pc[j]),
                    "observation_delta_max": float(np.max(np.abs(d[j]))),
                }
    checked, prediction_error = [], 0.0
    for saved in rows(packet / "02/predictions.csv"):
        row = cached[int(saved["row"]), saved["model"]]
        if any(
            saved[k] != str(row[k])
            for k in (*FIELDS, "state_index", "exact_input_noop")
        ):
            raise ValueError("prediction identity mismatch")
        for k in (
            "observed_action",
            "predicted_action",
            "error",
            "linear_contribution",
            "coupling_contribution",
            "observation_delta_max",
        ):
            prediction_error = max(prediction_error, abs(row[k] - float(saved[k])))
        checked.append(row)
    if len(checked) != 8688 or len({(r["row"], r["model"]) for r in checked}) != 8688:
        raise ValueError("prediction coverage mismatch")
    metric_error = metric_count = 0
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
                        raise ValueError("degenerate correlation serialized")
                else:
                    metric_error = max(metric_error, abs(v - float(saved[k])))
            metric_count += 1
    gains = defaultdict(lambda: defaultdict(list))
    keys = ("role", "axis", "query", "arm", "model", "comparison")
    for row in checked:
        if row["model"] == "full_linear":
            continue
        for comparison in (
            ("full_linear", "shuffled_context")
            if row["model"] == "receiver_tensor"
            else ("full_linear",)
        ):
            gain = cached[row["row"], comparison]["error"] ** 2 - row["error"] ** 2
            for axis in ("all", row["axis"]):
                key = (
                    row["role"],
                    axis,
                    row["query"],
                    row["arm"],
                    row["model"],
                    comparison,
                )
                gains[key][row["family"]].append(gain)
    table = rows(packet / "02/paired_mechanism_gains.csv")
    if len(table) != len(gains):
        raise ValueError("paired mechanism coverage mismatch")
    for saved in table:
        family = gains[tuple(saved[k] for k in keys)]
        if int(saved["families"]) != len(family):
            raise ValueError("paired family count mismatch")
        for k, v in zip(
            ("mse_gain", "ci_low", "ci_high"),
            interval([np.mean(v) for v in family.values()], spec),
            strict=True,
        ):
            metric_error = max(metric_error, abs(v - float(saved[k])))
    noops = [r for r in checked if r["exact_input_noop"]]
    if any(r["observed_action"] != 0 or r["predicted_action"] != 0 for r in noops):
        raise ValueError("identical input nonzero action")
    if (
        max(prediction_error, metric_error, potential_error, potential_cycle_error)
        > 1e-10
    ):
        raise ValueError("alternate prediction/potential/metric replay failed")
    frozen = {
        str(p.relative_to(packet)): digest(p)
        for p in sorted(packet.rglob("*"))
        if p.is_file()
    }
    write_json(out / "frozen_primary_artifact_hashes.json", frozen)
    write_csv(out / "source_fits_replayed.csv", fit_rows)
    summary = {
        "status": "VERIFIED_FINITE_HM16_NOT_GLOBAL_SCOPE_COMPLETE",
        "new_model_calls": 0,
        "unique_full_states": len(seen),
        "paired_query_interventions": len(metadata),
        "predictions": len(checked),
        "source_rows": len(source),
        "held_rows": len(partitions["held"]),
        "new_source_functions": len(fit_rows),
        "inherited_source_functions": 3,
        "context_pairs_moved": sum(p != q for p, q in destinations.items()),
        "context_pairs_fixed": sum(p == q for p, q in destinations.items()),
        "context_strata": len(strata),
        "context_states_moved": int(np.sum(mapping != np.arange(len(mapping)))),
        "independent_absolute_observation_max_error": feature_error,
        "independent_categorical_action_max_error": action_error,
        "source_coordinate_max_error": coordinate_error,
        "source_kernel_unit_max_error": unit_error,
        "spectral_dual_max_error": dual_error,
        "gram_eigenvalue_max_error": eigen_error,
        "spectral_residual_max": residual_error,
        "signed_coordinate_isometry_gram_max_error": gauge_error,
        "endpoint_scalar_potential_max_error": potential_error,
        "endpoint_three_point_telescoping_max_error": potential_cycle_error,
        "potential_check_is_algebra_not_native_cycle_measurement": True,
        "serialized_prediction_max_error": prediction_error,
        "serialized_metric_max_error": metric_error,
        "metric_rows_recomputed": metric_count,
        "paired_ci_rows_recomputed": len(table),
        "exact_noop_predictions": len(noops),
        "frozen_primary_artifacts": len(frozen),
        "same_agent_alternate_arithmetic_not_independent_researcher": True,
        "original_18_protocols_100_cards_44_questions_complete": False,
        "primary_manifest_sha256": digest(packet / "00/manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
    }
    write_json(out / "summary.json", summary)
    print(summary, flush=True)


def main():
    """Run a bounded locked verifier with no model, API, or training call."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        verify(args.packet, args.out)


if __name__ == "__main__":
    main()
