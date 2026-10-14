#!/usr/bin/env python3
"""HM15 alternate complete-state, spectral-dual and kernel replay; zero forwards."""

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
from scripts.run_grain_math_native_joint import write_json
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_vector_action_kernel import DEFAULT, check
from scripts.run_grain_upstream_routes import collector_guard
from scripts.verify_grain_math_ordered_moments import (
    PROPERTY,
    SPANS,
    cell_metrics,
    interval,
    native_score,
    rows,
)

OUTPUT = ROOT / "results/grain_math_response_v1/full_vector_action_verification_r1"
MODELS = ("zero_action", "ordered_rank2", "ordered_full")


def observe(bodies):
    """Accumulate complete hidden-vector position powers token by token."""
    moments = {}
    for name in SPANS:
        h = np.asarray(bodies[name], float)
        t = len(h)
        p = [0.0] if t == 1 else [-1 + 2 * j / (t - 1) for j in range(t)]
        moments[name] = np.asarray(
            [
                sum(position**k * v for position, v in zip(p, h, strict=True)) / t
                for k in range(4)
            ]
        )
    return np.asarray(
        [moments[name][0] for name in SPANS]
        + [moments[name][k] for k in (1, 2, 3) for name in PROPERTY]
    ).reshape(-1)


def verify(packet, out):
    """Replay all states, native actions, source solves and serialized held metrics."""
    if out.exists():
        raise FileExistsError("preserve verification; use another output path")
    manifest = check(packet, models=True)
    spec = manifest["spec"]
    for name, expected in read_json(packet / "02/artifact_hashes.json").items():
        if digest(packet / name) != expected:
            raise ValueError(f"primary artifact changed: {name}")
    span, ordered = ROOT / spec["span_packet"], ROOT / spec["ordered_packet"]
    canonical = {
        r["doc_id"]: r
        for r in read_json(span / "00/design.json")
        if r["query"] == spec["queries"][0]
    }
    with np.load(span / "02/basis.npz") as arrays:
        frame = arrays["L16_U"].copy()
    bodies = {}
    for doc_id, row in canonical.items():
        with np.load(span / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            bodies[doc_id] = {name: arrays[f"L16_{name}"].copy() for name in SPANS}
    metadata = read_json(packet / "00/metadata.json")
    original = read_json(ordered / "00/metadata.json")
    original.extend(
        r
        for r in read_json(span / "00/patch_design.json")
        if r["layer"] == 16 and r["arm"] in ("projection", "complement")
    )
    if len(original) != len(metadata) or len(metadata) != 2172:
        raise ValueError("all inherited and extra addresses required")
    with np.load(packet / "00/differences.npz") as arrays:
        views = {
            "ordered_full": arrays["full"].copy(),
            "ordered_rank2": arrays["projected"].copy(),
        }
        if not np.array_equal(frame, arrays["frame"]):
            raise ValueError("frame mismatch")
    geometry = read_json(packet / "00/geometry.json")
    y = np.empty(len(metadata))
    partitions = {}
    for label in ("source", "held"):
        with np.load(packet / f"00/{label}_targets.npz") as arrays:
            partitions[label] = arrays["indices"].tolist()
            y[arrays["indices"]] = arrays["y"]
    source = [i for i, r in enumerate(metadata) if r["role"] == "source"]
    if (
        partitions["source"] != source
        or set(source) & set(partitions["held"])
        or sorted(source + partitions["held"]) != list(range(len(metadata)))
    ):
        raise ValueError("source/held partition mismatch")
    values = np.asarray(read_json(span / "00/manifest.json")["alphabet"]["values"])
    state_cache, records = {}, {}
    feature_error = action_error = geometry_error = 0.0
    fields = (
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
    for i, (saved, old) in enumerate(zip(metadata, original, strict=True)):
        if saved["row"] != i or any(saved[k] != old[k] for k in fields):
            raise ValueError("original intervention identity mismatch")
        key, name = (saved["pair_id"], saved["arm"]), saved["span"]
        if key not in state_cache:
            h, d = (bodies[saved[k]][name] for k in ("receiver_doc", "donor_doc"))
            ha, da = (
                canonical[saved[k]]["spans"][name]
                for k in ("receiver_doc", "donor_doc")
            )
            if ha["raw"] != da["raw"] or ha["token_ids"] != da["token_ids"]:
                raise ValueError("same-word/token identity mismatch")
            reference = h
            delta = d.astype(float) - h.astype(float)
            avg = np.broadcast_to(np.mean(delta, axis=0), delta.shape)
            arm = saved["arm"]
            if arm == "full_property":
                target = d
            elif arm == "mean_only":
                target = (h.astype(float) + avg).astype(np.float32)
            elif arm == "centered_only":
                target = (h.astype(float) + (delta - avg)).astype(np.float32)
            elif arm in ("receiver_reverse", "centered_reverse"):
                if arm == "centered_reverse":
                    reference = (h.astype(float) + (delta - avg)).astype(np.float32)
                target = reference[np.arange(len(reference) - 1, -1, -1)]
            elif arm in ("projection", "complement"):
                projected = np.asarray([frame @ (frame.T @ v) for v in delta])
                change = projected if arm == "projection" else delta - projected
                target = (h.astype(float) + change).astype(np.float32)
            else:
                raise ValueError("undeclared intervention")
            rb, tb = (
                dict(bodies[saved["receiver_doc"]]),
                dict(bodies[saved["receiver_doc"]]),
            )
            rb[name], tb[name] = reference, target
            dx = observe(tb) - observe(rb)
            projected = np.asarray([frame.T @ v for v in dx.reshape(12, -1)]).reshape(
                -1
            )
            state_cache[key] = (dx, projected, reference, target)
            j = saved["state_index"]
            g = geometry[j]
            if g["state_index"] != j or (g["pair_id"], g["arm"]) != key:
                raise ValueError("unique state geometry identity mismatch")
            for k, value in {
                "full_observation_norm": np.linalg.norm(dx),
                "projected_observation_norm": np.linalg.norm(projected),
                "projected_token_displacement_max": np.max(
                    np.abs((target.astype(float) - reference.astype(float)) @ frame)
                ),
            }.items():
                geometry_error = max(geometry_error, abs(value - g[k]))
        dx, projected, reference, target = state_cache[key]
        j = saved["state_index"]
        feature_error = max(
            feature_error,
            float(np.max(np.abs(dx - views["ordered_full"][j]))),
            float(np.max(np.abs(projected - views["ordered_rank2"][j]))),
        )
        if (
            saved["reference_state_sha256"]
            != hashlib.sha256(reference.tobytes()).hexdigest()
            or saved["target_state_sha256"]
            != hashlib.sha256(target.tobytes()).hexdigest()
            or saved["exact_input_noop"] != (reference.tobytes() == target.tobytes())
        ):
            raise ValueError("exact full-state identity mismatch")
        for field in ("unit_path", "reference_path"):
            path = ROOT / saved[field]
            if path not in records:
                records[path] = read_json(path)
        unit, ref = (records[ROOT / saved[k]] for k in ("unit_path", "reference_path"))
        if (
            any(unit["address"][k] != saved[k] for k in fields)
            or unit["live_body_error"] != 0
            or ref["address"]["query"] != saved["query"]
            or ref["address"].get("doc_id", ref["address"].get("receiver_doc"))
            != saved["receiver_doc"]
        ):
            raise ValueError("native readout address mismatch")
        action_error = max(
            action_error,
            abs(native_score(unit, values) - native_score(ref, values) - y[i]),
        )
    if (
        len(state_cache) != 724
        or feature_error > 1e-10
        or geometry_error > 1e-10
        or action_error > 1e-12
    ):
        raise ValueError("alternate state/readout reconstruction failed")
    functions = read_json(packet / "01/functions.json")
    fits, coefficients, duals = [], {}, {}
    scale_error = dual_error = coefficient_error = eigen_error = gauge_error = 0.0
    for query in spec["queries"]:
        indices = [i for i in source if metadata[i]["query"] == query]
        state_indices = np.asarray([metadata[i]["state_index"] for i in indices])
        counts = {
            f: sum(metadata[i]["family"] == f for i in indices)
            for f in {metadata[i]["family"] for i in indices}
        }
        w = np.asarray(
            [
                len(indices) / (len(counts) * counts[metadata[i]["family"]])
                for i in indices
            ]
        )
        root = np.sqrt(w)
        for model in MODELS:
            function = functions[query][model]
            x = (
                views[model][state_indices]
                if model != "zero_action"
                else np.empty((len(indices), 0))
            )
            scale2 = sum(w[j] * np.dot(v, v) for j, v in enumerate(x)) / sum(w)
            scale = float(np.sqrt(scale2)) if scale2 else 1.0
            gram = (x / scale) @ (x / scale).T
            system = root[:, None] * ((gram + gram.T) / 2) * root[None, :]
            eigen, vectors = np.linalg.eigh(system)
            weighted_dual = root * (
                vectors @ ((vectors.T @ (root * y[indices])) / (eigen + spec["alpha"]))
            )
            coefficient = x.T @ weighted_dual / scale**2
            with np.load(packet / function["path"]) as arrays:
                if not np.array_equal(arrays["source_state_indices"], state_indices):
                    raise ValueError("source support changed")
                dual_error = max(
                    dual_error,
                    float(np.max(np.abs(weighted_dual - arrays["weighted_dual"]))),
                )
                coefficient_error = max(
                    coefficient_error,
                    float(
                        np.max(np.abs(coefficient - arrays["coefficient"]), initial=0)
                    ),
                )
                eigen_error = max(
                    eigen_error,
                    float(
                        np.max(np.abs(np.linalg.eigvalsh(gram) - arrays["eigenvalues"]))
                    ),
                )
            scale_error = max(
                scale_error,
                abs(scale - function["scale"]),
                abs(scale2 - function["scale_squared"]),
            )
            if (
                function["intercept"] != 0
                or function["centering"]
                or function["source_rows"] != len(indices)
                or function["source_families"] != len(counts)
                or function["source_state_indices"] != state_indices.tolist()
            ):
                raise ValueError("source-only fit contract mismatch")
            if model == "ordered_full":
                signs = np.where(np.arange(spec["hidden_dimension"]) % 2, -1.0, 1.0)
                rotated = (x.reshape(len(x), 12, -1)[:, :, ::-1] * signs).reshape(
                    x.shape
                )
                gauge_error = max(
                    gauge_error,
                    float(
                        np.max(np.abs(gram - (rotated / scale) @ (rotated / scale).T))
                    ),
                )
            coefficients[query, model] = coefficient
            duals[query, model] = (x, weighted_dual, scale)
            fits.append(
                {
                    "query": query,
                    "model": model,
                    "source_rows": len(x),
                    "feature_dimension": x.shape[1],
                }
            )
    fm = read_json(packet / "01/functions_manifest.json")
    if fm["held_targets_loaded"] or fm["source_indices"] != source:
        raise ValueError("source fit provenance mismatch")
    if max(scale_error, dual_error, coefficient_error, eigen_error, gauge_error) > 1e-9:
        raise ValueError("spectral source fit/gauge replay failed")
    checked, prediction_error, kernel_error = [], 0.0, 0.0
    cached = {}
    for query in spec["queries"]:
        indices = [i for i, r in enumerate(metadata) if r["query"] == query]
        states = [metadata[i]["state_index"] for i in indices]
        for model in MODELS:
            x = (
                views[model][states]
                if model != "zero_action"
                else np.empty((len(indices), 0))
            )
            src, dual, scale = duals[query, model]
            p = x @ coefficients[query, model]
            expansion = ((x / scale) @ (src / scale).T) @ dual
            kernel_error = max(kernel_error, float(np.max(np.abs(p - expansion))))
            for i, value, dx in zip(indices, p, x, strict=True):
                cached[i, model] = {
                    **metadata[i],
                    "model": model,
                    "observed_action": float(y[i]),
                    "predicted_action": float(value),
                    "error": float(value - y[i]),
                    "observation_delta_max": float(np.max(np.abs(dx), initial=0)),
                }
    for saved in rows(packet / "02/predictions.csv"):
        r = cached[int(saved["row"]), saved["model"]]
        if any(
            saved[k] != str(r[k]) for k in (*fields, "state_index", "exact_input_noop")
        ):
            raise ValueError("prediction address mismatch")
        for k in (
            "observed_action",
            "predicted_action",
            "error",
            "observation_delta_max",
        ):
            prediction_error = max(prediction_error, abs(float(saved[k]) - r[k]))
        checked.append(r)
    if len(checked) != 6516 or len({(r["row"], r["model"]) for r in checked}) != 6516:
        raise ValueError("prediction coverage mismatch")
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
        for r in checked:
            grouped[tuple(str(r[k]) for k in keys)].append(r)
        table = rows(packet / f"02/{name}.csv")
        if len(table) != len(grouped):
            raise ValueError("metric coverage mismatch")
        for saved in table:
            values = cell_metrics(grouped[tuple(saved[k] for k in keys)], spec)
            for k, value in values.items():
                if value is None:
                    if saved[k] != "":
                        raise ValueError("degenerate correlation serialized")
                else:
                    metric_error = max(metric_error, abs(value - float(saved[k])))
            metric_count += 1
    paired = defaultdict(lambda: defaultdict(list))
    for r in checked:
        if r["model"] != "ordered_full":
            continue
        base = cached[r["row"], "ordered_rank2"]
        gain = base["error"] ** 2 - r["error"] ** 2
        for axis in ("all", r["axis"]):
            paired[r["role"], axis, r["query"], r["arm"]][r["family"]].append(gain)
    table = rows(packet / "02/paired_full_vs_rank2.csv")
    if len(table) != len(paired):
        raise ValueError("paired diagnostic coverage mismatch")
    for saved in table:
        key = tuple(saved[k] for k in ("role", "axis", "query", "arm"))
        values = [np.mean(v) for v in paired[key].values()]
        for k, value in zip(
            ("mse_gain_full_vs_rank2", "ci_low", "ci_high"),
            interval(values, spec),
            strict=True,
        ):
            metric_error = max(metric_error, abs(value - float(saved[k])))
    if max(prediction_error, kernel_error, metric_error) > 1e-10:
        raise ValueError("alternate prediction/metric replay failed")
    frozen = {
        str(p.relative_to(packet)): digest(p)
        for p in sorted(packet.rglob("*"))
        if p.is_file()
    }
    write_json(out / "frozen_primary_artifact_hashes.json", frozen)
    write_csv(out / "source_fits_replayed.csv", fits)
    summary = {
        "status": "VERIFIED_FINITE_HM15_NOT_GLOBAL_SCOPE_COMPLETE",
        "new_model_calls": 0,
        "paired_query_interventions": len(metadata),
        "unique_full_states": len(state_cache),
        "predictions": len(checked),
        "source_rows": len(source),
        "held_rows": len(partitions["held"]),
        "source_functions_including_zero": len(fits),
        "independent_full_observation_max_error": feature_error,
        "independent_geometry_max_error": geometry_error,
        "independent_categorical_action_max_error": action_error,
        "source_rms_max_error": scale_error,
        "spectral_dual_max_error": dual_error,
        "spectral_coefficient_max_error": coefficient_error,
        "gram_eigenvalue_max_error": eigen_error,
        "signed_hidden_coordinate_isometry_gram_max_error": gauge_error,
        "source_kernel_vs_primal_max_error": kernel_error,
        "serialized_prediction_max_error": prediction_error,
        "metric_rows_recomputed": metric_count,
        "paired_ci_rows_recomputed": len(table),
        "serialized_metric_max_error": metric_error,
        "frozen_primary_artifacts": len(frozen),
        "same_agent_alternate_arithmetic_not_independent_researcher": True,
        "original_18_protocols_100_cards_44_questions_complete": False,
        "primary_manifest_sha256": digest(packet / "00/manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
    }
    write_json(out / "summary.json", summary)
    print(summary, flush=True)


def main():
    """Run one locked zero-forward verification without replacing prior outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        verify(args.packet, args.out)


if __name__ == "__main__":
    main()
