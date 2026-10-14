#!/usr/bin/env python3
"""Test retained-coordinate fibers against cached real interventions; no refitting."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_prefix_compilation as M
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_prefix_compilation import check
from scripts.run_grain_math_response import write_csv

DEFAULT = ROOT / "results/grain_math_response_v1/prefix_compilation_r1"
OUTPUT = ROOT / "results/grain_math_response_v1/prefix_fiber_diagnostic_r1"


def coordinate_gradient(x, order, arm, function):
    """Differentiate the compiled polynomial in twelve fixed coordinates at fixed order."""
    a = np.asarray(x, float)
    if a.shape != (12,) or not np.isfinite(a).all() or order not in (0, 1):
        raise ValueError("finite body point and fixed order required")
    _, names = M.feature_map(a[None], np.array([order]), arm)
    weights = (
        np.asarray(function["raw_coefficient"]).reshape(len(names), -1)
        if names
        else np.empty((0, len(function["queries"])))
    )
    gradient = np.zeros((12, len(function["queries"])))
    values = {**dict(zip(M.NAMES, a, strict=True)), "relation_order": order}
    for name, weight in zip(names, weights, strict=True):
        factors = name.split("*")
        for i, factor in enumerate(factors):
            if factor == "relation_order":
                continue
            multiplier = np.prod([values[v] for j, v in enumerate(factors) if j != i])
            gradient[M.NAMES.index(factor)] += multiplier * weight
    return gradient


def robust_pair_bound(output_difference, input_distance, lipschitz):
    """Triangle inequality: max endpoint error >= (|delta y|-L*epsilon)_+/2."""
    d, e, l = (float(v) for v in (output_difference, input_distance, lipschitz))
    if not np.isfinite([d, e, l]).all() or min(e, l) < 0:
        raise ValueError("finite difference and nonnegative distance/L required")
    return max(0.0, (abs(d) - l * e) / 2)


def reconstructed_coordinates(base, current, donor, frame, span, arm):
    """Reproduce the original float32 patch assignment, without executing the reader."""
    x, h, target, u = (np.asarray(v, float) for v in (base, current, donor, frame))
    if (
        x.shape != (12,)
        or h.ndim != 2
        or target.shape != h.shape
        or span not in M.SPANS
    ):
        raise ValueError("matched local token states and complete base point required")
    if not np.allclose(u.T @ u, np.eye(2), atol=1e-10):
        raise ValueError("orthonormal frozen frame required")
    d = target - h
    projected = (d @ u) @ u.T
    if arm in ("full_property", "neutral"):
        patched = target
    elif arm == "projection":
        patched = h + projected
    elif arm == "complement":
        patched = h + d - projected
    elif arm == "random_matched":
        raise ValueError("random arm requires its original matched random displacement")
    elif arm == "baseline":
        return x.copy()
    else:
        raise ValueError("unknown declared patch arm")
    # Original observer hook assigns torch.float32; preserve the rounding here.
    patched = patched.astype(np.float32).astype(float)
    start = M.SPANS.index(span) * 2
    x[start : start + 2] = patched.mean(axis=0) @ u
    return x


def analyze(packet, out):
    """Expose fiber collisions using saved outputs, with exact/rounded scopes distinct."""
    if out.exists():
        raise FileExistsError("preserve prior fiber diagnostic")
    manifest = check(packet, models=True)
    spec = manifest["spec"]
    primary = ROOT / spec["source_packet"]
    frozen_files(primary)
    for name, expected in read_json(packet / "03/artifact_hashes.json").items():
        if digest(packet / name) != expected:
            raise ValueError("frozen compilation artifact changed")
    source_ledger = read_json(
        ROOT / spec["source_verification"] / "frozen_primary_artifact_hashes.json"
    )
    for name, expected in source_ledger.items():
        if digest(primary / name) != expected:
            raise ValueError("original intervention artifact changed")
    models = read_json(packet / "01/functions.json")
    meta = read_json(packet / "00/metadata.json")
    index = {r["doc_id"]: i for i, r in enumerate(meta)}
    captures = read_json(primary / "00/design.json")
    lookup = {(r["doc_id"], r["query"]): r for r in captures}
    pairs = read_json(primary / "00/pairs.json")
    patch_design = [
        r
        for r in read_json(primary / "00/patch_design.json")
        if r["layer"] == 16 and r["arm"] != "noop"
    ]
    patch_index = {(r["pair_id"], r["query"], r["arm"]): r for r in patch_design}
    with np.load(packet / "00/prefix_inputs.npz", allow_pickle=False) as a:
        x, order, u = a["x"].copy(), a["order"].copy(), a["frame"].copy()
    with np.load(primary / "02/basis.npz", allow_pickle=False) as a:
        random = a["L16_R"].copy()
    predictions, witnesses, summary = [], [], []
    gradient_error, bound_violation, hidden_identity_error = 0.0, 0.0, 0.0
    consumed = {}
    for pair in pairs:
        i = index[pair["receiver_doc"]]
        axis = pair["axis"]
        current_row = lookup[pair["receiver_doc"], spec["queries"][0]]
        donor_row = lookup[pair["donor_doc"], spec["queries"][0]]
        with np.load(
            primary / "01/units" / f"unit_{current_row['unit']:04d}.npz",
            allow_pickle=False,
        ) as a:
            current = {
                span: a[f"L16_{span}"].astype(float)
                for span in (f"{axis}_property", f"{axis}_neutral")
            }
        with np.load(
            primary / "01/units" / f"unit_{donor_row['unit']:04d}.npz",
            allow_pickle=False,
        ) as a:
            donor = {span: a[f"L16_{span}"].astype(float) for span in current}
        prop = f"{axis}_property"
        d = donor[prop] - current[prop]
        projected = (d @ u) @ u.T
        hidden_identity_error = max(
            hidden_identity_error, float(np.max(np.abs((d - projected) @ u)))
        )
        views = {"baseline": x[i].copy()}
        for arm in ("full_property", "projection", "complement"):
            views[arm] = reconstructed_coordinates(
                x[i].copy(), current[prop], donor[prop], u, prop, arm
            )
        span = f"{axis}_neutral"
        views["neutral"] = reconstructed_coordinates(
            x[i].copy(), current[span], donor[span], u, span, "neutral"
        )
        z = (d @ random) @ random.T
        norm, target_norm = np.linalg.norm(z, axis=1), np.linalg.norm(projected, axis=1)
        shift = (
            z
            * np.divide(target_norm, norm, out=np.zeros_like(norm), where=norm > 0)[
                :, None
            ]
        )
        views["random_matched"] = x[i].copy()
        start = M.SPANS.index(prop) * 2
        views["random_matched"][start : start + 2] = (current[prop] + shift).astype(
            np.float32
        ).astype(float).mean(axis=0) @ u
        for query in spec["queries"]:
            q = spec["queries"].index(query)
            capture = lookup[pair["receiver_doc"], query]
            capture_path = primary / "01/units" / f"unit_{capture['unit']:04d}.json"
            base = read_json(capture_path)["native"]["conditional_score"]
            observed = {"baseline": base}
            for arm in spec_primary_arms():
                r = patch_index[pair["pair_id"], query, arm]
                path = primary / "03/units" / f"unit_{r['unit']:05d}.json"
                saved = read_json(path)
                if saved["address"] != r or saved["live_body_error"] != 0:
                    raise ValueError("live patch did not use the saved body state")
                observed[arm] = saved["native"]["conditional_score"]
                consumed[str(path.relative_to(primary))] = digest(path)
            address = {
                k: pair[k]
                for k in ("pair_id", "family", "role", "axis", "polarity", "order")
            }
            for model_arm in spec["arms"]:
                fn = models[model_arm]
                predicted, gradients = {}, {}
                for arm, point in views.items():
                    phi, _ = M.feature_map(point[None], np.array([order[i]]), model_arm)
                    predicted[arm] = float(M.apply_function(phi, fn)[0, q])
                    gradients[arm] = coordinate_gradient(
                        point, int(order[i]), model_arm, fn
                    )[:, q]
                    integral = (
                        (gradients["baseline"] + gradients[arm])
                        @ (point - views["baseline"])
                        / 2
                    )
                    gradient_error = max(
                        gradient_error,
                        abs(integral - predicted[arm] + predicted["baseline"]),
                    )
                    predictions.append(
                        {
                            **address,
                            "query": query,
                            "model_arm": model_arm,
                            "patch_arm": arm,
                            "observed": observed[arm],
                            "predicted": predicted[arm],
                            "observed_action": observed[arm] - base,
                            "predicted_action": predicted[arm] - predicted["baseline"],
                            "retained_input_distance": float(
                                np.linalg.norm(point - views["baseline"])
                            ),
                        }
                    )
                for kind, left, right in (
                    ("complement_vs_baseline", "complement", "baseline"),
                    ("full_vs_projection", "full_property", "projection"),
                ):
                    distance = float(np.linalg.norm(views[left] - views[right]))
                    delta = observed[left] - observed[right]
                    lipschitz = float(
                        max(
                            np.linalg.norm(gradients[left]),
                            np.linalg.norm(gradients[right]),
                        )
                    )
                    bound = robust_pair_bound(delta, distance, lipschitz)
                    endpoint_error = max(
                        abs(predicted[left] - observed[left]),
                        abs(predicted[right] - observed[right]),
                    )
                    bound_violation = max(bound_violation, bound - endpoint_error)
                    witnesses.append(
                        {
                            **address,
                            "query": query,
                            "model_arm": model_arm,
                            "kind": kind,
                            "input_distance_float32": distance,
                            "output_difference": delta,
                            "local_compiled_L_bound": lipschitz,
                            "minimum_max_endpoint_error": bound,
                            "observed_compiled_max_endpoint_error": endpoint_error,
                            "ideal_retained_coordinate_difference": 0.0,
                            "ideal_statement_is_algebraic_not_recorded_float_equality": True,
                        }
                    )
    groups = defaultdict(list)
    for row in witnesses:
        groups[
            row["role"],
            row["axis"],
            row["order"],
            row["query"],
            row["model_arm"],
            row["kind"],
        ].append(row)
    for key, rows in sorted(groups.items()):
        summary.append(
            {
                **dict(
                    zip(
                        ("role", "axis", "order", "query", "model_arm", "kind"),
                        key,
                        strict=True,
                    )
                ),
                "pairs": len(rows),
                "mean_abs_output_difference": float(
                    np.mean([abs(r["output_difference"]) for r in rows])
                ),
                "maximum_input_distance_float32": max(
                    r["input_distance_float32"] for r in rows
                ),
                "mean_error_lower_bound": float(
                    np.mean([r["minimum_max_endpoint_error"] for r in rows])
                ),
                "positive_bounds": sum(
                    r["minimum_max_endpoint_error"] > 0 for r in rows
                ),
            }
        )
    if (
        len(predictions) != 9216
        or len(witnesses) != 3072
        or len(consumed) != 1920
        or len(summary) != 288
    ):
        raise ValueError("all fixed pair/query/model/fiber addresses required")
    if (
        gradient_error > 1e-10
        or bound_violation > 1e-10
        or hidden_identity_error > 1e-10
    ):
        raise ValueError("numeric gradient, fiber identity or triangle bound failed")
    write_csv(out / "intervention_reading.csv", predictions)
    write_csv(out / "fiber_witnesses.csv", witnesses)
    write_csv(out / "fiber_summary.csv", summary)
    write_json(out / "consumed_primary_hashes.json", consumed)
    verification = {
        "status": "COMPLETE_RESULT_GUIDED_FIBER_DIAGNOSTIC",
        "new_model_calls": 0,
        "new_fit": False,
        "pairs": len(pairs),
        "actual_saved_interventions_used": len(consumed),
        "compiled_intervention_rows": len(predictions),
        "bound_rows": len(witnesses),
        "gradient_integral_max_error": gradient_error,
        "triangle_bound_max_violation": bound_violation,
        "ideal_complement_projection_max_error": hidden_identity_error,
        "float32_rounding_reconstructed": True,
        "ideal_algebra_scope": "orthonormal projection fibers; neural outputs at unrounded ideal endpoints were not measured",
        "conditioning_scope": "same receiver and original query-conditioned layer16 states fixed across patch arms",
        "rounded_claim_scope": "endpoint error lower bounds for the four fixed compiled functions, not impossibility for arbitrary discontinuous functions",
        "observed_capture_replay_error_is_zero": True,
        "human_personality_validity": False,
        "original_scope_complete": False,
        "source_compilation_manifest_sha256": digest(packet / "00/manifest.json"),
        "source_function_manifest_sha256": digest(
            packet / "01/functions_manifest.json"
        ),
        "code_sha256": digest(Path(__file__)),
    }
    write_json(out / "verification.json", verification)
    write_json(
        out / "artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in out.iterdir() if p.is_file()},
    )
    print(verification, flush=True)


def spec_primary_arms():
    """The original five fixed arms, never an outcome-selected subset."""
    return ("full_property", "neutral", "projection", "complement", "random_matched")


def main():
    """Read-only analysis of completed old interventions with separate output scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    analyze(args.source, args.out)


if __name__ == "__main__":
    main()
