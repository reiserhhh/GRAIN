#!/usr/bin/env python3
"""HM14 alternate state, weighted least-squares and scalar-action replay; no forwards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import write_json
from scripts.run_grain_math_ordered_moments import DEFAULT, check
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_upstream_routes import collector_guard

OUTPUT = ROOT / "results/grain_math_response_v1/ordered_moment_verification_r1"
SPANS = ("A_property", "A_actor", "A_neutral", "T_property", "T_actor", "T_neutral")
PROPERTY = ("A_property", "T_property")
WIDTH = {"zero_action": 0, "span_mean": 12, "ordered_first": 16, "ordered_complete": 24}


def rows(path):
    """Load an output table without calling primary summarization."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def observe(bodies, frame):
    """Project each token first, then explicitly accumulate its position powers."""
    moments = {}
    for name in SPANS:
        h = bodies[name].astype(float)
        t = len(h)
        positions = [0.0] if t == 1 else [-1 + 2 * j / (t - 1) for j in range(t)]
        projected = np.asarray([frame.T @ vector for vector in h])
        moments[name] = np.asarray(
            [
                sum(p**k * v for p, v in zip(positions, projected, strict=True)) / t
                for k in range(4)
            ]
        )
    return np.asarray(
        [v for name in SPANS for v in moments[name][0]]
        + [
            moments[name][k, j]
            for k in (1, 2, 3)
            for name in PROPERTY
            for j in range(2)
        ]
    )


def native_score(record, values):
    """Recalculate a native categorical expectation from saved logits, not its score."""
    logits = np.asarray(record["native"]["candidate_logits"], float)
    weights = np.exp(logits - max(logits))
    return float(np.dot(weights / sum(weights), values))


def basic_metrics(actual, predicted):
    """Recalculate correlation by centered dot products and scalar action losses."""
    y, p = np.asarray(actual), np.asarray(predicted)
    e = p - y
    yc, pc = y - y.mean(), p - p.mean()
    correlation = None
    if y.std() > 1e-12 and p.std() > 1e-12:
        correlation = float(yc @ pc / np.sqrt((yc @ yc) * (pc @ pc)))
    return {
        "n": len(y),
        "rmse": float(np.sqrt(e @ e / len(e))),
        "bias": float(sum(e) / len(e)),
        "r": correlation,
        "observed_mean": float(sum(y) / len(y)),
        "predicted_mean": float(sum(p) / len(p)),
        "sign_agreement": float(sum(np.sign(p) == np.sign(y)) / len(y)),
    }


def interval(values, spec):
    """Resample lexical families, keeping the frozen source law conditional."""
    x = np.asarray(values, float)
    indices = np.random.default_rng(spec["bootstrap_seed"]).integers(
        len(x), size=(spec["bootstrap_replicates"], len(x))
    )
    means = np.sum(x[indices], axis=1) / len(x)
    return float(x.mean()), *np.percentile(means, [2.5, 97.5]).tolist()


def cell_metrics(cell, spec):
    """Derive every primary metric from independently checked predictions."""
    family = defaultdict(list)
    for row in cell:
        y, p = row["observed_action"], row["predicted_action"]
        family[row["family"]].append((p - y, (p - y) ** 2, y**2 - (p - y) ** 2))
    values = np.asarray([np.average(v, axis=0) for v in family.values()])
    _, lo, hi = interval(values[:, 0], spec)
    gain, gain_lo, gain_hi = interval(values[:, 2], spec)
    return {
        **basic_metrics(
            [r["observed_action"] for r in cell], [r["predicted_action"] for r in cell]
        ),
        "families": len(family),
        "family_rmse": float(np.sqrt(values[:, 1].mean())),
        "mean_error_ci_low": lo,
        "mean_error_ci_high": hi,
        "mse_gain_vs_zero": gain,
        "mse_gain_ci_low": gain_lo,
        "mse_gain_ci_high": gain_hi,
        "noops": sum(r["exact_input_noop"] for r in cell),
        "max_observation_difference": max(r["observation_delta_max"] for r in cell),
    }


def verify(packet, out):
    """Reconstruct all addressed interventions and fit source-only laws by augmented LS."""
    if out.exists():
        raise FileExistsError("preserve earlier verification; use a new output path")
    manifest = check(packet, models=True)
    spec = manifest["spec"]
    original_ledger = read_json(packet / "02/artifact_hashes.json")
    for name, expected in original_ledger.items():
        if digest(packet / name) != expected:
            raise ValueError(f"primary artifact changed: {name}")
    span, mean, permutation = (
        ROOT / spec[k] for k in ("span_packet", "mean_packet", "permutation_packet")
    )
    captures = read_json(span / "00/design.json")
    addresses = {(r["doc_id"], r["query"]): r for r in captures}
    canonical = {r["doc_id"]: r for r in captures if r["query"] == spec["queries"][0]}
    with np.load(span / "02/basis.npz") as arrays:
        frame = arrays["L16_U"].copy()
    bodies = {}
    reconstruction_max = parity_max = 0.0
    for doc_id, row in canonical.items():
        with np.load(span / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            bodies[doc_id] = {name: arrays[f"L16_{name}"].copy() for name in SPANS}
        for name in PROPERTY:
            h = bodies[doc_id][name].astype(float)
            t = len(h)
            p = np.asarray(
                [0.0] if t == 1 else [-1 + 2 * j / (t - 1) for j in range(t)]
            )
            vandermonde = np.asarray(
                [[position**k for position in p] for k in range(t)]
            )
            moments = np.asarray(
                [
                    sum(
                        position**k * vector
                        for position, vector in zip(p, h, strict=True)
                    )
                    / t
                    for k in range(t)
                ]
            )
            rebuilt = np.linalg.lstsq(vandermonde, t * moments, rcond=None)[0]
            reconstruction_max = max(
                reconstruction_max, float(np.max(np.abs(rebuilt - h)))
            )
            for k in range(4):
                forward = (
                    sum(
                        position**k * vector
                        for position, vector in zip(p, h, strict=True)
                    )
                    / t
                )
                reverse = (
                    sum(
                        position**k * vector
                        for position, vector in zip(p, h[::-1], strict=True)
                    )
                    / t
                )
                parity_max = max(
                    parity_max, float(np.max(np.abs(reverse - (-1) ** k * forward)))
                )
    oldmean = read_json(mean / "00/design.json")
    centered = {
        (r["pair_id"], r["query"]): r for r in oldmean if r["arm"] == "centered_only"
    }
    design_blocks = (
        (
            span,
            "03/units",
            5,
            [
                r
                for r in read_json(span / "00/patch_design.json")
                if r["layer"] == 16 and r["arm"] == "full_property"
            ],
        ),
        (mean, "01/units", 4, oldmean),
        (permutation, "01/units", 4, read_json(permutation / "00/design.json")),
    )
    metadata = read_json(packet / "00/metadata.json")
    with np.load(packet / "00/observations.npz") as arrays:
        references, targets = arrays["reference"].copy(), arrays["target"].copy()
        if not np.array_equal(frame, arrays["frame"]):
            raise ValueError("source frame mismatch")
    y, source_indices, held_indices = np.empty(len(metadata)), [], []
    for label, bucket in (("source", source_indices), ("held", held_indices)):
        with np.load(packet / f"00/{label}_targets.npz") as arrays:
            bucket.extend(arrays["indices"].tolist())
            y[arrays["indices"]] = arrays["y"]
    actual_source = [i for i, r in enumerate(metadata) if r["role"] == "source"]
    if (
        source_indices != actual_source
        or set(source_indices) & set(held_indices)
        or sorted(source_indices + held_indices) != list(range(len(metadata)))
    ):
        raise ValueError("source/held role partition mismatch")
    values = np.asarray(
        read_json(span / "00/manifest.json")["alphabet"]["values"], float
    )
    state_cache, records = {}, {}
    feature_error = action_error = 0.0
    count = 0
    for oldpacket, folder, width, design in design_blocks:
        for oldrow in design:
            saved = metadata[count]
            if saved["row"] != count or any(
                saved[k] != oldrow[k]
                for k in (
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
            ):
                raise ValueError("independent intervention address mismatch")
            name = f"{oldrow['axis']}_property"
            key = oldrow["pair_id"], oldrow["arm"]
            if key not in state_cache:
                h = bodies[oldrow["receiver_doc"]][name]
                d = bodies[oldrow["donor_doc"]][name]
                caddr, daddr = (
                    canonical[oldrow[k]]["spans"][name]
                    for k in ("receiver_doc", "donor_doc")
                )
                if (
                    caddr["raw"] != daddr["raw"]
                    or caddr["token_ids"] != daddr["token_ids"]
                ):
                    raise ValueError("original word/token identity mismatch")
                reference = h
                delta = d.astype(float) - h.astype(float)
                avg = np.broadcast_to(np.mean(delta, axis=0), delta.shape)
                if oldrow["arm"] == "full_property":
                    target = d
                elif oldrow["arm"] == "mean_only":
                    target = (h.astype(float) + avg).astype(np.float32)
                elif oldrow["arm"] == "centered_only":
                    target = (h.astype(float) + (delta - avg)).astype(np.float32)
                else:
                    if oldrow["arm"] == "centered_reverse":
                        reference = (h.astype(float) + (delta - avg)).astype(np.float32)
                    target = reference[np.arange(len(h) - 1, -1, -1)]
                rb, tb = (
                    dict(bodies[oldrow["receiver_doc"]]),
                    dict(bodies[oldrow["receiver_doc"]]),
                )
                rb[name], tb[name] = reference, target
                state_cache[key] = (
                    observe(rb, frame),
                    observe(tb, frame),
                    reference,
                    target,
                )
            a, b, reference, target = state_cache[key]
            feature_error = max(
                feature_error,
                float(np.max(np.abs(a - references[count]))),
                float(np.max(np.abs(b - targets[count]))),
            )
            if (
                saved["reference_state_sha256"]
                != hashlib.sha256(reference.tobytes()).hexdigest()
                or saved["target_state_sha256"]
                != hashlib.sha256(target.tobytes()).hexdigest()
            ):
                raise ValueError("reference/target full-state hash mismatch")
            if saved["exact_input_noop"] != (reference.tobytes() == target.tobytes()):
                raise ValueError("exact input-noop mismatch")
            oldunit = oldpacket / folder / f"unit_{oldrow['unit']:0{width}d}.json"
            baseaddr = addresses[oldrow["receiver_doc"], oldrow["query"]]
            refpath = span / "01/units" / f"unit_{baseaddr['unit']:04d}.json"
            if oldrow["arm"] == "centered_reverse":
                refpath = (
                    mean
                    / "01/units"
                    / f"unit_{centered[oldrow['pair_id'], oldrow['query']]['unit']:04d}.json"
                )
            if saved["unit_path"] != str(oldunit.relative_to(ROOT)) or saved[
                "reference_path"
            ] != str(refpath.relative_to(ROOT)):
                raise ValueError("actual readout reference-address mismatch")
            for path in (oldunit, refpath):
                if path not in records:
                    records[path] = read_json(path)
            unit, refunit = records[oldunit], records[refpath]
            if unit["address"] != oldrow:
                raise ValueError("actual intervention unit mismatch")
            action = native_score(unit, values) - native_score(refunit, values)
            action_error = max(action_error, abs(action - y[count]))
            count += 1
    if (
        count != spec["paired_query_rows"]
        or feature_error > 1e-10
        or action_error > 1e-12
        or reconstruction_max > 1e-10
        or parity_max > 1e-12
    ):
        raise ValueError("independent full state/feature/readout reconstruction failed")
    # Fit from the verified serialized source observation differences. The
    # alternate QR/SVD augmented solve isolates fitting from summation roundoff.
    delta = targets - references
    functions = read_json(packet / "01/functions.json")
    coeff_error = fit_score_error = 0.0
    for query in spec["queries"]:
        indices = [i for i in source_indices if metadata[i]["query"] == query]
        family_counts = {
            f: sum(metadata[i]["family"] == f for i in indices)
            for f in {metadata[i]["family"] for i in indices}
        }
        weights = np.asarray(
            [
                len(indices)
                / (len(family_counts) * family_counts[metadata[i]["family"]])
                for i in indices
            ]
        )
        for model, width in WIDTH.items():
            function = functions[query][model]
            x = delta[indices, :width]
            scale = np.sqrt((x * x).T @ weights / weights.sum())
            threshold = (
                100 * np.finfo(float).eps * max(1, float(np.max(np.abs(x), initial=0)))
            )
            active = scale > threshold
            safe = np.where(active, scale, 1)
            z = x / safe
            z[:, ~active] = 0
            design = np.vstack(
                (np.sqrt(weights)[:, None] * z, np.sqrt(spec["alpha"]) * np.eye(width))
            )
            target = np.concatenate((np.sqrt(weights) * y[indices], np.zeros(width)))
            coefficient = (
                np.linalg.lstsq(design, target, rcond=None)[0] if width else np.empty(0)
            )
            saved_coefficient = np.asarray(function["coefficient"])
            coeff_error = max(
                coeff_error,
                float(np.max(np.abs(coefficient - saved_coefficient), initial=0)),
            )
            fit_score_error = max(
                fit_score_error,
                float(
                    np.max(
                        np.abs(
                            delta[:, :width] @ (coefficient / safe)
                            - delta[:, :width] @ np.asarray(function["raw_coefficient"])
                        ),
                        initial=0,
                    )
                ),
            )
            if (
                function["intercept"] != 0
                or function["centering"]
                or function["source_rows"] != len(indices)
                or function["source_families"] != len(family_counts)
                or not np.array_equal(active, function["active"])
                or not np.allclose(safe, function["scale"], atol=1e-12, rtol=1e-12)
            ):
                raise ValueError("source-only scale/intercept/family fit mismatch")
    fitted_manifest = read_json(packet / "01/functions_manifest.json")
    if (
        fitted_manifest["held_targets_loaded"]
        or fitted_manifest["source_indices"] != source_indices
    ):
        raise ValueError("held targets in source-fit provenance")
    if coeff_error > 1e-9 or fit_score_error > 1e-10:
        raise ValueError("alternate augmented least-squares fit failed")
    predicted_rows, table_error = [], 0.0
    for saved in rows(packet / "02/predictions.csv"):
        i, model, query = int(saved["row"]), saved["model"], saved["query"]
        width = WIDTH[model]
        value = float(
            delta[i, :width] @ np.asarray(functions[query][model]["raw_coefficient"])
        )
        row = {
            **metadata[i],
            "model": model,
            "observed_action": float(y[i]),
            "predicted_action": value,
            "error": value - y[i],
            "observation_delta_max": float(np.max(np.abs(delta[i, :width]), initial=0)),
        }
        if any(
            saved[k] != str(row[k])
            for k in (
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
                "row",
                "model",
                "exact_input_noop",
            )
        ):
            raise ValueError("prediction row identity mismatch")
        for key in (
            "observed_action",
            "predicted_action",
            "error",
            "observation_delta_max",
        ):
            table_error = max(table_error, abs(float(saved[key]) - row[key]))
        predicted_rows.append(row)
    if len(predicted_rows) != len(metadata) * len(WIDTH) or len(
        {(r["row"], r["model"]) for r in predicted_rows}
    ) != len(predicted_rows):
        raise ValueError("full prediction coverage failed")
    metric_count = 0
    for name, keys in (
        ("action_metrics", ("role", "query", "model", "arm")),
        (
            "order_token_metrics",
            ("role", "axis", "query", "model", "arm", "order", "opportunity", "tokens"),
        ),
        ("family_metrics", ("role", "family", "query", "model", "arm")),
    ):
        grouped = defaultdict(list)
        for row in predicted_rows:
            grouped[tuple(str(row[k]) for k in keys)].append(row)
        saved_table = rows(packet / f"02/{name}.csv")
        if len(saved_table) != len(grouped):
            raise ValueError("metric group coverage mismatch")
        for saved in saved_table:
            values = cell_metrics(grouped[tuple(saved[k] for k in keys)], spec)
            for key, value in values.items():
                if value is None:
                    if saved[key] != "":
                        raise ValueError("degenerate correlation reported as numeric")
                else:
                    table_error = max(table_error, abs(float(saved[key]) - value))
            metric_count += 1
    if table_error > 1e-10:
        raise ValueError("serialized scalar action metrics failed")
    # Result-after paired diagnostic, not a new primary model selection.
    by_model = {(r["row"], r["model"]): r for r in predicted_rows}
    paired = defaultdict(list)
    for row in predicted_rows:
        if row["model"] not in ("ordered_first", "ordered_complete"):
            continue
        base = by_model[row["row"], "span_mean"]
        gain = base["error"] ** 2 - row["error"] ** 2
        paired[row["role"], row["axis"], row["query"], row["arm"], row["model"]].append(
            (row["family"], gain)
        )
    paired_rows = []
    for key, values in sorted(paired.items()):
        family = defaultdict(list)
        for f, gain in values:
            family[f].append(gain)
        gain, lo, hi = interval([np.mean(v) for v in family.values()], spec)
        paired_rows.append(
            {
                **dict(
                    zip(("role", "axis", "query", "arm", "model"), key, strict=True)
                ),
                "n": len(values),
                "families": len(family),
                "mse_gain_vs_span_mean": gain,
                "ci_low": lo,
                "ci_high": hi,
                "result_after_diagnostic": True,
            }
        )
    frozen = {
        str(p.relative_to(packet)): digest(p)
        for p in sorted(packet.rglob("*"))
        if p.is_file()
    }
    write_json(out / "frozen_primary_artifact_hashes.json", frozen)
    write_csv(out / "paired_ordered_vs_mean.csv", paired_rows)
    summary = {
        "status": "VERIFIED_FINITE_HM14_NOT_GLOBAL_SCOPE_COMPLETE",
        "new_model_calls": 0,
        "paired_query_interventions": count,
        "predictions": len(predicted_rows),
        "source_rows": len(source_indices),
        "held_rows": len(held_indices),
        "geometry_rows": len(canonical) * 2,
        "source_functions": len(spec["queries"]) * len(WIDTH),
        "full_dimensional_reconstruction_max": reconstruction_max,
        "reversal_parity_max": parity_max,
        "independent_feature_max_error": feature_error,
        "independent_categorical_action_max_error": action_error,
        "augmented_lstsq_coefficient_max_error": coeff_error,
        "augmented_lstsq_score_max_error": fit_score_error,
        "metric_rows_recomputed": metric_count,
        "serialized_metric_max_error": table_error,
        "paired_diagnostic_rows": len(paired_rows),
        "frozen_primary_artifacts": len(frozen),
        "same_agent_alternate_arithmetic_not_independent_researcher": True,
        "original_18_protocols_100_cards_44_questions_complete": False,
        "primary_manifest_sha256": digest(packet / "00/manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
    }
    write_json(out / "summary.json", summary)
    print(summary, flush=True)


def main():
    """Run one locked zero-forward verification and preserve its complete outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        verify(args.packet, args.out)


if __name__ == "__main__":
    main()
