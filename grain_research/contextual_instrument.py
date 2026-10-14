#!/usr/bin/env python3
"""Source-constructed local measurement instruments with addressed evidence kernels.

Both examples use the same Gaussian working readout; their latent objects,
permitted invariances, clocks, calibration and source parameters differ.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from .pilot_transport import restore_units
from .temporal_measurement import state_covariance

PROFILE_TARGETS = {
    "level": np.array([0.5, 0.5]),
    "condition": np.array([-0.5, 0.5]),
    "low_condition": np.array([1.0, 0.0]),
    "high_condition": np.array([0.0, 1.0]),
}
RECORD_FIELDS = {
    "author",
    "event",
    "text",
    "scope",
    "condition",
    "time",
    "partner",
    "replicate",
}


def text_key(query, text):
    """Bind an exact raw premise to its query without any event/author latent label."""
    return hashlib.sha256((query + "\0" + text).encode()).hexdigest()


def evidence_rows(records, margins, bundle):
    """Map qualified raw records to calibrated observations and preserve omissions."""
    spec = bundle["spec"]
    output, seen = [], {}
    for record in records:
        if set(record) != RECORD_FIELDS:
            raise ValueError(
                "Use the exact raw evidence schema; labels/scores are not inputs"
            )
        if any(
            not isinstance(record[k], str) or not record[k].strip()
            for k in ("author", "event", "text", "scope")
        ):
            raise ValueError("Nonempty author/event/text/scope required")
        identity = (record["author"], record["event"])
        if identity in seen:
            if seen[identity] != record:
                raise ValueError(
                    "Conflicting event report requires an explicit revision model"
                )
            continue
        seen[identity] = record.copy()
        row = {
            **record,
            "text_sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
            "span_start": 0,
            "span_end": len(record["text"]),
            "value": None,
            "status": "unqualified_evidence",
        }
        if record["scope"] == spec["allowed_scope"]:
            if record["condition"] not in (-1, 1):
                row["status"] = "condition_outside_design"
            elif not np.isfinite([record["time"], record["partner"]]).all():
                raise ValueError("Finite recorded clock and partner input required")
            else:
                key = text_key(spec["query"], record["text"])
                if key not in margins or not np.isfinite(margins[key]):
                    raise ValueError("Missing finite raw-text reader result")
                row["value"] = float(
                    restore_units(
                        np.array([margins[key]]),
                        np.array([record["condition"]]),
                        bundle["calibration"],
                    )[0]
                )
                row["status"] = "qualified_by_observation_design"
        output.append(row)
    return output


def gaussian_readout(mean, covariance, design, noise):
    """Compute reusable affine kernels and conditional covariance, without outcomes."""
    m, s, h, r = (np.asarray(a, float) for a in (mean, covariance, design, noise))
    if (
        s.shape != (len(m), len(m))
        or h.ndim != 2
        or h.shape[1] != len(m)
        or r.shape != (len(h),)
    ):
        raise ValueError("Aligned latent and observation matrices required")
    if not all(np.isfinite(a).all() for a in (m, s, h, r)) or np.any(r <= 0):
        raise ValueError("Finite values and positive observation variances required")
    if not np.allclose(s, s.T) or np.linalg.eigvalsh(s).min() <= 0:
        raise ValueError("Positive definite working covariance required")
    factor = cho_factor(h @ s @ h.T + np.diag(r), lower=True)
    gain = cho_solve(factor, h @ s).T
    post = s - gain @ h @ s
    return {
        "kernel": gain,
        "offset": m - gain @ h @ m,
        "covariance": (post + post.T) / 2,
    }


def profile_model(rows, fitted):
    """Build two-cell observations with event-level precision and stable identity."""
    counts = np.array([sum(r["condition"] == z for r in rows) for z in (-1, 1)])
    selected = np.flatnonzero(counts)
    pools = [
        [i for i, r in enumerate(rows) if r["condition"] == (-1, 1)[cell]]
        for cell in selected
    ]
    y = np.array([np.mean([rows[i]["value"] for i in pool]) for pool in pools])
    design = np.eye(2)[selected]
    noise = np.asarray(fitted["noise"])[selected] / counts[selected]
    result = gaussian_readout(fitted["mean"], fitted["covariance"], design, noise)
    return result, PROFILE_TARGETS, y, pools, counts, design


def process_model(rows, fitted, pilot_reports):
    """Build an ordered occasion process; varying report counts change precision."""
    times = np.array(sorted({r["time"] for r in rows}), float)
    if len(times) < 2:
        raise ValueError("Process requires at least two observed occasions")
    pools = [[i for i, r in enumerate(rows) if r["time"] == t] for t in times]
    condition, partner = [], []
    for pool in pools:
        cells = {(rows[i]["condition"], rows[i]["partner"]) for i in pool}
        if len(cells) != 1:
            raise ValueError(
                "One explicit condition/partner input per occasion required"
            )
        z, u = next(iter(cells))
        condition.append(z)
        partner.append(u)
    x = np.c_[np.ones(len(times)), condition, partner]
    design = np.c_[x, np.eye(len(times))]
    covariance = block_diag(
        np.asarray(fitted["covariance"]),
        state_covariance(
            times, np.array([[-fitted["rate"]]]), np.array([[fitted["stationary"]]])
        ),
    )
    mean = np.r_[fitted["mean"], np.zeros(len(times))]
    counts = np.array([len(p) for p in pools])
    noise = fitted["noise"] * pilot_reports / counts
    y = np.array([np.mean([rows[i]["value"] for i in p]) for p in pools])
    result = gaussian_readout(mean, covariance, design, noise)
    targets = {}
    for i, name in enumerate(("level", "condition", "partner")):
        targets[name] = np.eye(len(mean))[i]
    targets["current_state"] = np.eye(len(mean))[-1]
    targets["current_response"] = targets["current_state"].copy()
    targets["current_response"][:3] = x[-1]
    gaps = np.diff(times)
    weights = np.r_[gaps[0] / 2, (gaps[:-1] + gaps[1:]) / 2, gaps[-1] / 2] / (
        times[-1] - times[0]
    )
    targets["window_state"] = np.r_[np.zeros(3), weights]
    return result, targets, y, pools, counts, design


def source_reference(name, bundle):
    """Reference position uses frozen source moments, never the application batch."""
    m = bundle["main"]
    if bundle["spec"]["kind"] == "conditional_profile":
        t = PROFILE_TARGETS[name]
        return float(t @ m["mean"]), float(t @ np.asarray(m["covariance"]) @ t)
    if name in ("level", "condition", "partner"):
        i = ("level", "condition", "partner").index(name)
        return float(m["mean"][i]), float(m["covariance"][i][i])
    if name == "current_state":
        return 0.0, float(m["stationary"])
    return None, None


def score_person(rows, bundle, *, explain=True):
    """Score one person with source-bootstrap mixture and exact evidence accounting."""
    spec = bundle["spec"]
    qualified = [r for r in rows if r["value"] is not None]
    if not qualified or (
        spec["kind"] == "occasion_process" and len({r["time"] for r in qualified}) < 2
    ):
        return {
            "status": "insufficient_qualified_evidence",
            "targets": {},
            "qualified_events": len(qualified),
            "retained_events": len(rows),
        }
    models = bundle["bootstrap"]
    if len(models) < 2:
        raise ValueError("At least two frozen source constructions required")
    scores, variances, offsets, kernels, gains = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for fitted in models:
        if spec["kind"] == "conditional_profile":
            result, targets, y, pools, counts, design = profile_model(qualified, fitted)
        else:
            result, targets, y, pools, counts, design = process_model(
                qualified, fitted, bundle["pilot_reports_per_occasion"]
            )
        for name in spec["target_names"]:
            t = targets[name]
            kernel = t @ result["kernel"]
            offset = float(t @ result["offset"])
            score = float(offset + kernel @ y)
            variance = float(t @ result["covariance"] @ t)
            if variance < -1e-9:
                raise ArithmeticError("Negative conditional variance")
            scores[name].append(score)
            variances[name].append(max(0.0, variance))
            offsets[name].append(offset)
            weights = np.zeros(len(qualified))
            for j, pool in enumerate(pools):
                weights[pool] = kernel[j] / len(pool)
            kernels[name].append(weights)
            candidate = (
                np.eye(2) if spec["kind"] == "conditional_profile" else design[-1:]
            )
            candidate_noise = (
                np.asarray(fitted["noise"])
                if spec["kind"] == "conditional_profile"
                else np.array([fitted["noise"] * bundle["pilot_reports_per_occasion"]])
            )
            cov = result["covariance"]
            gains[name].append(
                (t @ cov @ candidate.T) ** 2
                / (
                    np.einsum("ij,jk,ik->i", candidate, cov, candidate)
                    + candidate_noise
                )
            )
    answer = {}
    for name in spec["target_names"]:
        score = float(np.mean(scores[name]))
        within, between = (
            float(np.mean(variances[name])),
            float(np.var(scores[name], ddof=1)),
        )
        variance = within + between
        kernel, offset = np.mean(kernels[name], axis=0), float(np.mean(offsets[name]))
        value = np.array([r["value"] for r in qualified])
        error = abs(score - offset - kernel @ value)
        if error > 1e-8:
            raise ArithmeticError("Addressed evidence contributions fail to sum")
        mean, refvar = source_reference(name, bundle)
        position = (
            (score - mean) / np.sqrt(refvar)
            if refvar is not None and refvar > 1e-10
            else None
        )
        item = {
            "score": score,
            "variance": variance,
            "within_variance": within,
            "construction_variance": between,
            "lower": score - 1.959963984540054 * np.sqrt(variance),
            "upper": score + 1.959963984540054 * np.sqrt(variance),
            "source_term": offset,
            "evidence_term": float(kernel @ value),
            "source_relative_position": position,
            "reference_mean": mean,
            "reference_variance": refvar,
            "next_independent_report_gain": np.mean(gains[name], axis=0).tolist(),
            "perfect_repeat_gain": 0.0,
            "accounting_error": error,
            "interpretation": f"{spec['target_description']}; target={name}; conditional on recorded design and frozen source reader/model",
        }
        if explain:
            item["evidence"] = [
                {
                    "event": r["event"],
                    "text_sha256": r["text_sha256"],
                    "span_start": r["span_start"],
                    "span_end": r["span_end"],
                    "time": r["time"],
                    "condition": r["condition"],
                    "value": r["value"],
                    "weight": float(w),
                    "contribution": float(w * r["value"]),
                }
                for r, w in zip(qualified, kernel, strict=True)
            ]
        answer[name] = item
    return {
        "status": "source_constructed_working_measurement",
        "targets": answer,
        "qualified_events": len(qualified),
        "retained_events": len(rows),
        "counts": counts.tolist(),
        "empirically_absent_condition": len({r["condition"] for r in qualified}) < 2,
        "empirical_profile_design_rank": int(
            np.linalg.matrix_rank(
                design if spec["kind"] == "conditional_profile" else design[:, :3]
            )
        ),
        "profile_parameter_count": 2 if spec["kind"] == "conditional_profile" else 3,
        "clock_unit": spec["clock_unit"],
        "current_time": max(r["time"] for r in qualified)
        if spec["kind"] == "occasion_process"
        else None,
        "scope": "Text-coded expectations/occasion deviations, not personality or observed external behavior truth",
    }


def score_records(records, margins, bundle, *, clock_unit, explain=True):
    """Public raw-record entry; apply authors separately with no batch normalization."""
    if clock_unit != bundle["spec"]["clock_unit"]:
        raise ValueError("Clock units differ from the constructed instrument")
    rows = evidence_rows(records, margins, bundle)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["author"]].append(row)
    return {
        author: score_person(value, bundle, explain=explain)
        for author, value in sorted(grouped.items())
    }, rows
