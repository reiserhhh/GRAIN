#!/usr/bin/env python3
"""Alternate finite HM92 arithmetic; never call a neural model or fit a reader."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_cold_evidence as C
from scripts.run_grain_hierarchical_query_hm88 import load, sha, unit, verify, write

OUT = ROOT / "results/grain_math_response_v1/event_history_observation_r1"


def reading(raw, row, bundle):
    """Use direct strict-past exponentials and per-head span normalization."""
    ix = row["body_indices"]
    keys, values = raw["k"][ix].astype(float), raw["v"][ix].astype(float)
    n, heads = raw["q_body"].shape[:2]
    labels = np.asarray(row["root_labels"])
    gate = np.zeros((heads, n))
    for head in range(heads):
        kh = head // (heads // keys.shape[1])
        scores = raw["q_body"][:, head] @ keys[:, kh].T
        allowed = np.arange(n)[None] < np.arange(1, n)[:, None]
        log = np.where(allowed, scores[1:], -np.inf)
        prob = np.exp(log - log.max(1)[:, None])
        prob /= prob.sum(1)[:, None]
        gate[head, 1:] = prob[:, labels == 0].sum(1)
        gate[head, labels >= 0] = (labels[labels >= 0] == 0).astype(float)
    groups = [np.asarray(g, int) for g in row["action_groups"]]
    events = [
        [
            j
            for j, pos in enumerate(ix)
            if pos in row["groups"][row["event_order"].index(e)]
        ]
        for e in range(2)
    ]
    fields, factors = np.zeros((10, 6)), []
    for qi in range(6):
        query = C.rotate(
            bundle["raw_queries"][qi],
            ix[-1] + bundle["gaps"][qi],
            bundle["frequencies"],
            bundle["scaling"],
        )
        pp, cc, locals_p = [], [], [[], []]
        for head in range(heads):
            kh = head // (heads // keys.shape[1])
            log = keys[:, kh] @ query[head]
            p = np.exp(log - log.max())
            p /= p.sum()
            c = values[:, kh] @ np.asarray(bundle["directions"])[qi, head]
            pp.append(p)
            cc.append(c)
            fields[0, qi] += p @ c
            fields[1, qi] += np.sum(p * gate[head] * c)
            for event, group in enumerate(groups):
                local = np.exp(log[group] - log[group].max())
                local /= local.sum()
                locals_p[event].append(local)
                fields[2 + 3 * event, qi] += local @ c[group]
                fields[3 + 3 * event, qi] += np.sum(
                    local * gate[head, group] * c[group]
                )
                fields[4 + 3 * event, qi] += sum(c[group]) / len(group)
                eg = events[event]
                fields[8 + event, qi] += np.sum(p[eg] * gate[head, eg] * c[eg])
        c = np.asarray(cc)
        factors.append(
            [
                (np.asarray(locals_p[e]), gate[:, g], c[:, g])
                for e, g in enumerate(groups)
            ]
        )
    # Reproduce the recorded float32 mean unit; also check a float64 summation.
    recorded = np.stack(
        [
            (np.add.reduce(raw["v"][ix][g], axis=0) / np.float32(len(g))).ravel()
            for g in groups
        ]
    )
    precise = np.stack([values[g].sum(0).ravel() / len(g) for g in groups])
    return fields, factors, recorded, float(np.max(np.abs(recorded - precise)))


def read_table(name):
    """Read every row of a sealed primary table."""
    with (OUT / "02" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    """Check raw observations, every character, laws, accounts and controls."""
    manifest = load(OUT / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(OUT, manifest["inputs"])
    verify(OUT, load(OUT / "01/raw_seal.json"))
    verify(OUT, load(OUT / "02/output_seal.json"))
    if load(OUT / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Complete captures required")
    spec, rows = manifest["spec"], load(OUT / "00/rows.json")
    frames, targets, fields = spec["frames"], spec["targets"], manifest["fields"]
    points = list(itertools.product((-1, 1), repeat=6))
    masks = list(itertools.product((0, 1), repeat=6))
    names = [
        "".join(c for c, bit in zip("xyghab", mask, strict=True) if bit) or "1"
        for mask in masks
    ]
    with np.load(OUT / "02/observations.npz", allow_pickle=False) as saved:
        actual = np.zeros_like(saved["fields"])
        saved_fields, saved_coeff, saved_means = (
            saved["fields"],
            saved["coefficients"],
            saved["raw_value_means"],
        )
        raw_means = np.zeros_like(saved_means)
        if not np.array_equal(saved["points"], points):
            raise ValueError("Point layout changed")
    errors = dict.fromkeys(
        (
            "fields",
            "coefficients",
            "means",
            "products",
            "laws",
            "tables",
            "parseval",
            "reader",
        ),
        0.0,
    )
    checks, float32_mean_roundoff, previous, products = 0, 0.0, {}, {}
    bundle = load(OUT / "00/bundle.json")
    for row in rows:
        fi, order, ti = (
            frames.index(row["frame"]),
            row["order"],
            targets.index(row["target"]),
        )
        point = tuple(row["coordinates"])
        key = fi, order, ti, points.index(point)
        observation, factor, means, roundoff = reading(
            unit(OUT, row["unit"]), row, bundle
        )
        actual[key], raw_means[key] = observation, means
        errors["fields"] = max(
            errors["fields"], float(np.max(np.abs(observation - saved_fields[key])))
        )
        errors["means"] = max(
            errors["means"], float(np.max(np.abs(means - saved_means[key])))
        )
        float32_mean_roundoff = max(float32_mean_roundoff, roundoff)
        checks += observation.size + means.size
        x, y, g, h, a, b = point
        base = fi, order, ti, x, y, g, h, b
        if a < 0:
            previous[base] = factor
        else:
            for qi in range(6):
                old, new = previous[base][qi][1], factor[qi][1]
                delta = [new[j] - old[j] for j in range(3)]
                labels = (
                    "selection",
                    "subject_gate",
                    "value_content",
                    "selection_gate",
                    "selection_value",
                    "gate_value",
                    "triple",
                )
                subsets = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
                terms = {
                    name: float(
                        np.sum(
                            np.prod(
                                [delta[j] if j in subset else old[j] for j in range(3)],
                                axis=0,
                            )
                        )
                    )
                    for name, subset in zip(labels, subsets, strict=True)
                }
                terms["delta"] = float(
                    np.sum(np.prod(new, axis=0)) - np.sum(np.prod(old, axis=0))
                )
                terms["identity_error"] = abs(
                    terms["delta"] - sum(terms[n] for n in labels)
                )
                products[row["unit"], targets[qi]] = terms
                errors["products"] = max(errors["products"], terms["identity_error"])
    coef = np.zeros_like(saved_coeff)
    for ki, mask in enumerate(masks):
        for ci, point in enumerate(points):
            sign = np.prod([point[j] for j in range(6) if mask[j]])
            coef[:, :, :, ki] += sign * actual[:, :, :, ci] / 64
    errors["coefficients"] = float(np.max(np.abs(coef - saved_coeff)))
    for row in read_table("coefficients.csv"):
        key = (
            frames.index(row["frame"]),
            int(row["order"]),
            targets.index(row["family"]),
            names.index(row["component"]),
            fields.index(row["field"]),
            targets.index(row["readout"]),
        )
        errors["tables"] = max(errors["tables"], abs(float(row["value"]) - coef[key]))
        checks += 1
    plus = np.array(
        [(1 + spec["epsilon"] * a * y * h) / 64 for x, y, g, h, a, b in points]
    )
    minus = np.array(
        [(1 - spec["epsilon"] * a * y * h) / 64 for x, y, g, h, a, b in points]
    )
    for axes in ((1, 3, 5), (0, 2, 4), (4,)):
        gap = plus.reshape((2,) * 6).sum(axes) - minus.reshape((2,) * 6).sum(axes)
        errors["laws"] = max(errors["laws"], float(np.max(np.abs(gap))))
        checks += gap.size
    for row in read_table("author_laws.csv"):
        fi, order, ti, field, qi = (
            frames.index(row["frame"]),
            int(row["order"]),
            targets.index(row["family"]),
            fields.index(row["field"]),
            targets.index(row["readout"]),
        )
        contrast = sum(
            (plus[j] - minus[j]) * actual[fi, order, ti, j, field, qi]
            for j in range(64)
        )
        formula = (
            2 * spec["epsilon"] * coef[fi, order, ti, names.index("yha"), field, qi]
        )
        errors["laws"] = max(errors["laws"], abs(contrast - formula))
        for name, value in (("contrast", contrast), ("coefficient_formula", formula)):
            errors["tables"] = max(errors["tables"], abs(float(row[name]) - value))
            checks += 1
    mixed = np.array(
        [
            any(mask[j] for j in (0, 2, 4)) and any(mask[j] for j in (1, 3, 5))
            for mask in masks
        ]
    )
    for row in read_table("composition.csv"):
        fi, order, ti, field = (
            frames.index(row["frame"]),
            int(row["order"]),
            targets.index(row["family"]),
            fields.index(row["field"]),
        )
        c = coef[fi, order, ti, :, field]
        f = actual[fi, order, ti, :, field]
        one = f.reshape((2,) * 6 + (6,)).mean((1, 3, 5))
        two = f.reshape((2,) * 6 + (6,)).mean((0, 2, 4))
        remainder = np.stack(
            [
                f[j]
                - one[(int(x > 0), int(g > 0), int(a > 0))]
                - two[(int(y > 0), int(h > 0), int(b > 0))]
                + f.mean(0)
                for j, (x, y, g, h, a, b) in enumerate(points)
            ]
        )
        total, cross = float(np.sum(c[1:] ** 2)), float(np.sum(c[mixed] ** 2))
        rms = float(np.sqrt(np.mean(remainder**2)))
        expected = {
            "nonconstant_energy": total,
            "cross_event_energy": cross,
            "cross_event_fraction": cross / total if total else 0.0,
            "cross_event_rms": rms,
            "parseval_error": abs(rms * rms - cross / 6),
        }
        errors["parseval"] = max(errors["parseval"], expected["parseval_error"])
        for name, value in expected.items():
            errors["tables"] = max(errors["tables"], abs(float(row[name]) - value))
            checks += 1
    for row in read_table("reader_accounts.csv"):
        fi, order, field = (
            frames.index(row["frame"]),
            int(row["order"]),
            fields.index(row["field"]),
        )
        matrix = 2 * spec["epsilon"] * coef[fi, order, :, names.index("yha"), field]
        common = matrix.mean(1)[:, None] * np.ones((1, 6))
        total, shared, specific = (
            float(np.sum(v**2)) for v in (matrix, common, matrix - common)
        )
        expected = {
            "full_energy": total,
            "battery_common_energy": shared,
            "reader_specific_energy": specific,
            "identity_error": abs(total - shared - specific),
        }
        errors["reader"] = max(errors["reader"], expected["identity_error"])
        for name, value in expected.items():
            errors["tables"] = max(errors["tables"], abs(float(row[name]) - value))
            checks += 1
    for row in read_table("past_future_controls.csv"):
        fi, order, ti, event = (
            frames.index(row["frame"]),
            int(row["order"]),
            targets.index(row["family"]),
            int(row["event"]) - 1,
        )
        axis = 4 if event == 1 else 5
        vector, readings = [], []
        for j, point in enumerate(points):
            if point[axis] > 0:
                other = list(point)
                other[axis] *= -1
                k = points.index(tuple(other))
                vector.append(
                    raw_means[fi, order, ti, j, event]
                    - raw_means[fi, order, ti, k, event]
                )
                readings.append(
                    actual[fi, order, ti, j, 2 + 3 * event : 5 + 3 * event]
                    - actual[fi, order, ti, k, 2 + 3 * event : 5 + 3 * event]
                )
        vector, readings = np.array(vector), np.array(readings)
        expected = {
            "value_max_abs": float(np.abs(vector).max()),
            "value_rms": float(np.sqrt(np.mean(vector**2))),
            "local_cold_rms": float(np.sqrt(np.mean(readings[:, 0] ** 2))),
            "local_paired_rms": float(np.sqrt(np.mean(readings[:, 1] ** 2))),
            "uniform_value_rms": float(np.sqrt(np.mean(readings[:, 2] ** 2))),
        }
        for name, value in expected.items():
            errors["tables"] = max(errors["tables"], abs(float(row[name]) - value))
            checks += 1
    for row in read_table("K2_scope_product_accounts.csv"):
        for name, value in products[int(row["unit"]), row["readout"]].items():
            errors["tables"] = max(errors["tables"], abs(float(row[name]) - value))
            checks += 1
    summary = load(OUT / "02/summary.json")
    status = (
        "VERIFIED_FINITE_HM92_NOT_GLOBAL"
        if max(errors.values()) < 1e-10 and float32_mean_roundoff < 1e-5
        else "FAILED_NUMERICAL_REVIEW"
    )
    write(
        OUT / "verification.json",
        {
            "status": status,
            "errors": errors,
            "checks": checks,
            "float32_mean_vs_float64_roundoff": float32_mean_roundoff,
            "arithmetic_atol": 1e-10,
            "mean_roundoff_atol": 1e-5,
            "future_prefix_control": summary["future_control"],
            "new_model_calls": 0,
            "same_agent_alternate_formula_not_third_party": True,
            "code_sha256": sha(Path(__file__)),
            "original_scope_complete": False,
        },
    )
    print(status, errors, checks, "mean roundoff", float32_mean_roundoff, flush=True)
    if status.startswith("FAILED"):
        raise ValueError("Preserve failed verification; do not relax tolerance")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
