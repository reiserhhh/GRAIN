#!/usr/bin/env python3
"""Alternate arithmetic for every HM91 reading; no new neural/model requests."""

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

OUT = ROOT / "results/grain_math_response_v1/relation_scope_r1"


def reading(raw, row, bundle):
    """Use strict-past exponentials directly, not HM91's normalized graph code."""
    ix = row["body_indices"]
    keys, values = raw["k"][ix].astype(float), raw["v"][ix].astype(float)
    n, heads = raw["q_body"].shape[:2]
    labels = np.asarray(row["root_labels"])
    gate = np.zeros((heads, n))
    for head in range(heads):
        scores = raw["q_body"][:, head] @ keys[:, head // (heads // keys.shape[1])].T
        allowed = np.arange(n)[None] < np.arange(1, n)[:, None]
        log = np.where(allowed, scores[1:], -np.inf)
        prob = np.exp(log - log.max(1)[:, None])
        prob /= prob.sum(1)[:, None]
        gate[head, 1:] = prob[:, labels == 0].sum(1)
        gate[head, labels >= 0] = (labels[labels >= 0] == 0).astype(float)
    core = [i for group in row["action_groups"] for i in group]
    results, factors = np.zeros((6, 6)), []
    for qi in range(6):
        query = C.rotate(
            bundle["raw_queries"][qi],
            ix[-1] + bundle["gaps"][qi],
            bundle["frequencies"],
            bundle["scaling"],
        )
        ps, cs, nums, masses = [], [], [], []
        for head in range(heads):
            kh = head // (heads // keys.shape[1])
            log = keys[:, kh] @ query[head]
            p = np.exp(log - log.max())
            p /= p.sum()
            c = values[:, kh] @ np.asarray(bundle["directions"])[qi, head]
            ps.append(p)
            cs.append(c)
            nums.append(np.sum(p * gate[head] * c))
            masses.append(np.sum(p * gate[head]))
        p, c, masses = np.array(ps), np.array(cs), np.array(masses)
        results[:, qi] = [
            np.sum(p * c),
            sum(nums),
            sum(np.array(nums) / masses),
            masses.mean(),
            np.sum(p[:, core] * c[:, core]),
            np.sum(p[:, core] * gate[:, core] * c[:, core]),
        ]
        factors.append((p[:, core], gate[:, core], c[:, core]))
    return results, factors


def read_table(name):
    """Inspect every serialized number in a named sealed table."""
    with (OUT / "02" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    """Recompute fields, complete contrasts, geometry and same-marginal laws."""
    manifest = load(OUT / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(OUT, manifest["inputs"])
    verify(OUT, load(OUT / "01/raw_seal.json"))
    verify(OUT, load(OUT / "02/output_seal.json"))
    spec, rows = manifest["spec"], load(OUT / "00/rows.json")
    frames, modes, targets = spec["frames"], spec["modes"], spec["targets"]
    field_names = manifest["fields"]
    corners = list(itertools.product((-1, 1), repeat=4))
    masks = list(itertools.product((0, 1), repeat=4))
    component_names = [
        "".join(c for c, bit in zip("xygh", mask, strict=True) if bit) or "1"
        for mask in masks
    ]
    with np.load(OUT / "02/observations.npz", allow_pickle=False) as data:
        saved_fields, saved_coeff = data["fields"], data["coefficients"]
    actual = np.zeros_like(saved_fields)
    bundle = load(OUT / "00/bundle.json")
    errors, checks = (
        {
            "fields": 0.0,
            "coefficients": 0.0,
            "geometry": 0.0,
            "products": 0.0,
            "author": 0.0,
            "scope": 0.0,
            "table": 0.0,
        },
        0,
    )
    references, expected_products = {}, {}
    for row in rows:
        key = (
            frames.index(row["frame"]),
            modes.index(row["mode"]),
            targets.index(row["target"]),
            corners.index(tuple(row["coordinates"])),
        )
        result, factor = reading(unit(OUT, row["unit"]), row, bundle)
        actual[key] = result
        errors["fields"] = max(
            errors["fields"], float(np.max(np.abs(result - saved_fields[key])))
        )
        checks += result.size
        refkey = key[0], key[2], key[3]
        if key[1] == 0:
            references[refkey] = factor
        else:
            for qi in range(6):
                old, new = references[refkey][qi], factor[qi]
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
                expected_products[row["unit"], targets[qi]] = terms
                errors["products"] = max(errors["products"], terms["identity_error"])
    coeff = np.zeros_like(saved_coeff)
    for ki, mask in enumerate(masks):
        for ci, point in enumerate(corners):
            sign = np.prod([point[j] for j in range(4) if mask[j]])
            coeff[:, :, :, ki] += sign * actual[:, :, :, ci] / 16
    errors["coefficients"] = float(np.max(np.abs(coeff - saved_coeff)))
    for row in read_table("coefficients.csv"):
        key = (
            frames.index(row["frame"]),
            modes.index(row["mode"]),
            targets.index(row["target"]),
            component_names.index(row["component"]),
            field_names.index(row["field"]),
            targets.index(row["readout"]),
        )
        errors["table"] = max(errors["table"], abs(float(row["value"]) - coeff[key]))
        checks += 1
    for row in read_table("scalar_action.csv"):
        fi, mi, ai = (
            frames.index(row["frame"]),
            modes.index(row["mode"]),
            field_names.index(row["field"]),
        )
        u, v = coeff[fi, 0, :, 1:, ai].ravel(), coeff[fi, mi, :, 1:, ai].ravel()
        uu, vv, uv = float(u @ u), float(v @ v), float(u @ v)
        exterior = np.triu(np.outer(u, v) - np.outer(v, u), 1)
        wedge = float(np.sum(exterior**2))
        expected = {
            "reference_energy": uu,
            "response_energy": vv,
            "multiplier": uv / uu,
            "cosine": uv / np.sqrt(uu * vv),
            "non_scalar_energy_fraction": wedge / (uu * vv),
            "wedge_squared": wedge,
        }
        for name, value in expected.items():
            errors["geometry"] = max(errors["geometry"], abs(float(row[name]) - value))
            checks += 1
    for row in read_table("core_product_accounts.csv"):
        for name, value in expected_products[int(row["unit"]), row["readout"]].items():
            errors["table"] = max(errors["table"], abs(float(row[name]) - value))
            checks += 1
    law = np.zeros((2, 6, 16))
    for mi, ci in itertools.product(range(6), range(16)):
        x, y, g, h = corners[ci]
        for li, sign in enumerate((1, -1)):
            law[li, mi, ci] = (
                1
                + sign
                * spec["epsilon"]
                * spec["relation_weights"][mi]
                * (x * g + y * h)
                / 2
            ) / 96
    for row in read_table("author_laws.csv"):
        fi, ti, ai, qi = (
            frames.index(row["frame"]),
            targets.index(row["target"]),
            field_names.index(row["field"]),
            targets.index(row["readout"]),
        )
        surface = actual[fi, :, ti, :, ai, qi]
        contrast = sum(
            (law[0, mi, ci] - law[1, mi, ci]) * surface[mi, ci]
            for mi, ci in itertools.product(range(6), range(16))
        )
        errors["author"] = max(
            errors["author"],
            abs(float(row["joint_law_contrast"]) - contrast),
            abs(float(row["scope_formula"]) - contrast),
            abs(float(row["mode_blind_contrast"])),
        )
        checks += 3
    for row in read_table("scope_contrasts.csv"):
        fi, ti, ai, qi = (
            frames.index(row["frame"]),
            targets.index(row["target"]),
            field_names.index(row["field"]),
            targets.index(row["readout"]),
        )
        delta = actual[fi, 1, ti, :, ai, qi] - actual[fi, 2, ti, :, ai, qi]
        expected = {
            "inner_minus_outer_mean": float(np.mean(delta)),
            "scope_rms": float(np.sqrt(np.mean(delta**2))),
            "binding_scope_difference": float(
                sum(
                    (x * g + y * h) * delta[ci]
                    for ci, (x, y, g, h) in enumerate(corners)
                )
                / 32
            ),
        }
        for name, value in expected.items():
            errors["scope"] = max(errors["scope"], abs(float(row[name]) - value))
            checks += 1
    status = (
        "VERIFIED_FINITE_HM91_NOT_GLOBAL"
        if max(errors.values()) < 1e-10
        else "FAILED_NUMERICAL_REVIEW"
    )
    write(
        OUT / "verification.json",
        {
            "status": status,
            "errors": errors,
            "checks": checks,
            "new_model_calls": 0,
            "same_agent_alternate_formula_not_third_party": True,
            "code_sha256": sha(Path(__file__)),
            "original_scope_complete": False,
        },
    )
    print(status, errors, checks, flush=True)
    if status.startswith("FAILED"):
        raise ValueError("Preserve failed numerical review; do not change tolerance")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
