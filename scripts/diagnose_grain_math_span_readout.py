#!/usr/bin/env python3
"""Separate pre-link action from categorical-link curvature without forwards."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_native_joint import interaction
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_response import write_csv

DEFAULT = ROOT / "results/grain_math_response_v1/span_binding_r1"
OUTPUT = ROOT / "results/grain_math_response_v1/span_binding_readout_diagnostic_r1"


def _probability(logits):
    """Evaluate the categorical distribution independently of the primary helper."""
    z = np.asarray(logits, float)
    if z.ndim < 1 or z.shape[-1] < 2 or not np.isfinite(z).all():
        raise ValueError("finite categorical logits required")
    weights = np.exp(z - z.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def gradient_integral(origin, displacement, values, order=32):
    """Integrate the score's exact gradient on a declared natural-coordinate path."""
    z, d, a = (np.asarray(x, float) for x in (origin, displacement, values))
    if z.ndim != 1 or d.shape != z.shape or a.shape != z.shape or order < 2:
        raise ValueError("matched logit path and finite answer values required")
    if not all(np.isfinite(x).all() for x in (z, d, a)):
        raise ValueError("finite logit path and answer values required")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    probabilities = _probability(z + (nodes[:, None] + 1) * d / 2)
    centered_values = a - (probabilities @ a)[:, None]
    return float(weights @ ((probabilities * centered_values) @ d) / 2)


def split_four_logits(logits, values):
    """Split an activation rectangle at its declared additive-logit reference.

    Rows are baseline, projection, complement, full. This is a signed,
    reference-dependent identity, not a unique allocation of causal mediation.
    Centering removes only the exact scalar gauge of categorical logits.
    """
    raw, a = np.asarray(logits, float), np.asarray(values, float)
    if raw.ndim != 2 or len(raw) != 4 or a.shape != (raw.shape[-1],):
        raise ValueError("complete four-corner logit table and answer values required")
    if not np.isfinite(raw).all() or not np.isfinite(a).all():
        raise ValueError("finite four-corner table required")
    z = raw - raw.mean(axis=1, keepdims=True)
    additive = z[1] + z[2] - z[0]
    p = _probability(z)
    reference_p = _probability(additive)
    means, reference_mean = p @ a, float(reference_p @ a)
    pre_link = float(means[3] - reference_mean)
    link = float(reference_mean - means[1] - means[2] + means[0])
    total = float(means[3] - means[1] - means[2] + means[0])
    displacement = z[3] - additive
    kl = float(p[3] @ (np.log(p[3]) - np.log(reference_p)))
    # Compare with the same KL in log-partition coordinates, not another score.
    partition = lambda v: float(np.max(v) + np.log(np.exp(v - np.max(v)).sum()))
    bregman = partition(additive) - partition(z[3]) + float(p[3] @ displacement)
    pre_integral = gradient_integral(additive, displacement, a)
    link_integral = gradient_integral(z[2], z[1] - z[0], a) - gradient_integral(
        z[0], z[1] - z[0], a
    )
    alternate = interaction(raw, a)
    return {
        "mixed_score": total,
        "pre_link_action": pre_link,
        "link_curvature": link,
        "pre_link_norm": float(np.linalg.norm(displacement)),
        "pre_link_kl": kl,
        "representation_integral_error": abs(pre_integral - pre_link),
        "link_integral_error": abs(link_integral - link),
        "kl_bregman_error": abs(kl - bregman),
        "decomposition_error": abs(total - pre_link - link),
        "alternate_formula_error": max(
            abs(total - alternate["total"]),
            abs(link - alternate["link_only"]),
            abs(pre_link - alternate["representation_action"]),
        ),
    }


def _summarize(rows, fields):
    """Keep both orders, every layer and disjoint roles; never select a winner."""
    groups = defaultdict(list)
    for row in rows:
        groups[
            row["role"], row["axis"], row["order"], row["layer"], row["query"]
        ].append(row)
    output = []
    for key, members in sorted(groups.items()):
        output.append(
            {
                "role": key[0],
                "axis": key[1],
                "order": key[2],
                "layer": key[3],
                "query": key[4],
                "families": len({r["family"] for r in members}),
                "observations": len(members),
                **{
                    f"mean_{field}": float(np.mean([r[field] for r in members]))
                    for field in fields
                },
            }
        )
    return output


def analyze(source, out):
    """Read saved numeric observations and preserve a new immutable diagnostic."""
    if out.exists():
        raise FileExistsError("preserve prior diagnostic; choose a new output")
    manifest = frozen_files(source)
    values = manifest["alphabet"]["values"]
    spec = manifest["spec"]
    designs = read_json(source / "00/design.json")
    patches = read_json(source / "00/patch_design.json")
    captures, hashes, baseline_error = {}, {}, 0.0
    for row in designs:
        path = source / "01/units" / f"unit_{row['unit']:04d}.json"
        saved = read_json(path)
        if saved["address"] != row:
            raise ValueError("capture address changed")
        logits = np.asarray(saved["native"]["candidate_logits"], float)
        score = float(_probability(logits) @ values)
        baseline_error = max(
            baseline_error, abs(score - saved["native"]["conditional_score"])
        )
        captures[row["doc_id"], row["query"]] = logits
        hashes[str(path.relative_to(source))] = digest(path)
    cells, visited, patch_error = defaultdict(dict), [], 0.0
    for row in patches:
        path = source / "03/units" / f"unit_{row['unit']:05d}.json"
        if not path.exists():
            continue
        saved = read_json(path)
        if saved["address"] != row:
            raise ValueError("patch address changed")
        hashes[str(path.relative_to(source))] = digest(path)
        visited.append(row["unit"])
        logits = np.asarray(saved["native"]["candidate_logits"], float)
        baseline = captures[row["receiver_doc"], row["query"]]
        expected = float((_probability(logits) - _probability(baseline)) @ values)
        patch_error = max(patch_error, abs(expected - saved["patch_delta"]))
        if row["arm"] not in {"full_property", "projection", "complement"}:
            continue
        key = (
            row["family"],
            row["axis"],
            row["polarity"],
            row["order"],
            row["layer"],
            row["query"],
        )
        group = cells[key]
        if row["arm"] in group:
            raise ValueError("duplicate actual intervention")
        group[row["arm"]] = {"row": row, "logits": logits}
    split_rows = []
    for key, arms in sorted(cells.items()):
        if set(arms) != {"full_property", "projection", "complement"}:
            continue
        row = arms["full_property"]["row"]
        if any(a["row"]["receiver_doc"] != row["receiver_doc"] for a in arms.values()):
            raise ValueError("four corners must share the exact receiver")
        table = np.stack(
            [
                captures[row["receiver_doc"], row["query"]],
                arms["projection"]["logits"],
                arms["complement"]["logits"],
                arms["full_property"]["logits"],
            ]
        )
        split_rows.append(
            {
                "family": key[0],
                "role": row["role"],
                "axis": key[1],
                "polarity": key[2],
                "order": key[3],
                "layer": key[4],
                "query": key[5],
                **split_four_logits(table, values),
            }
        )
    polarities = defaultdict(dict)
    for row in split_rows:
        key = row["family"], row["axis"], row["order"], row["layer"], row["query"]
        polarities[key][row["polarity"]] = row
    binding_rows = []
    fields = ("mixed_score", "pre_link_action", "link_curvature")
    for key, rows in sorted(polarities.items()):
        if set(rows) != {0, 1}:
            continue
        high, low = rows[1], rows[0]
        binding_rows.append(
            {
                "family": key[0],
                "role": high["role"],
                "axis": key[1],
                "order": key[2],
                "layer": key[3],
                "query": key[4],
                **{f"{field}_binding": high[field] - low[field] for field in fields},
            }
        )
    errors = {
        "capture_score_max_error": baseline_error,
        "saved_patch_max_error": patch_error,
        **{
            f"max_{name}": max((r[name] for r in split_rows), default=0.0)
            for name in (
                "representation_integral_error",
                "link_integral_error",
                "kl_bregman_error",
                "decomposition_error",
                "alternate_formula_error",
            )
        },
    }
    if max(errors.values()) > 1e-10 or not split_rows:
        raise ValueError(f"numeric readout checks failed: {errors}")
    role_totals = {
        role: {
            "rectangles": len(members),
            "mean_abs_pre_link_action": float(
                np.mean([abs(r["pre_link_action"]) for r in members])
            ),
            "mean_abs_link_curvature": float(
                np.mean([abs(r["link_curvature"]) for r in members])
            ),
            "mean_abs_mixed_score": float(
                np.mean([abs(r["mixed_score"]) for r in members])
            ),
            "mean_pre_link_norm": float(np.mean([r["pre_link_norm"] for r in members])),
            "nonzero_pre_link_action_at_1e_6": sum(
                abs(r["pre_link_action"]) > 1e-6 for r in members
            ),
            "opposite_signed_terms_at_1e_6": sum(
                r["pre_link_action"] * r["link_curvature"] < 0
                and min(abs(r["pre_link_action"]), abs(r["link_curvature"])) > 1e-6
                for r in members
            ),
        }
        for role in ("source", "validation", "application")
        if (members := [r for r in split_rows if r["role"] == role])
    }
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "readout_split_by_rectangle.csv", split_rows)
    write_csv(
        out / "readout_split_summary.csv",
        _summarize(split_rows, fields + ("pre_link_norm",)),
    )
    write_csv(out / "binding_split_by_family.csv", binding_rows)
    write_csv(
        out / "binding_split_summary.csv",
        _summarize(binding_rows, tuple(f"{f}_binding" for f in fields)),
    )
    hashes["00/manifest.json"] = digest(source / "00/manifest.json")
    write_json(out / "checked_source_hashes.json", hashes)
    result = {
        "status": "COMPLETE_CHECKED"
        if len(visited) == len(patches)
        else "PARTIAL_CHECKED_NO_PROMOTION",
        "source": str(source),
        "new_model_calls": 0,
        "saved_captures_checked": len(designs),
        "saved_interventions_checked": len(visited),
        "expected_interventions": len(patches),
        "complete_rectangles": len(split_rows),
        "complete_binding_contrasts": len(binding_rows),
        "layers": spec["layers"],
        "queries": spec["queries"],
        "roles": role_totals,
        "numeric_errors": errors,
        "diagnostic_code_sha256": digest(Path(__file__)),
        "result_after_diagnostic_not_preregistered_primary": True,
        "same_agent_alternative_formula_not_independent_researcher": True,
        "reference": "z_add = z_projection + z_complement - z_receiver",
        "interpretation": "signed conditional action, not causal mediation or psychological information fractions",
    }
    write_json(out / "verification.json", result)
    print(result, flush=True)


def main():
    """Never collect observations or overwrite an existing diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    analyze(args.source, args.out)


if __name__ == "__main__":
    main()
