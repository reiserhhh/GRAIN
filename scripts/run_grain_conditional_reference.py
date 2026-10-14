#!/usr/bin/env python3
"""Build fixed condition-response references without questionnaire or neural fits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.conditional_reference import (
    completion_envelope,
    contrasts,
    fit_reference,
    score_reference,
)
from grain_research.local_bilinear import keyed_directions
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_local_construction import load_texts

PRIOR = BASE / "all_plan_closure_v1/local_bilinear_r2"
PLAN = ROOT / "docs/GRAIN_CONDITIONAL_REFERENCE_PLAN_V1.md"
COMPONENTS = (
    "actual",
    "context",
    "departure",
    "unseen",
    "center",
    "reference_target_mean",
)


def anchors():
    """Recover the exact fixed 300-item encoder cache and keyed 30 directions."""
    vectors, stop, inputs = [], 0, {}
    for path in sorted((DEFAULT / "sentence_batches").glob("batch_*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["start"] != stop or meta["sha256"] != digest(path):
            raise ValueError("Changed or gapped anchor cache")
        with np.load(path, allow_pickle=False) as z:
            vector = z["vectors"]
            if len(vector) != meta["stop"] - meta["start"]:
                raise ValueError("Wrong anchor batch shape")
            vectors.append(vector)
        inputs[str(path)] = digest(path)
        stop = meta["stop"]
        if stop >= 300:
            break
    items_path = DEFAULT / "ipip_neo300.json"
    inputs[str(items_path)] = digest(items_path)
    items = json.loads(items_path.read_text())["items"]
    d, keys = keyed_directions(np.concatenate(vectors)[:300], items)
    return d, keys, inputs


def export_reference(reference):
    """Produce a source-frozen, ordinary JSON reference with no class loading."""
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in reference.items()
    }


def quantiles(x):
    """Summarize a defined support quantity without outcome-based filtering."""
    return {
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "max": float(np.max(x)),
    }


def run(out):
    """Build source references and apply them to both full and held author panels."""
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    data = load_texts()
    raw = json.loads((DEFAULT / "private_material.json").read_text())
    direction, facets, anchor_inputs = anchors()
    targets, target_names = contrasts(facets)
    np.savez_compressed(
        out / "private_directions.npz", direction=direction, targets=targets
    )
    write(
        out / "contract.json",
        {
            "protocol": "GRAIN_CONDITIONAL_REFERENCE_V1",
            "plan_sha256": digest(PLAN),
            "runner_sha256": digest(Path(__file__)),
            "module_sha256": digest(ROOT / "grain_research/conditional_reference.py"),
            "material_manifest_sha256": digest(DEFAULT / "material_manifest.json"),
            "anchor_inputs": anchor_inputs,
            "facets": facets,
            "contrast_names": target_names,
            "references": ["full_source_distribution", "top5_source_author_coverage"],
            "criterion_columns_accessed": False,
            "neural_calls": 0,
            "bounds_scope": "empirical observed-cell means fixed; unit-ball missing-cell completions, not sampling CIs",
        },
    )
    unit_radius = np.linalg.norm(direction @ targets, axis=0)
    unit_box = np.linalg.norm(direction, axis=0) @ abs(targets)
    geometry = pd.DataFrame(
        {
            "contrast": target_names,
            "joint_halfwidth_per_missing_mass": unit_radius,
            "box_halfwidth_per_missing_mass": unit_box,
            "joint_over_box": unit_radius / unit_box,
        }
    )
    geometry.to_csv(out / "contrast_geometry.csv", index=False)
    all_summary = []
    for domain in ("pandora", "essays"):
        root, old = out / domain, PRIOR / domain
        root.mkdir()
        metadata = json.loads((old / "material.json").read_text())
        for name, sha in metadata["files"].items():
            if digest(old / name) != sha:
                raise ValueError("Changed original units")
        names = json.loads((old / "private_authors.json").read_text())
        if names["ids"] != data[domain]["ids"] or metadata["facet_names"] != facets:
            raise ValueError("Author or evidence ordering mismatch")
        with np.load(old / "private_units.npz", allow_pickle=False) as z:
            h, author = np.asarray(z["h"], float), z["authors"]
            norms = np.linalg.norm(h, axis=1)
            if np.any(norms < 1e-12):
                raise ValueError("Zero semantic vector")
            h = h / norms[:, None]
            scores = h @ direction
            inherited_score_error = float(abs(scores - z["semantic"]).max())
            # The prior cache normalized float32 vectors; this scorer freezes float64.
            np.testing.assert_allclose(scores, z["semantic"], rtol=0, atol=1e-7)
        addresses = pd.read_csv(old / "private_addresses.csv")
        np.testing.assert_array_equal(addresses.author_index, author)
        if domain == "pandora":
            conditions = np.array(
                [c["subreddit"] for b in raw["bundles"] for v in b["views"] for c in v],
                str,
            )
        else:
            conditions = np.repeat("essay_protocol", len(author))
        if len(conditions) != len(author):
            raise ValueError("Unit condition mismatch")
        addresses["condition"] = conditions
        addresses.to_csv(root / "private_addresses.csv", index=False)
        source, held = data[domain]["train"], data[domain]["held"]
        ref = fit_reference(conditions, scores, author, source)
        write(root / "private_reference.json", export_reference(ref))
        write(root / "private_authors.json", names)
        np.savez_compressed(
            root / "private_units.npz",
            scores=scores,
            authors=author,
            conditions=conditions,
            source=source,
            held=held,
        )
        pd.DataFrame(
            {
                "condition": ref["keys"],
                "source_authors": ref["source_authors_per_cell"],
                "full_weight": ref["weights"],
                "common5_weight": ref["common5_weights"],
            }
        ).to_csv(root / "reference_conditions.csv", index=False)
        summaries, counts, reveals, budgets, selection_rows = [], [], [], [], []
        for target in ("full", "common5"):
            accum = {
                k: []
                for k in (
                    *COMPONENTS,
                    "observed",
                    "lower",
                    "upper",
                    "coverage",
                    "missing",
                    "unknown_material_mass",
                    "complete",
                )
            }
            nu = ref["weights" if target == "full" else "common5_weights"]
            for u in range(len(names["ids"])):
                ix = np.flatnonzero(author == u)
                c, x = conditions[ix], scores[ix]
                s = score_reference(c, x, ref, target)
                np.testing.assert_allclose(
                    s["actual"], s["context"] + s["departure"] + s["unseen"], atol=1e-12
                )
                lo, hi, _, _ = completion_envelope(
                    s["center"], s["missing"], direction, targets
                )
                for key in COMPONENTS:
                    accum[key].append(s[key])
                for key in (
                    "observed",
                    "coverage",
                    "missing",
                    "unknown_material_mass",
                    "complete",
                ):
                    accum[key].append(s[key])
                accum["lower"].append(lo)
                accum["upper"].append(hi)
                counts.append(
                    {
                        "author_index": u,
                        "role": "held" if u in held else "source",
                        "target": target,
                        "material_units": len(ix),
                        "observed_conditions": len(s["selection"]),
                        "covered_target_cells": int(np.sum(s["observed"] & (nu > 0))),
                        "coverage": s["coverage"],
                        "missing": s["missing"],
                        "unknown_material_mass": s["unknown_material_mass"],
                        "complete": s["complete"],
                    }
                )
                if target == "full":
                    selection_rows.extend(
                        {"author_index": u, **row} for row in s["selection"]
                    )
                if u in held:
                    ranked = s["missing_cells_ranked"]
                    for budget in (0, 1, 2, 5):
                        recovered = float(nu[ranked[:budget]].sum())
                        budgets.append(
                            {
                                "author_index": u,
                                "target": target,
                                "extra_complete_cells": budget,
                                "coverage_if_empirical_cells_observed": min(
                                    1.0, s["coverage"] + recovered
                                ),
                                "missing_mass_after": max(
                                    0.0, s["missing"] - recovered
                                ),
                                "scope": "equal cost per full empirical cell; prospective bookkeeping, not new observations",
                            }
                        )
                    choices = [
                        j for j in range(len(nu)) if s["observed"][j] and nu[j] > 0
                    ]
                    if choices:
                        j = min(choices, key=lambda i: (-nu[i], ref["keys"][i]))
                        keep = c != ref["keys"][j]
                        removed_mean = x[~keep].mean(0)
                        hidden_center = s["center"] - nu[j] * removed_mean
                        hidden_missing = s["missing"] + nu[j]
                        hlo, hhi, _, _ = completion_envelope(
                            hidden_center, hidden_missing, direction, targets
                        )
                        np.testing.assert_allclose(
                            (hhi - hlo) - (hi - lo), 2 * nu[j] * unit_radius, atol=1e-12
                        )
                        if np.any(hlo > lo + 1e-12) or np.any(hhi < hi - 1e-12):
                            raise ArithmeticError(
                                "Restored empirical cell left completion envelope"
                            )
                        reveals.append(
                            {
                                "author_index": u,
                                "target": target,
                                "condition": ref["keys"][j],
                                "cell_weight": float(nu[j]),
                                "removed_units": int((~keep).sum()),
                                "max_width_identity_error": float(
                                    abs(
                                        ((hhi - hlo) - (hi - lo))
                                        - 2 * nu[j] * unit_radius
                                    ).max()
                                ),
                                "envelope_nested": True,
                            }
                        )
            arrays = {k: np.asarray(v) for k, v in accum.items()}
            np.savez_compressed(root / f"private_scores_{target}.npz", **arrays)
            # Keep signed covariance terms; centering here summarizes variation only.
            a = arrays["actual"][held]
            b = np.stack(
                [arrays[k][held] for k in ("context", "departure", "unseen")], axis=1
            )
            ac, bc = a - a.mean(0), b - b.mean(0)
            variance_rows = []
            for j, key in enumerate(facets):
                cov = bc[:, :, j].T @ bc[:, :, j] / (len(held) - 1)
                total = float(ac[:, j] @ ac[:, j] / (len(held) - 1))
                np.testing.assert_allclose(cov.sum(), total, atol=1e-13)
                variance_rows.append(
                    {
                        "facet": key,
                        "var_actual": total,
                        "var_context": float(cov[0, 0]),
                        "var_departure": float(cov[1, 1]),
                        "var_unseen": float(cov[2, 2]),
                        "twice_cov_context_departure": float(2 * cov[0, 1]),
                        "twice_cov_context_unseen": float(2 * cov[0, 2]),
                        "twice_cov_departure_unseen": float(2 * cov[1, 2]),
                    }
                )
            if target == "full":
                pd.DataFrame(variance_rows).to_csv(
                    root / "held_variance_accounting.csv", index=False
                )
            summaries.append(
                {
                    "domain": domain,
                    "target": target,
                    "source_authors": len(source),
                    "held_authors": len(held),
                    "reference_cells": int((nu > 0).sum()),
                    "held_complete": int(arrays["complete"][held].sum()),
                    "held_no_target_support": int(
                        (arrays["coverage"][held] == 0).sum()
                    ),
                    "held_coverage": quantiles(arrays["coverage"][held]),
                    "held_unknown_material_mass": quantiles(
                        arrays["unknown_material_mass"][held]
                    ),
                    "held_median_raw_facet_sd": float(np.median(a.std(0, ddof=1))),
                    "reference_covariance_scope": "actual context/departure/new-context covariance, not independent personality information",
                }
            )
        pd.DataFrame(counts).to_csv(root / "private_support.csv", index=False)
        pd.DataFrame(selection_rows).to_csv(root / "private_selection.csv", index=False)
        pd.DataFrame(reveals).to_csv(root / "private_reveal_checks.csv", index=False)
        budget_table = pd.DataFrame(budgets)
        budget_table.to_csv(root / "private_completion_budget.csv", index=False)
        budget_summary = budget_table.groupby(["target", "extra_complete_cells"])[
            ["coverage_if_empirical_cells_observed", "missing_mass_after"]
        ].agg(["mean", "median"])
        budget_summary.columns = ["_".join(key) for key in budget_summary.columns]
        budget_summary.reset_index().to_csv(
            root / "completion_budget_summary.csv", index=False
        )
        write(
            root / "summary.json",
            {
                "rows": summaries,
                "input_material_sha256": digest(old / "material.json"),
                "units": len(author),
                "normalization_dtype": "float64",
                "inherited_float32_score_max_difference": inherited_score_error,
                "reveal_checks": len(reveals),
                "source_reference_rule": "author-balanced choice fractions and author-balanced cell means",
                "files": {
                    p.name: digest(p)
                    for p in root.iterdir()
                    if p.is_file() and not p.name.startswith("._")
                },
            },
        )
        all_summary.extend(summaries)
        print(
            json.dumps(
                {"stage": "domain_complete", "domain": domain, "rows": summaries},
                indent=2,
            ),
            flush=True,
        )
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "rows": all_summary,
            "wall_seconds": time.monotonic() - start,
            "criterion_columns_accessed": False,
            "neural_calls": 0,
            "root_files": {
                p.name: digest(p)
                for p in out.iterdir()
                if p.is_file() and not p.name.startswith("._")
            },
            "domain_summaries": {
                d: digest(out / d / "summary.json") for d in ("pandora", "essays")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "all_plan_closure_v1/conditional_reference_r2",
    )
    with threadpool_limits(limits=2):
        run(parser.parse_args().output)
