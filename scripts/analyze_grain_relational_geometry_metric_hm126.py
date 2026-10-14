#!/usr/bin/env python3
"""Exploratory fixed seven-case metric decomposition; parent results retained."""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_relational_geometry_metric as M
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_relational_dual_frame_hm125 import budget, save
from scripts.run_grain_relational_isometry_hm126 import check


def main(out):
    """Decompose full geometry, preserve all components, and directly check gains."""
    folder = out / "metric_supplement_r1"
    if folder.exists():
        raise FileExistsError("Preserve prior exploratory outcomes")
    if H.load(out / "status.json")["state"] != "VERIFIED":
        raise RuntimeError("Primary packet must be verified first")
    check(out)
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    H.verify_hashes(out, H.load(out / "audit_seal.json"))
    folder.mkdir()
    code = (
        "docs/GRAIN_RELATIONAL_GEOMETRY_SUPPLEMENT_HM126.md",
        "grain_research/construction/math_relational_geometry_metric.py",
        "tests/test_grain_relational_geometry_metric.py",
        "scripts/analyze_grain_relational_geometry_metric_hm126.py",
    )
    sources = H.load(out / "output_seal.json")
    H.write(
        folder / "manifest.json",
        {
            "code": {n: H.sha(ROOT / n) for n in code},
            "source": sources,
            "known_primary_results_before_design": True,
            "new_model_calls": 0,
            "all_components_retained": True,
        },
    )
    with budget(out, "metric_supplement"):
        rows, arrays, maximum = {}, {}, 0.0
        for case in H.load(out / "00/cases.json"):
            key = case["key"]
            cells = [tuple(c) for c in case["cells"]]
            with np.load(out / f"01/{key}.npz", allow_pickle=False) as z:
                data = dict(z)
            for phase in ("calibration", "held"):
                gx, gy = data[phase + "_source_gram"], data[phase + "_target_gram"]
                bx, sx = M.decompose_gram(gx)
                by, sy = M.decompose_gram(gy)
                metric = M.metric_deformation(gx, gy)
                t = metric["coefficient_directions"]
                ratios = np.diag(t.T @ gy @ t) / np.diag(t.T @ gx @ t)
                error = max(
                    float(np.max(abs(sum(bx.values()) - gx))),
                    float(np.max(abs(sum(by.values()) - gy))),
                    float(np.max(abs(ratios - metric["directional_energy_gains"]))),
                )
                maximum = max(maximum, error)
                specific_x, specific_y = bx["specific"], by["specific"]
                denominator = np.linalg.norm(specific_x) * np.linalg.norm(specific_y)
                alignment = float(np.sum(specific_x * specific_y) / denominator)
                null = []
                for permutation in itertools.permutations(range(6)):
                    ids = [cells.index((permutation[f], c)) for f, c in cells]
                    g = specific_y[np.ix_(ids, ids)]
                    null.append(float(np.sum(specific_x * g) / denominator))
                record = key + "__" + phase
                rows[record] = {
                    "source_common_trace_share": sx,
                    "target_common_trace_share": sy,
                    "specific_gram_alignment": alignment,
                    "specific_correct_family_rank": 1
                    + int(np.sum(np.array(null) > alignment + 1e-12)),
                    "directional_energy_gains": metric[
                        "directional_energy_gains"
                    ].tolist(),
                    "numeric_max_error": error,
                }
                arrays.update({record + "_" + n: v for n, v in metric.items()})
                arrays.update({record + "_source_" + n: v for n, v in bx.items()})
                arrays.update({record + "_target_" + n: v for n, v in by.items()})
                arrays[record + "_specific_permutation_alignments"] = np.array(null)
        if maximum > 1e-9:
            raise ValueError("Metric block/direct-ratio account failed")
        save(folder / "account.npz", **arrays)
        H.write(
            folder / "summary.json",
            {
                "status": "EXPLORATORY_ALL_COMPONENTS_ACCOUNTED_NOT_NEW_TRAITS",
                "cases": rows,
                "alternate_max_error": maximum,
                "all_modes_retained_no_scoring_metric_selected": True,
                "second_formula_same_agent_not_blind_replication": True,
            },
        )
        H.write(folder / "seal.json", H.seals(out, folder))
    print("HM126 metric supplement accounted all14geometry cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/relational_isometry_r1",
    )
    args = parser.parse_args()
    main(args.out.resolve())
