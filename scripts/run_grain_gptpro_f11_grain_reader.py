#!/usr/bin/env python3
"""Measure a registered frozen GRAIN comment view on the same F08 text bank.

The representation is the historical TF-IDF/SVD24 comment channel, not the entire
GRAIN measurement system. Two source-only Ridge readouts put its coordinates in
the designed pole/eligibility units. There is no neural or questionnaire fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from scripts.run_grain_gptpro_f11_reader_interface import (
    correlation,
    crossed_decomposition,
)


def fit_readout(features, targets, source):
    """Fit fixed-alpha linear units using only the declared source materials."""
    features, targets = np.asarray(features, float), np.asarray(targets, float)
    source = np.asarray(source, int)
    if len(source) != len(set(source)) or not len(source):
        raise ValueError("unique nonempty source material rows required")
    if not np.isfinite(features).all() or not np.isfinite(targets[source]).all():
        raise ValueError("finite source features and targets required")
    center, scale = features[source].mean(0), features[source].std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = (features[source] - center) / scale
    heads = Ridge(alpha=1.0, fit_intercept=True, solver="svd").fit(x, targets[source])
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coef": heads.coef_.tolist(),
        "intercept": heads.intercept_.tolist(),
        "source_target_mean": targets[source].mean(0).tolist(),
        "alpha": 1.0,
        "unit": ["designed_content_pole", "designed_actual_self_qualification"],
        "output": "clipped calibrated readout, not guaranteed posterior probability",
    }


def predict_readout(features, bundle):
    """Score with explicit coefficients, independent of sklearn object loading."""
    x = (np.asarray(features, float) - bundle["center"]) / bundle["scale"]
    raw = x @ np.asarray(bundle["coef"]).T + bundle["intercept"]
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite readout")
    return raw, np.clip(raw, 0, 1)


def main():
    """Freeze provenance then run all materials and matched-edit comparisons."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    base = ROOT / "results/grain_gptpro_foundation_v3"
    lexical_path = base / "F08/reader_recovery_r1/reader_scores.csv"
    nli_path = base / "F08/nli_reader_r1/reader_scores.csv"
    historical = base / "F11/reader_interface_r1/summary.json"
    runtime_path = (
        ROOT
        / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
    )
    preflight_path = (
        ROOT / "results/grain_vector_mechanisms/vm05_preflight_r1/manifest.json"
    )
    inputs = [lexical_path, nli_path, historical, runtime_path, preflight_path]
    hashes = {str(p): sha256_file(p) for p in inputs}
    old = json.loads(historical.read_text())
    if (
        old["hashes"]["lexical"] != hashes[str(lexical_path)]
        or old["hashes"]["nli"] != hashes[str(nli_path)]
    ):
        raise ValueError("changed original reader materials")
    preflight = json.loads(preflight_path.read_text())
    if preflight["runtime_sha256"] != hashes[str(runtime_path)]:
        raise ValueError("untrusted or changed registered representation")
    contract = {
        "inputs": hashes,
        "runner_sha256": sha256_file(Path(__file__)),
        "representation_code_sha256": sha256_file(
            ROOT / "suica_core/v7_psychometric.py"
        ),
        "representation": "registered historical GRAIN comment TFIDF1-2/SVD24; not full GRAIN",
        "readout": "source-only scaling, Ridge alpha1 SVD solver, clip to[0,1]; no selection grid",
        "scope": "same96TEXT_SYNTHETICmaterials;fixed reader comparison,not author/trait validation",
        "roles": "32source/32validation/32application,core wording shared but frames disjoint",
        "neural_training": False,
        "questionnaire_or_private_human_text": False,
        "parent_promotion": False,
    }
    (a.output / "contract_before_readout.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    lexical, nli = [
        pd.read_csv(path).sort_values("material_id").reset_index(drop=True)
        for path in (lexical_path, nli_path)
    ]
    identity = [
        "material_id",
        "role",
        "family",
        "pole",
        "subject_status",
        "qualification",
        "variant",
        "text",
    ]
    if (
        len(lexical) != 96
        or lexical.material_id.nunique() != 96
        or not lexical[identity].equals(nli[identity])
    ):
        raise ValueError("exact material and role alignment required")
    runtime = joblib.load(runtime_path)
    features = runtime.transform(lexical.text.tolist())
    if features.shape != (96, 24) or not np.isfinite(features).all():
        raise ValueError("registered feature shape changed")
    source = np.flatnonzero(lexical.role.eq("source"))
    targets = lexical[["pole", "qualification"]].to_numpy(float)
    bundle = fit_readout(features, targets, source)
    (a.output / "readout_bundle.json").write_text(
        json.dumps(bundle, indent=2, allow_nan=False) + "\n"
    )
    raw, scores = predict_readout(features, bundle)
    # Alternate normal-equation ridge solution; no sklearn predictor reused.
    x = (features[source] - bundle["center"]) / bundle["scale"]
    xc = x - x.mean(0)
    yc = targets[source] - targets[source].mean(0)
    beta = np.linalg.solve(xc.T @ xc + np.eye(x.shape[1]), xc.T @ yc)
    intercept = targets[source].mean(0) - x.mean(0) @ beta
    independently = ((features - bundle["center"]) / bundle["scale"]) @ beta + intercept
    error = float(np.max(abs(independently - raw)))
    if error > 1e-9:
        raise ValueError("source readout numerical audit failed")
    scored = lexical[identity].copy()
    scored["pole_raw"], scored["qualification_raw"] = raw.T
    scored["pole_score"], scored["qualification_score"] = scores.T
    scored.to_csv(a.output / "reader_scores.csv", index=False)
    np.savez_compressed(
        a.output / "representation.npz", features=features, source=source
    )
    readers = {"lexical": lexical, "nli": nli, "grain_comment_view": scored}
    rows, agreement, edits, decomposition = [], [], [], []
    for role in ("source", "validation", "application"):
        sel = lexical.role.eq(role).to_numpy()
        for target in ("pole", "qualification", "signed_qualified"):
            truth = (
                targets[:, 0]
                if target == "pole"
                else targets[:, 1]
                if target == "qualification"
                else targets[:, 1] * (2 * targets[:, 0] - 1)
            )
            values = {}
            baseline = float(truth[source].mean())
            for name, r in readers.items():
                value = (
                    r.pole_score.to_numpy()
                    if target == "pole"
                    else r.qualification_score.to_numpy()
                    if target == "qualification"
                    else r.qualification_score.to_numpy()
                    * (2 * r.pole_score.to_numpy() - 1)
                )
                values[name] = value[sel]
                rows.append(
                    {
                        "role": role,
                        "reader": name,
                        "target": target,
                        "materials": int(sel.sum()),
                        "mse": float(np.mean((value[sel] - truth[sel]) ** 2)),
                        "r": correlation(value[sel], truth[sel]),
                        "source_constant_mse": float(
                            np.mean((baseline - truth[sel]) ** 2)
                        ),
                    }
                )
                group = lexical.loc[sel].copy()
                group["score"], group["truth"] = value[sel], truth[sel]
                index = group.set_index(["pole", "subject_status", "variant"])
                for pole in (0, 1):
                    for variant in range(4):
                        actual = index.loc[(pole, "actual", variant)]
                        for status in ("quoted", "hypothetical", "negated"):
                            other = index.loc[(pole, status, variant)]
                            edits.append(
                                {
                                    "role": role,
                                    "reader": name,
                                    "target": target,
                                    "pole": pole,
                                    "variant": variant,
                                    "edit": "actual_to_" + status,
                                    "observed_delta": float(other.score - actual.score),
                                    "designed_delta": float(other.truth - actual.truth),
                                }
                            )
            names = list(values)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    va, vb = values[names[i]], values[names[j]]
                    agreement.append(
                        {
                            "role": role,
                            "target": target,
                            "reader_a": names[i],
                            "reader_b": names[j],
                            "r": correlation(va, vb),
                            "error_r": correlation(va - truth[sel], vb - truth[sel]),
                            "rmse": float(np.sqrt(np.mean((va - vb) ** 2))),
                        }
                    )
            terms = crossed_decomposition(np.column_stack(list(values.values())))
            decomposition.append(
                {
                    "role": role,
                    "target": target,
                    "material": terms[0],
                    "fixed_reader_offset": terms[1],
                    "interaction": terms[2],
                    "total": terms.sum(),
                }
            )
    pd.DataFrame(rows).to_csv(a.output / "reader_metrics.csv", index=False)
    pd.DataFrame(agreement).to_csv(a.output / "reader_agreement.csv", index=False)
    pd.DataFrame(edits).to_csv(a.output / "edit_signatures.csv", index=False)
    pd.DataFrame(decomposition).to_csv(
        a.output / "fixed_method_decomposition.csv", index=False
    )
    pair_distances = []
    for ids in lexical.groupby(["role", "subject_status", "variant"]).groups.values():
        if len(ids) != 2:
            raise ValueError("expected opposite-pole text pair")
        i, j = list(ids)
        pair_distances.append(float(np.linalg.norm(features[i] - features[j])))
    summary = {
        "status": "REGISTERED_GRAIN_COMMENT_READER_SAME_MATERIAL_COMPLETE",
        "materials": 96,
        "features": 24,
        "calibration_source_rows": len(source),
        "paired_edit_rows": len(edits),
        "normal_equation_max_error": error,
        "opposite_pole_feature_distance_min_median_max": np.quantile(
            pair_distances, [0, 0.5, 1]
        ).tolist(),
        "identical_opposite_pole_pairs": int(
            np.sum(np.asarray(pair_distances) <= 1e-12)
        ),
        "application_metrics": [r for r in rows if r["role"] == "application"],
        "new_neural_calls": 0,
        "scope": contract["scope"],
        "representation": contract["representation"],
        "F11_parent_complete": False,
        "remaining": [
            "actual text history integration with matching observation law",
            "other original F11 dependencies and uncertainty",
        ],
    }
    (a.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (a.output / "output_manifest.json").write_text(
        json.dumps(
            {p.name: sha256_file(p) for p in sorted(a.output.iterdir()) if p.is_file()},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
