#!/usr/bin/env python3
"""Locate lost material contrasts and test one declared two-channel follow-up.

The cross-reader calculations are post-result diagnostics on the same bank, not
an independent replication or a new head/prompt search. No source head refitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from scripts.run_grain_gptpro_f11_grain_reader import predict_readout


def main():
    """Verify all readings and locate whether collision precedes SVD."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.run / "output_manifest.json").read_text())
    for name, expected in manifest.items():
        if Path(name).name != name or sha256_file(a.run / name) != expected:
            raise ValueError("changed reader output")
    contract = json.loads((a.run / "contract_before_readout.json").read_text())
    for path, expected in contract["inputs"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("changed input")
    paths = [Path(k) for k in contract["inputs"]]
    runtime_path = next(k for k in paths if k.suffix == ".joblib")
    nli_path = next(k for k in paths if k.parent.name == "nli_reader_r1")
    lexical_path = next(k for k in paths if k.parent.name == "reader_recovery_r1")
    runtime = joblib.load(runtime_path)
    f = (
        pd.read_csv(a.run / "reader_scores.csv")
        .sort_values("material_id")
        .reset_index(drop=True)
    )
    nli, lex = [
        pd.read_csv(path).sort_values("material_id").reset_index(drop=True)
        for path in (nli_path, lexical_path)
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
    if not f[identity].equals(nli[identity]) or not f[identity].equals(lex[identity]):
        raise ValueError("follow-up exact material/role mismatch")
    followup = {
        "stage": "post-result fixed diagnostic,not independent confirmation",
        "run_manifest_sha256": sha256_file(a.run / "output_manifest.json"),
        "arms": [
            "nli_pole_nli_qualification",
            "nli_pole_grain_qualification",
            "nli_pole_lexical_qualification",
        ],
        "operation": "q*(2*p-1), with component losses retained separately; not an exact posterior",
        "no_new_fit_or_arm_selection": True,
        "runner_sha256": sha256_file(Path(__file__)),
    }
    (a.output / "followup_before_calculation.json").write_text(
        json.dumps(followup, indent=2) + "\n"
    )
    x = runtime.vectorizer.transform(f.text.tolist())
    actual = runtime.svd.transform(x)
    with np.load(a.run / "representation.npz", allow_pickle=False) as z:
        expected = z["features"]
    error = float(np.max(abs(actual - expected)))
    if error > 1e-12:
        raise ValueError("frozen text representation changed")
    bundle = json.loads((a.run / "readout_bundle.json").read_text())
    raw, clipped = predict_readout(actual, bundle)
    score_error = float(
        np.max(abs(raw - f[["pole_raw", "qualification_raw"]].to_numpy()))
    )
    if score_error > 1e-12 or not np.allclose(
        clipped, f[["pole_score", "qualification_score"]], atol=1e-12, rtol=0
    ):
        raise ValueError("readout score changed")
    analyzer = runtime.vectorizer.build_analyzer()
    rows = []
    differing, retained = set(), set()
    for (role, status, variant), ids in f.groupby(
        ["role", "subject_status", "variant"]
    ).groups.items():
        i, j = list(ids)
        diff = set(analyzer(f.text.iloc[i])) ^ set(analyzer(f.text.iloc[j]))
        differing.update(diff)
        retained.update(diff & set(runtime.vectorizer.vocabulary_))
        rows.append(
            {
                "role": role,
                "status": status,
                "variant": int(variant),
                "material_a": f.material_id.iloc[i],
                "material_b": f.material_id.iloc[j],
                "tfidf_difference_nonzero": int((x[i] - x[j]).nnz),
                "svd_distance": float(np.linalg.norm(actual[i] - actual[j])),
                "pole_target_gap": float(abs(f.pole.iloc[i] - f.pole.iloc[j])),
            }
        )
    pd.DataFrame(rows).to_csv(a.output / "collision_certificate.csv", index=False)
    metrics = []
    p_score = nli.pole_score.to_numpy()
    truth = f.qualification.to_numpy() * (2 * f.pole.to_numpy() - 1)
    for name, q in {
        "nli": nli.qualification_score.to_numpy(),
        "grain": f.qualification_score.to_numpy(),
        "lexical": lex.qualification_score.to_numpy(),
    }.items():
        value = q * (2 * p_score - 1)
        for role in ("source", "validation", "application"):
            mask = f.role.eq(role).to_numpy()
            mse = float(np.mean((value[mask] - truth[mask]) ** 2))
            metrics.append(
                {
                    "role": role,
                    "pole_reader": "nli",
                    "qualification_reader": name,
                    "materials": int(mask.sum()),
                    "signed_evidence_mse": mse,
                    "mean_qualified_evidence": float(value[mask].mean()),
                    "designed_mean": float(truth[mask].mean()),
                    "scope": "same-bank post-result diagnostic",
                }
            )
    pd.DataFrame(metrics).to_csv(a.output / "two_channel_followup.csv", index=False)
    summary = {
        "status": "RAW_TEXT_REPLAY_AND_COLLISION_CERTIFICATE_VERIFIED",
        "materials": len(f),
        "representation_replay_max_error": error,
        "score_replay_max_error": score_error,
        "paired_contrasts": len(rows),
        "tfidf_identical_pairs": sum(r["tfidf_difference_nonzero"] == 0 for r in rows),
        "svd_identical_pairs": sum(r["svd_distance"] == 0 for r in rows),
        "differing_analyzed_terms": sorted(differing),
        "retained_differing_terms": sorted(retained),
        "vocabulary_size": len(runtime.vectorizer.vocabulary_),
        "finite_bound": "Each equally-weighted opposite-pole pair with identical representation has mean squared pole risk (g-.5)^2+.25>=.25 for ANY deterministic downstream function; not a bound on other representations or natural-personality accuracy",
        "followup_application": [r for r in metrics if r["role"] == "application"],
        "code_and_output_audited_same_agent": True,
        "new_neural_calls": 0,
        "parent_promotion": False,
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
