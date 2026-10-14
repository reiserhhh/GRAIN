#!/usr/bin/env python3
"""Result-after coupling diagnosis, not a rescued primary metric or new run."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H


def gram(values):
    """Cosines of full signed action configurations, retaining every target."""
    x = np.asarray(values, float).reshape(6, -1)
    n = np.linalg.norm(x, axis=1)
    if (n <= 1e-12).any():
        raise ValueError("Unavailable action cannot be a confirmed direction")
    x = x / n[:, None]
    return x @ x.T


def run(parent):
    """Compare source mark overlap to actual input/output action overlap."""
    m = H.load(parent / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(parent, m["inputs"])
    H.verify_hashes(parent, H.load(parent / "02/seal.json"))
    out = parent / "post_result_coupling_r1"
    out.mkdir(exist_ok=False)
    pairs = H.load(parent / "00/pairs.json")
    with np.load(parent / "00/interventions.npz", allow_pickle=False) as z:
        axes, cov, arms = z["axes"], z["covariance"], z["arms"]
    with np.load(parent / "02/responses.npz", allow_pickle=False) as z:
        odd, scale = z["actor_odd"], z["read_scale"]
    summaries = []
    arrays = {"source_mark_gram": axes @ cov @ axes.T}
    for variant in range(3):
        ids = sorted(
            {p[k] for p in pairs if p["key"][0] == variant for k in ("low", "high")}
        )
        arm = arms[ids, 1:].reshape(96, 6, 2, 2, 105)
        change = (arm[:, :, 0, 1] - arm[:, :, 0, 0]) / 2
        input_gram = gram(change.transpose(1, 0, 2))
        output = odd[variant, :, :, :, :, 0, 1, :] / scale[1]
        output_gram = gram(output.transpose(3, 0, 1, 2, 4))
        arrays[f"input_gram_{variant}"] = input_gram
        arrays[f"output_gram_{variant}"] = output_gram
        summaries.append(
            {
                "variant": variant,
                "input_E_A_cosine": float(input_gram[0, 2]),
                "output_E_A_cosine": float(output_gram[0, 2]),
                "input_O_anthro_cosine": float(input_gram[4, 5]),
                "output_O_anthro_cosine": float(output_gram[4, 5]),
                "actor_odd_response_RMS_source_scaled": float(
                    np.sqrt(np.mean(output**2))
                ),
            }
        )
    with (out / "coupling.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    H.write(
        out / "summary.json",
        {
            "status": "POST_RESULT_DIAGNOSTIC_NOT_PRIMARY",
            "comparisons": summaries,
            "source_mark_gram": arrays["source_mark_gram"].tolist(),
            "numeric_or_neural_reexecution": False,
            "primary_failure_retained": True,
            "scope": "Overlap of designed GRAIN action cues, not psychological trait correlations",
        },
    )
    H.write(
        out / "manifest.json",
        {
            "code": H.sha(Path(__file__)),
            "source": {
                n: H.sha(parent / n)
                for n in ("00/interventions.npz", "02/responses.npz", "02/seal.json")
            },
            "outputs": H.seals(out, out),
            "registered_after_primary": True,
        },
    )
    print(summaries, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    run(p.parse_args().parent.resolve())
