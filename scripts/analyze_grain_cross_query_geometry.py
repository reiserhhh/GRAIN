#!/usr/bin/env python3
"""Compare frozen query-specific response spaces and functional directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def geometry(u, v, a, b):
    """Separate basis overlap from signed functional alignment in a common instrument."""
    u, v, a, b = (np.asarray(t, float) for t in (u, v, a, b))
    if u.shape[0] != v.shape[0] or a.shape != (len(u),) or b.shape != a.shape:
        raise ValueError("Response operators must use the same hidden coordinate space")
    if not np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-6) or not np.allclose(
        v.T @ v, np.eye(v.shape[1]), atol=1e-6
    ):
        raise ValueError("Principal angles require orthonormal bases")
    singular = np.clip(np.linalg.svd(u.T @ v, compute_uv=False), 0, 1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not all(np.isfinite(t).all() for t in (u, v, a, b)) or min(na, nb) == 0:
        raise ValueError("Finite nonzero response operators required")
    normalized = np.column_stack([a / na, b / nb])
    sv = np.linalg.svd(normalized, compute_uv=False)
    return {
        "hidden_dimensions": len(u),
        "ranks": [u.shape[1], v.shape[1]],
        "principal_cosines": singular.tolist(),
        "principal_angles_degrees": np.degrees(np.arccos(singular)).tolist(),
        "normalized_projector_overlap": float(
            np.sum(singular**2) / min(u.shape[1], v.shape[1])
        ),
        "isotropic_random_overlap_expectation": max(u.shape[1], v.shape[1]) / len(u),
        "functional_cosine": float(a @ b / (na * nb)),
        "a_energy_inside_b_space": float(np.linalg.norm(v.T @ a) ** 2 / na**2),
        "b_energy_inside_a_space": float(np.linalg.norm(u.T @ b) ** 2 / nb**2),
        "normalized_operator_rank1_energy": float(sv[0] ** 2 / np.sum(sv**2)),
        "operator_norms": [float(na), float(nb)],
    }


def main():
    """Analyze saved development operators with no new author labels or forwards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.operator_dir / "frozen_selection.json").read_text())
    spaces, vectors = [], []
    tasks = ("TF_cont", "neuroticism")
    for task in tasks:
        path = args.operator_dir / frozen["models"][task]["shared"]["model_file"]
        with np.load(path, allow_pickle=False) as m:
            spaces.append(m["basis"])
            vectors.append(m["basis"] @ (m["weights"] / m["response_scale"]))
    result = {
        "status": "COMPLETE_EXPLORATORY_GEOMETRY",
        "tasks": list(tasks),
        "data_role": "development-only operators; different author populations across queries",
        "interpretation": "geometric and functional overlap, not questionnaire correlation",
        "metrics": geometry(*spaces, *vectors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
