#!/usr/bin/env python3
"""Write exact finite Gaussian observation-law certificates required by F05."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f05 import eta_from_rho, grouped
from grain_research.gptpro_foundation_v3.f05_invariants import (
    conditional_future,
    covariance_jacobian,
    observation_covariance,
    similarity_transform,
    stationary_covariance,
)
from grain_research.gptpro_foundation_v3.f05_recovery import (
    dense_posterior,
    generate_dynamics,
    kalman_loglik,
)


def dump(path, value):
    """Serialize computed certificates, not prefilled acceptance flags."""
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: x.tolist(), allow_nan=False)
        + "\n"
    )


def main():
    """Verify transformations, parameter restrictions and ten-person filters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior certificates")
    args.output.mkdir(parents=True)
    a, q, c, r = (
        np.array([[0.6, 0.15], [0.0, 0.3]]),
        np.diag([0.25, 0.16]),
        np.array([[1.0, 1.0]]),
        0.25,
    )
    p = stationary_covariance(a, q)
    theta = np.pi / 5
    transforms = {
        "rotation": np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        ),
        "scaling": np.diag([2.0, 0.5]),
    }
    original = observation_covariance(a, q, c, r, np.arange(64))
    rng = np.random.default_rng(20260915)
    history = rng.multivariate_normal(np.zeros(48), original[:48, :48])
    certificates = []
    for name, s in transforms.items():
        aa, qq, cc, pp = similarity_transform(a, q, c, p, s)
        changed = observation_covariance(aa, qq, cc, r, np.arange(64))
        expected = conditional_future(a, q, c, r, history, 16)
        actual = conditional_future(aa, qq, cc, r, history, 16)
        errors = {
            "covariance": float(np.max(np.abs(changed - original))),
            "forecast_mean": float(np.max(np.abs(expected[0] - actual[0]))),
            "forecast_covariance": float(np.max(np.abs(expected[1] - actual[1]))),
            "stationary_covariance": float(
                np.max(np.abs(stationary_covariance(aa, qq) - pp))
            ),
        }
        assert max(errors.values()) < 1e-10
        certificates.append(
            {
                "name": name,
                "S": s,
                "A": aa,
                "Q": qq,
                "C": cc,
                "P0": pp,
                "errors": errors,
                "preserves_fixed_C": bool(np.allclose(cc, c)),
                "preserves_fixed_Q": bool(np.allclose(qq, q)),
            }
        )
    ranks = {
        free: covariance_jacobian(a, q, c, r, free=free)
        for free in ("all", "fixed_CQ", "fixed_CQR")
    }
    v = np.array([1.0, -1.0]) / np.sqrt(2)
    dp = 0.02 * np.outer(v, v)
    q2 = q + dp - a @ dp @ a.T
    extra_error = float(
        np.max(np.abs(observation_covariance(a, q2, c, r, np.arange(64)) - original))
    )
    assert extra_error < 1e-10 and np.linalg.eigvalsh(q2).min() > 0
    rows, _ = generate_dynamics(
        "core", 20260915, {"source": 10, "validation": 0, "application": 0}, 64
    )
    checks = []
    for index, (author, events) in enumerate(grouped(rows).items()):
        psi = [0.2, -0.1, eta_from_rho(0.2 + 0.06 * index)]
        for gap in (1, 3):
            observed = events[::gap]
            actual = kalman_loglik(psi, observed, np.log([0.5, 0.5]))
            expected = dense_posterior(psi, observed, np.log([0.5, 0.5]))
            errors = [
                abs(actual["loglik"] - expected["loglik"]),
                abs(actual["state_mean"][-1] - expected["state_mean"]),
                abs(actual["state_variance"][-1] - expected["state_variance"]),
            ]
            assert max(errors) < 1e-10
            checks.append({"author_id": author, "gap": gap, "errors": errors})
    shift, rho = 1.7, 0.98
    t = np.arange(64)
    unit_difference = (0.3 + shift) + (-0.2 - shift) * np.ones(64) - (0.3 - 0.2)
    near_difference = shift * (1 - rho**t)
    dump(
        args.output / "similarity_invariants.json",
        {
            "A": a,
            "Q": q,
            "C": c,
            "R": r,
            "P0": p,
            "transforms": certificates,
            "local_jacobian_ranks": ranks,
            "additional_noise_equivalence": {
                "delta_P": dp,
                "alternative_Q": q2,
                "max_covariance_error": extra_error,
                "observability_rank": int(np.linalg.matrix_rank(np.vstack((c, c @ a)))),
                "proof": "delta_P C^T=0; Q'=Q+delta_P-A delta_P A^T yields P'=P+delta_P. All output autocovariances unchanged. Fixed observable A,C allow no nonidentity state similarity preserving both.",
            },
        },
    )
    dump(
        args.output / "filter_verification.json",
        {
            "authors": 10,
            "comparisons": checks,
            "unit_root_shift_max_error": float(np.max(np.abs(unit_difference))),
            "near_root_shift_at63": float(near_difference[-1]),
            "near_root_rho": rho,
        },
    )
    dump(
        args.output / "summary.json",
        {
            "status": "FINITE_MATHEMATICAL_CERTIFICATES_VERIFIED",
            "scope": "stationary linear Gaussian observation law and local Jacobian; not psychological construct identity or global restricted-model uniqueness",
            "ranks": {k: v["rank"] for k, v in ranks.items()},
            "two_transforms": len(certificates),
            "filter_cases": len(checks),
        },
    )
    paths = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f05_invariants.py",
        ROOT / "grain_research/gptpro_foundation_v3/f05_recovery.py",
    ]
    (args.output / "source_archive").mkdir()
    for path in paths:
        (args.output / "source_archive" / (sha256(path) + "_" + path.name)).write_bytes(
            path.read_bytes()
        )
    dump(
        args.output / "input_manifest.json",
        {"files": [{"path": str(p), "sha256": sha256(p)} for p in paths]},
    )
    dump(
        args.output / "output_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(args.output)), "sha256": sha256(p)}
                for p in sorted(args.output.rglob("*"))
                if p.is_file() and not p.name.startswith("._")
            ]
        },
    )


if __name__ == "__main__":
    main()
