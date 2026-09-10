#!/usr/bin/env python3
"""Publish NEXT66 with a bounded, post-outcome source-only spectral account."""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_paired_response_objective import OUT, manifest, old, prior


def spectral_account(gram, rhs, coef, alpha, relative_cutoff=1e-10):
    """Resolved source response capacity and ridge-only shrinkage, not an applied fit.

    Target squared norm is one by the independently audited source normalization.
    The Gram cutoff is explicit; discarded numerical modes are NOT proven noise.
    """
    gram, rhs, coef = [np.asarray(x, float) for x in (gram, rhs, coef)]
    if gram.shape != (len(rhs), len(rhs)) or coef.shape != rhs.shape or alpha <= 0:
        raise ValueError("Aligned source normal system required")
    if not all(np.isfinite(x).all() for x in (gram, rhs, coef)):
        raise ValueError("Finite source normal system required")
    lam, v = np.linalg.eigh((gram + gram.T) / 2)
    if lam[-1] <= 0 or lam[0] < -1e-10 * lam[-1]:
        raise ValueError("Positive semidefinite response Gram required")
    keep = lam > relative_cutoff * lam[-1]
    b = v.T @ rhs
    resolved = float(np.sum(b[keep] ** 2 / lam[keep]))
    if not -1e-8 <= resolved <= 1 + 1e-8:
        raise ValueError("Unit target projection inconsistent")
    e_min = v[:, keep] @ (b[keep] / lam[keep])
    e_ridge = v @ (b / (np.maximum(lam, 0) + alpha))
    error = lambda e: float(1 - 2 * e @ rhs + e @ gram @ e)
    ridge_loss = error(e_ridge)
    eigen_loss = float(1 + np.sum(lam * (v.T @ e_ridge) ** 2 - 2 * b * (v.T @ e_ridge)))
    return {
        "conditional_parameters": len(rhs),
        "resolved_rank": int(keep.sum()),
        "gram_relative_cutoff": relative_cutoff,
        "smallest_eigenvalue": float(lam[0]),
        "largest_eigenvalue": float(lam[-1]),
        "resolved_target_energy": resolved,
        "unresolved_or_dictionary_external_energy": 1 - resolved,
        "resolved_minimum_loss": error(e_min),
        "resolved_minimum_coefficient_norm": float(np.linalg.norm(e_min)),
        "resolved_minimum_ridge_cost": float(alpha * (e_min @ e_min)),
        "ridge_only_delta_loss": ridge_loss,
        "ridge_only_penalty": float(alpha * (e_ridge @ e_ridge)),
        "max_mode_ridge_retention": float(lam[-1] / (lam[-1] + alpha)),
        "current_delta_loss": error(coef),
        "current_delta_penalty": float(alpha * (coef @ coef)),
        "normal_equation_error": float(
            np.max(abs((gram + np.eye(len(rhs)) * alpha) @ e_ridge - rhs))
        ),
        "spectral_objective_error": abs(ridge_loss - eigen_loss),
    }


def publish():
    """Publish aggregate source diagnostics; never apply diagnostic inverses to new8."""
    start = time.process_time()
    m = manifest(OUT)
    audit = old.read(OUT / "audit.json")
    if audit["status"] != "PASS":
        raise ValueError("Primary audit required")
    fm = old.frozen(OUT, "models")
    summary = old.read(OUT / "analysis/summary.json")
    old.verify_hashes(OUT / "analysis", summary["files"])
    dst = ROOT / "reports/grain_vector_mechanisms/paired_response_objective_v1"
    dst.mkdir(exist_ok=True)
    hashes = {}
    for name, digest in summary["files"].items():
        if name.startswith("private_"):
            continue
        src = OUT / "analysis" / name
        if set(pd.read_csv(src, nrows=0).columns) & {
            "author_key",
            "user_id",
            "text",
            "tau",
        }:
            raise ValueError("Private field in publication")
        shutil.copyfile(src, dst / name)
        hashes[name] = digest
    rows = []
    for q in old.QUERIES:
        path = prior.model_path(OUT, q)
        old.verify_hashes(OUT / "models", {path.name: fm["files"][path.name]})
        model = old.arrays(path)
        prefix = (
            1
            + 2 * model["basis"].shape[1]
            + len(model["landmarks"]) * (model["basis"].shape[1] + 1)
        )
        if np.any(model["delta_gram"][:prefix]) or np.any(model["delta_rhs"][:prefix]):
            raise ValueError("Additive response prefix nonzero")
        args = (
            model["delta_gram"][prefix:, prefix:],
            model["delta_rhs"][prefix:],
            model["coef"][prefix:],
            float(model["alpha"]),
        )
        first = spectral_account(*args)
        replay = spectral_account(*args)
        if first != replay:
            raise ValueError("Spectral replay changed")
        expected = old.read(OUT / "models" / (q + "_status.json"))["diagnostics"][
            "source_delta_loss"
        ]
        np.testing.assert_allclose(
            first["current_delta_loss"], expected, atol=1e-10, rtol=1e-10
        )
        rows.append({"query_name": q, "source_authors": 4, **first})
    pd.DataFrame(rows).to_csv(dst / "source_spectral_account.csv", index=False)
    hashes["source_spectral_account.csv"] = old.sha(dst / "source_spectral_account.csv")
    cpu = time.process_time() - start
    if cpu > 30:
        raise TimeoutError("Post-outcome source account exceeded30sCPU")
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": audit,
            "files": hashes,
            "promotion": summary["promotion"],
            "criteria": summary["criteria"],
            "post_outcome_scope": "Source contrast Gram only; two eigendecompositions plus replay, no new8 diagnostic application or predictive refits",
            "post_outcome_cpu_seconds": cpu,
            "post_outcome_code": {
                str(p.relative_to(ROOT)): old.sha(p)
                for p in (
                    Path(__file__),
                    ROOT / "tests/test_grain_paired_response_spectrum.py",
                )
            },
            "scope": "source4 in-fit, exposed new8 development; fixed model L and GRAIN actions, no humanY",
        },
    )
    print({"public_tables": len(hashes), "post_cpu_seconds": cpu}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        publish()
