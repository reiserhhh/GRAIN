#!/usr/bin/env python3
"""Four-cell T1 source bootstrap and exact conditional Gaussian mixtures.

Source resampling re-estimates the original four-cell moment estimator. The
linear two-coefficient F13 experiment is not substituted for this target model.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS = (-1.0, -1 / 3, 1 / 3, 1.0)
CONTRAST = np.array([-1.0, 0.0, 0.0, 1.0])
READER_HASH = hashlib.sha256(b"F16-T1:numeric-four-cell-source-mixture:v1").hexdigest()


def code_hash():
    """Bind the standalone scoring mathematics to its frozen implementation."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sufficient(rows):
    """Validate one author's IID response records and retain exact cell statistics."""
    counts, sums, squares = np.zeros(4), np.zeros(4), np.zeros(4)
    ids, author, role = set(), None, None
    for row in rows:
        if set(row) != {"author_id", "role", "world", "event_id", "t", "z", "y"}:
            raise ValueError("only the declared F04 observation fields are accepted")
        if row["world"] != "L" or row["role"] not in ("source", "application"):
            raise ValueError("wrong T1 observation world or role")
        if author is None:
            author, role = row["author_id"], row["role"]
        if row["author_id"] != author or row["role"] != role:
            raise ValueError("mixed authors or roles")
        if row["event_id"] in ids:
            raise ValueError("duplicate event")
        ids.add(row["event_id"])
        if type(row["t"]) is not int or row["t"] < 0:
            raise ValueError("nonnegative integer time required")
        if isinstance(row["y"], bool) or not np.isfinite(row["y"]):
            raise ValueError("finite observation required")
        if isinstance(row["z"], bool) or row["z"] not in CONDITIONS:
            raise ValueError("unsupported condition")
        k = CONDITIONS.index(row["z"])
        counts[k] += 1
        sums[k] += row["y"]
        squares[k] += row["y"] ** 2
    return counts, sums, squares


def fit_source(statistics, selected):
    """Refit noise-corrected four-cell moments to bootstrap author copies."""
    counts, sums, squares = (np.asarray(array)[selected] for array in statistics)
    if len(selected) < 3 or np.any(counts <= 0):
        raise ValueError("three source authors and every condition required")
    means = sums / counts
    df = float(np.sum(counts - 1))
    if df <= 0:
        raise ValueError("positive within-cell residual degrees of freedom required")
    sigma2 = float(np.sum(squares - sums * means) / df)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("nonpositive source noise variance")
    sampling = np.diag(sigma2 * np.mean(1 / counts, axis=0))
    raw = np.cov(means, rowvar=False) - sampling
    eigen, vectors = np.linalg.eigh((raw + raw.T) / 2)
    covariance = (vectors * np.maximum(eigen, 1e-6)) @ vectors.T
    return {
        "mu": means.mean(axis=0).tolist(),
        "Sigma": covariance.tolist(),
        "sigma2": sigma2,
        "raw_eigenvalues": eigen.tolist(),
        "spectral_floor": 1e-6,
        "noise_df": df,
        "sampling_covariance": sampling.tolist(),
        "source_authors": len(selected),
        "method": "source_noise_corrected_four_cell_moments",
    }


def posterior(counts, sums, model):
    """Condition on sufficient cell means; missing conditions retain source borrowing."""
    mu = np.asarray(model["mu"], dtype=float)
    covariance = np.asarray(model["Sigma"], dtype=float)
    selected = np.flatnonzero(counts)
    if not len(selected):
        return mu.copy(), covariance.copy()
    cross = covariance[:, selected]
    k = covariance[np.ix_(selected, selected)] + np.diag(
        model["sigma2"] / counts[selected]
    )
    solve = np.linalg.solve(
        k, np.column_stack([sums[selected] / counts[selected] - mu[selected], cross.T])
    )
    mean = mu + cross @ solve[:, 0]
    variance = covariance - cross @ solve[:, 1:]
    return mean, (variance + variance.T) / 2


def raw_score(rows, bundle):
    """Integrate the four-dimensional posterior; keep analytic and 200-draw versions."""
    counts, sums, _ = sufficient(rows)
    parts = []
    for b, model in enumerate(bundle["source_models"]):
        mean, cov = posterior(counts, sums, model)
        score = float(CONTRAST @ mean)
        variance = float(CONTRAST @ cov @ CONTRAST)
        # Shared numerical integration arrays prevent ID/batch-dependent scores.
        rng = np.random.default_rng(np.random.SeedSequence([bundle["seed"], 16, 1, b]))
        parameters = (
            mean
            + rng.standard_normal((bundle["personal_draws"], 4))
            @ np.linalg.cholesky(cov).T
        )
        scores = parameters @ CONTRAST
        parts.append(
            {
                "bootstrap": b,
                "mean": score,
                "conditional_variance": variance,
                "mc_mean": float(scores.mean()),
                "mc_variance": float(scores.var(ddof=0)),
            }
        )
    means = np.asarray([row["mean"] for row in parts])
    mc_means = np.asarray([row["mc_mean"] for row in parts])
    within = float(np.mean([row["conditional_variance"] for row in parts]))
    between = float(means.var(ddof=0))
    mc_within = float(np.mean([row["mc_variance"] for row in parts]))
    mc_between = float(mc_means.var(ddof=0))
    score, variance = float(means.mean()), within + between
    radius = 1.96 * np.sqrt(variance)
    mc_score, mc_variance = float(mc_means.mean()), mc_within + mc_between
    mc_radius = 1.96 * np.sqrt(mc_variance)
    return {
        "target_id": "T1",
        "score": score,
        "variance": variance,
        "lower": float(score - radius),
        "upper": float(score + radius),
        "components": {"personal_conditional": within, "source": between},
        "source_components": parts,
        "mc_score": mc_score,
        "mc_variance": mc_variance,
        "mc_lower": float(mc_score - mc_radius),
        "mc_upper": float(mc_score + mc_radius),
        "mc_components": {"personal_conditional": mc_within, "source": mc_between},
        "support": "observed" if counts[0] > 0 and counts[3] > 0 else "model_borrowed",
        "prior_only": not rows,
        "observed_count": len(rows),
        "source_uncertainty_included": True,
        "calibration": "NOT_EVALUATED",
        "approximation": "empirical_source_bootstrap_mixture;exact_conditional_Gaussians;Gaussian_moment_interval",
        "mc_scope": "finite200_joint_four_cell_draw_comparator;unadjusted_finite_MC_moments_not_an_extra_psychological_component",
    }


def verify_bundle(bundle):
    """Reject changed mathematics, reference or incomplete source ensembles."""
    if (
        bundle["scorer_code_hash"] != code_hash()
        or bundle["formula_version"] != "T1_four_cell_source_mixture_v1"
    ):
        raise ValueError("changed T1 scorer/formula")
    if (
        bundle["target_id"] != "T1"
        or bundle["world_contract"] != "F04_L_four_condition"
        or bundle["reader_hash"] != READER_HASH
    ):
        raise ValueError("wrong T1 target/world/reader")
    if len(bundle["source_models"]) != bundle["source_bootstraps"]:
        raise ValueError("incomplete source ensemble")
    for model in bundle["source_models"]:
        mu, cov = np.asarray(model["mu"]), np.asarray(model["Sigma"])
        if (
            mu.shape != (4,)
            or cov.shape != (4, 4)
            or not np.isfinite(mu).all()
            or not np.isfinite(cov).all()
        ):
            raise ValueError("nonfinite four-cell prior")
        if (
            not np.allclose(cov, cov.T, rtol=0, atol=1e-12)
            or not np.isfinite(model["sigma2"])
            or model["sigma2"] <= 0
        ):
            raise ValueError("invalid covariance or noise")
        np.linalg.cholesky(cov)
    if (
        not np.isfinite(bundle["reference_mu"])
        or not np.isfinite(bundle["reference_sigma"])
        or bundle["reference_sigma"] <= 0
    ):
        raise ValueError("source reference not frozen")


def cold_score(raw, bundle):
    """Score using bundle and this author's prefix only; never read source files."""
    verify_bundle(bundle)
    if (
        set(raw) != {"world_contract", "reader_hash", "records"}
        or raw["world_contract"] != bundle["world_contract"]
        or raw["reader_hash"] != READER_HASH
    ):
        raise ValueError("input target/world/reader mismatch")
    result = raw_score(raw["records"], bundle)
    result["reference_score"] = (result["score"] - bundle["reference_mu"]) / bundle[
        "reference_sigma"
    ]
    result["reference_variance"] = result["variance"] / bundle["reference_sigma"] ** 2
    return result


def main():
    """Exact-byte authenticated JSON cold interface; no pickle or source access."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    payload = args.bundle.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.sha256:
        raise ValueError("bundle hash mismatch")
    bundle = json.loads(payload)
    with args.input.open() as stream, args.output.open("x") as out:
        for line in stream:
            out.write(
                json.dumps(
                    cold_score(json.loads(line), bundle),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
