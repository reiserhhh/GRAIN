#!/usr/bin/env python3
"""Finite conditional-current/GRAIN interface, without predictor or neural fitting."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_author_current as A
from grain_research.construction import math_conditional_current as M
from scripts import run_grain_opportunity_current_hm94 as R

CFG = ROOT / "configs/grain_conditional_current_lift_hm95.json"
CODE = (
    CFG,
    ROOT / "docs/GRAIN_CONDITIONAL_CURRENT_LIFT_PLAN_HM95.md",
    ROOT / "grain_research/construction/math_conditional_current.py",
    ROOT / "scripts/run_grain_conditional_current_hm95.py",
    ROOT / "scripts/verify_grain_conditional_current_hm95.py",
    ROOT / "tests/test_grain_math_conditional_current.py",
)
SPACES = ("resolved", "visible", "hidden")
CHANNELS = ("raw", "opportunity", "response")
TOL = 1e-12


def verify(out):
    """Authenticate explicit numerical products and this protocol's frozen code."""
    manifest = R.read(out / "00/manifest.json")
    for path, digest in manifest["hashes"].items():
        if R.sha(ROOT / path) != digest:
            raise ValueError("Frozen input/code changed: " + path)
    return manifest["config"]


def begin(out, name):
    """Create a non-overwriting recoverable finite stage."""
    cfg = verify(out)
    path = out / name
    path.mkdir(exist_ok=False)
    R.write(path / "checkpoint.json", {"status": "RUNNING"})
    return cfg, path, time.monotonic()


def prepare(out):
    """Freeze protocol and new author ledgers before numerical outcomes."""
    cfg = R.read(CFG)
    natural, vector = (ROOT / cfg[k] for k in ("natural_source", "vector_source"))
    R.sealed(natural)
    for stage in ("01", "02", "03", "04_matched_null"):
        R.previous(natural, stage)
    if (
        R.read(natural / "verification.json")["status"]
        != "VERIFIED_FINITE_HM94_NOT_GLOBAL"
    ):
        raise ValueError("Natural predecessor not verified")
    if (
        R.read(vector / "verification.json")["status"]
        != "VERIFIED_FINITE_HM93_NOT_GLOBAL"
    ):
        raise ValueError("Vector predecessor not verified")
    for filename, digest in R.read(vector / "02/output_seal.json").items():
        if R.sha(vector / filename) != digest:
            raise ValueError("Vector kernel seal changed")
    inputs = [
        natural / f
        for f in (
            "01/source_model.npz",
            "01/partition.json",
            "02/private_author_accounts.npz",
            "02/private_metadata.json",
        )
    ]
    inputs += [
        vector / f for f in ("02/kernels.npz", "00/manifest.json", "verification.json")
    ]
    (out / "00").mkdir(parents=True, exist_ok=False)
    paths, truth = [], []
    for pattern, sign, seed, replicate in itertools.product(
        range(3), (-1, 1), cfg["seeds"], range(cfg["profiles_per_sign_seed"])
    ):
        author = f"toy_{len(paths):03d}"
        sequences = [
            A.euler_path(
                A.cycle_counts(int(sign * M.walsh_patterns()[pattern, c])),
                seed * 10000 + replicate * 10 + c,
            )[:-1].tolist()
            for c in range(4)
        ]
        paths.append({"author": author, "paths": sequences})
        truth.append(
            {
                "author": author,
                "pattern": pattern,
                "sign": sign,
                "seed": seed,
                "replicate": replicate,
            }
        )
    R.write(out / "00/application_paths.json", paths)
    R.write(out / "00/evaluation_truth.json", truth)
    inputs += [out / "00/application_paths.json", out / "00/evaluation_truth.json"]
    R.write(
        out / "00/manifest.json",
        {
            "config": cfg,
            "hashes": {str(p.relative_to(ROOT)): R.sha(p) for p in (*CODE, *inputs)},
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "new_neural_calls": 0,
            "new_application_ledgers": len(paths),
            "original_scope_complete": False,
            "raw_or_external_labels_read": False,
        },
    )
    print(
        "PREPARED_HM95",
        len(paths),
        "new numeric ledgers; outcomes not read",
        flush=True,
    )


def source(out):
    """Construct source-common comment laws and retain actual vector kernels."""
    cfg, path, started = begin(out, "01")
    natural, vector = (ROOT / cfg[k] for k in ("natural_source", "vector_source"))
    with np.load(natural / "01/source_model.npz", allow_pickle=False) as data:
        counts = data["source_counts"]
        references = np.stack([data["reference_" + arm] for arm in cfg["arms"]])
    patterns, probabilities, initial, transitions = M.source_markov(
        counts, cfg["source_markov_pseudocount"]
    )
    pair_laws = np.zeros((6, 3, 4, 4))
    for z, r in itertools.product(range(6), range(3)):
        np.add.at(
            pair_laws[z, r], (patterns[:, r], patterns[:, r + 1]), probabilities[z]
        )
    with np.load(vector / "02/kernels.npz", allow_pickle=False) as data:
        kernel = data["kernel"].reshape(4, 6, 3, 3, 3, 6)
        reset = data["reset"].reshape(4, 6, 3, 3, 6)
    np.savez_compressed(
        path / "source_bundle.npz",
        patterns=patterns,
        probabilities=probabilities,
        initial=initial,
        transitions=transitions,
        pair_laws=pair_laws,
        references=references,
        kernel=kernel,
        reset=reset,
    )
    R.finish(
        path,
        {
            "status": "FROZEN_SOURCE_HM95",
            "source_authors": len(counts),
            "markov_patterns": len(patterns),
            "source_common_not_personal": True,
            "new_neural_calls": 0,
            "new_predictors": 0,
            "wall_seconds": time.monotonic() - started,
        },
    )


def apply(out):
    """Cold numerical application without score proxies or toy truth inputs."""
    cfg, path, started = begin(out, "02")
    R.previous(out, "01")
    natural = ROOT / cfg["natural_source"]
    metadata = R.read(natural / "02/private_metadata.json")
    with np.load(
        natural / "02/private_author_accounts.npz", allow_pickle=False
    ) as data:
        counts = np.stack([data[r["key"] + "_counts"] for r in metadata])
        states = np.concatenate([data[r["key"] + "_states"] for r in metadata])
        conditions = np.concatenate([data[r["key"] + "_conditions"] for r in metadata])
        views = np.concatenate([data[r["key"] + "_views"] for r in metadata])
    authors = np.concatenate(
        [np.full(r["comments"], i) for i, r in enumerate(metadata)]
    )
    strata = np.array([len(r["queries"]) for r in metadata])
    with np.load(out / "01/source_bundle.npz", allow_pickle=False) as data:
        references, pairs, kernel, reset = (
            data[k] for k in ("references", "pair_laws", "kernel", "reset")
        )
    reconstructed = M.counts_for_comments(
        states, authors, views, conditions, len(metadata), 6
    )
    if not np.array_equal(reconstructed, counts):
        raise AssertionError("Natural addresses/counts changed")
    features, q = M.lift(counts, references)
    expected_counts = counts.sum((-2, -1))[..., None, None] * pairs[None, None]
    expected, _ = M.lift(expected_counts, references)
    np.savez_compressed(
        path / "private_features.npz",
        **features,
        **{"expected_" + key: value for key, value in expected.items()},
        counts=counts,
        authors=authors,
        views=views,
        conditions=conditions,
        strata=strata,
        q=q,
    )
    paths = R.read(out / "00/application_paths.json")
    if any(set(p) != {"author", "paths"} for p in paths):
        raise ValueError("Unexpected application truth fields")
    scores, laws = [], []
    names = R.read(ROOT / cfg["vector_source"] / "00/manifest.json")["spec"]["targets"]
    for record in paths:
        local = np.stack([A.path_joint(p, closed=True) for p in record["paths"]]) / 4
        pooled = local.sum(0)
        full_score = np.einsum("cij,cfijht->fht", local, kernel)
        flat_score = np.einsum("ij,fijht->fht", pooled, kernel.mean(0))
        reset_kernel = np.broadcast_to(reset[:, :, None], kernel.shape)
        reset_score = np.einsum("cij,cfijht->fht", local, reset_kernel)
        laws.append(local)
        for family, field, target in itertools.product(range(6), range(3), range(6)):
            scores.append(
                {
                    "author": record["author"],
                    "family": names[family],
                    "field": ("uniform_value", "local_query", "local_bound_query")[
                        field
                    ],
                    "target": names[target],
                    "conditional": full_score[family, field, target],
                    "pooled": flat_score[family, field, target],
                    "reset": reset_score[family, field, target],
                }
            )
    np.savez_compressed(path / "toy_laws.npz", laws=laws)
    pd.DataFrame(scores).to_csv(path / "toy_scores.csv", index=False)
    R.finish(
        path,
        {
            "status": "COLD_APPLIED_HM95",
            "application_authors": len(metadata),
            "application_comments": len(states),
            "toy_ledgers": len(paths),
            "toy_scores": len(scores),
            "personality_scores_input": False,
            "new_fits": 0,
            "new_neural_calls": 0,
            "wall_seconds": time.monotonic() - started,
        },
    )


def covariance_accounts(features, strata):
    """Preserve all arm/channel/space covariance contributions in fixed order."""
    return np.stack([M.author_covariance(features[k], strata) for k in SPACES], axis=-1)


def evaluate(out):
    """Test hidden author structure and frozen target action, never select predictors."""
    cfg, path, started = begin(out, "03")
    for stage in ("01", "02"):
        R.previous(out, stage)
    with np.load(out / "02/private_features.npz", allow_pickle=False) as data:
        features = {k: data[k] for k in SPACES}
        expected = {k: data["expected_" + k] for k in SPACES}
        authors, views, conditions, strata, real_counts = (
            data[k] for k in ("authors", "views", "conditions", "strata", "counts")
        )
    with np.load(out / "01/source_bundle.npz", allow_pickle=False) as data:
        patterns, probabilities, references = (
            data[k] for k in ("patterns", "probabilities", "references")
        )
    contributions = covariance_accounts(features, strata)
    mean_contributions = covariance_accounts(expected, strata)
    observed, expected_mean = contributions.mean(0), mean_contributions.mean(0)
    rng, ghost, max_q_error = np.random.default_rng(cfg["seed"]), [], 0.0
    for draw in range(cfg["ghost_draws"]):
        states = M.sample_comments(patterns, probabilities, conditions, rng)
        counts = M.counts_for_comments(
            states, authors, views, conditions, len(strata), 6
        )
        if not np.array_equal(counts.sum((-2, -1)), real_counts.sum((-2, -1))):
            raise AssertionError("Ghost changed opportunity or text budget")
        gf, _ = M.lift(counts, references)
        value = covariance_accounts(gf, strata).mean(0)
        max_q_error = max(max_q_error, float(np.max(abs(value[:, 1] - observed[:, 1]))))
        ghost.append(value)
        if draw % 100 == 0:
            R.write(
                path / "checkpoint.json", {"status": "RUNNING", "ghost_draws": draw + 1}
            )
            print("HM95 opportunity-preserving null", draw + 1, flush=True)
        if time.monotonic() - started > cfg["max_cpu_seconds"]:
            raise TimeoutError("Finite null budget exhausted; preserve partial stage")
    ghost = np.array(ghost)
    ai = cfg["arms"].index(cfg["primary_account"][0])
    ci, si = (
        CHANNELS.index(cfg["primary_account"][1]),
        SPACES.index(cfg["primary_account"][2]),
    )
    primary_bootstrap = []
    boot_rng = np.random.default_rng(cfg["seed"] + 1)
    for _ in range(cfg["bootstrap_draws"]):
        indices = np.concatenate(
            [
                boot_rng.choice(
                    np.flatnonzero(strata == g), np.sum(strata == g), replace=True
                )
                for g in np.unique(strata)
            ]
        )
        a = M.author_covariance(
            features[SPACES[si]][indices, :, ai, ci], strata[indices]
        ).mean()
        b = M.author_covariance(
            expected[SPACES[si]][indices, :, ai, ci], strata[indices]
        ).mean()
        primary_bootstrap.append(a - b)
    rows = []
    for a, c, s in itertools.product(range(4), range(3), range(3)):
        value, null = observed[a, c, s], ghost[:, a, c, s]
        rows.append(
            {
                "arm": cfg["arms"][a],
                "channel": CHANNELS[c],
                "space": SPACES[s],
                "covariance": value,
                "source_common_expectation": expected_mean[a, c, s],
                "delta_over_source_common": value - expected_mean[a, c, s],
                "ghost_mean": null.mean(),
                "ghost_lower": np.quantile(null, 0.025),
                "ghost_upper": np.quantile(null, 0.975),
                "ghost_p": (1 + np.count_nonzero(null >= value - TOL))
                / (len(null) + 1),
                "primary": (a, c, s) == (ai, ci, si),
            }
        )
    pd.DataFrame(rows).to_csv(path / "covariance_accounts.csv", index=False)
    np.savez_compressed(
        path / "null_and_covariance.npz",
        ghost=ghost,
        observed=observed,
        expected=expected_mean,
        contributions=contributions,
        primary_bootstrap=primary_bootstrap,
    )
    truth = pd.DataFrame(R.read(out / "00/evaluation_truth.json"))
    scores = pd.read_csv(out / "02/toy_scores.csv").merge(
        truth, on="author", validate="many_to_one"
    )
    grouped = scores.groupby(["pattern", "family", "field", "target", "sign"])[
        ["conditional", "pooled", "reset"]
    ].mean()
    contrast = grouped.xs(1, level="sign") - grouped.xs(-1, level="sign")
    contrast.reset_index().to_csv(path / "conditional_target_actions.csv", index=False)
    base, tangents, fisher, pooled_fisher = M.fisher_witness()
    np.savez_compressed(
        path / "fisher_witness.npz",
        base=base,
        tangents=tangents,
        full=fisher,
        pooled=pooled_fisher,
    )
    primary = rows[(ai * 3 + ci) * 3 + si]
    summary = {
        "status": "EVALUATED_FINITE_HM95_NOT_GLOBAL",
        "primary": primary,
        "primary_delta_author_CI": np.quantile(
            primary_bootstrap, (0.025, 0.975)
        ).tolist(),
        "opportunity_ghost_invariance_error": max_q_error,
        "covariance_account_error": float(
            np.max(
                abs(
                    contributions[..., 0]
                    - contributions[..., 1]
                    - contributions[..., 2]
                )
            )
        ),
        "all_toy_contrasts": len(contrast),
        "toy_conditional_contrast_abs_range": [
            float(contrast.conditional.abs().min()),
            float(contrast.conditional.abs().max()),
        ],
        "toy_pooled_contrast_max": float(contrast.pooled.abs().max()),
        "toy_reset_contrast_max": float(contrast.reset.abs().max()),
        "fisher_full": fisher.tolist(),
        "fisher_pooled": pooled_fisher.tolist(),
        "source_null_not_exact_natural_truth": True,
        "new_neural_calls": 0,
        "new_predictors": 0,
        "original_scope_complete": False,
        "wall_seconds": time.monotonic() - started,
    }
    R.finish(path, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "source", "apply", "evaluate"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    destination = args.out or ROOT / R.read(CFG)["output"]
    globals()[args.stage](destination)
