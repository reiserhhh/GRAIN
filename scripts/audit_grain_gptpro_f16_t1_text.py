#!/usr/bin/env python3
"""Alternate finite latent-state and product-likelihood replay of T1 text scores."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from grain_research.gptpro_foundation_v3 import f16_t1_text as scorer
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file


def alternate(counts, bundle, arm):
    """Enumerate four latent bit/qualification states; multiply event likelihoods."""
    n = len(counts)
    sums, squares, component_squares = np.zeros(n), np.zeros(n), np.zeros(n)
    for b, prior in enumerate(bundle["source_models"]):
        seed = int(
            np.random.SeedSequence([bundle["seed"], 161, b, 0]).generate_state(1)[0]
        )
        u = qmc.Sobol(4, scramble=True, seed=seed).random_base2(bundle["qmc_power"])
        z = ndtri(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
        points = np.asarray(prior["mu"]) + z @ np.linalg.cholesky(prior["Sigma"]).T
        bit1 = ndtr(points / np.sqrt(prior["sigma2"]))
        bit0 = ndtr(-points / np.sqrt(prior["sigma2"]))
        states = np.stack(
            [
                np.full_like(bit0, 0.25),
                0.5 * bit0,
                np.full_like(bit1, 0.25),
                0.5 * bit1,
            ],
            axis=-1,
        )
        if arm == "oracle":
            channel = np.eye(4).reshape(2, 2, 4)
        else:
            key = "nli_joint" if arm == "nli_joint" else "hybrid_joint"
            channel = np.array(bundle["channel_draws"][key][b])
            if arm == "hybrid_factorized":
                rebuilt = np.empty_like(channel)
                for pole in (0, 1):
                    for q in (0, 1):
                        cell = channel[pole, q].reshape(2, 2)
                        for c in range(4):
                            rebuilt[pole, q, c] = (
                                cell[c // 2, :].sum() * cell[:, c % 2].sum()
                            )
                channel = rebuilt
        probabilities = np.einsum("njs,sc->njc", states, channel.reshape(4, 4))
        weights = np.ones((len(points), n))
        for condition in range(4):
            for category in range(4):
                weights *= (
                    probabilities[:, condition, category, None]
                    ** counts[None, :, condition, category]
                )
        mass = weights.sum(axis=0)
        if (mass <= 0).any():
            raise ValueError("product replay underflow, do not skip")
        weights /= mass
        delta = points[:, 3] - points[:, 0]
        mu = np.sum(weights * delta[:, None], axis=0)
        second = np.sum(weights * delta[:, None] ** 2, axis=0)
        sums += mu
        squares += second
        component_squares += mu**2
    b = len(bundle["source_models"])
    mean = sums / b
    return mean, squares / b - mean**2, component_squares / b - mean**2


def raw_counts(inputs, bundle, arm):
    """Reconstruct unedited input counts without using the scorer's history parser."""
    output = np.zeros((len(inputs), 4, 4))
    conditions = (-1.0, -1 / 3, 1 / 3, 1.0)
    for i, raw in enumerate(inputs):
        ids = set()
        for op in raw["history"]:
            if op["op"] != "new":
                raise ValueError("unexpected edited base history")
            row = op["observation"]
            event = row["event_id"]
            if event in ids or row["author_id"] != raw["author_id"]:
                raise ValueError("event identity error")
            ids.add(event)
            text = bundle["materials"][row["material_id"]]
            q = (
                text["nli_qualification"]
                if arm == "nli_joint"
                else text["grain_qualification"]
            )
            c = 2 * int(text["pole_readout"] >= 0.5) + int(q >= 0.5)
            output[i, conditions.index(row["z"]), c] += 1
    return output


def gaussian_replay(inputs, evaluator, bundle, naive):
    """Use whitened-prior normal equations, not the archived cell-mean solver."""
    n = len(inputs)
    counts, sums = np.zeros((n, 4)), np.zeros((n, 4))
    index = {raw["author_id"]: i for i, raw in enumerate(inputs)}
    for row in evaluator:
        i, k = index[row["author_id"]], (-1.0, -1 / 3, 1 / 3, 1.0).index(row["z"])
        weight = row["grain_qualification"] if naive else row["qualification"]
        value = 2 * row["pole_readout"] - 1 if naive else row["y"]
        counts[i, k] += weight
        sums[i, k] += weight * value
    means, variances = [], []
    contrast = np.array([-1.0, 0.0, 0.0, 1.0])
    for prior in bundle["source_models"]:
        mu = np.asarray(prior["mu"])
        lower = np.linalg.cholesky(prior["Sigma"])
        precision = np.eye(4) + np.einsum(
            "ki,nk,kj->nij", lower, counts / prior["sigma2"], lower
        )
        rhs = ((sums - counts * mu) / prior["sigma2"]) @ lower
        latent_mean = np.linalg.solve(precision, rhs[..., None])[..., 0]
        target_loading = contrast @ lower
        value = mu @ contrast + latent_mean @ target_loading
        solved = np.linalg.solve(
            precision, np.broadcast_to(target_loading, (n, 4))[..., None]
        )[..., 0]
        variance = solved @ target_loading
        means.append(value)
        variances.append(variance)
    means = np.array(means)
    return means.mean(0), np.mean(variances, axis=0) + means.var(0)


def main():
    """Audit all saved application arms, source norm and actual history controls."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    began = time.monotonic()
    manifest = json.loads((a.run / "output_manifest.json").read_text())
    for name, value in manifest.items():
        if sha256_file(a.run / name) != value:
            raise ValueError(f"changed result {name}")
    protocol = json.loads((a.run / "protocol_before_run.json").read_text())
    for key in ("source_hashes", "code_hashes"):
        for name, value in protocol[key].items():
            if sha256_file(Path(name)) != value:
                raise ValueError(f"changed source/code {name}")
    all_errors = []
    history_checks = []
    formula_objects = 0
    bank = pd.read_csv(
        Path(__file__).resolve().parents[1]
        / "results/grain_gptpro_foundation_v3/F08/nli_reader_r1/reader_scores.csv"
    ).set_index("material_id")
    for seed in protocol["seeds"]:
        root = a.run / str(seed)
        bundle = json.loads((root / "bundle.json").read_text())
        inputs = [
            json.loads(line)
            for line in (root / "application_inputs.jsonl").read_text().splitlines()
        ]
        saved = json.loads((root / "arm_scores.json").read_text())
        arrays = np.load(root / "counts.npz")
        evaluation = json.loads(
            (root / "application_rendering_evaluator.json").read_text()
        )
        author_index = {r["author_id"]: i for i, r in enumerate(inputs)}
        reconstructed_oracle = np.zeros_like(arrays["oracle"])
        for r in evaluation:
            k = (-1.0, -1 / 3, 1 / 3, 1.0).index(r["z"])
            i = author_index[r["author_id"]]
            reconstructed_oracle[i, k, 2 * r["rendered_pole"] + r["qualification"]] += 1
        np.testing.assert_array_equal(reconstructed_oracle, arrays["oracle"])
        for arm in ("hybrid_joint", "nli_joint", "hybrid_factorized", "oracle"):
            if arm != "oracle":
                np.testing.assert_array_equal(
                    raw_counts(inputs, bundle, arm), arrays[arm]
                )
            mean, var, between = alternate(arrays[arm], bundle, arm)
            ref = np.array(
                [
                    [r[k] for k in ("score", "variance", "source_and_reader_variance")]
                    for r in saved[arm]
                ]
            )
            error = float(np.max(np.abs(np.column_stack([mean, var, between]) - ref)))
            if error > 1e-10:
                raise ValueError("alternate posterior replay mismatch")
            all_errors.append(
                {"seed": seed, "arm": arm, "authors": len(mean), "max_error": error}
            )
            formula_objects += len(mean)
        for arm in ("naive_gaussian", "eligible_continuous"):
            mean, var = gaussian_replay(
                inputs, evaluation, bundle, arm == "naive_gaussian"
            )
            reference = np.array([[r["score"], r["variance"]] for r in saved[arm]])
            error = float(np.max(abs(np.column_stack([mean, var]) - reference)))
            if error > 1e-9:
                raise ValueError("Gaussian comparator alternate replay mismatch")
            all_errors.append(
                {"seed": seed, "arm": arm, "authors": len(mean), "max_error": error}
            )
            formula_objects += len(mean)
        source_inputs = [
            json.loads(line)
            for line in (root / "source_inputs.jsonl").read_text().splitlines()
        ]
        mean, var, _ = alternate(
            raw_counts(source_inputs, bundle, "hybrid_joint"), bundle, "hybrid_joint"
        )
        source_saved = json.loads((root / "source_scores.json").read_text())
        np.testing.assert_allclose(
            mean, [r["score"] for r in source_saved], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            var, [r["variance"] for r in source_saved], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            [mean.mean(), mean.std(ddof=1)],
            [bundle["reference_mu"], bundle["reference_sigma"]],
            atol=1e-10,
            rtol=0,
        )
        # Real authored event histories, not just unit-test fixtures.
        base = copy.deepcopy(inputs[0])
        base["target_time"] = 34
        parent = base["history"][-1]["record_id"]
        annotated = copy.deepcopy(base)
        annotated["history"] += [
            {
                "op": "alias",
                "record_id": "audit_alias",
                "recorded_at": 32,
                "parent_id": parent,
            },
            {
                "op": "summary",
                "record_id": "audit_summary",
                "recorded_at": 32,
                "parents": [parent],
            },
        ]
        revised = copy.deepcopy(annotated)
        observation = copy.deepcopy(base["history"][-1]["observation"])
        old = bank.loc[observation["material_id"]]
        match = bank[
            (bank.role == old.role)
            & (bank.subject_status == old.subject_status)
            & (bank.variant == old.variant)
            & (bank.pole == 1 - old.pole)
        ]
        if len(match) != 1:
            raise ValueError("missing same-style opposite-pole material")
        observation["material_id"] = match.index[0]
        revised["history"].append(
            {
                "op": "revise",
                "record_id": "audit_revision",
                "recorded_at": 33,
                "parent_id": parent,
                "observation": observation,
            }
        )
        retracted = copy.deepcopy(revised)
        retracted["history"].append(
            {
                "op": "retract",
                "record_id": "audit_retraction",
                "recorded_at": 34,
                "parent_id": "audit_revision",
            }
        )
        direct = copy.deepcopy(base)
        direct["history"] = direct["history"][:-1]
        results = scorer.cold_scores(
            [base, annotated, revised, retracted, direct], bundle
        )
        if (
            abs(results[0]["score"] - results[1]["score"]) > 1e-12
            or abs(results[3]["score"] - results[4]["score"]) > 1e-12
        ):
            raise ValueError("alias/retraction changes wrong evidence")
        if results[2]["stale_annotations"] != ["audit_alias", "audit_summary"]:
            raise ValueError("stale summaries not tracked")
        history_checks.append(
            {
                "seed": seed,
                "scores": results,
                "original_material": old.name,
                "revised_material": observation["material_id"],
            }
        )
        print(
            json.dumps({"audited_seed": seed, "posterior_objects": formula_objects}),
            flush=True,
        )
    f = pd.read_csv(a.run / "scores.csv")
    table = pd.read_csv(a.run / "metrics_by_seed.csv")
    for r in table.to_dict("records"):
        sub = f[(f.seed == r["seed"]) & (f.arm == r["arm"])]
        d = sub.score - sub.truth
        actual = [
            np.mean(d * d),
            d.mean(),
            sub[["score", "truth"]].corr().iloc[0, 1],
            ((sub.lower <= sub.truth) & (sub.truth <= sub.upper)).mean(),
            (sub.upper - sub.lower).mean(),
        ]
        np.testing.assert_allclose(
            actual,
            [r[k] for k in ("mse", "bias", "r", "coverage", "width")],
            atol=1e-12,
            rtol=0,
        )
    # Preserve row order when replaying seeded paired resampling.
    for row in pd.read_csv(a.run / "paired_contrasts.csv").to_dict("records"):
        joined = f[f.arm.eq("hybrid_joint")].merge(
            f[f.arm.eq(row["reference"])],
            on=["seed", "author_id"],
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        d = (joined.score_a - joined.truth_a) ** 2 - (
            joined.score_b - joined.truth_b
        ) ** 2
        rng = np.random.default_rng(164)
        boot = np.zeros(2000)
        for seed in protocol["seeds"]:
            x = d[joined.seed.eq(seed)].to_numpy()
            boot += x[rng.integers(0, len(x), (2000, len(x)))].mean(1) / len(
                protocol["seeds"]
            )
        np.testing.assert_allclose(
            [d.mean(), *np.quantile(boot, [0.025, 0.975])],
            [row[k] for k in ("delta_mse", "ci_low", "ci_high")],
            atol=1e-12,
            rtol=0,
        )
    cards = json.loads((a.run / "evidence_cards.json").read_text())
    card_effects = []
    for seed in protocol["seeds"]:
        group = [r for r in cards if r["seed"] == seed]
        lookup = {r["name"]: r for r in group}
        for r in group:
            base_name = (
                "complete_b"
                if r["name"]
                in (
                    "complete_b",
                    "missing_positive",
                    "condition_dependent_quotation",
                    "all_hypothetical",
                )
                else "complete_a"
            )
            card_effects.append(
                {
                    "seed": seed,
                    "card": r["name"],
                    "score": r["score"]["score"],
                    "delta_from_same_author_complete": r["score"]["score"]
                    - lookup[base_name]["score"]["score"],
                    "events": r["score"]["observed_events"],
                    "support": r["score"]["support"],
                }
            )
    pd.DataFrame(card_effects).to_csv(a.output / "card_effects.csv", index=False)
    (a.output / "history_checks.json").write_text(
        json.dumps(history_checks, indent=2, allow_nan=False) + "\n"
    )
    result = {
        "status": "ALTERNATE_T1_TEXT_POSTERIOR_AND_METRICS_REPLAY_PASS",
        "output_hashes": len(manifest),
        "application_posterior_objects": formula_objects,
        "source_reference_objects": 300 * len(protocol["seeds"])
        if not protocol["smoke"]
        else 8,
        "maximum_posterior_error": max(r["max_error"] for r in all_errors),
        "detail": all_errors,
        "metric_rows": len(table),
        "paired_intervals": 5,
        "history_controls": 5 * len(protocol["seeds"]),
        "scope": "same-agent alternate latent enumeration/product likelihood;not independenthuman validation",
        "remaining": "fullstate numericalquadrature convergence notcertifiedby thissame-node replay;see independent scramble checks",
        "seconds": time.monotonic() - began,
    }
    (a.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
