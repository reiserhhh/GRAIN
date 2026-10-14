#!/usr/bin/env python3
"""Construct and evaluate a calibrated text observation law for the original T1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3 import f16_t1_text as model
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from grain_research.gptpro_foundation_v3.f16_t1_source import (
    verify_bundle as verify_numeric,
)
from scripts.run_grain_gptpro_f16_t1_source import group


def save(path, value):
    """Checkpoint strict JSON without nonfinite placeholders."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def material_bank(base):
    """Join identical texts; preserve truth only in experiment-side material table."""
    nli = pd.read_csv(base / "F08/nli_reader_r1/reader_scores.csv").set_index(
        "material_id"
    )
    grain = pd.read_csv(
        base / "F11/grain_comment_reader_r1/reader_scores.csv"
    ).set_index("material_id")
    identity = [
        "text",
        "role",
        "family",
        "pole",
        "qualification",
        "subject_status",
        "variant",
    ]
    if not nli[identity].equals(grain[identity]):
        raise ValueError("exact material identity and splits required")
    result = nli[identity].copy()
    result["pole_readout"] = nli.pole_score
    result["nli_qualification"] = nli.qualification_score
    result["grain_qualification"] = grain.qualification_score
    return result.reset_index()


def render(groups, bank, role, seed):
    """Render archived continuous events; latent values never enter scorer inputs."""
    pools = {
        (b, q): bank[
            (bank.role == role) & (bank.pole == b) & (bank.qualification == q)
        ].to_dict("records")
        for b in (0, 1)
        for q in (0, 1)
    }
    if any(not p for p in pools.values()):
        raise ValueError("missing role-specific rendering stratum")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 162, 0 if role == "source" else 1])
    )
    inputs, evaluator = [], []
    for author in sorted(groups):
        history = []
        for row in groups[author]:
            eligible = int(rng.integers(2))
            bit = int(row["y"] >= 0) if eligible else int(rng.integers(2))
            choices = pools[bit, eligible]
            text = choices[int(rng.integers(len(choices)))]
            observed = {k: row[k] for k in ("author_id", "event_id", "t", "z")}
            observed["material_id"] = text["material_id"]
            history.append(
                {
                    "record_id": row["event_id"],
                    "op": "new",
                    "recorded_at": row["t"],
                    "observation": observed,
                }
            )
            evaluator.append(
                {
                    **row,
                    **{
                        k: text[k]
                        for k in (
                            "material_id",
                            "pole_readout",
                            "nli_qualification",
                            "grain_qualification",
                        )
                    },
                    "qualification": eligible,
                    "rendered_pole": bit,
                }
            )
        inputs.append(
            {
                "author_id": author,
                "target_time": 31,
                "history": history,
                "world_contract": "F04_L_thresholded_text_eligibility_v1",
            }
        )
    return inputs, evaluator


def write_inputs(path, inputs):
    """Save observation-only input independently of the evaluation truth file."""
    with path.open("x") as out:
        for raw in inputs:
            out.write(json.dumps(raw, allow_nan=False) + "\n")


def cold_run(directory, bundle_path, inputs, name):
    """Call a separate process with only bundle, inputs and authenticated code."""
    source = directory / f"{name}_inputs.jsonl"
    output = directory / f"{name}_cold_scores.jsonl"
    write_inputs(source, inputs)
    with (directory / f"{name}_cold.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grain_research.gptpro_foundation_v3.f16_t1_text",
                "--bundle",
                str(bundle_path.resolve()),
                "--sha256",
                sha256_file(bundle_path),
                "--input",
                str(source.resolve()),
                "--output",
                str(output.resolve()),
            ],
            cwd=directory,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return [json.loads(line) for line in output.read_text().splitlines()]


def numeric_difference(left, right):
    """Check complete structures, tolerating only floating point number differences."""
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("cold object fields differ")
        return max((numeric_difference(left[k], right[k]) for k in left), default=0.0)
    if isinstance(left, list):
        if len(left) != len(right):
            raise ValueError("cold object sizes differ")
        return max(
            (numeric_difference(a, b) for a, b in zip(left, right, strict=True)),
            default=0.0,
        )
    if type(left) in (float, int) and type(right) in (float, int):
        return abs(left - right)
    if left != right:
        raise ValueError("cold object values differ")
    return 0.0


def evidence_cards(inputs, bank, bundle):
    """Eight actual text histories plus copy/version controls, not prose placeholders."""
    a, b = copy.deepcopy(inputs[:2])
    cards = []
    for name, raw in [("complete_a", a), ("complete_b", b)]:
        cards.append((name, copy.deepcopy(raw)))
    for name, raw, omit in [
        ("missing_negative", a, -1.0),
        ("missing_positive", b, 1.0),
    ]:
        item = copy.deepcopy(raw)
        item["history"] = [
            op for op in item["history"] if op["observation"]["z"] != omit
        ]
        cards.append((name, item))
    selected = copy.deepcopy(a)
    selected["history"] = [
        op
        for op in selected["history"]
        if bundle["materials"][op["observation"]["material_id"]]["pole_readout"] >= 0.5
    ]
    cards.append(("positive_report_selection", selected))
    for name, raw, status, condition_only in [
        ("condition_dependent_quotation", b, "quoted", True),
        ("all_quoted", a, "quoted", False),
        ("all_hypothetical", b, "hypothetical", False),
    ]:
        item = copy.deepcopy(raw)
        for op in item["history"]:
            row = op["observation"]
            if condition_only and row["z"] < 0:
                continue
            old = bank[bank.material_id.eq(row["material_id"])].iloc[0]
            match = bank[
                (bank.role == "application")
                & (bank.pole == old.pole)
                & (bank.subject_status == status)
                & (bank.variant == old.variant)
            ]
            if len(match) != 1:
                raise ValueError("counterfactual material pairing failed")
            row["material_id"] = match.iloc[0].material_id
        cards.append((name, item))
    return cards


def metrics(frame):
    """Summarize the declared continuous contrast, not classification accuracy."""
    result = []
    for (seed, arm), rows in frame.groupby(["seed", "arm"]):
        y, p = rows.truth.to_numpy(), rows.score.to_numpy()
        r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 1e-12 else None
        result.append(
            {
                "seed": int(seed),
                "arm": arm,
                "authors": len(rows),
                "mse": float(np.mean((p - y) ** 2)),
                "bias": float(np.mean(p - y)),
                "r": r,
                "coverage": float(np.mean((rows.lower <= y) & (y <= rows.upper))),
                "width": float(np.mean(rows.upper - rows.lower)),
            }
        )
    return pd.DataFrame(result)


def main():
    """Freeze all source constructs before application; run six fixed comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    began = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    base = ROOT / "results/grain_gptpro_foundation_v3"
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    code = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f16_t1_text.py",
        ROOT / "tests/test_grain_gptpro_f16_t1_text.py",
        ROOT / "docs/GRAIN_F16_T1_TEXT_OBSERVATION_PROTOCOL_V1.md",
    ]
    sources = [
        base / "F08/nli_reader_r1/reader_scores.csv",
        base / "F11/grain_comment_reader_r1/reader_scores.csv",
    ]
    for seed in seeds:
        sources += [
            base / f"F16/t1_source_full_r1/{seed}/bundle.json",
            base / f"F04/source_full_r1/L_{seed}/source.jsonl",
        ]
    save(
        args.output / "protocol_before_run.json",
        {
            "source_hashes": {str(p): sha256_file(p) for p in sources},
            "code_hashes": {str(p): sha256_file(p) for p in code},
            "seeds": seeds,
            "source_models": 4 if args.smoke else 200,
            "qmc_power": 8 if args.smoke else 10,
            "reader_calibration": "32validation materials disjoint from original32source Ridge fitting;Dirichlet counts+.5",
            "rendering": "source/application frames kept separate;QiidBernoulli.5;eligibleB=signY,otherwiseindependentfairB",
            "arms": [*model.ARMS, "naive_gaussian", "oracle", "eligible_continuous"],
            "smoke": args.smoke,
            "new_neural_calls": 0,
            "old_source_refit": False,
            "scope": "TEXT_SYNTHETIC observation law on archivedF04L;not humanpersonalityaccuracy",
            "remaining": "T2/T3 observationlaw,transport,fullparentcontracts and44+6goal",
        },
    )
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for p in code:
        (snapshot / p.name).write_bytes(p.read_bytes())
    bank = material_bank(base)
    calibration = bank[bank.role.eq("validation")].to_dict("records")
    material_cache = {}
    for row in bank.to_dict("records"):
        item = {
            k: row[k]
            for k in (
                "text",
                "pole_readout",
                "nli_qualification",
                "grain_qualification",
            )
        }
        item["text_sha256"] = hashlib.sha256(item["text"].encode()).hexdigest()
        material_cache[row["material_id"]] = item
    prepared = []
    for seed in seeds:
        directory = args.output / str(seed)
        directory.mkdir()
        old = json.loads(
            (base / f"F16/t1_source_full_r1/{seed}/bundle.json").read_text()
        )
        verify_numeric(old)
        n_models = 4 if args.smoke else 200
        counts = {
            arm: model.channel_counts(calibration, arm)
            for arm in ("hybrid_joint", "nli_joint")
        }
        bundle = {
            "version": model.VERSION,
            "target_id": "T1",
            "object_class": "conditional_response_function",
            "unit": "archived numeric source endpoint contrast;not identified by binarytextalone",
            "world_contract": "F04_L_thresholded_text_eligibility_v1",
            "seed": seed,
            "scorer_code_hash": model.code_hash(),
            "qmc_power": 8 if args.smoke else 10,
            "source_components": n_models,
            "source_models": old["source_models"][:n_models],
            "channel_counts": {k: v.tolist() for k, v in counts.items()},
            "channel_draws": {
                k: model.channel_draws(
                    v, np.random.SeedSequence([seed, 163, i]), n_models
                ).tolist()
                for i, (k, v) in enumerate(counts.items())
            },
            "qualification_rate": 0.5,
            "reference_mu": 0.0,
            "reference_sigma": 1.0,
            "materials": material_cache,
            "source_bundle_hash": sha256_file(
                base / f"F16/t1_source_full_r1/{seed}/bundle.json"
            ),
            "support_rule": "observed text endpoints;eligibility latent underdeclaredchannel;missing conditions borrowed",
            "error_model": "jointcategoricalreader+probit;equalcut source/reader posterior mixture;QMC;momentnormalinterval",
            "invalidating_result": "unmodeled selection,wrong noiseunit or reader shift changes target law",
            "explanation_rule": "trace material/category/eligibility likelihood/history/target;no psychological naming",
            "calibration": "NOT_EVALUATED",
        }
        source = group(base / f"F04/source_full_r1/L_{seed}/source.jsonl", "source")
        if len(source) != 300:
            raise ValueError("source coverage mismatch")
        if args.smoke:
            source = dict(sorted(source.items())[:8])
        inputs, evaluator = render(source, bank, "source", seed)
        source_scores = model.cold_scores(inputs, bundle)
        v = np.array([x["score"] for x in source_scores])
        bundle["reference_mu"] = float(v.mean())
        bundle["reference_sigma"] = float(v.std(ddof=1))
        bundle["reference_authors"] = len(source)
        for score in source_scores:
            score["reference_score"] = (
                score["score"] - bundle["reference_mu"]
            ) / bundle["reference_sigma"]
            score["reference_variance"] = (
                score["variance"] / bundle["reference_sigma"] ** 2
            )
        save(directory / "bundle.json", bundle)
        save(directory / "source_scores.json", source_scores)
        write_inputs(directory / "source_inputs.jsonl", inputs)
        save(directory / "source_rendering_evaluator.json", evaluator)
        prepared.append((seed, directory, bundle))
        print(
            json.dumps(
                {"stage": "source_frozen", "seed": seed, "authors": len(source)}
            ),
            flush=True,
        )
    rows, audits, qmc_checks, cards_out = [], [], [], []
    for seed, directory, bundle in prepared:
        archive = base / f"F04/source_full_r1/L_{seed}"
        app = group(archive / "application.jsonl", "application")
        if len(app) != 200:
            raise ValueError("application coverage mismatch")
        if args.smoke:
            app = dict(sorted(app.items())[:4])
        inputs, evaluator = render(app, bank, "application", seed)
        save(directory / "application_rendering_evaluator.json", evaluator)
        save(
            directory / "application_hashes.json",
            {
                "observations": sha256_file(archive / "application.jsonl"),
                "truth": sha256_file(archive / "truth.json"),
            },
        )
        counts = {
            arm: np.stack(
                [model.counts_from_input(raw, bundle, arm)[0] for raw in inputs]
            )
            for arm in model.ARMS
        }
        warm = model.cold_scores(inputs, bundle)
        cold = cold_run(directory, directory / "bundle.json", inputs, "application")
        error = numeric_difference(warm, cold)
        if error > 1e-10:
            raise ValueError("cold/warm mismatch")
        arms = {"hybrid_joint": warm}
        for arm in ("nli_joint", "hybrid_factorized"):
            arms[arm] = model.score_counts(counts[arm], bundle, arm)
            print(
                json.dumps({"stage": "application_arm", "seed": seed, "arm": arm}),
                flush=True,
            )
        ev_by = {raw["author_id"]: [] for raw in inputs}
        for row in evaluator:
            ev_by[row["author_id"]].append(row)
        oracle_counts = np.zeros_like(counts["hybrid_joint"])
        for i, raw in enumerate(inputs):
            for row in ev_by[raw["author_id"]]:
                oracle_counts[
                    i,
                    model.CONDITIONS.index(row["z"]),
                    2 * row["rendered_pole"] + row["qualification"],
                ] += 1
        arms["oracle"] = model.score_counts(oracle_counts, bundle, "oracle")
        for arm, naive in [("naive_gaussian", True), ("eligible_continuous", False)]:
            arms[arm] = [
                model.numeric_comparator(ev_by[raw["author_id"]], bundle, naive)
                for raw in inputs
            ]
        np.savez_compressed(directory / "counts.npz", **counts, oracle=oracle_counts)
        save(directory / "arm_scores.json", arms)
        # Truth is opened only after scoring every arm. It never enters calibration.
        truth = {
            r["author_id"]: float(r["means"][-1] - r["means"][0])
            for r in json.loads((archive / "truth.json").read_text())
            if r["role"] == "application"
        }
        for arm, scored in arms.items():
            for raw, score in zip(inputs, scored, strict=True):
                rows.append(
                    {
                        "seed": seed,
                        "author_id": raw["author_id"],
                        "arm": arm,
                        "truth": truth[raw["author_id"]],
                        **{
                            k: score[k] for k in ("score", "variance", "lower", "upper")
                        },
                    }
                )
        audit_counts = counts["hybrid_joint"][:12]
        second = model.score_counts(audit_counts, bundle, scramble=1)
        delta = max(abs(x["score"] - y["score"]) for x, y in zip(warm, second))
        sd_delta = max(
            abs(np.sqrt(x["variance"]) - np.sqrt(y["variance"]))
            for x, y in zip(warm, second)
        )
        check = {
            "seed": seed,
            "authors": len(second),
            "power": bundle["qmc_power"],
            "score_diff": delta,
            "sd_diff": sd_delta,
        }
        if max(delta, sd_delta) > 0.02 and not args.smoke:
            fine = copy.deepcopy(bundle)
            fine["qmc_power"] = 12
            f0 = model.score_counts(audit_counts, fine)
            f1 = model.score_counts(audit_counts, fine, scramble=1)
            check["fine_score_diff"] = max(
                abs(a["score"] - b["score"]) for a, b in zip(f0, f1, strict=True)
            )
            check["fine_sd_diff"] = max(
                abs(np.sqrt(a["variance"]) - np.sqrt(b["variance"]))
                for a, b in zip(f0, f1, strict=True)
            )
            save(
                directory / "fine_qmc_comparison.json",
                {"scramble0": f0, "scramble1": f1},
            )
        qmc_checks.append(check)
        save(directory / "qmc_check.json", check)
        audits.append(
            {
                "seed": seed,
                "authors": len(inputs),
                "cold_warm_max_error": error,
                "min_ess": min(x["qmc_ess_min"] for x in warm),
                "median_ess": float(np.median([x["qmc_ess_median"] for x in warm])),
            }
        )
        # One fixed pair of authors per seed; no selection by score or latent truth.
        cards = evidence_cards(inputs, bank, bundle)
        card_inputs = [raw for _, raw in cards]
        scored = model.cold_scores(card_inputs, bundle)
        cold_cards = cold_run(
            directory, directory / "bundle.json", card_inputs, "cards"
        )
        if numeric_difference(scored, cold_cards) > 1e-10:
            raise ValueError("card cold mismatch")
        for (name, raw), score in zip(cards, scored, strict=True):
            cards_out.append(
                {
                    "seed": seed,
                    "name": name,
                    "author_id": raw["author_id"],
                    "input": raw,
                    "score": score,
                }
            )
        # A change of units leaves all observable text probabilities identical.
        scaled = copy.deepcopy(bundle)
        for item in scaled["source_models"]:
            item["mu"] = (2 * np.asarray(item["mu"])).tolist()
            item["Sigma"] = (4 * np.asarray(item["Sigma"])).tolist()
            item["sigma2"] *= 4
        scaled_scores = model.score_counts(audit_counts[:2], scaled)
        scale_error = max(
            abs(s["score"] - 2 * r["score"]) for s, r in zip(scaled_scores, warm)
        )
        variance_error = max(
            abs(s["variance"] - 4 * r["variance"]) for s, r in zip(scaled_scores, warm)
        )
        if max(scale_error, variance_error) > 1e-9:
            raise ValueError("scale equivariance failure")
        audits[-1].update(
            {"scale_score_error": scale_error, "scale_variance_error": variance_error}
        )
        save(directory / "audit_summary.json", audits[-1])
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "scores.csv", index=False)
    table = metrics(frame)
    table.to_csv(args.output / "metrics_by_seed.csv", index=False)
    table.groupby("arm")[["mse", "bias", "r", "coverage", "width"]].mean().to_csv(
        args.output / "metrics.csv"
    )
    save(args.output / "evidence_cards.json", cards_out)
    contrasts = []
    for arm in [
        "nli_joint",
        "hybrid_factorized",
        "naive_gaussian",
        "oracle",
        "eligible_continuous",
    ]:
        p = frame[frame.arm.eq("hybrid_joint")].merge(
            frame[frame.arm.eq(arm)],
            on=["seed", "author_id"],
            suffixes=("_primary", "_ref"),
            validate="one_to_one",
        )
        d = (p.score_primary - p.truth_primary) ** 2 - (p.score_ref - p.truth_ref) ** 2
        rng = np.random.default_rng(164)
        boot = np.zeros(2000)
        for seed in seeds:
            x = d[p.seed.eq(seed)].to_numpy()
            boot += x[rng.integers(0, len(x), (2000, len(x)))].mean(1) / len(seeds)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        contrasts.append(
            {
                "reference": arm,
                "authors": len(p),
                "delta_mse": float(d.mean()),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )
    pd.DataFrame(contrasts).to_csv(args.output / "paired_contrasts.csv", index=False)
    save(
        args.output / "summary.json",
        {
            "status": "T1_TEXT_OBSERVATION_LAW_EXECUTED",
            "smoke": args.smoke,
            "application_authors": len(frame) // 6,
            "arms": 6,
            "cards": len(cards_out),
            "audits": audits,
            "qmc_checks": qmc_checks,
            "new_neural_calls": 0,
            "scope": "calibrated controlled-text construction,not natural/human validity",
            "F16_parent_complete": False,
            "remaining": "T2/T3 text observation,matched transport,full protocol acceptance",
            "seconds": time.monotonic() - began,
        },
    )
    save(
        args.output / "output_manifest.json",
        {
            str(p.relative_to(args.output)): sha256_file(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file() and not p.name.startswith("._")
        },
    )
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
