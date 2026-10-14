#!/usr/bin/env python3
"""M09 controlled common-input mechanisms and typed conversational evidence."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.interaction_response import apply_response, estimate_response
from scripts.run_grain_evidence_calibration import infer_pairs
from scripts.run_grain_query_moment_operator import read, sha, write


def public_bank(cfg):
    """Cross topics/wordings with acts; labels grade rather than train the observer."""
    rows = []
    templates = {
        "accept": ["Yes.", "I agree with your suggestion and will try it."],
        "reject": ["No.", "I disagree with that suggestion and do not want to try it."],
        "help": [
            "Could you help me with {topic} now?",
            "Please show me how to handle {topic}.",
        ],
        "repair": [
            "That is not what I meant. I was talking about {topic}, not money.",
            "You misunderstood me. Let me clarify: my problem is {topic}.",
        ],
        "past": [
            "Yesterday I asked my colleague to help with {topic}.",
            "Last week I requested help from a friend with {topic}.",
        ],
        "future": [
            "Tomorrow I will ask my colleague to help with {topic}.",
            "Next week I plan to request help from a friend with {topic}.",
        ],
    }
    for ti, topic in enumerate(cfg["topics"]):
        for act, variants in templates.items():
            context = f"For {topic}, I suggest making a short checklist first. Would you like to try that?"
            if act == "repair":
                context = "It sounds as if your main concern is how much money you will receive."
            elif act in ("past", "future"):
                context = f"What can you tell me about getting help with {topic}?"
            elif act == "help":
                context = f"You mentioned {topic}. What would you like from this conversation?"
            for vi, template in enumerate(variants):
                rows.append(
                    {
                        "case_id": f"topic{ti}:{act}:{vi}",
                        "topic": topic,
                        "act": act,
                        "variant": vi,
                        "partner": context,
                        "reply": template.format(topic=topic),
                    }
                )
    return pd.DataFrame(rows)


def prepare(out, config):
    """Freeze exact prompts, math worlds and existing local observer dependencies."""
    if out.exists():
        raise FileExistsError("Fresh output directory required")
    out.mkdir(parents=True)
    cfg = read(config)
    write(out / "config.json", cfg)
    public_bank(cfg).to_csv(out / "public_bank.csv", index=False)
    paths = [
        Path(__file__),
        config,
        ROOT / "grain_research/interaction_response.py",
        ROOT / "scripts/run_grain_interaction_natural.py",
        ROOT / "scripts/run_grain_evidence_calibration.py",
        ROOT / "scripts/run_grain_entailment_reference.py",
        ROOT / "grain_research/vector_mechanisms/meps_material.py",
        ROOT / "suica_core/meps_ai_conv.py",
        ROOT / "suica_core/meps_layout.py",
        ROOT / "docs/GRAIN_INTERACTION_RESPONSE_PACKET_V1.md",
    ]
    write(
        out / "contract.json",
        {
            "files": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "bank_sha256": sha(out / "public_bank.csv"),
            "config_sha256": sha(out / "config.json"),
            "human_scale_fit": False,
            "private_raw_output": False,
        },
    )


def verify(out, stage):
    """Verify exact source inputs; do not overwrite a running or completed stage."""
    c = read(out / "contract.json")
    for name, digest in c["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    for name, key in (
        ("public_bank.csv", "bank_sha256"),
        ("config.json", "config_sha256"),
    ):
        if sha(out / name) != c[key]:
            raise ValueError(f"Frozen material changed: {name}")
    folder = out / stage
    if folder.exists():
        raise FileExistsError("Inspect existing stage before a separate retry")
    folder.mkdir()
    return read(out / "config.json"), folder


def math(out):
    """Separate conditional response coefficients from endogenous partner inputs."""
    cfg, folder = verify(out, "math")
    start = time.process_time()
    estimates, observations, predictions, fits, boundaries = [], [], [], {}, []
    n = cfg["fit_opportunities"] + cfg["held_opportunities"]
    split = cfg["fit_opportunities"]
    for wi, world in enumerate(cfg["worlds"]):
        for seed in cfg["seeds"]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, wi]))
            for person in range(cfg["authors"]):
                beta = rng.normal([0.0, 0.1, 0.3, 0.1], [0.15, 0.1, 0.12, 0.08])
                z, r = rng.choice([-1.0, 1.0], (2, n))
                s, u, y = np.zeros(n), np.zeros(n), np.zeros(n)
                gamma = 0.0 if world == "exogenous" else 0.9
                pi = 0.08 if world == "weak_assignment" else 1.0
                if world == "no_assignment":
                    pi = 0.0
                delta = 0.3 if world == "direct_assignment" else 0.0
                for j in range(n):
                    s[j] = (
                        cfg["state_ar"] * s[j - 1]
                        + rng.normal(
                            0, cfg["state_sd"] * np.sqrt(1 - cfg["state_ar"] ** 2)
                        )
                        if j
                        else rng.normal(0, cfg["state_sd"])
                    )
                    u[j] = pi * r[j] + gamma * s[j] + rng.normal(0, 0.35)
                    if world == "feedback" and j:
                        u[j] += 0.45 * y[j - 1]
                    if world == "constant_partner":
                        u[j] = 1.0
                    y[j] = (
                        beta @ [1, z[j], u[j], z[j] * u[j]]
                        + s[j]
                        + delta * r[j]
                        + rng.normal(0, cfg["measurement_sd"])
                    )
                assignment = (
                    None if world in ("no_assignment", "constant_partner") else r
                )
                fit = estimate_response(
                    y[:split],
                    z[:split],
                    u[:split],
                    None if assignment is None else r[:split],
                    lags=cfg["hac_lags"],
                )
                key = f"{world}:{seed}:{person}"
                fits[key] = fit
                header = {"world": world, "seed": seed, "person": person}
                boundaries.append(
                    {
                        **header,
                        "status": fit["status"],
                        "minimum_first_stage_f": np.min(fit["first_stage_f"])
                        if "first_stage_f" in fit
                        else np.nan,
                    }
                )
                observations.extend(
                    {
                        **header,
                        "occasion": j,
                        "partition": "fit" if j < split else "held",
                        "condition": z[j],
                        "partner": u[j],
                        "assignment": r[j] if assignment is not None else np.nan,
                        "response": y[j],
                        "truth_state": s[j],
                        "truth_level": beta[0],
                        "truth_condition": beta[1],
                        "truth_partner": beta[2],
                        "truth_condition_partner": beta[3],
                    }
                    for j in range(n)
                )
                if "ols" not in fit:
                    continue
                for arm in ("ols", "iv", "control_function"):
                    if arm != "ols" and "iv" not in fit:
                        continue
                    coefficient = fit["ols"] if arm == "ols" else fit["iv"]
                    cov = fit["ols_cov"] if arm == "ols" else fit["iv_cov"]
                    if arm != "control_function":
                        for j, target in enumerate(
                            ("level", "condition", "partner", "condition_partner")
                        ):
                            estimates.append(
                                {
                                    **header,
                                    "arm": arm,
                                    "target": target,
                                    "score": coefficient[j],
                                    "variance": cov[j, j],
                                    "truth": beta[j],
                                }
                            )
                    held = apply_response(
                        fit,
                        z[split:],
                        u[split:],
                        None if assignment is None else r[split:],
                        arm=arm,
                    )
                    for target, p, truth in (
                        ("current_expression", held["response"], y[split:]),
                        ("common_state_evidence", held["common_input"], s[split:]),
                        ("remaining_state", y[split:] - held["profile"], s[split:]),
                    ):
                        predictions.append(
                            {
                                **header,
                                "arm": arm,
                                "target": target,
                                "mse": np.mean((p - truth) ** 2),
                                "bias": np.mean(p - truth),
                                "mean_common_input": held["common_input"].mean(),
                            }
                        )
                if time.process_time() - start > cfg["math_cpu_budget"]:
                    write(folder / "partial_fits.json", fits)
                    raise TimeoutError("Bounded CPU budget reached")
    df = pd.DataFrame(estimates)
    df.to_csv(folder / "coefficient_predictions.csv", index=False)
    pd.DataFrame(observations).to_csv(folder / "observations.csv", index=False)
    held = pd.DataFrame(predictions)
    held.to_csv(folder / "held_errors.csv", index=False)
    pd.DataFrame(boundaries).to_csv(folder / "identification.csv", index=False)
    write(folder / "fits.json", fits)
    metric = []
    for (world, arm, target), part in df.groupby(["world", "arm", "target"]):
        e = part.score - part.truth
        metric.append(
            {
                "world": world,
                "arm": arm,
                "target": target,
                "n": len(part),
                "rmse": np.sqrt(np.mean(e**2)),
                "bias": e.mean(),
                "median_abs_error": np.median(abs(e)),
                "p90_abs_error": np.quantile(abs(e), 0.9),
                "working95_coverage": np.mean(
                    abs(e) <= 1.96 * np.sqrt(np.maximum(part.variance, 0))
                ),
                "abs_score_gt5": int((part.score.abs() > 5).sum()),
            }
        )
    pd.DataFrame(metric).to_csv(folder / "coefficient_metrics.csv", index=False)
    aggregate = (
        held.groupby(["world", "arm", "target"])
        .agg(n=("mse", "size"), mse=("mse", "mean"), bias=("bias", "mean"))
        .reset_index()
    )
    aggregate["rmse"] = np.sqrt(aggregate.mse)
    aggregate.to_csv(folder / "held_metrics.csv", index=False)
    write(
        folder / "summary.json",
        {
            "stage": "complete",
            "persons": len(fits),
            "observations": len(observations),
            "coefficient_rows": len(df),
            "max_iv_cf_difference": max(
                f.get("iv_cf_max_difference", 0) for f in fits.values()
            ),
            "cpu_seconds": time.process_time() - start,
            "natural_causal_fit": False,
            "assignment_exclusion_verified_in_data": False,
        },
    )


def nli(out):
    """Read current acts and reported history separately, with no observer training."""
    cfg, folder = verify(out, "nli")
    bank = pd.read_csv(out / "public_bank.csv")
    pairs = []
    for row in bank.itertuples():
        scopes = {
            "reply": f"Respondent: {row.reply}",
            "pair": f"Partner: {row.partner}\nRespondent: {row.reply}",
        }
        if row.act in ("accept", "reject"):
            scopes["swapped"] = (
                f"Partner: Would you reject my suggestion about {row.topic}?\nRespondent: {row.reply}"
            )
        for scope, premise in scopes.items():
            expected = (
                {"accept": "reject", "reject": "accept"}[row.act]
                if scope == "swapped" and row.variant == 0
                else row.act
            )
            for query, text in cfg["nli_queries"].items():
                pairs.append(
                    {
                        "audit_id": f"{row.case_id}/{scope}/{query}",
                        "case_id": row.case_id,
                        "scope": scope,
                        "query": query,
                        "designed_act": expected,
                        "sentence": premise,
                        "item_text": text,
                    }
                )
    pairs = pd.DataFrame(pairs)
    start = time.monotonic()
    result = pairs.merge(infer_pairs(pairs), on="audit_id", validate="one_to_one")
    result.to_csv(folder / "public_nli.csv", index=False)
    result["contrast"] = result.p_entailment - result.p_contradiction
    result.groupby(["designed_act", "scope", "query"])[
        ["p_entailment", "contrast"]
    ].mean().reset_index().to_csv(folder / "act_profiles.csv", index=False)
    rows = []
    for (case, scope), part in result[result["query"] != "habit"].groupby(
        ["case_id", "scope"]
    ):
        winner = part.loc[part.p_entailment.idxmax()]
        rows.append(
            {
                "case_id": case,
                "scope": scope,
                "designed_act": winner.designed_act,
                "top_act": winner["query"],
                "top_entailment": winner.p_entailment,
                "top_matches_design": winner.designed_act == winner["query"],
            }
        )
    pd.DataFrame(rows).to_csv(folder / "top_act.csv", index=False)
    write(
        folder / "summary.json",
        {
            "stage": "complete",
            "public_texts": len(bank),
            "nli_pairs": len(result),
            "seconds": time.monotonic() - start,
            "raw_hash": sha(folder / "public_nli.csv"),
            "scope": "Controlled act evidence; NLI numbers are not calibrated probabilities of a human behavior or trait",
        },
    )


def main():
    """Stage-local execution with preserved raw outputs and no implicit restarts."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "math", "nli"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_interaction_response.json"
    )
    a = p.parse_args()
    with threadpool_limits(limits=1):
        prepare(a.output, a.config) if a.stage == "prepare" else globals()[a.stage](
            a.output
        )
    print(f"Completed {a.stage}: {a.output}", flush=True)


if __name__ == "__main__":
    main()
