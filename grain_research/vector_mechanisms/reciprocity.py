"""Synthetic, disjoint-scenario persona expression/recognition interventions.

These materials operationalize social initiation, not the entire Extraversion
construct. No human records or personality labels enter this module.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from .analysis import grouped_bootstrap
from .controls import material_hash

SYSTEM = "Follow the task using only the supplied profile or account."
SCENARIOS = (
    ("development", "orientation", "an orientation meeting for new volunteers"),
    ("development", "workshop", "a weekend pottery workshop"),
    ("development", "bookclub", "the first meeting of a book club"),
    ("development", "walking", "a neighborhood walking group"),
    ("development", "language", "a language exchange evening"),
    ("development", "museum", "a small museum tour"),
    ("evaluation", "garden", "a community gardening afternoon"),
    ("evaluation", "choir", "a rehearsal with a new choir"),
    ("evaluation", "conference", "a break at a professional conference"),
    ("evaluation", "cooking", "a group cooking class"),
    ("evaluation", "games", "a board game evening"),
    ("evaluation", "exhibition", "the opening of a local art exhibition"),
)
PROFILES = {
    "trait": {
        1: "The person is socially outgoing and extraverted.",
        -1: "The person is socially reserved and introverted.",
    },
    "behavior": {
        1: "The person generally prefers to make the first move when meeting others.",
        -1: "The person generally prefers others to make the first move when they meet.",
    },
}
DEVELOPMENT_ACTIONS = {
    1: "I introduced myself to people I did not know and began talking with them.",
    -1: "I stayed quiet around people I did not know until they began talking with me.",
}
EVALUATION_ACTIONS = {
    1: "I went over to an unfamiliar participant and got a conversation going.",
    -1: "I kept to myself until another participant came over and spoke to me.",
}
GENERATION_ANSWERS = (
    "I approach someone and start a conversation.",
    "I wait quietly until someone approaches me.",
)


def build_material(*, smoke: bool = False) -> list[dict]:
    """Build paired task variants with disjoint development/evaluation settings."""
    rows = []
    for split, group, situation in SCENARIOS:
        if smoke and group not in {"orientation", "garden", "choir"}:
            continue
        actions = DEVELOPMENT_ACTIONS if split == "development" else EVALUATION_ACTIONS
        variants = (
            ("G_trait", "R_direct")
            if split == "development"
            else (
                "G_trait",
                "G_behavior",
                "R_direct",
                "R_negative",
                "R_trait",
                "R_trait_negative",
                "R_owner",
                "R_fact",
            )
        )
        for variant in variants:
            for polarity in (-1, 1):
                task = variant[0]
                if task == "G":
                    profile = PROFILES[variant.split("_")[1]][polarity]
                    evidence = (
                        f"Profile: {profile}\n"
                        f"Situation: The person is attending {situation}. "
                        "Several unfamiliar people are nearby."
                    )
                    question = (
                        "Speak as the person in the profile. Write one short first-person "
                        "sentence about what you do next."
                    )
                    answers = GENERATION_ANSWERS
                    expected = polarity
                else:
                    first, other = actions[polarity], actions[-polarity]
                    if variant == "R_owner":
                        # Swapping clause ownership preserves the bag of action words.
                        evidence = (
                            f"The writer attended {situation}. The writer's account: "
                            f"\"My own experience: {first} My companion's experience, "
                            f'quoted verbatim: {other} End of account." The venue was open.'
                        )
                    else:
                        evidence = (
                            f"The writer attended {situation}. The writer's account: "
                            f'"{first}" The venue was open.'
                        )
                    question = (
                        "In this account, did the writer personally initiate social contact "
                        "rather than wait for another person to do so? Answer Yes or No."
                    )
                    answers, expected = ("Yes", "No"), polarity
                    if variant == "R_negative":
                        question = (
                            "In this account, did the writer wait for another person to "
                            "initiate social contact rather than initiate it personally? "
                            "Answer Yes or No."
                        )
                        answers = ("No", "Yes")
                    elif variant == "R_trait":
                        question = (
                            "Based only on this account, does the writer seem more "
                            "extraverted than introverted? Answer Yes or No."
                        )
                    elif variant == "R_trait_negative":
                        question = (
                            "Based only on this account, does the writer seem more "
                            "introverted than extraverted? Answer Yes or No."
                        )
                        answers = ("No", "Yes")
                    elif variant == "R_fact":
                        question = "According to the account, was the venue open? Answer Yes or No."
                        expected = 1
                rows.append(
                    {
                        "case_id": f"{group}_{variant}_{'high' if polarity == 1 else 'low'}",
                        "group_id": group,
                        "split": split,
                        "task": task,
                        "variant": variant,
                        "polarity": polarity,
                        "expected_sign": expected,
                        "evidence": evidence,
                        "user": f"{evidence}\n\nTask: {question}",
                        "high_answer": answers[0],
                        "low_answer": answers[1],
                    }
                )
    validate_material(rows)
    return rows


def validate_material(rows: list[dict]) -> None:
    """Reject duplicate IDs, split overlap, or incomplete counterfactual pairs."""
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate case ID")
    groups = defaultdict(set)
    pairs = defaultdict(list)
    for row in rows:
        groups[row["split"]].add(row["group_id"])
        pairs[row["split"], row["group_id"], row["variant"]].append(row)
        if row["high_answer"] == row["low_answer"]:
            raise ValueError("Identical answer candidates")
    if groups["development"] & groups["evaluation"]:
        raise ValueError("Scenario split leakage")
    for pair in pairs.values():
        if len(pair) != 2 or {r["polarity"] for r in pair} != {-1, 1}:
            raise ValueError("Incomplete polarity pair")


def fit_directions(material: list[dict], states: dict, sites: list[str]) -> dict:
    """Average matched high-minus-low development activations, without eval fit."""
    lookup = {
        (r["group_id"], r["task"], r["polarity"]): r
        for r in material
        if r["split"] == "development"
    }
    result = {}
    for source in ("G", "R"):
        groups = sorted({g for g, task, _ in lookup if task == source})
        if not groups:
            raise ValueError("Missing development task")
        for site in sites:
            deltas = np.stack(
                [
                    states[lookup[g, source, 1]["case_id"]][site][0].astype(float)
                    - states[lookup[g, source, -1]["case_id"]][site][0].astype(float)
                    for g in groups
                ]
            )
            direction = deltas.mean(axis=0)
            if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-10:
                raise ValueError("Degenerate development direction")
            result[f"{source}__{site}"] = direction
    return result


def random_direction(direction: np.ndarray, key: str) -> np.ndarray:
    """Generate a fixed seeded isotropic control with exactly matched L2 norm."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    value = np.random.default_rng(seed).normal(size=direction.shape)
    return value * (np.linalg.norm(direction) / np.linalg.norm(value))


def arm_schedule(doses: list[float], random_replicates: int) -> list[tuple]:
    """Freeze signed signal doses and paired random controls; no target selection."""
    return [("signal", -1, dose) for dose in doses] + [
        ("random", replica, dose)
        for replica in range(random_replicates)
        for dose in (-1.0, 1.0)
    ]


def analysis_tables(baselines: pd.DataFrame, effects: pd.DataFrame) -> tuple:
    """Report native separation and paired intervention effects by scenario.

    Confidence intervals are pointwise, exploratory scenario bootstraps. They are
    not author-population intervals or family-wise corrected discovery claims.
    """
    capabilities = []
    for (split, variant), sub in baselines.groupby(["split", "variant"]):
        paired = sub.pivot(index="group_id", columns="polarity", values="margin")
        gap = paired[1] - paired[-1]
        ci = grouped_bootstrap(gap.to_numpy(), gap.index)
        correct = (sub.margin.to_numpy() * sub.expected_sign.to_numpy()) > 0
        capabilities.append(
            {
                "split": split,
                "variant": variant,
                "n_cases": len(sub),
                "n_scenarios": len(paired),
                "accuracy": float(correct.mean()),
                "positive_pair_fraction": float((gap > 0).mean()),
                "high_minus_low_gap": ci["mean"],
                "gap_ci_low": ci["ci_low"],
                "gap_ci_high": ci["ci_high"],
                "is_fact_control": variant == "R_fact",
            }
        )
    if effects.empty:
        return pd.DataFrame(capabilities), pd.DataFrame()
    required = {-1.0, 1.0}
    records = []
    keys = ["case_id", "group_id", "variant", "source", "site", "kind", "replica"]
    for key, sub in effects.groupby(keys):
        if not required <= set(sub.dose):
            continue
        scores = sub.set_index("dose").margin
        records.append(
            dict(zip(keys, key)) | {"slope": float((scores[1] - scores[-1]) / 2)}
        )
    slopes = pd.DataFrame(records)
    summaries = []
    for key, sub in slopes.groupby(["variant", "source", "site"]):
        variant, source, site = key
        signal = sub[sub.kind == "signal"].set_index("case_id")
        controls = sub[sub.kind == "random"].groupby("case_id").slope.mean()
        if set(signal.index) != set(controls.index):
            raise ValueError("Unpaired signal/random controls")
        delta = signal.slope - controls.reindex(signal.index)
        ci = grouped_bootstrap(signal.slope.to_numpy(), signal.group_id)
        paired_ci = grouped_bootstrap(delta.to_numpy(), signal.group_id)
        summaries.append(
            {
                "variant": variant,
                "source": source,
                "site": site,
                "n_scenarios": ci["clusters"],
                "signal_slope": ci["mean"],
                "signal_ci_low": ci["ci_low"],
                "signal_ci_high": ci["ci_high"],
                "random_slope": float(controls.mean()),
                "signal_minus_random": paired_ci["mean"],
                "paired_ci_low": paired_ci["ci_low"],
                "paired_ci_high": paired_ci["ci_high"],
                "positive_scenario_fraction": float(
                    (signal.groupby("group_id").slope.mean() > 0).mean()
                ),
            }
        )
    return pd.DataFrame(capabilities), pd.DataFrame(summaries)


def weighted_effect_summary(effects, weights) -> dict:
    """Aggregate already-author-averaged effects; expose heterogeneity and ESS.

    This numerical primitive does not establish GRAIN similarity or align model
    coordinates. Callers must provide disjoint-view, independently frozen weights.
    """
    effects, weights = np.asarray(effects, float), np.asarray(weights, float)
    if (
        effects.ndim != 2
        or weights.shape != (len(effects),)
        or not np.isfinite(effects).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid author effect matrix or weights")
    weights = weights / weights.sum()
    mean = weights @ effects
    residual = effects - mean
    return {
        "mean": mean,
        "population_covariance": (residual * weights[:, None]).T @ residual,
        "positive_weight_fraction": weights @ (effects > 0),
        "effective_authors": float(1 / (weights @ weights)),
    }


def material_identity(rows: list[dict]) -> str:
    """Bind exact synthetic prompts, splits and answer candidates to an identity."""
    return material_hash({"system": SYSTEM, "rows": rows})
