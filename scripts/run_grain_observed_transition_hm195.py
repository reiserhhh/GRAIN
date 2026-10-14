#!/usr/bin/env python3
"""One fixed observed-continuation mechanism experiment, not predictor tuning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observed_transition as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_observed_transition_hm195.json"
DEFAULT = ROOT / "results/grain_math_response_v1/observed_transition_target_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_OBSERVED_TRANSITION_PLAN_HM195.md",
    "grain_research/construction/math_observed_transition.py",
    "scripts/run_grain_observed_transition_hm195.py",
    "scripts/score_grain_observed_transition.py",
    "tests/test_grain_observed_transition.py",
    "grain_research/construction/math_observed_family.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def prepare(out):
    """Load only SOURCE numeric events, inherited profiles and frozen units."""
    cfg = R.read(ROOT / CONFIG)
    parent, event = ROOT / cfg["source_parent"], ROOT / cfg["event_parent"]
    R.check(parent, ("computed_seal.json", "verification_seal.json"))
    R.check(event, ("compiled_seal.json",))
    with np.load(parent / "observations.npz", allow_pickle=False) as z:
        source, codes, folds, prior = (
            z[k] for k in ("source", "codes", "folds", "prior")
        )
    if (
        len(source) != cfg["source_authors"]
        or np.bincount(folds).tolist() != cfg["banks"]
    ):
        raise ValueError("Changed SOURCE authors or frozen banks")
    with np.load(event / "observations.npz", allow_pickle=False) as z:
        obs = {
            k: z[k][source]
            for k in (
                "coordinates",
                "conditions",
                "times",
                "venue_codes",
                "original_units",
            )
        }
        pairs = z["pairs"]
    cut = R.read(event / "source_law.json")["thresholds"]
    st = M.F.states(obs["coordinates"].reshape(-1, 15), cut).reshape(len(source), 36)
    current, outcome, units, stamps = [], [], [], []
    for start, stop in cfg["windows"]:
        ts = np.arange(start, stop - 1)
        if np.any(obs["times"][:, ts] >= obs["times"][:, ts + 1]):
            raise ValueError(
                "A registered chronological pair is missing; scope cannot shrink"
            )
        current.append(4 * obs["conditions"][:, ts] + st[:, ts])
        outcome.append(
            4 * (obs["venue_codes"][:, ts] == obs["venue_codes"][:, ts + 1])
            + st[:, ts + 1]
        )
        units.append(
            np.stack(
                [obs["original_units"][:, ts], obs["original_units"][:, ts + 1]], -1
            )
        )
        stamps.append(np.stack([obs["times"][:, ts], obs["times"][:, ts + 1]], -1))
    current, outcome = np.stack(current, 1), np.stack(outcome, 1)
    wanted = {
        (int(u), w, t, t + 1)
        for u in source
        for w, (start, stop) in enumerate(cfg["windows"])
        for t in range(start, stop - 1)
    }
    observed = {tuple(row) for row in pairs[np.isin(pairs[:, 0], source)].tolist()}
    if wanted != observed:
        raise ValueError(
            "Source transitions disagree with the original recorded family"
        )
    p = M.frequencies(current, outcome)
    files = [
        parent / k
        for k in (
            "manifest.json",
            "observations.npz",
            "computed_seal.json",
            "verification_seal.json",
        )
    ]
    files += [
        event / k
        for k in (
            "manifest.json",
            "observations.npz",
            "source_law.json",
            "prepared_seal.json",
            "compiled_seal.json",
        )
    ]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "source_codes_only_no_application_labels": True,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        source=source,
        codes=codes,
        folds=folds,
        prior=prior,
        current=current,
        outcome=outcome,
        units=np.stack(units, 1),
        timestamps=np.stack(stamps, 1),
        probabilities=p,
        reference=p.mean((0, 1)),
    )
    author, start = int(source[0]), cfg["windows"][0][0]
    record = {
        "regime": M.REGIME,
        "representation_id": R.digest(event / "source_law.json"),
    }
    for name, j in (("current", start), ("next", start + 1)):
        record[name] = {
            "author_key": f"historical-source-{author}",
            "event_id": f"author-{author}:comment-{int(obs['original_units'][0, j])}",
            "timestamp": float(obs["times"][0, j]),
            "venue_code": int(obs["venue_codes"][0, j]),
            "condition": int(obs["conditions"][0, j]),
            "coordinates": obs["coordinates"][0, j].tolist(),
        }
    R.write(out / "example_transition.json", record)
    R.write(
        out / "input_quality.json",
        {
            "source_authors": len(source),
            "banks": np.bincount(folds).tolist(),
            "profiles": 32,
            "directed_observed_transitions": int(current.size),
            "distinct_vector_events": len(source) * 36,
            "adjacent_pairs_share_records": True,
            "matches_registered_HM170_pairs": True,
            "frozen_thresholds": cut,
            "representation_id": record["representation_id"],
            "source_author_min_profile_per_bank": min(
                int(np.bincount(codes[folds == b], minlength=32).min()) for b in (0, 1)
            ),
            "next_is_next_cached_record_not_controlled_behavior": True,
            "raw_private_application_labels_neural_API_or_training": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "example_transition.json", "input_quality.json"),
    )
    print(json.dumps(R.read(out / "input_quality.json")))


def matching(joint, screened):
    """Check declared preserved margins without selecting a favorable outcome."""
    jf, sf = M.target_fields(joint), M.target_fields(screened)
    return {
        "profile_current_error": float(
            np.max(np.abs(joint.sum(-1) - screened.sum(-1)))
        ),
        "current_outcome_error": float(
            np.max(np.abs(joint.sum(-3) - screened.sum(-3)))
        ),
        "singleton_target_reading_error": float(
            np.nanmax(np.abs(jf["before"] - sf["before"]))
        ),
        "screened_increment_error": float(np.nanmax(np.abs(sf["increment"]))),
    }


def compute(out):
    """Run matched laws and the fixed whole-profile exchange reference once."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    start = time.perf_counter()
    j, screened = M.compile_laws(z["probabilities"], z["codes"], z["folds"], z["prior"])
    fields = M.target_fields(j)
    stat = M.agreement(fields["increment"], fields["supported"], z["reference"])
    rng = np.random.default_rng(cfg["null_seed"])
    permutations, null_values, null_mass, null_available = [], [], [], []
    for _ in range(cfg["null_count"]):
        ix = np.arange(len(z["source"]))
        for b in (0, 1):
            group = np.flatnonzero(z["folds"] == b)
            ix[group] = rng.permutation(group)
        nj, _ = M.compile_laws(
            z["probabilities"], z["codes"][ix], z["folds"], z["prior"]
        )
        nf = M.target_fields(nj)
        ns = M.agreement(nf["increment"], nf["supported"], z["reference"])
        permutations.append(ix)
        null_values.append(ns["by_target"])
        null_mass.append(ns["reference_mass"])
        null_available.append(ns["full_primary_available"])
    null_values = np.asarray(null_values)
    full_available = stat["full_primary_available"] and all(null_available)
    rows = []
    for q, t in enumerate(M.TARGETS):
        rows.append(
            {
                "target": t,
                "signed_cross_author_cross_window": float(stat["by_target"][q]),
                "null_mean": float(null_values[:, q].mean()),
                "exploratory_upper_rank": float(
                    (1 + np.count_nonzero(null_values[:, q] >= stat["by_target"][q]))
                    / (1 + len(null_values))
                )
                if full_available
                else None,
                "source_increment_rms": float(np.sqrt(stat["energy"][:, :, q].mean())),
            }
        )
    kl = []
    for b in (0, 1):
        for w in (0, 1):
            use = j[b, w] > 0
            kl.append(
                {
                    "bank": b,
                    "window": w,
                    "profile_continuation_conditional_KL": float(
                        np.sum(
                            j[b, w][use] * np.log(j[b, w][use] / screened[b, w][use])
                        )
                    ),
                }
            )
    aggregate = np.array(
        [
            z["prior"][g] * z["probabilities"][z["codes"] == g].mean((0, 1))
            for g in range(32)
        ]
    )
    quality = R.read(out / "input_quality.json")
    bundle = M.compile_bundle(
        aggregate, quality["frozen_thresholds"], quality["representation_id"]
    )
    R.write(out / "transition_bundle.json", bundle)
    R.write(
        out / "example_reading.json",
        M.query(bundle, R.read(out / "example_transition.json")),
    )
    R.archive(
        out / "mechanism_accounts.npz",
        joint=j,
        screened=screened,
        aggregate=aggregate,
        before=fields["before"],
        after=fields["after"],
        increments=fields["increment"],
        support=fields["supported"],
        null_permutations=np.asarray(permutations),
        null_values=null_values,
        null_mass=np.asarray(null_mass),
    )
    R.table(out / "target_transition_accounts.csv", rows)
    R.table(out / "law_accounts.csv", kl)
    primary_null = null_values.mean(1)
    R.write(
        out / "summary.json",
        {
            "packet": "HM195",
            "constructive_ability": "identified_observed_transition_to_target_evidence_without_source_author_catalogue",
            "primary": stat["restricted_mean"] if full_available else None,
            "restricted_primary": stat["restricted_mean"],
            "full_primary_available": full_available,
            "common_reference_mass": stat["reference_mass"],
            "missing_reference_mass": stat["missing_mass"],
            "null_reference_mass_range": [min(null_mass), max(null_mass)],
            "primary_null_mean": float(primary_null.mean()),
            "primary_exploratory_upper_rank": float(
                (1 + np.count_nonzero(primary_null >= stat["restricted_mean"]))
                / (1 + len(primary_null))
            )
            if full_available
            else None,
            "by_target": rows,
            "matching": matching(j, screened),
            "KL_accounts": kl,
            "same_observable_predictions_BY_CONSTRUCTION_not_accuracy_gain": True,
            "source_only_historical_cohort_no_new_psychological_effect_claim": True,
            "compute_seconds": time.perf_counter() - start,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "computed_seal.json",
        (
            "transition_bundle.json",
            "example_reading.json",
            "mechanism_accounts.npz",
            "target_transition_accounts.csv",
            "law_accounts.csv",
            "summary.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")))


def verify(out):
    """One literal count and conditional-mean audit, not a precision campaign."""
    cfg, z = R.check(out, ("computed_seal.json",))["config"], R.prepared(out)
    with np.load(out / "mechanism_accounts.npz", allow_pickle=False) as v:
        saved = {
            k: v[k] for k in ("joint", "screened", "increments", "null_permutations")
        }
    literal = np.zeros((2, 2, 32, 8, 8))
    counts = np.array(
        [np.bincount(z["codes"][z["folds"] == b], minlength=32) for b in (0, 1)]
    )
    for u, (g, b) in enumerate(zip(z["codes"], z["folds"], strict=True)):
        weight = z["prior"][g] / (counts[b, g] * z["current"].shape[-1])
        for w in (0, 1):
            for a, o in zip(z["current"][u, w], z["outcome"][u, w], strict=True):
                literal[b, w, g, a, o] += weight
    errors = {
        "literal_source_count_error": float(np.max(np.abs(literal - saved["joint"]))),
        **matching(saved["joint"], saved["screened"]),
    }
    conditional_error = 0.0
    for b in (0, 1):
        for w in (0, 1):
            for a in range(8):
                prior = literal[b, w, :, a].sum(-1)
                for o in range(8):
                    post = literal[b, w, :, a, o]
                    if prior.sum() and post.sum():
                        delta = (post / post.sum() - prior / prior.sum()) @ M.BITS
                        conditional_error = max(
                            conditional_error,
                            float(
                                np.max(
                                    np.abs(delta - saved["increments"][b, w, :, a, o])
                                )
                            ),
                        )
    errors["literal_target_update_error"] = conditional_error
    for ix in saved["null_permutations"]:
        if sorted(ix.tolist()) != list(range(len(ix))) or not np.array_equal(
            z["folds"][ix], z["folds"]
        ):
            raise ValueError("An exchange changed the author bank")
    if max(errors.values()) > cfg["audit_tolerance"]:
        raise ArithmeticError("Registered alternate count/conditional audit failed")
    R.write(
        out / "audit_receipt.json",
        {
            **errors,
            "actual_directed_source_pairs": int(z["current"].size),
            "same_agent_not_blind": True,
            "old_inputs_and_science_seals_unchanged": True,
        },
    )
    R.seal(out, "verification_seal.json", ("audit_receipt.json",))
    print(json.dumps(R.read(out / "audit_receipt.json")))


def main():
    """Execute one exclusive registered stage; no automatic experiment queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
