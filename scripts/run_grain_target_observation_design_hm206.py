#!/usr/bin/env python3
"""Frozen source-linked acquisition, followed by disjoint-author archive evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_target_observation_design as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_target_observation_design_hm206.json"
DEFAULT = ROOT / "results/grain_math_response_v1/target_observation_design_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TARGET_OBSERVATION_DESIGN_PLAN_HM206.md",
    "grain_research/construction/math_target_observation_design.py",
    "scripts/run_grain_target_observation_design_hm206.py",
    "tests/test_grain_target_observation_design.py",
    "scripts/run_grain_observed_family_hm170.py",
)
LINKS = ("source_link", "profile_permutation")


def finite(value):
    """Keep unavailable observations null, rather than assigning neutral zeros."""
    return float(value) if np.isfinite(value) else None


def ratio(numerator, denominator):
    """An empty common-support experiment has no mean, not a zero gain."""
    return finite(numerator / denominator) if denominator > 0 else None


def labels(codes):
    """Decode only the inherited E/N/A/C/O source profile, without new thresholds."""
    return ((codes[:, None] >> np.arange(5)) & 1).astype(float)


def prepare(out):
    """Authenticate distinct events, source links and the inherited author folds."""
    start = time.perf_counter()
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent, ("computed_seal.json",))
    old = R.read(parent / "audit_receipt.json")
    if old["direct_event_pair_law_error"] > cfg["numeric_tolerance"]:
        raise ValueError("Parent literal event audit failed")
    with np.load(parent / "observations.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in ("source", "cells", "units", "codes", "folds")}
    n, t = cfg["source_authors"], cfg["source_events_per_window"]
    if data["cells"].shape != (n, 2, t) or data["units"].shape != (n, 2, t):
        raise ValueError("Changed addressed source windows")
    if len(set(data["source"].tolist())) != n or data["codes"].shape != (n,):
        raise ValueError("Changed author or target links")
    if not np.isin(data["codes"], np.arange(32)).all():
        raise ValueError("Changed five-target profile alphabet")
    f = data["folds"]
    if (
        not np.isin(f, (0, 1)).all()
        or np.bincount(f).tolist() != cfg["source_fold_counts"]
    ):
        raise ValueError("Changed inherited author split")
    p = M.event_probabilities(data["cells"], data["units"])
    M.validate(p, labels(data["codes"]))
    rng = np.random.default_rng(cfg["label_control_seed"])
    shuffled = data["codes"].copy()
    permutation = np.arange(n)
    for fold in (0, 1):
        ids = np.flatnonzero(f == fold)
        permutation[ids] = rng.permutation(ids)
        shuffled[ids] = data["codes"][permutation[ids]]
    if not np.array_equal(f[permutation], f):
        raise ValueError("Control crossed author folds")
    inputs = (
        "manifest.json",
        "observations.npz",
        "prepared_seal.json",
        "computed_seal.json",
        "audit_receipt.json",
    )
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p)
                for p in inputs
            },
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        **data,
        probabilities=p,
        shuffled_codes=shuffled,
        label_permutation=permutation,
    )
    mask, _ = M.archive_menu(p)
    quality = {
        "source_authors": n,
        "fold_counts": np.bincount(f).tolist(),
        "distinct_addressed_events": int(data["units"].size),
        "availability_masks": np.bincount(mask, minlength=4).tolist(),
        "profile_counts_by_fold": [
            np.bincount(data["codes"][f == s], minlength=32).tolist() for s in (0, 1)
        ],
        "historical_global_feature_thresholds_retained": True,
        "untouched_full_pipeline_CV": False,
        "source_labels_used": True,
        "application_labels_or_raw_private_texts_read": 0,
        "new_model_calls": 0,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(quality))


def load_design(out, fold, link):
    """Read a frozen source constructor, never fit on the evaluation authors."""
    with np.load(out / "designs.npz", allow_pickle=False) as z:
        prefix = f"f{fold}_l{link}_"
        return {k[len(prefix) :]: z[k] for k in z.files if k.startswith(prefix)}


def build(out):
    """Freeze acquisition policies and source scores before held-author evaluation."""
    R.check(out)
    start, z = time.perf_counter(), R.prepared(out)
    arrays, bundles, values, choices, disagreements, source_checks = (
        {},
        {},
        [],
        [],
        [],
        [],
    )
    for fold in (0, 1):
        ix = z["folds"] == fold
        for link, name in enumerate(LINKS):
            codes = z["codes"] if link == 0 else z["shuffled_codes"]
            p, y = z["probabilities"][ix], labels(codes[ix])
            d = M.compile_design(p, y)
            arrays.update({f"f{fold}_l{link}_{k}": np.asarray(v) for k, v in d.items()})
            bundles[f"{fold}/{link}"] = M.bundle(d)
            self_eval = M.evaluate(d, p, y)
            source_checks.append(
                {
                    "calibration_fold": fold,
                    "link": name,
                    "tower_error": d["tower_error"],
                    "variance_error": d["variance_error"],
                    "expected_risk_reduction_error": float(
                        np.max(
                            np.abs(
                                self_eval["gain"].sum(0) - self_eval["forecast"].sum(0)
                            )
                        )
                        / len(p)
                    ),
                    "source_support": float(self_eval["support"].mean()),
                }
            )
            for m, a in np.argwhere(d["mass"] > 0):
                for q, target in enumerate(M.TARGETS):
                    for c in (0, 1):
                        if not int(m) & (1 << c):
                            continue
                        values.append(
                            {
                                "calibration_fold": fold,
                                "link": name,
                                "mask": int(m),
                                "early_cell": int(a),
                                "target": target,
                                "condition": c,
                                "source_state_mass": float(d["mass"][m, a]),
                                "before": float(d["before"][m, a, q]),
                                "expected_risk_gain": float(d["value"][m, a, c, q]),
                                "target_information": float(d["target_mi"][m, a, c, q]),
                                "identity_information": float(
                                    d["identity_mi"][m, a, c]
                                ),
                            }
                        )
                    for k, policy in enumerate(M.POLICIES):
                        choices.append(
                            {
                                "calibration_fold": fold,
                                "link": name,
                                "mask": int(m),
                                "early_cell": int(a),
                                "target": target,
                                "policy": policy,
                                "condition0": float(d["policies"][k, m, a, q, 0]),
                                "condition1": float(d["policies"][k, m, a, q, 1]),
                            }
                        )
            both = d["mass"][3].sum()
            for q, target in enumerate(M.TARGETS):
                supported = d["mass"][3] > 0
                different = np.argmax(
                    d["policies"][1, 3, supported, q], axis=-1
                ) != np.argmax(d["policies"][2, 3, supported, q], axis=-1)
                target_difference = (
                    np.argmax(d["policies"][1, 3, supported], axis=-1)
                    != np.argmax(d["policies"][1, 3, supported, q], axis=-1)[:, None]
                )
                disagreements.append(
                    {
                        "calibration_fold": fold,
                        "link": name,
                        "target": target,
                        "available_both_source_mass": float(both),
                        "target_vs_identity_information_disagreement": ratio(
                            d["mass"][3, supported] @ different, both
                        ),
                        "target_vs_other_targets_disagreement": ratio(
                            np.sum(d["mass"][3, supported, None] * target_difference),
                            both * 4,
                        ),
                    }
                )
    R.archive(out / "designs.npz", **arrays)
    R.write(out / "selector_bundles.json", bundles)
    R.table(out / "state_values.csv", values)
    R.table(out / "choice_policies.csv", choices)
    R.table(out / "policy_disagreement.csv", disagreements)
    R.write(
        out / "build_summary.json",
        {
            "source_checks": source_checks,
            "elapsed_seconds": time.perf_counter() - start,
            "acquisition_and_scoring_frozen_before_evaluation": True,
            "common_support_required_for_all_policies": True,
        },
    )
    R.seal(
        out,
        "built_seal.json",
        (
            "designs.npz",
            "selector_bundles.json",
            "state_values.csv",
            "choice_policies.csv",
            "policy_disagreement.csv",
            "build_summary.json",
        ),
    )
    print(json.dumps(R.read(out / "build_summary.json")))


def evaluate(out):
    """Keep all signed held-author gains and source forecasts, not a best-policy grid."""
    R.check(out, ("built_seal.json",))
    start, z = time.perf_counter(), R.prepared(out)
    arrays, rows, author_rows, coverage = {}, [], [], []
    pooled_gain = np.zeros((2, 6, 5))
    pooled_forecast = np.zeros_like(pooled_gain)
    total_support = np.zeros(2)
    for fold in (0, 1):
        held = z["folds"] != fold
        for link, name in enumerate(LINKS):
            d = load_design(out, fold, link)
            e = M.evaluate(d, z["probabilities"][held], labels(z["codes"][held]))
            arrays.update({f"f{fold}_l{link}_{k}": v for k, v in e.items()})
            weight = e["support"].sum()
            total_support[link] += weight
            pooled_gain[link] += e["gain"].sum(0)
            pooled_forecast[link] += e["forecast"].sum(0)
            coverage.append(
                {
                    "calibration_fold": fold,
                    "link": name,
                    "held_authors": int(held.sum()),
                    "any_support_authors": int(np.count_nonzero(e["support"])),
                    "complete_support_authors": int(
                        np.count_nonzero(np.isclose(e["support"], 1, atol=1e-12))
                    ),
                    "shared_support_mean": float(e["support"].mean()),
                    "missing_prefix_mass": float(e["missing_pre"].mean()),
                    "missing_outcome_mass": float(e["missing_outcome"].mean()),
                }
            )
            for k, policy in enumerate(M.POLICIES):
                for q, target in enumerate(M.TARGETS):
                    rows.append(
                        {
                            "calibration_fold": fold,
                            "evaluation_fold": 1 - fold,
                            "link": name,
                            "policy": policy,
                            "target": target,
                            "shared_support": weight,
                            "before_risk": ratio(e["before_risk"][:, q].sum(), weight),
                            "after_risk": ratio(e["after_risk"][:, k, q].sum(), weight),
                            "realized_gain": ratio(e["gain"][:, k, q].sum(), weight),
                            "source_forecast": ratio(
                                e["forecast"][:, k, q].sum(), weight
                            ),
                        }
                    )
            for u, author in enumerate(z["source"][held]):
                for k, policy in enumerate(M.POLICIES):
                    for q, target in enumerate(M.TARGETS):
                        author_rows.append(
                            {
                                "calibration_fold": fold,
                                "link": name,
                                "author_source_index": int(author),
                                "policy": policy,
                                "target": target,
                                "support": float(e["support"][u]),
                                "before_risk": float(e["before_risk"][u, q]),
                                "after_risk": float(e["after_risk"][u, k, q]),
                                "realized_gain": float(e["gain"][u, k, q]),
                                "source_forecast": float(e["forecast"][u, k, q]),
                            }
                        )
    pooled = []
    for link, name in enumerate(LINKS):
        for k, policy in enumerate(M.POLICIES):
            for q, target in enumerate(M.TARGETS):
                pooled.append(
                    {
                        "link": name,
                        "policy": policy,
                        "target": target,
                        "realized_gain": ratio(
                            pooled_gain[link, k, q], total_support[link]
                        ),
                        "source_forecast": ratio(
                            pooled_forecast[link, k, q], total_support[link]
                        ),
                        "gain_vs_identity_information": ratio(
                            pooled_gain[link, k, q] - pooled_gain[link, 2, q],
                            total_support[link],
                        ),
                        "gain_vs_uniform": ratio(
                            pooled_gain[link, k, q] - pooled_gain[link, 5, q],
                            total_support[link],
                        ),
                    }
                )
    R.archive(out / "evaluation.npz", **arrays)
    R.table(out / "policy_evaluation.csv", rows)
    R.table(out / "author_evaluation.csv", author_rows)
    R.table(out / "pooled_policy_evaluation.csv", pooled)
    summary = {
        "packet": "HM206",
        "coverage": coverage,
        "pooled": pooled,
        "elapsed_seconds": time.perf_counter() - start,
        "source_links_used_for_calibration": True,
        "no_neural_private_API_training_or_application_labels": True,
        "all_targets_policies_and_negative_gains_retained": True,
        "held_result_used_for_policy_selection": False,
        "original_scope_complete": False,
        "estimand": "one_step_retrospective_randomized_archive_sampling_Brier_risk_change_not_human_prompt_intervention",
        "label_permutation_is_descriptive_not_significance": True,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "evaluated_seal.json",
        (
            "evaluation.npz",
            "policy_evaluation.csv",
            "author_evaluation.csv",
            "pooled_policy_evaluation.csv",
            "summary.json",
        ),
    )
    print(json.dumps({"coverage": coverage, "source_link_pooled": pooled[:30]}))


def literal_law(cells, y):
    """Alternate count-based archive law, without module frequency extraction."""
    early = np.array([np.bincount(row[0], minlength=8) for row in cells], float)
    early /= cells.shape[2]
    late = np.array([np.bincount(row[1], minlength=8) for row in cells], float).reshape(
        -1, 2, 4
    )
    totals = late.sum(2)
    mask = (totals[:, 0] > 0).astype(int) + 2 * (totals[:, 1] > 0)
    late = np.divide(
        late, totals[:, :, None], out=np.zeros_like(late), where=totals[:, :, None] > 0
    )
    return early, late, mask, y


def verify(out):
    """Re-sum source and held-author accounts once using alternate literal arithmetic."""
    cfg = R.check(out, ("built_seal.json", "evaluated_seal.json"))["config"]
    start, z = time.perf_counter(), R.prepared(out)
    errors, states, outcomes, held_states, cold_checks = [], 0, 0, 0, 0
    bits = (np.arange(32)[:, None] >> np.arange(5)) & 1
    bundles = R.read(out / "selector_bundles.json")
    with np.load(out / "evaluation.npz", allow_pickle=False) as a:
        for fold in (0, 1):
            ix, held = z["folds"] == fold, z["folds"] != fold
            for link in (0, 1):
                codes = z["codes"] if link == 0 else z["shuffled_codes"]
                early, late, mask, y = literal_law(z["cells"][ix], labels(codes[ix]))
                d = load_design(out, fold, link)
                for m, c0 in np.argwhere(d["mass"] > 0):
                    w = early[:, c0] * (mask == m)
                    w /= w.sum()
                    mu = np.array(
                        [
                            sum(float(w[u] * y[u, q]) for u in range(len(w)))
                            for q in range(5)
                        ]
                    )
                    errors.append(float(np.max(np.abs(mu - d["before"][m, c0]))))
                    errors.append(
                        float(np.max(np.abs(d["before_profile"][m, c0] @ bits - mu)))
                    )
                    states += 1
                    for c in (0, 1):
                        if not int(m) & (1 << c):
                            continue
                        pb = np.array(
                            [
                                sum(w[u] * late[u, c, b] for u in range(len(w)))
                                for b in range(4)
                            ]
                        )
                        pos = pb > 0
                        post = np.array(
                            [
                                [
                                    sum(
                                        w[u] * late[u, c, b] * y[u, q]
                                        for u in range(len(w))
                                    )
                                    / pb[b]
                                    for q in range(5)
                                ]
                                for b in np.flatnonzero(pos)
                            ]
                        )
                        v = sum(
                            pb[b] * (post[j] - mu) ** 2
                            for j, b in enumerate(np.flatnonzero(pos))
                        )
                        hb = -sum(x * np.log(x) for x in pb if x > 0)
                        hu = sum(
                            w[u] * (-sum(x * np.log(x) for x in late[u, c] if x > 0))
                            for u in range(len(w))
                        )
                        errors.extend(
                            (
                                float(np.max(np.abs(pb - d["outcome"][m, c0, c]))),
                                float(np.max(np.abs(post - d["after"][m, c0, c, pos]))),
                                float(np.max(np.abs(v - d["value"][m, c0, c]))),
                                float(abs(hb - hu - d["identity_mi"][m, c0, c])),
                            )
                        )
                        errors.append(
                            float(
                                np.max(
                                    np.abs(
                                        d["after_profile"][m, c0, c, pos] @ bits - post
                                    )
                                )
                            )
                        )
                        outcomes += int(pos.sum())
                    for q in range(5):
                        actual = M.recommend(
                            bundles[f"{fold}/{link}"],
                            {
                                "event_id": "literal-current-event",
                                "current_cell": int(c0),
                                "availability": [
                                    c for c in (0, 1) if int(m) & (1 << c)
                                ],
                                "target": q,
                            },
                        )
                        for k, policy in enumerate(M.POLICIES):
                            errors.append(
                                float(
                                    np.max(
                                        np.abs(
                                            np.array(
                                                actual["condition_probabilities"][
                                                    policy
                                                ]
                                            )
                                            - d["policies"][k, m, c0, q]
                                        )
                                    )
                                )
                            )
                        cold_checks += 1
                ep, lp, mp, yp = literal_law(z["cells"][held], labels(z["codes"][held]))
                br = np.zeros((len(ep), 5))
                ar = np.zeros((len(ep), 6, 5))
                pred = np.zeros_like(ar)
                supported = np.zeros(len(ep))
                for u in range(len(ep)):
                    for c0 in range(8):
                        pa, m = ep[u, c0], mp[u]
                        if not pa or not d["mass"][m, c0]:
                            continue
                        available = [c for c in (0, 1) if int(m) & (1 << c)]
                        if any(
                            lp[u, c, b] > 0 and d["outcome"][m, c0, c, b] == 0
                            for c in available
                            for b in range(4)
                        ):
                            continue
                        supported[u] += pa
                        held_states += 1
                        for q in range(5):
                            br[u, q] += pa * (yp[u, q] - d["before"][m, c0, q]) ** 2
                            for k in range(6):
                                for c in available:
                                    pc = d["policies"][k, m, c0, q, c]
                                    pred[u, k, q] += pa * pc * d["value"][m, c0, c, q]
                                    for b in range(4):
                                        if lp[u, c, b] > 0:
                                            ar[u, k, q] += (
                                                pa
                                                * pc
                                                * lp[u, c, b]
                                                * (
                                                    yp[u, q]
                                                    - d["after"][m, c0, c, b, q]
                                                )
                                                ** 2
                                            )
                for key, literal in (
                    ("support", supported),
                    ("before_risk", br),
                    ("after_risk", ar),
                    ("gain", br[:, None] - ar),
                    ("forecast", pred),
                ):
                    errors.append(
                        float(np.max(np.abs(literal - a[f"f{fold}_l{link}_{key}"])))
                    )
    maximum = max(errors)
    if maximum > cfg["numeric_tolerance"]:
        raise ValueError(f"Literal audit failed: {maximum}")
    receipt = {
        "PASS": True,
        "literal_account_max_error": maximum,
        "checks": len(errors),
        "source_states": states,
        "supported_outcomes": outcomes,
        "held_pre_states": held_states,
        "cold_interface_checks": cold_checks,
        "same_agent_not_independent": True,
        "scientific_rerun": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verified_seal.json", ("verification.json",))
    print(json.dumps(receipt))


def main():
    """Run one exclusive stage, or a read-only cold selector invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "build", "evaluate", "verify", "recommend")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--link", type=int, choices=(0, 1), default=0)
    parser.add_argument("--record")
    args = parser.parse_args()
    if args.stage == "recommend":
        R.check(args.output, ("built_seal.json",))
        bank = R.read(args.output / "selector_bundles.json")[f"{args.fold}/{args.link}"]
        print(json.dumps(M.recommend(bank, json.loads(args.record))))
    else:
        {"prepare": prepare, "build": build, "evaluate": evaluate, "verify": verify}[
            args.stage
        ](args.output)


if __name__ == "__main__":
    main()
