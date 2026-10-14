#!/usr/bin/env python3
"""One SOURCE-profile dependence discriminator, without precision selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_evidence as J
from grain_research.construction import math_target_profile as P
from scripts import run_grain_joint_evidence_hm174 as H
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_target_profile_hm175.json"
DEFAULT = ROOT / "results/grain_math_response_v1/target_profile_composition_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TARGET_PROFILE_PLAN_HM175.md",
    "grain_research/construction/math_target_profile.py",
    "scripts/run_grain_target_profile_hm175.py",
    "scripts/score_grain_target_profile_cold.py",
    "scripts/verify_grain_target_profile_hm175.py",
    "tests/test_grain_target_profile.py",
)


def prepare(out):
    """Seal the new dependence law while preserving old evidence and outcomes."""
    spec = R.read(ROOT / CONFIG)
    parent = ROOT / spec["parent"]
    manifest = H.check(parent, ("scored_seal.json", "analysis_seal.json"))
    bundle = P.build_bundle(R.read(parent / "source_bundle.json"))
    if len(bundle["parent_bundle"]["source_mass"]) != spec["source_authors"]:
        raise ValueError("Changed SOURCE exclusion")
    inputs = dict(manifest["inputs"])
    for stage in ("prepared_seal.json", "scored_seal.json", "analysis_seal.json"):
        for name in (stage, *R.read(parent / stage)):
            path = str((parent / name).relative_to(ROOT))
            inputs[path] = R.digest(ROOT / path)
    inputs[str((parent / "manifest.json").relative_to(ROOT))] = R.digest(
        parent / "manifest.json"
    )
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": manifest["code"] | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "original_scope_complete": False,
        },
    )
    R.write(out / "source_bundle.json", bundle)
    (out / "cold_input.jsonl").write_bytes((parent / "cold_input.jsonl").read_bytes())
    inv = P.invariants(bundle)
    if max(inv.values()) > 2e-12:
        raise ValueError("Target prior or singleton meaning changed")
    number = np.asarray(bundle["profile_counts"])
    quality = {
        "source_authors": spec["source_authors"],
        "panel_authors": spec["panel_authors"],
        "observed_profiles": len(number),
        "minimum_profile_count": int(number.min()),
        "maximum_profile_count": int(number.max()),
        "profiles_with_at_most_five_authors": int(np.sum(number <= 5)),
        "invariants": inv,
        "new_label_or_neural_reads": 0,
        "historical_representation_overlap_retained": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("source_bundle.json", "cold_input.jsonl", "input_quality.json"),
    )
    print(json.dumps(quality))


def score(out):
    """Invoke an outcome-free CLI once on the old480 evidence families."""
    R.check(out)
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_target_profile_cold.py"),
            "--bundle",
            str(out / "source_bundle.json"),
            "--input",
            str(out / "cold_input.jsonl"),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        check=True,
    )
    shape = (2, 24, 2, 5, 3, 5)
    prob, odds = np.full(shape, np.nan), np.full(shape, np.nan)
    support, seen = np.zeros(shape, bool), set()
    with (out / "cold_output.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(map(int, row["author_key"].split("/")))
            if key in seen:
                raise ValueError("Duplicate cold family")
            seen.add(key)
            prob[key], odds[key] = row["probabilities"], row["binary_log_odds"]
            support[key] = row["posterior_supported"]
    if len(seen) != 480:
        raise ValueError("Missing fixed families")
    R.archive(
        out / "cold_accounts.npz",
        probabilities=prob,
        binary_log_odds=odds,
        posterior_supported=support,
    )
    R.write(
        out / "cold_receipt.json",
        {
            "rows": len(seen),
            "seconds": time.perf_counter() - start,
            "new_neural_calls": 0,
            "native_fit": False,
        },
    )
    R.seal(
        out,
        "scored_seal.json",
        ("cold_output.jsonl", "cold_accounts.npz", "cold_receipt.json"),
    )
    print(json.dumps(R.read(out / "cold_receipt.json")))


def values(x):
    """Return finite descriptive magnitudes, not psychometric effect sizes."""
    return H.stats(x)


def contexts(bundle):
    """Enumerate all registered prior and one-event contextual log increments."""
    prior = np.asarray(bundle["parent_bundle"]["prior"])
    prior_odds = np.log(prior / (1 - prior))
    plain, conditional = np.empty((3, 8, 8, 5)), np.empty((3, 8, 8, 5))
    for b in range(8):
        old = P.evaluate(bundle, [b])["binary_log_odds"]
        for a in range(8):
            plain[:, b, a] = P.evaluate(bundle, [a])["binary_log_odds"] - prior_odds
            conditional[:, b, a] = P.evaluate(bundle, [b, a])["binary_log_odds"] - old
    return plain, conditional


def analyze(out):
    """Account for dependence sources; never pick a high-r law or erase states."""
    start = time.perf_counter()
    manifest = R.check(out, ("scored_seal.json",))
    bundle, spec = (
        P.validate_bundle(R.read(out / "source_bundle.json")),
        manifest["config"],
    )
    parent = ROOT / spec["parent"]
    with np.load(out / "cold_accounts.npz", allow_pickle=False) as a:
        prob, odds = a["probabilities"], a["binary_log_odds"]
    with np.load(parent / "cold_accounts.npz", allow_pickle=False) as a:
        old_p, old_o = a["probabilities"], a["binary_log_odds"]
    p_error = float(np.max(np.abs(prob[..., [0, 2], :] - old_p[..., [1, 2], :])))
    o_error = float(np.max(np.abs(odds[..., [0, 2], :] - old_o)))
    if max(p_error, o_error) > 2e-12:
        raise ValueError("Changed parent comparator")
    joint = P.two_event_accounts(bundle)
    defect = (
        joint["total_dependence"] - joint["within_profile"] - joint["between_profiles"]
    )
    jmean = J.composition(np.moveaxis(prob, 3, 0), 0.5)
    jsum = J.composition(np.moveaxis(odds, 3, 0))
    plain, conditional = contexts(bundle)
    rev = plain * conditional < 0
    finite = np.isfinite(plain) & np.isfinite(conditional)
    z = H.load(parent)
    joint_rows, context_rows, composition_rows, signed_rows = [], [], [], []
    per_trait = []
    for q, trait in enumerate(spec["traits"]):
        valid = finite[1, :, :, q] & finite[2, :, :, q]
        rp, ra = rev[1, :, :, q], rev[2, :, :, q]
        per_trait.append(
            {
                "trait": trait,
                "valid_context_pairs": int(valid.sum()),
                "profile_reversals": int(np.sum(rp & valid)),
                "author_reversals": int(np.sum(ra & valid)),
                "both_reverse": int(np.sum(rp & ra & valid)),
                "profile_only_reverse": int(np.sum(rp & ~ra & valid)),
                "author_only_reverse": int(np.sum(~rp & ra & valid)),
                "conditional_profile_author_difference": values(
                    conditional[2, :, :, q] - conditional[1, :, :, q]
                ),
            }
        )
        w, b, t = (
            joint[k][q] for k in ("signed_within", "signed_between", "signed_total")
        )
        signed_rows.append(
            {
                "trait": trait,
                "within_squared_norm": float(np.sum(w * w)),
                "between_squared_norm": float(np.sum(b * b)),
                "signed_cross_term": float(2 * np.sum(w * b)),
                "total_squared_norm": float(np.sum(t * t)),
                "energy_identity_error": float(
                    abs(
                        np.sum(t * t)
                        - np.sum(w * w)
                        - np.sum(b * b)
                        - 2 * np.sum(w * b)
                    )
                ),
            }
        )
        for g in (0, 1):
            for a in range(8):
                for b in range(8):
                    joint_rows.append(
                        {
                            "trait": trait,
                            "group": g,
                            "event_a": a,
                            "event_b": b,
                            **{
                                name: float(joint["joint_laws"][k, q, g, a, b])
                                for k, name in enumerate(P.MODELS)
                            },
                            **{
                                name: float(joint[name][q, g, a, b])
                                for name in (
                                    "total_dependence",
                                    "within_profile",
                                    "between_profiles",
                                )
                            },
                        }
                    )
        for k, model in enumerate(P.MODELS):
            for b in range(8):
                for a in range(8):
                    context_rows.append(
                        {
                            "model": model,
                            "trait": trait,
                            "context_cell": b,
                            "event_cell": a,
                            "plain_log_Bayes_increment": H.finite(plain[k, b, a, q]),
                            "conditional_log_Bayes_increment": H.finite(
                                conditional[k, b, a, q]
                            ),
                            "increment_change": H.finite(
                                conditional[k, b, a, q] - plain[k, b, a, q]
                            ),
                            "sign_reversed": bool(rev[k, b, a, q])
                            if finite[k, b, a, q]
                            else None,
                        }
                    )
            for arm in range(2):
                for u in range(24):
                    for v in range(2):
                        composition_rows.append(
                            {
                                "arm": spec["arms"][arm],
                                "author_id": str(z["author_ids"][u]),
                                "view": v,
                                "split": str(z["splits"][u]),
                                "trait": trait,
                                "model": model,
                                "J_mean_probability": H.finite(jmean[arm, u, v, k, q]),
                                "J_sum_binary_log_odds": H.finite(
                                    jsum[arm, u, v, k, q]
                                ),
                            }
                        )
    per_model = [
        {
            "model": name,
            "source_context_reversals": int(np.sum(rev[k] & finite[k])),
            "missing_contexts": int(np.sum(~finite[k])),
            "context_increment_change": values(conditional[k] - plain[k]),
            "natural_J_mean_probability": values(jmean[..., k, :]),
            "natural_J_sum_binary_log_odds": values(jsum[..., k, :]),
        }
        for k, name in enumerate(P.MODELS)
    ]
    summary = {
        "source_authors": 386,
        "panel_authors": 24,
        "views": 48,
        "cold_families": 480,
        "quality": R.read(out / "input_quality.json"),
        "per_model": per_model,
        "per_trait_context": per_trait,
        "signed_dependence": signed_rows,
        "weighted_within_profile_joint_RMS": values(joint["within_profile"]),
        "weighted_between_profiles_joint_RMS": values(joint["between_profiles"]),
        "joint_identity_max_error": float(np.max(np.abs(defect))),
        "parent_probability_error": p_error,
        "parent_log_odds_error": o_error,
        "profile_author_cold_probability_difference": values(
            prob[..., 2, :] - prob[..., 1, :]
        ),
        "profile_author_cold_log_odds_difference": values(
            odds[..., 2, :] - odds[..., 1, :]
        ),
        "posterior_missing": int(np.sum(~np.isfinite(prob))),
        "nonfinite_log_odds": int(np.sum(~np.isfinite(odds))),
        "seconds": time.perf_counter() - start,
        "new_neural_calls": 0,
        "native_answers_fit": False,
        "psychological_accuracy_promotion": False,
        "original_scope_complete": False,
    }
    R.table(out / "two_event_joint_accounts.csv", joint_rows)
    R.table(out / "source_context_accounts.csv", context_rows)
    R.table(out / "natural_composition_accounts.csv", composition_rows)
    R.table(out / "signed_dependence_accounts.csv", signed_rows)
    R.archive(
        out / "mechanism_accounts.npz",
        **joint,
        plain=plain,
        conditional=conditional,
        probability_J_mean=jmean,
        binary_log_odds_J_sum=jsum,
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "two_event_joint_accounts.csv",
            "source_context_accounts.csv",
            "natural_composition_accounts.csv",
            "signed_dependence_accounts.csv",
            "mechanism_accounts.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def main():
    """Execute one stage once, preserving old and partial products."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "score": score, "analyze": analyze}[args.stage](args.out)


if __name__ == "__main__":
    main()
