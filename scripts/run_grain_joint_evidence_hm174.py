#!/usr/bin/env python3
"""Finite SAME-event evidence composition experiment; no accuracy search."""

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
from grain_research.construction import math_trait_anchor as A
from scripts import run_grain_event_trait_action_hm173 as E
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_trait_anchor_hm171 as T

CONFIG = "configs/grain_joint_evidence_hm174.json"
DEFAULT = ROOT / "results/grain_math_response_v1/joint_evidence_law_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_JOINT_EVIDENCE_PLAN_HM174.md",
    "grain_research/construction/math_joint_evidence.py",
    "scripts/run_grain_joint_evidence_hm174.py",
    "scripts/score_grain_joint_evidence_cold.py",
    "scripts/verify_grain_joint_evidence_hm174.py",
    "tests/test_grain_joint_evidence.py",
    "grain_research/construction/math_trait_anchor.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "scripts/run_grain_event_trait_action_hm173.py",
    "scripts/run_grain_trait_anchor_hm171.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def check(out, stages=()):
    """Check exact methods, parent identities and completed artifacts."""
    return R.check(out, stages)


def load(out):
    """Load only prepared numerical events, not raw text or private scales."""
    return R.prepared(out)


def prepare(out):
    """Seal the source-only bundle and evidence identities before composition."""
    spec = R.read(ROOT / CONFIG)
    parent = ROOT / spec["parent"]
    m = E.check(parent, ("compiled_seal.json", "analysis_seal.json"))
    z = E.load(parent)
    y = T.labels(m["old_source_config"])[z["source_indices"]]
    old = R.read(parent / "source_law.json")
    if len(y) != spec["source_authors"]:
        raise ValueError("Changed SOURCE/application exclusion")
    binary = (y > np.asarray(old["label_cutoffs"])).astype(float)
    bundle = J.build_bundle(z["source_mass"], binary)
    if not np.allclose(
        bundle["event_probability"], old["probabilities"], rtol=0, atol=2e-15
    ):
        raise ValueError("One-event target law changed")
    cells = np.stack(
        [
            4 * z["conditions"]
            + R.F.states(
                z["coordinates"][:, arm].reshape(-1, 15), old["coordinate_cutoffs"]
            ).reshape(24, 2, 12)
            for arm in range(2)
        ]
    )
    inputs = dict(m["inputs"])
    for name in (
        "manifest.json",
        "observations.npz",
        "source_law.json",
        "action_accounts.npz",
        "analysis_seal.json",
    ):
        path = str((parent / name).relative_to(ROOT))
        inputs[path] = R.digest(ROOT / path)
    inputs[CONFIG] = R.digest(ROOT / CONFIG)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "parent_config": m["config"],
            "original_scope_complete": False,
        },
    )
    R.write(out / "source_bundle.json", bundle)
    R.archive(
        out / "observations.npz",
        cells=cells,
        selected=z["selected"],
        control=z["control"],
        author_ids=z["author_ids"],
        splits=z["splits"],
    )
    rows = 0
    with (out / "cold_input.jsonl").open("x") as stream:
        for arm in range(2):
            for u in range(24):
                for v in range(2):
                    groups = J.group_indices(z["selected"][u, v], z["control"][u, v])
                    for variant, ids in enumerate(groups):
                        record = {
                            "author_key": f"{arm}/{u}/{v}/{variant}",
                            "events": [
                                {
                                    "event_id": f"{u}/{v}/{int(i)}",
                                    "cell": int(cells[arm, u, v, i]),
                                }
                                for i in ids
                            ],
                        }
                        stream.write(json.dumps(record) + "\n")
                        rows += 1
    quality = {
        "source_authors": len(y),
        "application_authors": 24,
        "views": 48,
        "cold_inputs": rows,
        "one_event_law_error": float(
            np.max(
                np.abs(np.asarray(bundle["event_probability"]) - old["probabilities"])
            )
        ),
        "new_neural_calls": 0,
        "new_application_label_reads": 0,
        "historical_representation_overlap_retained": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        (
            "source_bundle.json",
            "observations.npz",
            "cold_input.jsonl",
            "input_quality.json",
        ),
    )
    print(json.dumps(quality))


def score(out):
    """Run a separate cold process; never supply native outputs to the scorer."""
    check(out)
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_joint_evidence_cold.py"),
            "--bundle",
            str(out / "source_bundle.json"),
            "--input",
            str(out / "cold_input.jsonl"),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        check=True,
    )
    p = np.full((2, 24, 2, 5, 3, 5), np.nan)
    odds = np.full((2, 24, 2, 5, 2, 5), np.nan)
    support = np.zeros_like(odds, bool)
    seen = set()
    with (out / "cold_output.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            key = tuple(map(int, r["author_key"].split("/")))
            if key in seen:
                raise ValueError("Duplicated cold evidence family")
            seen.add(key)
            p[key] = np.asarray(r["probabilities"], float)
            odds[key] = np.asarray(r["binary_log_odds"], float)
            support[key] = r["posterior_supported"]
    if len(seen) != 480:
        raise ValueError("Incomplete fixed cold family")
    R.archive(
        out / "cold_accounts.npz",
        probabilities=p,
        binary_log_odds=odds,
        posterior_supported=support,
    )
    R.write(
        out / "cold_receipt.json",
        {
            "rows": len(seen),
            "seconds": time.perf_counter() - start,
            "fit": False,
            "new_neural_calls": 0,
        },
    )
    R.seal(
        out,
        "scored_seal.json",
        ("cold_output.jsonl", "cold_accounts.npz", "cold_receipt.json"),
    )
    print(json.dumps(R.read(out / "cold_receipt.json")))


def finite(value):
    """Keep unsupported posterior quantities missing, never replace with zero."""
    return float(value) if np.isfinite(value) else None


def stats(x):
    """Describe all finite values and explicitly account for missing support."""
    x = np.asarray(x, float)
    valid = x[np.isfinite(x)]
    return {
        "finite": len(valid),
        "missing": int(x.size - len(valid)),
        "rms": None if not len(valid) else float(np.sqrt(np.mean(valid**2))),
        "max_abs": None if not len(valid) else float(np.max(np.abs(valid))),
    }


def native_logits(z, spec):
    """Restore all ten candidates from the same five historical natural sets."""
    base = ROOT / spec["intervention_parent"] / "private_packets"
    result = np.empty((24, 2, 5, 10))
    for u, uid in enumerate(z["author_ids"]):
        for v in range(2):
            p = R.read(base / f"{uid}_{v}.json")
            if (p["author_id"], p["view"], p["task"]) != (uid, v, "neuroticism"):
                raise ValueError("Changed native endpoint identity")
            for k, variant in enumerate(J.VARIANTS):
                readout = p["readouts"][variant]
                result[u, v, k] = readout["candidate_logits"]
    return result


def analyze(out):
    """Discriminate composition laws, not select the highest-r reader."""
    start = time.perf_counter()
    m, z = check(out, ("scored_seal.json",)), load(out)
    b = R.read(out / "source_bundle.json")
    J.validate_bundle(b)
    with np.load(out / "cold_accounts.npz", allow_pickle=False) as a:
        prob, odds = a["probabilities"], a["binary_log_odds"]
    parent = ROOT / m["config"]["parent"]
    with np.load(parent / "action_accounts.npz", allow_pickle=False) as a:
        native = a["native_readouts"][..., :5]
    logits = native_logits(z, m["parent_config"])
    expected = np.exp(logits - logits.max(-1, keepdims=True))
    expected /= expected.sum(-1, keepdims=True)
    native_replay_error = float(np.max(np.abs(expected @ (np.arange(10) / 9) - native)))
    if native_replay_error > 2e-12:
        raise ValueError("Native ordinal readout changed")
    np_jmean = J.composition(np.moveaxis(native, 2, 0), 0.5)
    nlog_j = J.composition(np.moveaxis(J.categorical_contrasts(logits), 2, 0))
    pmean_j = J.composition(np.moveaxis(prob, 3, 0), 0.5)
    psum_j = J.composition(np.moveaxis(prob, 3, 0))
    odds_j = J.composition(np.moveaxis(odds, 3, 0))
    rows = []
    for arm in range(2):
        for u in range(24):
            for v in range(2):
                for model, name in enumerate(J.MODELS):
                    for q, trait in enumerate(A.TRAITS):
                        row = {
                            "arm": m["config"]["arms"][arm],
                            "author_id": str(z["author_ids"][u]),
                            "view": v,
                            "split": str(z["splits"][u]),
                            "model": name,
                            "trait": trait,
                            "J_mean_probability": finite(pmean_j[arm, u, v, model, q]),
                            "J_sum_probability": finite(psum_j[arm, u, v, model, q]),
                            "J_sum_binary_log_odds": None
                            if model == 0
                            else finite(odds_j[arm, u, v, model - 1, q]),
                            "native_J_mean_ordinal_score": float(np_jmean[u, v]),
                        }
                        row.update(
                            {
                                f"prob_{name}": finite(prob[arm, u, v, i, model, q])
                                for i, name in enumerate(J.VARIANTS)
                            }
                        )
                        rows.append(row)
    native_rows = [
        {
            "author_id": str(z["author_ids"][u]),
            "view": v,
            "digit": k + 1,
            "reference_digit": 0,
            "J_sum_categorical_log_odds": float(nlog_j[u, v, k]),
        }
        for u in range(24)
        for v in range(2)
        for k in range(9)
    ]
    context_rows = []
    for context in range(8):
        before = J.evaluate(b, [context])
        for event in range(8):
            singleton, joint = J.evaluate(b, [event]), J.evaluate(b, [context, event])
            for model in range(2):
                prior_odds = np.log(
                    np.asarray(b["prior"]) / (1 - np.asarray(b["prior"]))
                )
                plain = singleton["binary_log_odds"][model] - prior_odds
                contextual = (
                    joint["binary_log_odds"][model] - before["binary_log_odds"][model]
                )
                for q, trait in enumerate(A.TRAITS):
                    context_rows.append(
                        {
                            "model": J.MODELS[model + 1],
                            "trait": trait,
                            "context_cell": context,
                            "event_cell": event,
                            "plain_log_Bayes_increment": finite(plain[q]),
                            "context_log_Bayes_increment": finite(contextual[q]),
                            "increment_change": finite(contextual[q] - plain[q]),
                            "sign_reversed": bool(plain[q] * contextual[q] < 0)
                            if np.isfinite([plain[q], contextual[q]]).all()
                            else None,
                        }
                    )
    links = []
    for arm in range(2):
        for model, name in enumerate(J.MODELS):
            for q, trait in enumerate(A.TRAITS):
                x, y = pmean_j[arm, :, :, model, q].mean(1), np_jmean.mean(1)
                good = np.isfinite(x) & np.isfinite(y)
                links.append(
                    {
                        "arm": m["config"]["arms"][arm],
                        "model": name,
                        "trait": trait,
                        "authors": int(good.sum()),
                        "descriptive_J_mean_r": E.E.pearson(x[good], y[good]),
                        "selection": False,
                    }
                )
    catalogue_context = [r for r in context_rows if r["model"] == "shared_author"]
    summary = {
        "authors": 24,
        "views": 48,
        "source_authors": 386,
        "cold_families": 480,
        "event_account_rows": len(rows),
        "native_log_odds_rows": len(native_rows),
        "source_context_rows": len(context_rows),
        "native_ordinal_composition": stats(np_jmean),
        "native_pre_softmax_composition": stats(nlog_j),
        "native_author_mean_pre_softmax_composition": stats(nlog_j.mean(1)),
        "native_per_digit": [stats(nlog_j[..., k]) for k in range(9)],
        "event_mean_identity": stats(pmean_j[..., 0, :]),
        "target_iid_log_odds_identity": stats(odds_j[..., 0, :]),
        "shared_author_log_odds_composition": stats(odds_j[..., 1, :]),
        "posterior_missing": int(np.isnan(prob).sum()),
        "nonfinite_binary_log_odds": int(np.sum(~np.isfinite(odds))),
        "source_context_sign_reversals": sum(
            r["sign_reversed"] is True for r in catalogue_context
        ),
        "source_context_increment_change": stats(
            [
                np.nan if r["increment_change"] is None else r["increment_change"]
                for r in catalogue_context
            ]
        ),
        "native_score_replay_error": native_replay_error,
        "elapsed_seconds": time.perf_counter() - start,
        "new_neural_calls": 0,
        "native_answer_fit": False,
        "psychological_accuracy_promotion": False,
        "original_scope_complete": False,
    }
    R.table(out / "event_composition_accounts.csv", rows)
    R.table(out / "native_composition_accounts.csv", native_rows)
    R.table(out / "source_context_updates.csv", context_rows)
    R.table(out / "descriptive_links.csv", links)
    R.archive(
        out / "composition_accounts.npz",
        probabilities=prob,
        binary_log_odds=odds,
        native_scores=native,
        native_logits=logits,
        probability_J_mean=pmean_j,
        probability_J_sum=psum_j,
        binary_log_odds_J_sum=odds_j,
        native_score_J_mean=np_jmean,
        native_log_odds_J_sum=nlog_j,
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "event_composition_accounts.csv",
            "native_composition_accounts.csv",
            "source_context_updates.csv",
            "descriptive_links.csv",
            "composition_accounts.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def main():
    """Run only the requested finite stage; no retries or automatic new packets."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "score", "analyze"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    {"prepare": prepare, "score": score, "analyze": analyze}[a.stage](a.out)


if __name__ == "__main__":
    main()
