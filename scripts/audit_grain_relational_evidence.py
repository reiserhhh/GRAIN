#!/usr/bin/env python3
"""Independent tensor contraction, block regression and inherited-address audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_query_moment_operator import read, sha, write


def independently_read(out):
    """Reconstruct frozen candidate coordinates without calling the production reader."""
    bank = pd.read_csv(out / "public_bank.csv").set_index("case_id")
    nli = pd.read_csv(out / "readers/public_nli.csv")
    if len(nli) != 256 or nli.audit_id.duplicated().any():
        raise AssertionError("Invalid NLI inventory")
    probabilities = nli[["p_contradiction", "p_neutral", "p_entailment"]].to_numpy()
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise AssertionError("Invalid NLI class probabilities")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    literal, candidates = np.zeros((32, 2, 2, 2, 2)), np.zeros((32, 2, 2, 2, 2))
    for ix, row in bank.iterrows():
        actor = int("your colleague" in row.partner)
        channel = int(" will " in row.partner)
        pole = int(" accept " in row.partner)
        stance = int(row.reply == "Yes.")
        if (actor, channel, pole, stance) != (
            row.actor,
            row.channel,
            row.orientation,
            row.stance,
        ):
            raise AssertionError("Public generator labels disagree with raw reference")
        literal[ix, actor, channel, pole, stance] = 1
        f = nli[nli.case_id == ix]
        total = f.p_entailment.sum()
        for candidate in f.itertuples():
            s = int(candidate.candidate_target == pole)
            candidates[
                ix, candidate.candidate_actor, candidate.candidate_channel, pole, s
            ] = candidate.p_entailment / total
    with np.load(out / "readers/observer_tensors.npz", allow_pickle=False) as observed:
        np.testing.assert_allclose(observed["literal"], literal, atol=1e-14)
        np.testing.assert_allclose(observed["nli"], candidates, atol=1e-12)
        np.testing.assert_allclose(observed["condition"], bank.condition)
    return bank, {
        "known_relation_upper": literal,
        "literal_reference": literal,
        "fixed_nli_candidates": candidates,
    }


def audit_blocks(out, bank, tensors):
    """Recompute raw tensor sums and the dependence-deleting contraction by hand."""
    events = pd.read_csv(out / "math/observations.csv")
    blocks = pd.read_csv(out / "math/block_scores.csv").set_index(
        ["seed", "person", "block", "world", "reader", "channel"]
    )
    if (
        events.duplicated(["seed", "person", "block", "occasion"]).any()
        or blocks.index.duplicated().any()
    ):
        raise AssertionError("Duplicate observations/blocks")
    maximum, count, unknown = 0.0, 0, 0
    signs = np.array([[1.0, -1.0], [-1.0, 1.0]])
    for key, group in events.groupby(["seed", "person", "block"]):
        rows = blocks.loc[key]
        cases = group.case_id.to_numpy()
        if group.condition.nunique() != 1 or not np.all(
            bank.loc[cases].condition == group.condition.iloc[0]
        ):
            raise AssertionError("Condition mismatch")
        for (world, reader, channel), row in rows.iterrows():
            expected = None
            if reader == "reply_only":
                expected = 2 * bank.loc[cases].stance.mean() - 1
            elif reader == "known_relation_upper" or world != "missing_antecedent":
                joint = tensors[reader][cases].mean(axis=0)
                if world == "unbound_relations" and reader != "known_relation_upper":
                    total = joint.sum()
                    p = joint.sum(axis=(0, 1, 3)) / total
                    s = joint.sum(axis=(0, 1, 2)) / total
                    expected = (p[1] - p[0]) * (s[1] - s[0])
                else:
                    selected = joint[0, channel]
                    expected = float((selected * signs).sum() / selected.sum())
            if expected is None:
                if pd.notna(row.score) or row.status != "unresolved":
                    raise AssertionError("Missing context became a scored person")
                unknown += 1
            else:
                maximum = max(maximum, abs(expected - row.score))
                count += 1
    if maximum > 1e-10:
        raise AssertionError(f"Wrong contraction {maximum}")
    return {
        "block_values_checked": count,
        "missing_blocks_checked": unknown,
        "maximum_error": maximum,
        "unique_observations": len(events),
    }


def audit_profiles(out, cfg):
    """Independent least squares and cluster sandwich, followed by all reported metrics."""
    blocks = pd.read_csv(out / "math/block_scores.csv")
    predictions = pd.read_csv(out / "math/profile_predictions.csv")
    index = ["world", "reader", "seed", "person", "channel"]
    scores = predictions.set_index(index + ["target"])
    maximum, fits, missing = 0.0, 0, 0
    for key, group in blocks.groupby(index):
        f = group.dropna(subset=["score"])
        if not len(f):
            if scores.loc[key].score.notna().any():
                raise AssertionError("Unavailable measurement was imputed")
            missing += 1
            continue
        x, y = np.c_[np.ones(len(f)), f.condition.to_numpy()], f.score.to_numpy()
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        bread = np.linalg.inv(x.T @ x)
        g = x * (y - x @ beta)[:, None]
        covariance = len(y) / (len(y) - 2) * bread @ g.T @ g @ bread
        for target, d in (
            ("level", [1, 0]),
            ("response", [0, 1]),
            ("new_condition", [1, cfg["new_condition"]]),
            ("finite_change", [0, 1 + cfg["new_condition"]]),
        ):
            d = np.asarray(d)
            point = float(d @ beta)
            radius = t.ppf(0.975, len(y) - 2) * np.sqrt(max(0.0, d @ covariance @ d))
            row = scores.loc[key + (target,)]
            maximum = max(
                maximum,
                abs(row.score - point),
                abs(row.lower - point + radius),
                abs(row.upper - point - radius),
            )
        fits += 1
    metrics = pd.read_csv(out / "math/metrics.csv")
    for row in metrics.itertuples():
        f = predictions[
            (predictions.world == row.world)
            & (predictions.reader == row.reader)
            & (predictions.channel == row.channel)
            & (predictions.target == row.target)
        ].dropna(subset=["score"])
        if len(f) != row.n:
            raise AssertionError("Metric sample size mismatch")
        if len(f):
            error = f.score - f.truth
            maximum = max(
                maximum,
                abs(np.sqrt(np.mean(error**2)) - row.rmse),
                abs(error.mean() - row.bias),
            )
            if f.score.std() <= 1e-10 and pd.notna(row.pearson):
                raise AssertionError("Correlation assigned to a constant score")
    if maximum > 1e-9:
        raise AssertionError(f"Profile replay failure {maximum}")
    contrasts = []
    rng = np.random.default_rng(91599)
    for channel in range(2):
        for target in ("level", "response", "new_condition", "finite_change"):
            pairs = [
                (
                    "literal_reference",
                    "unbound_relations",
                    "literal_reference",
                    "complete_relation",
                ),
                (
                    "reply_only",
                    "complete_relation",
                    "fixed_nli_candidates",
                    "complete_relation",
                ),
            ]
            for a_reader, a_world, b_reader, b_world in pairs:
                a = scores.xs((a_world, a_reader), level=("world", "reader")).xs(
                    (channel, target), level=("channel", "target")
                )
                b = (
                    scores.xs((b_world, b_reader), level=("world", "reader"))
                    .xs((channel, target), level=("channel", "target"))
                    .loc[a.index]
                )
                ea, eb = (
                    (a.score - a.truth).to_numpy() ** 2,
                    (b.score - b.truth).to_numpy() ** 2,
                )
                seed_ix = [
                    np.flatnonzero(a.index.get_level_values("seed") == s)
                    for s in cfg["seeds"]
                ]
                draws = []
                for _ in range(cfg["bootstrap_draws"]):
                    ix = np.concatenate(
                        [rng.choice(v, len(v), replace=True) for v in seed_ix]
                    )
                    draws.append(np.sqrt(ea[ix].mean()) - np.sqrt(eb[ix].mean()))
                contrasts.append(
                    {
                        "channel": channel,
                        "target": target,
                        "comparison": f"{a_world}/{a_reader} minus {b_world}/{b_reader}",
                        "delta_rmse": np.sqrt(ea.mean()) - np.sqrt(eb.mean()),
                        "lower95": np.quantile(draws, 0.025),
                        "upper95": np.quantile(draws, 0.975),
                        "positive_seeds": sum(
                            ea[ix].mean() > eb[ix].mean() for ix in seed_ix
                        ),
                    }
                )
    pd.DataFrame(contrasts).to_csv(out / "paired_comparisons.csv", index=False)
    return {
        "profile_fits_checked": fits,
        "unavailable_profiles_checked": missing,
        "metric_rows": len(metrics),
        "maximum_error": maximum,
    }


def audit_natural(out):
    """Recheck source-event addresses and the deliberate absence of semantic truth."""
    relations = {
        r["event_id"]: r
        for r in (
            json.loads(v)
            for v in (out / "natural/private_relations.jsonl").read_text().splitlines()
        )
    }
    joins = pd.read_csv(out / "natural/private_query_relation_join.csv")
    for row in joins.itertuples():
        relation = relations[row.event_id]
        if relation["predicate_span"] != [row.predicate_start, row.predicate_end]:
            raise AssertionError("Raw event address changed")
        if relation["subject_reference"] != row.subject_reference:
            raise AssertionError("Actor binding changed")
        for key in (
            "target_orientation",
            "semantic_actor",
            "event_time",
            "personal_score",
        ):
            if relation[key] is not None:
                raise AssertionError("Unobserved semantic variable was invented")
    if joins.personal_score.notna().any() or joins.semantic_binding_resolved.any():
        raise AssertionError("Syntax was promoted to personal interpretation")
    return {
        "query_event_addresses_checked": len(joins),
        "unique_relations": len(relations),
        "human_accuracy_claim": False,
    }


def main():
    """Verify code freeze and save portable computational evidence, not human validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    contract, cfg = read(out / "contract.json"), read(out / "config.json")
    for group in ("code", "natural"):
        for name, expected in contract[group].items():
            if sha(ROOT / name) != expected:
                raise AssertionError(f"Frozen file changed: {name}")
    for name, expected in contract["inputs"].items():
        if sha(out / name) != expected:
            raise AssertionError(f"Frozen packet changed: {name}")
    with threadpool_limits(limits=1):
        bank, tensors = independently_read(out)
        result = {
            "status": "PASS_COMPUTATIONAL_AND_SCOPE",
            "blocks": audit_blocks(out, bank, tensors),
            "profiles": audit_profiles(out, cfg),
            "natural": audit_natural(out),
            "model_forward_repeated": False,
            "candidate_weights_calibrated": False,
        }
    write(out / "audit.json", result)
    files = [
        v
        for v in sorted(out.rglob("*"))
        if v.is_file()
        and not v.name.startswith("._")
        and v.name != "delivery_manifest.json"
    ]
    write(
        out / "delivery_manifest.json", {str(v.relative_to(out)): sha(v) for v in files}
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
