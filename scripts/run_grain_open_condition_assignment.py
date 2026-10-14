#!/usr/bin/env python3
"""WP2 runner: learn condition assignment from source clauses, propagate into the
frozen two-cell instrument, and compare oracle / soft / hard / abstain scoring.

Reuses cached clause vectors (condition_scope_r1, paraphrase-distilroberta) and
the handbook rebuild of the social instrument (all three seeds). No new neural
calls; application truth is opened only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.contextual_instrument import PROFILE_TARGETS, score_records
from grain_research.open_condition import (
    assign,
    flip_mixture,
    learn_prototype_assigner,
    score_person_open,
    source_clause_error,
)
from grain_research.pilot_transport import restore_units

BASE = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1"
SCOPE = BASE / "condition_scope_r1"
BANK = BASE / "condition_accumulation_r1/public_scored_bank.csv"
REBUILD = (
    BASE / "claude_todo_closure_20260913/d02_handbook_rebuild/local_instrument_rebuild"
)


def loadlines(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metrics(pred, truth, var=None):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if len(pred) == 0:
        return {"n": 0, "rmse": None, "bias": None, "r": None, "coverage95": None}
    err = pred - truth
    out = {
        "n": len(pred),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r": float(np.corrcoef(pred, truth)[0, 1]) if pred.std() > 1e-12 else None,
    }
    if var is not None:
        out["coverage95"] = float(
            np.mean(np.abs(err) <= 1.959963984540054 * np.sqrt(np.asarray(var)))
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--query", default="social_enjoyment_event")
    parser.add_argument("--instrument", default="social_condition_profile_v1")
    parser.add_argument("--posterior", default="lda1d", choices=("isotropic", "lda1d"))
    parser.add_argument(
        "--clause-correlated",
        action="store_true",
        help="share assignment uncertainty across events with the same clause",
    )
    parser.add_argument("--source-error-mixture", action="store_true")
    parser.add_argument("--full-assignment-moment", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    spans = pd.read_csv(SCOPE / "public_condition_spans.csv")
    vectors = np.load(SCOPE / "vectors.npy")
    bank = pd.read_csv(BANK)
    assert len(spans) == len(vectors) == len(bank)
    merged = bank.merge(
        spans[["case_id", "condition_text"]], on="case_id", validate="one_to_one"
    )
    merged["vector_index"] = np.arange(len(merged))
    # identical clauses must have identical vectors (alignment check)
    clause_vec = {}
    align_ok = True
    for text, group in merged.groupby("condition_text"):
        vs = vectors[group.vector_index.to_numpy()]
        align_ok &= bool(np.allclose(vs, vs[0], atol=1e-6))
        clause_vec[text] = vs[0]
    q = merged[merged["query"] == args.query]
    source, held = q[q.partition == "source"], q[q.partition == "held"]
    model = learn_prototype_assigner(
        vectors[source.vector_index.to_numpy()],
        source.condition.to_numpy(),
        posterior=args.posterior,
    )
    error_model = source_clause_error(
        vectors[source.vector_index.to_numpy()],
        source.condition.to_numpy(),
        source.condition_text.to_numpy(),
    )
    epsilon = error_model["epsilon"] if args.source_error_mixture else 0.0
    pi_held = assign(vectors[held.vector_index.to_numpy()], model)
    pi_source = assign(vectors[source.vector_index.to_numpy()], model)
    assignment = {
        "vector_alignment_identical_clauses": align_ok,
        "source_rows": len(source),
        "held_rows": len(held),
        "source_unique_clauses": int(source.condition_text.nunique()),
        "held_unique_clauses": int(held.condition_text.nunique()),
        "source_accuracy": float(
            np.mean((pi_source >= 0.5) == (source.condition.to_numpy() == 1))
        ),
        "held_accuracy": float(
            np.mean((pi_held >= 0.5) == (held.condition.to_numpy() == 1))
        ),
        "held_mean_abs_margin": float(np.mean(np.abs(pi_held - 0.5))),
        "held_abstain_share_tau": float(np.mean(np.abs(pi_held - 0.5) < args.tau)),
        "held_posteriors_by_clause": {
            t: float(assign(np.array([clause_vec[t]]), model)[0])
            for t in sorted(held.condition_text.unique())
        },
    }
    # Instrument propagation over all rebuilt seeds.
    margins = json.loads((REBUILD / "reader_cache.json").read_text())["margins"]
    records_out, per_variant = (
        [],
        {
            v: {t: {"pred": [], "truth": [], "var": []} for t in PROFILE_TARGETS}
            for v in ("oracle", "soft", "hard", "abstain")
        },
    )
    assign_eval = {"events": 0, "correct": 0, "abstained": 0}
    for bundle_path in sorted(
        (REBUILD / "instruments").glob(f"{args.instrument}_*.json")
    ):
        bundle = json.loads(bundle_path.read_text())
        seed = bundle["seed"]
        raw = loadlines(
            REBUILD / "raw" / f"{args.instrument}_{seed}_evaluation_held.jsonl"
        )
        truth = {
            t["author"]: t["nominal"]
            for t in loadlines(
                REBUILD / "truth" / f"{args.instrument}_{seed}_evaluation_held.jsonl"
            )
        }
        oracle, rows = score_records(
            raw, margins, bundle, clock_unit=bundle["spec"]["clock_unit"], explain=False
        )
        by_author = {}
        for row in rows:
            by_author.setdefault(row["author"], []).append(row)
        for author, evs in by_author.items():
            clauses = [r["text"].split(", ")[0] for r in evs]
            pis = assign(np.array([clause_vec[c] for c in clauses]), model)
            pis = flip_mixture(pis, epsilon)
            truth_z = np.array([r["condition"] for r in evs])
            assign_eval["events"] += len(evs)
            assign_eval["correct"] += int(np.sum((pis >= 0.5) == (truth_z == 1)))
            assign_eval["abstained"] += int(np.sum(np.abs(pis - 0.5) < args.tau))
            # observation-level mixture of the conditional calibration
            margins_v = np.array(
                [
                    margins[
                        __import__(
                            "grain_research.contextual_instrument",
                            fromlist=["text_key"],
                        ).text_key(bundle["spec"]["query"], r["text"])
                    ]
                    for r in evs
                ]
            )
            y_minus = restore_units(
                margins_v, -np.ones(len(evs)), bundle["calibration"]
            )
            y_plus = restore_units(margins_v, np.ones(len(evs)), bundle["calibration"])
            y_mix = (1 - pis) * y_minus + pis * y_plus
            clause_ids = np.array(clauses) if args.clause_correlated else None
            for variant in ("soft", "hard", "abstain"):
                hard_pi = (pis >= 0.5).astype(float)
                values = (
                    (1 - hard_pi) * y_minus + hard_pi * y_plus
                    if variant == "hard"
                    else y_mix
                )
                res = score_person_open(
                    values,
                    pis,
                    bundle,
                    mode=variant,
                    tau=args.tau,
                    clause_ids=clause_ids,
                    full_moment=args.full_assignment_moment,
                )
                if res["status"] != "open_condition_working_measurement":
                    continue
                for t in PROFILE_TARGETS:
                    per_variant[variant][t]["pred"].append(res["targets"][t]["score"])
                    per_variant[variant][t]["truth"].append(truth[author][t])
                    per_variant[variant][t]["var"].append(res["targets"][t]["variance"])
            for t in PROFILE_TARGETS:
                per_variant["oracle"][t]["pred"].append(
                    oracle[author]["targets"][t]["score"]
                )
                per_variant["oracle"][t]["truth"].append(truth[author][t])
                per_variant["oracle"][t]["var"].append(
                    oracle[author]["targets"][t]["variance"]
                )
            records_out.append(
                {
                    "seed": seed,
                    "author": author,
                    "events": len(evs),
                    "mean_pi_margin": float(np.mean(np.abs(pis - 0.5))),
                    "assign_acc": float(np.mean((pis >= 0.5) == (truth_z == 1))),
                }
            )
    table = []
    for variant, targets in per_variant.items():
        for t, d in targets.items():
            m = metrics(d["pred"], d["truth"], d["var"])
            table.append({"variant": variant, "target": t, **m})
    table = pd.DataFrame(table)
    table.to_csv(out / "variant_metrics.csv", index=False)
    pd.DataFrame(records_out).to_csv(out / "author_assignment.csv", index=False)
    summary = {
        "assignment": assignment,
        "instrument_events": {
            **assign_eval,
            "accuracy": assign_eval["correct"] / max(assign_eval["events"], 1),
            "abstain_share": assign_eval["abstained"] / max(assign_eval["events"], 1),
        },
        "authors_scored": {
            v: len(per_variant[v]["level"]["pred"]) for v in per_variant
        },
        "metrics": {
            v: {
                t: metrics(
                    per_variant[v][t]["pred"],
                    per_variant[v][t]["truth"],
                    per_variant[v][t]["var"],
                )
                for t in PROFILE_TARGETS
            }
            for v in per_variant
        },
        "tau": args.tau,
        "posterior": args.posterior,
        "clause_correlated": args.clause_correlated,
        "new_neural_calls": 0,
        "source_error_model": error_model,
        "epsilon_applied": epsilon,
        "full_assignment_moment": args.full_assignment_moment,
        "scope": "public 128-sentence bank clause vectors (frozen encoder) and simulated instrument authors; not natural conditions or human validity",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "assignment": {
                    k: v
                    for k, v in assignment.items()
                    if k != "held_posteriors_by_clause"
                },
                "instrument_events": summary["instrument_events"],
                "metrics": summary["metrics"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
