#!/usr/bin/env python3
"""Bounded relational evidence, fixed NLI application and author response study."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_accumulation import profile_fit
from grain_research.relational_evidence import (
    atom,
    candidate_joint,
    factorize,
    marginalize,
    query_kernel,
    query_measure,
    read_controlled_pair,
    symbolic_relations,
)
from scripts.run_grain_evidence_calibration import infer_pairs
from scripts.run_grain_query_moment_operator import read, sha, write

BASE = ROOT / "results/grain_vector_mechanisms"
NATURAL = BASE / "all_plan_closure_v1/query_event_r2"
GRAPH = BASE / "all_plan_closure_v1/source_event_r4"


def public_bank():
    """Cross 32 declared observations; latent codes are for evaluation only."""
    rows = []
    for i, (z, o, c, p, s) in enumerate(itertools.product(range(2), repeat=5)):
        condition = ("practical", "social")[z]
        tense, when = ("will", "tomorrow") if c else ("did", "yesterday")
        subject, action = ("you", "your colleague")[o], ("refuse", "accept")[p]
        partner = f"In a {condition} situation, {tense} {subject} {action} the offered help {when}?"
        rows.append(
            {
                "case_id": i,
                "condition": 2 * z - 1,
                "actor": o,
                "channel": c,
                "orientation": p,
                "stance": s,
                "target_sign": (2 * p - 1) * (2 * s - 1),
                "partner": partner,
                "reply": ("No.", "Yes.")[s],
            }
        )
    return pd.DataFrame(rows)


def requests_for_bank(bank):
    """Pass raw premises and fixed joint queries, never latent codes, to NLI."""
    rows = []
    for row in bank[["case_id", "partner", "reply"]].itertuples():
        for o, c, target in itertools.product(range(2), repeat=3):
            person = "the respondent" if o == 0 else "the respondent's colleague"
            action = (
                ("refused", "accepted") if c == 0 else ("will refuse", "will accept")
            )[target]
            when = "yesterday" if c == 0 else "tomorrow"
            rows.append(
                {
                    "audit_id": f"{row.case_id}:{o}:{c}:{target}",
                    "case_id": row.case_id,
                    "candidate_actor": o,
                    "candidate_channel": c,
                    "candidate_target": target,
                    "sentence": f"Partner: {row.partner}\nRespondent: {row.reply}",
                    "item_text": f"The respondent says that {person} {action} the offered help {when}.",
                }
            )
    return pd.DataFrame(rows)


def prepare(out, config):
    """Freeze a new experiment directory before inspecting model responses."""
    if out.exists():
        raise FileExistsError("Use a fresh directory; do not overwrite evidence")
    out.mkdir(parents=True)
    cfg = read(config)
    write(out / "config.json", cfg)
    bank = public_bank()
    bank.to_csv(out / "public_bank.csv", index=False)
    paths = [
        Path(__file__),
        config,
        ROOT / "grain_research/relational_evidence.py",
        ROOT / "grain_research/evidence_accumulation.py",
        ROOT / "scripts/run_grain_evidence_calibration.py",
        ROOT / "scripts/run_grain_entailment_reference.py",
        ROOT / "docs/GRAIN_RELATIONAL_EVIDENCE_PACKET_V1.md",
    ]
    write(
        out / "contract.json",
        {
            "code": {str(v.relative_to(ROOT)): sha(v) for v in paths},
            "inputs": {
                "public_bank.csv": sha(out / "public_bank.csv"),
                "config.json": sha(out / "config.json"),
            },
            "natural": {
                str(v.relative_to(ROOT)): sha(v)
                for v in (
                    NATURAL / "private_candidates.csv",
                    NATURAL / "summary.json",
                    GRAPH / "private_graphs.jsonl",
                )
            },
            "nli_pairs": len(requests_for_bank(bank)),
            "human_labels_loaded": False,
            "new_training": False,
            "meps_access": False,
        },
    )


def verify(out, stage):
    """Check frozen code/data and make a single-use stage directory."""
    contract = read(out / "contract.json")
    for mapping in (contract["code"], contract["natural"]):
        for name, expected in mapping.items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Frozen file changed: {name}")
    for name, expected in contract["inputs"].items():
        if sha(out / name) != expected:
            raise ValueError(f"Changed packet input: {name}")
    folder = out / stage
    if folder.exists():
        raise FileExistsError(f"Stage already exists: {stage}")
    folder.mkdir()
    return read(out / "config.json"), folder


def readers(out):
    """Apply exactly 256 fixed English NLI queries and a literal grammar reference."""
    cfg, folder = verify(out, "readers")
    bank = pd.read_csv(out / "public_bank.csv")
    requests = requests_for_bank(bank)
    if len(requests) > cfg["nli_pair_budget"]:
        raise ValueError("NLI budget exceeded")
    start = time.monotonic()
    scores = infer_pairs(requests[["audit_id", "sentence", "item_text"]])
    frame = requests.merge(scores, on="audit_id", validate="one_to_one")
    frame.to_csv(folder / "public_nli.csv", index=False)
    literal, nli, conditions, summaries = [], [], [], []
    for row in bank[["case_id", "partner", "reply"]].itertuples():
        parsed = read_controlled_pair(row.partner, row.reply)
        weights = np.zeros((2, 2, 2))
        f = frame[frame.case_id == row.case_id]
        for candidate in f.itertuples():
            weights[
                candidate.candidate_actor,
                candidate.candidate_channel,
                candidate.candidate_target,
            ] = candidate.p_entailment
        joint = candidate_joint(weights, parsed["orientation"])
        literal.append(parsed["joint"])
        nli.append(joint)
        conditions.append(parsed["condition"])
        top = np.unravel_index(np.argmax(weights), weights.shape)
        summaries.append(
            {
                "case_id": row.case_id,
                "top_actor": top[0],
                "top_channel": top[1],
                "top_target": top[2],
                "max_entailment": weights.max(),
                "candidate_sum": weights.sum(),
                "source_sha256": parsed["source_sha256"],
            }
        )
    np.savez_compressed(
        folder / "observer_tensors.npz",
        literal=np.array(literal),
        nli=np.array(nli),
        condition=conditions,
    )
    summary = pd.DataFrame(summaries).merge(bank, on="case_id", validate="one_to_one")
    summary["top_correct"] = (
        (summary.top_actor == summary.actor)
        & (summary.top_channel == summary.channel)
        & (2 * summary.top_target - 1 == summary.target_sign)
    )
    summary.to_csv(folder / "public_reader_comparison.csv", index=False)
    write(
        folder / "summary.json",
        {
            "public_cases": len(bank),
            "nli_pairs": len(frame),
            "top_joint_correct": int(summary.top_correct.sum()),
            "literal_scope": "Exact declared grammar only; not open-language accuracy",
            "nli_scope": "Normalized candidate support, NOT calibrated joint posterior",
            "new_neural_calls": len(frame),
            "wall_seconds": time.monotonic() - start,
            "training": False,
            "private_data": False,
        },
    )


def simulate(cfg, bank, tensors):
    """Generate events; evaluation truth never enters the observation readers."""
    keys = {
        (int(r.condition), r.actor, r.channel, r.orientation, r.stance): r.case_id
        for r in bank.itertuples()
    }
    known = np.array(
        [atom(r.actor, r.channel, r.orientation, r.stance) for r in bank.itertuples()]
    )
    views = {
        "known_relation_upper": known,
        "literal_reference": tensors["literal"],
        "fixed_nli_candidates": tensors["nli"],
    }
    blocks, events, truths = [], [], []
    started = time.process_time()
    reply_signs = bank.set_index("case_id").stance.to_numpy() * 2 - 1
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        for person in range(cfg["authors_per_seed"]):
            if time.process_time() - started > cfg["math_cpu_budget"]:
                raise TimeoutError("Declared math CPU budget exceeded")
            beta = rng.uniform(-0.3, 0.3, (2, 2, 2))
            for o, c in itertools.product(range(2), repeat=2):
                truths.append(
                    {
                        "seed": seed,
                        "person": person,
                        "actor": o,
                        "channel": c,
                        "level": beta[o, c, 0],
                        "response": beta[o, c, 1],
                    }
                )
            for zi, z in enumerate(cfg["conditions"]):
                for block in range(cfg["blocks_per_condition"]):
                    ids = []
                    for o, c in itertools.product(range(2), repeat=2):
                        size = cfg["events_per_actor_channel_block"]
                        poles = np.tile([0, 1], size // 2)
                        rng.shuffle(poles)
                        for j, p in enumerate(poles):
                            y = (
                                1
                                if rng.random() < (1 + beta[o, c] @ [1, z]) / 2
                                else -1
                            )
                            s = int(y * (2 * p - 1) > 0)
                            case = keys[(int(z), o, c, int(p), s)]
                            ids.append(case)
                            events.append(
                                {
                                    "seed": seed,
                                    "person": person,
                                    "block": zi * cfg["blocks_per_condition"] + block,
                                    "occasion": len(ids) - 1,
                                    "condition": z,
                                    "case_id": case,
                                }
                            )
                    mixtures = {
                        name: value[ids].mean(axis=0) for name, value in views.items()
                    }
                    reply = float(reply_signs[ids].mean())
                    for world in cfg["worlds"]:
                        for reader in cfg["readers"]:
                            if reader == "reply_only":
                                measures = [
                                    {
                                        "score": reply,
                                        "eligible_mass": None,
                                        "unresolved_mass": None,
                                        "status": "untyped_reply_polarity",
                                    }
                                ] * 2
                            else:
                                joint = mixtures[reader]
                                if reader != "known_relation_upper":
                                    if world == "missing_antecedent":
                                        joint = np.zeros_like(joint)
                                    elif world == "unbound_relations":
                                        joint = factorize(joint)
                                measures = [
                                    query_measure(joint, actor=0, channel=c)
                                    for c in range(2)
                                ]
                            for c, measure in enumerate(measures):
                                blocks.append(
                                    {
                                        "world": world,
                                        "reader": reader,
                                        "seed": seed,
                                        "person": person,
                                        "block": zi * cfg["blocks_per_condition"]
                                        + block,
                                        "condition": z,
                                        "channel": c,
                                        **measure,
                                    }
                                )
    return pd.DataFrame(blocks), pd.DataFrame(events), pd.DataFrame(truths)


def score_profiles(blocks, truths, cfg):
    """Reuse event-profile WLS with independent blocks and separate evidence channels."""
    truth = truths[truths.actor == 0].set_index(["seed", "person", "channel"])
    fits, scores = {}, []
    for keys, f in blocks.groupby(["world", "reader", "seed", "person", "channel"]):
        world, reader, seed, person, channel = keys
        eligible = f[f.score.notna()].sort_values("block")
        fit = None
        if len(eligible) >= 3:
            fit = profile_fit(
                eligible.condition.to_numpy(),
                eligible.score.to_numpy(),
                eligible.block.to_numpy(),
                balance_events=False,
                cluster_errors=True,
            )
        key = ":".join(map(str, keys))
        fits[key] = fit
        true = truth.loc[(seed, person, channel), ["level", "response"]].to_numpy(float)
        for target, direction in (
            ("level", [1, 0]),
            ("response", [0, 1]),
            ("new_condition", [1, cfg["new_condition"]]),
            ("finite_change", [0, 1 + cfg["new_condition"]]),
        ):
            d = np.asarray(direction, float)
            score = None if fit is None else float(d @ fit["beta"])
            radius = (
                None
                if fit is None
                else float(
                    t.ppf(0.975, len(eligible) - 2)
                    * np.sqrt(max(0, d @ fit["covariance"] @ d))
                )
            )
            scores.append(
                {
                    "world": world,
                    "reader": reader,
                    "seed": seed,
                    "person": person,
                    "channel": channel,
                    "target": target,
                    "score": score,
                    "truth": float(d @ true),
                    "lower": None if score is None else score - radius,
                    "upper": None if score is None else score + radius,
                    "blocks": len(eligible),
                }
            )
    return fits, pd.DataFrame(scores)


def math(out):
    """Run three information ablations, not three new human populations."""
    cfg, folder = verify(out, "math")
    start = time.process_time()
    bank = pd.read_csv(out / "public_bank.csv")
    with np.load(out / "readers/observer_tensors.npz", allow_pickle=False) as loaded:
        tensors = {k: loaded[k] for k in loaded.files}
    blocks, events, truths = simulate(cfg, bank, tensors)
    fits, scores = score_profiles(blocks, truths, cfg)
    for name, frame in (
        ("block_scores", blocks),
        ("observations", events),
        ("truth_profiles", truths),
        ("profile_predictions", scores),
    ):
        frame.to_csv(folder / f"{name}.csv", index=False)
    write(folder / "fits.json", fits)
    rows = []
    for keys, part in scores.groupby(["world", "reader", "channel", "target"]):
        f = part[part.score.notna()]
        error = f.score - f.truth
        rows.append(
            dict(zip(["world", "reader", "channel", "target"], keys, strict=True))
            | {
                "n": len(f),
                "missing": int(part.score.isna().sum()),
                "rmse": np.sqrt(np.mean(error**2)) if len(f) else None,
                "bias": error.mean() if len(f) else None,
                "pearson": f.score.corr(f.truth)
                if len(f) and f.score.std() > 1e-10 and f.truth.std() > 1e-10
                else None,
                "coverage95_working": (
                    (f.lower <= f.truth) & (f.truth <= f.upper)
                ).mean()
                if len(f)
                else None,
            }
        )
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    positive = (atom(0, 1, 0, 0) + atom(0, 1, 1, 1)) / 2
    negative = (atom(0, 1, 0, 1) + atom(0, 1, 1, 0)) / 2
    basis = np.eye(16).reshape(16, 2, 2, 2, 2)
    operator = np.array([np.concatenate(marginalize(x)) for x in basis]).T
    q = query_kernel().ravel()
    projection = operator.T @ np.linalg.lstsq(operator.T, q, rcond=None)[0]
    write(
        folder / "information_counterexample.json",
        {
            "positive": positive,
            "negative": negative,
            "shared_marginals": marginalize(positive),
            "positive_measure": query_measure(positive),
            "negative_measure": query_measure(negative),
            "marginalized_measure": query_measure(factorize(positive)),
            "marginal_map_rank": np.linalg.matrix_rank(operator),
            "nullity": 16 - np.linalg.matrix_rank(operator),
            "unrepresented_query_norm": np.linalg.norm(q - projection),
        },
    )
    write(
        folder / "summary.json",
        {
            "authors": len(truths) // 4,
            "unique_observations": len(events),
            "block_rows": len(blocks),
            "profile_rows": len(scores),
            "cpu_seconds": time.process_time() - start,
            "no_new_condition_observations": True,
            "natural_personality_accuracy": False,
            "missing_context_nli_rerun": False,
        },
    )


def natural(out):
    """Bind existing source-only PANDORA event/query addresses without new inference."""
    _, folder = verify(out, "natural")
    source_summary = read(NATURAL / "summary.json")
    if (
        sha(NATURAL / "private_candidates.csv")
        != source_summary["files"]["private_candidates.csv"]
    ):
        raise ValueError("Inherited query artifact changed")
    candidates = pd.read_csv(NATURAL / "private_candidates.csv")
    selected_ids = set(candidates.event_id)
    records = {}
    with (GRAPH / "private_graphs.jsonl").open() as handle:
        for line in handle:
            graph = json.loads(line)
            for event in graph["events"]:
                if event["event_id"] in selected_ids:
                    records[event["event_id"]] = symbolic_relations(
                        event, graph["raw_sha256"]
                    )
    if set(records) != selected_ids:
        raise ValueError("Missing source event")
    with (folder / "private_relations.jsonl").open("w") as handle:
        for row in records.values():
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    fields = [
        "audit_id",
        "event_id",
        "source_row",
        "predicate_start",
        "predicate_end",
        "route",
        "subject_reference",
        "intention_embedded",
        "ancestor_negation",
        "local_negation",
        "modal",
        "factuality",
    ]
    joins = candidates[fields].copy()
    joins["semantic_binding_resolved"] = False
    joins["personal_score"] = None
    joins.to_csv(folder / "private_query_relation_join.csv", index=False)
    counts = (
        joins.groupby(["route", "intention_embedded"])
        .size()
        .rename("query_events")
        .reset_index()
    )
    counts.to_csv(folder / "relation_coverage.csv", index=False)
    write(
        folder / "summary.json",
        {
            "queries": int(joins.audit_id.nunique()),
            "query_events": len(joins),
            "unique_events": len(records),
            "source_comments": int(joins.source_row.nunique()),
            "intention_cue_query_events": int(joins.intention_embedded.sum()),
            "source_nested_query_events": int(
                (joins.route == "writer_subject_embedded_source").sum()
            ),
            "unresolved_participant_query_events": int(
                (joins.route == "unresolved_participant").sum()
            ),
            "natural_complete_semantic_bindings": 0,
            "new_parser_or_neural_calls": 0,
            "labels_or_meps_loaded": False,
            "fitting": False,
            "scope": "Addressed syntax/roles are usable; observed event time, factuality and target stance remain unresolved",
        },
    )


def main():
    """Run one explicit frozen stage; failed stages are retained for diagnosis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "readers", "math", "natural"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_relational_evidence.json"
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == "prepare":
            prepare(args.output, args.config)
        else:
            globals()[args.stage](args.output)
    print(f"{args.stage}: complete", flush=True)


if __name__ == "__main__":
    main()
