#!/usr/bin/env python3
"""Bounded natural-anchor probes and history-aware author construction.

No fitting on natural text or psychological labels. Raw corpus, modified natural
probes and pseudonymous scores remain inside the local gitignored results tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_grain_math_response import encode, sha, write_csv, write_json

from grain_research import evidence_history as H
from grain_research.construction import math_evidence as E
from grain_research.construction import math_response as M

OLD = ROOT / "results/grain_math_response_v1/r1"
PAIRS = (
    ROOT
    / "results/grain_vector_mechanisms/all_plan_logic_v1"
    / "natural_antecedent_scan_r1/private_antecedent_pairs.csv"
)
DEFAULT_OUT = ROOT / "results/grain_math_response_v1/evidence_r2"


def freeze_inputs(out: Path, old: Path, pairs: Path) -> None:
    """Freeze target, selection rules, source programs and finite resource scope."""
    contract = {
        "pair_cache": str(pairs),
        "pair_sha256": sha(pairs),
        "reader_bundles": {
            r: sha(old / f"06/reader_bundles/{r}_joint_binding.npz")
            for r in ("e5", "bge")
        },
        "prior_confirmation_sha256": sha(old / "06/00/material_bank.jsonl"),
        "target": "reported own initiation/coordination expression",
        "natural_selection": "English 8..150 words, explicit first-person past action and team cue; <=56-word exact action span; source-order first128, no score selection",
        "qualification": "natural candidates are not independent human annotations; condition unknown unless explicitly established",
        "natural_protocol": "frozen scoring of whole text, span, self envelope, denial, other quotation, hypothetical and mixed subjects; edits are counterfactuals, not observed events",
        "aggregation_protocol": "frozen confirmation scores; four blocks x2 conditions x2 distinct events x2 reports; source256/application128,3 seeds; learn source covariance without latent parameters",
        "fixed_reference": {"-1": 0.5, "1": 0.5},
        "permission": "reuse previously authorized local PANDORA cache; no redistribution or external processing",
        "privacy": "bank, originals, exact spans and per-anchor scores local only; reports contain aggregates only",
        "neural_training": 0,
        "api": 0,
        "max_natural_anchors": 128,
        "max_new_text_encodings_per_reader": 1152,
        "code": {
            p: sha(ROOT / p)
            for p in (
                "grain_research/construction/math_evidence.py",
                "scripts/run_grain_math_evidence.py",
            )
        },
    }
    path = out / "00/protocol_before_run.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != contract:
            raise ValueError("Resume rejected: input/protocol/code changed")
    else:
        write_json(path, contract)


def prepare(out: Path, old: Path, pairs: Path) -> None:
    """Stream existing natural pair cache; never print its words or identities."""
    freeze_inputs(out, old, pairs)
    if (out / "00/material_bank.jsonl").exists():
        expected = json.loads((out / "00/selection.json").read_text())
        if expected["bank_sha256"] != sha(out / "00/material_bank.jsonl"):
            raise ValueError("Changed natural probe bank")
        return
    anchors, variants = [], []
    reasons = Counter()
    total = candidates = explicit_time = 0
    seen_ids = set()
    for chunk in pd.read_csv(pairs, chunksize=40000, low_memory=False):
        total += len(chunk)
        eligible = chunk[
            chunk.lang.eq("en")
            & chunk.word_count.between(8, 150)
            & chunk.body.astype(str).str.contains(E.ACTION, regex=True)
            & chunk.body.astype(str).str.contains(E.TEAM, regex=True)
        ]
        candidates += len(eligible)
        for row in eligible.itertuples():
            if str(row.id) in seen_ids:
                continue
            seen_ids.add(str(row.id))
            item = E.candidate_slice(str(row.body))
            reasons[item["status"]] += 1
            if item["status"] != "past_self_action_candidate" or len(anchors) >= 128:
                continue
            aid = f"a{len(anchors):03d}"
            anchors.append(
                {
                    "anchor_id": aid,
                    "private_comment_id": str(row.id),
                    "private_author_id": str(row.author),
                    "private_subreddit": str(row.subreddit),
                    "created_utc": row.created_utc,
                    "text": str(row.body),
                    **item,
                }
            )
            probe = {
                "raw": str(row.body),
                **E.probe_variants(item["fragment"], item["verb"]),
            }
            for name, text in probe.items():
                variants.append(
                    {
                        "material_id": f"{aid}:{name}",
                        "anchor_id": aid,
                        "variant": name,
                        "text": text,
                        "role": "diagnostic",
                        "condition": None,
                        "kind": "natural_observed"
                        if name == "raw"
                        else "natural_span"
                        if name == "fragment"
                        else "synthetic_reframing",
                    }
                )
    bank = out / "00/material_bank.jsonl"
    bank.write_text("".join(json.dumps(v) + "\n" for v in variants))
    write_json(out / "00/private_anchors.json", anchors)
    write_json(
        out / "00/reader_registry.json",
        json.loads((old / "00/reader_registry.json").read_text()),
    )
    write_json(
        out / "00/selection.json",
        {
            "cache_rows": total,
            "lexical_candidates": candidates,
            "span_statuses": dict(reasons),
            "anchors": len(anchors),
            "distinct_authors": len({a["private_author_id"] for a in anchors}),
            "distinct_contexts": len({a["private_subreddit"] for a in anchors}),
            "confirmed_endpoint_conditions": explicit_time,
            "new_texts": len(variants),
            "bank_sha256": sha(bank),
            "scope": "target-enriched candidate coverage, not prevalence/accuracy/author trait scoring",
        },
    )
    print(
        f"prepared {len(anchors)} natural anchors, {len(variants)} bounded probes",
        flush=True,
    )


def load_bundle(old: Path, reader: str) -> dict:
    """Load a frozen source-only program, without refitting any coefficient."""
    with np.load(old / f"06/reader_bundles/{reader}_joint_binding.npz") as b:
        return {k: b[k] for k in b.files}


def natural_scores(out: Path, old: Path) -> None:
    """Measure frozen action responses; natural rows have no assigned gold score."""
    rows = [
        json.loads(s) for s in (out / "00/material_bank.jsonl").read_text().splitlines()
    ]
    metrics, paired_rows = {}, []
    for reader in ("e5", "bge"):
        x = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        bundle = load_bundle(old, reader)
        scores = M.score_bundle(bundle, x)
        z = (x - bundle["factor_mu"]) @ bundle["factor_w"] + bundle["factor_intercept"]
        np.save(out / f"scores_{reader}.npy", scores)
        write_csv(
            out / f"private_scores_{reader}.csv",
            [
                {
                    "material_id": row["material_id"],
                    "variant": row["variant"],
                    "T1": scores[i, 0],
                    "T2_raw_not_admissible": scores[i, 1],
                    "pole_coordinate": z[i, 0],
                    "ownership_coordinate": z[i, 1],
                    "condition_coordinate": z[i, 2],
                    "conditional_author_score": None,
                }
                for i, row in enumerate(rows)
            ],
        )
        indexed = {
            (row["anchor_id"], row["variant"]): scores[i, 0]
            for i, row in enumerate(rows)
        }
        reader_metrics = {}
        for high, low, meaning in (
            ("fragment", "denied_action", "assertion_minus_denial_not_waiting"),
            ("own_envelope", "other_account", "same_content_own_minus_other"),
            ("fragment", "hypothetical", "past_report_minus_counterfactual"),
            ("mixed_other", "wait_only", "irrelevant_other_report_contamination"),
            ("own_envelope", "fragment", "envelope_shift"),
            ("raw", "fragment", "whole_context_shift"),
        ):
            delta = np.array(
                [
                    indexed[a, high] - indexed[a, low]
                    for a in sorted({row["anchor_id"] for row in rows})
                ]
            )
            reader_metrics[meaning] = {
                "mean": float(delta.mean()),
                "ci": M.bootstrap_interval(delta),
                "positive_count": int((delta > 0).sum()),
                "n": len(delta),
                "rms": float(np.sqrt(np.mean(delta**2))),
            }
            paired_rows.extend(
                {"reader": reader, "contrast": meaning, "anchor": a, "delta": value}
                for a, value in zip(
                    sorted({row["anchor_id"] for row in rows}), delta, strict=True
                )
            )
        reader_metrics["T2_unknown_condition"] = {
            "raw_coordinate_rms": float(np.sqrt(np.mean(scores[:, 1] ** 2))),
            "admissible_conditional_person_scores": 0,
        }
        metrics[reader] = reader_metrics
    write_json(out / "natural_response_summary.json", metrics)
    write_csv(out / "private_paired_effects.csv", paired_rows)


def posterior_rows(model, y, records, truth, reader, seed, mode, prefix):
    """Evaluate one frozen working predictor, keeping oracle outside the fit."""
    result = H.score_records(model, y, records, mode)
    mse = np.mean((result["mean"] - truth) ** 2, axis=0)
    var = np.diag(result["covariance"])
    coverage = np.mean(np.abs(result["mean"] - truth) <= 1.96 * np.sqrt(var), axis=0)
    return {
        "reader": reader,
        "seed": seed,
        "mode": mode,
        "reports": len(records),
        "events": len({r["event_id"] for r in records}),
        "prefix_blocks": prefix,
        "level_rmse": float(np.sqrt(mse[0])),
        "response_rmse": float(np.sqrt(mse[1])),
        "level_coverage": float(coverage[0]),
        "response_coverage": float(coverage[1]),
        "level_working_sd": float(np.sqrt(var[0])),
        "response_working_sd": float(np.sqrt(var[1])),
    }, result


def history_experiment(out: Path, old: Path) -> None:
    """Connect new score units to source-learned event/block/report moments."""
    bank = [
        json.loads(s)
        for s in (old / "06/00/material_bank.jsonl").read_text().splitlines()
    ]
    summaries, duplicate_rows, update_rows, models, check_rows = [], [], [], {}, []
    for reader in ("e5", "bge"):
        bundle = load_bundle(old, reader)
        x = np.load(old / f"06/01B/embeddings/{reader}.npy").astype(float)
        scores = M.score_bundle(bundle, x)
        lookup = {}
        for row, score in zip(bank, scores, strict=True):
            p, r, o, c = row["bits"]
            if o == 1:
                lookup[row["family"], p, r, c] = float(score[0])
        families = sorted({r["family"] for r in bank})
        for seed in (71, 72, 73):
            source, _ = E.simulate_panels(lookup, families[:8], 256, seed)
            observed, evaluation = E.simulate_panels(
                lookup, families[8:], 128, seed + 1000
            )
            model = H.fit_source(source["values"], source["records"])
            models[f"{reader}:{seed}"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in model.items()
                if k != "empirical_covariance"
            }
            y, records = observed["values"], observed["records"]
            truth = evaluation["level_response"]
            for blocks in (1, 2, 4):
                take = [
                    i
                    for i, row in enumerate(records)
                    if int(row["block_id"][1:]) < blocks
                ]
                selected = [records[i] for i in take]
                for mode in ("full", "independent", "event_only"):
                    row, result = posterior_rows(
                        model, y[:, take], selected, truth, reader, seed, mode, blocks
                    )
                    summaries.append(row)
                    if blocks == 4 and mode == "full":
                        np.savez_compressed(
                            out / f"panel_{reader}_{seed}.npz",
                            values=y,
                            truth=truth,
                            mean=result["mean"],
                            covariance=result["covariance"],
                        )
            target = records[0]
            history = [
                {**r, "author_id": "toy_0", "value": float(y[0, i])}
                for i, r in enumerate(records)
            ]
            base = E.replay_qualified(history)
            profile = E.fixed_reference_profile(base["records"])
            aliases = [
                {
                    "author_id": "toy_0",
                    "record_id": f"alias{i}",
                    "kind": "alias",
                    "parent_id": target["record_id"],
                }
                for i in range(20)
            ]
            repeated = E.replay_qualified(history + aliases)
            after = E.fixed_reference_profile(repeated["records"])
            # Naive control treats each copy as a fresh row in the same condition.
            bad = [
                {
                    **target,
                    "record_id": f"bad{i}",
                    "event_id": f"bad_event{i}",
                    "value": float(y[0, 0]),
                }
                for i in range(20)
            ]
            naive = E.fixed_reference_profile(base["records"] + bad)
            duplicate_rows.append(
                {
                    "reader": reader,
                    "seed": seed,
                    "aliases": 20,
                    "ledger_level_change": after["level"] - profile["level"],
                    "ledger_response_change": after["response"] - profile["response"],
                    "naive_level_change": naive["level"] - profile["level"],
                    "naive_response_change": naive["response"] - profile["response"],
                    "original_events": profile["events"],
                    "after_events": after["events"],
                }
            )
            old_rows = records[:8]
            update = H.append_innovation(model, y[:, :8], old_rows, y[:, 8], records[8])
            direct = H.score_records(model, y[:, :9], records[:9])
            update_rows.append(
                {
                    "reader": reader,
                    "seed": seed,
                    "new_event_innovation_variance": update["innovation_variance"],
                    "update_batch_error": float(
                        np.max(np.abs(update["mean"] - direct["mean"]))
                    ),
                }
            )
            alias_record = {
                **records[0],
                "record_id": "numeric_alias",
                "origin_id": records[0]["record_id"],
            }
            numeric_records = [{**r, "origin_id": r["record_id"]} for r in records[:8]]
            exact = H.append_innovation(
                model, y[:, :8], numeric_records, y[:, 0], alias_record
            )
            revision = {
                **history[0],
                "kind": "revise",
                "record_id": "corrected",
                "parent_id": history[0]["record_id"],
                "value": -history[0]["value"],
            }
            revised = E.replay_qualified(history + [revision])
            retracted = E.replay_qualified(
                history
                + [
                    revision,
                    {
                        "author_id": "toy_0",
                        "record_id": "withdrawal",
                        "kind": "retract",
                        "parent_id": "corrected",
                    },
                ]
            )
            excluded = E.replay_qualified(
                history
                + [
                    {
                        **history[0],
                        "record_id": "unknown",
                        "event_id": "unknown_event",
                        "condition": None,
                    },
                    {
                        **history[0],
                        "record_id": "other",
                        "event_id": "other_event",
                        "qualification": "other_actual",
                    },
                ]
            )
            dequalified = E.replay_qualified(
                history
                + [
                    {
                        **history[0],
                        "kind": "revise",
                        "record_id": "dequalified",
                        "parent_id": history[0]["record_id"],
                        "qualification": "other_actual",
                    }
                ]
            )
            check_rows.append(
                {
                    "reader": reader,
                    "seed": seed,
                    "duplicate_innovation_variance": exact["innovation_variance"],
                    "duplicate_gain_norm": float(np.linalg.norm(exact["gain"])),
                    "revision_active_events": revised["active_events"],
                    "revision_removes_parent": not any(
                        r["record_id"] == history[0]["record_id"]
                        for r in revised["records"]
                    ),
                    "withdrawal_keeps_other_report": any(
                        r["event_id"] == history[0]["event_id"]
                        for r in retracted["records"]
                    ),
                    "excluded_rows": len(excluded["excluded"]),
                    "dequalified_revision_removes_old_parent": not any(
                        r["record_id"] == history[0]["record_id"]
                        for r in dequalified["records"]
                    ),
                    "dequalified_revision_not_scored": not any(
                        r["record_id"] == "dequalified" for r in dequalified["records"]
                    ),
                }
            )
    write_csv(out / "aggregation_metrics.csv", summaries)
    write_csv(out / "duplicate_controls.csv", duplicate_rows)
    write_csv(out / "innovation_updates.csv", update_rows)
    write_csv(out / "ledger_checks.csv", check_rows)
    write_json(out / "source_working_models.json", models)
    write_json(
        out / "history_scope.json",
        {
            "matched_base_authors": 3 * (256 + 128),
            "distinct_events_per_author": 16,
            "reports_per_author": 32,
            "paired_reader_evaluations": 2,
            "natural_authors_used_in_source_or_fit": 0,
            "simulation_noise": {"block_sd": 0.12, "report_sd": 0.04},
            "source_families": families[:8],
            "application_families": families[8:],
            "working_covariance_not_exact_posterior": True,
            "meaning": "author level/conditional response of reported expression in simulated event panels, not personality or emotional state",
        },
    )


def main():
    """Run finite stages with verified resumption, leaving old results untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "encode", "score", "history", "all")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old", type=Path, default=OLD)
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    started = time.monotonic()
    out, old = args.out.resolve(), args.old.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prepare(out, old, args.pairs.resolve())
    if args.stage in ("encode", "all"):
        for reader in ("e5", "bge"):
            encode(out, reader, args.device)
    if args.stage in ("score", "all"):
        natural_scores(out, old)
    if args.stage in ("history", "all"):
        history_experiment(out, old)
    write_json(
        out / "execution_progress.json",
        {
            "stage": args.stage,
            "status": "finite_stage_executed_pending_arithmetic_review",
            "elapsed_seconds": time.monotonic() - started,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": "NEXT-01 bounded candidate probes / NEXT-02 numerical event integration",
        },
    )
    print(
        f"finished finite {args.stage} stage in {time.monotonic() - started:.1f}s; no background queue",
        flush=True,
    )


if __name__ == "__main__":
    main()
