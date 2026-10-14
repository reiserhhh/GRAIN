#!/usr/bin/env python3
"""Collect conditioned-writer choices and cold-test a source-observed vector bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_semantic_anchor as S
from grain_research.construction import math_writer_history_bridge as M
from scripts import run_grain_joint_persona_style_hm150 as OLD
from scripts import run_grain_observed_text_anchor_hm164 as A

DEFAULT = ROOT / "results/grain_math_response_v1/writer_history_bridge_r1"
CONFIG = ROOT / "configs/grain_writer_history_bridge_hm165.json"
SCIENCE = (
    "configs/grain_writer_history_bridge_hm165.json",
    "grain_research/construction/math_writer_history_bridge.py",
    "scripts/run_grain_writer_history_bridge_hm165.py",
    "tests/test_grain_writer_history_bridge.py",
    "docs/GRAIN_WRITER_HISTORY_BRIDGE_PLAN_HM165.md",
    "grain_research/construction/math_semantic_anchor.py",
    "scripts/run_grain_observed_text_anchor_hm164.py",
    "grain_research/construction/math_observed_text_anchor.py",
    "scripts/run_grain_joint_persona_style_hm150.py",
    "scripts/run_grain_semantic_anchor_hm147.py",
    "grain_research/vector_mechanisms/trace.py",
)


def status(out, **fields):
    """Atomically checkpoint runtime state, without modifying scientific artifacts."""
    path = out / "collection_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(fields, indent=2) + "\n")
    temporary.replace(path)


def check(out):
    """Authenticate frozen new science and old public generated inputs."""
    manifest = A.read(out / "manifest.json")
    for section, base in (
        ("science_hashes", ROOT),
        ("input_hashes", ROOT),
        ("prepared_hashes", out),
    ):
        for name, digest in manifest[section].items():
            if A.digest(base / name) != digest:
                raise ValueError("Sealed dependency changed: " + name)
    return manifest


def prepare(out):
    """Seal actual old-article corners, vector addresses and new response opportunities."""
    from transformers import AutoTokenizer

    config = A.read(CONFIG)
    source, rows, records = A.authenticated(A.read(A.CONFIG))
    if str(source.relative_to(ROOT)) != config["source"]:
        raise ValueError("Historical corpus differs from registered source")
    persona = A.read(ROOT / config["persona_manifest"])
    pairs, metadata = A.M.persona_pairs(rows)
    selected = [
        (pair, meta)
        for pair, meta in zip(pairs, metadata, strict=True)
        if meta["family"] == config["family"]
    ]
    if len(selected) != 108:
        raise ValueError("All named2targets/6tasks/3styles/3seeds pairs required")
    neutral = {
        (r["situation"], r["style"], r["seed"]): r["system"]
        for r in rows
        if r["axis"] == "none"
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(OLD.MODEL), local_files_only=True, trust_remote_code=False
    )
    ab_ids = [tokenizer.encode(s, add_special_tokens=False) for s in ("A", "B")]
    if any(len(ids) != 1 for ids in ab_ids) or ab_ids[0] == ab_ids[1]:
        raise ValueError("Protocol alphabet must be two distinct single tokens")
    design, articles, requests = [], [], []
    for pair_index, (indices, meta) in enumerate(selected):
        hi, low = map(int, indices)
        if rows[hi]["user"] != rows[low]["user"]:
            raise ValueError("Actual historical pair changes the prior task")
        if meta["seed"] not in (11, 23, 37):
            raise ValueError("Unknown generation replicate")
        item = meta | {
            "pair_id": f"pair{pair_index:03d}",
            "old_user": rows[hi]["user"],
            "high_index": hi,
            "low_index": low,
            "high_article": len(articles),
            "low_article": len(articles) + 1,
        }
        design.append(item)
        for pole, index in ((1, hi), (-1, low)):
            articles.append(
                meta
                | {
                    "pair_id": item["pair_id"],
                    "pole": pole,
                    "old_index": index,
                    "text_id": rows[index]["text_id"],
                    "old_finish_reason": records[index]["finish_reason"],
                    "old_audit": records[index]["audit"],
                }
            )
        systems = {
            -1: rows[low]["system"],
            1: rows[hi]["system"],
            0: neutral[meta["situation"], meta["style"], meta["seed"]],
        }
        for pole in (-1, 1):
            if not systems[pole].startswith(
                persona["aligned_pairs"][f"{meta['axis']}:named"][
                    "high" if pole == 1 else "low"
                ]
            ):
                raise ValueError("Policy is not the original writer assignment")
        body = {-1: records[low]["text"], 1: records[hi]["text"], 0: ""}
        requests.extend(M.bridge_requests(item, systems, body, config["opportunities"]))
    unique, addresses, lookup = [], [], {}
    role_order = {"discovery": 0, "development": 1, "evaluation": 2}
    requests.sort(key=lambda r: (role_order[r["role"]], r["request_id"]))
    for request in requests:
        ids = M.rendered_token_ids(
            tokenizer.apply_chat_template(
                request["messages"], tokenize=True, add_generation_prompt=True
            )
        )
        if len(ids) > config["max_prompt_tokens"]:
            raise ValueError("Full article exceeds support; no truncation rescue")
        key = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append({"unit_id": key, "prompt_ids": ids, "role": request["role"]})
        index = lookup[key]
        if unique[index]["role"] != request["role"]:
            raise ValueError(
                "Source and application share the same full response input"
            )
        request["unit_index"] = index
        addresses.append(index)
    if len(requests) != 7776:
        raise ValueError("Full finite response design missing")
    source_mask = np.array([r["role"] == "discovery" for r in rows])
    with np.load(source / "numeric_observations.npz", allow_pickle=False) as data:
        all_slices = data["grain4_slices"]
    indices = [r["old_index"] for r in articles]
    with np.load(ROOT / config["grain_units"], allow_pickle=False) as old:
        z = S.standardize(all_slices[indices], old["origin"], old["scale"])
        features = S.tensor_features(z)
        old_readings, _ = S.score_views(
            all_slices[indices], old["origin"], old["scale"], old["coefficients"]
        )
    if source_mask.sum() != 351:
        raise ValueError("Historical cohort changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "batches").mkdir()
    A.write(out / "pairs.json", design)
    A.write(out / "articles.json", articles)
    A.write(out / "requests.json", requests)
    A.write(out / "units.json", unique)
    OLD.H.write_npz(
        out / "features.npz",
        features=features,
        old_readings=old_readings,
        request_unit=np.array(addresses),
    )
    inputs = [
        source / name
        for name in (
            "manifest.json",
            "materials.json",
            "generation_seal.json",
            "observation_seal.json",
            "numeric_observations.npz",
        )
    ] + [
        ROOT / config["grain_units"],
        ROOT / config["persona_manifest"],
        OLD.MODEL / "tokenizer.json",
        OLD.MODEL / "tokenizer_config.json",
        OLD.MODEL / "config.json",
    ]
    A.write(
        out / "manifest.json",
        {
            "packet": "HM165",
            "config": config,
            "science_hashes": {name: A.digest(ROOT / name) for name in SCIENCE},
            "input_hashes": {str(p.relative_to(ROOT)): A.digest(p) for p in inputs},
            "prepared_hashes": {
                name: A.digest(out / name)
                for name in (
                    "pairs.json",
                    "articles.json",
                    "requests.json",
                    "units.json",
                    "features.npz",
                )
            },
            "model_weights_sha256": A.read(source / "manifest.json")[
                "model_weights_sha256"
            ],
            "logical_requests": len(requests),
            "unique_histories": len(unique),
            "articles": len(articles),
            "pairs": len(design),
            "option_token_ids": [ids[0] for ids in ab_ids],
            "prompt_tokens_max": max(len(u["prompt_ids"]) for u in unique),
            "scope": "controlled writer protocol;not human behaviors;not LLM-free encoder",
            "prepared_before_new_outcomes": True,
        },
    )
    print(
        json.dumps(
            {"prepared": str(out), "logical": len(requests), "unique": len(unique)}
        )
    )


def results(out, source_only=False):
    """Load actual completed units; source compilation never opens application files."""
    found = {}
    for path in sorted((out / "batches").glob("[0-9][0-9][0-9][0-9].json")):
        if source_only and int(path.stem) >= 1000:
            continue
        for row in A.read(path):
            if row["unit_index"] in found:
                raise ValueError("Duplicate real capture")
            found[row["unit_index"]] = row
    return found


def check_capture_seal(out, name):
    """Reject edited captures before source construction or outcome analysis."""
    for filename, expected in A.read(out / name).items():
        if A.digest(out / filename) != expected:
            raise ValueError("Captured observation changed: " + filename)


def collect(out, role):
    """Capture new next-choice laws once per exact input; no trace/fidelity search."""
    import torch

    manifest = check(out)
    config = manifest["config"]
    units = A.read(out / "units.json")
    done = results(out)
    if role == "remaining":
        for name in ("response_program_seal.json", "cold_readings_seal.json"):
            seal = A.read(out / name)
            if A.digest(out / seal["file"]) != seal["sha256"]:
                raise ValueError(
                    "Source program/cold output changed before application"
                )
    wanted = [
        i
        for i, u in enumerate(units)
        if (u["role"] == "discovery") == (role == "source")
    ]
    if any(i in done for i in wanted):
        raise ValueError(
            "Do not rerun or automatically retry an executed collection stage"
        )
    observer = OLD.offline_model(manifest)
    if not (out / "model_manifest.json").exists():
        A.write(out / "model_manifest.json", observer.manifest())
    elapsed = sum(r["model_seconds_share"] for r in done.values())
    eos = observer.tokenizer.eos_token_id
    index0 = 0 if role == "source" else 1000
    for batch, offset in enumerate(range(0, len(wanted), config["batch_size"])):
        if elapsed >= config["model_seconds_cap"]:
            status(
                out,
                status="PARTIAL_BUDGET",
                role=role,
                model_seconds=elapsed,
                completed_unique=len(done),
                total_unique=len(units),
                pid=os.getpid(),
            )
            return
        ids = wanted[offset : offset + config["batch_size"]]
        group = [units[i]["prompt_ids"] for i in ids]
        width = max(map(len, group))
        padded = torch.tensor(
            [[eos] * (width - len(p)) + p for p in group], device="mps"
        )
        mask = torch.tensor(
            [[0] * (width - len(p)) + [1] * len(p) for p in group], device="mps"
        )
        status(
            out,
            status="RUNNING",
            role=role,
            current_batch=index0 + batch,
            model_seconds=elapsed,
            completed_unique=len(done),
            total_unique=len(units),
            pid=os.getpid(),
        )
        start = time.monotonic()
        with torch.inference_mode():
            hidden = observer.model.model(
                input_ids=padded, attention_mask=mask, use_cache=False, return_dict=True
            ).last_hidden_state[:, -1]
            logits = observer.model.lm_head(hidden)
            normalizer = torch.logsumexp(logits, dim=-1).cpu().numpy().astype(float)
            ab = logits[:, manifest["option_token_ids"]].cpu().numpy().astype(float)
        seconds = time.monotonic() - start
        elapsed += seconds
        captured = []
        for i, values, total in zip(ids, ab, normalizer, strict=True):
            # Semantic order is request metadata; raw A/B measurements are retained.
            qa = M.choice_account(values, total, 0)
            draw = M.sample_protocol_choice(
                qa["q_positive"], units[i]["unit_id"], config["draw_seed"]
            )
            item = {
                "unit_index": i,
                "unit_id": units[i]["unit_id"],
                "role": units[i]["role"],
                "ab_logits": values.tolist(),
                "log_normalizer": float(total),
                "draw_a": draw,
                "protocol_output": "A" if draw else "B",
                "model_seconds_share": seconds / len(ids),
            }
            captured.append(item)
            done[i] = item
        A.write(out / "batches" / f"{index0 + batch:04d}.json", captured)
        print(
            json.dumps(
                {
                    "batch": index0 + batch,
                    "unique": len(done),
                    "total": len(units),
                    "model_seconds": elapsed,
                }
            ),
            flush=True,
        )
    seal = {
        str(p.relative_to(out)): A.digest(p)
        for p in sorted((out / "batches").glob("*.json"))
    }
    A.write(out / f"{role}_collection_seal.json", seal)
    status(
        out,
        status="SOURCE_COMPLETE" if role == "source" else "COMPLETE",
        role=role,
        completed_unique=len(done),
        total_unique=len(units),
        model_seconds=elapsed,
        pid=os.getpid(),
        all_stage_batches_terminal=True,
    )


def response_tables(out, source_only=False):
    """Map registered semantic choices back to exact event and opportunity corners."""
    captured = results(out, source_only=source_only)
    requests = A.read(out / "requests.json")
    tables = {}
    for request in requests:
        if source_only and request["role"] != "discovery":
            continue
        observed = captured.get(request["unit_index"])
        if observed is None:
            raise ValueError("Incomplete real event collection")
        key = (request["pair_id"], request["opportunity"], request["order"])
        q = M.choice_account(
            observed["ab_logits"], observed["log_normalizer"], request["order"]
        )
        values = tables.setdefault(key, {})
        cell = (request["policy"], request["history"])
        if cell in values:
            raise ValueError("Duplicate registered corner")
        values[cell] = q | {
            "draw_positive": observed["draw_a"]
            if not request["order"]
            else 1 - observed["draw_a"]
        }
    if any(len(cells) != 9 for cells in tables.values()):
        raise ValueError("A complete nine-corner law is required")
    return tables


def compile_source(out):
    """Freeze a new observed-response witness before application choices are captured."""
    manifest = check(out)
    check_capture_seal(out, "source_collection_seal.json")
    units = A.read(out / "units.json")
    if any(units[i]["role"] != "discovery" for i in results(out)):
        raise ValueError("Application outcomes exist before source compilation")
    tables = response_tables(out, source_only=True)
    articles = A.read(out / "articles.json")
    ids = [i for i, r in enumerate(articles) if r["role"] == "discovery"]
    opportunities = [r["id"] for r in manifest["config"]["opportunities"]]
    observed = np.array(
        [
            [
                np.mean(
                    [
                        tables[r["pair_id"], a, o][0, r["pole"]]["q_positive"]
                        for o in (0, 1)
                    ]
                )
                for a in opportunities
            ]
            for r in (articles[i] for i in ids)
        ]
    )
    with np.load(out / "features.npz", allow_pickle=False) as saved:
        x = saved["features"][ids]
    origin, coefficients, norms = M.response_witness(x, observed)
    OLD.H.write_npz(
        out / "source_response_program.npz",
        origin=origin,
        coefficients=coefficients,
        norms=norms,
        source_features=x,
        source_responses=observed,
        source_article_indices=np.array(ids),
    )
    A.write(
        out / "response_program_seal.json",
        {
            "file": "source_response_program.npz",
            "sha256": A.digest(out / "source_response_program.npz"),
            "source_articles": len(ids),
            "application_captures": 0,
            "input_roles": ["discovery"],
            "meaning": "source-choice witness;not probability predictor",
        },
    )
    print(
        json.dumps(
            {
                "compiled_source_articles": len(ids),
                "nonzero_witnesses": int((norms > 0).sum()),
            }
        )
    )


def cold(out):
    """Run a standalone numeric scorer before any new application choice outcomes."""
    check(out)
    with (
        np.load(out / "features.npz", allow_pickle=False) as f,
        np.load(out / "source_response_program.npz", allow_pickle=False) as p,
    ):
        readings = M.cold_response_reading(
            f["features"], p["origin"], p["coefficients"], p["norms"]
        )
    OLD.H.write_npz(out / "cold_readings.npz", readings=readings)
    A.write(
        out / "cold_readings_seal.json",
        {
            "file": "cold_readings.npz",
            "sha256": A.digest(out / "cold_readings.npz"),
            "application_capture_allowed_after_this": True,
        },
    )
    print(
        json.dumps(
            {"cold_numeric_articles": len(readings), "opportunities": readings.shape[1]}
        )
    )


def analyze(out):
    """Compare completed actual response laws and frozen numeric readings without refit."""
    manifest = check(out)
    check_capture_seal(out, "remaining_collection_seal.json")
    for seal_name in ("response_program_seal.json", "cold_readings_seal.json"):
        seal = A.read(out / seal_name)
        if A.digest(out / seal["file"]) != seal["sha256"]:
            raise ValueError("Sealed source/cold artifact changed")
    captured = results(out)
    if len(captured) != manifest["unique_histories"]:
        raise ValueError("Finite design incomplete; do not announce completion")
    start = time.monotonic()
    tables = response_tables(out)
    pairs = {r["pair_id"]: r for r in A.read(out / "pairs.json")}
    with np.load(out / "cold_readings.npz", allow_pickle=False) as saved:
        readings = saved["readings"]
    with np.load(out / "features.npz", allow_pickle=False) as saved:
        old_readings = saved["old_readings"]
    opportunity_ids = [r["id"] for r in manifest["config"]["opportunities"]]
    rows, errors = [], []
    for (pair_id, opportunity, order), cells in sorted(tables.items()):
        pair = pairs[pair_id]
        q = np.array(
            [[cells[p, h]["q_positive"] for h in (-1, 0, 1)] for p in (-1, 0, 1)]
        )
        odds = np.array(
            [[cells[p, h]["positive_log_odds"] for h in (-1, 0, 1)] for p in (-1, 0, 1)]
        )
        account, log_account = M.nine_corner_account(q), M.nine_corner_account(odds)
        index = opportunity_ids.index(opportunity)
        gap = (
            readings[pair["high_article"], index] - readings[pair["low_article"], index]
        )
        row = {
            k: pair[k]
            for k in ("pair_id", "axis", "situation", "style", "seed", "role")
        }
        row |= {"opportunity": opportunity, "order": order} | account
        row |= {"log_odds_" + k: v for k, v in log_account.items()}
        row["cold_observed_choice_witness_difference"] = float(gap)
        target = ("extraversion", "neuroticism").index(pair["axis"])
        for j, arm in enumerate(("full", "single", "source_relinked")):
            row["old_persona_" + arm + "_difference"] = float(
                old_readings[pair["high_article"], j, target]
                - old_readings[pair["low_article"], j, target]
            )
        row["option_mass_min"] = min(
            v["unrestricted_option_mass"] for v in cells.values()
        )
        row["option_mass_mean"] = float(
            np.mean([v["unrestricted_option_mass"] for v in cells.values()])
        )
        row["natural_draw_difference"] = (
            cells[1, 1]["draw_positive"] - cells[-1, -1]["draw_positive"]
        )
        rows.append(row)
        errors.append(account["reconstruction_error"])
    A.table(out / "response_accounts.csv", rows)
    summaries = []
    for axis in ("extraversion", "neuroticism"):
        for opportunity in opportunity_ids:
            for role in ("discovery", "development", "evaluation"):
                subset = [
                    r
                    for r in rows
                    if r["axis"] == axis
                    and r["opportunity"] == opportunity
                    and r["role"] == role
                ]
                for style in ("all", "neutral", "newsletter", "spelling"):
                    group = (
                        subset
                        if style == "all"
                        else [r for r in subset if r["style"] == style]
                    )
                    result = {
                        "axis": axis,
                        "opportunity": opportunity,
                        "role": role,
                        "style": style,
                        "accounts": len(group),
                    }
                    for name in (
                        "natural_effect",
                        "policy_effect",
                        "history_effect",
                        "interaction_effect",
                        "joint_policy_history",
                        "option_mass_mean",
                        "natural_draw_difference",
                    ):
                        result[name + "_mean"] = float(
                            np.mean([r[name] for r in group])
                        )
                    for name in ("natural_effect", "history_effect"):
                        raw = np.array([r[name] for r in group])
                        reading = np.array(
                            [
                                r["cold_observed_choice_witness_difference"]
                                for r in group
                            ]
                        )
                        active = (
                            np.isfinite(reading)
                            & (abs(raw) > 1e-12)
                            & (abs(reading) > 1e-12)
                        )
                        result[name + "_witness_same"] = int(
                            (active & (raw * reading > 0)).sum()
                        )
                        result[name + "_witness_opposite"] = int(
                            (active & (raw * reading < 0)).sum()
                        )
                    summaries.append(result)
    A.table(out / "condition_summary.csv", summaries)
    # Alternate formulas recompute all measured choices and source cross-moments.
    probability_error = 0.0
    for item in captured.values():
        aa, bb = item["ab_logits"]
        expected = float(1 / (1 + np.exp(bb - aa)))
        actual = M.choice_account(item["ab_logits"], item["log_normalizer"], 0)[
            "q_positive"
        ]
        probability_error = max(probability_error, abs(expected - actual))
    with np.load(out / "source_response_program.npz", allow_pickle=False) as p:
        x, y, c = p["source_features"], p["source_responses"], p["coefficients"]
        alternate = np.einsum(
            "ijc,ijd->cd", y[:, None] - y[None, :], x[:, None] - x[None, :]
        ) / (2 * len(x) ** 2)
        covariance_error = float(np.max(abs(c - alternate)))
    audit = {
        "nine_corner_max_error": max(errors),
        "choice_sigmoid_max_error": probability_error,
        "source_pairwise_covariance_max_error": covariance_error,
        "independent_blind_audit": False,
        "same_agent_alternate_formulas": True,
        "old_science_and_input_hashes_match": True,
    }
    A.write(out / "alternate_audit.json", audit)
    A.write(
        out / "summary.json",
        {
            "packet": "HM165",
            "status": "FINITE_WRITER_HISTORY_BRIDGE_COMPLETE",
            "unique_model_histories": len(captured),
            "logical_accounts": len(rows),
            "logical_requests": manifest["logical_requests"],
            "actual_articles": manifest["articles"],
            "model_seconds": sum(r["model_seconds_share"] for r in captured.values()),
            "analysis_seconds": time.monotonic() - start,
            "source_program_before_application_capture": True,
            "cold_program_before_application_capture": True,
            "science_scope": "constrained next-token simulated writer choices;not human personality validity",
            "all_terms_retained": True,
            "original_18_100_44_complete": False,
            "audit": audit,
        },
    )
    A.write(
        out / "output_seal.json",
        {
            name: A.digest(out / name)
            for name in (
                "response_accounts.csv",
                "condition_summary.csv",
                "summary.json",
                "alternate_audit.json",
            )
        },
    )
    print(json.dumps(A.read(out / "summary.json")), flush=True)


def main():
    """Expose finite exclusive stages, including a separate metadata-free cold CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "source", "compile", "cold", "remaining", "analyze"),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage in ("source", "remaining"):
        collect(args.out, args.stage)
    else:
        {
            "prepare": prepare,
            "compile": compile_source,
            "cold": cold,
            "analyze": analyze,
        }[args.stage](args.out)


if __name__ == "__main__":
    main()
