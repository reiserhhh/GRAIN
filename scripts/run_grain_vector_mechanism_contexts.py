#!/usr/bin/env python3
"""VM02/03: label-free natural author-context pilot with shared-anchor controls."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import (
    ANCHOR,
    ANCHOR_SECOND,
    ARMS,
    QUESTIONS,
    SYSTEM,
    VIEWS,
    history_text,
    match_donors,
    render_context,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_controls import write_json


def run(args):
    """Preflight every prompt before inference and checkpoint private per-case states."""
    config = json.loads(args.config.read_text())
    if config.get("labels_allowed") is not False:
        raise ValueError("This runner never accepts questionnaire labels")
    all_bundles = json.loads(args.views.read_text(encoding="utf-8"))
    bundles = [b for b in all_bundles if b["split"] == args.split]
    if not bundles:
        raise ValueError("Requested author split is empty")
    if any(len(b["history"]) != 3 or len(b["current"]) != 2 for b in bundles):
        raise ValueError("This pilot requires exactly three histories and two currents")
    by_id = {b["author_id"]: b for b in bundles}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "private_cache"
    cache.mkdir(exist_ok=True)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    observer = DecoderObserver.load(
        **{
            k: config[k]
            for k in (
                "revision",
                "device",
                "dtype",
                "max_tokens",
                "local_files_only",
                "pad_to_length",
            )
        },
        name=config["model"],
    )
    yes = observer.tokenizer.encode("Yes", add_special_tokens=False)
    no = observer.tokenizer.encode("No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("This optimized pilot readout requires one-token candidates")
    matches = match_donors(
        bundles, lambda t: len(observer.tokenizer.encode(t, add_special_tokens=False))
    )
    donors = {
        (r["author_id"], r["rule"]): r["donor_id"]
        for r in matches
        if r["status"] == "MATCHED"
    }
    eligible = [
        b
        for b in bundles
        if all(
            (b["author_id"], r) in donors for r in ("other_length", "other_condition")
        )
    ]
    if args.independent_history:
        eligible = bundles
    arms = (
        ("history_0", "history_1", "history_2", "none")
        if args.independent_history
        else ARMS
    )
    use_anchors = args.anchor_only or args.independent_history
    cases = []
    content_ids = {}
    for b in eligible:
        uid = b["author_id"]
        current_texts = (
            [ANCHOR, ANCHOR_SECOND]
            if use_anchors
            else [c["body"] for c in b["current"]] + [ANCHOR]
        )
        for current_index, current in enumerate(current_texts):
            for arm in arms:
                donor_id = donors.get((uid, arm), uid)
                history = (
                    None
                    if arm == "none"
                    else history_text(by_id[donor_id], reverse=arm == "same_reversed")
                )
                history_body = None
                if args.independent_history and arm != "none":
                    history_body = b["history"][int(arm[-1])]["body"]
                    history = "Earlier excerpt 1:\n" + history_body
                for task in QUESTIONS:
                    user, anchor = render_context(history, current, task)
                    ids = observer.prompt_ids(SYSTEM, user)
                    if len(ids) > config["pad_to_length"]:
                        raise ValueError(
                            "Natural prompt exceeds fixed length; never truncate"
                        )
                    positions = [
                        observer.content_position(SYSTEM, user, anchor),
                        len(ids) - 1,
                    ]
                    indices = observer.content_indices(SYSTEM, user, current)
                    token_hash = material_hash([ids[i] for i in indices])
                    previous = content_ids.setdefault((uid, current_index), token_hash)
                    if previous != token_hash:
                        raise ValueError(
                            "Current content tokenization changed across arms"
                        )
                    cases.append(
                        {
                            "case_id": f"{uid}__c{current_index}__{arm}__{task}",
                            "author_id": uid,
                            "current_index": current_index,
                            "arm": arm,
                            "task": task,
                            "donor_id": None if arm == "none" else donor_id,
                            "history_comment_ids": []
                            if arm == "none"
                            else (
                                [b["history"][int(arm[-1])]["comment_id"]]
                                if args.independent_history
                                else [
                                    c["comment_id"] for c in by_id[donor_id]["history"]
                                ]
                            ),
                            "ids": ids,
                            "positions": positions,
                            "indices": indices,
                            "history_indices": observer.content_indices(
                                SYSTEM, user, history_body
                            )
                            if history_body
                            else [],
                            "content_token_sha256": token_hash,
                        }
                    )
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("No complete target with eligible historical donors")
    manifest = {
        "config": config,
        "split": args.split,
        "source_view_sha256": material_hash(all_bundles),
        "code_sha256": code_fingerprint(ROOT),
        "model": observer.manifest(),
        "tasks": QUESTIONS,
        "system": SYSTEM,
        "common_anchor": ANCHOR,
        "current_design": "two_shared_synthetic_anchors"
        if use_anchors
        else "two_author_currents_plus_shared_anchor",
        "second_anchor": ANCHOR_SECOND if use_anchors else None,
        "history_design": "three_nonoverlapping_single_comments"
        if args.independent_history
        else "shared_three_comment_bundle",
        "arms": arms,
        "cases": [
            {
                k: v
                for k, v in c.items()
                if k not in ("ids", "indices", "history_indices")
            }
            | {
                "prompt_token_sha256": material_hash(c["ids"]),
                "prompt_tokens": len(c["ids"]),
            }
            for c in cases
        ],
        "requested_authors": len(bundles),
        "eligible_authors": len(eligible),
        "excluded_no_prior_donor": len(bundles) - len(eligible),
        "labels_loaded": False,
        "shared_history_warning": "View similarity is NOT author reliability when the same history occurs in both views",
    }
    stable_manifest = {k: v for k, v in manifest.items() if k != "model"}
    stable_manifest["model"] = {
        k: v for k, v in manifest["model"].items() if k != "unix_time"
    }
    manifest["identity"] = material_hash(stable_manifest)
    if (output / "manifest.json").exists() and json.loads(
        (output / "manifest.json").read_text()
    )["identity"] != manifest["identity"]:
        raise ValueError("Run identity changed; use a new directory")
    write_json(output / "manifest.json", manifest)
    write_json(output / "model_manifest.json", observer.manifest())
    pd.DataFrame(matches).to_csv(output / "donor_matching.csv", index=False)
    archive_source(ROOT, output / "source_snapshot.zip")
    started = time.monotonic()
    first = cases[0]
    states, logits = observer.capture(first["ids"], positions=first["positions"])
    _, repeated = observer.capture(first["ids"], positions=first["positions"])
    _, sham = observer.capture(
        first["ids"],
        positions=first["positions"],
        patch=Patch("block_16", first["positions"][1], states["block_16"][1]),
    )
    checks = {
        "repeat_max_abs": float(np.max(np.abs(logits - repeated))),
        "self_max_abs": float(np.max(np.abs(logits - sham))),
    }
    if max(checks.values()) > 5e-4:
        raise ValueError("Natural-text instrumentation does not replay")
    write_json(output / "hook_replay_tests.json", checks)
    print(
        f"PREFLIGHT {len(cases)} cases, authors={len(eligible)}, max_tokens={max(len(c['ids']) for c in cases)}",
        flush=True,
    )
    completed = []
    for c in cases:
        path = cache / f"{c['case_id']}.json"
        fp = cache / f"{c['case_id']}.npz"
        if path.is_file() and fp.is_file():
            completed.append(json.loads(path.read_text()))
            continue
        if (
            observer.calls >= config["maximum_calls"]
            or time.monotonic() - started > config["maximum_seconds"]
        ):
            break
        if (
            sum(p.stat().st_size for p in cache.iterdir() if p.is_file())
            >= config["maximum_cache_bytes"]
        ):
            break
        pools = [("mean", c["indices"]), ("std", c["indices"])]
        if c["history_indices"]:
            pools.append(("mean", c["history_indices"]))
        states, logits = observer.capture(
            c["ids"],
            positions=c["positions"],
            pools=pools,
        )
        if not np.isfinite(logits).all() or not all(
            np.isfinite(s).all() for s in states.values()
        ):
            raise ValueError("Nonfinite natural-text trace")
        record = {
            k: v
            for k, v in c.items()
            if k not in ("ids", "indices", "positions", "history_indices")
        }
        record.update(
            margin=float(logits[yes[0]] - logits[no[0]]),
            prompt_tokens=len(c["ids"]),
            content_position=c["positions"][0],
            decision_position=c["positions"][1],
            content_tokens=len(c["indices"]),
        )
        saved = {
            f"{site}__{view}": value[i]
            for site, value in states.items()
            for i, view in enumerate(VIEWS)
        }
        if args.independent_history:
            saved.update(
                {
                    f"{site}__history_mean": value[4]
                    if c["history_indices"]
                    else np.zeros(value.shape[1], dtype=np.float32)
                    for site, value in states.items()
                }
            )
        np.savez_compressed(fp, **saved)
        write_json(path, record)
        completed.append(record)
        if len(completed) % 30 == 0:
            print(
                f"CONTEXT {len(completed)}/{len(cases)} seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
    pd.DataFrame(completed).to_csv(output / "context_predictions.csv", index=False)
    status = {
        "status": "COMPLETE" if len(completed) == len(cases) else "BUDGET_PARTIAL",
        "cases_complete": len(completed),
        "cases_planned": len(cases),
        "authors": len(eligible),
        "calls": observer.calls,
        "input_tokens": observer.input_tokens,
        "processed_tokens": observer.processed_tokens,
        "seconds": time.monotonic() - started,
        "labels_loaded": False,
    }
    if config["device"] == "cuda":
        status["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    write_json(output / "run_status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_vector_contexts_v1.json"
    )
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["technical_only", "pilot_development"], required=True
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--anchor-only",
        action="store_true",
        help="Use identical current texts for every author; isolate shared-history matching",
    )
    parser.add_argument(
        "--independent-history",
        action="store_true",
        help="Three nonoverlapping one-comment histories with two shared current anchors",
    )
    parsed = parser.parse_args()
    try:
        run(parsed)
    except Exception:
        parsed.output_dir.mkdir(parents=True, exist_ok=True)
        (parsed.output_dir / "error_log.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
