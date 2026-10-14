#!/usr/bin/env python3
"""Measure complete public behavior-item keys on unchanged HM251 evidence laws."""

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_behavior_anchor as B
from grain_research.construction import math_item_bridge as M
from grain_research.construction import math_selector_matrix as S
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_behavior_anchor_hm251 as P
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_item_bridge_hm255.json"
DEFAULT = ROOT / "results/grain_math_response_v1/item_observation_bridge_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_ITEM_BRIDGE_PLAN_HM255.md",
    "grain_research/construction/math_item_bridge.py",
    "scripts/run_grain_item_bridge_hm255.py",
    "tests/test_grain_item_bridge.py",
    "grain_research/construction/math_behavior_anchor.py",
    "grain_research/construction/math_selector_matrix.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "scripts/run_grain_behavior_anchor_hm251.py",
)
ACCOUNTS = ("high", "low", "balanced", "common", "official")


def prepare(out):
    """Freeze all new item prompts and unchanged public SOURCE laws before responses."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No overwrite/restart")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    old = R.check(
        parent,
        (
            "source_seal.json",
            "design_seal.json",
            "native_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        ),
    )
    P.local_environment()
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    item_rows = R.read(parent / "source_rows.json")[:50]
    items = [
        {k: r[k] for k in ("target", "pole", "original", "sentence")}
        for r in item_rows
        if r["target"] in cfg["targets"]
    ]
    if len(items) != 20 or [
        sum(r["target"] == t for r in items) for t in cfg["targets"]
    ] != [10, 10]:
        raise ValueError("Changed complete public item keys")
    alphabet = D.candidates("agreeableness", tok)
    alphabet = {
        "labels": ["0", "1"],
        "token_ids": alphabet["token_ids"][:2],
        "values": [0.0, 1.0],
    }
    old_rows = R.read(parent / "reader_rows.json")
    rows, sizes, prefixes = [], [], []
    for row in old_rows:
        body = M.public_bodies(
            tok.decode(row["input_ids"][0], skip_special_tokens=False)
        )
        if len(body) != 8:
            raise ValueError("Changed actual eight-event input")
        for j, target in enumerate(cfg["targets"]):
            if (
                D.tokenize_prompt(body, target, tok, cfg["max_tokens"])[0]
                != row["input_ids"][j]
            ):
                raise ValueError("Changed old native anonymous body/template")
        user, _ = D.render(body, cfg["targets"][0])
        prefix = user.split("Question: ", 1)[0]
        questions = []
        for item in items:
            question = (
                'Consider this first-person statement as a possible description of the writer: "'
                + item["sentence"]
                + "\" Does this description fit this writer's usual behavior? "
                + "Return 1 if yes or 0 if no. Return only one digit."
            )
            ids = tok.apply_chat_template(
                [
                    {"role": "system", "content": D.SYSTEM},
                    {
                        "role": "user",
                        "content": prefix + "Question: " + question + "\nAnswer:",
                    },
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            if not isinstance(ids, list) or len(ids) + 16 > cfg["max_tokens"]:
                raise ValueError("Invalid/overlong item input; no truncation")
            questions.append(ids)
            sizes.append(len(ids))
        prefixes.append(M.common_prefix(questions))
        rows.append(
            {
                "author": row["author"],
                "candidate": row["candidate"],
                "old_panel_author": row["old_panel_author"],
                "input_ids": questions,
            }
        )
    if (
        len(rows) != cfg["max_batches"]
        or len(rows) * len(items) != cfg["max_inquiries"]
    ):
        raise ValueError("Changed whole scope")
    design = H.selected(parent / "design.npz", ("weights", "scores"))
    if design["weights"].shape != (12, 2, 16):
        raise ValueError("Changed frozen positive law")
    model_files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if model_files != old["model_files"]:
        raise ValueError("Changed model identity")
    (out / "batches").mkdir(parents=True, exist_ok=False)
    R.write(out / "items.json", items)
    R.write(out / "units.json", rows)
    R.archive(out / "observations.npz", **design)
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "source_rows.json",
            "reader_rows.json",
            "source_seal.json",
            "design.npz",
            "design_seal.json",
            "native_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        )
    ]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "model_files": model_files,
            "alphabet": alphabet,
            "historical_named_outcomes_known": True,
            "new_item_outcomes_read": False,
            "original_scope_complete": False,
        },
    )
    quality = {
        "authors": 12,
        "families": len(rows),
        "items": len(items),
        "inquiries": len(rows) * len(items),
        "input_tokens_range": [min(sizes), max(sizes)],
        "shared_prefix_tokens_range": [min(prefixes), max(prefixes)],
        "same_old_body_native_prompts_replayed": True,
        "same_frozen_source_laws": True,
        "private_API_train": 0,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("items.json", "units.json", "observations.npz", "input_quality.json"),
    )
    H.status(out, {"status": "PREPARED", "model_seconds": 0, "completed_batches": 0})
    print(json.dumps(quality), flush=True)


def collect(out):
    """Collect each complete family once, with a fixed finite budget and exact prefix reuse."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg = manifest["config"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No retry of started or terminal collection")
    for name, digest in manifest["model_files"].items():
        if R.digest(MODEL / name) != digest:
            raise ValueError("Changed local model")
    lock, elapsed, completed = out / "ACTIVE.lock", 0.0, 0
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        P.local_environment()
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        alphabet = manifest["alphabet"]["token_ids"]
        R.archive(
            out / "head.npz",
            weights=observer.model.lm_head.weight.detach()[alphabet]
            .float()
            .cpu()
            .numpy(),
        )
        rows = R.read(out / "units.json")
        cache_error = None
        for index, row in enumerate(rows):
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Single model cap reached; no renewal/retry")
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "pending_batch": index,
                    "completed_batches": completed,
                    "model_seconds": elapsed,
                },
            )
            stamp = time.monotonic()
            try:
                result = M.native_items(
                    observer.model,
                    row["input_ids"],
                    alphabet,
                    cfg["device"],
                    cfg["query_batch_size"],
                )
                if index == 0:
                    cache_error = 0.0
                    with torch.inference_mode():
                        for j in (0, 19):
                            ids = torch.tensor(
                                [row["input_ids"][j]], device=cfg["device"]
                            )
                            h = observer.model.model(
                                input_ids=ids, use_cache=False, return_dict=True
                            ).last_hidden_state[0, -1]
                            logits = observer.model.lm_head(h).float().cpu().numpy()
                            cache_error = max(
                                cache_error,
                                float(
                                    np.max(abs(logits[alphabet] - result["logits"][j]))
                                ),
                                float(
                                    np.max(
                                        abs(
                                            h.float().cpu().numpy()
                                            - result["states"][j]
                                        )
                                    )
                                ),
                            )
                    if not np.isfinite(cache_error) or cache_error > 1e-3:
                        R.archive(out / "cache_guard_failure.npz", **result)
                        raise ValueError(
                            "Cache/direct mismatch; no precision rescue/retry"
                        )
                    R.write(
                        out / "cache_guard.json",
                        {
                            "fixed_items": [0, 19],
                            "max_error": cache_error,
                            "tolerance": 1e-3,
                            "changes_scientific_method": False,
                        },
                    )
                R.archive(out / "batches" / f"{index:03d}.npz", **result)
            finally:
                elapsed += time.monotonic() - stamp
            completed += 1
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "pending_batch": None,
                    "completed_batches": completed,
                    "model_seconds": elapsed,
                },
            )
            if completed % 16 == 0:
                print(
                    f"HM255 {completed}/{len(rows)} families; model_seconds={elapsed:.3f}",
                    flush=True,
                )
        runtime = {
            "status": "COLLECTED",
            "pid": None,
            "completed_batches": completed,
            "inquiries": completed * 20,
            "model_seconds": elapsed,
            "cache_direct_error": cache_error,
            "new_API_private_train": 0,
        }
        R.write(out / "collected_runtime.json", runtime)
        H.status(out, runtime)
        R.seal(
            out,
            "collected_seal.json",
            tuple(f"batches/{i:03d}.npz" for i in range(completed))
            + ("head.npz", "cache_guard.json", "collected_runtime.json"),
        )
        print(json.dumps(runtime), flush=True)
    except BaseException as exc:
        H.status(
            out,
            {
                "status": "FAILED",
                "completed_batches": completed,
                "model_seconds": elapsed,
                "error": repr(exc),
                "no_auto_retry": True,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def responses(out):
    """Load complete public item observations, not author questionnaires or partial subsets."""
    rows = [
        H.selected(
            out / "batches" / f"{i:03d}.npz", ("logits", "states", "normalizer", "top")
        )
        for i in range(192)
    ]
    return {
        k: np.asarray([r[k] for r in rows]).reshape(12, 16, 20, *rows[0][k].shape[1:])
        for k in rows[0]
    }


def analyze(out):
    """Keep every item, pole, key, common action, old named response and signed order."""
    manifest = R.check(out, ("collected_seal.json",))
    cfg, observed = manifest["config"], responses(out)
    items, inputs = R.read(out / "items.json"), R.prepared(out)
    keys = M.keyed_observations(observed["logits"], items, cfg["targets"])
    weights = inputs["weights"]
    arrays, summaries, author_rows = {}, {}, []
    for kind in ("probability", "log_odds"):
        levels, action = M.law_actions(keys[kind], weights)
        item_levels, item_action = M.law_actions(keys["item_" + kind], weights)
        orders = np.asarray(
            [
                B.walsh_accounts(keys[kind][u].reshape(16, -1), weights[u])[
                    "by_order"
                ].reshape(5, 2, 5)
                for u in range(12)
            ]
        )
        arrays[kind] = keys[kind]
        arrays[kind + "_levels"] = levels
        arrays[kind + "_actions"] = action
        arrays[kind + "_item_actions"] = item_action
        arrays[kind + "_orders"] = orders
        summaries[kind] = {
            "means": action.mean(0).tolist(),
            "CIs": S.bootstrap(
                action.reshape(12, -1), cfg["bootstrap_draws"], cfg["bootstrap_seed"]
            )
            .reshape(2, 5, 2)
            .tolist(),
            "positive_authors": (action > 0).sum(0).tolist(),
            "bank_means": [v.mean(0).tolist() for v in np.split(action, 2)],
            "order_means": orders.mean(0).tolist(),
            "balanced_C_minus_A": float((action[:, 1, 2] - action[:, 0, 2]).mean()),
            "balanced_C_minus_A_CI": S.bootstrap(
                action[:, 1, 2] - action[:, 0, 2],
                cfg["bootstrap_draws"],
                cfg["bootstrap_seed"],
            ).tolist(),
        }
        for u in range(12):
            for q, target in enumerate(cfg["targets"]):
                for j, account in enumerate(ACCOUNTS):
                    author_rows.append(
                        {
                            "kind": kind,
                            "author": u,
                            "target": target,
                            "account": account,
                            "high_arm": float(levels[u, 0, q, j]),
                            "low_arm": float(levels[u, 1, q, j]),
                            "action": float(action[u, q, j]),
                        }
                    )
        R.table(
            out / f"{kind}_items.csv",
            [
                {
                    "author": u,
                    "item": j,
                    **items[j],
                    "high_arm": float(item_levels[u, 0, j]),
                    "low_arm": float(item_levels[u, 1, j]),
                    "action": float(item_action[u, j]),
                    "keyed_action": float(
                        item_action[u, j] * (1 if items[j]["pole"] == "high" else -1)
                    ),
                }
                for u in range(12)
                for j in range(20)
            ],
        )
    # Old named outputs are a historical comparator, never new-response training data.
    old_logits, _ = P.native_arrays(ROOT / cfg["parent"])
    named = logsumexp(old_logits[..., 5:], -1) - logsumexp(old_logits[..., :5], -1)
    _, named_action = M.law_actions(named, weights)
    summaries["historical_named_log_odds"] = {
        "means": named_action.mean(0).tolist(),
        "positive_authors": (named_action > 0).sum(0).tolist(),
        "not_a_new_observation_or_same_item_units": True,
    }
    masses = np.exp(logsumexp(observed["logits"], -1) - observed["normalizer"])
    candidate = np.isin(observed["top"], manifest["alphabet"]["token_ids"])
    summaries["compliance"] = {
        "candidate_mass_min_mean_max": [
            float(masses.min()),
            float(masses.mean()),
            float(masses.max()),
        ],
        "greedy_first_token_in_alphabet_count": int(candidate.sum()),
        "inquiries": 3840,
        "complete_generated_response_not_measured": True,
    }
    summaries.update(
        {
            "authors": 12,
            "items": 20,
            "accounts": ACCOUNTS,
            "targets": cfg["targets"],
            "all_families_complete": True,
            "primary": "probability/conscientiousness/balanced",
            "model_seconds": R.read(out / "collected_runtime.json")["model_seconds"],
            "no_accuracy_optimization_or_human_trait_claim": True,
            "original_scope_complete": False,
        }
    )
    R.archive(
        out / "analysis.npz",
        named_actions=named_action,
        candidate_mass=masses,
        **arrays,
    )
    R.table(out / "author_accounts.csv", author_rows)
    R.write(out / "summary.json", summaries)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "author_accounts.csv",
            "probability_items.csv",
            "log_odds_items.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summaries), flush=True)


def verify(out):
    """Independently enumerate pole responses, arm expectations and exact head identities."""
    manifest = R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    z, inputs = responses(out), R.prepared(out)
    items = R.read(out / "items.json")
    saved = H.selected(
        out / "analysis.npz",
        (
            "probability",
            "log_odds",
            "probability_actions",
            "log_odds_actions",
            "probability_orders",
            "log_odds_orders",
        ),
    )
    head = H.selected(out / "head.npz", ("weights",))["weights"]
    error = float(
        np.max(abs(z["states"].astype(float) @ head.astype(float).T - z["logits"]))
    )
    value_error, action_error, key_identity, order_error = 0.0, 0.0, 0.0, 0.0
    count = 0
    for u in range(12):
        for q, target in enumerate(manifest["config"]["targets"]):
            hi = [
                j
                for j, r in enumerate(items)
                if r["target"] == target and r["pole"] == "high"
            ]
            lo = [
                j
                for j, r in enumerate(items)
                if r["target"] == target and r["pole"] == "low"
            ]
            for kind in ("probability", "log_odds"):
                alternate = []
                for k in range(16):
                    vals = []
                    for j in range(20):
                        a, b = map(float, z["logits"][u, k, j])
                        vals.append(
                            float(np.exp(b - np.logaddexp(a, b)))
                            if kind == "probability"
                            else b - a
                        )
                    h, l = (
                        sum(vals[j] for j in hi) / len(hi),
                        sum(vals[j] for j in lo) / len(lo),
                    )
                    off = 0.5 if kind == "probability" else 0.0
                    direct_official = (
                        sum(vals[j] for j in hi) + sum(2 * off - vals[j] for j in lo)
                    ) / 10
                    alternate.append(
                        [h, l, off + (h - l) / 2, (h + l) / 2, direct_official]
                    )
                alternate = np.array(alternate)
                value_error = max(
                    value_error, float(np.max(abs(alternate - saved[kind][u, :, q])))
                )
                diff = np.array(
                    [
                        sum(
                            float(
                                inputs["weights"][u, 0, k] - inputs["weights"][u, 1, k]
                            )
                            * float(alternate[k, j])
                            for k in range(16)
                        )
                        for j in range(5)
                    ]
                )
                action_error = max(
                    action_error,
                    float(np.max(abs(diff - saved[kind + "_actions"][u, q]))),
                )
                key_identity = max(
                    key_identity,
                    abs(diff[4] - diff[2] - ((len(hi) - len(lo)) / 10) * diff[3]),
                )
                order_error = max(
                    order_error,
                    float(np.max(abs(saved[kind + "_orders"][u, :, q].sum(0) - diff))),
                )
                count += 16 * 5
    result = {
        "PASS": bool(
            error < 1e-3
            and max(value_error, action_error, key_identity, order_error) < 1e-10
        ),
        "head_error": error,
        "value_error": value_error,
        "action_error": action_error,
        "key_identity_error": key_identity,
        "signed_order_error": order_error,
        "literal_accounts": count,
        "same_agent_alternate_not_blind": True,
        "new_forward_or_response_fit": 0,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result), flush=True)
    if not result["PASS"]:
        raise ValueError("Retain failed verification; no rewrite/rescue")


def main():
    """Run one explicit immutable scientific stage at a time."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.output,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM255 ALL SEALS PASS")
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
