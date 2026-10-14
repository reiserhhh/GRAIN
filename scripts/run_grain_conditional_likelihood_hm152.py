#!/usr/bin/env python3
"""Observe persona predictive paths and prefix-reset units, without score fitting."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_conditional_likelihood as M
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_semantic_anchor_hm147 as H
from scripts.run_grain_joint_persona_style_enumeration_hm150 import ScientificOutputPath

SOURCE = J.ROOT / "results/grain_math_response_v1/joint_persona_style_r2"
DEFAULT = ROOT / "results/grain_math_response_v1/conditional_likelihood_paths_r1"
CODE = (
    "grain_research/construction/math_conditional_likelihood.py",
    "scripts/run_grain_conditional_likelihood_hm152.py",
    "tests/test_grain_conditional_likelihood.py",
    "docs/GRAIN_CONDITIONAL_LIKELIHOOD_PATH_PLAN_HM152.md",
)


def prepare(out):
    """Freeze all actual articles and matched hypotheses before new observations."""
    parent = ScientificOutputPath(SOURCE)
    old = J.check(parent)
    rows, records = J.generated(parent)
    lookup = {
        tuple(r[k] for k in ("axis", "family", "pole", "situation", "style", "seed")): r
        for r in rows
    }
    cases = []
    for row, record in zip(rows, records, strict=True):
        hypotheses = (
            [(row["axis"], row["family"])]
            if row["pole"]
            else [(axis, family) for axis in J.M.TARGETS for family in J.M.FAMILIES]
        )
        tokens = record["new_ids"][:32]
        if not tokens or not all(isinstance(v, int) and v >= 0 for v in tokens):
            raise ValueError("Missing actual native continuation;do not impute")
        for axis, family in hypotheses:
            prompts = [
                lookup[axis, family, p, row["situation"], row["style"], row["seed"]][
                    "prompt_ids"
                ]
                for p in (1, -1)
            ]
            if not all(p and all(isinstance(v, int) for v in p) for p in prompts):
                raise ValueError("Flat actual native prompts required")
            cases.append(
                {
                    "case_id": len(cases),
                    "text_id": row["text_id"],
                    "axis": axis,
                    "family": family,
                    "situation": row["situation"],
                    "style": row["style"],
                    "seed": row["seed"],
                    "pole": row["pole"],
                    "role": row["role"],
                    "tokens": tokens,
                    "prompts": prompts,
                    "actual_full_tokens": len(record["new_ids"]),
                    "audit": record["audit"],
                    "finish_reason": record["finish_reason"],
                }
            )
    if len(cases) != 972 or len({c["text_id"] for c in cases}) != 702:
        raise ValueError("Full972 hypothesis-conditioned cases/702articles required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "captures").mkdir()
    H.write_json(out / "cases.json", cases)
    H.write_json(
        out / "manifest.json",
        {
            "packet": "HM152",
            "cases": 972,
            "actual_articles": 702,
            "code": {p: H.digest(ROOT / p) for p in CODE},
            "source": {
                str(p.relative_to(ROOT)): H.digest(p)
                for p in (
                    parent / "manifest.json",
                    parent / "materials.json",
                    parent / "generation_seal.json",
                )
            },
            "cases_sha256": H.digest(out / "cases.json"),
            "model_weights_sha256": old["model_weights_sha256"],
            "model_seconds_cap": 1200,
            "logical_sequence_cap": 3890,
            "prefix_tokens": 32,
            "boundary": 16,
            "batch_size": 4,
            "tolerance_absolute": 1e-4,
            "tolerance_relative": 1e-5,
            "probability_law": "Frozen raw-softmax,NOThistoricaltop-p/.8 sampler",
            "original_scope_complete": False,
        },
    )
    J.status(out, "capture", "PREPARED", done=0, seconds=0)
    print({"prepared": len(cases), "new_generations": 0}, flush=True)


def check(out):
    """Authenticate frozen cases, code and upstream hashes."""
    m = H.load(out / "manifest.json")
    for section in ("code", "source"):
        for name, checksum in m[section].items():
            if H.digest(ROOT / name) != checksum:
                raise ValueError("Frozen dependency changed: " + name)
    if H.digest(out / "cases.json") != m["cases_sha256"]:
        raise ValueError("Frozen actual cases changed")
    return m


def observe(observer, specs):
    """Compute normalized states and full-vocabulary accounts in small chunks."""
    import torch

    device = observer.model.device
    inputs = [p + t[:-1] for p, t in specs]
    length = max(map(len, inputs))
    ids = torch.zeros((len(specs), length), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, values in enumerate(inputs):
        ids[i, -len(values) :] = torch.tensor(values, device=device)
        mask[i, -len(values) :] = 1
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.inference_mode():
        hidden = observer.model.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        width = max(len(t) for _, t in specs)
        dim = hidden.shape[-1]
        result = {
            key: np.zeros((len(specs), width), dtype=np.float32)
            for key in ("chosen", "partition", "logprob", "logit_mean", "bias")
        }
        result["states"] = np.zeros((len(specs), width, dim), dtype=np.float32)
        result["covectors"] = np.zeros_like(result["states"])
        result["lengths"] = np.asarray([len(t) for _, t in specs], dtype=int)
        for i, (prompt, tokens) in enumerate(specs):
            start = length - len(inputs[i]) + len(prompt) - 1
            h = hidden[i, start : start + len(tokens)]
            target = torch.tensor(tokens, device=device)
            result["states"][i, : len(tokens)] = h.cpu().numpy()
            result["covectors"][i, : len(tokens)] = (
                observer.model.lm_head.weight[target].cpu().numpy()
            )
            if observer.model.lm_head.bias is not None:
                result["bias"][i, : len(tokens)] = (
                    observer.model.lm_head.bias[target].cpu().numpy()
                )
            for a in range(0, len(tokens), 8):
                b = min(a + 8, len(tokens))
                logits = observer.model.lm_head(h[a:b]).float()
                local_ids = target[a:b, None]
                result["chosen"][i, a:b] = (
                    logits.gather(1, local_ids).squeeze(1).cpu().numpy()
                )
                result["partition"][i, a:b] = torch.logsumexp(logits, -1).cpu().numpy()
                result["logprob"][i, a:b] = (
                    torch.log_softmax(logits, -1)
                    .gather(1, local_ids)
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
                result["logit_mean"][i, a:b] = logits.mean(-1).cpu().numpy()
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ArithmeticError("Nonfinite native observation")
    replay = (
        np.sum(result["states"].astype(float) * result["covectors"].astype(float), -1)
        + result["bias"]
    )
    if not np.allclose(replay, result["chosen"], atol=1e-4, rtol=1e-5):
        raise ArithmeticError("Actual selected logits disagree with W.h")
    if not np.allclose(
        result["chosen"] - result["partition"], result["logprob"], atol=1e-4, rtol=1e-5
    ):
        raise ArithmeticError("Actual vocabulary-normalized account differs")
    return result


def capture(out):
    """Observe every case once within the sealed cumulative compute budget."""
    import torch

    m = check(out)
    previous = H.load(out / "capture_status.json")
    if previous["status"] not in ("PREPARED", "PARTIAL_BUDGET", "RUNNING"):
        raise ValueError("Refuse terminal or failed experiment replay")
    charged = float(previous["seconds"])
    start = time.monotonic()
    cases = H.load(out / "cases.json")
    J.status(out, "capture", "RUNNING", done=previous["done"], seconds=charged)
    observer = J.offline_model(m)
    physical_calls = int(previous.get("physical_calls", 0))
    logical_sequences = int(previous.get("logical_sequences", 0))
    checks = dict(previous.get("causal_checks", {}))
    done = 0
    for case in cases:
        path = out / "captures" / f"{case['case_id']:04d}.npz"
        if path.exists():
            done += 1
            continue
        elapsed = charged + time.monotonic() - start
        if elapsed >= m["model_seconds_cap"]:
            J.status(
                out,
                "capture",
                "PARTIAL_BUDGET",
                done=done,
                seconds=elapsed,
                physical_calls=physical_calls,
                logical_sequences=logical_sequences,
                causal_checks=checks,
            )
            return
        tokens, prompts = case["tokens"], case["prompts"]
        specs = [(p, tokens) for p in prompts]
        if len(tokens) > 16:
            specs.extend((p, tokens[16:]) for p in prompts)
        physical_calls += 1
        logical_sequences += len(specs)
        values = observe(observer, specs)
        if case["axis"] not in checks:
            physical_calls += 1
            logical_sequences += 1
            prefix = observe(observer, [(prompts[0], tokens[:16])])
            n = min(16, len(tokens))
            error = float(
                np.max(np.abs(prefix["states"][0, :n] - values["states"][0, :n]))
            )
            if not np.allclose(
                prefix["states"][0, :n], values["states"][0, :n], atol=1e-4, rtol=1e-5
            ):
                raise ArithmeticError("Native first-slice causal replay differs")
            checks[case["axis"]] = {"case_id": case["case_id"], "state_error": error}
            H.write_npz(out / f"causal_check_{case['axis']}.npz", **prefix)
        if logical_sequences > m["logical_sequence_cap"]:
            raise ValueError("Sealed logical sequence cap exceeded")
        if torch.mps.current_allocated_memory() > 9 * 1024**3:
            raise MemoryError("No remote/CPU fallback above9GiB allocated MPS")
        H.write_npz(path, **values)
        done += 1
        elapsed = charged + time.monotonic() - start
        J.status(
            out,
            "capture",
            "RUNNING",
            done=done,
            seconds=elapsed,
            physical_calls=physical_calls,
            logical_sequences=logical_sequences,
            causal_checks=checks,
        )
        if done % 24 == 0 or done == len(cases):
            print(
                {
                    "cases": done,
                    "total": len(cases),
                    "seconds": elapsed,
                    "physical_calls": physical_calls,
                },
                flush=True,
            )
    H.write_json(
        out / "capture_seal.json",
        {
            f"captures/{i:04d}.npz": H.digest(out / "captures" / f"{i:04d}.npz")
            for i in range(len(cases))
        },
    )
    J.status(
        out,
        "capture",
        "COMPLETE_CAPTURE",
        done=done,
        seconds=charged + time.monotonic() - start,
        physical_calls=physical_calls,
        logical_sequences=logical_sequences,
        causal_checks=checks,
    )


def analyze(out):
    """Retain every path, slice and condition contrast without a classifier."""
    check(out)
    status = H.load(out / "capture_status.json")
    if status["status"] != "COMPLETE_CAPTURE":
        raise ValueError("Full finite capture not complete;do not promote partial")
    for name, checksum in H.load(out / "capture_seal.json").items():
        if H.digest(out / name) != checksum:
            raise ValueError("Changed immutable capture: " + name)
    token_rows, case_rows = [], []
    max_error, max_vector_error = 0.0, 0.0
    for case in H.load(out / "cases.json"):
        with np.load(
            out / "captures" / f"{case['case_id']:04d}.npz", allow_pickle=False
        ) as archive:
            raw = {key: archive[key] for key in archive.files}
        n = len(case["tokens"])
        a = M.paired_account(
            *[raw[k][:2, :n] for k in ("chosen", "partition", "logprob", "logit_mean")]
        )
        reset = None
        if n > 16:
            reset = M.paired_account(
                *[
                    raw[k][2:, : n - 16]
                    for k in ("chosen", "partition", "logprob", "logit_mean")
                ]
            )
        account = M.slice_account(a["ratio"], None if reset is None else reset["ratio"])
        info = {
            key: case[key]
            for key in (
                "case_id",
                "text_id",
                "axis",
                "family",
                "situation",
                "style",
                "seed",
                "pole",
                "role",
            )
        }
        case_rows.append(
            {
                **info,
                **account,
                "numerator_total": float(a["numerator"].sum()),
                "normalizer_total": float(a["normalizer"].sum()),
                "centered_numerator_total": float(a["centered_numerator"].sum()),
                "centered_normalizer_total": float(a["centered_normalizer"].sum()),
            }
        )
        for arm, values, ids in (
            ("carry", a, case["tokens"]),
            ("reset_second", reset, case["tokens"][16:]),
        ):
            if values is None:
                continue
            max_error = max(max_error, values["account_error"])
            for t, token in enumerate(ids):
                token_rows.append(
                    {
                        **info,
                        "arm": arm,
                        "token_index": t,
                        "native_token_id": token,
                        **{
                            key: float(values[key][t])
                            for key in (
                                "numerator",
                                "normalizer",
                                "ratio",
                                "centered_numerator",
                                "centered_normalizer",
                                "direct_ratio",
                            )
                        },
                    }
                )
        replay = (
            np.sum(raw["states"].astype(float) * raw["covectors"].astype(float), -1)
            + raw["bias"]
        )
        max_vector_error = max(
            max_vector_error, float(np.max(np.abs(replay - raw["chosen"])))
        )
    groups = {}
    for row in case_rows:
        key = tuple(row[k] for k in ("axis", "family", "situation", "style"))
        groups.setdefault(key, []).append(row)
    condition_rows = []
    for key, values in groups.items():
        high, low = (
            [r for r in values if r["pole"] == 1],
            [r for r in values if r["pole"] == -1],
        )
        control = [r for r in values if r["pole"] == 0]
        item = dict(zip(("axis", "family", "situation", "style"), key, strict=True))
        for field in (
            "full",
            "reset_sum",
            "numerator_total",
            "normalizer_total",
            "centered_numerator_total",
        ):
            h, l = (
                [r[field] for r in high if r[field] is not None],
                [r[field] for r in low if r[field] is not None],
            )
            item[field + "_high_minus_low"] = (
                float(np.mean(h) - np.mean(l)) if h and l else None
            )
        item["neutral_full_mean"] = float(np.mean([r["full"] for r in control]))
        condition_rows.append(item)
    H.write_csv(out / "token_accounts.csv", token_rows)
    H.write_csv(out / "slice_accounts.csv", case_rows)
    H.write_csv(out / "condition_summary.csv", condition_rows)
    short = sum(r["reset_sum"] is None for r in case_rows)
    summary = {
        "cases": len(case_rows),
        "condition_cells": len(condition_rows),
        "token_accounts": len(token_rows),
        "short_cases_without_second_slice": short,
        "max_normalized_account_error": max_error,
        "max_selected_vector_replay_error": max_vector_error,
        "mean_absolute_prefix_transport_effect": float(
            np.mean(
                [
                    abs(r["carry_minus_reset"])
                    for r in case_rows
                    if r["carry_minus_reset"] is not None
                ]
            )
        ),
        "normalizer_changes_condition_contrast_sign": sum(
            r["numerator_total_high_minus_low"] * r["full_high_minus_low"] < 0
            for r in condition_rows
        ),
        "reset_changes_condition_contrast_sign": sum(
            r["reset_sum_high_minus_low"] is not None
            and r["reset_sum_high_minus_low"] * r["full_high_minus_low"] < 0
            for r in condition_rows
        ),
        "capture": status,
        "new_generation_training_API_private_calls": 0,
        "scope": "Synthetic conditional predictive path,not humanpersona posterior oraccuracy",
        "original_scope_complete": False,
    }
    H.write_json(out / "summary.json", summary)
    H.write_json(
        out / "output_seal.json",
        {
            name: H.digest(out / name)
            for name in (
                "token_accounts.csv",
                "slice_accounts.csv",
                "condition_summary.csv",
                "summary.json",
            )
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Run one bounded phase, retaining any scientific failure and charged time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    started = time.monotonic()
    old = (
        H.load(args.output / "capture_status.json")
        if (args.output / "capture_status.json").exists()
        else {}
    )
    try:
        {"prepare": prepare, "capture": capture, "analyze": analyze}[args.stage](
            args.output
        )
    except Exception as error:
        if args.output.exists():
            if args.stage == "capture":
                current = H.load(args.output / "capture_status.json")
                J.status(
                    args.output,
                    "capture",
                    "FAILED_PRESERVED",
                    **{
                        key: value
                        for key, value in current.items()
                        if key not in ("status", "seconds")
                    },
                    seconds=float(old.get("seconds", 0)) + time.monotonic() - started,
                    error=f"{type(error).__name__}:{error}",
                )
            path = args.output / f"{args.stage}_failure.json"
            if not path.exists():
                H.write_json(
                    path,
                    {"stage": args.stage, "error": f"{type(error).__name__}:{error}"},
                )
        raise


if __name__ == "__main__":
    main()
