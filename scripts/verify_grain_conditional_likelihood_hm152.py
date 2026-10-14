#!/usr/bin/env python3
"""Re-sum actual HM152 paths; spot-check vocabulary partitions from frozen weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/grain_math_response_v1/conditional_likelihood_paths_r1"
MODEL = (
    ROOT
    / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)


def load(path):
    """Read a frozen public synthetic artifact."""
    return json.loads(path.read_text())


def digest(path):
    """Hash bytes without modifying a recorded scientific result."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def csv_rows(path):
    """Read a complete numerical account, preserving missing values."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify(out):
    """Check all output accounts with explicit differences and prefix sums."""
    m, summary = load(out / "manifest.json"), load(out / "summary.json")
    for section in ("code", "source"):
        for name, checksum in m[section].items():
            if digest(ROOT / name) != checksum:
                raise ValueError("Changed frozen dependency: " + name)
    if digest(out / "cases.json") != m["cases_sha256"]:
        raise ValueError("Case manifest changed")
    for name, checksum in {
        **load(out / "capture_seal.json"),
        **load(out / "output_seal.json"),
    }.items():
        if digest(out / name) != checksum:
            raise ValueError("Changed scientific artifact: " + name)
    cases = load(out / "cases.json")
    case_rows = csv_rows(out / "slice_accounts.csv")
    token_rows = csv_rows(out / "token_accounts.csv")
    condition_rows = csv_rows(out / "condition_summary.csv")
    if len(cases) != 972 or len(case_rows) != 972 or len(condition_rows) != 108:
        raise ValueError("Full original support not present")
    errors, vector_errors = [], []
    calculated = {}
    indices = {}
    for row in token_rows:
        key = (int(row["case_id"]), row["arm"], int(row["token_index"]))
        if key in indices:
            raise ValueError("Duplicated token observation")
        indices[key] = row

    def equal(actual, expected):
        error = float(abs(actual - expected))
        errors.append(error)
        if not np.isclose(actual, expected, atol=1e-9, rtol=1e-9):
            raise ArithmeticError(f"Independent path account differs:{error}")

    for case, row in zip(cases, case_rows, strict=True):
        i, n = case["case_id"], len(case["tokens"])
        if i != int(row["case_id"]) or case["text_id"] != row["text_id"]:
            raise ValueError("Actual article address differs")
        with np.load(out / "captures" / f"{i:04d}.npz", allow_pickle=False) as z:
            raw = {key: z[key].astype(float) for key in z.files}
        selected = np.sum(raw["states"] * raw["covectors"], -1) + raw["bias"]
        error = float(np.max(np.abs(selected - raw["chosen"])))
        vector_errors.append(error)
        if not np.allclose(selected, raw["chosen"], atol=1e-4, rtol=1e-5):
            raise ArithmeticError("Native selected-vector replay failed")
        ratios = []
        for arm, start, count, tokens in (
            ("carry", 0, n, case["tokens"]),
            ("reset_second", 2, max(0, n - 16), case["tokens"][16:]),
        ):
            values = []
            for t in range(count):
                r = indices[i, arm, t]
                if int(r["native_token_id"]) != tokens[t]:
                    raise ValueError("Native token identity changed")
                a, b = start, start + 1
                numerator = raw["chosen"][a, t] - raw["chosen"][b, t]
                norm = raw["partition"][a, t] - raw["partition"][b, t]
                gauge = raw["logit_mean"][a, t] - raw["logit_mean"][b, t]
                for field, value in (
                    ("numerator", numerator),
                    ("normalizer", norm),
                    ("ratio", numerator - norm),
                    ("centered_numerator", numerator - gauge),
                    ("centered_normalizer", norm - gauge),
                    ("direct_ratio", raw["logprob"][a, t] - raw["logprob"][b, t]),
                ):
                    equal(value, float(r[field]))
                values.append(numerator - norm)
            ratios.append(values)
        full, first = sum(ratios[0]), sum(ratios[0][:16])
        equal(full, float(row["full"]))
        equal(first, float(row["first_slice"]))
        delta = None
        if n > 16:
            delta = sum(ratios[0][16:]) - sum(ratios[1])
            equal(first + sum(ratios[1]), float(row["reset_sum"]))
            equal(delta, float(row["carry_minus_reset"]))
        elif row["reset_sum"]:
            raise ArithmeticError("Short unit was imputed")
        calculated[i] = {
            "full": full,
            "delta": delta,
            "reset_sum": None if delta is None else full - delta,
            "numerator_total": float(
                np.sum(raw["chosen"][0, :n] - raw["chosen"][1, :n])
            ),
            "normalizer_total": float(
                np.sum(raw["partition"][0, :n] - raw["partition"][1, :n])
            ),
            "centered_numerator_total": float(
                np.sum(
                    raw["chosen"][0, :n]
                    - raw["chosen"][1, :n]
                    - raw["logit_mean"][0, :n]
                    + raw["logit_mean"][1, :n]
                )
            ),
        }
    equal(
        np.mean(
            [abs(v["delta"]) for v in calculated.values() if v["delta"] is not None]
        ),
        summary["mean_absolute_prefix_transport_effect"],
    )
    if len(token_rows) != sum(
        len(c["tokens"]) + max(0, len(c["tokens"]) - 16) for c in cases
    ):
        raise ValueError("Incomplete actual token support")
    groups = {}
    for case in cases:
        key = tuple(case[k] for k in ("axis", "family", "situation", "style"))
        groups.setdefault(key, []).append(case)
    raw_reversals, centered_reversals, reset_reversals = 0, 0, 0
    for row in condition_rows:
        key = tuple(row[k] for k in ("axis", "family", "situation", "style"))
        group = groups.pop(key)
        contrasts = {}
        for field in (
            "full",
            "reset_sum",
            "numerator_total",
            "normalizer_total",
            "centered_numerator_total",
        ):
            high = [
                calculated[c["case_id"]][field]
                for c in group
                if c["pole"] == 1 and calculated[c["case_id"]][field] is not None
            ]
            low = [
                calculated[c["case_id"]][field]
                for c in group
                if c["pole"] == -1 and calculated[c["case_id"]][field] is not None
            ]
            contrast = (
                None
                if not high or not low
                else sum(high) / len(high) - sum(low) / len(low)
            )
            contrasts[field] = contrast
            reported = row[field + "_high_minus_low"]
            if contrast is None:
                if reported:
                    raise ArithmeticError("Missing contrast was imputed")
            else:
                equal(contrast, float(reported))
        neutral = [calculated[c["case_id"]]["full"] for c in group if c["pole"] == 0]
        equal(sum(neutral) / len(neutral), float(row["neutral_full_mean"]))
        raw_reversals += contrasts["full"] * contrasts["numerator_total"] < 0
        centered_reversals += (
            contrasts["full"] * contrasts["centered_numerator_total"] < 0
        )
        reset_reversals += (
            contrasts["reset_sum"] is not None
            and contrasts["full"] * contrasts["reset_sum"] < 0
        )
    if groups:
        raise ValueError("Unreported original condition")
    equal(raw_reversals, summary["normalizer_changes_condition_contrast_sign"])
    equal(reset_reversals, summary["reset_changes_condition_contrast_sign"])

    # A fixed small full-vocabulary replay avoids another model run or budget grid.
    import torch
    from safetensors import safe_open

    if digest(MODEL / "model.safetensors") != m["model_weights_sha256"]:
        raise ValueError("Frozen observer weights changed")
    with safe_open(
        MODEL / "model.safetensors", framework="pt", device="cpu"
    ) as archive:
        keys = set(archive.keys())
        key = (
            "lm_head.weight"
            if "lm_head.weight" in keys
            else "model.embed_tokens.weight"
        )
        if (
            key == "model.embed_tokens.weight"
            and not load(MODEL / "config.json")["tie_word_embeddings"]
        ):
            raise ValueError("Unspecified output projection")
        # The pinned file is BF16; convert its exact values, not a fitted decoder.
        weights = archive.get_tensor(key).to(torch.float32).numpy()
        bias = (
            archive.get_tensor("lm_head.bias").to(torch.float32).numpy()
            if "lm_head.bias" in keys
            else None
        )
        partitions = []
        for i in (0, 6, 971):
            with np.load(out / "captures" / f"{i:04d}.npz", allow_pickle=False) as z:
                for stream in range(len(z["lengths"])):
                    state = z["states"][stream, 0].astype(float)
                    logits = np.concatenate(
                        [
                            weights[a : a + 8192].astype(float) @ state
                            for a in range(0, len(weights), 8192)
                        ]
                    )
                    if bias is not None:
                        logits += bias
                    error = abs(
                        float(logsumexp(logits)) - float(z["partition"][stream, 0])
                    )
                    partitions.append(error)
                    if error > 1e-4:
                        raise ArithmeticError("Fixed full-vocabulary replay failed")
    return {
        "status": "PASS_ALTERNATE_ARITHMETIC_NOT_THIRD_PARTY",
        "cases": len(cases),
        "token_accounts": len(token_rows),
        "scalar_checks": len(errors),
        "max_account_error": max(errors),
        "max_selected_vector_error": max(vector_errors),
        "full_vocabulary_spot_checks": len(partitions),
        "max_partition_spot_error": max(partitions),
        "condition_contrasts_recomputed": len(condition_rows),
        "raw_gauge_sign_reversals": int(raw_reversals),
        "token_uniform_gauge_sign_reversals": int(centered_reversals),
        "reset_sign_reversals": int(reset_reversals),
        "coverage": {
            "cases_planned": len(cases),
            "cases_captured": len(case_rows),
            "unique_articles": len({c["text_id"] for c in cases}),
            "missing_cases": [],
            "prefix_tokens": m["prefix_tokens"],
            "boundary": m["boundary"],
            "short_cases_without_second_slice": sum(
                len(c["tokens"]) <= 16 for c in cases
            ),
            "source_short_articles": len(
                {c["text_id"] for c in cases if c["actual_full_tokens"] < 32}
            ),
            "source_refusals": len(
                {c["text_id"] for c in cases if c["audit"].get("refusal", False)}
            ),
            "source_finish_reasons": {
                reason: len(
                    {c["text_id"] for c in cases if c["finish_reason"] == reason}
                )
                for reason in sorted({c["finish_reason"] for c in cases})
            },
            "new_articles": 0,
            "generation_seeds_are_authors": False,
        },
        "all_vocabulary_partitions_independently_recomputed": False,
        "new_neural_forwards": 0,
        "original_scope_complete": False,
    }


def main():
    """Print an audit and optionally create a new exclusive receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.input)
    result["audit_code_sha256"] = digest(Path(__file__))
    if args.receipt:
        with args.receipt.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        with (args.input / "coverage.json").open("x") as stream:
            json.dump(result["coverage"], stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
