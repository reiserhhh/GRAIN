#!/usr/bin/env python3
"""Execute one sealed six-field discriminator; no decoder or predictor fitting."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_target_wording_fields as M
from grain_research.construction.math_context_diagnostics import aligned_states
from scripts.run_grain_context_diagnostics_hm153 import (
    MODEL,
    PARENT,
    authenticate_parent,
    digest,
    distribution,
    read,
    table,
    weights,
    write,
)

DEFAULT = ROOT / "results/grain_math_response_v1/target_wording_fields_r1"
CODE = (
    "grain_research/construction/math_target_wording_fields.py",
    "scripts/run_grain_target_wording_fields_hm154.py",
    "scripts/verify_grain_target_wording_fields_hm154.py",
    "tests/test_grain_target_wording_fields.py",
    "docs/GRAIN_TARGET_WORDING_FIELD_PLAN_HM154.md",
)
DEPENDENCIES = (
    "grain_research/construction/math_context_diagnostics.py",
    "scripts/run_grain_context_diagnostics_hm153.py",
)
ENERGIES = (
    "common_energy",
    "target_energy",
    "wording_energy",
    "interaction_energy",
    "total_energy",
)
SCALARS = ENERGIES + (
    "intended_support",
    "intended_margin",
    "mean_cross_family_cosine",
    "energy_closure_error",
    "orbit_mean_error",
    "support_identity_error",
)


def prepare(out):
    """Seal the full identical-text six-way crossing before projection."""
    parent, cases = authenticate_parent()
    by_text = {}
    for case in cases:
        if case["pole"] == 0:
            key = (case["axis"], case["family"])
            cell = by_text.setdefault(case["text_id"], {})
            if key in cell:
                raise ValueError("Duplicated neutral hypothesis case")
            cell[key] = case
    texts = []
    for text_id, cells in sorted(by_text.items()):
        six = [cells[(axis, family)] for axis in M.AXES for family in M.FAMILIES]
        first = six[0]
        if len(cells) != 6 or any(c["tokens"] != first["tokens"] for c in six):
            raise ValueError("Every six-way case must use identical native text")
        if any(
            any(c[k] != first[k] for k in ("situation", "style", "seed")) for c in six
        ):
            raise ValueError("Crossed article conditions differ")
        texts.append(
            {
                "text_index": len(texts),
                "text_id": text_id,
                "case_ids": [c["case_id"] for c in six],
                **{
                    k: first[k]
                    for k in ("situation", "style", "seed", "audit", "finish_reason")
                },
            }
        )
    if len(texts) != 54:
        raise ValueError("All54 neutral articles required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "accounts").mkdir()
    write(out / "texts.json", texts)
    write(
        out / "manifest.json",
        {
            "packet": "HM154",
            "code": {p: digest(ROOT / p) for p in CODE},
            "dependencies": {p: digest(ROOT / p) for p in DEPENDENCIES},
            "parent": {
                str((PARENT / name).relative_to(ROOT)): digest(PARENT / name)
                for name in (
                    "manifest.json",
                    "cases.json",
                    "capture_seal.json",
                    "output_seal.json",
                    "alternate_audit.json",
                )
            },
            "texts_sha256": digest(out / "texts.json"),
            "model_weights_sha256": parent["model_weights_sha256"],
            "analysis_seconds_cap": 300,
            "audit_seconds_cap": 120,
            "text_count": 54,
            "case_count": 324,
            "native_units": 864,
            "projection_batches": 54,
            "projected_rows": 6048,
            "field_order": [
                axis + ":" + family for axis in M.AXES for family in M.FAMILIES
            ],
            "orbit_signs": M.ORBIT.tolist(),
            "first_tie_tolerance": 1e-10,
            "absolute_projection_tolerance": 1e-4,
            "relative_projection_tolerance": 1e-5,
            "absolute_algebra_tolerance": 1e-9,
            "relative_algebra_tolerance": 1e-9,
            "fixed_full_vocabulary_spots": [
                [i, t] for i in (0, 26, 53) for t in (0, 15)
            ],
            "decoder_forwards": 0,
            "original_scope_complete": False,
        },
    )
    write(out / "status.json", {"status": "PREPARED", "texts_done": 0})
    print(
        {"prepared_texts": 54, "native_units": 864, "decoder_forwards": 0}, flush=True
    )


def check(out):
    """Authenticate all immutable implementation and observation dependencies."""
    manifest = read(out / "manifest.json")
    for section in ("code", "dependencies", "parent"):
        for name, checksum in manifest[section].items():
            if digest(ROOT / name) != checksum:
                raise ValueError("Frozen dependency changed: " + name)
    if digest(out / "texts.json") != manifest["texts_sha256"]:
        raise ValueError("Crossing addresses changed")
    return manifest


def capture(case_id):
    """Load stored final prediction states; no model execution or imputation."""
    with np.load(PARENT / "captures" / f"{case_id:04d}.npz", allow_pickle=False) as a:
        values = {k: a[k] for k in a.files}
    if not np.array_equal(values["lengths"], [32, 32, 16, 16]):
        raise ValueError("Aligned second-slice support missing")
    if np.any(values["bias"] != 0):
        raise ValueError("Nonzero head bias would change declared reference")
    return values


def mean_row(records, address):
    """Aggregate every position, retaining undefined cosine counts and signs."""
    row = {**address, "native_units": len(records)}
    for key in (
        SCALARS
        + tuple(f"orbit_{j}" for j in range(4))
        + tuple(f"cosine_{f}_{g}" for f, g in M.PAIRS)
    ):
        vals = [r[key] for r in records]
        finite = [v for v in vals if v is not None]
        row[key] = float(np.mean(finite)) if finite else None
        if "cosine" in key:
            row[key + "_undefined"] = len(vals) - len(finite)
    for key in ("strict_first", "tied_first", "not_first"):
        row[key + "_count"] = sum(r[key] for r in records)
    row["positive_support_count"] = sum(r["intended_support"] > 0 for r in records)
    return row


def analyze(out):
    """Compute every six-field Gram once, within one nonrenewing resource cap."""
    import torch

    start = time.monotonic()
    manifest = check(out)
    if read(out / "status.json")["status"] != "PREPARED":
        raise ValueError("Scientific runs cannot be silently repeated")
    _, cases = authenticate_parent()
    if digest(MODEL / "model.safetensors") != manifest["model_weights_sha256"]:
        raise ValueError("Model archive changed")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Declared MPS backend unavailable; no automatic fallback")
    texts = read(out / "texts.json")
    w = weights().to("mps")
    unit_rows, selected_rows, text_rows = [], [], []
    counts = {"texts_done": 0, "projection_batches": 0, "projected_rows": 0}

    def checkpoint(status):
        write(
            out / "status.json",
            {
                "status": status,
                **counts,
                "pid": os.getpid(),
                "seconds_including_load_IO": time.monotonic() - start,
                "decoder_forwards": 0,
            },
        )

    def guard():
        if time.monotonic() - start > manifest["analysis_seconds_cap"]:
            raise TimeoutError("Sealed300second budget exhausted; no renewal")

    checkpoint("RUNNING_OUTPUT_HEAD_ONLY")
    try:
        for text in texts:
            guard()
            six = [capture(i) for i in text["case_ids"]]
            aligned = [aligned_states(a["states"], a["partition"]) for a in six]
            deltas = np.stack([a[0] for a in aligned])
            reference = np.mean([a[1] for a in aligned], axis=0)
            normalizers = np.stack([a[2] for a in aligned])
            projection_input = np.r_[deltas.reshape(96, 1536), reference]
            with torch.inference_mode():
                projected = (
                    (
                        torch.as_tensor(
                            projection_input, dtype=torch.float32, device="mps"
                        )
                        @ w.T
                    )
                    .cpu()
                    .numpy()
                    .astype(float)
                )
            counts["projection_batches"] += 1
            counts["projected_rows"] += len(projection_input)
            ids = np.asarray(cases[text["case_ids"][0]]["tokens"][16:])
            account = M.field_gram(
                projected[:96].reshape(6, 16, -1), projected[96:], normalizers, ids
            )
            native = np.stack(
                [
                    a["logprob"][0, 16:]
                    - a["logprob"][1, 16:]
                    - a["logprob"][2, :16]
                    + a["logprob"][3, :16]
                    for a in six
                ]
            ).astype(float)
            if not np.allclose(
                account["reconstructed_k"], native, atol=1e-4, rtol=1e-5
            ):
                raise ArithmeticError(
                    "Selected diagnostic account does not close native contrast"
                )
            account["native_k"] = native
            np.savez_compressed(
                out / "accounts" / f"{text['text_index']:04d}.npz", **account
            )
            metrics = M.gram_accounts(account["gram"])
            for key in (
                "energy_closure_error",
                "orbit_mean_error",
                "support_identity_error",
            ):
                if not np.allclose(metrics[key], 0, atol=1e-9, rtol=1e-9):
                    raise ArithmeticError("Finite identity failure: " + key)
            local = []
            for t in range(16):
                row = {
                    "text_index": text["text_index"],
                    "text_id": text["text_id"],
                    "token_index": t + 16,
                    **{k: text[k] for k in ("situation", "style", "seed")},
                }
                for key in SCALARS:
                    value = float(metrics[key][t])
                    row[key] = value if np.isfinite(value) else None
                for j in range(4):
                    row[f"orbit_{j}"] = float(metrics["orbit_scores"][t, j])
                for j, (f, g) in enumerate(M.PAIRS):
                    value = float(metrics["cross_family_cosines"][t, j])
                    row[f"cosine_{f}_{g}"] = value if np.isfinite(value) else None
                for key in ("strict_first", "tied_first", "not_first"):
                    row[key] = bool(metrics[key][t])
                row["reference_min_probability"] = float(
                    account["reference_min_probability"][t]
                )
                local.append(row)
                for j, case_id in enumerate(text["case_ids"]):
                    selected_rows.append(
                        {
                            "text_index": text["text_index"],
                            "case_id": case_id,
                            "token_index": t + 16,
                            "axis": M.AXES[j // 3],
                            "family": M.FAMILIES[j % 3],
                            "native_token_id": int(ids[t]),
                            **{
                                k: float(account[k][j, t])
                                for k in (
                                    "selected_u",
                                    "reference_mean_u",
                                    "selected_shape",
                                    "calibration",
                                    "reconstructed_k",
                                    "native_k",
                                )
                            },
                        }
                    )
            unit_rows.extend(local)
            text_rows.append(
                mean_row(
                    local,
                    {
                        "text_index": text["text_index"],
                        "text_id": text["text_id"],
                        **{k: text[k] for k in ("situation", "style", "seed")},
                    },
                )
            )
            counts["texts_done"] += 1
            checkpoint("RUNNING_OUTPUT_HEAD_ONLY")
            if counts["texts_done"] % 9 == 0:
                print({**counts, "seconds": time.monotonic() - start}, flush=True)
        conditions = []
        for situation, style in sorted(
            {(r["situation"], r["style"]) for r in unit_rows}
        ):
            records = [
                r
                for r in unit_rows
                if (r["situation"], r["style"]) == (situation, style)
            ]
            if len(records) != 48:
                raise ValueError("Full18 groups of3texts x16positions required")
            conditions.append(
                mean_row(records, {"situation": situation, "style": style})
            )
        table(out / "unit_accounts.csv", unit_rows)
        table(out / "selected_token_accounts.csv", selected_rows)
        table(out / "text_accounts.csv", text_rows)
        table(out / "condition_accounts.csv", conditions)
        unique_source = {c["text_id"]: c for c in cases}
        summary = {
            "packet": "HM154",
            **counts,
            "native_units": len(unit_rows),
            "selected_token_accounts": len(selected_rows),
            "condition_groups": len(conditions),
            "all_unit_means": mean_row(unit_rows, {}),
            "support_distribution": distribution(
                [r["intended_support"] for r in unit_rows]
            ),
            "cosine_distribution": distribution(
                [
                    r["mean_cross_family_cosine"]
                    if r["mean_cross_family_cosine"] is not None
                    else np.nan
                    for r in unit_rows
                ]
            ),
            "positive_text_mean_support": sum(
                r["intended_support"] > 0 for r in text_rows
            ),
            "intended_first_text_mean_orbit": sum(
                r["orbit_0"] > max(r[f"orbit_{j}"] for j in (1, 2, 3)) + 1e-10
                for r in text_rows
            ),
            "positive_condition_mean_support": sum(
                r["intended_support"] > 0 for r in conditions
            ),
            "intended_first_condition_mean_orbit": sum(
                r["orbit_0"] > max(r[f"orbit_{j}"] for j in (1, 2, 3)) + 1e-10
                for r in conditions
            ),
            "coverage": {
                "source_unique_articles": len(unique_source),
                "source_refusals": sum(
                    c["audit"]["refusal"] for c in unique_source.values()
                ),
                "source_generation_truncations": sum(
                    c["audit"]["truncated"] for c in unique_source.values()
                ),
                "neutral_refusals": sum(t["audit"]["refusal"] for t in texts),
                "neutral_generation_truncations": sum(
                    t["audit"]["truncated"] for t in texts
                ),
                "missing_crossed_cases": 0,
            },
            "seconds_including_load_IO": time.monotonic() - start,
            "decoder_forwards": 0,
            "new_generation_API_training_private_calls": 0,
            "scope": "Same actual synthetic neutral texts; target-conditioned diagnostic-field geometry, not human traits or token-independent inference",
            "original_scope_complete": False,
        }
        write(out / "summary.json", summary)
        files = [
            "unit_accounts.csv",
            "selected_token_accounts.csv",
            "text_accounts.csv",
            "condition_accounts.csv",
            "summary.json",
        ]
        files += [
            str(p.relative_to(out))
            for p in sorted((out / "accounts").glob("[0-9]*.npz"))
        ]
        write(out / "output_seal.json", {p: digest(out / p) for p in files})
        guard()
        checkpoint("COMPLETE_ANALYSIS")
        print(summary, flush=True)
    except Exception as exc:
        checkpoint("FAILED_PRESERVED")
        write(
            out / "failure.json",
            {"type": type(exc).__name__, "error": str(exc), **counts},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze}[args.command](args.out)
